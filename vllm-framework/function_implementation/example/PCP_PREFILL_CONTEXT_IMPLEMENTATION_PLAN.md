# Qwen3.5 PCP Implementation Plan

## Background

Current Qwen3.5 PCP-related changes live mainly in:

- `vllm_hcu/vllm_hcu/models/qwen3_5.py`
- `vllm_hcu/vllm_hcu/v1/attention/lightly_cp_utils.py`
- `vllm_hcu/vllm_hcu/v1/hcu_model_runner.py`
- `vllm/vllm/engine/arg_utils.py`
- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`

The current Qwen3.5 wrapper uses HCU `lightly_cp` to split full-attention
tokens. It does not use native vLLM PCP size directly.

Current behavior:

```text
Enable switch: --enable-lightly-cp
Split size: tensor_parallel_size
Split rank: tensor parallel rank
Gather group: tensor parallel group
```

Expected target behavior:

```text
Enable switch: --prefill-context-parallel-size / -pcp
Split size: prefill_context_parallel_size
Split rank: PCP rank
Gather group: PCP group
```

The target is not to simply replace a boolean flag. PCP and TP are different
parallel dimensions. The implementation must keep them separated:

```text
TP: tensor/model shard communication
PCP: prefill token/context split communication
```

For Qwen3.5 hybrid layers:

```text
full_attention: split by PCP, then gather back
linear_attention: keep full-token input unchanged
MLP/layernorm/residual: consume full-token hidden states
```

## Reference Design

Use `/public/home/guanyu1/my_work/my_workspace/vllm-ascend` as the design
reference, especially:

- `vllm_ascend/worker/pcp_utils.py`
- `vllm_ascend/attention/context_parallel/attention_cp.py`
- `vllm_ascend/attention/context_parallel/common_cp.py`
- `vllm_ascend/worker/block_table.py`
- `vllm_ascend/distributed/parallel_state.py`

Important ideas to borrow:

- A central `PCPManager` owns split metadata, padding metadata, restore indices,
  and PCP-related buffers.
- Decode and prefill requests are classified separately.
- PCP splits prefill tokens only.
- Decode tokens are replicated or kept un-split.
- DualChunkSwap uses head/tail chunk assignment for better load balance.
- All-gather restoration uses PCP group, not TP group.
- Qwen3.5 hybrid attention needs explicit layout transition indices between
  linear-attention layout and full-attention layout.
- Full-attention PCP should eventually run in the attention backend as a
  Head/Tail CP computation, not only as model-wrapper scatter/gather.
- Multi-path attention outputs must be merged with LSE-aware weighting, not
  simple concatenation or averaging.
- KV cache slot mapping must be aware of PCP/DCP rank layout for full native PCP.

Do not blindly copy Ascend code. Borrow the architecture and adapt it to the HCU
runner, attention backend, and Qwen3.5 hybrid attention behavior.

## Validated Ascend Qwen3.5 PCP Observations

The following observations from `/public/home/guanyu1/my_work/my_workspace/vllm-ascend`
are accurate and should guide this implementation:

1. DualChunkSwap is the primary prefill split strategy.
   Each prefill request is padded to a multiple of `2 * pcp_world_size`, split
   into `2 * pcp_world_size` chunks, and rank `r` receives head chunk `r` plus
   tail chunk `2 * pcp_world_size - r - 1`.

2. Decode requests are not meaningfully split.
   Decode tokens are copied or kept available on PCP ranks, and restore indices
   keep only the canonical copy when returning to original request order.

3. The minimal metadata set is not just scatter/gather.
   A complete implementation needs:

```text
pcp_scatter_indexes / pcp_allgather_restore_idx
pcp_unpad_mask
pcp_enter_fa_restore_idx
pcp_exit_fa_scatter_idx
q_head_idx / q_tail_idx / q_full_idx
kv_with_q_head_nomask_idx / kv_with_q_head_mask_idx
kv_with_q_tail_nomask_idx / kv_with_q_tail_mask_idx
```

4. Hybrid attention has two token layouts.
   Linear attention and full attention do not necessarily consume the same token
   layout. Ascend handles this with:

```text
pcp_enter_fa_restore_idx: enter full attention from linear-attention layout
pcp_exit_fa_scatter_idx: leave full attention back to linear-attention layout
```

5. The full target is attention-backend PCP.
   The backend splits full-attention prefill into head and tail paths. Each path
   separately handles:

```text
nomask KV: previous chunks that are fully visible
mask KV: current chunk that still needs causal masking
```

Then `q_full_idx` restores head/tail outputs to local query order.

6. Multi-path attention merge must be LSE-aware.
   When combining nomask/mask, chunked context/current chunk, DCP, or PCP paths,
   the output must be merged using log-sum-exp weighting:

```text
lse_final = logsumexp(lse_i)
out_final = sum(exp(lse_i - lse_final) * out_i)
```

Do not replace this with simple output addition, averaging, or concatenation.

Current `_site_mooncake_pcp_qwen3_5` Phase 1/2 work is intentionally smaller:
it validates PCP-oriented split/restore metadata and wires Qwen3.5 wrapper-level
PCP gather for full attention. Phase 4C now adds a backend PCP entry path, but
it is still a bridge rather than the final Ascend-style head/tail algorithm.

## Phase 1: Add PCP-Oriented Split/Restore Utilities

Goal:

Create a PCP split/restore path that uses `prefill_context_parallel_size`,
`get_pcp_group().world_size`, and `get_pcp_group().rank_in_group`.

Implementation direction:

- Add or extend utility code near:
  - `vllm_hcu/vllm_hcu/v1/attention/lightly_cp_utils.py`
- Keep the existing `lightly_cp` path intact at first.
- Add PCP-specific utility functions instead of overloading TP-based helpers.

The new utility should produce:

```text
pcp_scatter_indexes_tensor
pcp_gather_indexes_tensor
pcp_unpad_mask
pcp_local_positions
pcp_local_q_lens
pcp_local_kv_lens
```

Future extensions of this utility layer should also generate the hybrid/full
attention metadata listed above (`pcp_enter_fa_restore_idx`,
`pcp_exit_fa_scatter_idx`, `q_head_idx`, `q_tail_idx`, `q_full_idx`, and
head/tail KV indices). Keep these separate from the minimal Phase 1 metadata so
the first utility remains easy to test.

The algorithm should follow the Ascend DualChunkSwap idea:

```text
1. For each prefill request, pad token count to multiple of 2 * pcp_world_size.
2. Split into 2 * pcp_world_size chunks.
3. Rank i takes:
   - head chunk i
   - tail chunk 2 * pcp_world_size - 1 - i
4. Build restore index for allgather output.
5. Decode requests are not split.
```

Keep this distinction explicit:

```text
total_q_len: original number of scheduled query tokens
pcp_padded_q_len: query token count after PCP padding
local_q_len: token count processed by this PCP rank
```

Validation for this phase:

- Unit-test the split/restore index generation with small examples:
  - `pcp_size=1`
  - `pcp_size=2`
  - `pcp_size=4`
  - mixed decode + prefill
  - non-divisible token counts
- Verify all ranks compute the same restore index.
- Verify gather + restore returns original token order.

## Phase 2: Switch Qwen3.5 Full Attention to PCP Group

Goal:

Make Qwen3.5 full-attention splitting controlled by
`prefill_context_parallel_size`, not `tensor_parallel_size`.

Current code to replace conceptually:

```python
self.tp_size = get_tensor_model_parallel_world_size()
self.tp_rank = get_tensor_model_parallel_rank()
tensor_model_parallel_all_gather(..., dim=0)
```

Target concept:

```python
pcp_group = get_pcp_group()
self.pcp_size = pcp_group.world_size
self.pcp_rank = pcp_group.rank_in_group
pcp_group.all_gather(..., dim=0)
```

Target behavior in `vllm_hcu/vllm_hcu/models/qwen3_5.py`:

```text
linear_attention:
  do not split
  input/output shape stays [total_tokens, hidden_size]

full_attention with pcp_size > 1:
  split hidden_states and positions by PCP metadata
  run self.self_attn on local tokens
  all-gather local outputs across PCP group
  restore original token order
  output shape returns to [total_tokens, hidden_size]

full_attention with pcp_size == 1:
  keep current non-PCP behavior
```

Important:

- Gather token dimension only.
- Do not gather hidden dimension.
- Do not use TP group for PCP token restoration.
- After every full-attention layer, hidden states must be restored before
  post-attention layernorm and MLP.
- This wrapper-level path is a bridge, not the final full-attention PCP design.
  It does not yet implement head/tail backend attention, nomask/mask KV splits,
  or LSE-aware output merging.

Validation for this phase:

- Add temporary debug logs or shape assertions during local testing:
  - before split
  - after split
  - after full attention
  - after PCP gather/restore
- Confirm linear attention sees full-token hidden states.
- Confirm full attention local token count changes with `-pcp`.

## Phase 3: Drive PCP from `prefill_context_parallel_size`

Goal:

Use native vLLM PCP size as the control switch. Do not require
`--enable-lightly-cp`.

Current control path:

```text
parallel_config.enable_lightly_cp
  -> runner.enable_lightly_cp
  -> forward_context.enable_lightly_cp
  -> qwen3_5.py split branch
```

Target control path:

```text
parallel_config.prefill_context_parallel_size > 1
  -> runner.enable_pcp_for_prefill
  -> forward_context.enable_pcp_for_prefill
  -> qwen3_5.py full_attention PCP branch
```

Suggested implementation:

- Add a new explicit forward-context field instead of reusing
  `enable_lightly_cp`, for example:

```python
enable_pcp_for_prefill: bool = False
```

- Set it in HCU model runner based on:

```python
parallel_config.prefill_context_parallel_size > 1
```

- Preserve `enable_lightly_cp` for the old TP-based path until the new PCP path
  is verified.

Important:

- Do not make `--enable-lightly-cp` silently mean native PCP.
- Do not make `prefill_context_parallel_size > 1` use TP group by accident.
- Keep naming clear enough that future debugging can distinguish:
  - `lightly_cp`
  - native PCP
  - DCP

Validation for this phase:

Run combinations:

```bash
--tensor-parallel-size 1 --prefill-context-parallel-size 1
--tensor-parallel-size 2 --prefill-context-parallel-size 1
--tensor-parallel-size 1 --prefill-context-parallel-size 2
--tensor-parallel-size 2 --prefill-context-parallel-size 2
```

Expected:

```text
pcp_size == 1: no PCP split
pcp_size > 1: full attention split by PCP rank
tp_size changes: should not change PCP token split count
```

Implementation status:

- Implemented in `vllm_hcu/v1/hcu_model_runner.py` by deriving
  `enable_pcp_for_prefill` from
  `parallel_config.prefill_context_parallel_size > 1`.
- `prepare_pcp_metadata()` now builds PCP scatter/gather metadata for the
  runner without requiring `--enable-lightly-cp`.
- `vllm_hcu/patches/vllm__forward_context.patch.py` exposes
  `enable_pcp_for_prefill`, `pcp_scatter_indexes_tensor`, and
  `pcp_gather_indexes_tensor` in forward context.
- `vllm_hcu/models/qwen3_5.py` uses the PCP fields only in full-attention
  layers; linear-attention layers continue to see the unsplit hidden states.
- This is still the wrapper-level PCP path. Phase 4 is still needed for the
  full head/tail attention backend and LSE-aware merge.

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_wrapper.py \
  tests/test_pcp_split_restore_utils.py \
  tests/test_qwen35_pcp_phase3_integration.py
```

## Phase 4A: PCP Backend Bootstrap

Goal:

Make native PCP initialization proceed past the HCU FlashAttention compatibility
check, while keeping the implementation boundary explicit. This phase is a
bootstrap, not the final Ascend-style full-attention PCP backend.

Why this phase is necessary:

- Wrapper-level scatter/gather can verify token routing, but it does not model
  the full prefill attention algorithm.
- vLLM/HCU initializes native PCP only when the selected attention
  implementation advertises PCP support. Startup can otherwise fail at:

```text
PCP requires attention impls' support, but the impl FlashAttentionImpl does not
support PCP.
```

- The backend metadata and KV cache update path must at least agree with the
  PCP-local token layout before deeper backend work is possible.

Implementation direction:

- Identify the HCU full-attention implementation used by Qwen3.5
  (`FlashAttentionImpl` in the current run).
- Add a PCP flag to FlashAttention metadata so PCP-enabled requests are visible
  in backend metadata.
- Let `FlashAttentionImpl` pass vLLM's PCP compatibility check.
- Localize `slot_mapping` in `prepare_pcp_metadata()` using the PCP scatter
  indexes, and map padding scatter entries to slot `-1`.
- Add source/unit tests for the compatibility declaration and PCP-local
  slot_mapping construction.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/backends/flash_attn.py`:
  - `FlashAttentionMetadata.pcp_enabled`
  - PCP metadata detection from `prefill_context_parallel_size` plus
    scatter/gather metadata
  - `FlashAttentionImpl.supports_pcp = True`
- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`:
  - PCP-local `slot_mapping` derived from `pcp_scatter_indexes_tensor`
  - padding entries mapped to `-1`
- Tested by:
  - `tests/test_qwen35_pcp_phase4_backend.py`
  - `tests/test_pcp_split_restore_utils.py`

Out of scope:

- Real Head/Tail full-attention computation.
- nomask/mask KV partitioning.
- LSE-aware multi-path attention merge.
- Native PCP for Qwen3.5 linear attention.

Validation for this phase:

- Verify native PCP no longer fails `FlashAttentionImpl.supports_pcp` startup
  compatibility checks.
- Verify PCP-local `slot_mapping` length and padding entries match local
  Q/K/V token layout.
- Run the focused Phase 1-4A tests:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase4_backend.py \
  tests/test_qwen35_pcp_phase3_integration.py \
  tests/test_qwen35_pcp_wrapper.py \
  tests/test_pcp_split_restore_utils.py
```

## Phase 4B: Hybrid Metadata Scope Isolation

Goal:

Keep Qwen3.5 linear-attention layers on original/global metadata while
full-attention layers use PCP-local metadata. This must happen before the real
full-attention PCP backend work, because linear attention currently does not
support native PCP.

Why this phase is necessary:

- Qwen3.5 is a hybrid model. Full attention and linear attention consume
  different runtime metadata:
  - full attention can use PCP-local query/KV metadata
  - linear attention/GDN state-scan still expects the original global request
    metadata and state indices
- Phase 4A makes native PCP start and creates PCP-local attention metadata, but
  using that single PCP-local metadata for every layer corrupts linear-attention
  state handling.
- The observed failure after Phase 4A is:

```text
IndexError: The shape of the mask [2] at index 0 does not match the shape of
the indexed tensor [1, 32, 128, 128] at index 0
```

This happens because GDN linear attention reads a `has_initial_state` mask and
state indices that no longer match its full-token hidden-state layout.

Implementation direction:

- Preserve both metadata views in the runner or forward context:
  - global/original metadata for linear attention
  - PCP-local metadata for full attention
- Make Qwen3.5 wrapper select metadata by layer type:
  - `linear_attention`: use global/original metadata and full-token hidden states
  - `full_attention`: use PCP-local metadata, PCP scatter, PCP gather/restore
- Keep the invariant that linear attention receives full-token hidden states
  until Phase 6.
- Avoid implementing native linear-attention PCP in this phase.
- Do not let linear-attention GDN state indices, `has_initial_state`, or
  sequence metadata come from PCP-local full-attention metadata.

Possible implementation shapes:

- Add explicit forward-context fields such as:

```text
global_attn_metadata
pcp_attn_metadata
```

Then temporarily select the right one around each layer type.

- Or keep `attn_metadata` global by default and temporarily switch to PCP-local
  metadata only inside Qwen3.5 full-attention execution.

Prefer the smaller change that matches current HCU/vLLM forward-context
patterns.

Validation for this phase:

- Unit-test that Qwen3.5 linear-attention layers do not see PCP-local metadata.
- Unit-test that Qwen3.5 full-attention layers still receive PCP scatter/gather
  metadata.
- Re-run the serve path and verify requests no longer fail in GDN with
  `has_initial_state` / `initial_state` shape mismatch.
- Verify linear attention still receives full-token hidden states.
- Verify full attention still restores output to full-token length after PCP
  gather.

Implementation status:

- Implemented in `vllm/vllm/forward_context.py` and the HCU forward-context
  patch by adding `pcp_attn_metadata`.
- Implemented in `vllm_hcu/v1/hcu_model_runner.py`:
  - default `attn_metadata` remains global/original metadata
  - `pcp_attn_metadata` is built separately from PCP-local common metadata
  - `pcp_attn_metadata` is passed through forward context
- Implemented in `vllm_hcu/models/qwen3_5.py`:
  - linear-attention layers continue to use default/global metadata
  - full-attention PCP calls temporarily switch to `pcp_attn_metadata`
  - full-attention PCP requires `pcp_attn_metadata` instead of silently using
    global metadata
- Tested by:
  - `tests/test_qwen35_pcp_phase4b_metadata_scope.py`
  - `tests/test_qwen35_pcp_wrapper.py`

Out of scope:

- Head/Tail full-attention backend.
- KV cache physical layout redesign.
- Native linear-attention PCP.

## Phase 4C: Real Full-Attention PCP Backend

Goal:

Move from wrapper-level PCP gather and Phase 4A startup compatibility to a
PCP-aware full-attention backend entry path for Qwen3.5 full-attention layers.

Why this phase is necessary:

- Phase 4A only makes the current backend PCP-compatible enough to start and
  consume PCP-local metadata.
- Phase 4B keeps hybrid metadata scoped correctly.
- Neither Phase 4A nor Phase 4B implements the real full-attention PCP
  algorithm.
- Full-attention PCP should avoid unnecessary full KV all-gather where possible.
- Head/tail query chunks need different KV index sets and different causal mask
  behavior.
- Multi-path outputs require LSE-aware merging.

Implementation direction:

- Add a PCP-aware full-attention execution path in the HCU FlashAttention
  backend so PCP can be routed separately from the default paged-attention
  path.
- Extend the PCP metadata builder to produce full-attention backend indices:

```text
q_head_idx
q_tail_idx
q_full_idx
kv_with_q_head_nomask_idx
kv_with_q_head_mask_idx
kv_with_q_tail_nomask_idx
kv_with_q_tail_mask_idx
head_attn_nomask_seqlens
tail_attn_nomask_seqlens
attn_mask_seqlens
```

- In the Qwen3.5 full-attention backend, wire the execution path so the
  backend receives PCP-specific metadata and can later execute:

```text
head path:
  Q = q_head_idx
  KV nomask = previous visible chunks
  KV mask = current head chunk

tail path:
  Q = q_tail_idx
  KV nomask = previous visible chunks
  KV mask = current tail chunk

post:
  concatenate head/tail outputs
  restore with q_full_idx
```

- Implement or reuse an LSE-aware merge helper equivalent to:

```text
lse_final = logsumexp(lse_i)
out_final = sum(exp(lse_i - lse_final) * out_i)
```

- Treat `supports_pcp=True` as fully justified only when the backend path
  actually handles:
  - local Q tokens
  - full or correctly gathered KV context
  - causal mask semantics
  - LSE-aware output merge
  - mixed decode/prefill behavior

Out of scope:

- Native PCP for Qwen3.5 linear attention. Linear attention has different
  recurrent/state-scan semantics and is planned separately in Phase 6.

Validation for this phase:

- Unit-test generated head/tail/Q/KV indices for `pcp_size=2` and `pcp_size=4`.
- Unit-test `q_full_idx` restores concatenated head/tail outputs.
- Unit-test LSE merge against a direct full softmax attention reference on small
  tensors.
- Verify full-attention output numerically matches non-PCP for small prompts.
- Verify mixed decode/prefill behavior stays correct.
- Verify the backend routes PCP-enabled full attention through the PCP entry
  branch before the default paged-attention path.

## Phase 5: Align KV Cache and Slot Mapping

Goal:

Make Qwen3.5 KV cache physical layout and slot mapping consistent with PCP/DCP
rank topology. Keep this separate from Phase 4A's local `slot_mapping` fix:
Phase 4A aligns the current local Q/K/V update; Phase 5 validates and implements
the final cache address mapping across PCP and DCP ranks.

Reference:

- `vllm_ascend/worker/block_table.py`
- Ascend uses:

```text
total_cp_world_size = pcp_world_size * dcp_world_size
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
```

Current files to inspect in HCU/vLLM:

- `vllm/vllm/v1/worker/block_table.py`
- `vllm/vllm/v1/core/kv_cache_utils.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/vllm/v1/core/kv_cache_coordinator.py`
- `vllm_hcu/vllm_hcu/v1/hcu_model_runner.py`

Risks:

- Hybrid attention models may have multiple KV cache groups with different page
  sizes or different state types.
- Qwen3.5 combines full attention and linear attention/GDN-like state. Native PCP
  should not assume every layer has the same KV-cache or hidden-state layout
  semantics.
- Existing upstream assertions may reject PCP for hybrid attention, mamba, or
  sliding-window patterns.
- Phase 4A's PCP-local `slot_mapping` may be enough to avoid shape errors, but
  it does not prove the physical KV cache layout is globally correct for
  prefill followed by decode.
- Phase 4B keeps metadata scoped by layer type, but it does not prove the
  physical cache address mapping is correct across prefill/decode.

Implementation direction:

- Audit HCU/vLLM block table and slot mapping ownership for Qwen3.5 full
  attention under PCP.
- Make final slot mapping aware of:

```text
total_cp_world_size = pcp_world_size * dcp_world_size
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
```

- Ensure prefill writes KV to the same physical layout that decode later reads.
- Ensure `block_table_tensor`, `slot_mapping`, `seq_lens`, and
  `num_computed_tokens` describe the same PCP-local/global token contract.
- Ensure `cp_kv_cache_interleave_size` is respected.
- Keep linear attention on its current full-token, non-native-PCP layout until
  Phase 6.

Validation for this phase:

- Check no page-size mismatch in hybrid KV cache setup.
- Check prefill writes KV to the expected physical slots.
- Check decode reads the same cache layout.
- Test mixed prefill/decode batches.
- Compare short prefill + decode output against `pcp_size=1` within acceptable
  numerical tolerance.
- Add tests around slot mapping for:
  - PCP only
  - DCP only
  - PCP * DCP
  - padding and mixed decode/prefill

## Phase 5B: Hybrid Hidden-State Layout Switching

Goal:

Add explicit layout transition metadata for the hybrid Qwen3.5 boundary while
preserving the current invariant that linear attention sees full-token hidden
states.

Why this may be needed:

- Ascend-style Qwen3.5 PCP can maintain different token layouts for linear
  attention and full attention.
- If full-attention backend PCP starts returning PCP-local layout instead of
  restoring full-token layout after each full-attention layer, Qwen3.5 needs
  explicit transition indexes:

```text
pcp_enter_fa_restore_idx: enter full attention from linear-attention layout
pcp_exit_fa_scatter_idx: leave full attention back to linear-attention layout
```

Current rule before Phase 6:

- Linear attention must continue to receive full-token hidden states.
- Full attention may process PCP-local tokens internally, but wrapper/backend
  output must be restored before post-attention layernorm, MLP, and the next
  linear-attention layer unless this phase explicitly changes that contract.
- Phase 4B handles metadata scope; this phase is only about hidden-state layout
  transitions if the implementation later stops restoring full-token layout
  after full attention.

Implementation status:

- Implemented preparatory metadata in
  `vllm_hcu/v1/attention/lightly_cp_utils.py`:
  - `PCPHybridLayoutMetadata`
  - `build_pcp_hybrid_layout_metadata()`
  - `pcp_enter_fa_restore_idx`
  - `pcp_exit_fa_scatter_idx`
- The current active layout contract is unchanged:
  - `pcp_enter_fa_restore_idx` is the safe full-token-to-PCP-local full
    attention scatter index, with padding entries mapped to token 0.
  - `pcp_exit_fa_scatter_idx` is the PCP all-gather restore index that returns
    full-attention output to original full-token order.
  - Qwen3.5 full attention still scatters internally and gathers/restores before
    post-attention layernorm, MLP, and the next layer.
  - Linear attention still receives full-token hidden states and global metadata.
- Implemented forward-context plumbing in:
  - `vllm/vllm/forward_context.py`
  - `vllm_hcu/patches/vllm__forward_context.patch.py`
  - `vllm_hcu/v1/hcu_model_runner.py`
- `vllm_hcu/models/qwen3_5.py` now validates that the hybrid layout metadata is
  available when native PCP is enabled, but does not change execution semantics.

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase5b_layout_switching.py \
  tests/test_qwen35_pcp_wrapper.py
```

Validation for this phase:

- Verify linear-attention layers receive the intended token layout before and
  after full-attention transitions.
- Verify hidden/residual lengths stay aligned across full-attention and
  linear-attention layer boundaries.
- Add focused tests for enter/exit layout indexes if these indexes become active.

## Phase 6: Linear-Attention Native PCP and End-to-End Tests

Goal:

Evaluate and implement native PCP for Qwen3.5 linear-attention layers if the
GatedDeltaNet/FLA backend can correctly handle PCP-local token layouts.

Why this is separate from Phases 4 and 5:

- Linear attention is not softmax attention.
- DualChunkSwap and head/tail causal softmax decomposition do not directly apply
  to GatedDeltaNet/FLA state-scan kernels.
- Native PCP for linear attention likely needs cross-rank prefix/recurrent state
  transfer or an equivalent associative scan formulation.
- It should not be attempted until full-attention PCP and KV cache layout are
  stable enough to provide a trustworthy baseline.

Files to inspect:

- `vllm/vllm/model_executor/layers/fla/`
- HCU GDN/linear-attention backend implementation files
- Qwen3.5 linear-attention wrapper and state/cache ownership
- Any existing DCP/CP handling in FLA kernels

Implementation direction:

- Define the linear-attention PCP contract:
  - token layout per PCP rank
  - required prefix state per rank
  - state handoff or scan merge across ranks
  - decode behavior
- Prototype small reference tests against non-PCP linear attention.
- Only after correctness is established, allow linear-attention layers to
  consume PCP-local token slices.
- Keep full-attention PCP backend and KV/slot mapping from Phases 4C/5 as the
  baseline path.
- If Phase 5B layout switching is not needed, preserve the simpler invariant:
  linear attention receives full-token hidden states.

Phase 6A implementation status:

- The codebase currently does not implement true native linear-attention PCP.
- A safety guard is in place in `vllm_hcu/models/qwen3_5.py`:
  - `enable_pcp_for_linear_attention` defaults to `False`
  - if it is ever enabled, Qwen3.5 raises a clear `NotImplementedError`
- This keeps the validated invariant intact:
  - linear attention still receives full-token hidden states
  - full attention still uses the PCP split/gather path
- The missing piece for a real Phase 6 implementation is a GDN state contract:
  - prefix/recurrent state handoff across PCP ranks
  - cross-rank state merge or scan
  - correctness tests against the non-PCP reference

Phase 6A verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6_linear_guard.py \
  tests/test_qwen35_pcp_phase5b_layout_switching.py \
  tests/test_qwen35_pcp_wrapper.py
```

## Phase 6B: Linear-Attention PCP Reference Contract

Goal:

Define the metadata contract for future native Qwen3.5 linear-attention PCP
without changing the active execution path.

Why this phase exists:

- GatedDeltaNet is recurrent/stateful. A rank that processes later tokens needs
  the final recurrent state from earlier tokens.
- DualChunkSwap is good for full softmax attention, but it is not a safe token
  layout for linear attention unless the GDN state transfer problem is solved.
- Before enabling native linear PCP, the code needs explicit metadata for:
  - contiguous per-rank linear-attention token ownership
  - decode token replication
  - all-gather restore order
  - predecessor/successor rank state handoff

Implementation direction:

- Add `PCPLinearAttentionMetadata` near the existing PCP metadata utilities.
- Build this metadata from the original full-token request layout, not from the
  full-attention DualChunkSwap layout.
- Use contiguous per-request chunks for prefill tokens:

```text
prefill request tokens -> pad to pcp_size -> split into pcp_size contiguous chunks
rank r gets chunk r
rank r requires prefix/recurrent state from rank r-1 when r > 0
rank r sends final recurrent state to rank r+1 when r < pcp_size - 1
```

- Keep decode tokens replicated or full-token for now, matching the existing
  safe decode behavior.
- Attach the metadata to forward context for future GDN work, but do not make
  `linear_attention` consume PCP-local hidden states in this phase.

Validation for this phase:

- Unit-test contiguous linear PCP split metadata for:
  - `pcp_size=1`
  - `pcp_size=2`
  - non-divisible prefill length
  - mixed decode + prefill
- Verify restore indices reconstruct original full-token order after all-gather.
- Verify state source/sink rank metadata is correct.
- Verify Qwen3.5 linear attention still receives full-token hidden states while
  native linear PCP remains disabled.

Implementation status:

- Implemented `PCPLinearAttentionMetadata` in
  `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Implemented `build_pcp_linear_attention_metadata()`:
  - prefill tokens use contiguous per-request chunks by PCP rank
  - decode tokens remain replicated/full-token
  - restore indices reconstruct original full-token order after all-gather
  - state source/destination ranks are explicit for future GDN handoff
- `prepare_pcp_metadata()` attaches `pcp_linear_attention_metadata` to the PCP
  common metadata object.
- This metadata is not active in Qwen3.5 linear attention yet. The active
  runtime invariant remains unchanged: linear attention receives full-token
  hidden states.

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6b_linear_contract.py \
  tests/test_qwen35_pcp_phase6_linear_guard.py
```

Phase 6C implementation status:

- `pcp_linear_attention_metadata` is now carried through:
  - `vllm/vllm/forward_context.py`
  - `vllm_hcu/patches/vllm__forward_context.patch.py`
  - `vllm_hcu/v1/hcu_model_runner.py`
  - `vllm_hcu/models/qwen3_5.py`
- Qwen3.5 linear attention can observe the contract metadata, but it still
  executes on full-token hidden states.
- Native linear PCP remains disabled until a real GDN cross-rank state merge or
  handoff is implemented.

## Phase 6D: Linear-Attention PCP Reference State Handoff

Goal:

Add a small reference-only execution helper that proves the Phase 6B contiguous
linear-attention split can reproduce a serial recurrent computation when each
PCP rank receives the final recurrent state from the previous rank.

Why this phase exists:

- GDN/linear attention is stateful. A later chunk cannot be computed correctly
  from token slices alone.
- Before touching FLA/GDN kernels, the code needs a testable reference for:
  - scatter local linear-attention tokens by `PCPLinearAttentionMetadata`
  - zero out padding entries
  - pass recurrent state rank-by-rank
  - gather/restore rank-local outputs to original token order
- This is still not the production native linear PCP path.

Implementation direction:

- Add helpers near the existing PCP metadata utilities:

```text
scatter_pcp_linear_attention_input()
restore_pcp_linear_attention_output()
run_pcp_linear_recurrent_reference()
```

- Keep the reference helper intentionally narrow:
  - prefill-only
  - one sequence at a time
  - caller-provided recurrent `step_fn(token, state) -> (output, next_state)`
- Use it to prove the state-handoff contract before wiring real GDN kernels.
- Prefer contiguous per-rank token chunks for the first native linear PCP
  implementation.
- Do not use `position % pcp_size` as the first production layout for GDN:
  it still preserves the token-order dependency chain and needs either per-token
  synchronization or a true associative scan/merge formulation to be useful.
- Treat `position % pcp_size` as a later optimization path only after the
  contiguous state-handoff path is correct and tested.

Validation for this phase:

- Unit-test that scatter masks padding positions to zero.
- Unit-test that rank-local recurrent execution with state handoff restores the
  same outputs as a serial full-token recurrent reference.
- Unit-test that the helper rejects multi-sequence/decode metadata instead of
  implying broader support.

Out of scope:

- Enabling `enable_pcp_for_linear_attention`.
- Integrating with `GatedDeltaNet` or FLA kernels.
- Cross-rank runtime communication for recurrent states.

## Phase 6E: Linear-Attention GDN State Handoff Plan

Goal:

Turn the Phase 6B/6D linear-attention state contract into an explicit local
handoff plan that the future `gdn_attention_core/_forward_core` integration can
consume.

Why this phase exists:

- The real Qwen3.5 linear-attention entry is
  `Qwen3_5GatedDeltaNet -> torch.ops.vllm.gdn_attention_core ->
  Qwen3NextGatedDeltaNet._forward_core`.
- That path will need to know, per local prefill sequence:
  - whether this rank receives initial GDN state from a previous PCP rank
  - whether this rank sends final state to a later PCP rank
  - which rank owns the final full-sequence state that must be broadcast before
    decode can run replicated on every PCP rank
- Short prefill sequences may not have real tokens on every PCP rank, so state
  handoff must skip padding-only ranks.

Implementation direction:

- Extend `PCPLinearAttentionMetadata` with final-state source rank metadata.
- Add `PCPLinearStateHandoffPlan`.
- Add `build_pcp_linear_state_handoff_plan()`.
- Keep the plan local and metadata-only. Do not perform distributed send/recv or
  cache mutation in this phase.

Validation for this phase:

- Unit-test short prefill where later PCP ranks are padding-only.
- Unit-test decode replication has no state handoff or final-state broadcast.
- Unit-test mixed decode + prefill marks only the prefill request for state
  receive/broadcast.

Implementation status:

- Implemented metadata field:
  - `pcp_linear_final_state_src_ranks`
- Implemented local plan:
  - `PCPLinearStateHandoffPlan`
  - `build_pcp_linear_state_handoff_plan()`
- Fixed the contiguous linear split contract so `state_dst_ranks` points to the
  next rank that owns real tokens, not blindly to `rank + 1`.
- Still out of scope:
  - enabling `enable_pcp_for_linear_attention`
  - wiring `gdn_attention_core/_forward_core`
  - actual PCP send/recv or broadcast of GDN states

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6e_linear_state_handoff.py \
  tests/test_qwen35_pcp_phase6d_linear_reference.py
```

## Phase 6F: Linear-Attention GDN State IO Contract

Goal:

Define local payload/update helpers for the future runtime GDN state handoff.
This turns the Phase 6E plan into concrete tensors that can later be passed to
PCP send/recv and broadcast code.

Why this phase exists:

- Real Qwen3.5 linear PCP needs two state movements:
  - send each local chunk's final GDN state to the next rank that owns real
    tokens for the same sequence
  - broadcast the final full-sequence GDN state from the final token-owning rank
    to all PCP ranks before replicated decode
- Before adding distributed communication, the local row-selection and cache
  update contract should be testable without NPU runtime dependencies.

Implementation direction:

- Add small dataclasses:

```text
PCPLinearStateSendPayload
PCPLinearStateUpdate
```

- Add helpers:

```text
build_pcp_linear_state_send_payload()
build_pcp_linear_final_state_broadcast_payload()
build_pcp_linear_initial_state_update()
build_pcp_linear_final_state_update()
```

- Keep these helpers local only:
  - no distributed send/recv
  - no broadcast implementation
  - no real GDN cache mutation
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test that final-state send payloads select the rows for the next real
  rank.
- Unit-test that a rank requiring an initial GDN state fails clearly if the
  received state tensors are missing.
- Unit-test that the final token-owning rank can build a cache update from its
  local final state.
- Unit-test that non-final ranks require the broadcast final state before
  building their decode-ready cache update.
- Unit-test that replicated decode requests produce no state handoff payload.
- Unit-test that state payload rows are detached and moved to CPU before
  object-based PCP group communication.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Runtime note: object-based PCP communication (`send_object` and
  `broadcast_object`) should carry CPU payload tensors only. Cache update
  helpers move received payloads back to the local cache device/dtype before
  mutation.
- Still out of scope:
  - distributed GDN state communication
  - wiring `Qwen3_5GatedDeltaNet` / `gdn_attention_core`
  - mutating real GDN caches from the returned update objects
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6f_linear_state_io.py \
  tests/test_qwen35_pcp_phase6e_linear_state_handoff.py
```

## Phase 6G: Linear-Attention GDN Cache Update Contract

Goal:

Add a minimal local cache update helper for applying Phase 6F state updates to
GDN conv/recurrent state caches.

Why this phase exists:

- Phase 6F defines which state rows should be received, sent, or broadcast.
- The real `_forward_core` integration also needs a small, testable primitive
  for writing received or final GDN states into the local caches.
- This helper keeps cache mutation semantics separate from distributed
  communication so each piece can be verified independently.

Implementation direction:

- Add:

```text
apply_pcp_linear_state_update_to_cache()
```

- The helper updates both:
  - `conv_state_cache`
  - `recurrent_state_cache`
- Empty updates are no-ops.
- Non-empty updates require both conv and recurrent state tensors.
- Received payload tensors may come from another PCP rank/device. Before cache
  mutation, move `state_indices`, conv state, and recurrent state to the local
  cache device and target cache dtype.

Validation for this phase:

- Unit-test that selected cache rows are updated and unrelated rows are
  unchanged.
- Unit-test that empty updates are no-ops.
- Unit-test that partial conv/recurrent updates fail clearly.
- Unit-test that received payload tensors are converted to the local cache
  device/dtype before assignment.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Still out of scope:
  - distributed GDN state communication
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6g_linear_cache_update.py \
  tests/test_qwen35_pcp_phase6f_linear_state_io.py
```

## Phase 6H: Linear-Attention GDN Cache-to-Payload Contract

Goal:

Add local helpers that select GDN conv/recurrent cache rows by state index and
convert those rows into Phase 6F send/broadcast payloads.

Why this phase exists:

- After a local GDN prefill chunk runs, the rank's final conv/recurrent states
  live in the GDN caches for that request's `state_indices`.
- The future `_forward_core` wiring should not duplicate cache-row selection
  logic when preparing:
  - final-state handoff to the next real token-owning rank
  - final full-sequence state broadcast before replicated decode
- This keeps cache row selection, payload construction, and distributed
  communication as separate testable pieces.

Implementation direction:

- Add:

```text
select_pcp_linear_state_cache_rows()
build_pcp_linear_state_send_payload_from_cache()
build_pcp_linear_final_state_broadcast_payload_from_cache()
```

- Keep the helpers local only:
  - no distributed send/recv
  - no broadcast implementation
  - no mutation of model execution flow
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test direct cache row selection by `state_indices`.
- Unit-test next-rank send payload construction from cache rows.
- Unit-test final-state broadcast payload construction from cache rows.
- Unit-test non-final ranks produce empty final-state broadcast payloads.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Still out of scope:
  - distributed GDN state communication
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6h_linear_cache_payload.py \
  tests/test_qwen35_pcp_phase6g_linear_cache_update.py
```

## Phase 6I: Linear-Attention Payload-to-Update Contract

Goal:

Convert received PCP GDN state payloads into local cache update objects by
matching `seq_indexes`.

Why this phase exists:

- Real distributed communication may deliver state payload rows in send order,
  not necessarily in the receiver's local sequence order.
- The receiver must apply received states to its own `state_indices` in the
  order expected by `PCPLinearStateHandoffPlan`.
- Missing or duplicated payload sequence ids should fail before any cache
  mutation occurs.

Implementation direction:

- Add helpers:

```text
build_pcp_linear_initial_state_update_from_payload()
build_pcp_linear_final_state_update_from_payload()
```

- The helpers:
  - collect the sequence ids required by the local plan
  - reorder received payload rows by `seq_indexes`
  - reuse the existing Phase 6F update builders
  - keep empty receive paths as no-ops

Validation for this phase:

- Unit-test initial-state payload rows are reordered into local plan order.
- Unit-test missing required `seq_indexes` fail clearly.
- Unit-test duplicated payload `seq_indexes` fail clearly.
- Unit-test received final-state broadcast payload builds a local final update.
- Unit-test final token-owning ranks do not require a received final payload.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Still out of scope:
  - distributed GDN state communication
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6i_linear_payload_update.py \
  tests/test_qwen35_pcp_phase6h_linear_cache_payload.py
```

## Phase 6J: Linear-Attention Point-to-Point Payload Routing

Goal:

Add local routing helpers for point-to-point GDN state handoff payloads before
introducing real PCP distributed communication.

Why this phase exists:

- Phase 6H builds outgoing payload rows with per-row `dst_ranks`.
- Phase 6I consumes one received payload and converts it into local cache
  updates.
- The missing local contract between those two steps is routing:
  - group outgoing rows by destination PCP rank
  - merge multiple payload fragments for the same receiver
  - reject duplicate `seq_indexes` before a receiver applies state updates

Implementation direction:

- Add helpers:

```text
merge_pcp_linear_state_payloads()
route_pcp_linear_state_send_payloads()
```

- Keep the helper scoped to next-rank point-to-point handoff:
  - do not mix final-state broadcast semantics into this helper
  - no distributed send/recv implementation
  - no mutation of model execution flow
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test payload merge concatenates state rows.
- Unit-test duplicate `seq_indexes` fail clearly.
- Unit-test routing groups rows by destination rank.
- Unit-test routed payloads feed the Phase 6I initial-state update helper.
- Unit-test invalid destination ranks fail clearly.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Still out of scope:
  - final-state broadcast routing
  - distributed GDN state communication
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6j_linear_payload_routing.py \
  tests/test_qwen35_pcp_phase6i_linear_payload_update.py
```

## Phase 6K: Linear-Attention Final-State Broadcast Routing

Goal:

Add local routing helpers for final full-sequence GDN state broadcast payloads.

Why this phase exists:

- Phase 6J handles point-to-point handoff between adjacent real token-owning
  ranks.
- Replicated decode needs every PCP rank to have the final full-sequence GDN
  state after prefill.
- The final token-owning rank can use its local final state directly; the other
  ranks need the broadcast payload from that source rank.

Implementation direction:

- Add:

```text
route_pcp_linear_final_state_broadcast_payloads()
```

- The helper:
  - treats `payloads_by_rank` as source-rank ordered
  - requires broadcast payload rows to use `dst_ranks == -1`
  - routes each final-state row to all non-source PCP ranks
  - merges per-receiver rows with duplicate `seq_indexes` validation

- Keep this local only:
  - no distributed collective implementation
  - no mutation of model execution flow
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test broadcast payloads route to non-source ranks.
- Unit-test routed broadcast payloads feed the Phase 6I final-state update
  helper.
- Unit-test multiple source ranks merge correctly per receiver.
- Unit-test duplicate `seq_indexes` fail clearly.
- Unit-test non-broadcast destination ranks fail clearly.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Still out of scope:
  - real PCP broadcast/all-gather communication
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6k_linear_broadcast_routing.py \
  tests/test_qwen35_pcp_phase6j_linear_payload_routing.py
```

## Phase 6L: Linear-Attention Runtime Handoff Plan Context

Goal:

Expose `PCPLinearStateHandoffPlan` through forward context so future
`Qwen3NextGatedDeltaNet._forward_core` wiring can consume a single runtime plan
instead of rebuilding the plan inside each linear-attention layer.

Why this phase exists:

- Phase 6E defines the local handoff plan.
- Phases 6F-6K define payload/update/cache/routing contracts that consume that
  plan.
- The model execution path needs the plan in forward context before the real GDN
  `_forward_core` pre/post hooks can be attached safely.

Implementation direction:

- Build `pcp_linear_state_handoff_plan` in the HCU model runner immediately
  after `pcp_linear_attention_metadata` is created.
- Add `pcp_linear_state_handoff_plan` to:
  - `vllm_hcu/patches/vllm__forward_context.patch.py`
  - `vllm/vllm/forward_context.py`
  - both runner `set_forward_context()` call paths
- Make the Qwen3.5 linear-attention guard require both:
  - `pcp_linear_attention_metadata`
  - `pcp_linear_state_handoff_plan`

Validation for this phase:

- Unit/source-test forward context exposes the new field.
- Unit/source-test runner builds and passes the handoff plan.
- Unit/source-test Qwen3.5 guard reads and requires the handoff plan.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/hcu_model_runner.py`
  - `vllm_hcu/patches/vllm__forward_context.patch.py`
  - `vllm/vllm/forward_context.py`
  - `vllm_hcu/models/qwen3_5.py`
- Still out of scope:
  - real PCP send/recv or broadcast communication
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6l_linear_plan_context.py \
  tests/test_qwen35_pcp_phase6_linear_guard.py
```

## Phase 6M: Linear-Attention Pre/Post-Core State Exchange Contract

Goal:

Add local helper functions that compose the Phase 6F-6L primitives into the
operations a future GDN `_forward_core` hook needs immediately before and after
running the local linear-attention core.

Why this phase exists:

- The previous phases define individual pieces:
  - received payload -> update
  - update -> cache mutation
  - cache rows -> outgoing payloads
  - point-to-point and broadcast routing
- `_forward_core` should call a small set of helpers rather than manually
  reimplementing this sequence around the GDN kernels.
- Keeping this as a local contract makes the pre/post-core behavior testable
  before real PCP communication is introduced.

Implementation direction:

- Add helpers:

```text
apply_pcp_linear_initial_state_update_from_payload_to_cache()
build_pcp_linear_post_core_state_payloads_from_cache()
apply_pcp_linear_final_state_update_from_payload_to_cache()
```

- Add a small result dataclass:

```text
PCPLinearPostCoreStatePayloads
```

- Keep this local only:
  - no real send/recv or broadcast
  - no changes to `Qwen3NextGatedDeltaNet._forward_core`
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test received initial-state payloads are applied to caches before core.
- Unit-test post-core point-to-point send payloads are built from cache state.
- Unit-test post-core final-state broadcast payloads are built from cache state.
- Unit-test received final-state payloads update decode-ready caches.
- Unit-test final token-owning ranks can apply their local final state without a
  received payload.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Still out of scope:
  - real PCP communication
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6m_linear_state_exchange.py \
  tests/test_qwen35_pcp_phase6l_linear_plan_context.py
```

## Phase 6N: Linear-Attention State Exchange Adapter

Goal:

Add a local exchange adapter that converts all ranks' post-core state payloads
into per-rank received payloads for the next pre/post-core helper calls.

Why this phase exists:

- Phase 6M defines the local operations around a GDN core call.
- Real PCP communication should have one narrow replacement point:
  - gather all ranks' `PCPLinearPostCoreStatePayloads`
  - route point-to-point handoff payloads
  - route final-state broadcast payloads
  - return the payload each rank should consume
- Keeping the adapter local first lets the routing contract be tested before
  wiring object/tensor communication.

Implementation direction:

- Add:

```text
PCPLinearStateExchangePayloads
exchange_pcp_linear_post_core_state_payloads()
```

- The adapter composes:
  - `route_pcp_linear_state_send_payloads()`
  - `route_pcp_linear_final_state_broadcast_payloads()`

- Keep this local only:
  - no real `get_pcp_group()` communication yet
  - no changes to `Qwen3NextGatedDeltaNet._forward_core`
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test point-to-point and final broadcast payloads are routed by rank.
- Unit-test empty payloads remain no-ops.
- Unit-test invalid source count fails clearly.
- Unit-test duplicate sequence validation propagates from the routing helpers.

Implementation status:

- Implemented in `vllm_hcu/v1/attention/lightly_cp_utils.py`.
- Still out of scope:
  - real PCP all-rank object/tensor exchange
  - wiring into `Qwen3NextGatedDeltaNet._forward_core`
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6n_linear_state_exchange_adapter.py \
  tests/test_qwen35_pcp_phase6m_linear_state_exchange.py
```

## Phase 6O: Linear-Attention Pre-Enable Readiness Validation

Goal:

Complete the validation work that should pass before any attempt to truly enable
`enable_pcp_for_linear_attention`.

Why this phase exists:

- Phases 6B-6N define the metadata, state handoff, cache update, routing, and
  exchange contracts.
- Before removing the Qwen3.5 guard, there should be one focused readiness test
  that simulates the whole local state lifecycle:
  - rank 0 prefill chunk produces a final state
  - rank 1 consumes that state as its initial state
  - rank 1 produces the final full-sequence state
  - rank 0 receives the final broadcast state
  - both ranks end with the same decode-ready GDN cache state
- The test must also verify that the runtime guardrails are still in place until
  real communication and `_forward_core` wiring are implemented.

Implementation direction:

- Add a pre-enable readiness test file that composes the existing helpers.
- Keep the test local/fake-runtime only:
  - no real NPU kernels
  - no real PCP communication
  - no execution-path changes
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test a two-rank prefill handoff plus final broadcast makes both ranks'
  decode caches identical.
- Unit/source-test the linear PCP guard remains enabled.
- Unit/source-test `pcp_linear_state_handoff_plan` remains available in forward
  context.

Implementation status:

- Implemented in:
  - `tests/test_qwen35_pcp_phase6o_linear_pre_enable_readiness.py`
- Still out of scope and still required before enabling:
  - real PCP all-rank object/tensor communication
  - `Qwen3NextGatedDeltaNet._forward_core` pre/post hook wiring
  - correctness comparison against real non-PCP GDN outputs
  - removing the Qwen3.5 `NotImplementedError` guard

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6o_linear_pre_enable_readiness.py \
  tests/test_qwen35_pcp_phase6n_linear_state_exchange_adapter.py
```

## Phase 6P: Linear-Attention Forward-Core Hook Scaffold

Goal:

Add dormant GDN `_forward_core` pre/post hook points for native linear PCP
without enabling the feature or changing the current execution path.

Why this phase exists:

- Phase 6O completes the local pre-enable validation.
- The next production step needs stable hook locations around the real GDN core:
  - before local GDN core runs, where received initial state will be applied
  - after local GDN core runs, where outgoing handoff/broadcast payloads will be
    built
- The hook must be safe while disabled and fail clearly if someone bypasses the
  existing Qwen3.5 guard before real PCP communication is implemented.

Implementation direction:

- Add helper functions:

```text
prepare_pcp_linear_forward_core_state()
finalize_pcp_linear_forward_core_state()
```

- Patch `vllm.model_executor.models.qwen3_next._forward_core` through the HCU
  patch file to call these helpers around the prefill GDN core cache update.
- Keep default behavior no-op when `enable_pcp_for_linear_attention=False`.
- Keep enabled behavior blocked with a clear `NotImplementedError` until real
  PCP state exchange is wired.

Validation for this phase:

- Unit-test prepare hook is no-op while disabled.
- Unit-test prepare hook requires the handoff plan and linear metadata if
  enabled.
- Unit-test prepare hook blocks with `NotImplementedError` until real exchange
  exists.
- Unit-test finalize hook is no-op when prepare was no-op.
- Source-test the Qwen3Next patch contains both hook points.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `vllm_hcu/patches/vllm__model_executor__models__qwen3_next.patch.py`
- Still out of scope:
  - real PCP state communication
  - using received payloads inside `_forward_core`
  - removing the Qwen3.5 linear PCP guard
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6p_linear_forward_core_hooks.py \
  tests/test_qwen35_pcp_phase6o_linear_pre_enable_readiness.py
```

## Phase 6Q: Linear-Attention Forward-Core Local State Hook

Goal:

Complete production-integration step 1 at the local hook level: let the GDN
`_forward_core` hook consume a received initial-state payload before the core
and produce post-core handoff/broadcast payloads after the core, while still
leaving real cross-rank communication and the public linear-PCP switch disabled.

Why this phase exists:

- Phase 6P adds safe dormant hook points.
- Step 1 is not complete until the hook actually composes the already-tested
  Phase 6M helpers:
  - pre-core: apply received initial state to local GDN caches
  - pre-core: mark `has_initial_state` so GDN does not zero the received state
  - post-core: build outgoing point-to-point and broadcast payloads from local
    caches
  - post-core: expose those payloads on forward context for the future
    communication layer

Implementation direction:

- Add forward-context fields:

```text
pcp_linear_initial_state_payload
pcp_linear_post_core_state_payloads
```

- Update `prepare_pcp_linear_forward_core_state()`:
  - validate metadata and handoff plan
  - apply `pcp_linear_initial_state_payload` via the Phase 6M helper
  - mark received-state rows in `attn_metadata.has_initial_state`
  - return hook state for finalize

- Update `finalize_pcp_linear_forward_core_state()`:
  - build `PCPLinearPostCoreStatePayloads` from local caches
  - store the payloads on forward context

- Keep unchanged:
  - no real PCP communication yet
  - no removal of the Qwen3.5 guard
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test disabled hook remains no-op.
- Unit-test enabled pre-hook applies received initial state to caches and marks
  `has_initial_state`.
- Unit-test enabled post-hook builds/stores post-core payloads.
- Unit/source-test forward context exposes the hook payload fields.
- Keep full PCP focused regression passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `vllm_hcu/patches/vllm__forward_context.patch.py`
  - `vllm/vllm/forward_context.py`
  - `tests/test_qwen35_pcp_phase6p_linear_forward_core_hooks.py`
- Still out of scope:
  - real PCP state communication
  - feeding exchanged payloads between ranks at runtime
  - removing the Qwen3.5 linear PCP guard
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6p_linear_forward_core_hooks.py \
  tests/test_qwen35_pcp_phase6o_linear_pre_enable_readiness.py
```

## Phase 6R: Linear-Attention PCP Group State Exchange

Goal:

Complete production-integration step 2 at the communication-adapter level: use
the real PCP group object-broadcast primitive to collect each rank's post-core
GDN state payloads, then route them into per-rank initial-state and final-state
payloads.

Why this phase exists:

- Phase 6Q lets the GDN `_forward_core` hook produce local post-core payloads.
- The next missing production piece is not another local routing helper; it is a
  narrow communication seam that can gather those payloads across the PCP group.
- vLLM's `GroupCoordinator` exposes `broadcast_object()` but not a direct
  object all-gather API, so the least invasive adapter is:
  - loop over each PCP source rank
  - broadcast that rank's `PCPLinearPostCoreStatePayloads`
  - reuse `exchange_pcp_linear_post_core_state_payloads()` for routing

Implementation direction:

- Add:

```text
exchange_pcp_linear_post_core_state_payloads_with_group()
```

- Validate:
  - PCP group exists
  - group `rank_in_group` exists
  - `pcp_size` matches group `world_size` when available
  - every rank contributes non-empty post-core payload objects

- Keep unchanged:
  - no runtime call from model runner yet
  - no final-state cache update after group exchange yet
  - no removal of the Qwen3.5 guard
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test the adapter broadcasts each source rank's payload and routes the
  returned exchange payloads.
- Unit-test missing group rank metadata fails clearly.
- Unit-test `pcp_size` / group `world_size` mismatch fails clearly.
- Keep the previous local exchange tests passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `tests/test_qwen35_pcp_phase6r_linear_group_state_exchange.py`
- Still out of scope:
  - feeding exchanged payloads between real GDN layers at runtime
  - applying final-state broadcast payloads to caches in the model execution
    path
  - removing the Qwen3.5 linear PCP guard
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6r_linear_group_state_exchange.py \
  tests/test_qwen35_pcp_phase6n_linear_state_exchange_adapter.py
```

## Phase 6S: Linear-Attention Forward-Context State Exchange Bridge

Goal:

Complete production-integration step 3 at the forward-context bridge level:
consume the local post-core payload already stored on forward context, run the
PCP group state exchange adapter, and write this rank's received initial/final
state payloads back to forward context.

Why this phase exists:

- Phase 6R can collect and route all ranks' post-core payloads through the real
  PCP group object-broadcast primitive.
- The GDN hook should not know how cross-rank routing is performed; it should
  only consume payloads exposed on forward context.
- This phase creates the bridge that later runtime wiring can call at the
  correct GDN-layer boundary.
- The bridge must remain dormant while `enable_pcp_for_linear_attention=False`
  so current serve behavior is unchanged.

Important ordering note:

- This phase intentionally does not call the bridge from the whole-model runner.
  Initial-state handoff must happen before the relevant GDN core executes; doing
  a whole-model post-forward exchange would be too late for correctness.

Implementation direction:

- Add forward-context field:

```text
pcp_linear_final_state_payload
```

- Add:

```text
exchange_pcp_linear_forward_context_state()
```

- The bridge:
  - returns no-op when native linear PCP is disabled
  - requires `pcp_linear_post_core_state_payloads` when enabled
  - calls `exchange_pcp_linear_post_core_state_payloads_with_group()`
  - writes this rank's `pcp_linear_initial_state_payload`
  - writes this rank's `pcp_linear_final_state_payload`

- Keep unchanged:
  - no whole-model runner call yet
  - no GDN-layer boundary call yet
  - no final-state cache application in the execution path yet
  - no removal of the Qwen3.5 guard
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test disabled bridge is no-op.
- Unit-test enabled bridge writes this rank's initial-state payload.
- Unit-test enabled bridge writes this rank's final-state payload.
- Unit-test enabled bridge requires post-core payloads.
- Unit/source-test forward context exposes the final-state payload field.
- Keep full PCP focused regression passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `vllm_hcu/patches/vllm__forward_context.patch.py`
  - `vllm/vllm/forward_context.py`
  - `tests/test_qwen35_pcp_phase6s_linear_forward_context_exchange.py`
- Still out of scope:
  - invoking this bridge at the correct GDN-layer boundary
  - applying final-state broadcast payloads to GDN caches in runtime
  - removing the Qwen3.5 linear PCP guard
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6s_linear_forward_context_exchange.py \
  tests/test_qwen35_pcp_phase6r_linear_group_state_exchange.py
```

## Phase 6T: Linear-Attention Final-State Context Apply

Goal:

Complete production-integration step 4 at the local cache-apply level: after a
PCP state exchange has written this rank's final-state payload to forward
context, apply that payload to the local GDN caches so decode can start from the
full-sequence final linear-attention state.

Why this phase exists:

- Phase 6S writes `pcp_linear_final_state_payload` to forward context.
- The existing Phase 6M primitive can apply a final-state payload to cache, but
  runtime code should call a forward-context-level helper rather than manually
  re-reading plan/payload fields.
- This keeps the final-state path symmetric with the pre-core initial-state
  hook and gives the future GDN-layer runtime integration one narrow call site.

Implementation direction:

- Add:

```text
apply_pcp_linear_forward_context_final_state()
```

- The helper:
  - returns no-op when native linear PCP is disabled
  - requires `pcp_linear_state_handoff_plan` when enabled
  - requires `state_indices` when enabled
  - reads `pcp_linear_final_state_payload`
  - calls `apply_pcp_linear_final_state_update_from_payload_to_cache()`
  - clears `pcp_linear_final_state_payload` after applying it

- Keep unchanged:
  - no GDN-layer boundary call yet
  - no rank-ordered initial-state runtime handoff yet
  - no removal of the Qwen3.5 guard
  - no enablement of `enable_pcp_for_linear_attention`

Validation for this phase:

- Unit-test disabled helper is no-op and does not mutate caches.
- Unit-test received final-state payload updates local caches.
- Unit-test final-rank local final state remains valid without a received
  payload.
- Unit-test missing plan fails clearly.
- Unit-test missing state indices fails clearly.
- Keep full PCP focused regression passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `tests/test_qwen35_pcp_phase6t_linear_final_state_context_apply.py`
- Still out of scope:
  - invoking this helper at the correct GDN-layer boundary
  - sequencing rank-to-rank initial-state handoff around the real GDN core
  - removing the Qwen3.5 linear PCP guard
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6t_linear_final_state_context_apply.py \
  tests/test_qwen35_pcp_phase6s_linear_forward_context_exchange.py
```

## Phase 6U: Linear-Attention Runtime Post-Core Hook Composition

Goal:

Complete production-integration step 5 at the real GDN post-core boundary:
after the GDN core updates local cache state, let the existing finalize hook
optionally run PCP group exchange and apply this rank's final-state payload back
to the local GDN caches.

Why this phase exists:

- Phase 6Q already places a post-core hook at the real `_forward_core`
  boundary.
- Phases 6R-6T add the pieces needed after local GDN computation:
  - build local post-core payloads
  - exchange payloads across the PCP group
  - write this rank's final-state payload to forward context
  - apply that final-state payload to cache
- The runtime hook should compose these pieces in one place, instead of
  spreading exchange/apply logic into the model patch.

Implementation direction:

- Extend `finalize_pcp_linear_forward_core_state()` with optional:

```text
pcp_group
pcp_size
```

- Preserve existing behavior when `pcp_group` is not provided.
- When `pcp_group` is provided:
  - require forward context
  - build/store local post-core payloads
  - call `exchange_pcp_linear_forward_context_state()`
  - call `apply_pcp_linear_forward_context_final_state()`
- Update the Qwen3Next `_forward_core` patch to pass `get_pcp_group()` into the
  finalize hook.

Keep unchanged:

- `enable_pcp_for_linear_attention` remains disabled by default.
- The Qwen3.5 linear PCP guard remains in place.
- Rank-ordered initial-state handoff is still not solved in this phase.
- Native linear PCP must still not be enabled end to end.

Validation for this phase:

- Unit-test runtime post-core hook performs exchange and final-state cache apply
  when a PCP group is provided.
- Unit-test runtime exchange fails clearly if forward context is missing.
- Source-test Qwen3Next patch passes `get_pcp_group()` to the finalize hook.
- Keep full PCP focused regression passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `vllm_hcu/patches/vllm__model_executor__models__qwen3_next.patch.py`
  - `tests/test_qwen35_pcp_phase6u_linear_runtime_post_core_hook.py`
- Still out of scope:
  - rank-ordered pre-core initial-state handoff
  - removing the Qwen3.5 linear PCP guard
  - enabling `enable_pcp_for_linear_attention`

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6u_linear_runtime_post_core_hook.py \
  tests/test_qwen35_pcp_phase6t_linear_final_state_context_apply.py
```

## Phase 6V: Linear-Attention Rank-Ordered Initial-State Handoff

Goal:

Complete production-integration step 6: guarantee rank `r` receives rank
`r - 1`'s final GDN state before rank `r` starts its local GDN core.

Why this phase exists:

- Linear attention is recurrent. Rank `r` cannot run its local chunk until it
  has the final state produced by the previous chunk owner.
- The Phase 6R all-rank object-broadcast exchange is useful for final-state
  broadcast, but it is too late for pre-core initial-state handoff.
- A naive broadcast loop can deadlock because different ranks would enter
  collectives in different orders.
- `GroupCoordinator` exposes `send_object()` and `recv_object()`, which are the
  right primitives for the ordered chain:

```text
rank0 core -> send_object(rank1)
rank1 recv_object(rank0) -> core -> send_object(rank2)
rank2 recv_object(rank1) -> core -> ...
```

Implementation direction:

- Add:

```text
receive_pcp_linear_rank_ordered_initial_state()
send_pcp_linear_rank_ordered_initial_state()
```

- Extend `prepare_pcp_linear_forward_core_state()` with optional `pcp_group`:
  - if the current rank needs initial state, block on `recv_object()` from the
    required source rank(s)
  - store the received payload on forward context
  - then apply the existing pre-core cache update
- Extend `finalize_pcp_linear_forward_core_state()` runtime path:
  - after local post-core payloads are built, send handoff payloads with
    `send_object()` to downstream rank(s)
  - then continue the existing final-state exchange/apply path
  - clear any stale `pcp_linear_initial_state_payload` after runtime exchange
- Update the Qwen3Next `_forward_core` patch to pass `get_pcp_group()` into both
  pre-core and post-core hooks.

Keep unchanged:

- `enable_pcp_for_linear_attention` remains disabled by default.
- The Qwen3.5 linear PCP guard remains in place.
- Native linear PCP is still not enabled end to end.

Validation for this phase:

- Unit-test rank 1 pre-core hook receives rank 0 payload before applying cache
  state.
- Unit-test rank 0 pre-core hook does not receive and clears stale initial
  payload.
- Unit-test post-core hook sends rank-ordered payload before final-state apply.
- Source-test Qwen3Next patch passes `get_pcp_group()` to both hooks.
- Keep full PCP focused regression passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `vllm_hcu/patches/vllm__model_executor__models__qwen3_next.patch.py`
  - `tests/test_qwen35_pcp_phase6v_linear_rank_ordered_handoff.py`
- Still out of scope:
  - removing the Qwen3.5 linear PCP guard
  - enabling `enable_pcp_for_linear_attention`
  - end-to-end numerical comparison against non-PCP GDN

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6v_linear_rank_ordered_handoff.py \
  tests/test_qwen35_pcp_phase6u_linear_runtime_post_core_hook.py
```

## Phase 6W: Linear-Attention Runtime Guard Relaxation

Goal:

Complete production-integration step 7: remove the old unconditional Qwen3.5
linear-PCP `NotImplementedError` now that the GDN pre-core, post-core, group
exchange, final-state apply, and rank-ordered initial-state handoff pieces are
implemented and tested.

Why this phase exists:

- Earlier phases intentionally kept this guard in place while the GDN
  cross-rank state path was incomplete.
- After Phase 6U and Phase 6V, the wrapper should no longer block a ready
  runtime contract solely with the old message:

```text
linear-attention native PCP requires GDN...
```

- The guard should still fail clearly when required runtime pieces are missing.

Implementation direction:

- In `vllm_hcu/models/qwen3_5.py`, replace the unconditional
  `NotImplementedError` with runtime contract checks:
  - `pcp_linear_attention_metadata` must exist
  - `pcp_linear_state_handoff_plan` must exist
  - `get_pcp_group()` must provide:
    - `send_object`
    - `recv_object`
    - `broadcast_object`

- Keep unchanged:
  - `self.enable_pcp_for_linear_attention = False` remains the default in the
    runner
  - no CLI/config enablement yet
  - no end-to-end serve path with native linear PCP enabled yet

Validation for this phase:

- Unit-test ready runtime contract no longer raises the old
  `NotImplementedError`.
- Unit-test missing PCP group p2p support fails clearly.
- Source-test the old guard string is gone from Qwen3.5.
- Source-test linear PCP is still disabled by default in the runner.
- Keep full PCP focused regression passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/models/qwen3_5.py`
  - `tests/test_qwen35_pcp_phase6_linear_guard.py`
  - `tests/test_qwen35_pcp_phase6w_linear_guard_relaxation.py`
- Still out of scope:
  - enabling `enable_pcp_for_linear_attention` from config/CLI
  - end-to-end serve validation with native linear PCP enabled
  - numerical comparison against non-PCP GDN

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6w_linear_guard_relaxation.py \
  tests/test_qwen35_pcp_phase6_linear_guard.py
```

## Phase 6X: Linear-Attention Native PCP Enable Switch

Goal:

Complete production-integration step 8 at the switch level: allow native
Qwen3.5 linear-attention PCP to be explicitly enabled for serve validation,
while keeping the default path disabled.

Why this phase exists:

- Phases 6Q-6V implement the GDN state path.
- Phase 6W relaxes the old Qwen3.5 runtime guard.
- The runner still needs a controlled switch so native linear PCP can be tested
  without making it the default behavior.

Implementation direction:

- Add environment variable:

```text
VLLM_HCU_ENABLE_PCP_LINEAR_ATTENTION
```

- In the HCU model runner, set:

```text
enable_pcp_for_linear_attention =
    prefill_context_parallel_size > 1
    and VLLM_HCU_ENABLE_PCP_LINEAR_ATTENTION
```

- Treat that value as a config-level permission only. Each real or dummy forward
  step must enable native linear PCP in `ForwardContext` only when both are
  available:
  - `pcp_linear_attention_metadata`
  - `pcp_linear_state_handoff_plan`

- Keep unchanged:
  - default remains disabled
  - `prefill_context_parallel_size=1` cannot enable native linear PCP
  - metadata and runtime guard checks still run when enabled
  - profile/dummy runs can keep native linear PCP disabled when they do not
    build linear PCP metadata

Validation for this phase:

- Unit-test the environment variable defaults to disabled.
- Unit-test true values enable the env switch.
- Source-test runner requires both PCP world size and the env switch.
- Unit-test the per-step effective switch requires both linear PCP metadata and
  the state handoff plan before entering Qwen3.5 native linear PCP.
- Source-test the existing serve script does not force-enable native linear PCP.
- Keep full PCP focused regression passing.

Implementation status:

- Implemented in:
  - `vllm_hcu/platforms/envs.py`
  - `vllm_hcu/v1/hcu_model_runner.py`
  - `tests/test_qwen35_pcp_phase6x_linear_enable_switch.py`
- Still requires manual serve validation:
  - `pcp_size=2`
  - short prefill
  - long prefill
  - prefill followed by decode
  - no errors or hangs
  - optional text/logit comparison against a non-PCP or linear-PCP-disabled run

Verification added:

```bash
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest \
  tests/test_qwen35_pcp_phase6x_linear_enable_switch.py \
  tests/test_qwen35_pcp_phase6w_linear_guard_relaxation.py
```

Testing matrix:

```text
pcp_size=1, tp_size=1
pcp_size=1, tp_size=2
pcp_size=2, tp_size=1
pcp_size=2, tp_size=2
pcp_size=4 if hardware allows
```

## Phase 6Y: Linear-Attention Local Token Execution

Goal:

Fix the first native-linear-PCP serve correctness issue after Phase 6X: with
`VLLM_HCU_ENABLE_PCP_LINEAR_ATTENTION=1`, the runtime no longer hangs, but
client output is semantically wrong because GDN state handoff assumes local
linear chunks while the real linear-attention core still receives full-token
hidden states.

Observed evidence:

- Serve/client can complete without hanging.
- Debug logs show per-rank linear metadata is present:
  - rank0 linear scatter: `[0..16]`
  - rank1 linear scatter: `[17..32, -1]`
- But both ranks still enter GDN core with global shapes such as:
  - `mixed_qkv=(33, 8192)`
  - `query_core=(1, 33, ...)`
- That means state handoff and actual token execution disagree, which corrupts
  the linear-attention state/output.

Implementation direction:

- In the Qwen3.5 `linear_attention` branch, when native linear PCP is enabled:
  - scatter `hidden_states` by `pcp_linear_scatter_indexes_tensor`
  - drop `-1` padding before calling `self.linear_attn`
  - build temporary per-layer GDN metadata whose `num_actual_tokens` and
    `non_spec_query_start_loc` describe only this rank's real local linear
    tokens
  - run the real GDN core on local tokens
  - pad local output back to `local_padded_q_len`
  - all-gather across the PCP group
  - restore full-token order with `pcp_linear_gather_indexes_tensor`
- Keep the wrapper contract simple after each linear layer: return full-token
  layout so the next full-attention layer can use the existing full-attention
  PCP scatter path.
- Keep decode semantics replicated. The local metadata rewrite is for prefill
  GDN metadata; decode can continue to use the existing replicated path/state.

Validation for this phase:

- Unit-test unpadded linear scatter drops `-1` padding tokens before GDN core.
- Unit-test local linear output is padded back to the rank-local padded shape
  before all-gather.
- Unit-test local GDN metadata rewrites `num_actual_tokens`,
  `num_prefill_tokens`, and `non_spec_query_start_loc` to local real-token
  values without mutating the original metadata.
- Unit-test Qwen3.5 linear branch uses local metadata during `linear_attn` and
  restores the original forward-context metadata after the call.
- Re-run the full Phase 6 regression suite.

Implementation status:

- Implemented in:
  - `vllm_hcu/models/qwen3_5.py`
  - `vllm_hcu/v1/attention/lightly_cp_utils.py`
  - `tests/test_qwen35_pcp_phase6d_linear_reference.py`
  - `tests/test_qwen35_pcp_phase6_linear_guard.py`
- Manual serve validation still required with:

```bash
VLLM_HCU_ENABLE_PCP_LINEAR_ATTENTION=1 bash serve-pcp.sh
bash client.sh
```

Expected serve-log signal:

- During prefill, `[QWEN35_PCP_LINEAR_ATTN_CORE] mixed_qkv=` should show local
  token counts per PCP rank, e.g. around `17` and `16` for a 33-token prompt
  with `pcp_size=2`, instead of both ranks showing `33`.

Verification added:

```bash
/public/home/guanyu1/.local/bin/python3.11 -m py_compile \
  vllm_hcu/v1/attention/lightly_cp_utils.py \
  vllm_hcu/models/qwen3_5.py \
  vllm_hcu/patches/vllm__model_executor__models__qwen3_next.patch.py

uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest tests/test_qwen35_pcp_phase6*.py -q
```

Model behavior tests:

- short prompt
- long prompt above PCP threshold if threshold remains
- mixed batch with decode + prefill
- Qwen3.5 full-attention layers
- Qwen3.5 linear-attention layers with native linear PCP disabled
- Qwen3.5 linear-attention layers with native linear PCP enabled, after Phase 6
- speculative decoding if enabled
- multimodal/MRoPE if target model uses it

Correctness checks:

- No shape mismatch after each full-attention layer.
- No hidden/residual length mismatch.
- Before Phase 6 native PCP, linear attention receives full-token hidden states.
- After Phase 6 native PCP, linear-attention output matches non-PCP reference
  within acceptable tolerance.
- Logits indices still point to the correct final token for each request.
- Decode output matches non-PCP within acceptable numerical tolerance.
- No deadlock in TP/PCP mixed parallel setup.

## Minimal First Milestone

The smallest useful milestone is:

```text
Use -pcp to control Qwen3.5 full-attention token split, while linear attention
stays full-token.
```

Scope:

- Add PCP split/gather based on `get_pcp_group()`.
- Add forward-context flag derived from `prefill_context_parallel_size > 1`.
- Use PCP group all-gather in Qwen3.5 full attention.
- Keep existing `enable_lightly_cp` path intact until validated.

Out of scope for first milestone:

- Full Ascend-style CP attention backend port, including Head/Tail attention,
  nomask/mask KV split, and LSE-aware output merge.
- MTP/multimodal/MRoPE support unless immediately required by the target run.
- Large refactor of all HCU models.

## Things to Avoid

- Do not use `tensor_model_parallel_all_gather` for PCP token restoration.
- Do not use `get_tensor_model_parallel_world_size()` as PCP split size.
- Do not make linear attention consume PCP-local token slices before Phase 6.
- Do not enable Phase 6 linear-attention PCP without reference correctness tests.
- Do not ignore hybrid-attention layout transitions between linear attention and
  full attention.
- Do not merge multi-path attention outputs without LSE weighting.
- Do not silently equate `--enable-lightly-cp` with native PCP.
- Do not bypass `supports_pcp` assertions without verifying backend semantics.
- Do not assume TP and PCP ranks have the same layout.
- Do not make KV cache slot mapping changes without testing decode.

## Quick Diagnostic Checklist

When debugging a failure, check these in order:

1. Which flag enabled the path?
   - `--enable-lightly-cp`
   - `--prefill-context-parallel-size`
2. Which group was used for split/gather?
   - TP group
   - PCP group
3. What are the shapes around Qwen3.5 full attention?
   - before split
   - after split
   - after attention
   - after gather/restore
4. Did linear attention receive full-token hidden states?
5. Did `gather_indexes_tensor` or restore index remove padding correctly?
6. If hybrid attention is enabled, were `pcp_enter_fa_restore_idx` and
   `pcp_exit_fa_scatter_idx` applied at the correct boundaries?
7. If backend PCP is enabled, are head/tail outputs restored with `q_full_idx`?
8. If multiple attention paths are merged, was LSE-aware weighting used?
9. Did slot mapping use the same CP topology as the attention path?
10. Did backend `supports_pcp` reflect real support?

## Final Target

The final implementation should let users control Qwen3.5 prefill context
parallelism with:

```bash
--prefill-context-parallel-size N
```

without requiring:

```bash
--enable-lightly-cp
```

and with this behavior:

```text
N == 1:
  no PCP token split

N > 1:
  full attention split across PCP ranks
  full attention output restored to original token order
  linear attention remains full-token
  TP communication remains separate from PCP communication
  final backend path uses Head/Tail CP attention with LSE-aware merges
```
