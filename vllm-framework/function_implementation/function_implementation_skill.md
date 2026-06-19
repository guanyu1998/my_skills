# Qwen3.5 PCP (Prefill Context Parallelism) Implementation Skill

> 源文档: `_site_mooncake_pcp_qwen3_5/PCP_PREFILL_CONTEXT_IMPLEMENTATION_PLAN.md`
> 原则: 遵循 CLAUDE.md — 简单优先、外科手术式改动、目标驱动验证

---

## 核心思路

### 什么是 PCP
PCP = Prefill Context Parallelism，将 prefill 阶段的 token 按 PCP 维度拆分到多个 rank 并行计算，减少单 rank prefill 负载。

### 关键约束
- **PCP ≠ TP**: PCP 拆分 token，TP 拆分 hidden_size。两者是独立的并行维度，不能混用。
- **Hybrid Attention**: Qwen3.5 同时有 full-attention 和 linear-attention 层，两者对 PCP 的处理不同。
- **Linear Attention 是有状态的**: GDN (GatedDeltaNet) 是递归模型，后续 chunk 依赖前序 chunk 的最终 state。
- **死锁风险**: PCP 通信必须严格按 rank 顺序进行，不能让不同 rank 以不同顺序进入 collective。

### 最终目标
```bash
--prefill-context-parallel-size N   # 控制 PCP
# 不再需要 --enable-lightly-cp
```
```
N == 1: 无 PCP 拆分
N > 1:  full-attention 按 PCP rank 拆分 token，计算后 gather 恢复原始顺序
        linear-attention 保持 full-token 输入
        TP 通信与 PCP 通信完全独立
```

### 设计参考
- 参考 `vllm-ascend` 的 PCP 实现，但不要照搬
- 关键借鉴: DualChunkSwap、PCPManager、Head/Tail CP attention、LSE-aware merge

---

## 禁止事项 (红线)

1. **禁止用 `tensor_model_parallel_all_gather` 做 PCP token 恢复** — 用 PCP group
2. **禁止用 `get_tensor_model_parallel_world_size()` 作为 PCP split size**
3. **禁止在 Phase 6 之前让 linear attention 消费 PCP-local token**
4. **禁止用简单加法/平均合并多路 attention 输出** — 必须用 LSE-aware merge
5. **禁止静默等价 `--enable-lightly-cp` 和 native PCP**
6. **禁止在没有正确性测试的情况下启用 linear-attention PCP**
7. **禁止忽略 hybrid attention 的 layout transition**
8. **禁止假设 TP rank 和 PCP rank 有相同 layout**
9. **禁止修改 KV cache slot mapping 而不测试 decode**
10. **禁止绕过 `supports_pcp` 断言而不验证 backend 语义**

---

## 通用验证策略

### 每个 Phase 必须有
1. **pytest 单元测试** — 纯 CPU，无 NPU 依赖
2. **py_compile 检查** — 确保无语法错误
3. **回归测试** — 跑通之前所有 Phase 的测试

### 测试运行命令模板
```bash
# 编译检查
/public/home/guanyu1/.local/bin/python3.11 -m py_compile <changed_files>

# 单元测试
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest tests/test_pcp_<phase>.py -v

# 全量 PCP 回归
uv run --with pytest --python /public/home/guanyu1/.local/bin/python3.11 \
  python -m pytest tests/test_qwen35_pcp_*.py tests/test_pcp_*.py -q
```

### 测试覆盖矩阵
```
pcp_size=1, tp_size=1   (无 PCP)
pcp_size=1, tp_size=2   (纯 TP)
pcp_size=2, tp_size=1   (纯 PCP)
pcp_size=2, tp_size=2   (PCP + TP 混合)
pcp_size=4 (如果硬件允许)
混合 decode + prefill batch
非整除 token 数量
短序列 (padding-only rank)
```

---

## Phase 拆分与任务清单

### ━━━ 第一阶段: 基础设施 (Phase 1-3) ━━━

---

#### Phase 1: PCP Split/Restore 工具函数

**目标**: 创建 PCP 专用的 token 拆分/恢复工具，与 TP 路径完全独立。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 1.1 | 创建 `build_pcp_scatter_indexes()` — DualChunkSwap 算法 | pytest: pcp_size=1,2,4 |
| 1.2 | 创建 `build_pcp_gather_restore_indexes()` — allgather 恢复索引 | pytest: 恢复后顺序一致 |
| 1.3 | 创建 `build_pcp_unpad_mask()` — 标记 padding 位置 | pytest: padding 位置正确 |
| 1.4 | 创建 `build_pcp_local_positions()` — PCP-local 位置索引 | pytest: 局部位置连续 |
| 1.5 | 创建 `build_pcp_local_q_kv_lens()` — local query/KV 长度 | pytest: 长度计算正确 |
| 1.6 | 组合函数 `build_pcp_split_restore_metadata()` | pytest: 一站式生成全部元数据 |
| 1.7 | 集成测试: mixed decode + prefill | pytest: decode 不拆分 |
| 1.8 | 集成测试: 非整除 token 数量 | pytest: padding 正确 |

**DualChunkSwap 算法**:
```python
# 每个 prefill request:
# 1. padding 到 2 * pcp_world_size 的倍数
# 2. 切成 2 * pcp_world_size 个 chunk
# 3. rank r 取: head chunk r + tail chunk (2*N - 1 - r)
# 4. decode 不拆分，复制到所有 rank
```

**关键验证**:
- `pcp_size=1` 时不拆分
- gather + restore 后恢复原始 token 顺序
- 所有 rank 计算出相同的 restore index
- padding 位置用 `-1` 标记，不参与计算

**文件位置**: `vllm_hcu/v1/attention/lightly_cp_utils.py`

**风险点**:
- DualChunkSwap 的 head/tail chunk 分配要正确
- padding 对齐必须是 `2 * pcp_size` 的倍数
- restore index 必须考虑 padding entries

---

#### Phase 2: Qwen3.5 Full Attention 切换到 PCP Group

**目标**: 让 Qwen3.5 full-attention 层使用 PCP group 而非 TP group 进行 token 拆分。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 2.1 | 在 `qwen3_5.py` 中添加 `get_pcp_group()` 获取逻辑 | pytest: group 获取正确 |
| 2.2 | full_attention 分支: 用 PCP scatter 拆分 hidden_states | pytest: shape 正确 |
| 2.3 | full_attention 分支: self_attn 在 local tokens 上执行 | pytest: local token 数变化 |
| 2.4 | full_attention 分支: PCP all_gather 恢复 | pytest: 恢复后 shape 正确 |
| 2.5 | full_attention 分支: restore original token order | pytest: 顺序一致 |
| 2.6 | linear_attention 分支: 保持 full-token 不变 | pytest: shape 不变 |
| 2.7 | pcp_size == 1 时走原路径 | pytest: 无行为变化 |

**关键验证**:
```python
# 加临时 debug log:
# before split: hidden_states.shape = [total_tokens, hidden_size]
# after split:  hidden_states.shape = [local_tokens, hidden_size]
# after attn:   hidden_states.shape = [local_tokens, hidden_size]
# after gather: hidden_states.shape = [total_tokens, hidden_size]
```

**关键不变量**:
- gather 只在 token 维度，不在 hidden 维度
- 不用 TP group 做 PCP 恢复
- 每个 full-attention 层之后，hidden states 必须恢复到 full-token 长度
- linear attention 始终看到 full-token hidden states

**风险点**:
- ⚠️ PCP all_gather 通信 — 确保用 PCP group 而非 TP group
- ⚠️ shape 对齐 — split 后 local_tokens = total_tokens / pcp_size (含 padding)

---

#### Phase 3: 从 `prefill_context_parallel_size` 驱动 PCP

**目标**: 用原生 vLLM PCP size 作为控制开关，不再依赖 `--enable-lightly-cp`。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 3.1 | 在 `parallel_config` 中确认 `prefill_context_parallel_size` 字段 | source test |
| 3.2 | 在 runner 中添加 `enable_pcp_for_prefill` 派生逻辑 | pytest: 条件正确 |
| 3.3 | 在 forward_context 中暴露 `enable_pcp_for_prefill` | pytest: 字段存在 |
| 3.4 | `qwen3_5.py` 使用新 PCP 字段 | pytest: full_attn 走 PCP |
| 3.5 | 保留旧 `enable_lightly_cp` 路径不删除 | source test: 旧路径存在 |

**控制流**:
```
parallel_config.prefill_context_parallel_size > 1
  -> runner.enable_pcp_for_prefill = True
  -> forward_context.enable_pcp_for_prefill = True
  -> qwen3_5.py: full_attention 走 PCP 分支
```

**验证矩阵**:
```bash
--tensor-parallel-size 1 --prefill-context-parallel-size 1  # 无 PCP
--tensor-parallel-size 2 --prefill-context-parallel-size 1  # 纯 TP
--tensor-parallel-size 1 --prefill-context-parallel-size 2  # 纯 PCP
--tensor-parallel-size 2 --prefill-context-parallel-size 2  # 混合
```

**风险点**:
- ⚠️ 不要让 `prefill_context_parallel_size > 1` 意外使用 TP group
- ⚠️ 命名要清晰: `lightly_cp` vs native PCP vs DCP

---

### ━━━ 第二阶段: Backend 接入 (Phase 4) ━━━

---

#### Phase 4A: PCP Backend Bootstrap

**目标**: 让 native PCP 初始化通过 FlashAttention 兼容性检查。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 4A.1 | `FlashAttentionMetadata` 添加 `pcp_enabled` 字段 | pytest: 字段存在 |
| 4A.2 | `FlashAttentionImpl.supports_pcp = True` | pytest: 启动检查通过 |
| 4A.3 | PCP-local `slot_mapping` 构建 (padding → `-1`) | pytest: 长度/值正确 |
| 4A.4 | 从 `pcp_scatter_indexes_tensor` 推导 slot_mapping | pytest: 映射正确 |

**关键验证**:
- native PCP 不再因 `FlashAttentionImpl does not support PCP` 启动失败
- PCP-local `slot_mapping` 长度匹配 local Q/K/V token layout
- padding entries 映射到 slot `-1`

**Out of scope**: 真正的 Head/Tail attention、nomask/mask KV、LSE merge

---

#### Phase 4B: Hybrid Metadata Scope 隔离

**目标**: linear-attention 用 global metadata，full-attention 用 PCP-local metadata。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 4B.1 | forward_context 添加 `pcp_attn_metadata` 字段 | pytest: 字段存在 |
| 4B.2 | runner 分别构建 global 和 PCP-local metadata | pytest: 两份 metadata 不同 |
| 4B.3 | `qwen3_5.py` linear_attn 用 global metadata | pytest: 不看到 PCP metadata |
| 4B.4 | `qwen3_5.py` full_attn 用 `pcp_attn_metadata` | pytest: 看到 PCP metadata |

**观测到的失败模式** (如果不做此 Phase):
```
IndexError: The shape of the mask [2] at index 0 does not match
the shape of the indexed tensor [1, 32, 128, 128] at index 0
```
原因: GDN linear attention 读取了 PCP-local 的 `has_initial_state` mask 和 state indices。

**关键验证**:
- linear-attention 层不看到 PCP-local metadata
- full-attention 层收到 PCP scatter/gather metadata
- serve 路径 GDN 不再报 shape mismatch

---

#### Phase 4C: 真正的 Full-Attention PCP Backend

**目标**: 从 wrapper-level scatter/gather 升级到 attention-backend 级别的 PCP。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 4C.1 | 构建 head/tail query chunk 索引 (`q_head_idx`, `q_tail_idx`, `q_full_idx`) | pytest: pcp_size=2,4 |
| 4C.2 | 构建 head/tail KV 索引 (nomask/mask) | pytest: 索引正确 |
| 4C.3 | 构建 head/tail seqlens | pytest: 长度正确 |
| 4C.4 | 实现 LSE-aware merge 函数 | pytest: 对比 softmax 参考 |
| 4C.5 | backend PCP 入口路径 (在 default paged-attention 之前) | pytest: 路由正确 |
| 4C.6 | 集成: full-attention 输出数值匹配 non-PCP | pytest: 数值容差内 |
| 4C.7 | 集成: mixed decode/prefill 行为正确 | pytest: 无错误 |

**Head/Tail CP Attention 算法**:
```
head path:
  Q = q_head_idx → 对应 head chunk 的 query
  KV_nomask = 之前完全可见的 chunks (无需 causal mask)
  KV_mask = 当前 head chunk (需要 causal mask)

tail path:
  Q = q_tail_idx → 对应 tail chunk 的 query
  KV_nomask = 之前完全可见的 chunks
  KV_mask = 当前 tail chunk

post:
  concat head/tail 输出
  用 q_full_idx 恢复原始 query 顺序
```

**LSE-aware Merge (关键)**:
```python
# 绝对不能用简单加法或平均!
lse_final = logsumexp(lse_i)          # 各路 log-sum-exp
out_final = sum(exp(lse_i - lse_final) * out_i)  # 加权合并
```

**⚠️ 缺失算子用 Triton 实现**:
- 如果 HCU 平台没有对应的 flash attention 算子
- 用 Triton 实现 head/tail CP attention kernel
- 优先实现简单的 paged attention + causal mask
- 确保首先跑通，再优化性能

---

### ━━━ 第三阶段: KV Cache 对齐 (Phase 5) ━━━

---

#### Phase 5: KV Cache 和 Slot Mapping 对齐

**目标**: 确保 PCP/DCP 拓扑下 KV cache 物理布局和 slot mapping 一致。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 5.1 | 审计 block_table 和 slot_mapping 在 PCP 下的行为 | source review |
| 5.2 | slot_mapping 考虑 `total_cp_rank = pcp_rank * dcp_world + dcp_rank` | pytest |
| 5.3 | 确保 prefill 写入的 KV 位置 = decode 读取的位置 | pytest: prefill+decode |
| 5.4 | `block_table_tensor`, `slot_mapping`, `seq_lens` 一致 | pytest |
| 5.5 | 测试 PCP only / DCP only / PCP*DCP / padding / mixed | pytest |
| 5.6 | 短 prefill + decode 输出对比 pcp_size=1 | pytest: 数值容差内 |

**关键风险**:
- ⚠️ hybrid attention 可能有多个 KV cache group，page size 不同
- ⚠️ linear attention 的 state 不是 KV cache，不能假设相同 layout
- ⚠️ prefill 写入位置必须与 decode 读取位置一致，否则 decode 读到垃圾数据

---

#### Phase 5B: Hybrid Hidden-State Layout Switching

**目标**: 为 hybrid attention 边界添加显式的 layout transition metadata。

**任务拆分**:

| # | 任务 | 验证 |
|---|------|------|
| 5B.1 | 构建 `pcp_enter_fa_restore_idx` (linear→full layout) | pytest |
| 5B.2 | 构建 `pcp_exit_fa_scatter_idx` (full→linear layout) | pytest |
| 5B.3 | 在 forward_context 中暴露 hybrid layout metadata | pytest |
| 5B.4 | `qwen3_5.py` 验证 metadata 可用但不改变执行语义 | pytest |

**当前不变量** (Phase 6 前):
- linear attention 始终接收 full-token hidden states
- full attention 内部可以 PCP-local，但输出必须恢复后再传给 post-attention layernorm/MLP

---

### ━━━ 第四阶段: Linear Attention PCP (Phase 6A-6Y) ━━━

> **这是最复杂的部分。Linear attention 有状态，需要跨 rank 传递 recurrent state。**
> **严格按顺序实现，每一步都有 pytest 验证。**

---

#### Phase 6A: Safety Guard

**目标**: 添加安全守卫，防止未完成的 linear PCP 被意外启用。

| # | 任务 | 验证 |
|---|------|------|
| 6A.1 | `enable_pcp_for_linear_attention` 默认 `False` | pytest |
| 6A.2 | 启用时抛出 `NotImplementedError` | pytest |
| 6A.3 | linear attention 继续接收 full-token hidden states | pytest |

---

#### Phase 6B: Linear Attention PCP Reference Contract

**目标**: 定义 linear attention PCP 的 metadata 契约，不改变执行路径。

| # | 任务 | 验证 |
|---|------|------|
| 6B.1 | 实现 `PCPLinearAttentionMetadata` | pytest: pcp_size=1,2 |
| 6B.2 | contiguous per-request chunk split (非 DualChunkSwap) | pytest |
| 6B.3 | decode tokens 保持 replicated | pytest |
| 6B.4 | state source/destination rank 元数据 | pytest |

**Linear Attention Token Layout** (与 Full Attention 不同):
```
# linear attention 用 contiguous chunks，不用 DualChunkSwap
# 因为 GDN state 有顺序依赖
prefill request tokens → pad to pcp_size → split into pcp_size contiguous chunks
rank r gets chunk r
rank r needs state from rank r-1 (when r > 0)
```

---

#### Phase 6C: Metadata Forward Context 透传

| # | 任务 | 验证 |
|---|------|------|
| 6C.1 | `pcp_linear_attention_metadata` 透传到 forward context | pytest |
| 6C.2 | Qwen3.5 linear attention 可观察但不消费 | pytest |

---

#### Phase 6D: Reference State Handoff

**目标**: 证明 contiguous split + state handoff 能复现串行 recurrent 计算。

| # | 任务 | 验证 |
|---|------|------|
| 6D.1 | `scatter_pcp_linear_attention_input()` — padding 置零 | pytest |
| 6D.2 | `restore_pcp_linear_attention_output()` | pytest |
| 6D.3 | `run_pcp_linear_recurrent_reference()` — 逐 rank 传 state | pytest: 对比串行 |
| 6D.4 | 验证 reject multi-sequence/decode | pytest |

**⚠️ 状态传递顺序关键**:
```
rank 0: process tokens[0..N/2] → final_state_0
rank 1: recv final_state_0 → process tokens[N/2..N] → final_state_1
# 必须严格按顺序，否则死锁或结果错误
```

---

#### Phase 6E: GDN State Handoff Plan

| # | 任务 | 验证 |
|---|------|------|
| 6E.1 | `PCPLinearStateHandoffPlan` 数据结构 | pytest |
| 6E.2 | `build_pcp_linear_state_handoff_plan()` | pytest: 短序列 |
| 6E.3 | padding-only rank 跳过 state handoff | pytest |
| 6E.4 | `state_dst_ranks` 跳过 padding-only rank (不是 rank+1) | pytest |

---

#### Phase 6F: GDN State IO Contract

| # | 任务 | 验证 |
|---|------|------|
| 6F.1 | `PCPLinearStateSendPayload` 数据结构 | pytest |
| 6F.2 | `PCPLinearStateUpdate` 数据结构 | pytest |
| 6F.3 | `build_pcp_linear_state_send_payload()` | pytest |
| 6F.4 | `build_pcp_linear_final_state_broadcast_payload()` | pytest |
| 6F.5 | `build_pcp_linear_initial_state_update()` | pytest |
| 6F.6 | `build_pcp_linear_final_state_update()` | pytest |
| 6F.7 | state payload tensors 必须 detach + move to CPU | pytest |

**⚠️ GPU/CPU 通信安全**:
- `send_object` / `broadcast_object` 只传 CPU tensor
- cache update 前把 received payload 移到 local cache device/dtype
- 不要在 GPU tensor 上直接做 object-based communication，会死锁

---

#### Phase 6G: GDN Cache Update Contract

| # | 任务 | 验证 |
|---|------|------|
| 6G.1 | `apply_pcp_linear_state_update_to_cache()` | pytest: 选中行更新 |
| 6G.2 | 空 update 是 no-op | pytest |
| 6G.3 | received tensors 转换到 local device/dtype | pytest |

---

#### Phase 6H: Cache-to-Payload Contract

| # | 任务 | 验证 |
|---|------|------|
| 6H.1 | `select_pcp_linear_state_cache_rows()` | pytest |
| 6H.2 | `build_pcp_linear_state_send_payload_from_cache()` | pytest |
| 6H.3 | `build_pcp_linear_final_state_broadcast_payload_from_cache()` | pytest |
| 6H.4 | non-final rank 产生空 broadcast payload | pytest |

---

#### Phase 6I: Payload-to-Update Contract

| # | 任务 | 验证 |
|---|------|------|
| 6I.1 | `build_pcp_linear_initial_state_update_from_payload()` | pytest |
| 6I.2 | `build_pcp_linear_final_state_update_from_payload()` | pytest |
| 6I.3 | 按 `seq_indexes` 重排 received payload rows | pytest |
| 6I.4 | 缺失/重复 `seq_indexes` 失败清晰 | pytest |

---

#### Phase 6J: Point-to-Point Payload Routing

| # | 任务 | 验证 |
|---|------|------|
| 6J.1 | `merge_pcp_linear_state_payloads()` | pytest |
| 6J.2 | `route_pcp_linear_state_send_payloads()` | pytest |
| 6J.3 | 重复 `seq_indexes` 检测 | pytest |
| 6J.4 | 无效 destination rank 检测 | pytest |

---

#### Phase 6K: Final-State Broadcast Routing

| # | 任务 | 验证 |
|---|------|------|
| 6K.1 | `route_pcp_linear_final_state_broadcast_payloads()` | pytest |
| 6K.2 | broadcast payload 路由到所有 non-source rank | pytest |
| 6K.3 | 多 source rank 正确 merge | pytest |

---

#### Phase 6L: Runtime Handoff Plan Context

| # | 任务 | 验证 |
|---|------|------|
| 6L.1 | runner 构建 `pcp_linear_state_handoff_plan` | pytest |
| 6L.2 | forward context 暴露该字段 | pytest |
| 6L.3 | Qwen3.5 guard 要求同时有 metadata 和 plan | pytest |

---

#### Phase 6M: Pre/Post-Core State Exchange Contract

| # | 任务 | 验证 |
|---|------|------|
| 6M.1 | `apply_pcp_linear_initial_state_update_from_payload_to_cache()` | pytest |
| 6M.2 | `build_pcp_linear_post_core_state_payloads_from_cache()` | pytest |
| 6M.3 | `apply_pcp_linear_final_state_update_from_payload_to_cache()` | pytest |
| 6M.4 | `PCPLinearPostCoreStatePayloads` 数据结构 | pytest |

---

#### Phase 6N: State Exchange Adapter

| # | 任务 | 验证 |
|---|------|------|
| 6N.1 | `PCPLinearStateExchangePayloads` | pytest |
| 6N.2 | `exchange_pcp_linear_post_core_state_payloads()` | pytest |
| 6N.3 | 空 payload 是 no-op | pytest |

---

#### Phase 6O: Pre-Enable Readiness Validation

**目标**: 模拟完整的两-rank prefill handoff + final broadcast 流程。

| # | 任务 | 验证 |
|---|------|------|
| 6O.1 | 两-rank prefill handoff 测试 | pytest: caches 一致 |
| 6O.2 | 验证 linear PCP guard 仍然启用 | pytest |
| 6O.3 | 验证 handoff plan 在 forward context 中 | pytest |

---

#### Phase 6P: Forward-Core Hook Scaffold

| # | 任务 | 验证 |
|---|------|------|
| 6P.1 | `prepare_pcp_linear_forward_core_state()` (dormant) | pytest: no-op |
| 6P.2 | `finalize_pcp_linear_forward_core_state()` (dormant) | pytest: no-op |
| 6P.3 | Patch `qwen3_next._forward_core` 添加 hook 点 | source test |

---

#### Phase 6Q: Forward-Core Local State Hook

| # | 任务 | 验证 |
|---|------|------|
| 6Q.1 | forward context 添加 `pcp_linear_initial_state_payload` | pytest |
| 6Q.2 | forward context 添加 `pcp_linear_post_core_state_payloads` | pytest |
| 6Q.3 | prepare hook: apply initial state + mark `has_initial_state` | pytest |
| 6Q.4 | finalize hook: build/store post-core payloads | pytest |

---

#### Phase 6R: PCP Group State Exchange

| # | 任务 | 验证 |
|---|------|------|
| 6R.1 | `exchange_pcp_linear_post_core_state_payloads_with_group()` | pytest |
| 6R.2 | 用 `broadcast_object` 逐 rank 广播 | pytest |
| 6R.3 | group rank/world_size 校验 | pytest |

**⚠️ 死锁防护**:
- 用 `broadcast_object` 而非手动 send/recv 循环
- 所有 rank 必须以相同顺序进入 collective
- 如果某个 rank payload 为空，也要参与 broadcast

---

#### Phase 6S: Forward-Context State Exchange Bridge

| # | 任务 | 验证 |
|---|------|------|
| 6S.1 | forward context 添加 `pcp_linear_final_state_payload` | pytest |
| 6S.2 | `exchange_pcp_linear_forward_context_state()` | pytest |
| 6S.3 | disabled 时是 no-op | pytest |

**⚠️ 时序关键**:
- initial-state handoff 必须在 GDN core 之前
- 不能做 whole-model post-forward exchange — 太晚了

---

#### Phase 6T: Final-State Context Apply

| # | 任务 | 验证 |
|---|------|------|
| 6T.1 | `apply_pcp_linear_forward_context_final_state()` | pytest |
| 6T.2 | disabled 时不修改 cache | pytest |
| 6T.3 | apply 后清除 `pcp_linear_final_state_payload` | pytest |

---

#### Phase 6U: Runtime Post-Core Hook Composition

| # | 任务 | 验证 |
|---|------|------|
| 6U.1 | `finalize_pcp_linear_forward_core_state()` 扩展: 可选 `pcp_group` | pytest |
| 6U.2 | 有 pcp_group 时: build → exchange → apply | pytest |
| 6U.3 | Qwen3Next patch 传入 `get_pcp_group()` | source test |

---

#### Phase 6V: Rank-Ordered Initial-State Handoff ⚠️ 死锁关键

**目标**: 保证 rank r 在 GDN core 之前收到 rank r-1 的 final state。

| # | 任务 | 验证 |
|---|------|------|
| 6V.1 | `receive_pcp_linear_rank_ordered_initial_state()` | pytest |
| 6V.2 | `send_pcp_linear_rank_ordered_initial_state()` | pytest |
| 6V.3 | prepare hook: recv_object from source rank | pytest |
| 6V.4 | finalize hook: send_object to downstream rank(s) | pytest |
| 6V.5 | Qwen3Next patch 传入 pcp_group 到两个 hook | source test |

**⚠️⚠️ 死锁防护 — 最关键的部分**:
```
# 必须严格按 rank 顺序:
rank 0: core → send_object(rank1)
rank 1: recv_object(rank0) → core → send_object(rank2)
rank 2: recv_object(rank1) → core → ...
# 如果顺序错乱 → 死锁!
# 如果用 broadcast 代替有序 recv → 死锁!
```

**通信原语选择**:
- pre-core: `recv_object()` from previous rank (blocking)
- post-core: `send_object()` to next rank (blocking)
- 不要用 `all_gather` 或 `broadcast` 做 initial-state handoff
- `broadcast_object` 只用于 final-state broadcast (在所有 core 完成后)

---

#### Phase 6W: Runtime Guard Relaxation

| # | 任务 | 验证 |
|---|------|------|
| 6W.1 | 移除旧的无条件 `NotImplementedError` | source test |
| 6W.2 | 添加运行时契约检查 (metadata, plan, pcp_group) | pytest |
| 6W.3 | 默认仍然 disabled | pytest |

---

#### Phase 6X: Native PCP Enable Switch

| # | 任务 | 验证 |
|---|------|------|
| 6X.1 | 环境变量 `VLLM_HCU_ENABLE_PCP_LINEAR_ATTENTION` | pytest |
| 6X.2 | runner 派生 `enable_pcp_for_linear_attention` | pytest |
| 6X.3 | per-step effective switch 要求 metadata + plan | pytest |
| 6X.4 | profile/dummy run 保持 disabled | pytest |

---

#### Phase 6Y: Local Token Execution (首次 serve 验证)

**目标**: 修复 native linear PCP serve 时的正确性问题。

**观测到的问题**:
- serve 不 hang 了，但输出语义错误
- GDN state handoff 假设 local chunks，但 core 仍接收 full-token

| # | 任务 | 验证 |
|---|------|------|
| 6Y.1 | linear 分支: scatter hidden_states by PCP indexes | pytest |
| 6Y.2 | 丢弃 `-1` padding tokens | pytest |
| 6Y.3 | 构建 local GDN metadata (num_actual_tokens 等) | pytest |
| 6Y.4 | GDN core 在 local tokens 上执行 | pytest |
| 6Y.5 | local output pad 回 local_padded_q_len | pytest |
| 6Y.6 | all-gather + restore full-token order | pytest |
| 6Y.7 | 恢复 forward-context metadata | pytest |

**Serve 验证**:
```bash
VLLM_HCU_ENABLE_PCP_LINEAR_ATTENTION=1 bash serve-pcp.sh
bash client.sh
# 检查 log: mixed_qkv 应显示 local token counts (如 17 和 16)
# 而不是两个 rank 都显示 33
```

---

## 关键风险总结

### 1. 死锁风险
| 场景 | 原因 | 防护 |
|------|------|------|
| Phase 6V rank-ordered handoff | 不同 rank 以不同顺序进入 collective | 严格 rank 顺序 send/recv |
| Phase 6R group exchange | 某些 rank payload 为空不参与 | 所有 rank 都参与 broadcast |
| PCP + TP 混合通信 | PCP group 和 TP group 交错使用 | 严格分离通信维度 |

### 2. GPU/CPU 通信风险
| 场景 | 原因 | 防护 |
|------|------|------|
| object-based communication | GPU tensor 不能直接序列化 | 先 detach + move to CPU |
| cache update device mismatch | received payload 在错误 device | 更新前 move to local device |
| dtype mismatch | received payload dtype 不同 | 更新前转换到 cache dtype |

### 3. 正确性风险
| 场景 | 原因 | 防护 |
|------|------|------|
| LSE merge 错误 | 用简单加法代替 LSE | 严格实现 LSE-aware merge |
| slot_mapping 不一致 | prefill 写入 ≠ decode 读取 | 测试 prefill+decode 流程 |
| linear attention 收到 PCP metadata | metadata scope 泄漏 | Phase 4B 隔离 |
| DualChunkSwap 用于 linear attention | linear attention 需要 contiguous chunks | 分别处理 |

### 4. 缺失算子风险
| 场景 | 方案 |
|------|------|
| HCU 无 flash attention PCP 算子 | 用 Triton 实现，先跑通再优化 |
| HCU 无 LSE merge 算子 | 用 Triton 实现 `logsumexp` + weighted sum |
| HDN state scan 无跨 rank 版本 | 先用 reference 串行实现验证正确性 |

---

## 实现顺序建议

```
Phase 1  → pytest pass → commit
Phase 2  → pytest pass → commit
Phase 3  → pytest pass → commit (最小可用里程碑)
Phase 4A → pytest pass → commit
Phase 4B → pytest pass → commit
Phase 4C → pytest pass → commit (full-attention PCP 完成)
Phase 5  → pytest pass → commit
Phase 5B → pytest pass → commit
Phase 6A → pytest pass → commit
Phase 6B → 6C → 6D → 6E → 6F → 6G → 6H → 6I → 6J → 6K → 6L → 6M → 6N → 6O
  每步 pytest pass → commit (linear attention PCP 契约完成)
Phase 6P → 6Q → 6R → 6S → 6T → 6U → 6V
  每步 pytest pass → commit (linear attention PCP 运行时完成)
Phase 6W → 6X → 6Y
  每步 pytest pass → commit (linear attention PCP 启用)
```

**每个 commit 必须**:
1. 编译通过 (`py_compile`)
2. 本 Phase 测试通过
3. 所有之前 Phase 测试回归通过
4. 无新增 lint 警告

---

## 文件清单

### 修改文件
- `vllm_hcu/v1/attention/lightly_cp_utils.py` — PCP 工具函数主文件
- `vllm_hcu/models/qwen3_5.py` — Qwen3.5 wrapper
- `vllm_hcu/v1/hcu_model_runner.py` — HCU model runner
- `vllm_hcu/v1/attention/backends/flash_attn.py` — FlashAttention backend
- `vllm_hcu/patches/vllm__forward_context.patch.py` — forward context patch
- `vllm_hcu/patches/vllm__model_executor__models__qwen3_next.patch.py` — GDN hook patch
- `vllm_hcu/platforms/envs.py` — 环境变量
- `vllm/vllm/forward_context.py` — upstream forward context (如果需要)

### 测试文件
- `tests/test_pcp_split_restore_utils.py`
- `tests/test_qwen35_pcp_wrapper.py`
- `tests/test_qwen35_pcp_phase3_integration.py`
- `tests/test_qwen35_pcp_phase4_backend.py`
- `tests/test_qwen35_pcp_phase4b_metadata_scope.py`
- `tests/test_qwen35_pcp_phase5b_layout_switching.py`
- `tests/test_qwen35_pcp_phase6_linear_guard.py`
- `tests/test_qwen35_pcp_phase6b_linear_contract.py`
- `tests/test_qwen35_pcp_phase6d_linear_reference.py`
- `tests/test_qwen35_pcp_phase6e_linear_state_handoff.py`
- `tests/test_qwen35_pcp_phase6f_linear_state_io.py`
- `tests/test_qwen35_pcp_phase6g_linear_cache_update.py`
- `tests/test_qwen35_pcp_phase6h_linear_cache_payload.py`
- `tests/test_qwen35_pcp_phase6i_linear_payload_update.py`
- `tests/test_qwen35_pcp_phase6j_linear_payload_routing.py`
- `tests/test_qwen35_pcp_phase6k_linear_broadcast_routing.py`
- `tests/test_qwen35_pcp_phase6l_linear_plan_context.py`
- `tests/test_qwen35_pcp_phase6m_linear_state_exchange.py`
- `tests/test_qwen35_pcp_phase6n_linear_state_exchange_adapter.py`
- `tests/test_qwen35_pcp_phase6o_linear_pre_enable_readiness.py`
- `tests/test_qwen35_pcp_phase6p_linear_forward_core_hooks.py`
- `tests/test_qwen35_pcp_phase6r_linear_group_state_exchange.py`
- `tests/test_qwen35_pcp_phase6s_linear_forward_context_exchange.py`
- `tests/test_qwen35_pcp_phase6t_linear_final_state_context_apply.py`
- `tests/test_qwen35_pcp_phase6u_linear_runtime_post_core_hook.py`
- `tests/test_qwen35_pcp_phase6v_linear_rank_ordered_handoff.py`
- `tests/test_qwen35_pcp_phase6w_linear_guard_relaxation.py`
- `tests/test_qwen35_pcp_phase6x_linear_enable_switch.py`

---

## 快速诊断清单

失败时按顺序检查:

1. **哪个 flag 启用了路径?** `--enable-lightly-cp` vs `--prefill-context-parallel-size`
2. **用了哪个 group?** TP group vs PCP group
3. **Qwen3.5 full attention 周围的 shapes?** before split → after split → after attn → after gather
4. **Linear attention 收到了 full-token hidden states?**
5. **gather/restore index 正确移除了 padding?**
6. **Hybrid attention: `pcp_enter_fa_restore_idx` 和 `pcp_exit_fa_scatter_idx` 在正确边界?**
7. **Backend PCP: `q_full_idx` 恢复了 head/tail 输出?**
8. **多路 attention 合并用了 LSE weighting?**
9. **slot_mapping 与 attention path 使用相同的 CP 拓扑?**
10. **backend `supports_pcp` 反映了真实支持?**
