---
name: vllm-hcu-aiter-verify
description: |
  AITER MoE 通用自迭代验证。
  Agent 自动执行 修改→启动→测试→判定→再修改 的闭环，
  适用于量化/非量化 MoE 模型，每轮日志归档到 agent-log/ 目录。
license: MIT
---

# AITER MoE 自迭代验证

自动迭代验证 AITER MoE 推理路径，适用于量化（W8A8 等）和非量化模型，直到通过或达到最大迭代次数。

**前置**：需要提供两个可执行的 serve 脚本：

- `serve_baseline.sh` — 无 AITER kernel 的 serve 启动脚本，用于 Step 0 基线评测
- `serve.sh` — 启用 AITER kernel 的 serve 启动脚本，用于后续每轮迭代

两个脚本除 AITER 开关外，其余参数（模型路径、GPU、量化参数等）应保持一致。Agent 不负责构造 serve 命令。

## 迭代总循环

```
ITER=1, MAX_ITER=5
while ITER <= MAX_ITER:
    1. [MODIFY]    根据上一轮失败原因修改
    2. [SERVE]     启动 serve，日志归档到 agent-log/iter_{ITER}/
    3. [TEST]      发送测试请求
    4. [VERIFY]    4a-4d 基本功能检查
    5. [EVALSCOPE] 若 4a-4d 全 PASS，HumanEval 精度验证
    6. [DECIDE]    PASS → 结束；FAIL → 分析原因，ITER++，回步骤1
```

日志目录：`agent-log/iter_NNN/{serve.log, test_request.json, evalscope_results.json, analysis.md}`

## 0. BASELINE — 无 AITER 精度基线

**仅执行一次，在迭代循环之前。**

用 `serve_baseline.sh` 启动无 AITER kernel 的 serve，跑一次 HumanEval 全量评测作为精度基线：

1. 启动无 AITER 的 serve：直接执行 `bash serve_baseline.sh`
2. 确认 serve 启动成功
3. 运行 evalscope HumanEval 全量 164 题（参数同 5b，batch-size=32）
4. 将精度结果写入 `agent-log/baseline_accuracy.txt`（格式：`0.XXXX`）

后续每轮迭代的 5c 精度判定以此基线为准，而非硬编码阈值。

**基线只需跑一次。** 若因代码改动（非 aiter 相关）导致基线失效，需重新跑基线。

## 1. MODIFY — 按失败原因修改

**第一轮不改代码。只检查 serve.sh 配置是否合理。**

后续迭代根据上一轮 serve.log和analysis.md进行下一次修改，每轮在 analysis.md 记录：改了什么文件、为什么改。


## 2. SERVE — 启动服务

```bash
ITER_DIR=./agent-log/iter_$(printf '%03d' $ITER)
mkdir -p "$ITER_DIR"
bash serve.sh 2>&1 | tee "$ITER_DIR/serve.log"
```
- 600s 内没出现 "Uvicorn running" → serve 启动失败

## 3. TEST — 测试请求

```bash
curl -s http://localhost:9645/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "模型路径（与 serve.sh 一致）",
       "prompt": "Hello, who are you?", "max_tokens": 50, "temperature": 0}' \
  | tee "$ITER_DIR/test_request.json"
```

连接拒绝最多重试 3 次，每次等 10s。

## 4. VERIFY — 基本功能判定

**全部 4 项 PASS 才算通过。**

- **4a serve 启动**：`grep -q "Uvicorn running" "$ITER_DIR/serve.log"`
- **4b aiter moe 路径**：日志中出现 aiter moe 相关调用痕迹
- **4c 推理成功**：HTTP 200，choices 有文本
- **4d 输出合理**：前 10 个 token 不全是重复（unique_ratio >= 0.3）

## 5. EVALSCOPE — 精度对比验证

**仅当 4a-4d 全 PASS 才执行，否则跳过直接进 DECIDE。**

### 5a 安装

```bash
pip install evalscope
```

### 5b 运行

```bash
evalscope eval \
  --model "模型名称" \
  --eval-type openai_api \
  --api-url "http://localhost:9645/v1" \
  --api-key EMPTY \
  --generation-config temperature=0,max_tokens=10240,top_p=0.95 
  --datasets humaneval \
  --eval-batch-size 32 \
  --work-dir "$ITER_DIR/evalscope" \
  2>&1 | tee "$ITER_DIR/evalscope.log"
```

HumanEval 共 164 题全量评测。batch-size=32。耗时 5-15 分钟，调试时用 limit=20 快速迭代。

### 5c 精度判定

**与基线对比，而非硬编码阈值。基线来自 Step 0 的无 AITER 评测结果。**

```bash
BASELINE_FILE=./agent-log/baseline_accuracy.txt

python3 -c "
import re

def extract_accuracy(log_path):
    with open(log_path) as f: log = f.read()
    matches = re.findall(r'(?:accuracy|score|correct|pass@1)[^\d]*?(\d+\.?\d*)', log, re.I)
    if not matches:
        return None
    scores = [float(m) for m in matches]
    for s in reversed(scores):
        if 0 < s <= 1: return s
        elif 1 < s <= 100: return s / 100
    return None

# 读取基线
try:
    with open('$BASELINE_FILE') as f:
        baseline = float(f.read().strip())
    print(f'Baseline (AITER OFF): {baseline*100:.1f}%')
except:
    print('FAIL: baseline not found — run Step 0 first')
    exit(1)

# 当前轮精度
acc = extract_accuracy('$ITER_DIR/evalscope.log')
if acc is None:
    log_text = open('$ITER_DIR/evalscope.log').read().lower()
    if 'error' in log_text or 'traceback' in log_text:
        print('FAIL: evalscope error')
    else:
        print('WARN: no score found — manual check needed')
    exit(1)

diff = (acc - baseline) * 100
status = 'PASS' if acc >= baseline else 'FAIL'
print(f'AITER ON : {acc*100:.1f}%')
print(f'Delta    : {diff:+.1f}%')
print(f'Result   : {status}')
" 2>&1
```

**基线**：Step 0 中无 AITER kernel 时测得的 HumanEval pass@1，存在 `agent-log/baseline_accuracy.txt`。

**判定**：AITER ON 精度 ≥ 基线精度 → PASS；否则 FAIL。

精度低于基线的根因方向：
- `aiter_moe()` 中 token indices / expert weights / dtype 是否正确
- 若为量化模型，检查量化参数解析和 weight 转换是否与原始路径一致
- 对比 AITER ON vs OFF 的逐题输出差异，定位具体哪些题退化

## 6. DECIDE — 分支决策

**PASS（4a-4d 全过 + evalscope PASS/WARN）** → 结束。写入 summary：
```
## Iter NNN — SUCCESS
- 修改: [做了什么]
- 功能: 4a-4d 全 PASS
- 精度: [PASS/WARN — AITER ON XX% vs 基线 YY%, Delta ±ZZ%]
```

**FAIL (precision)（4a-4d 过但 evalscope FAIL，AITER ON 精度低于基线）** → 精度不达标。对比逐题输出差异、查 dtype/索引。写入 summary 后 ITER++。

**FAIL (functional)（4a-4d 任一 FAIL）** → 跳过 evalscope。写 analysis.md（含 4a-4d 结果 + 根因分析 + 修改方向），写入 summary 后 ITER++。

**ITER > 3** → 停止，报告需人工介入。

## 判定速查

| 检查项 | PASS 条件 | FAIL 含义 |
|--------|---------|---------|
| 4a | 日志含 "Uvicorn running" | GPU/显存/配置 |
| 4b | 日志含 aiter moe 调用痕迹 | 未进入 aiter moe 路径 |
| 4c | HTTP 200 + choices 有文本 | 推理流程异常 |
| 4d | 非纯重复 token | aiter moe 计算有误 |
| 5 | AITER ON pass@1 ≥ 基线 (AITER OFF) | AITER 引入精度退化 |

## 执行注意事项

- **每轮前杀旧进程**：`pkill -f "vllm serve"` 或 `kill $(lsof -t -i:9645)`
- **serve 启动等 30-60s**，不要并行启动多个
- **改代码后确认生效**：`_site` 下的安装包确认修改到了；源码目录确认 PYTHONPATH
- **记录一切**：serve.log、test_request.json、evalscope.log、analysis.md
- **evalscope 全量 164 题 5-15min**，调试用 limit=20

