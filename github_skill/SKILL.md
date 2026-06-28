---
name: commit
description: 分析当前 diff 生成标准规范的 commit message，然后执行 git commit -s 提交。
---

# Commit — 标准 Git 提交

分析当前工作目录的 staged/unstaged 变更，按项目 Conventional Commits 规范生成 commit message，然后执行 `git commit -s`。

## 工作流程

### 1. 分析变更

先执行 `git diff --staged`，如果为空则 fallback 到 `git diff`。同时执行 `git diff --stat` 获取概览。

提取信息：
- **Type**: `feat`（新功能）、`fix`（修 bug）、`perf`（性能）、`refactor`（重构）、`test`（测试）、`docs`（文档）、`chore`（杂务）
- **Scope**（可选）：受影响的模块/组件（如 `npu`、`model_runner`、`attention`）
- **修改文件**：列出所有修改的文件，每个文件一句话说明改了什么

### 2. 生成 commit message

严格按照以下格式：

```
<type>(<scope>): <中文摘要>

<body — 说明改了什么、为什么改>

Signed-off-by: git config user.name <git config user.email>
```

**规则：**
- 标题行 ≤ 72 字符，中文用动宾结构（如"修复 xxx"、"新增 xxx 支持"），不加句号
- Body 说明**改了什么**和**为什么改**，不写怎么改
- 末尾列出修改文件及每个文件的一句话总结
- 如果是解决某个具体问题，用"问题："段落描述
- 如果引入了新的机制或方案，用"解决方案："段落描述
- 如果有环境变量或配置变更，用"环境变量："段落描述
- 强制带上 `Signed-off-by:` 行（使用 `git config user.name` 和 `git config user.email`）

**示例：**

```
perf(causal_conv1d): H2D copy stream 优化，消除 build 阶段空泡

问题：
- batch_ptr.copy_() 在默认 stream 上执行，H2D copy 阻塞 GPU
- 导致 2-19ms 空泡

解决方案：
- 预分配 GPU buffer 和 pinned CPU staging buffer，避免每次 CUDA malloc
- 使用独立 copy stream 异步传输，和计算 stream 并行
- 4-slot ring buffer 复用，三方并行互不干扰
- kernel 启动前 wait_event 确保数据传输完成

环境变量：
- VLLM_HCU_GDN_CAUSAL_CONV1D_COPY_STREAM=True 开启优化（默认 False）

修改文件：
- gdn_attn.patch.py: 预分配 buffer + copy stream 调用
- utils.patch.py: compute_causal_conv1d_metadata 支持 copy stream
- causal_conv1d.patch.py: kernel 启动前 wait_event
- envs.py: 添加环境变量定义

Signed-off-by: Lai Bao <laibao@example.com>
```

### 3. 展示并确认

将生成的 commit message 展示给用户，确认后再提交。

### 4. 执行提交

```bash
git commit -s -m "<subject>" -m "<body>"
```

如果 body 中已经包含 `Signed-off-by:` 行，直接用 `git commit -m "..."` 避免重复。

## 常用模式

- **性能优化**: `perf(scope): ...` + 问题/解决方案 结构
- **修 bug**: `fix(scope): ...` + 描述根因和修复
- **新功能**: `feat(scope): ...` + 功能描述和使用方式
- **重构**: `refactor(scope): ...` + 描述简化了什么、为什么
