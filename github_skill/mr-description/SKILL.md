---
name: mr-description
description: 分析当前分支与目标分支的差异，生成标准规范的 Merge Request 描述。
---

# MR Description — 生成 Merge Request 描述

分析当前分支与目标分支的差异，生成结构化的 MR 描述供 Code Review 使用。

## 工作流程

### 1. 确认分支信息

先确认目标分支（默认 `v0.21.0`，可从用户输入获取），然后获取变更概览：

```bash
# 当前分支和目标分支
git branch --show-current
# 变更文件列表 + 统计
git diff <target_branch>...HEAD --stat
# commit 列表
git log <target_branch>..HEAD --oneline
```

### 2. 分析变更

对每个 commit 提取：
- **标题**：commit subject
- **改动文件及内容**：`git diff <target_branch>...HEAD <file>` 了解关键变更

汇总并归类：
- 新增功能
- Bug 修复
- 重构/代码优化
- 配置/环境变量变更

### 3. 生成 MR 描述

严格按照以下格式，**用中文**：

```markdown
## 变更概要

从 `main` 分支 cherry-pick N 个 commit 到 `v0.21.0`：

1. **`<short_sha>`** <commit subject>
2. **`<short_sha>`** <commit subject>

## 改动内容

### <分类标题>（commit X & Y）

- 每个改动点用一句话说明做了什么、为什么做
- 列出新增的文件
- 列出新增的测试（如有）

### <另一个分类>（commit Z）

- ...

## 影响的模块

| 模块 | 变更类型 |
|---|---|
| `path/to/file.py` | 描述（如"新增函数"、"逻辑重构"） |
| ... | ... |

## 验证方式

- [ ] 新增测试通过
- [ ] 具体功能验证点 1
- [ ] 具体功能验证点 2
- [ ] 端到端推理回归

## 风险评估

- **低/中/高**：一句话说明主要风险点
- 重点关注场景或已知局限
```

### 4. 规则

- **标题建议**：`[<目标分支简写>] <变更主题概括>`
  - 例如：`[021] Flash Attention 后端模式 & MoE 后端显式选择合入`
- **变更概要**：一行一个 commit，格式为 `` `<short_sha>` `` + subject
- **改动内容**：按功能聚合 commit，不逐条罗列文件；用"问题/解决方案"结构说明动机
- **影响模块**：用表格列出关键变更文件，标注变更类型（新增/重构/适配/修复）
- **验证方式**：用 checkbox 格式，标注哪些已完成 `[x]`，哪些待验证 `[ ]`
- **风险评估**：给出等级（低/中/高）+ 一句话说明 + 需关注的具体场景

### 5. 输出

将生成的 MR 描述展示给用户，包括：
- MR 标题建议
- 完整 MR 描述正文

不自动执行任何 git 操作。
