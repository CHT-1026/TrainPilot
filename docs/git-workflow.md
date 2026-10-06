# Git 开发与 GitHub 展示规范

## 当前仓库

独立仓库位于 `E:\agent项目\trainpilot`，默认分支为 `main`。当前不预设 GitHub 用户名或远程地址。完成本地初始化后，通过 `git log --oneline` 查看实际提交。

## 提交方式

每次提交对应一个可解释的成果。先运行 `git diff` 检查修改，再用 `git add <具体路径>` 暂存。不要将数据、权重、密钥或全部运行日志放入 Git。

建议提交类型：`docs` 写方案与报告，`feat` 增加能力，`fix` 修复行为，`perf` 做性能改进，`test` 增加有意义的验证，`chore` 更新工具与配置。

示例：`feat(training): add fixed validation split`；`perf(training): overlap data transfer after profiling`。提交说明应写出问题、改动与验证方法，不虚构测量结果。

## 分支与里程碑

第一阶段可在 main 上做小步文档提交。涉及完整模块时创建 `feat/training-baseline`、`feat/experiment-runner` 或 `feat/diagnostic-agent` 分支。进入 main 前检查差异和相应验证；如使用 PR，在正文附验证与效果证据。

真实完成训练基线后再创建 `v0.1.0-baseline` 标签；完成 Agent 闭环和评测后再创建 `v0.2.0-demo`。不提前生成表示已实现功能的标签。

## 远程仓库

仓库所有者提供实际地址后再设置 origin。若新建空的 GitHub 仓库，可按以下流程同步，URL 必须替换为自己的地址：

```bash
git remote -v
git remote add origin <你的实际仓库地址>
git push -u origin main
```

这些是使用说明，不代表已经执行远程操作。若远程已有提交，先 fetch 并比较历史，再决定合并方式；不要直接 force push。远程公开性、项目许可证和作者邮箱公开方式由所有者决定。

## 面试官阅读路径

README 用于一分钟理解任务、架构、当前状态与核心结果。docs 保存方案、设计和决策；reports 保存经过测量的结果；examples 保存小规模可公开输入和输出；演示视频在完成后链接。

每份实验报告标注代码 SHA、配置、上游版本、硬件、数据指纹、验证集、原始结果位置和失败案例。大文件使用独立存储，只在仓库记录校验值与可访问位置。

## 文件管理

`.env.example` 仅含变量名和空值；真实 `.env` 忽略。数据、检查点、运行目录、数据库和本地排版检查产物均忽略。提交经过脱敏的小型报告和必要的汇总指标，不能只靠 `.gitignore` 代替提交前检查。

Markdown 是项目文档的权威源。更新 `docs/product-plan.md` 后重新运行文档生成脚本，同一提交中更新 Word 副本，避免两个版本内容漂移。

## 第一阶段完成后的检查

- `git status --short` 是否干净。
- README 的功能状态是否与代码一致。
- 文档本地链接是否可访问。
- 当前提交中是否出现凭证、个人数据或大模型权重。
- 报告中的性能数字是否能追溯到实验记录。
