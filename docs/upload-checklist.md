# 本版本上传清单与 VS Code 操作

本版本交付项目方案、实施路线、Word 副本与 Git 管理基础。应用代码尚未实现，报告中没有实测性能结论。GitHub 已创建，上传由仓库作者在 VS Code 中完成。

## 本版本需要上传的文件

| 文件或目录 | 内容 |
| --- | --- |
| README.md | 面试官阅读入口、项目架构与真实实现状态 |
| CHANGELOG.md | 本版本的修改记录 |
| THIRD_PARTY_NOTICES.md | 开源来源、复用边界与许可证待决事项 |
| .gitignore | 忽略数据、凭证、权重、运行结果和本地缓存 |
| .gitattributes 与 .editorconfig | 文本换行、编码和格式规则 |
| .env.example | 空值配置模板，不含真实密钥 |
| .github/ | Issue 与 Pull Request 模板 |
| docs/product-plan.md | 产品与技术实施方案，文档权威源 |
| docs/TrainPilot项目产品与实施方案.docx | 同源 Word 副本，视觉分页尚待核验 |
| docs/assets/architecture.png | 已检查的三端架构图 |
| docs/roadmap.md | 学习顺序、里程碑与关键时间节点 |
| docs/git-workflow.md | 提交、分支、双人协作和报告规范 |
| docs/document-build.md | 文档生成方式与排版检查状态 |
| docs/upload-checklist.md | 本清单及首次上传步骤 |
| reports/README.md | 后续实验报告要求，当前未执行实验 |
| scripts/build_project_doc.py | 从 Markdown 生成 Word 的脚本 |

最终上传范围以 `git ls-files` 为准。上述文件均提交到 main；已提交文件不会再出现在 VS Code 的未提交修改列表中。源代码管理显示没有修改并不代表目录为空。

## 本版本不上传的内容

本地 .git/ 由 Git 自行管理，不作为普通文件手动上传。忽略 .qa/ 排版中间文件、.env 真实凭证、数据集、模型权重、检查点、运行数据库、原始日志、虚拟环境与编辑器个人配置。当前仓库没有模型训练产物。

## 在 VS Code 中推送

1. 用“文件 → 打开文件夹”打开项目仓库根目录 `trainpilot`，不要打开它的上级目录作为本次 Git 操作目标。
2. 进入“源代码管理”，核对当前分支为 main。已有本地提交，无需重新初始化仓库或重新提交全部文件。
3. 在源代码管理的更多操作中选择“推送 Push”。若提示没有上游分支，按提示发布当前 main 分支到已存在的 origin；也可在 VS Code 终端执行下方命令。
4. 若出现 GitHub 登录提示，由作者完成登录。origin 已指向现有 TrainPilot 仓库，不需再创建另一个 GitHub 仓库。
5. 推送完成后，在 GitHub 网页检查 README、docs 和提交历史，确认页面中的最近提交与本地一致。

```bash
git remote -v
git push -u origin main
```

以上命令在本次整理工作中不会执行。遇到远程已有内容或推送被拒绝时，先检查远程历史，不使用 force push 覆盖。

## 上传前检查

- 核对远程地址为 `https://github.com/CHT-1026/TrainPilot.git`。
- 查看 Git 提交使用的作者身份；如果希望调整公开邮箱，在推送前由作者决定，不在这里重写历史。
- 阅读 README，确认训练、Agent 和评测仍标为待实现。
- 检查 .env.example 保持空值，没有真实账号凭证。
- Word 内容已检查，实际分页与视觉排版待具备渲染环境后核验。
- 项目自身许可证尚未选定，当前不创建未经确认的 LICENSE。

## 官方操作参考

- [VS Code 源代码管理](https://code.visualstudio.com/docs/sourcecontrol/overview)
- [VS Code 推送与同步](https://code.visualstudio.com/docs/sourcecontrol/repos-remotes#push-pull-and-sync)
