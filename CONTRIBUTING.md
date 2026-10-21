# 贡献指南（CONTRIBUTING）

感谢参与 **澜图智能企业智能体平台** 的建设。本项目为与「曹浩田」合作完成的项目，欢迎通过 Issue 与 Pull Request 参与改进。

## 一、提交前必须遵守

1. **不得提交任何真实密钥与个人信息**
   - 不提交 `.env`、`.env.local`、API Key、Token、数据库口令、JWT 密钥、个人邮箱、手机号、本机绝对路径。
   - 本地配置从 `.env.example` 复制生成，提交前用 `git diff` 自查。
2. **不得提交本地运行产物**
   - `.venv312/`、`__pycache__/`、`local_runtime/logs/`、`local_runtime/processes.json`、`local_runtime/model_check.json` 已在 `.gitignore` 中忽略。
3. **保持语料与评测可复现**
   - 修改检索、切分、重排、提示词或答案生成逻辑时，必须同步运行相关测试与评测，并在 PR 中附上结果。
4. **不要破坏可移植性**
   - 代码与脚本使用相对路径，不依赖某个人的本机目录。

## 二、开发环境

```powershell
# 1. 复制配置模板并填写本地值（.env.local 不会入库）
Copy-Item .env.example .env.local

# 2. 创建虚拟环境并安装依赖
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt

# 3. 启动（需要本机 MySQL 8 与 Redis 8）
.\.venv312\Scripts\python.exe 启动本地服务.py --no-browser
```

## 三、代码与提交规范

- 分支命名：`feature/<简述>`、`fix/<简述>`、`docs/<简述>`。
- 提交信息：使用祈使句简述“为什么改”，例如 `fix: 企业问题误路由到 OA 制度库`。
- 一次提交只做一件事，避免把格式化、重构与功能改动混在一起。
- Python 代码保持现有风格：类型标注按需使用，不引入无关依赖。

## 四、测试门禁

提交 PR 前至少完成：

```powershell
# 单元测试
.\.venv312\Scripts\python.exe -m unittest discover -s tests -t . -q

# 企业 RAG 固定评测（带门禁）
.\.venv312\Scripts\python.exe -m enterprise_rag.eval.run_eval --gate

# OA 检索评测
.\.venv312\Scripts\python.exe eval\run_eval.py --gate
```

涉及在线问答、推理或提示词改动时，还需在服务运行状态下执行：

```powershell
.\.venv312\Scripts\python.exe scripts/evaluate_reasoning.py --phase regression --gate
.\.venv312\Scripts\python.exe scripts/audit_http.py --phase regression
```

## 五、Pull Request 检查清单

- [ ] 未包含真实密钥、个人信息与本机绝对路径
- [ ] 新增/修改逻辑有对应测试，或说明了无法自动测试的原因
- [ ] 单元测试通过
- [ ] 相关评测通过，并在 PR 描述中粘贴指标
- [ ] README / 技术文档在行为变化时同步更新
- [ ] 语料或评测集变更时，说明了数据来源与预期影响

## 六、问题反馈

提交 Issue 时请说明：

1. 复现步骤与期望结果；
2. 实际结果（可附检索轨迹或引用信息，注意脱敏）；
3. 运行环境（操作系统、Python 版本）；
4. 是否使用了自定义模型或语料。
