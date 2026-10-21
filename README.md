# 澜图智能企业智能体平台

面向虚构企业 **澜图智能装备股份有限公司（LANTU，工业机器人/智能制造）** 的企业智能体平台，集企业知识问答、制度检索、请假与审批于一体。本目录是完整、可独立复制运行的交付包。

> **合作说明**：本项目为与「**曹浩田**」合作完成的项目。欢迎通过 Issue 与 Pull Request 参与共建，参与方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

平台不是单一的 RAG 项目：底层包含 FastAPI 服务、LangGraph 对话智能体、鉴权与审批状态机、BM25 + 向量混合检索、版本冲突消解、引用溯源、评测框架，以及多格式企业语料。

**2026-10-21 已完成整体实测与修复。** 架构核验、根因、逐条原始回答、修复文件、回归结果及后续方案见 [项目整体评估与修复报告](项目整体评估与修复报告_20261021.md)。企业问答现在共用生成引擎：显式事实优先使用有来源的有限推理算子，其他问题在配置模型密钥时默认调用 LLM。`RAG_USE_LLM=false` 可保留离线模式；默认 hash 向量不等于语义 Embedding。

本轮“公司名称/业务问法”的专项根因、代码修改与 62 题动态文档验收见 [推理能力修复与动态验收报告](推理能力修复与动态验收报告_20261021.md)。当前模型：`qwen3.8-flash`。

架构对比修复、多跳关系推理、114 条规模测试与 GraphRAG 优化规划见 [澜图智能推理能力与GraphRAG优化规划](澜图智能推理能力与GraphRAG优化规划_20261021.md)。

完整架构、模块衔接、数据设计、接口、测试方法与评审指南见 [PROJECT_TECHNICAL_DOCUMENT.md](PROJECT_TECHNICAL_DOCUMENT.md)。

## 一、快速开始

1. 双击 **`启动服务.cmd`**。
2. 浏览器自动打开 `http://127.0.0.1:8080/`。
3. 使用演示账号登录，可直接在对话框询问“公司叫什么名字？”。

| 账号 | 密码 | 角色 |
| --- | --- | --- |
| `admin` | `admin123` | 管理员 |
| `employee_zhao` | `admin123` | 员工 |
| `manager_zhang` | `admin123` | 主管 |

启动器会自动完成：

1. 检查/创建本机 Python 3.12 虚拟环境 `.venv312` 并安装依赖；
2. 启动本机 WSL 中的 MySQL 与 Redis；
3. 建表并执行 `sql/apply_migrations.py`；
4. 检查并自动入库企业语料（136 篇）与 OA 制度库（14 篇）；
5. 启动 FastAPI 服务并打开操作台。

首次启动需要几分钟；之后再次启动会复用环境和数据。

## 二、企业知识与 RAG 接口

登录后调用（PowerShell 示例）：

```powershell
$token = (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/auth/login `
  -ContentType application/json -Body '{"username":"admin","password":"admin123"}').data.token

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/rag/query `
  -Headers @{Authorization="Bearer $token"} -ContentType application/json `
  -Body '{"question":"2025年7月之后一线城市差旅住宿标准是多少？","top_k":8}'
```

接口列表：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/rag/query` | 企业知识库问答，返回答案、引用、检索轨迹 |
| GET | `/api/rag/stats` | 语料命名空间、文档数、块数、版本族 |
| GET | `/api/rag/documents` | 文档目录（仅管理员） |
| POST | `/api/rag/reindex` | 刷新进程内索引（仅管理员） |
| GET | `/api/rag/eval/latest` | 最近一次评测结果 |

## 三、复制到其他文件夹

整个目录可整体复制，**不要拆散 `app/`、`enterprise_rag/`、`data/`、`sql/`、`.venv312/`**。

- **同一台电脑复制到另一个文件夹**：直接复制整个目录，双击 `启动服务.cmd` 即可。`.venv312` 已包含依赖。
- **复制到另一台电脑**：删除 `.venv312` 文件夹后复制，双击 `启动服务.cmd`，启动器会自动创建虚拟环境并安装 `requirements.txt`。
- **数据库**：默认连接本机 `127.0.0.1:3306` 的 MySQL 与 `127.0.0.1:6379` 的 Redis，配置在 `.env.local`。如目标电脑没有 WSL 本机数据库，请先安装 MySQL/Redis，或修改 `.env.local` 指向可用实例。
- 所有代码均使用相对路径（基于文件自身位置），不依赖原项目目录。

> `.env.local` 只在本机保存真实密钥与数据库口令，已由 `.gitignore` 排除，**不会进入 Git 仓库或开源包**。开源仓库只保留占位模板 `.env.example`。

## 四、数据与评测

- 语料：136 篇文档、616 个文本块、12 类、5 种格式（md/docx/pdf/csv/xlsx）。
- 版本冲突：7 组同族多版本；默认最新版，问题含时间时按生效日期回退。
- 评测集：74 条固定用例 + 40 条新增推理用例（版本对比、跨版本冲突、多跳汇报关系），合计 114 条；覆盖单文档、多文档、跨部门、冲突、时间范围、模糊、跨格式、拒答、版本对比与多跳关系。
- 原固定题集的关键词命中、文档命中与引用指标不能证明推理或语义正确。本次另建 20 条推理用例与 13 条独立变式，保留修复前后实际回答，详见上方报告及 `audit_artifacts/reasoning/`。
- 114 条规模测试报告见 `audit_artifacts/reasoning_suite/`；`qwen3.8-flash` 与 `qwen3.7-plus` 均为 113/114（99.12%），多跳与版本类用例均 100%。

重新评测：

```powershell
.\.venv312\Scripts\python.exe -m enterprise_rag.eval.run_eval --gate
```

报告位于 `enterprise_rag/eval/reports/report.md`，逐条结果见 `per_case.jsonl`。

推理与在线回归（需服务运行）：

```powershell
.\.venv312\Scripts\python.exe scripts/evaluate_reasoning.py --phase regression --gate
.\.venv312\Scripts\python.exe scripts/evaluate_reasoning.py --phase regression --fixtures audit_artifacts/reasoning/holdout_cases.jsonl --output-name holdout_regression --gate
.\.venv312\Scripts\python.exe scripts/audit_http.py --phase regression

# 114 条规模测试（版本对比 / 多跳关系 / 冲突 / 原有八类）
.\.venv312\Scripts\python.exe scripts/run_reasoning_suite.py --label current --model-hint qwen3.8-flash
```

知识回答通过校验后发送 `streaming_mode=verified` 的完整文本；普通对话使用模型实际 token 流，标记为 `model`。`buffered` 表示模板整段输出，不再拆字模拟模型流式。

## 五、常用命令

```powershell
# 停止后台服务
.\停止服务.cmd

# 重新导入 134 篇语料（全量重建 lantu 命名空间）
.\重新导入语料.cmd

# 运行评测
.\运行评测.cmd

# 手动启动（前台）
.\.venv312\Scripts\python.exe 启动本地服务.py --no-browser
```

## 六、目录说明

```
澜图智能企业智能体平台/
├── 启动服务.cmd / 停止服务.cmd      # 一键启停
├── 重新导入语料.cmd / 运行评测.cmd
├── PROJECT_TECHNICAL_DOCUMENT.md     # 完整技术文档与评审指南
├── main.py                          # FastAPI 入口（对话、审批、/api/rag）
├── app/                             # LangGraph 智能体、鉴权、请假与审批
├── enterprise_rag/                  # 企业知识检索与问答（RAG）模块
│   ├── corpus/                      # 134 篇企业语料 + manifest.jsonl
│   ├── eval/                        # 评测集与报告
│   └── README.md                    # 技术方案与模块衔接说明
├── data/oa_docs/                    # 14 篇 OA 制度语料（随包携带）
├── sql/                             # 初始化脚本与幂等迁移
├── tests/                           # 单元测试与 HTTP 冒烟
├── local_runtime/                   # 本机操作台、启动器、数据库检查
├── .venv312/                        # 本机 Python 3.12 运行环境
├── .env.local                       # 本机配置（数据库、模型密钥、RAG 参数）
└── requirements.txt                 # 依赖清单
```

技术方案、完整模块衔接、指标定义与扩展建议见 [enterprise_rag/README.md](enterprise_rag/README.md)。

## 七、常见问题

| 现象 | 处理 |
| --- | --- |
| 双击启动后浏览器打不开 | 等待 30-60 秒；查看 `local_runtime/logs/backend.log` |
| 提示模型密钥无效 | 修改 `.env.local` 的 `OPENAI_API_KEY` 后重启 |
| 提示 MySQL/Redis 未就绪 | 确认本机 WSL `Ubuntu-24.04` 可用，或修改 `.env.local` 指向现有实例 |
| 复制到新电脑后启动失败 | 删除 `.venv312` 后重新运行 `启动服务.cmd`，让启动器重建环境 |
| 想恢复初始语料 | 运行 `重新导入语料.cmd` |

## 八、开源、安全与贡献

### 8.1 仓库内容边界

- 本仓库只包含源码、虚构企业语料、测试与评测脚本、配置模板。
- **不包含**：`.env` / `.env.local`、真实 API Key、Token、数据库口令、JWT 密钥、个人邮箱/手机号、本机绝对路径、虚拟环境、日志与进程状态文件。
- 首次运行请从模板生成本地配置：

```powershell
Copy-Item .env.example .env.local
# 然后编辑 .env.local，填写模型密钥、数据库口令与随机 SECRET_KEY
```

### 8.2 安全约定

- 演示账号（`admin` / `employee_zhao` / `manager_zhang`，口令 `admin123`）仅用于本地演示，生产部署必须替换并修改 `SECRET_KEY`。
- 语料中的公司名、人物、电话、邮箱、地址均为虚构，仅用于演示，不对应任何真实主体。
- 若发现密钥泄露或安全漏洞，请勿公开提交 Issue，先通过仓库维护者私下反馈。

### 8.3 贡献方式

开发环境、分支与提交规范、测试门禁与 PR 检查清单见 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 8.4 许可证

本项目采用 [MIT License](LICENSE)。
