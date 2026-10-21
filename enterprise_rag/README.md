# 企业知识检索与问答（RAG）模块

> 本模块已在 2026-10-21 完成整体实测与修复；当前默认生成配置、有限推理、查询分解、权限/版本、向量一致性及评测边界详见 [整体报告](../项目整体评估与修复报告_20261021.md)。旧文档中的“默认抽取式”和“准确率100%”属于修复前实现与关键词评测口径；不表示通用推理或语义准确率。

## 1. 模块定位

本模块是 **澜图智能企业智能体平台** 的知识检索与问答子系统，为虚构企业 **澜图智能装备股份有限公司（LANTU，工业机器人 / 智能制造解决方案）** 提供可运行、可评测、可追溯的企业知识库问答能力。

- 语料命名空间：`lantu`，与原有 OA 制度库 `default` 物理隔离，互不污染。
- 全部代码位于 `enterprise_rag/`，复用 OA 的 MySQL/Redis 连接、BM25、权限判断与 FastAPI 应用。
- 默认使用离线哈希向量与抽取式答案，保证无外网、无大模型也能完整跑通；可通过环境变量切换通义千问向量或 LLM 生成。

## 2. 目录结构

```
./
├── enterprise_rag/
│   ├── __init__.py
│   ├── config.py                 # 命名空间、检索/向量/答案参数
│   ├── corpus_gen.py             # 语料落盘 CLI（md/docx/pdf/csv/xlsx）
│   ├── corpus_specs.py           # 公司/组织/产品/销售语料定义
│   ├── corpus_specs2.py          # 制度/财务/会议/法务/HR/技术/售后/战略语料定义
│   ├── corpus/
│   │   ├── manifest.jsonl        # 134 篇文档的完整元数据
│   │   └── docs/<category>/*     # 已生成的 5 种格式文档
│   ├── loaders.py                # 多格式解析：md/txt/docx/pdf/csv/xlsx
│   ├── chunking.py               # 结构感知切分 + 表头重复
│   ├── embedding.py              # 离线哈希向量 / DashScope 向量
│   ├── store.py                  # 命名空间索引、向量矩阵、版本族消解
│   ├── ingest.py                 # 入库 CLI：解析→切分→向量→写库→刷新版本
│   ├── query.py                  # 查询改写、同义词、时间范围抽取
│   ├── retrieval.py              # BM25+向量 RRF→权限/版本过滤→特征重排→去重
│   ├── citations.py              # 上下文、引用、来源格式化
│   ├── answer.py                 # 抽取式/LLM 答案、意图槽位、引用校验、拒答
│   ├── pipeline.py               # 端到端编排
│   ├── api.py                    # /api/rag/* 接口
│   ├── eval/
│   │   ├── testset.jsonl         # 74 条复杂评测集
│   │   ├── run_eval.py           # 指标计算与报告生成
│   │   └── reports/
│   │       ├── latest.json       # 机器可读指标
│   │       ├── per_case.jsonl    # 逐条结果
│   │       └── report.md         # 人类可读报告
│   └── README.md                 # 本文档
├── tests/
│   ├── test_enterprise_rag.py    # 纯函数单元测试
│   └── smoke_rag.py              # HTTP 冒烟（鉴权/查询/冲突/拒答）
└── sql/
    ├── migrations/004_enterprise_rag.sql
    └── apply_migrations.py       # 幂等迁移入口
```

## 3. 语料与元数据

### 3.1 公司设定

| 项目 | 内容 |
| --- | --- |
| 公司 | 澜图智能装备股份有限公司（LANTU） |
| 行业 | 工业机器人、智能产线与工业软件 |
| 成立 | 2011-06-18，苏州工业园区 |
| 规模 | 员工 3,860 人；2025 年营收 42.63 亿元；机器人年出货 21,400 台 |
| 基地 | 苏州、东莞、成都三大基地，总产能 30,000 台/年 |

### 3.2 语料规模

- 文档：**134 篇**（要求 ≥100）
- 文本块：**603 个**
- 12 类：`company` 8、`org` 9、`product` 15、`sales` 18、`policy` 16、`finance` 11、`meeting` 10、`legal` 8、`hr` 10、`tech` 10、`support` 12、`strategy` 7
- 5 种格式：`md` 112、`docx` 10、`pdf` 5、`xlsx` 4、`csv` 3
- 6 组同族多版本（共 13 篇）：差旅标准 v1/v2/v3、考勤 2024/2025、采购 2024/2025、销售折扣 2024/2025、组织架构 2024/2025、服务 SLA 2024/2025

### 3.3 元数据字段

`manifest.jsonl` 与 `kb_documents` 同时保存：

| 字段 | 说明 |
| --- | --- |
| `doc_id` / `title` | 稳定 ID 与标题 |
| `namespace` | 固定 `lantu` |
| `category` / `department` / `author` | 类型、所属部门、作者 |
| `doc_family` / `version` | 文档族与版本号 |
| `created_at` / `updated_at` / `effective_date` | 创建、更新、生效时间 |
| `format` / `visibility` / `status` | 格式、可见性、状态 |
| `path` / `checksum` / `size_bytes` | 源文件、校验和、大小 |

Markdown/DOCX/PDF 正文首部会渲染元数据块；入库时该块被 loader 过滤，避免“更新时间/部门”等词污染检索，元数据以数据库字段为准。

## 4. 端到端链路与模块衔接

```
文档 → loaders 解析 → chunking 切分 → embedding 向量化 → ingest 写库
                                                           ↓
用户问题 → query 改写/时间抽取 → retrieval 混合召回 → 版本消解 → 重排 → 去重
                                                           ↓
                                   citations 上下文/编号 → answer 生成 → 引用校验/拒答
                                                           ↓
                                                      pipeline → API / 评测
```

| 环节 | 模块 | 输入 → 输出 | 关键设计 |
| --- | --- | --- | --- |
| 解析 | `loaders.py` | 文件 → sections/blocks | 表格统一为管道文本；PDF/DOCX 保留章节 |
| 切分 | `chunking.py` | blocks → chunks | 480 字窗口/70 字重叠；表格每 10 行重复表头 |
| 向量 | `embedding.py` | 文本 → 384 维向量 | 字符 n-gram 哈希，L2 归一化；可切 DashScope |
| 入库 | `ingest.py` | manifest → MySQL | 每篇事务写入；批量向量；刷新 Redis 索引版本 |
| 索引 | `store.py` | MySQL → BM25 + 向量矩阵 | 按命名空间缓存；同族版本映射 |
| 改写 | `query.py` | 原问题 → 扩展问题 + as_of | 同义词/缩写扩展；年月/季度/半年解析 |
| 召回 | `retrieval.py` | 问题 → Top-K 块 | BM25 Top60 + 向量 Top60，RRF 融合 |
| 过滤 | `retrieval.py` | 候选 → 合规候选 | 复用 OA 可见性/部门 ACL；按 as_of 选版本 |
| 重排 | `retrieval.py` | 候选 → 排序 | IDF 覆盖、向量相似、标题命中、实体/数字/日期/版本加成 |
| 引用 | `citations.py` | 块 → 上下文/引用 | 标题、版本、部门、作者、更新时间、片段 |
| 答案 | `answer.py` | 上下文 → 答案 | 抽取式逐句引用；意图槽位保证关键事实；无依据拒答 |
| 编排 | `pipeline.py` | API 请求 → 完整响应 | 返回改写、检索轨迹、引用、延迟 |

## 5. 依赖与配置

### 5.1 新增依赖

在 `requirements.txt` 中已追加：

```
numpy>=2.0.0
openpyxl>=3.1.0
reportlab>=4.0.0
jieba>=0.42.1
```

### 5.2 环境变量（`.env.local`）

```ini
RAG_NAMESPACE=lantu
RAG_TOP_K=8
RAG_CANDIDATE_K=60
RAG_RRF_K=60
RAG_MAX_CHUNKS_PER_DOC=4
RAG_MIN_SCORE=0.18
RAG_EMBEDDING_PROVIDER=hash      # hash | dashscope
RAG_EMBEDDING_DIM=384
RAG_EMBEDDING_BATCH=16
RAG_USE_LLM=auto                # auto: 有密钥则生成；false: 离线抽取/确定性推理
RAG_LLM_MODEL=qwen3.8-flash
```

`OPENAI_API_KEY` / `OPENAI_API_BASE` 与 OA 共用；`RAG_EMBEDDING_PROVIDER=dashscope` 时使用 `text-embedding-v3`。

## 6. 运行步骤

所有命令在本目录下执行，Python 使用 `.venv312\Scripts\python.exe`。

```powershell
# 1) 数据库迁移（幂等，新增 7 个文档字段、3 个块字段与索引）
.\.venv312\Scripts\python.exe sql\apply_migrations.py

# 2) 生成 134 篇语料（md/docx/pdf/csv/xlsx + manifest.jsonl）
.\.venv312\Scripts\python.exe -m enterprise_rag.corpus_gen

# 3) 入库（全量重建 lantu 命名空间）
.\.venv312\Scripts\python.exe -m enterprise_rag.ingest

# 抽样验收（仅入库前 10 篇，不清理已有数据）
.\.venv312\Scripts\python.exe -m enterprise_rag.ingest --limit 10

# 4) 评测（74 条，含门槛）
.\.venv312\Scripts\python.exe -m enterprise_rag.eval.run_eval --gate

# 5) 启动服务（会自动加载 main.py 中的 /api/rag 路由）
.\启动服务.cmd
```

### 6.1 API 示例

```powershell
# 登录
$token = (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/auth/login `
  -ContentType application/json -Body '{"username":"admin","password":"admin123"}').data.token

# 查询
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/rag/query `
  -Headers @{Authorization="Bearer $token"} -ContentType application/json `
  -Body '{"question":"2025年7月之后一线城市差旅住宿标准是多少？","top_k":8}'

# 统计 / 文档目录 / 重建索引
Invoke-RestMethod -Uri http://127.0.0.1:8080/api/rag/stats -Headers @{Authorization="Bearer $token"}
Invoke-RestMethod -Uri http://127.0.0.1:8080/api/rag/documents -Headers @{Authorization="Bearer $token"}
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/rag/reindex -Headers @{Authorization="Bearer $token"}
```

响应核心字段：`answer`、`citations`、`grounded`、`refused`、`query_rewrite`、`retrieval.chunks`（含 `score`、`bm25_rank`、`vector_rank`、`features`）、`latency_ms`。

## 7. 评测设计

### 7.1 用例构成（74 条）

| 类别 | 数量 | 覆盖点 |
| --- | --- | --- |
| `single_doc` | 16 | 单文档事实、参数、制度条款、公司名称/简介/总部/使命 |
| `multi_doc` | 10 | 跨文档联合（方案+案例、财报+研发） |
| `cross_department` | 8 | 财务/HR/法务/供应链/质量综合 |
| `conflict` | 8 | 同族多版本，默认取最新 |
| `time_range` | 8 | 2024 版、2025 年 1 月版、Q3、上半年 |
| `fuzzy` | 8 | 同义词、口语化、缩写 |
| `cross_format` | 8 | CSV/XLSX/PDF/DOCX/长文表格 |
| `refusal` | 8 | 无依据问题必须拒答 |

### 7.2 指标定义

- **Recall@K**：Top-K 块所属文档中是否出现任一期望文档。
- **全量召回@10**：多文档问题的所有期望文档是否都进入 Top10。
- **MRR**：首个期望文档排名的倒数。
- **nDCG@10**：二值相关性下的归一化折损累计增益。
- **答案准确率**：答案包含全部标准关键词（数字去空格/逗号后匹配）。
- **引用覆盖率**：引用文档命中期望文档的比例。
- **拒答准确率**：拒答类问题正确拒答比例。
- **误拒率**：可回答问题的拒答比例。
- **冲突准确率**：`conflict` 类别答案准确率。

### 7.3 实测结果（2026-10-21）

| 指标 | 数值 |
| --- | --- |
| Recall@5 | **96.97%** |
| Recall@10 | **98.48%** |
| 全部期望文档命中 Top10 | **93.94%** |
| MRR | **0.8477** |
| nDCG@10 | **0.8651** |
| 答案准确率 | **100.00%** |
| 引用覆盖率 | **93.94%** |
| 有依据回答率 | **100.00%** |
| 拒答准确率 | **100.00%** |
| 非拒答误拒率 | **0.00%** |
| 平均延迟 | **12.6 ms** |

分类结果中 `conflict`、`time_range`、`single_doc`、`multi_doc`、`cross_department` 答案准确率均为 100%；`fuzzy` 与 `cross_format` 的 Recall@5 为 87.5%，其余用例可由 Top10 或相邻块补足。逐条结果见 `eval/reports/per_case.jsonl`，完整报告见 `eval/reports/report.md`。

### 7.4 结果分析

- **同族版本消解有效**：默认取 `doc_updated_at`/版本号最新，如差旅 v3 返回 600 元；问“2024 年”时按 `effective_date` 回退到 v1 返回 450 元。
- **混合检索提升召回**：BM25 解决型号、金额、专有名词；哈希向量补足近义表达；RRF 后重排将“文档标题+章节+数字+实体”共同纳入。
- **表格类效果稳定**：CSV/XLSX 以“表头重复 + 每 10 行一块”切分，12 月营收、Q3 订单、备件价格、基地人数均可定位。
- **拒答可控**：以最高块分数、句级覆盖率和“名词性未登录实体”三重门禁，8 条无依据问题全部拒答，且无误拒。
- **局限**：离线哈希向量不具备真正的语义泛化，跨表述问题依赖同义词表；抽取式答案可读性弱于 LLM；当前评测是单轮、单语料，未覆盖多轮指代与超长文档。

## 8. 与平台其他模块的关系

- 命名空间隔离：OA 制度库仍为 `default`，新语料为 `lantu`；`kb_documents.namespace` 与索引缓存均按命名空间分区。
- 复用能力：MySQL/Redis 连接池、BM25 分词、`UserContext` 权限模型、`chunk_allowed` 判断、FastAPI 异常处理。
- 兼容性：`app/capabilities/retrieval/index.py` 与 `search.py` 的默认参数保持 `default`，原有 OA 问答、请假流程、知识库管理均不受影响。
- 新增接口：`/api/rag/query`、`/api/rag/stats`、`/api/rag/documents`、`/api/rag/reindex`、`/api/rag/eval/latest`。
- 对话联动：`app/workflows/chat_nodes/policy_qa.py` 对企业知识问题优先调用本模块检索，未命中时回退 OA 制度库；`enterprise_scope.py` 负责问题范围判定。

## 9. 扩展建议

1. 将 `RAG_EMBEDDING_PROVIDER` 切换为 `dashscope`，重建索引即可获得语义向量；评测集可对比两种向量的指标。
2. 将 `RAG_USE_LLM=true`，用通义千问基于同一上下文生成答案，抽取式作为降级路径。
3. 引入跨编码器重排（bge-reranker）替换当前特征重排；接口保持不变，仅替换 `retrieval._rerank`。
4. 增加多轮会话状态与指代消解；在 `pipeline.query` 前合并会话上下文。
5. 将 `manifest.jsonl` 作为语料版本基线，接入 CI 做“语料变更 → 重建 → 评测门槛”的流水线。
