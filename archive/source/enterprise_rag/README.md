# 企业知识检索与问答（RAG）模块

> 语雀46项合并方案中有36项被旧实施清单标为已实现，其中包含部分接线、可选实验和契约原型，不能据此宣称36项已完整验收。2026-10-23实测配置为 `qwen-plus` 与 DashScope `text-embedding-v3`（1024维）；实际模型以运行状态和本次响应为准。详见 [当前逐项核对](../research/project_assessment_20261023/yuque_current_crosswalk.md) 与 [完整改进方案](../research/project_assessment_20261023/完整改进方案与实施顺序_20261023.md)。

2026-10-24追加切换：聊天和企业问答使用北邮网关 `deepseek-v4-flash`，模型名经网关 `/models` 确认；向量继续使用独立配置的DashScope接口，保留现有1024维索引。切换结果见 [最新报告](../research/project_assessment_20261023/项目全面检测与修复报告_20261023.md)。

## 1. 模块定位

本模块是 **澜图智能企业智能体平台** 的知识检索与问答子系统，为虚构企业 **澜图智能装备股份有限公司（LANTU，工业机器人 / 智能制造解决方案）** 提供可运行、可评测、可追溯的企业知识库问答能力。

- 语料命名空间：`lantu`，与 OA 制度库 `default` 共用 MySQL 表，以 namespace 字段和检索 scope 逻辑隔离；它不构成独立物理数据库。
- 全部代码位于 `enterprise_rag/`，复用 OA 的 MySQL/Redis 连接、BM25、权限判断与 FastAPI 应用。
- 生成和向量调用分别受配置与预算控制。显式 `offline=true` 使用离线检索/确定性或抽取式回答；`use_llm=false` 仅禁生成模型，仍可能调用语义向量。不得给已有语义索引临时换成hash向量；hash实验须使用独立、同配置重建的索引。

## 2. 目录结构

```
enterprise_rag/
├── config.py                 # 命名空间、检索/向量/答案/预算/开关参数
├── scope.py                  # EvidenceScope：权限、时点、namespace、发布快照、ACL 指纹
├── release.py                # 版本化发布、校验、ACL 失效与回滚入口
├── budget.py                 # 请求级预算、阶段耗时、模型/向量调用与 token 用量
├── scoring.py                # BM25/向量/特征分数方向、量纲与阈值语义
├── prompts.py                # 版本化提示词、反例约束
├── facts.py                  # 有出处的类型化事实抽取
├── verification.py           # 句级事实核验（数字/主体/否定/年份）
├── model_router.py           # 按任务与复杂度选择模型和 token 预算
├── rerank.py                 # 可选词法/特征配对排序，非神经 cross-encoder
├── skills.py                 # 版本化按需业务能力包（请假/审批/报销/采购）
├── feedback.py               # 反馈采集、汇总与待审核队列
├── corpus_gen.py             # 语料落盘 CLI（md/docx/pdf/csv/xlsx）
├── corpus_specs.py           # 公司/组织/产品/销售语料定义
├── corpus_specs2.py          # 制度/财务/会议/法务/HR/技术/售后/战略语料定义
├── corpus/
│   ├── manifest.jsonl        # 136 篇文档的完整元数据
│   └── docs/<category>/*     # 5 种格式文档
├── loaders.py                # 多格式解析：md/txt/docx/pdf/csv/xlsx（PDF 逐页）
├── chunking.py               # 结构感知切分、父子/邻窗、页码/锚点
├── embedding.py              # 离线哈希向量 / DashScope 向量 + 查询向量 LRU/single-flight
├── store.py                  # 索引、向量矩阵、多表示召回、版本族消解、ANN 建议
├── ingest.py                 # 解析→切分→向量→写库→刷新版本→发布 ACL 失效
├── query.py                  # 查询改写、同义词、时间抽取、有限查询变体、HyDE
├── retrieval.py              # BM25+向量 RRF→EvidenceScope 过滤→重排→图扩展
├── citations.py              # 分层上下文、可逆压缩、页码/锚点/角色引用
├── answer.py                 # LLM/抽取式答案、引用校验、事实核验、技能包、模型路由
├── pipeline.py               # 端到端编排、一次有界纠错检索、观测与预算
├── graph/                    # JSON 图谱构建、检索、质量评估、版本与回滚
├── eval/
│   ├── testset.jsonl         # 74 条固定用例
│   ├── reasoning_cases.jsonl # 40 条推理用例
│   ├── unseen_spec.json      # 冻结未见集规格（不含题目正文）
│   └── run_eval.py           # 指标计算与分层报告
└── README.md                 # 本文档
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

- 文档：**136 篇**
- 文本块：**616 个**（索引实测）
- 12 类：`company` 8、`org` 10、`product` 15、`sales` 18、`policy` 16、`finance` 11、`meeting` 10、`legal` 8、`hr` 10、`tech` 11、`support` 12、`strategy` 7
- 5 种格式：`md` 114、`docx` 10、`pdf` 5、`xlsx` 4、`csv` 3
- 7 组同族多版本（共 15 篇）：差旅标准 v1/v2/v3、考勤 2024/2025、采购付款 2024/2025、销售折扣 2024/2025、组织架构 2024/2025、服务 SLA 2024/2025、控制架构 v2.4/v3.3

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

`kb_chunks` 额外保存 `metadata_json`（页码、锚点、父子块、抽取器版本、OCR 标记）与 `representation`（问题/标题/表格摘要多表示文本）。Markdown/DOCX/PDF 正文首部的元数据块在入库时被 loader 过滤，避免“更新时间/部门”等词污染检索。

## 4. 端到端链路与模块衔接

```
文档 → loaders 解析 → chunking 切分 → embedding 向量化 → ingest 写库 → release 发布
                                                                      ↓
用户问题 → scope 证据边界 → query 改写/时间抽取 → retrieval 混合召回 → 版本消解 → 重排/扩展
                                                                      ↓
                                   citations 分层上下文/压缩 → answer 生成 → 引用校验/事实核验
                                                                      ↓
                                                      pipeline → API / 评测 / 反馈
```

| 环节 | 模块 | 输入 → 输出 | 关键设计 |
| --- | --- | --- | --- |
| 证据边界 | `scope.py` | 请求 + UserContext → `EvidenceScope` | 权限只来自服务端；统一 user/namespace/as_of/snapshot/acl_version |
| 解析 | `loaders.py` | 文件 → sections/blocks | 表格统一为管道文本；PDF 逐页保留页码，支持 OCR 标记 |
| 切分 | `chunking.py` | blocks → chunks | 480 字窗口/70 字重叠；父子/前后邻块；表格每 10 行重复表头 |
| 向量 | `embedding.py` | 文本 → 1024 维向量 | DashScope `text-embedding-v3`；查询向量 LRU+TTL+single-flight |
| 入库 | `ingest.py` | manifest → MySQL | 每篇事务写入；批量向量；写入块元数据与多表示；使发布 ACL 失效 |
| 索引 | `store.py` | MySQL → BM25 + 向量矩阵 | 按命名空间缓存；同族版本映射；ANN 规模建议 |
| 改写 | `query.py` | 原问题 → 扩展问题 + as_of | 同义词/缩写扩展；年月/季度/半年解析；有限查询变体与 HyDE |
| 召回 | `retrieval.py` | 问题 → Top-K 块 | BM25 + 向量 RRF；多表示召回；预算内图扩展与邻居扩展 |
| 过滤 | `retrieval.py` | 候选 → 合法候选 | `EvidenceScope` 统一过滤；按 as_of 选版本；图路径只从合法块重建 |
| 重排 | `retrieval.py` | 候选 → 排序 | IDF覆盖、向量相似、标题/实体/日期加成；可选词法配对实验，未调用神经cross-encoder |
| 引用 | `citations.py` | 块 → 上下文/引用 | 分层上下文；可逆句级压缩；页码、锚点、证据角色 |
| 答案 | `answer.py` | 上下文 → 答案 | 版本化提示词、模型路由、技能包、引用校验、独立事实核验、无依据拒答 |
| 编排 | `pipeline.py` | API 请求 → 完整响应 | 一次有界纠错检索、阶段观测、请求预算、停止原因 |

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
RAG_EMBEDDING_PROVIDER=dashscope    # dashscope | hash
RAG_EMBEDDING_MODEL=text-embedding-v3
RAG_EMBEDDING_DIM=1024
RAG_EMBEDDING_BATCH=10
RAG_USE_LLM=auto                    # auto: 有密钥则生成；false: 离线抽取
RAG_LLM_MODEL=deepseek-v4-flash
RAG_REQUEST_TIMEOUT=300
RAG_MAX_QUERY_VARIANTS=8
RAG_MAX_EMBEDDING_CALLS=8
RAG_MAX_MODEL_CALLS=4
RAG_QUERY_CACHE_SIZE=256
RAG_QUERY_CACHE_TTL=900
RAG_MULTI_REPRESENTATION=false
RAG_NEIGHBOR_EXPANSION=false
RAG_CONTEXT_COMPRESSION=false
RAG_CROSS_ENCODER=false
RAG_HYDE_ENABLED=false
RAG_MAX_HOPS=2
RAG_ANN_CHUNK_THRESHOLD=20000
RAG_MAX_CONCURRENT=8
```

`OPENAI_API_KEY` / `OPENAI_API_BASE` 与 OA 共用。查询向量缓存 key 包含 provider/model/dim/version/text；`offline=true` 时禁止模型与外部向量调用。

## 6. 运行步骤

所有命令在本目录下执行，Python 使用 `.venv312\Scripts\python.exe`。

```powershell
# 1) 数据库迁移（幂等）
.\.venv312\Scripts\python.exe sql\apply_migrations.py

# 2) 生成 136 篇语料（md/docx/pdf/csv/xlsx + manifest.jsonl）
.\.venv312\Scripts\python.exe -m enterprise_rag.corpus_gen

# 3) 入库（全量重建 lantu 命名空间）
.\.venv312\Scripts\python.exe -m enterprise_rag.ingest

# 4) 回填块级页码/锚点/父子关系/多表示（不重新调用向量服务）
.\.venv312\Scripts\python.exe scripts\backfill_chunk_metadata.py

# 5) 构建图谱
.\.venv312\Scripts\python.exe -m enterprise_rag.graph.build --force

# 6) 发布并校验快照（索引 + 图谱 + ACL）
.\.venv312\Scripts\python.exe scripts\publish_release.py

# 7) 固定题集评测（74 条，含门槛）
.\.venv312\Scripts\python.exe -m enterprise_rag.eval.run_eval --gate --llm

# 8) 冻结未见集（运行时生成题目，规格不含题目正文）
.\.venv312\Scripts\python.exe scripts\run_frozen_eval.py --dry-run
.\.venv312\Scripts\python.exe scripts\run_frozen_eval.py --label frozen --mode hybrid

# 9) 启动服务（会自动加载 main.py 中的 /api/rag 路由）
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
  -Body '{"question":"2025年7月之后一线城市差旅住宿标准是多少？","top_k":8,"use_llm":true,"retrieval_mode":"hybrid"}'

# 统计 / 文档目录 / 重建索引
Invoke-RestMethod -Uri http://127.0.0.1:8080/api/rag/stats -Headers @{Authorization="Bearer $token"}
Invoke-RestMethod -Uri http://127.0.0.1:8080/api/rag/documents -Headers @{Authorization="Bearer $token"}
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/rag/reindex -Headers @{Authorization="Bearer $token"}
```

响应核心字段：`answer`、`citations`（含 `page`/`anchor`/`role`/`compressed_spans`）、`grounded`、`refused`、`query_rewrite`、`retrieval.chunks`（含 `score`、`bm25_rank`、`vector_rank`、`features`）、`evidence_scope`、`fact_check`、`observability`、`score_semantics`、`skill_packs`、`latency_ms`。

## 7. 评测设计

### 7.1 用例构成

| 集合 | 数量 | 覆盖点 |
| --- | --- | --- |
| 固定 `testset.jsonl` | 74 | 单文档、跨文档、跨部门、版本冲突、时间范围、模糊、跨格式、拒答 |
| 推理 `reasoning_cases.jsonl` | 40 | 版本对比、多跳汇报关系、计算、反例 |
| 动态文档验收 | 62 | 运行时从当前 136 篇文档归纳事实并生成问题 |
| 冻结未见集 | 83 | 排除固定集已用文档后运行时生成；版本 7、多跳 10、反例 5、数值 1 等 |

冻结规格保存在 `eval/unseen_spec.json`，只记录 seed、样本量与排除文档 ID；题目正文在运行时由当前 manifest/索引/图谱生成，不进入规格文件。

### 7.2 指标分层

- **词法层**：关键词命中率（兼容旧 `answer_accuracy`），只作代理指标。
- **检索层**：任一期望文档命中@5/@10、全部期望文档召回@10、MRR、nDCG@10。
- **引用结构层**：引用编号合法性、至少/全部期望来源被引用。
- **事实支持层**：句级数字/主体/否定/年份核验报告；不是语义蕴含。
- **人工核验层**：否定与矛盾、跨文档多跳完整性、引用是否支持全部前提。

### 7.3 本轮结果

本轮实际数值（模型随额度切换，报告逐份记录）：

| 测试 | 规模 | 结果 |
| --- | --- | --- |
| 固定集 LLM（qwen3.6-plus） | 74 | 关键词 100%、至少一期望来源引用 100%、拒答 100%、gate 通过 |
| 推理回归 + 留出 | 20 + 13 | 33/33 |
| 114 条规模（qwen3.6-plus） | 114 | 114/114（100%） |
| 动态文档（qwen3.8-flash） | 62 | 62/62（100%） |
| 冻结文档子集（qwen3.6-plus） | 73 | 71/73 |
| 冻结生成子集（deepseek-v4-flash） | 22 | 17/22（多跳 10/10、反例 5/5、版本 2/7） |
| DeepSeek 网关抽样 | 37 | 37/37 |

详细记录：

- `enterprise_rag/eval/reports/latest.json` / `report.md` / `per_case.jsonl`
- `audit_artifacts/reasoning_suite/`（推理与 114 条规模）
- `audit_artifacts/frozen_eval/`（冻结未见集，运行后生成题目与逐条结果）
- `audit_artifacts/dynamic_acceptance/`（动态文档验收）

结果解读以报告中的配置快照、执行支路与逐条答案为准；`fact_support_rate` 是严格机械核验，生成式同义改写的误报需人工复核，不等同于语义忠实度。制度/组织版本的宽泛对比仍可能因只召回单版而拒答，属于下一轮跨版本召回改进项。

### 7.4 结果分析要点

- **同族版本消解**：默认取最新版，问“2024 年”时按 `effective_date` 回退。
- **混合检索**：BM25 解决型号、金额、专有名词；语义向量补足跨表述；RRF 后重排综合标题、章节、数字与实体。
- **图路径**：只从本次合法块重建实体/关系/路径，`graph` 模式不回退向量。
- **拒答**：最高块分数、句级覆盖率与未登录实体三重门禁。
- **局限**：事实核验只覆盖可机械核对的数字、主体、否定与年份；离线 hash 向量不具备语义泛化；单轮评测不覆盖多轮指代。

## 8. 与平台其他模块的关系

- 命名空间隔离：OA 制度库仍为 `default`，企业语料为 `lantu`；`kb_documents.namespace` 与索引缓存均按命名空间分区。
- 复用能力：MySQL/Redis 连接池、BM25 分词、`UserContext` 权限模型、`chunk_allowed` 判断、FastAPI 异常处理。
- 兼容性：`app/capabilities/retrieval/index.py` 与 `search.py` 的默认参数保持 `default`，原有 OA 问答、请假流程、知识库管理均不受影响。
- 新增接口：`/api/rag/query`、`/api/rag/stats`、`/api/rag/documents`、`/api/rag/reindex`、`/api/rag/eval/latest`、`/api/rag/graph/stats`、`/api/rag/graph/quality`、`/api/rag/feedback`。
- 对话联动：`app/workflows/chat_nodes/policy_qa.py` 对企业知识问题优先调用本模块检索，未命中时回退 OA 制度库；`enterprise_scope.py` 负责问题范围判定。

## 9. 扩展建议

1. 规模超过 2 万块后再评估带过滤能力的 ANN 或向量数据库；当前 616 块使用精确扫描。
2. 在标注样本上验证 `RAG_CROSS_ENCODER=true` 的收益后再考虑替换默认精排。
3. 用冻结未见集与控制变量评测比较 embedding/HyDE/多表示开关，不以单次演示推断提升。
4. 将 `manifest.jsonl` 与发布快照接入 CI，执行“语料变更 → 重建 → 评测门槛 → 发布校验”。
5. 反馈只进入待审核队列；经审计后再决定是否用于提示词或检索参数迭代，不自动训练。
