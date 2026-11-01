# 澜图智能企业智能体平台 — 归档包

本目录是**澜图智能企业智能体平台**的完整归档，包含三部分：**完整源代码**、**测评题库**、**与题库对应的答案**，以及本轮自主检索能力测评的全部产出。

题目与答案通过 **`qid` 严格一一对应**（1239 ↔ 1239），构建时逐条比对，缺一即中止。

---

## 一、目录结构

```
archive/
├── README.md                          本文件
├── MANIFEST.json                      全部 1651 个文件的 sha256 清单（可校验完整性）
│
├── source/                            ① 完整源代码
│   ├── app/                           平台应用层：路由、工作流、能力、领域模型
│   ├── enterprise_rag/                企业 RAG 引擎：检索、图谱、校验、引用
│   ├── scripts/                       脚本工具（评测执行器、判分器、报告生成）
│   ├── tests/                         测试套件
│   ├── sql/                           数据库迁移脚本
│   ├── eval/                          评测入口
│   ├── local_runtime/                 本地运行支撑（服务安装、数据库准备、流程校验）
│   ├── corpus/                        平台实际使用的知识语料（137 份文档）
│   ├── main.py                        启动入口
│   ├── requirements.txt               依赖
│   └── 启动服务.cmd / 停止服务.cmd 等  Windows 快捷脚本
│
├── questions/                         ② 测评题库
│   ├── question_bank.jsonl            1239 道题（题干 + 题型 + 难度 + 16 维度 + 期望文档）
│   ├── question_bank_dimensioned.jsonl 同上，评测直接消费的文件
│   ├── dimensions.json                16 个能力维度的定义与题量分布
│   └── frozen/
│       ├── core400.json               冻结的共同 400 题题单（两云端模型共用）
│       └── q4_extra400.json           本地模型追加题单
│
├── answers/                           ③ 与题库对应的答案
│   ├── answer_bank.jsonl              1239 份答案（金标准 + 证据出处 + 判分 rubric）
│   └── grading_rules.md               判分规则说明
│
└── evaluation/                        本轮测评产出
    ├── deliverables/                  成绩报告、结论与建议、题单、来源对照表
    ├── runs/                          A/B 两模型各 400 题的完整轨迹（逐题工具调用与回答）
    ├── bank/                          冻结题单、人名白名单等支撑数据
    ├── 判分器校准记录.md               判分器 v1→v6 的校准全过程与七项反向验证
    ├── 执行记录_v1.md                  执行全流程记录
    ├── 清理前清单.md                   归档前的清理依据（删了什么、为什么能删）
    ├── 人工抽审清单.md                 待人工复核的分歧题
    └── 完整检测方案.md                 测评方案原文
```

---

## 二、三部分的关系

```
              questions/question_bank.jsonl
                        │  qid
                        ▼
              answers/answer_bank.jsonl
                        │
                        │  expected_doc_ids
                        ▼
              source/corpus/docs/<doc_id>
```

- **题库**给出问题、题型、难度、所属能力维度，以及**期望在哪几份文档里能找到答案**（`expected_doc_ids`）。
- **答案**给出对应的标准答案、**逐字证据出处**（`evidence[].segments[].locator` 精确到行号）、关键词与判分 rubric。
- **源代码**里的 `source/corpus/` 就是这些 `expected_doc_ids` 指向的原始文档，因此答案可被独立复核。

一个 qid 在两个文件里的 `question` 字段完全相同，`expected_doc_ids` 也完全相同——这是"题目 ↔ 答案"对应关系的机器可校验保证。

---

## 三、如何运行

### 3.1 启动平台

```bash
cd archive/source
pip install -r requirements.txt
python main.py
```

Windows 下可直接双击 `启动服务.cmd` / `停止服务.cmd`。

平台依赖 MySQL 与 Redis；若本地没有，先用 `local_runtime/install_services.sh` 与
`local_runtime/setup_database.py` 准备环境，SQL 迁移见 `sql/migrations/`。

### 3.2 用题库与答案评测自己的模型

评测执行器在 `source/scripts/retrieval_eval.py`：

```bash
cd archive/source
python scripts/retrieval_eval.py snapshot          # 生成隔离的资料快照（排除题库/答案/密钥）
python scripts/retrieval_eval.py run \
    --models A --cases 400 \
    --qids-file ../questions/frozen/core400.json     # 用冻结题单，保证可复现
```

执行器只给模型三样东西：**用户问题、工具说明、工具真实返回的资料**。
标准答案与评分证据清单对模型不可见（`snapshot` 会检查并拒绝把答案放进工作区）。

### 3.3 匹配题目与答案

```python
import json

def load(path):
    return {json.loads(l)["qid"]: json.loads(l)
            for l in open(path, encoding="utf-8") if l.strip()}

questions = load("questions/question_bank.jsonl")
answers   = load("answers/answer_bank.jsonl")

assert set(questions) == set(answers)      # 一一对应保证

q = questions["A0001"]
a = answers["A0001"]
print(q["question"])        # 沈亦舟的直接上级是谁？该上级在公司担任什么职务？
print(a["answer"])          # 沈亦舟的直接上级是顾承澜；顾承澜的职务是董事长兼总经理。
print(a["evidence"])        # 证据出处，含 doc_id 与 line 级定位
```

命令行查表：

```bash
grep '"qid": "A0001"' questions/question_bank.jsonl
grep '"qid": "A0001"' answers/answer_bank.jsonl
```

### 3.4 校验归档完整性

```bash
# MANIFEST.json 记录了每个文件的 sha256，可逐一比对
python -c "
import json,hashlib,pathlib
m=json.load(open('MANIFEST.json',encoding='utf-8'))
bad=[f for f in m['files']
     if hashlib.sha256(pathlib.Path(f['path']).read_bytes()).hexdigest()!=f['sha256']]
print('不一致文件数:',len(bad))
"
```

---

## 四、题库说明

- **规模**：1239 道（主库 1126 + 补建 113）
- **题型**：11 类，覆盖事实检索、单跳/多跳检索、跨文档推理、人物关系推理、时间与因果推断、冲突判断等
- **难度**：L1–L4 四级
- **能力维度**：16 个检索能力维度，题库按维度分层抽样、等配额宏平均
- **溯源**：每题标注 `expected_doc_ids`；每份答案标注 `evidence`，精确到 `doc_id` 与行号定位

16 个维度：

| 编号 | 维度 | 编号 | 维度 |
| --- | --- | --- | --- |
| 01 | 精确定位 | 09 | 历史与生效版本 |
| 02 | 同义与别名 | 10 | 冲突判断 |
| 03 | 模糊需求定位 | 11 | 多条件筛选 |
| 04 | 两跳检索 | 12 | 表格与跨表 |
| 05 | 三跳以上 | 13 | 跨格式取证 |
| 06 | 汇总去重 | 14 | 长文与干扰 |
| 07 | 数值与单位 | 15 | 多轮检索更新 |
| 08 | 时间推断 | 16 | 无答案与证据不足 |

---

## 五、本轮测评结果（供参考）

在 400 道真检索题（16 维度各 25 题）上：

| 槽位 | 模型 | 主指标（正确且引用支持） | 期望文档全部检出 |
| --- | --- | ---: | ---: |
| A | mimo-v2.6-flash | **141/400 = 35.2%** | 67% |
| B | qwen3.6-flash | **127/400 = 31.8%** | 80% |

期望文档检出率远高于主指标通过率，说明**瓶颈在证据运用与答案完整性，不在检索召回**。

判分器经 v1→v6 六轮校准，七项反向验证全部零违规（详见 `evaluation/判分器校准记录.md`）。

> 本地 Q6_K / Q4_K_M 两个槽位**无成绩**（16GB 显存下 27B 量化模型单轮生成 >400 秒，
> 超出 8 轮/600 秒预算），平台 HTTP/流式子集因 MySQL/Redis 不可用未执行。

---

## 六、整理与清理说明

本次归档同时清理了项目内的临时文件与构建产物。**每一项删除都在删除前做过替代性核查**，
依据记录在 `evaluation/清理前清单.md`：

| 删除项 | 规模 | 依据 |
| --- | ---: | --- |
| OCR 渲染页图像（`pdf_extracted/_work/`） | 10.4 GB | 178 个目录与 178 份 OCR 文本产出**一一对应、零缺失**；逐页 OCR 记录（text/language/error）已提取到 `pdf_extracted/_ocr_meta/`；原始 PDF 保留在 `attachments/` |
| `__pycache__` / `*.pyc` / `.pytest_cache` | 约 0.1 GB | 可再生构建产物 |
| 一次性任务推进脚本（checkpoint/sync/install 等 46 个） | 256 KB | 题库核心链路 21 个模块 **import 自检全部通过**，不依赖任何过程脚本 |

**确定保留**（经核查具不可替代性）：原始 PDF、语雀原始抓取、OCR 文本产出、平台语料、
本轮运行轨迹、题库构建源脚本。

---

## 七、许可与出处

- 代码与文档遵循仓库根目录的 `LICENSE`。
- `source/corpus/` 为澜图智能企业知识文档，用于本平台的检索评测。
- 题目由本项目依据上述语料文档生成，答案逐条标注文档出处，未使用外部题库。
