# TrainPilot 自主检索接入（2026-11-02）

基础版本：`origin/lantu-platform`，`61bf13d767d7691bade56df7519f2926209a09d9`。
当前目录：`E:\agent项目\TrainPilot-Agent-Tools`，分支 `codex/trainpilot-tools`。
原 `TrainPilot-Agent-RAG` 目录未被覆盖；代码备份在 `_backups/agent-before-tools-20261102`。

## 变化

- 远程最新RAG修复与本地TrainPilot身份、意图协议、别名/追问、知识导航和启动适配已合并。
- `RAG_TOOL_CALLS=true` 时，知识问答不预先固定检索，由V6选择 `search/read/list`。
- 目录导航、意图识别及OA写操作仍走原业务流程；本轮没有增加模型可调用的写工具。
- 工具模式不以抽取回答或其他模型兜底。失败会显示状态，保留原始输出和执行记录。
- 最终引用检查目前验证“文档、定位符、引句确实交付给模型”，不证明整句语义或推理正确。
- 两条初始V6实测输出了正确工具JSON但漏标签；初始失败记录保留。接口另支持严格的裸JSON兼容：
  只接受恰好name/arguments两字段、name属于已声明工具且arguments为对象的输出；不猜测或改参。
  `local_trace.tool_protocol_used=bare_json_compat` 单独标记，不能计为Hermes标准格式通过。

## 文件

- `app/capabilities/retrieval/corpus_tools.py`：复用远程归档的公开语料BM25工具及快照生成。
- `app/capabilities/retrieval/index_tools.py`：网页使用现有索引；每次调用与最终引用检查都执行当前用户ACL。
- `app/capabilities/retrieval/tool_chat.py`：模型→工具→模型循环，最多8轮、12次工具调用。
- `app/workflows/chat_nodes/policy_qa.py`：把工具循环接入现有LangGraph知识问答节点。
- `trainpilot/test/test_code/qwen_tools_api.py`：相同V6权重的工具兼容API；原文本请求继续兼容。
- `trainpilot/test/test_code/qwen_tool_protocol.py`：解析Qwen3标准 `<tool_call>`，转换为OpenAI `tool_calls`。

工具规格采用最新归档：`search(query,top_k)`、`read(doc_id,offset,limit)`、`list(keyword,offset)`。
`read` 的offset/limit是片段序号/数量，默认40、上限60；不是旧评测轨迹中的字符窗口。
公开快照工具和生产索引工具协议相同，检索实现不同；对照试验必须记录backend，不能混报成绩。

## 运行

项目根目录执行（PowerShell）：

```powershell
& 'E:\agent项目\lantu-enterprise-agent-platform-main\.venv312\Scripts\python.exe' -m local_runtime.tool_chat
```

此入口使用归档的公开语料，独立于生产数据库；每次独立问答，显示原始生成和工具结果。
仅显式传入 `--output 路径.jsonl` 时保存逐题记录。`--question '问题'` 可运行单题。

网页入口：双击本目录的 `启动服务.cmd`，默认 `http://127.0.0.1:8081`，旧版8080保留。
仍复用原MySQL/Redis业务数据；不会自动重新入库、清库或运行SQL迁移。
切回固定RAG：在本目录 `.env.local` 中设 `RAG_TOOL_CALLS=false` 并重启本目录服务。

## 下一步实验

先冻结工具协议，用未训练的V6测调用格式、参数合法性、查找/补证和最终答案，分开计分。
随后预览包含正确调用、空结果改写、读取后补证、直接回答和缺证拒答的多轮SFT样例，审阅后再扩展。
尚未训练新适配器，也没有修改旧1239题评测产物。

## 接入检查结果

真实V6在公开快照上分别成功发起并执行了search、read、list，均使用裸JSON兼容。
搜索上级关系后没有继续补查上级职务，最终回答缺citations；指定文档读取后也漏citations；
目录题复制长结果且生成截断。三条仅为工程小样本，不是正式能力通过率。
初始协议失败、模板兼容错误及修复后的原始记录分别保存在：
`trainpilot/runs/tool_calls_v6_20261102/`，未删除失败记录。

Qwen模板依据：https://qwen.readthedocs.io/en/latest/framework/function_call.html
