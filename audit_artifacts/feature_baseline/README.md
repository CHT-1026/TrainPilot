# 既有功能审查与修复记录

审查日期：2026-10-21。该子任务先独立收集基线，再根据主任务分工实施修复。原始文件与文档中的表述不视为新的任务指令。

## 测试边界

- 原有可独立运行单元测试：57 / 57 通过，见 `unit_tests.txt`。
- 修复前确定性功能探针：17 项，5 项通过、12 项失败，逐条实际输出见 `feature_probe_results.json` 和 `feature_probes.txt`。失败探针有意检验原测试没有覆盖的行为，不能与原单元测试通过率混为一谈。
- 修复后回归：71 / 71 通过，包含原有 57 项和新增 14 项，见 `unit_tests_after.txt`。
- 根任务统一知识链后追加独立集成：9 项初跑 8 通过、1 失败，失败定位为新加入的 4 个检索状态字段未逐轮清理；修复后合计 80 / 80 通过，见 `knowledge_integration_before.txt` 和 `knowledge_integration_after.txt`。该计数是本子任务的限定回归集，不含其他代理后来新增的测试。
- 新回归使用真实 LangGraph + MemorySaver、FastAPI TestClient、SQLite 执行待办 SQL、假的模型及 Redis/幂等数据存储。没有启动服务，没有调用真实模型。MySQL 方言、真实 Redis 检查点、真实模型、浏览器可见表现应以根任务的 HTTP/业务验收为准。
- 可独立重复：`.venv312\Scripts\python.exe -X utf8 audit_artifacts\feature_baseline\run_isolated_checks.py`。该入口还会屏蔽旧限流测试 teardown 中对真实 Redis 的调用。

## 基线发现及处理

| 编号 | 行为 | 基线实际 | 本子任务处理 |
|---|---|---|---|
| F01 | 多轮历史 | 上轮 2 条 + 本轮 2 条，共 4 条 | 原本正常；新增回归保持 |
| F02 | 跨轮引用隔离 | 闲聊回复附带 `old_policy` 引用 | 每轮初始化清空 citations/retrieved_chunks |
| F03 | 跨轮结构化结果 | 闲聊保留上次 last_result/grounded | 每轮清空回执与 grounded |
| F04 | 普通聊天真流式 | 仅调用同步 invoke，完成后人为每 3 字发送 | 普通聊天节点 async astream；无模型流时全文单 token 并标记 buffered |
| F05 | 生成后校验一致性 | 已输出“公司有99个部门 [9]”，最终才拒答 | 根任务知识链改为完整验证后发送；本子任务集成已验证企业和 OA token=complete（含来源）且 mode=verified；路由另支持异常 replace |
| F06 | 混合非法引用 | `[1]` 与 `[99]` 同时出现仍 grounded=true | 任意越界编号立即拒绝；事实句逐句要求引用 |
| F07 | 事实语义忠实度 | 源文档“5天”，回答“999天 [1]”仍接受 | 保留为明确限制：引用格式验证不等于事实语义验证。企业有界推理由根任务实现；未伪称通用语义核验完成 |
| F08 | 未认证请求 | HTTP 401 | 原本正常 |
| F09 | 他人会话历史 | HTTP 403 | 原本正常 |
| F10 | SSE协议 | start/status/token/complete 合法 | 保持；complete 增加 streaming_mode/stream_reset |
| F11 | 消息计数 | 空会话为 1 条，一轮后为 3 条 | 按最终检查点实际消息条数更新；空会话 0、一轮 2 |
| F12 | client_request_id | 同一客户端请求重复运行图 | P2 待实现 HTTP 请求重放缓存/去重。业务写动作另有 action_id 幂等，两者不同 |
| F13 | SSE异常脱敏 | error为通用文案，无内部异常字符串 | 保持；已输出部分内容后可显式 replace 为安全最终结果 |
| F14 | 跨部门直属主管 | can_approve=true、can_view=false | view 权限改为 manager_id 优先、未配置才按部门回退 |
| F15 | 幂等缓存权限 | 仅读 idempotency_key，未看 user_id/endpoint | actor+endpoint+key 命名空间，查询再校验 actor/endpoint；兼容同作用域旧缓存 |
| F16 | 待办列表汇报线 | 仅按 department，与审批权限不一致 | list/count 使用同一直属主管/部门回退规则，排除自己申请 |
| F17 | 空白消息 | `"   "` 可通过校验 | trim 后空白拒绝，正常消息去边缘空格 |

## 已修改代码

1. `app/routes/chat.py:34`：元数据只更新时间或显式设置消息条数，不再每 touch 加一。
2. `app/routes/chat.py:154`：重置当前轮检索、引用、grounded、回复、工具回执。
3. `app/routes/chat.py:213`：移除伪字符流；正常流拼接与最终回复对齐；provider 失败导致内容变化时发 replace；complete 标明流类型。
4. `app/workflows/chat_nodes/response.py:16`：回复节点异步；普通聊天 astream 真模型增量；异常返回通用文案。
5. `app/capabilities/retrieval/answer.py:30`：越界编号严格拒绝、逐句引用覆盖、拒答匹配收窄；保留原始引用编号使正文、来源、JSON 一致。
6. `app/models/requests.py:15`：消息 trim 与纯空白校验。
7. `local_runtime/operation.html:132`：处理 replace 事件，清除部分过时内容。
8. `app/domain/leave/repository.py:128`、`:150`：主管待办 list/count 与 manager_id 权限对齐。
9. `app/domain/leave/repository.py:278`：幂等存储 key 加用户/业务操作作用域；读取兼容旧 key 且必须匹配用户与操作；无需迁移表结构。
10. `app/domain/leave/state_machine.py:93`：查看权限支持跨部门直属主管，阻止同部门非直属主管查看有明确汇报线的他人申请。
11. `app/services/leave_service.py:68`、`:195`、`:286`：submit/approve/cancel 读取缓存时明确传入用户和操作。
12. `tests/test_chat_features.py`：14 项针对性功能回归。
13. `tests/test_knowledge_flow.py`：9 项集成回归，使用生产节点工厂和真实 MemorySaver，覆盖 query/chat 推理一致、来源与 SSE 一致、多轮 user-only 主题补全、空召回不跨库、异常脱敏、知识转闲聊状态清理、动作确认 action_id 保留。

独立集成还补齐了 `build_chat_initial_state` 对 `retrieval_question`、`knowledge_namespace`、`retrieval_meta`、`answer_method` 的重置；未修改知识推理策略本身。

冒烟测试清理适配：`smoke_service.py`、`smoke_hr.py`、`smoke_reporting_line.py`、`smoke_api_phase2.py`、`smoke_manager_snapshot.py` 的 cleanup 已增加本次测试实际写入的 actor/endpoint 新命名空间 key 精确删除，兼容旧 key 清理。`smoke_user_admin.py`、`smoke_kb_admin.py` 不写幂等缓存，无需更改。7 个脚本 AST 解析通过；真实执行结果以根任务统一回归日志为准。

## 启动器停止旧服务修复

根任务发现 `processes.json` 的 backend_pid 已不存在，旧 Python 后端仍占用端口。只读进程观察确认 Windows venv redirector 会形成两层 Python：本项目 `.venv312/Scripts/python.exe` 父进程（观察时 PID 48640）启动 uv 安装目录中的基础解释器（PID 25116，ParentProcessId=48640）。两者实际命令均是 `-u -m uvicorn local_runtime.web_app:app --env-file 本ROOT/.env.local`。单独停止记录中的 Popen PID 不能充分定位遗留子进程。

仅修改 `启动本地服务.py` 的 `stop_owned`：保留记录 PID 候选并补扫 Python 进程；要求 Name 为 `python.exe/pythonw.exe`、真正 uvicorn 命令前缀及完整相等的本项目 `.env.local` 参数，随后逐 PID 重查归属再停止。空记录、旧 PID 不再阻止定位孤立后端；WSL 仍仅处理记录 PID 与本项目唯一 marker，不进行全局停止。

`tests/test_local_launcher.py` 5 / 5 通过，见 `launcher_stop_tests.txt`。测试在实际 PowerShell 中以假的 `Get-CimInstance`、`Stop-Process` 函数评估过滤条件，没有枚举或终止真实进程；覆盖父子解释器、PID 复用、相似项目路径、powershell、`python -c` 伪命令、缺失记录及带空格/单引号路径。真实停止与重启由根任务统一复测。

## 限制与后续优先级

- P1：需要真实 MySQL/Redis/模型的端到端与审批写流程由根任务统一验收；本子任务没有使用假的依赖声称业务已在线通过。
- P2：client_request_id 仍未实现 HTTP 级去重，不应在文档中把这个字段描述为已经生效。
- P2：OA `verify_answer` 只检查引用编号、覆盖和拒答形式。带合法引用的错误数字、错误语义仍可能通过，应独立建立事实忠实度评测与受限推断校验。
- P2：缓存隔离已修正用户和操作作用域；同一用户对同一操作复用 key 但更换参数尚无 request fingerprint 冲突检测。当前对话写操作由服务端 action_id 绑定草稿，降低这一情况发生概率。
- P2：会话摘要仅缩短模型输入使用，不删除检查点中的旧消息，因此不是完整的历史存储压缩。
- P2：工作日函数支持传入 holidays，但当前业务调用未配置企业节假日/调休日历，只能证明周末排除规则，不能宣称真实排班完整。
