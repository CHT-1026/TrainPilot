# TrainPilot V3 多轮样例预览

状态：用户已认可12条样例；用户已确认从原Full SFT新建LoRA，并混合筛选后的V2训练数据及新增多轮数据。只完成样例与实际状态链路检查，尚未生成正式数据或训练。

已确认沿用 V2 的完整已知字段协议。下面的 JSON 是人工编写的预期输出，不是当前模型的回答。
上下文由脚本化识别结果驱动原 Agent 图生成；初始审批状态为模拟数据，数据库节点明确不可用。

## 本次重点

追问补齐、修改后保留字段、确认与审批区分、取消草稿与撤销已提交申请区分、临时查询后返回原任务。

## 样例

### a1 · leave_collect_modify_confirm · 第 1 轮

用户：我想申请事假，先帮我整理一下。

输入包含历史 0 条；待确认动作：`None`。

预期模型输出：

```json
{"intent":"submit_leave","extracted_info":{"leave_type":"事假"}}
```

预期节点：submit_leave

原 Agent 在预设标签下的实际回复：

```text
请提供以下信息：
- 请假时间（格式：2026-06-23 到 2026-06-28）
- 请假原因
```

执行范围：原Agent本地草稿或状态更新；未调用模型、未访问数据库。

### a2 · leave_collect_modify_confirm · 第 2 轮

用户：日期是2026年11月9日到11月10日，原因填写家中设备维修。

输入包含历史 2 条；待确认动作：`None`。

预期模型输出：

```json
{"intent":"submit_leave","extracted_info":{"leave_type":"事假","start_date":"2026-11-09","end_date":"2026-11-10","reason":"家中设备维修"}}
```

预期节点：submit_leave

原 Agent 在预设标签下的实际回复：

```text
请确认以下请假申请：
- 类型：事假
- 时间：2026-11-09 至 2026-11-10
- 工作日：2 天
- 原因：家中设备维修
- 审批路径：主管审批

回复「确认」提交，回复「取消」放弃。
```

执行范围：原Agent本地草稿或状态更新；未调用模型、未访问数据库。

### a3 · leave_collect_modify_confirm · 第 3 轮

用户：结束日期调整为2026年11月11日，开始日期、假别和原因都不变。

输入包含历史 4 条；待确认动作：`submit_leave`。

预期模型输出：

```json
{"intent":"submit_leave","extracted_info":{"leave_type":"事假","start_date":"2026-11-09","end_date":"2026-11-11","reason":"家中设备维修"}}
```

预期节点：submit_leave

原 Agent 在预设标签下的实际回复：

```text
请确认以下请假申请：
- 类型：事假
- 时间：2026-11-09 至 2026-11-11
- 工作日：3 天
- 原因：家中设备维修
- 审批路径：主管审批

回复「确认」提交，回复「取消」放弃。
```

执行范围：原Agent本地草稿或状态更新；未调用模型、未访问数据库。

### a4 · leave_collect_modify_confirm · 第 4 轮

用户：内容已经核对完毕，就按现在这份提交。

输入包含历史 6 条；待确认动作：`submit_leave`。

预期模型输出：

```json
{"intent":"confirm_action","extracted_info":{}}
```

预期节点：execute_pending

原 Agent 在预设标签下的实际回复：

```text
当前终端尚未接入业务数据库：知识库检索、申请查询、审批和正式提交暂不可用。可以继续普通对话、收集请假信息或取消草稿。当前没有执行数据库操作。
```

执行范围：仅验证进入确认执行节点；未连接数据库，草稿在当前终端保留，不声称已提交。

### b1 · cancel_draft_then_new_request · 第 1 轮

用户：请帮我整理2026年11月16日到11月17日的年假，原因为安排休息。

输入包含历史 0 条；待确认动作：`None`。

预期模型输出：

```json
{"intent":"submit_leave","extracted_info":{"leave_type":"年假","start_date":"2026-11-16","end_date":"2026-11-17","reason":"安排休息"}}
```

预期节点：submit_leave

原 Agent 在预设标签下的实际回复：

```text
请确认以下请假申请：
- 类型：年假
- 时间：2026-11-16 至 2026-11-17
- 工作日：2 天
- 原因：安排休息
- 审批路径：主管审批

回复「确认」提交，回复「取消」放弃。
```

执行范围：原Agent本地草稿或状态更新；未调用模型、未访问数据库。

### b2 · cancel_draft_then_new_request · 第 2 轮

用户：先别提交了，刚才这份年假草稿我不要了。

输入包含历史 2 条；待确认动作：`submit_leave`。

预期模型输出：

```json
{"intent":"cancel_action","extracted_info":{}}
```

预期节点：clear_pending

原 Agent 在预设标签下的实际回复：

```text
已取消本次操作，未提交任何变更。
```

执行范围：原Agent本地草稿或状态更新；未调用模型、未访问数据库。

### b3 · cancel_draft_then_new_request · 第 3 轮

用户：另起一份病假申请，日期和原因我稍后补充，不沿用上次的。

输入包含历史 4 条；待确认动作：`None`。

预期模型输出：

```json
{"intent":"submit_leave","extracted_info":{"leave_type":"病假"}}
```

预期节点：submit_leave

原 Agent 在预设标签下的实际回复：

```text
请提供以下信息：
- 请假时间（格式：2026-06-23 到 2026-06-28）
- 请假原因
```

执行范围：原Agent本地草稿或状态更新；未调用模型、未访问数据库。

### c1 · confirm_existing_approval · 第 1 轮

用户：这份驳回意见不用改，确认执行刚才的审批。

输入包含历史 2 条；待确认动作：`approve`。

预期模型输出：

```json
{"intent":"confirm_action","extracted_info":{}}
```

预期节点：execute_pending

原 Agent 在预设标签下的实际回复：

```text
当前终端尚未接入业务数据库：知识库检索、申请查询、审批和正式提交暂不可用。可以继续普通对话、收集请假信息或取消草稿。当前没有执行数据库操作。
```

执行范围：仅验证确认路由，不执行审批落库。

### d1 · cancel_other_submitted_request · 第 1 轮

用户：先暂停刚才的审批。我现在要撤回自己已提交的leave_52f7b0e649ad，这次操作对象不是刚才那条。

输入包含历史 2 条；待确认动作：`approve`。

预期模型输出：

```json
{"intent":"delete_application","extracted_info":{"app_id":"leave_52f7b0e649ad"}}
```

预期节点：delete

原 Agent 在预设标签下的实际回复：

```text
当前终端尚未接入业务数据库：知识库检索、申请查询、审批和正式提交暂不可用。可以继续普通对话、收集请假信息或取消草稿。当前没有执行数据库操作。
```

执行范围：终端delete节点为数据库不可用提示；生产环境如何替换原待确认动作需与Agent端确认，不将当前保留旧草稿当作生产规范。

### e1 · side_query_then_resume_draft · 第 1 轮

用户：这份草稿先留着。我想先查一下公司关于年假申请的制度。

输入包含历史 2 条；待确认动作：`submit_leave`。

预期模型输出：

```json
{"intent":"policy_qa","extracted_info":{}}
```

预期节点：retrieve → answer

原 Agent 在预设标签下的实际回复：

```text
当前终端尚未接入业务数据库：知识库检索、申请查询、审批和正式提交暂不可用。可以继续普通对话、收集请假信息或取消草稿。当前没有执行数据库操作。
```

执行范围：未连接知识库，回复明确不可用；不编造制度内容，也不声称已完成检索。

### e2 · side_query_then_resume_draft · 第 2 轮

用户：回到那份事假草稿，把结束日期改为2026年11月25日，其他信息沿用。

输入包含历史 4 条；待确认动作：`submit_leave`。

预期模型输出：

```json
{"intent":"submit_leave","extracted_info":{"leave_type":"事假","start_date":"2026-11-23","end_date":"2026-11-25","reason":"处理搬家事项"}}
```

预期节点：submit_leave

原 Agent 在预设标签下的实际回复：

```text
请确认以下请假申请：
- 类型：事假
- 时间：2026-11-23 至 2026-11-25
- 工作日：3 天
- 原因：处理搬家事项
- 审批路径：主管审批

回复「确认」提交，回复「取消」放弃。
```

执行范围：原Agent本地草稿或状态更新；未调用模型、未访问数据库。

### e3 · side_query_then_resume_draft · 第 3 轮

用户：现在草稿没问题了，确认提交当前这份。

输入包含历史 6 条；待确认动作：`submit_leave`。

预期模型输出：

```json
{"intent":"confirm_action","extracted_info":{}}
```

预期节点：execute_pending

原 Agent 在预设标签下的实际回复：

```text
当前终端尚未接入业务数据库：知识库检索、申请查询、审批和正式提交暂不可用。可以继续普通对话、收集请假信息或取消草稿。当前没有执行数据库操作。
```

执行范围：仅验证确认路由，不执行提交落库。

## 审核后再进行的工作

1. 确认表达、标签和状态边界，尤其是操作另一对象时如何处理旧待确认动作。
2. 扩展真实轨迹和表达变化，按整条轨迹及模板族划分，不将旧评测题写入训练集。
3. 用 Qwen tokenizer 检查完整历史和监督目标是否被截断，再决定长度和批次。
4. 保留 V2；从同一 Full SFT 底座建立新的 LoRA 对照，并重测完整业务流程。

详细上下文、模型输入、源代码哈希及状态检查见 [samples.json](samples.json)；人工场景定义见 [scenarios.json](scenarios.json)。

## 编码检查

Qwen tokenizer 已检查全部12条预览；最长 864 token，max_length=1536 下无截断，所有JSON答案与结束标记均完整纳入监督。此处只是模板检查，未训练或评测模型。
