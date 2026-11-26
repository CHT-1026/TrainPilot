# TrainPilot V4 数据构建记录

按用户批准的覆盖规范构造；没有运行模型训练。

模型身份为TrainPilot；完整已知字段；多任务先澄清；字段清空不训练伪执行。

## 数据规模

|划分|记录数|来源|模式|
|---|---:|---|---|
|train|4842|{"v4_new_scenario": 2413, "v3_train_replay": 960, "v4_agent_runtime": 240, "v2_train_replay": 1133, "v4_agent_identity": 96}|{"intent_parse": 4746, "direct_reply": 72, "agent_reply": 24}|
|val|375|{"v4_new_scenario": 311, "v4_agent_runtime": 40, "v4_agent_identity": 24}|{"intent_parse": 357, "agent_reply": 6, "direct_reply": 12}|
|test|373|{"v4_new_scenario": 309, "v4_agent_runtime": 40, "v4_agent_identity": 24}|{"intent_parse": 355, "direct_reply": 12, "agent_reply": 6}|

旧回放只来自旧train；V3内部已经包含的V2回放不再收录。恢复六类有效旧场景的候选共336条，每条保留来源ID。排除理由逐条见replay_selection.json。

检查范围：来源SHA256、结构化字段与可见依据、原Agent提示词逐条一致性、输入重复/冲突、分组与轨迹划分、已知旧留出输入排除。
新增脚本化原Agent请求数：392；多轮实际上下文和状态证据见trajectories.jsonl。

标签是规则构造的期望答案；脚本化Agent不访问外部模型或数据库。程序通过不能证明模型效果或数据库业务成功。

## 训练类别覆盖

|类别|训练记录数|
|---|---:|
|advice|48|
|advice_missing|4|
|ambiguous_intent|4|
|ambiguous_object|48|
|approval_comment|48|
|approval_edit|48|
|approval_subset|381|
|approve|70|
|approve_missing_decision|30|
|approve_missing_id|30|
|cancel|156|
|cancel_action|56|
|cancel_missing_id|30|
|cancel_then_new|312|
|clarify|40|
|collect_modify_confirm|416|
|confirm|156|
|confirm_action|56|
|delete_application|50|
|delete_missing|4|
|general_chat|40|
|get_suggestion|40|
|history_fill|56|
|history_modify|56|
|identity|24|
|identity_correction|16|
|identity_execution|16|
|identity_greeting|16|
|identity_missing|16|
|identity_name|16|
|identity_role|16|
|invalid_range|48|
|leave_full|80|
|leave_relative|40|
|leave_subset|766|
|missing_all|30|
|missing_dates|30|
|missing_reason|30|
|missing_type|23|
|modify_both|48|
|modify_end|48|
|modify_reason|48|
|modify_start|48|
|modify_type|46|
|multi_task|48|
|negated_modification|48|
|other_object|176|
|pending_cancel_existing|56|
|pending_decision|80|
|pending_other_task|56|
|policy|48|
|policy_qa|90|
|query_pending|44|
|query_status|54|
|quoted_policy|48|
|relative|24|
|side_query_resume|312|
|suggest_missing_id|30|
|uncertain_dates|48|
|withdraw_abandon|48|
|withdraw_confirm|48|

## 数据与评测边界

- val/test按新表达族和整条轨迹划分，是合成开发留出集。旧V2/V3测试作为回归。最终独立验收题尚未制作，不能用本次生成器的成功率充当模型成绩。
- 来源语义经模板检查及逐条规则检查；可见字符串证据不证明所有语义正确。仍需人工抽查具体表达和标签。
- 只有用户需求理解与结构化动作选择，以及实际普通回复模式；没有训练可执行的原生function calling或隐藏思维链。
- 字段清空、直接切换未取消任务、闲聊/澄清后保留待办，以及真正的数据库操作存在Agent接口或集成待办，详见spec_v4.json。

## 编码检查

最大长度1130 tokens；上限1536；截断0条；无有效答案0条。

只监督assistant模板正文及im_end结束符，用户输入和padding不监督。Qwen模板的空think前缀不含人工思维链。

监督token分布见encoding_check.json；样本条数与独立语义数量不同。

## 下一步

1. 按类别抽查数据与标签，重点检查确认/放弃和编号对象。
2. 提示词单独做V3对照；Agent部署时须使用intent_prompt_v4.txt同版，当前尚未修改原Agent。
3. 从原Full SFT重新初始化LoRA，保留V2/V3，保存各epoch导出并根据旧/新验证联合选取。
4. 使用独立验收题及真实Agent多轮测试确认能力，最后再接数据库业务验证执行结果。

