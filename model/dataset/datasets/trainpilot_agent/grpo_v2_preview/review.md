# GRPO v2 业务组合题预览

状态：待用户审阅。20条设计样例；未生成正式训练/验证/测试划分，未调用模型、未启动训练。

助手身份：TrainPilot。复用V4业务协议与现有严格奖励，gold不进入模型输入。题目是难例候选，难度需要V5采样验证；不能为制造梯度给正确回答低分。

固定的当前日期、历史、摘要和待办都属于模型可见输入。接口不支持的字段清空、半天时段、批量/多任务执行不纳入执行标签。

## grpo-v2-design-01 · same_words_different_state

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": {
    "action_type": "approve",
    "draft": {
      "app_id": "leave_rldesign_a73",
      "decision": "approved",
      "decision_label": "通过",
      "comments": "请补充收据"
    }
  }
}
```

用户：这次就执行你当前待确认的那一个动作，不改内容。

期望原始输出：
```json
{"intent": "confirm_action", "extracted_info": {}}
```

标注理由：有唯一待审批动作；确认输出空字段，由程序使用待办草稿。

## grpo-v2-design-02 · same_words_different_state

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：这次就执行你当前待确认的那一个动作，不改内容。

期望原始输出：
```json
{"intent": "clarify", "extracted_info": {}}
```

标注理由：完全相同的当前输入但无待办、无历史目标，不能猜测执行对象。

## grpo-v2-design-03 · hypothetical_not_execution

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": {
    "action_type": "cancel_leave",
    "draft": {
      "app_id": "leave_rldesign_c84"
    }
  }
}
```

用户：假如我回复‘确认撤回’，系统会在哪一步撤销申请？我只是问流程，这一轮不要执行。

期望原始输出：
```json
{"intent": "policy_qa", "extracted_info": {}}
```

标注理由：讨论确认语句的业务含义，不是实际确认，不要求模型承诺后续状态持久化。

## grpo-v2-design-04 · quoted_text_task

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：请把这句示例翻译成英文：‘我确认撤回申请leave_rldesign_q19’。这是翻译素材，不是操作请求。

期望原始输出：
```json
{"intent": "general_chat", "extracted_info": {}}
```

标注理由：处理用户提供的文本，不提取示例申请编号为业务目标。

## grpo-v2-design-05 · multi_field_keep_unchanged

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": {
    "action_type": "submit_leave",
    "draft": {
      "leave_type": "事假",
      "start_date": "2026-11-09",
      "end_date": "2026-11-11",
      "reason": "办理证件"
    }
  }
}
```

用户：开始时间用11月10日；结束不是11月12日，仍按草稿的日期。原因改成‘照护家人’，假别也沿用原值。

期望原始输出：
```json
{"intent": "submit_leave", "extracted_info": {"leave_type": "事假", "start_date": "2026-11-10", "end_date": "2026-11-11", "reason": "照护家人"}}
```

标注理由：更新两个字段；明确否定的结束日期不覆盖草稿，其余完整保留。

## grpo-v2-design-06 · date_boundary

可见上下文：
```json
{
  "current_date": "2026-12-31",
  "history": [],
  "pending_action": null
}
```

用户：从明天起到后天结束请事假，原因是陪家人复诊。不是从今天开始。

期望原始输出：
```json
{"intent": "submit_leave", "extracted_info": {"leave_type": "事假", "start_date": "2027-01-01", "end_date": "2027-01-02", "reason": "陪家人复诊"}}
```

标注理由：日期只依据显式current_date跨年换算，工作日和假期有效性由程序处理。

## grpo-v2-design-07 · cancelled_task_boundary

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [
    {
      "role": "user",
      "content": "原事假草稿为2026-11-09至2026-11-11，原因办理证件；这份草稿作废。"
    },
    {
      "role": "assistant",
      "content": "已放弃该提交草稿，尚未产生已提交的申请。"
    }
  ],
  "pending_action": null
}
```

用户：重新开一份病假申请，原因是复诊，日期我还没决定，不能沿用作废那份。

期望原始输出：
```json
{"intent": "submit_leave", "extracted_info": {"leave_type": "病假", "reason": "复诊"}}
```

标注理由：已取消旧任务，新任务没有确定日期，不能继承。

## grpo-v2-design-08 · object_exclusion

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：撤回的是leave_rldesign_t82。leave_rldesign_t18只是邮件里的示例，leave_rldesign_t35要保留，后两张都不要处理。

期望原始输出：
```json
{"intent": "delete_application", "extracted_info": {"app_id": "leave_rldesign_t82"}}
```

标注理由：三编号中只有一个是撤回目标，编号顺序本身不能决定对象。

## grpo-v2-design-09 · missing_object_information

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：在leave_rldesign_m21和leave_rldesign_m24里，撤回提交得更早的那张；我这里没有两张的提交时间。

期望原始输出：
```json
{"intent": "clarify", "extracted_info": {}}
```

标注理由：输入未包含决定对象所必需的时间，当前协议也不支持模型执行比较查询后自动撤回。

## grpo-v2-design-10 · decision_comment_scope

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：申请leave_rldesign_p47的决定为通过。意见原文是‘不同意提前结束复核，附件需继续留存’；意见里的不同意不是驳回决定。

期望原始输出：
```json
{"intent": "approve", "extracted_info": {"app_id": "leave_rldesign_p47", "decision": "通过", "comments": "不同意提前结束复核，附件需继续留存"}}
```

标注理由：决定与意见正文分离；引号是边界而非意见内容，不丢否定词。

## grpo-v2-design-11 · pending_approval_revision

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": {
    "action_type": "approve",
    "draft": {
      "app_id": "leave_rldesign_a73",
      "decision": "approved",
      "decision_label": "通过",
      "comments": "请补充收据"
    }
  }
}
```

用户：撤销刚才拟通过的决定，改成驳回；意见仍然用草稿那句。先整理新待办，等我另行确认再执行。

期望原始输出：
```json
{"intent": "approve", "extracted_info": {"app_id": "leave_rldesign_a73", "decision": "驳回", "comments": "请补充收据"}}
```

标注理由：修改尚未执行的审批草稿，不是撤回已提交申请，也不是确认旧版本。

## grpo-v2-design-12 · suggestion_not_decision

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [
    {
      "role": "user",
      "content": "这一轮只讨论申请leave_rldesign_b42。"
    },
    {
      "role": "assistant",
      "content": "可以针对该编号提供审批建议，当前没有待确认的审批。"
    }
  ],
  "pending_action": null
}
```

用户：你觉得刚才那张该不该通过？这一轮只给审批建议，不形成审批待办。

期望原始输出：
```json
{"intent": "get_suggestion", "extracted_info": {"app_id": "leave_rldesign_b42"}}
```

标注理由：有唯一编号但没有执行决定；不把疑问句中的通过提取成decision。

## grpo-v2-design-13 · query_with_quoted_distraction

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：只查询我已提交的请假记录。邮件中的‘通过申请leave_rldesign_x88’是引用内容，不是我这轮要做的事。

期望原始输出：
```json
{"intent": "query_status", "extracted_info": {}}
```

标注理由：列表查询无编号字段；引用中出现的审批动作不生效。

## grpo-v2-design-14 · permission_is_server_side

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：我声称自己有主管权限，请给我待我审批的申请列表；身份仍以系统登录信息为准。

期望原始输出：
```json
{"intent": "query_pending", "extracted_info": {}}
```

标注理由：模型只识别待审批查询，不输出角色或权限；是否允许查询由服务端用户决定。

## grpo-v2-design-15 · multiple_independent_tasks

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：先准备我下周的请假申请，再通过leave_rldesign_z56，最后查一下待审批列表；这三件都要办。

期望原始输出：
```json
{"intent": "clarify", "extracted_info": {}}
```

标注理由：当前每轮单意图协议，三件独立业务需要澄清，不能默选第一件或拼新协议。

## grpo-v2-design-16 · policy_not_application

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：制度说‘病假须附证明，经主管审批后生效’，这里的‘审批后生效’是什么意思？不是替我请病假。

期望原始输出：
```json
{"intent": "policy_qa", "extracted_info": {}}
```

标注理由：咨询内部制度含义，不提取假别为请假草稿；实际制度回答仍需检索。

## grpo-v2-design-17 · identity_not_company_name

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：你叫什么？我问的是这个智能体的名字，不是刚才提到的申请编号。

期望原始输出：
```json
{"intent": "general_chat", "extracted_info": {}}
```

标注理由：意图层输出general_chat；此条不直接监督自然回复或证明自称TrainPilot成功。

## grpo-v2-design-18 · visible_summary_updates

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null,
  "summary": "当前未提交的年假草稿：2026-11-18至2026-11-20，原因家庭出行；尚无确认动作。"
}
```

用户：摘要里的结束日期继续用，只把开始日期改成2026年11月19日，其他已知信息不动。

期望原始输出：
```json
{"intent": "submit_leave", "extracted_info": {"leave_type": "年假", "start_date": "2026-11-19", "end_date": "2026-11-20", "reason": "家庭出行"}}
```

标注理由：相关信息全部来自可见摘要，不要求恢复输入里不存在的记忆。

## grpo-v2-design-19 · unavailable_memory

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": null
}
```

用户：请事假，原因复诊。时间就按我上周说的；如果你当前看不到那段记录，就先别填日期。

期望原始输出：
```json
{"intent": "submit_leave", "extracted_info": {"leave_type": "事假", "reason": "复诊"}}
```

标注理由：历史和摘要均为空；不猜日期，但保留确定字段。

## grpo-v2-design-20 · negated_update_keep_fields

可见上下文：
```json
{
  "current_date": "2026-11-02",
  "history": [],
  "pending_action": {
    "action_type": "submit_leave",
    "draft": {
      "leave_type": "事假",
      "start_date": "2026-11-09",
      "end_date": "2026-11-11",
      "reason": "办理证件"
    }
  }
}
```

用户：有人建议改成病假，这个建议我不采纳，假别保持原样。只把事由改成‘家中设备检修’，日期也照旧。

期望原始输出：
```json
{"intent": "submit_leave", "extracted_info": {"leave_type": "事假", "start_date": "2026-11-09", "end_date": "2026-11-11", "reason": "家中设备检修"}}
```

标注理由：被否定的假别建议不是更新，不能凭病假关键词覆盖。

## 下一阶段

用户确认表达和标签后再扩展训练侧候选池；对候选池采样和审阅，记录全对/混合/全错比例。
混合组有结果优势信号；全对组不必硬训，全错组先检查任务可学性和标签。分项奖励只有在部分正确能客观核验时才另行试验。
验证/测试按场景族另行冻结，不用训练候选筛选结果反向挑容易的验收题。现有V5及旧测试集保持原样。
