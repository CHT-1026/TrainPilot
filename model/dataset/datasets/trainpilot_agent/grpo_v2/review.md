# GRPO v2 正式候选数据

训练候选600，验证90，测试150。训练有效池仍待V5采样筛选。三类范围与数量已由用户批准。

每类训练200/验证30/测试50，表达家族按20/3/5隔离；每家族10个实例。原V5、旧测试集与奖励保持原样。

以下仅展示训练侧样例；验证和测试不用于挑选难例。

## rl2-train-state_confirmation-00-00

上下文：
```json
{
  "current_date": "2027-02-01",
  "history": [],
  "pending_action": {
    "action_type": "approve",
    "draft": {
      "app_id": "leave_rl2_st_00_00",
      "decision": "approved",
      "decision_label": "通过",
      "comments": "同意安排"
    }
  }
}
```

我核对的是你此刻列出的唯一待办，内容不变，现在照它执行。

```json
{"intent": "confirm_action", "extracted_info": {}}
```

## rl2-train-state_confirmation-00-01

上下文：
```json
{
  "current_date": "2027-02-01",
  "history": [],
  "pending_action": null
}
```

我核对的是你此刻列出的唯一待办，内容不变，现在照它执行。

```json
{"intent": "clarify", "extracted_info": {}}
```

## rl2-train-hypothetical_policy-00-00

上下文：
```json
{
  "current_date": "2027-02-01",
  "history": [],
  "pending_action": {
    "action_type": "approve",
    "draft": {
      "app_id": "leave_rl2_hy_00_00",
      "decision": "approved",
      "decision_label": "通过",
      "comments": "同意安排"
    }
  }
}
```

我想弄懂流程：如果回复“确认执行当前审批”，系统会进入哪个环节？这只是提问，当前不操作。

```json
{"intent": "policy_qa", "extracted_info": {}}
```

## rl2-train-hypothetical_policy-00-01

上下文：
```json
{
  "current_date": "2027-02-01",
  "history": [],
  "pending_action": {
    "action_type": "cancel_leave",
    "draft": {
      "app_id": "leave_rl2_hy_00_01"
    }
  }
}
```

我想弄懂流程：如果回复“确认撤回这张申请”，系统会进入哪个环节？这只是提问，当前不操作。

```json
{"intent": "policy_qa", "extracted_info": {}}
```

## rl2-train-decision_comment-00-00

上下文：
```json
{
  "current_date": "2027-02-01",
  "history": [
    {
      "role": "user",
      "content": "先拟另一张leave_rl2_de_00_00_old通过，意见为原材料已收到。"
    },
    {
      "role": "assistant",
      "content": "已准备leave_rl2_de_00_00_old的审批预览，尚未执行。"
    }
  ],
  "pending_action": {
    "action_type": "approve",
    "draft": {
      "app_id": "leave_rl2_de_00_00_old",
      "decision": "approved",
      "decision_label": "通过",
      "comments": "原材料已收到"
    }
  }
}
```

leave_rl2_de_00_00的审批决定是通过。意见请原样记为“不同意提前结束复核，附件需继续留存”，不能从意见里的词反推另一种决定。

```json
{"intent": "approve", "extracted_info": {"app_id": "leave_rl2_de_00_00", "decision": "通过", "comments": "不同意提前结束复核，附件需继续留存"}}
```

## rl2-train-decision_comment-00-01

上下文：
```json
{
  "current_date": "2027-02-01",
  "history": [],
  "pending_action": {
    "action_type": "approve",
    "draft": {
      "app_id": "leave_rl2_de_00_01",
      "decision": "approved",
      "decision_label": "通过",
      "comments": "旧意见暂存"
    }
  }
}
```

leave_rl2_de_00_01的审批决定是驳回。意见请原样记为“同意安排，但不同意省略交接材料”，不能从意见里的词反推另一种决定。

```json
{"intent": "approve", "extracted_info": {"app_id": "leave_rl2_de_00_01", "decision": "驳回", "comments": "同意安排，但不同意省略交接材料"}}
```

## 限制

- 84措辞家族，每家族10个状态/字段实例，840条不等于840种独立语义
- 同一业务语义在各划分共有；表达族留出不是业务语义完全隔离，也不是真实用户独立验收
- 训练候选是否适合GRPO须由采样验证；不因凑数量把正确回答改判为错
- 不更改Agent权限/确认机制；这里只验证原始意图和字段，不承诺咨询后待办持久化