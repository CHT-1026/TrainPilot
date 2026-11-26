# V6修复型SFT正式数据

2026-10-30。用户已确认预览与OA部分回答规则。训练起点：原Full SFT + 已有V5 LoRA；建立新优化器，保留V5/GRPO50对照。

| 划分 | 新修复 | 旧任务回放 | 合计 |
|---|---:|---:|---:|
| train | 520 | 520 | 1040 |
| val | 104 | 0 | 104 |
| test | 156 | 0 | 156 |

新样本包括可信待办确认/取消、引用咨询与执行、能力边界与制度咨询、撤回方向、完整字段更新、新任务不继承旧字段、部分/全量/无证据RAG、表格唯一行定位、身份职责。旧任务回放按模式与意图/类别分层，保留全部11种业务意图。

## 规模与划分

第一轮桥接采用520条新修复与520条旧任务回放的1:1配比。新修复共15组表述与证据家族，每组52条状态/任务变体，按10/2/3组分入训练/验证/测试；同句有无待办、同题不同证据、同证据不同输出模式不跨划分。1300条记录不等于1300种独立语义，也不承诺全部业务达标。

回放仅来自V5/train.jsonl，不读取旧验证/测试答案构造训练标签；旧评测内容仅用于排重。过滤歧义或无法直接核验的表格标签、重复prompt及与历史评测重合的prompt。已修改身份/模板的prompt也与同样规范化的历史评测比较，避免靠改称呼掩盖重合。

## 示例

### v6-train-f02-confirm-3

用户：当前预览无需再改，我同意按这一版执行。

```json
{"current_date": "2027-04-01", "history": [{"role": "assistant", "content": "刚才仅查询过申请列表，没有创建新的待办。"}], "pending_action": null}
```

目标：
```text
{"extracted_info":{},"intent":"clarify"}
```

### v6-train-f05-quoted-2

用户：请把“撤掉已提交申请”当作流程示例解释，我尚未决定做任何变更。

```json
{"current_date": "2027-04-01", "history": [], "pending_action": {"action_type": "approve", "draft": {"app_id": "leave_v6_f05_a", "decision": "approved", "decision_label": "通过", "comments": "第6批书面材料已收齐"}}}
```

目标：
```text
{"extracted_info":{},"intent":"policy_qa"}
```

### v6-train-f08-capability

用户：你在整理需求时会擅自假设我未表达的日期吗？

```json
{"current_date": "2027-04-01", "history": [], "pending_action": null}
```

目标：
```text
{"extracted_info":{},"intent":"general_chat"}
```

### v6-train-f08-abandon-withdraw

用户：原申请我要继续保留，当前撤回步骤不要落实。

```json
{"current_date": "2027-04-01", "history": [], "pending_action": {"action_type": "cancel_leave", "draft": {"app_id": "leave_v6_f08_w"}}}
```

目标：
```text
{"extracted_info":{},"intent":"cancel_action"}
```

### v6-train-f02-leave-end

用户：沿用当前任务，仅作这些修改：只把结束日期延到2027-04-10，其他已知信息保留。

```json
{"current_date": "2027-04-01", "history": [], "pending_action": {"action_type": "submit_leave", "draft": {"leave_type": "事假", "start_date": "2027-04-07", "end_date": "2027-04-08", "reason": "核对第3批设备交付清单"}}}
```

目标：
```text
{"extracted_info":{"end_date":"2027-04-10","leave_type":"事假","reason":"核对第3批设备交付清单","start_date":"2027-04-07"},"intent":"submit_leave"}
```

### v6-train-f08-oa-price_distractor

用户：需要核实参数样机 TP-V6-08 的额定承载值和采购价。

证据：
[1] TrainPilot干扰参数（合成资料） · 核验记录
其他参数样机TP-OTHER-08的负载为999 kg，采购价格为999999元。
[2] TrainPilot演练参数（合成资料） · 核验记录
参数样机TP-V6-08的采购价格为254400元。

目标：
```text
参数样机TP-V6-08的采购价格为254400元。[2]
当前资料未提供该样机的额定负载。[2]
```

### v6-train-f00-oa-none

用户：参数样机 TP-V6-00 的额定负载和采购价格分别是多少？

证据：
[1] TrainPilot演练参数（合成资料） · 核验记录
参数样机TP-V6-00运输时应固定底座。

目标：
```text
现有企业资料未覆盖该问题，建议联系对应业务部门确认。
```

### v6-train-f01-table-specific

用户：根据给定表格查明：类别“申请”、事项“信息更正2”的办理时限是多少？

证据：
[1] TrainPilot流程演练表（合成制度） · 核验记录
| 类别 | 事项 | 办理时限 |
| 申请 | 资料补录2 | 2个工作日 |
| 申请 | 信息更正2 | 4个工作日 |

目标：
```text
{"claims":[{"sources":[1],"text":"类别为“申请”、事项为“信息更正2”的办理时限为4个工作日。"}],"refused":false}
```

### v6-train-f02-role-direct_system

用户：你负责理解哪些业务需求，实际办理又由谁执行？

目标：
```text
我是 TrainPilot，负责理解业务需求、整理请假和审批信息，并依据给定资料协助查询制度。实际业务变更由系统在确认和权限校验后执行，我不会把未执行的操作说成已完成。
```

## 运行状态

数据结构与划分检查通过。编码/Agent实际提示词一致性结果另存encoding_check.json与integration_check.json；没有模型效果评测。训练入口已支持--init-adapter加载已有V5权重，使用专用run_trainpilot_lora_v6.sh；--resume只恢复本轮V6。默认1轮、130次更新、学习率1e-5。

## 限制

- 15 synthetic wording/evidence families, each with 52 state/task variants; records are not independent user semantics
- new train/val/test repair families separated 10/2/3; all old replay only in train
- exact overlap checks do not prove semantic or fact-level independence
- RAG supports supplied evidence only; not a live retrieval/tool execution evaluation
- ambiguous table input is refused under current RAG protocol; no new clarification JSON field
- no formal model training started; use the V6 continuation launcher

## 本地检查结果

- 数据7项、Agent提示词3项、既有RAG回归24项测试通过。
- 1300条全部完成V5 tokenizer编码检查；最长1543 tokens（2048上限），无截断、无零监督样本。
- 1248次实际Agent输入/引用校验通过：805条意图上下文结构一致（忽略JSON对象键顺序），260条企业RAG、160条OA、23条实际普通回复节点模板一致。未调用外部模型、数据库或真实业务工具。
- 520条旧业务回放的意图部分按11类均衡抽取，每类25或26条；原始目标和证据保持一致。
- 原评测文件及V5 adapter摘要未变。没有训练或模型效果评测；V6接续入口已实现，真实模型尚未训练。
