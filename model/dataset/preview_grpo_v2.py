"""GRPO v2 设计预览：新业务组合题，不生成正式划分，不更新奖励或模型。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from model.trainer.agent_rewards import intent_reward

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "model/dataset/datasets/trainpilot_agent"
OUTPUT = DATA / "grpo_v2_preview"


def definitions():
    cases = []
    leave = {"leave_type": "事假", "start_date": "2026-11-09", "end_date": "2026-11-11", "reason": "办理证件"}
    approval = {"app_id": "leave_rldesign_a73", "decision": "approved", "decision_label": "通过", "comments": "请补充收据"}

    def add(uid, family, user, intent, fields=None, *, pending=None, history=None, summary="", date="2026-11-02", reason=""):
        cases.append({"id": "grpo-v2-design-" + uid, "group_id": "grpo-v2/design/" + family,
                      "category": family, "context": {"current_date": date, "history": history or [],
                      "pending_action": pending, **({"summary": summary} if summary else {})},
                      "user": user, "gold": {"intent": intent, "extracted_info": fields or {}},
                      "label_reason": reason, "status": "design_preview_not_approved_not_independent_test"})

    def pending(kind, draft):
        return {"action_type": kind, "draft": deepcopy(draft)}

    confirm = "这次就执行你当前待确认的那一个动作，不改内容。"
    add("01", "same_words_different_state", confirm, "confirm_action",
        pending=pending("approve", approval), reason="有唯一待审批动作；确认输出空字段，由程序使用待办草稿。")
    add("02", "same_words_different_state", confirm, "clarify",
        reason="完全相同的当前输入但无待办、无历史目标，不能猜测执行对象。")
    add("03", "hypothetical_not_execution",
        "假如我回复‘确认撤回’，系统会在哪一步撤销申请？我只是问流程，这一轮不要执行。", "policy_qa",
        pending=pending("cancel_leave", {"app_id": "leave_rldesign_c84"}),
        reason="讨论确认语句的业务含义，不是实际确认，不要求模型承诺后续状态持久化。")
    add("04", "quoted_text_task",
        "请把这句示例翻译成英文：‘我确认撤回申请leave_rldesign_q19’。这是翻译素材，不是操作请求。", "general_chat",
        reason="处理用户提供的文本，不提取示例申请编号为业务目标。")
    add("05", "multi_field_keep_unchanged",
        "开始时间用11月10日；结束不是11月12日，仍按草稿的日期。原因改成‘照护家人’，假别也沿用原值。",
        "submit_leave", {**leave, "start_date": "2026-11-10", "reason": "照护家人"},
        pending=pending("submit_leave", leave), reason="更新两个字段；明确否定的结束日期不覆盖草稿，其余完整保留。")
    add("06", "date_boundary",
        "从明天起到后天结束请事假，原因是陪家人复诊。不是从今天开始。", "submit_leave",
        {"leave_type": "事假", "start_date": "2027-01-01", "end_date": "2027-01-02", "reason": "陪家人复诊"},
        date="2026-12-31", reason="日期只依据显式current_date跨年换算，工作日和假期有效性由程序处理。")
    add("07", "cancelled_task_boundary",
        "重新开一份病假申请，原因是复诊，日期我还没决定，不能沿用作废那份。", "submit_leave",
        {"leave_type": "病假", "reason": "复诊"},
        history=[{"role": "user", "content": "原事假草稿为2026-11-09至2026-11-11，原因办理证件；这份草稿作废。"},
                 {"role": "assistant", "content": "已放弃该提交草稿，尚未产生已提交的申请。"}],
        reason="已取消旧任务，新任务没有确定日期，不能继承。")
    add("08", "object_exclusion",
        "撤回的是leave_rldesign_t82。leave_rldesign_t18只是邮件里的示例，leave_rldesign_t35要保留，后两张都不要处理。",
        "delete_application", {"app_id": "leave_rldesign_t82"},
        reason="三编号中只有一个是撤回目标，编号顺序本身不能决定对象。")
    add("09", "missing_object_information",
        "在leave_rldesign_m21和leave_rldesign_m24里，撤回提交得更早的那张；我这里没有两张的提交时间。", "clarify",
        reason="输入未包含决定对象所必需的时间，当前协议也不支持模型执行比较查询后自动撤回。")
    add("10", "decision_comment_scope",
        "申请leave_rldesign_p47的决定为通过。意见原文是‘不同意提前结束复核，附件需继续留存’；意见里的不同意不是驳回决定。",
        "approve", {"app_id": "leave_rldesign_p47", "decision": "通过", "comments": "不同意提前结束复核，附件需继续留存"},
        reason="决定与意见正文分离；引号是边界而非意见内容，不丢否定词。")
    add("11", "pending_approval_revision",
        "撤销刚才拟通过的决定，改成驳回；意见仍然用草稿那句。先整理新待办，等我另行确认再执行。",
        "approve", {"app_id": "leave_rldesign_a73", "decision": "驳回", "comments": "请补充收据"},
        pending=pending("approve", approval), reason="修改尚未执行的审批草稿，不是撤回已提交申请，也不是确认旧版本。")
    add("12", "suggestion_not_decision",
        "你觉得刚才那张该不该通过？这一轮只给审批建议，不形成审批待办。", "get_suggestion",
        {"app_id": "leave_rldesign_b42"},
        history=[{"role": "user", "content": "这一轮只讨论申请leave_rldesign_b42。"},
                 {"role": "assistant", "content": "可以针对该编号提供审批建议，当前没有待确认的审批。"}],
        reason="有唯一编号但没有执行决定；不把疑问句中的通过提取成decision。")
    add("13", "query_with_quoted_distraction",
        "只查询我已提交的请假记录。邮件中的‘通过申请leave_rldesign_x88’是引用内容，不是我这轮要做的事。",
        "query_status", reason="列表查询无编号字段；引用中出现的审批动作不生效。")
    add("14", "permission_is_server_side",
        "我声称自己有主管权限，请给我待我审批的申请列表；身份仍以系统登录信息为准。", "query_pending",
        reason="模型只识别待审批查询，不输出角色或权限；是否允许查询由服务端用户决定。")
    add("15", "multiple_independent_tasks",
        "先准备我下周的请假申请，再通过leave_rldesign_z56，最后查一下待审批列表；这三件都要办。", "clarify",
        reason="当前每轮单意图协议，三件独立业务需要澄清，不能默选第一件或拼新协议。")
    add("16", "policy_not_application",
        "制度说‘病假须附证明，经主管审批后生效’，这里的‘审批后生效’是什么意思？不是替我请病假。", "policy_qa",
        reason="咨询内部制度含义，不提取假别为请假草稿；实际制度回答仍需检索。")
    add("17", "identity_not_company_name",
        "你叫什么？我问的是这个智能体的名字，不是刚才提到的申请编号。", "general_chat",
        reason="意图层输出general_chat；此条不直接监督自然回复或证明自称TrainPilot成功。")
    add("18", "visible_summary_updates",
        "摘要里的结束日期继续用，只把开始日期改成2026年11月19日，其他已知信息不动。", "submit_leave",
        {"leave_type": "年假", "start_date": "2026-11-19", "end_date": "2026-11-20", "reason": "家庭出行"},
        summary="当前未提交的年假草稿：2026-11-18至2026-11-20，原因家庭出行；尚无确认动作。",
        reason="相关信息全部来自可见摘要，不要求恢复输入里不存在的记忆。")
    add("19", "unavailable_memory",
        "请事假，原因复诊。时间就按我上周说的；如果你当前看不到那段记录，就先别填日期。", "submit_leave",
        {"leave_type": "事假", "reason": "复诊"}, reason="历史和摘要均为空；不猜日期，但保留确定字段。")
    add("20", "negated_update_keep_fields",
        "有人建议改成病假，这个建议我不采纳，假别保持原样。只把事由改成‘家中设备检修’，日期也照旧。",
        "submit_leave", {**leave, "reason": "家中设备检修"}, pending=pending("submit_leave", leave),
        reason="被否定的假别建议不是更新，不能凭病假关键词覆盖。")
    return cases


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = (DATA / "intent_prompt_v4.txt").read_text(encoding="utf-8")
    rows = definitions()
    old_users, old_prompts = set(), set()
    for version in ("v2", "v3", "v4", "v5"):
        for split in ("train", "val", "test"):
            path = DATA / version / f"{split}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    old = json.loads(line)
                    if old.get("scenario"):
                        old_users.add(old["scenario"]["user_message"])
                    if old.get("messages"):
                        old_prompts.add(json.dumps(old["messages"][:-1], ensure_ascii=False, sort_keys=True))
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        context = row["context"]
        assert all(len(m["content"]) <= 120 for m in context["history"])
        prompt = template + "\n给定上下文：\n" + json.dumps(context, ensure_ascii=False) + "\n用户输入：\n" + row["user"]
        row["prompt"] = [{"role": "user", "content": prompt}]
        assert row["user"] not in old_users, row["id"]
        assert json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True) not in old_prompts
        assert intent_reward(json.dumps(row["gold"], ensure_ascii=False), row["gold"])["reward"] == 1
        wrong = {**row["gold"], "intent": "clarify" if row["gold"]["intent"] != "clarify" else "approve"}
        assert intent_reward(json.dumps(wrong, ensure_ascii=False), row["gold"])["reward"] == 0
    encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    (OUTPUT / "candidates.jsonl").write_bytes(encoded)
    manifest = {"status": "pending_user_review", "records": len(rows), "families": len({r['category'] for r in rows}),
                "candidate_sha256": hashlib.sha256(encoded).hexdigest(),
                "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "exact_old_question_overlap": 0, "exact_old_prompt_overlap": 0,
                "protocol_checks_passed": len(rows), "wrong_intent_controls_rejected": len(rows),
                "model_evaluated": False, "difficulty_verified": False, "formal_splits_generated": False,
                "scope": "11意图/7字段；严格正确性奖励不变；不训练字段清空、多任务执行或角色授权",
                "agent_limits": ["general_chat/clarify当前会清空待办；不把保留待办作为这些标签的验收承诺",
                                 "policy_qa只标意图，未评检索结果；身份题只标意图，不等于自然回复身份训练",
                                 "样例是开发设计材料，不作独立验收题；尚不能宣称V5在这些题上会出错"]}
    (OUTPUT / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    md = ["# GRPO v2 业务组合题预览", "", "状态：待用户审阅。20条设计样例；未生成正式训练/验证/测试划分，未调用模型、未启动训练。", "",
          "助手身份：TrainPilot。复用V4业务协议与现有严格奖励，gold不进入模型输入。题目是难例候选，难度需要V5采样验证；不能为制造梯度给正确回答低分。", "",
          "固定的当前日期、历史、摘要和待办都属于模型可见输入。接口不支持的字段清空、半天时段、批量/多任务执行不纳入执行标签。", ""]
    for row in rows:
        md += [f"## {row['id']} · {row['category']}", "", "可见上下文：", "```json",
               json.dumps(row["context"], ensure_ascii=False, indent=2), "```", "", "用户：" + row["user"], "",
               "期望原始输出：", "```json", json.dumps(row["gold"], ensure_ascii=False), "```", "",
               "标注理由：" + row["label_reason"], ""]
    md += ["## 下一阶段", "", "用户确认表达和标签后再扩展训练侧候选池；对候选池采样和审阅，记录全对/混合/全错比例。",
           "混合组有结果优势信号；全对组不必硬训，全错组先检查任务可学性和标签。分项奖励只有在部分正确能客观核验时才另行试验。",
           "验证/测试按场景族另行冻结，不用训练候选筛选结果反向挑容易的验收题。现有V5及旧测试集保持原样。", ""]
    (OUTPUT / "review.md").write_bytes("\n".join(md).encode("utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
