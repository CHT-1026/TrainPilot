"""V6修复型SFT预览；只起草26条样例，不生成正式训练划分、不训练。"""
from collections import Counter
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path

from model.dataset.build_agent_data import render_prompt
from model.dataset.build_agent_v5 import canonical, rag_row, synthetic_chunk
from model.trainer.agent_rewards import intent_reward

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "model/dataset/datasets/trainpilot_agent"
OUTPUT = DATA / "v6_preview"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def questions(row):
    found = set()
    for key in ("user", "question"):
        if isinstance(row.get(key), str):
            found.add(row[key])
    if row.get("scenario", {}).get("user_message"):
        found.add(row["scenario"]["user_message"])
    messages = row.get("prompt") or row.get("messages", [])[:-1]
    if messages:
        text = messages[-1]["content"]
        if "\n用户输入：\n" in text:
            found.add(text.rsplit("\n用户输入：\n", 1)[1])
        elif messages[-1]["role"] == "user":
            found.add(text)
    return found


def build():
    template = (DATA / "intent_prompt_v4.txt").read_text(encoding="utf-8")
    original_sources = json.loads((DATA / "v5/sources.json").read_text(encoding="utf-8"))
    sources = deepcopy(original_sources)
    enterprise_first = sources["enterprise_system"].splitlines()[0]
    sources["enterprise_system"] = sources["enterprise_system"].replace(
        enterprise_first, "你是 TrainPilot，一名企业知识与业务助手。只能依据【资料】回答，禁止使用资料外的知识。", 1)
    sources["oa_template"] = sources["oa_template"].replace(
        sources["oa_template"].splitlines()[0], "你是 TrainPilot，一名企业知识与业务助手。只能依据【资料】回答，不能使用资料之外的知识。", 1)
    old_rule = f"3. 如果资料不足以回答，直接回答“{sources['oa_refusal']}”，不要猜测。"
    new_rule = ("3. 资料仅覆盖部分问题时，先回答有依据的部分，并明确哪些信息未提供；"
                f"整题没有可支持的答案时，直接回答“{sources['oa_refusal']}”。不得猜测缺失值。")
    assert old_rule in sources["oa_template"]
    sources["oa_template"] = sources["oa_template"].replace(old_rule, new_rule)
    records = []

    def intent(uid, group, category, user, target, pending=None, fields=None, history=None, note=""):
        scenario = {"current_date": "2027-03-01", "history": history or [],
                    "pending_action": deepcopy(pending), "user_message": user}
        expected = {"intent": target, "extracted_info": fields or {}}
        assert intent_reward(canonical(expected), expected)["reward"] == 1
        prompt = render_prompt(template, scenario)
        records.append({"id": "v6-preview-"+uid, "group_id": "v6-preview/"+group,
            "mode": "intent_parse", "category": category, "scenario": scenario, "expected": expected,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": canonical(expected)}],
            "review_note": note, "requires_agent_prompt_sync": False})

    leave = {"action_type": "submit_leave", "draft": {"leave_type": "事假", "start_date": "2027-03-09",
             "end_date": "2027-03-10", "reason": "整理设备交付清单"}}
    approval = {"action_type": "approve", "draft": {"app_id": "leave_pv6_7b13", "decision": "approved",
                "decision_label": "通过", "comments": "已收齐书面资料"}}
    withdrawal = {"action_type": "cancel_leave", "draft": {"app_id": "leave_pv6_9e42"}}
    for name, pending in (("none", None), ("leave", leave), ("approval", approval), ("withdrawal", withdrawal)):
        history = [{"role": "assistant", "content": "上一项流程已经结束，目前没有待确认草稿。"}] if pending is None else []
        intent("confirm-"+name, "confirm-state-pair", "confirmation_state",
            "我核过你准备的内容了，就让那一份现在生效。", "confirm_action" if pending else "clarify",
            pending, history=history, note="同一句话配四种可信状态；用户自述不能创建待办。")
        intent("cancel-"+name, "cancel-state-pair", "cancellation_state",
            "先把手头这一步停掉，我不继续办理这一项了。", "cancel_action" if pending else "clarify",
            pending, history=history, note="取消的是当前待确认动作；没有待办时不能凭空取消业务。")
    intent("quoted-submit", "quoted-submit-vs-execute", "quoted_policy",
        "培训材料把“确认送出申请”当作例句，我想弄懂这一步的含义，不要动当前草稿。", "policy_qa", leave)
    intent("actual-submit", "quoted-submit-vs-execute", "actual_confirmation",
        "当前请假草稿的日期和事由都核对过了，我授权把这份申请送出。", "confirm_action", leave)
    intent("quoted-withdraw", "quoted-withdraw-vs-execute", "quoted_policy",
        "流程课件中的“撤掉已提交申请”是什么意思？这里只讲规则，不是在给你撤单指令。", "policy_qa", withdrawal)
    intent("actual-withdraw", "quoted-withdraw-vs-execute", "explicit_withdrawal",
        "我要撤回已经提交的 leave_pv6_c071，请先整理这张单的撤回预览。", "delete_application", withdrawal,
        fields={"app_id": "leave_pv6_c071"}, note="新目标编号明确，不沿用当前草稿里的其他编号。")
    intent("capability", "capability-vs-policy", "capability_boundary",
        "你整理业务资料时，会把我没写出的必要条件当成既定事实吗？", "general_chat")
    intent("company-policy", "capability-vs-policy", "company_policy",
        "企业的请假制度对申请材料缺项规定了怎样的补交步骤？", "policy_qa")
    intent("abandon-withdrawal", "withdrawal-direction", "withdraw_abandon",
        "撤单这一步作废，原来已经提交的申请继续保留。", "cancel_action", withdrawal)
    intent("execute-withdrawal", "withdrawal-direction", "withdraw_confirm",
        "就按当前撤回预览执行，让这张已提交申请进入撤回处理。", "confirm_action", withdrawal)
    changed_leave = {**leave["draft"], "end_date": "2027-03-12"}
    intent("edit-leave", "full-fields-update", "full_leave_fields",
        "只把这份请假草稿的结束日期顺延到2027-03-12，其余已知内容沿用，先给我预览。", "submit_leave", leave,
        fields=changed_leave, note="完整已知字段输出；不是仅返回本轮改动字段。")
    intent("edit-approval", "full-fields-update", "full_approval_fields",
        "当前审批结论改成驳回，意见原文写“预算口径尚未统一，请保留原始明细。”，先整理草稿，不落库。", "approve", approval,
        fields={"app_id": approval["draft"]["app_id"], "decision": "驳回", "comments": "预算口径尚未统一，请保留原始明细。"})

    for mode, device, load in (("enterprise", "TP-RP6101", 72), ("oa", "TP-RP6201", 93)):
        group = "partial-vs-no-evidence-"+mode
        question = f"参数样机 {device} 的额定负载和采购价格分别是多少？"
        for present in (True, False):
            text = f"参数样机 {device} 的额定负载为{load} kg。" if present else f"参数样机 {device} 运输时应固定底座。"
            chunk = synthetic_chunk("v6-preview-"+device+("-load" if present else "-transport"), text,
                                    title="TrainPilot数据演练参数（非真实企业资料）")
            claims = [{"text": f"参数样机 {device} 的额定负载为{load} kg；当前资料未提供采购价格。", "sources": [1]}] if present else []
            row = rag_row(sources, f"v6-preview-{mode}-{'partial' if present else 'unsupported'}", "v6-preview/"+group,
                          mode, question, [chunk], claims, "partial" if present else "unsupported", synthetic=True)
            row.update(review_note="同一问题仅改变证据覆盖程度：能答的先答，缺失值不猜；整题无依据才拒答。",
                       requires_agent_prompt_sync=True)
            records.append(row)
    table = synthetic_chunk("v6-preview-table", "| 类别 | 事项 | 办理时限 |\n| 申请 | 资料补录 | 2个工作日 |\n| 申请 | 信息更正 | 5个工作日 |",
                            title="TrainPilot数据演练流程表（非真实企业制度）")
    for label, question, claims in (
        ("ambiguous", "演练表中“申请”这一类的办理时限是多少？请只给唯一一个时限，不要自行选事项。", []),
        ("specific", "请查演练表：类别为“申请”、事项为“信息更正”的办理时限是多少？", [{"text": "类别为“申请”、事项为“信息更正”的办理时限为5个工作日。", "sources": [1]}]),
    ):
        row = rag_row(sources, "v6-preview-table-"+label, "v6-preview/table-row-identity", "enterprise", question,
                      [table], claims, "table_ambiguity" if label=="ambiguous" else "table_unique_key", synthetic=True)
        row.update(review_note="缺少唯一行定位时不随意挑选一个值；正式表格问答必须有足够行标识。", requires_agent_prompt_sync=True)
        records.append(row)

    user = "请用两句话界定你的业务职责，并说明谁来真正执行变更。"
    answer = "我是 TrainPilot，负责理解业务需求、整理请假和审批信息，并依据给定资料协助查询制度。实际业务变更由系统在确认和权限校验后执行，我不会把未执行的操作说成已完成。"
    records.append({"id": "v6-preview-role-direct", "group_id": "v6-preview/role-boundary", "mode": "direct_reply",
        "category": "identity_scope", "question": user, "messages": [{"role":"user","content":user},{"role":"assistant","content":answer}],
        "review_note": "无system也不自称具备考勤规则修改、日程管理等未接入业务。", "requires_agent_prompt_sync": False})
    agent_template = """
你是 TrainPilot，一名企业业务助手，负责：
1. 企业知识与制度问答：依据检索到的资料回答；
2. 请假流程：提交、查询、取消；
3. 审批：待审批列表、审批建议、通过/驳回。

限制：
- 只能依据系统提供的资料回答企业知识问题，不得编造。
- 资料未覆盖时明确说明，并建议联系对应业务部门确认。
- 不要声称已经执行任何写操作；写操作必须经过用户确认。
- 回答简洁、直接。

用户：preview_account（未填部门）
已记住的用户事实（不是企业证据，不能替代检索）：无
会话摘要：无
最近对话：
用户: @@USER@@
用户消息：@@USER@@
系统已知结果：
"""
    records.append({"id":"v6-preview-role-agent", "group_id":"v6-preview/role-boundary", "mode":"agent_reply",
        "category":"identity_scope", "question":user,
        "messages":[{"role":"user","content":agent_template.replace("@@USER@@",user)},{"role":"assistant","content":answer}],
        "review_note":"候选Agent回复模板；沿用业务范围，用户标识是演练账号，不是助手姓名。",
        "requires_agent_prompt_sync":True})
    assert len(records)==26
    for row in records:
        row.update(origin="assistant_authored_rule_checked_preview", status="pending_user_review", synthetic=True)
    return records, original_sources, sources, agent_template


def main():
    records, original, candidate, agent_template = build()
    OUTPUT.mkdir(exist_ok=False)
    sources, held_questions, held_prompts, train_questions = {}, set(), set(), set()
    for version in ("v2","v3","v4","v5","grpo_v1","grpo_v2"):
        for split in ("train","val","test"):
            path=DATA/version/(split+".jsonl")
            if not path.exists():
                continue
            sources[path.relative_to(DATA).as_posix()]=sha(path)
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    if split=="train":
                        train_questions.update(questions(row))
                    else:
                        held_questions.update(questions(row))
                        held_prompts.add(canonical(row.get("prompt") or row.get("messages",[])[:-1]))
    collisions=[]
    for row in records:
        if questions(row)&held_questions or canonical(row["messages"][:-1]) in held_prompts:
            collisions.append(row["id"])
    assert not collisions, collisions
    content=("".join(canonical(r)+"\n" for r in records)).encode("utf-8")
    (OUTPUT/"samples.jsonl").write_bytes(content)
    (OUTPUT/"enterprise_system_candidate.txt").write_bytes(candidate["enterprise_system"].encode("utf-8"))
    (OUTPUT/"oa_template_candidate.txt").write_bytes(candidate["oa_template"].encode("utf-8"))
    (OUTPUT/"agent_reply_candidate.txt").write_bytes(agent_template.encode("utf-8"))
    spec={"version":"v6_preview","assistant_name":"TrainPilot","status":"pending_user_review",
          "training_start":"existing_v5_adapter_on_original_full_sft","new_optimizer":True,"resume_old_training_state":False,
          "initial_adapter_sha256":"2c6d865c84c8b1f432250e69ab9d72217263fda4a8811e15ca341abc0d830eb1",
          "grpo50_usage":"independent_comparison_only","samples":26,"modes":dict(Counter(r["mode"] for r in records)),
          "categories":dict(Counter(r["category"] for r in records)),"sources_sha256":sources,
          "samples_sha256":sha(OUTPUT/"samples.jsonl"),"intent_prompt_sha256":sha(DATA/"intent_prompt_v4.txt"),
          "heldout_exact_question_overlap":0,"heldout_exact_prompt_overlap":0,
          "new_questions_repeated_in_old_train":sum(bool(questions(r)&train_questions) for r in records),
          "leakage_limits":"只有精确问句/完整prompt检查，不代表近义语义或事实完全独立；同一反事实组正式划分时必须一起留出。",
          "agent_prompt_sync":"candidate_only_not_applied","model_evaluated":False,"formal_splits_generated":False,
          "proposed_replay":"仅抽V5训练侧，覆盖全部业务意图、两类RAG与身份；排除歧义表格，不从评测集合回填",
          "formal_data_size_and_hyperparameters":"待预览确认与数据审阅后确定"}
    dump(OUTPUT/"spec.json",spec)
    lines=["# V6 修复型SFT · 26条预览", "", "2026-10-30。仅供审阅，尚未生成正式train/val/test，也未训练。助手身份统一为TrainPilot。参数样机、日期和演练账号均为合成信息，不是真实企业事实。", "",
      "## 已确认的训练起点", "", "使用原Full SFT底座加载V5现有LoRA，继续训练这份adapter；重建优化器，不恢复旧数据上的优化器状态。另存V6目录，V5与GRPO50保留作对照。具体学习率、轮次和正式规模待审阅后决定。", "",
      "## 本轮重点", "", "- 可信待办状态优先；有待办确认/取消，无待办则澄清。", "- 咨询引用不等于执行；能力边界询问不等于制度咨询。", "- 放弃撤回操作与真正撤回申请方向相反；完整字段更新继续保留。", "- RAG部分可答先答有据部分；整题无据拒答；表格须唯一定位。", "- 旧任务回放仅来自训练侧，不回填本次验证或回归题。", "",
      "## 需要确认的提示词调整", "", "候选文本只保存在本目录，尚未改Agent。企业RAG、OA和自然回复模板的助手称呼统一为TrainPilot；企业资料里的公司/产品名仍是证据事实。OA原先笼统规定资料不足就拒答，本轮候选明确允许部分回答；若不接受此规则变化，OA部分回答样例不能进入正式训练。", ""]
    for key in ("enterprise_system","oa_template"):
        diff="\n".join(difflib.unified_diff(original[key].splitlines(),candidate[key].splitlines(),fromfile="现有快照",tofile="候选"))
        lines += ["### "+key,"","```diff",diff,"```",""]
    lines += ["## 样例", "", "每条都包含对应完整messages，以下仅展示上下文、用户输入、目标和解释。相同group_id为状态对照，正式划分时不得拆到训练和验证两侧。", ""]
    for row in records:
        lines += ["### "+row["id"]+" · "+row["mode"],"","组：`"+row["group_id"]+"`。"]
        if row.get("scenario"):
            lines += ["","上下文：","```json",json.dumps({k:v for k,v in row["scenario"].items() if k!="user_message"},ensure_ascii=False,indent=2),"```", "", "用户："+row["scenario"]["user_message"]]
        else:
            lines += ["","用户："+row["question"]]
        if row.get("evidence"):
            lines += ["","证据："]+[f"[{i}] "+e["text"] for i,e in enumerate(row["evidence"],1)]
        lines += ["","目标：","```text",row["messages"][-1]["content"],"```","",row["review_note"],""]
    lines += ["## 检查与限制", "", "26条均通过既有意图协议或RAG结构检查；与V2—V5及GRPO既有验证/测试的精确用户问句、完整prompt均无重合。没有据此声称近义语义或事实完全隔离。尚未加载模型测能力，未完成tokenizer长度检查，也未核对候选提示词在Agent运行时的一致性。", ""]
    (OUTPUT/"review.md").write_bytes("\n".join(lines).encode("utf-8"))
    print(json.dumps({"samples":26,"modes":spec["modes"],"heldout_exact_overlap":0,"old_train_question_repeats":spec["new_questions_repeated_in_old_train"],"output":str(OUTPUT)},ensure_ascii=False))


if __name__=="__main__":
    main()
