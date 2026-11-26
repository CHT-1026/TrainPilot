"""按已批准的V4规范构造项目专项数据，不调用外部API、不训练模型。

Agent环境：python -m model.dataset.build_agent_v4 --agent-project PATH
训练环境：python -m model.dataset.build_agent_v4 --tokenizer-dir PATH
V4 test是合成表达族留出集；最终独立业务验收另行封存。
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import sys
from unittest.mock import patch

from model.dataset.build_agent_data import DATA_ROOT, FIELDS, INTENTS, canonical, read_jsonl, render_prompt
from model.dataset.build_agent_v3 import pending_seed, run_trajectory, AGENT_SOURCES
from model.dataset.preview_agent_v4 import ScriptedModel, CANDIDATE_RULES, specification

ROOT = Path(__file__).resolve().parents[2]
PREVIEW = DATA_ROOT / "v4_preview"
OUTPUT = DATA_ROOT / "v4"
SEED = 20261023
TYPES = ("年假", "病假", "事假", "婚假", "产假", "丧假")
REASONS = {
    "年假": ("返乡探亲", "安排家庭出游", "集中休息"),
    "病假": ("门诊复查", "接受康复治疗", "身体不适就医"),
    "事假": ("办理证件", "处理房屋交接", "家中设备检修"),
    "婚假": ("办理结婚手续", "筹办自己的婚礼", "安排婚礼事宜"),
    "产假": ("分娩休养", "产后恢复", "安排生育休假"),
    "丧假": ("处理亲属丧事", "办理亲属身后事", "参加亲属葬礼"),
}
MODIFICATIONS = ("modify_start", "modify_end", "modify_both", "modify_type", "modify_reason", "negated_modification")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def values(family, category, index):
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{SEED}/{family}/{category}/{index}".encode()).digest()[:8], "big"))
    start = date(2027, 1, 4) + timedelta(weeks=index * 3 + rng.randrange(10))
    today = start - timedelta(days=7 + rng.randrange(8))
    leave_type = TYPES[index % len(TYPES)]
    decision = ("通过", "驳回")[index % 2]
    return {"today": today.isoformat(), "start_date": start.isoformat(), "end_date": (start + timedelta(days=1)).isoformat(),
            "new_start": (start + timedelta(days=7)).isoformat(), "new_end": (start + timedelta(days=8)).isoformat(),
            "leave_type": leave_type, "new_type": TYPES[(index + 1) % len(TYPES)],
            "reason": REASONS[leave_type][index % 3], "new_reason": REASONS[leave_type][(index + 1) % 3],
            "pending_id": "leave_" + f"{rng.getrandbits(48):012x}", "app_id": "leave_" + f"{rng.getrandbits(48):012x}",
            "other_id": "leave_" + f"{rng.getrandbits(48):012x}",
            "decision_label": decision, "decision": "approved" if decision == "通过" else "rejected",
            "new_decision": "驳回" if decision == "通过" else "通过",
            "comments": "必要材料已核对" if decision == "通过" else "请补齐必要材料",
            "new_comments": "已核验提交材料" if decision == "通过" else "请说明起止日期",
            "topic": ("年假申请", "病假证明", "请假审批", "报销申请", "考勤补卡", "信息安全")[index % 6],
            "quoted": ("撤回申请", "通过审批", "提交请假")[index % 3]}


def plain_scenario(c, question, state=None):
    state = state or {}
    return {"current_date": c["today"], "user_message": question, "history": state.get("messages", []),
            "pending_action": state.get("pending_action"), "summary": state.get("summary", "")}


def intent_row(uid, category, group, c, question, intent, fields=None, state=None):
    scenario = plain_scenario(c, question, state)
    target = {"intent": intent, "extracted_info": fields or {}}
    return {"id": uid, "category": category, "group_id": group, "mode": "intent_parse", "origin": "v4_new_scenario",
            "scenario": scenario, "expected": target,
            "messages": [{"role": "user", "content": render_prompt(CANDIDATE_RULES, scenario)},
                         {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}]}


def review_replay():
    """全量规则审查及逐条来源记录，不声称自动检查可以取代人工语义审查。"""
    rows, decisions = [], []
    for version in ("v2", "v3"):
        folder = DATA_ROOT / version
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assert sha(folder / "train.jsonl") == manifest["outputs"]["train"]["sha256"]
        for old in read_jsonl(folder / "train.jsonl"):
            if version == "v3" and old["origin"] != "agent_runtime_v3":
                continue  # V3内的V2回放不能再收一遍。
            reason = "retained_after_schema_and_visible_evidence_review"
            if old["category"] == "natural":
                reason = "excluded_unrelated_general_exercise"
            elif version == "v2" and old["group_id"] == "missing_type/template-3":
                reason = "excluded_ambiguous_reason_span"
            row = deepcopy(old)
            if "expected" in row and reason.startswith("retained"):
                validate_target(row)
                validate_evidence(row)
                if row["expected"]["intent"] in {"confirm_action", "cancel_action"}:
                    assert row["scenario"].get("pending_action"), row["id"]
                row["messages"][0]["content"] = render_prompt(CANDIDATE_RULES, row["scenario"])
            include = reason.startswith("retained")
            decisions.append({"version": version, "source_id": old["id"], "category": old["category"],
                              "included": include, "reason": reason})
            if include:
                row.update(id=f"v4-replay-{version}-" + old["id"], origin=f"{version}_train_replay",
                           source_id=old["id"], source_file=f"{version}/train.jsonl", group_id=f"replay/{version}/" + old["group_id"])
                if "expected" not in row:
                    row["mode"] = "direct_reply"
                rows.append(row)
    return rows, decisions


def static_rows(templates, family, index):
    """标签来自明确的字段/状态变换；文案族与参数生成独立。"""
    group = f"v4/family-{family}"
    output = []

    def emit(category, c, intent, fields=None, state=None, text=None, suffix=""):
        question = text if text is not None else templates[category][family].format(**c)
        uid = f"v4-{family}-{category}-{index:02d}" + suffix
        row = intent_row(uid, category, group + "/" + category, c, question, intent, fields, state)
        output.append(row)

    for kind in ("confirm", "cancel"):
        for pending_kind in ("submit_leave", "approve", "cancel_leave", None):
            # 无待办样本也包含已取消、已完成和空历史，避免只认空上下文。
            c = values(family, kind, index)
            if pending_kind:
                state = pending_seed(pending_kind, c)
                target = "confirm_action" if kind == "confirm" else "cancel_action"
            else:
                if index >= 3:
                    continue
                state = {} if index == 0 else {"messages": [{"role": "assistant", "content": (
                    "上一项操作已经取消；目前没有任何待确认动作。" if index == 1 else "上一项业务已有执行回执并结束；目前没有待确认动作。")}]}
                target = "clarify"
            emit(kind, c, target, state=state, suffix="-" + (pending_kind or "none"))
    for kind in ("withdraw_confirm", "withdraw_abandon"):
        c = values(family, kind, index)
        emit(kind, c, "confirm_action" if kind.endswith("confirm") else "cancel_action", state=pending_seed("cancel_leave", c))
    for kind in MODIFICATIONS:
        c = values(family, kind, index)
        if kind == "modify_end":
            c["new_end"] = (date.fromisoformat(c["start_date"]) + timedelta(days=2)).isoformat()
        if kind == "modify_type":
            c.update(leave_type="事假", new_type="年假", reason="处理个人事务")
        original = {k: c[k] for k in ("leave_type", "start_date", "end_date", "reason")}
        changes = {"modify_start": {"start_date": c["new_start"], "end_date": c["end_date"]},
                   "modify_end": {"end_date": c["new_end"]},
                   "modify_both": {"start_date": c["new_start"], "end_date": c["new_end"]},
                   "modify_type": {"leave_type": c["new_type"]}, "modify_reason": {"reason": c["new_reason"]},
                   "negated_modification": {"reason": c["new_reason"]}}[kind]
        if kind == "modify_start":
            # 只改首日仍保持合法区间；不能把坏日期混进正常修改样本。
            c["new_start"] = c["end_date"]
            changes = {"start_date": c["new_start"]}
        context_kind = index % 3
        state = pending_seed("submit_leave", c)
        if context_kind == 1:
            state.pop("pending_action")  # 原简化历史仍是有效输入，不整类删除。
        elif context_kind == 2:
            state = {"summary": "当前尚未提交的请假草稿：" + json.dumps(original, ensure_ascii=False)}
        emit(kind, c, "submit_leave", {**original, **changes}, state)
    for kind in ("approval_edit", "approval_comment"):
        c = values(family, kind, index)
        state = pending_seed("approve", c)
        info = {"app_id": c["pending_id"], "decision": c["decision_label"], "comments": c["comments"]}
        info.update({"decision": c["new_decision"]} if kind == "approval_edit" else {"comments": c["new_comments"]})
        emit(kind, c, "approve", info, state)
    for state_kind in ("approve", "cancel_leave"):
        c = values(family, "other_object-" + state_kind, index)
        emit("other_object", c, "delete_application", {"app_id": c["app_id"]}, pending_seed(state_kind, c), suffix="-" + state_kind)
    c = values(family, "ambiguous_object", index)
    emit("ambiguous_object", c, "clarify", state={"messages": [{"role": "assistant", "content": f"提到过两张已提交申请：{c['pending_id']}、{c['app_id']}，还没指定撤回哪一张。"}]})

    # 请假4字段的16种已知/缺失组合（含全缺与全有）；缺项绝不填null或猜测。
    for mask in range(16):
        c = values(family, "leave_subset-" + str(mask), index)
        keys = ("leave_type", "start_date", "end_date", "reason")
        info = {k: c[k] for j, k in enumerate(keys) if mask & (1 << j)}
        names = {"leave_type": "假别", "start_date": "开始日期", "end_date": "结束日期", "reason": "事由"}
        c["known"] = "；".join(f"{names[k]}“{v}”" for k, v in info.items()) or "尚无任何已确定字段"
        emit("leave_subset", c, "submit_leave", info, suffix=f"-mask-{mask}")
    # 审批ID、结论、可选意见的8种组合。
    for mask in range(8):
        c = values(family, "approval_subset-" + str(mask), index)
        available = {"app_id": c["app_id"], "decision": c["decision_label"], "comments": c["comments"]}
        info = {k: v for j, (k, v) in enumerate(available.items()) if mask & (1 << j)}
        names = {"app_id": "申请编号", "decision": "审批结论", "comments": "意见正文"}
        c["known"] = "；".join(f"{names[k]}“{v}”" for k, v in info.items()) or "尚无已确定的编号、决定或意见"
        emit("approval_subset", c, "approve", info, suffix=f"-mask-{mask}")
    for kind, target in (("advice", "get_suggestion"), ("advice_missing", "get_suggestion"), ("delete_missing", "delete_application"),
                         ("policy", "policy_qa"), ("quoted_policy", "policy_qa"), ("multi_task", "clarify"), ("uncertain_dates", "submit_leave")):
        if kind in {"advice_missing", "delete_missing"} and index > 0:
            continue
        c = values(family, kind, index)
        info = {"app_id": c["app_id"]} if kind == "advice" else (
            {"leave_type": c["leave_type"], "reason": c["reason"]} if kind == "uncertain_dates" else {})
        emit(kind, c, target, info)
    for kind, target in (("query_status", "query_status"), ("query_pending", "query_pending"), ("ambiguous_intent", "clarify")):
        if index == 0:
            emit(kind, values(family, kind, index), target)
    c = values(family, "relative", index)
    c["today"] = ("2026-12-31", "2027-01-31", "2027-02-27", "2028-02-28", "2027-11-30", "2027-06-29")[index % 6]
    offset = (index + family) % 3
    c["relative"] = ("今天", "明天", "后天")[offset]
    relative_date = (date.fromisoformat(c["today"]) + timedelta(days=offset)).isoformat()
    emit("relative", c, "submit_leave", {"leave_type": c["leave_type"], "start_date": relative_date, "end_date": relative_date, "reason": c["reason"]})
    c = values(family, "invalid_range", index)
    emit("invalid_range", c, "submit_leave", {"leave_type": c["leave_type"], "start_date": c["end_date"], "end_date": c["start_date"], "reason": c["reason"]})
    return output


def trajectory(family, index, kind, templates, split):
    c = values(family, kind, index)
    leave = {key: c[key] for key in ("leave_type", "start_date", "end_date", "reason")}
    definition = {"id": f"v4-{split}-{family}-{kind}-{index}", "group_id": f"v4/family-{family}/" + kind,
                  "split": split, "kind": kind, "current_date": c["today"], "role": "employee", "turns": []}

    def add(key, intent, fields, node, pending_type):
        definition["turns"].append({"user": templates[key][family].format(**c),
                                    "expected": {"intent": intent, "extracted_info": fields},
                                    "expected_nodes": [node], "expected_pending_type": pending_type,
                                    **({"expected_draft": fields} if intent == "submit_leave" and pending_type else {})})
    if kind == "collect_modify_confirm":
        add("collect_start", "submit_leave", {"leave_type": c["leave_type"]}, "submit_leave", None)
        add("collect_fill", "submit_leave", leave, "submit_leave", "submit_leave")
        add("modify_both", "submit_leave", {**leave, "start_date": c["new_start"], "end_date": c["new_end"]}, "submit_leave", "submit_leave")
        add("confirm", "confirm_action", {}, "execute_pending", "submit_leave")
    elif kind == "cancel_then_new":
        definition["initial_state"] = pending_seed("submit_leave", c)
        add("cancel", "cancel_action", {}, "clear_pending", None)
        add("reopen", "submit_leave", {"leave_type": c["new_type"]}, "submit_leave", None)
        c.update(leave_type=c["new_type"], reason=REASONS[c["new_type"]][index % 3])
        add("collect_fill", "submit_leave", {key: c[key] for key in leave}, "submit_leave", "submit_leave")
    else:
        c["new_end"] = (date.fromisoformat(c["start_date"]) + timedelta(days=2)).isoformat()
        definition["initial_state"] = pending_seed("submit_leave", c)
        add("policy", "policy_qa", {}, "retrieve", "submit_leave")
        add("resume", "submit_leave", {**leave, "end_date": c["new_end"]}, "submit_leave", "submit_leave")
        add("confirm", "confirm_action", {}, "execute_pending", "submit_leave")
        definition["scope_note"] = "制度查询后保留待办只在终端占位节点检查；生产知识库和状态恢复需单独验收。"
    return definition


async def reply_rows(topic, family, split, graph_factory, run_turn, user_class):
    teacher = ScriptedModel()
    teacher.expected = {"intent": "general_chat", "extracted_info": {}}
    teacher.reply = topic["answer"]
    graph = graph_factory(teacher)
    question = topic["questions"][family]
    uid = f"v4-identity-{topic['id']}-{family}"
    user = user_class(user_id="synthetic_account", name="", role="employee")
    class Clock:
        @staticmethod
        def now():
            return datetime(2026, 11, 2)
    with patch("app.workflows.chat_nodes.recognition.datetime", Clock):
        state = await run_turn(graph, uid, question, user, lambda _: None)
    assert state["intent"] == "general_chat" and state["current_task"] == "general_chat", uid
    assert state["response"] == topic["answer"]
    assert [mode for mode, _ in teacher.calls] == ["intent_parse", "agent_reply"]
    rows = []
    for mode, prompt in teacher.calls:
        answer = json.dumps(teacher.expected, ensure_ascii=False, separators=(",", ":")) if mode == "intent_parse" else topic["answer"]
        row = {"id": uid + "/" + mode, "group_id": f"v4/family-{family}/identity-{topic['id']}", "category": "identity_" + topic["id"],
               "origin": "v4_agent_identity", "mode": mode, "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}
        if mode == "intent_parse":
            context = json.loads(prompt.split("\n给定上下文：\n", 1)[1].split("\n用户输入：\n", 1)[0])
            row.update(expected=teacher.expected, scenario={**context, "user_message": question})
        rows.append(row)
    for system in (False, True):
        messages = ([{"role": "system", "content": "你是 TrainPilot，一名企业业务助手。"}] if system else []) + [
            {"role": "user", "content": question}, {"role": "assistant", "content": topic["answer"]}]
        rows.append({"id": uid + ("/direct_system" if system else "/direct_plain"), "group_id": f"v4/family-{family}/identity-{topic['id']}",
                     "category": "identity_" + topic["id"], "origin": "v4_agent_identity", "mode": "direct_reply", "messages": messages})
    return rows


def validate_target(row):
    expected = row["expected"]
    assert json.loads(row["messages"][-1]["content"]) == expected, row["id"]
    assert set(expected) == {"intent", "extracted_info"} and expected["intent"] in INTENTS, row["id"]
    info = expected["extracted_info"]
    assert isinstance(info, dict) and set(info) <= FIELDS, row["id"]
    assert all(isinstance(v, str) and v for v in info.values()), row["id"]
    assert info.get("decision", "通过") in {"通过", "驳回"}, row["id"]
    for key in ("start_date", "end_date"):
        if key in info:
            date.fromisoformat(info[key])


def validate_evidence(row):
    info, scene = row["expected"]["extracted_info"], row["scenario"]
    visible = canonical(scene)
    for key in ("leave_type", "reason", "app_id", "comments"):
        assert key not in info or info[key] in visible, (row["id"], key)
    for key in ("start_date", "end_date"):
        if key in info and info[key] not in visible:
            match = next((name for name in ("后天", "明天", "今天") if name in scene["user_message"]), None)
            assert match, (row["id"], key, "缺少日期证据")
            expected = date.fromisoformat(scene["current_date"]) + timedelta(days={"今天": 0, "明天": 1, "后天": 2}[match])
            assert info[key] == expected.isoformat(), (row["id"], key, "相对日期错误")
    if row["expected"]["intent"] in {"confirm_action", "cancel_action"}:
        assert scene.get("pending_action") and info == {}, row["id"]


def input_key(row):
    if "scenario" in row and "expected" in row:
        return render_prompt("", row["scenario"])
    return canonical(row["messages"][:-1])


def deduplicate(splits):
    seen, removed = {}, []
    for split, rows in splits.items():
        kept = []
        for row in rows:
            key = canonical(row["messages"][:-1])
            if key in seen:
                original = seen[key]
                assert row["messages"][-1]["content"] == original["messages"][-1]["content"], ("同输入冲突标签", row["id"], original["id"])
                removed.append({"id": row["id"], "split": split, "same_input_as": original["id"], "reason": "identical_input_and_target"})
            else:
                seen[key] = row
                kept.append(row)
        splits[split] = kept
    return removed


def audit(splits, runtime_rows, agent_prompt_builder):
    seen, groups, trajectories, ids = {}, {}, {}, set()
    protected = {input_key(row) for version in ("v2", "v3") for split in ("val", "test")
                 for row in read_jsonl(DATA_ROOT / version / f"{split}.jsonl")}
    for split, rows in splits.items():
        for row in rows:
            assert row["id"] not in ids
            ids.add(row["id"])
            key = canonical(row["messages"][:-1])
            assert key not in seen, ("重复输入或冲突标签", row["id"], seen.get(key))
            seen[key] = row["id"]
            assert input_key(row) not in protected, ("旧留出输入泄漏", row["id"])
            for collection, name in ((groups, row["group_id"]), (trajectories, row.get("trajectory_id"))):
                if name:
                    assert collection.get(name, split) == split, ("分组跨划分", name)
                    collection[name] = split
            if "expected" not in row:
                continue
            validate_target(row)
            validate_evidence(row)
            scene = row["scenario"]
            state = {"user_message": scene["user_message"], "messages": scene.get("history", []),
                     "pending_action": scene.get("pending_action"), "summary": scene.get("summary", "")}
            expected_prompt = agent_prompt_builder(state, now=datetime.fromisoformat(scene["current_date"]))
            assert row["messages"][0]["content"] == expected_prompt, ("Agent实际提示词不一致", row["id"])
    return {"unique_inputs": True, "source_heldout_excluded_after_prompt_normalization": True,
            "group_and_trajectory_split_disjoint": True, "all_intent_rows_runtime_prompt_parity": True,
            "all_structured_targets_schema_and_visible_evidence_checked": True,
            "scripted_runtime_requests": runtime_rows, "external_model_calls": 0, "database_writes": 0,
            "note": "可见字符串检查不等于全面语义审阅；合成模板留出不等于真实用户泛化。"}


async def build(agent_project, output):
    assert not output.exists(), "正式V4已存在，不能覆盖已冻结数据"
    assert json.loads((PREVIEW / "approval.json").read_text(encoding="utf-8"))["status"] == "user_approved"
    definitions = json.loads((DATA_ROOT / "scenarios_v4.json").read_text(encoding="utf-8"))
    assert all(len(v) == len(definitions["family_splits"]) for v in definitions["templates"].values())
    sys.path.insert(0, str(agent_project))
    from app.domain.user.models import UserContext
    from app.workflows import intent_protocol
    from local_runtime.chat_cli import create_terminal_graph, run_turn
    logging.disable(logging.INFO)
    spec = specification()
    spec.update(version="v4", status="user_approved_generation", approval_record="v4_preview/approval.json",
                seed=SEED, static_variants={"train": 12, "val": 6, "test": 6}, trajectory_variants={"train": 6, "val": 4, "test": 4},
                test_scope="synthetic_template_family_holdout_development; independent_acceptance_pending")
    replay, decisions = review_replay()
    splits, trajectories = {"train": list(replay), "val": [], "test": []}, []
    runtime_count = 0
    # 所有结构化输入与原prompt builder逐条核对；多轮及身份请求真正经过脚本化原图。
    with patch.object(intent_protocol, "RULES", CANDIDATE_RULES):
        for family, split in enumerate(definitions["family_splits"]):
            count = spec["static_variants"][split]
            for index in range(count):
                splits[split].extend(static_rows(definitions["templates"], family, index))
            for kind in ("collect_modify_confirm", "cancel_then_new", "side_query_resume"):
                for index in range(spec["trajectory_variants"][split]):
                    definition = trajectory(family, index, kind, definitions["templates"], split)
                    rows, evidence = await run_trajectory(definition, create_terminal_graph, run_turn, UserContext)
                    for row, step in zip(rows, evidence["turns"]):
                        row["id"] = row["id"].removeprefix("v3-")
                        step["sample_id"] = row["id"]
                        row["origin"] = "v4_agent_runtime"
                    splits[split].extend(rows)
                    trajectories.append(evidence)
                    runtime_count += len(rows)
            for topic in definitions["reply_topics"]:
                splits[split].extend(await reply_rows(topic, family, split, create_terminal_graph, run_turn, UserContext))
                runtime_count += 2
            print(f"已构造family-{family} ({split})", flush=True)
        duplicates = deduplicate(splits)
        checks = audit(splits, runtime_count, intent_protocol.build_intent_prompt)
    # V3运行轨迹来源数据可追溯，但不冒充V4重新执行过的轨迹。
    for i, rows in enumerate(splits.values()):
        random.Random(SEED + i).shuffle(rows)
    output.mkdir(parents=True)
    outputs = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
        outputs[split] = {"file": path.name, "records": len(rows), "sha256": sha(path),
                          "origins": dict(Counter(r["origin"] for r in rows)), "modes": dict(Counter(r["mode"] for r in rows)),
                          "categories": dict(Counter(r["category"] for r in rows)),
                          "intents": dict(Counter(r["expected"]["intent"] for r in rows if "expected" in r)),
                          "groups": len({r["group_id"] for r in rows}),
                          "trajectories": len({r["trajectory_id"] for r in rows if r.get("trajectory_id")})}
    save(DATA_ROOT / "spec_v4.json", spec)
    (DATA_ROOT / "intent_prompt_v4.txt").write_bytes(CANDIDATE_RULES.encode("utf-8"))
    save(output / "replay_selection.json", {"audit": "schema, source hash, visible field evidence, category rules; not manual review of every paraphrase",
                                           "counts": dict(Counter(item["reason"] for item in decisions)), "decisions": decisions})
    save(output / "duplicates_removed.json", duplicates)
    (output / "trajectories.jsonl").write_bytes("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in trajectories).encode("utf-8"))
    inputs = ("spec_v4.json", "scenarios_v4.json", "intent_prompt_v4.txt", "v2/train.jsonl", "v2/val.jsonl", "v2/test.jsonl",
              "v3/train.jsonl", "v3/val.jsonl", "v3/test.jsonl", "v4_preview/approval.json", "v4_preview/samples.json")
    manifest = {"version": "v4", "status": "generated_pending_encoding_check", "assistant_name": "TrainPilot",
                "generation_date": "2026-10-23", "seed": SEED, "outputs": outputs, "checks": checks,
                "inputs_sha256": {name: sha(DATA_ROOT / name) for name in inputs}, "generator_sha256": sha(Path(__file__)),
                "agent_sources_sha256": {name: sha(agent_project / name) for name in AGENT_SOURCES},
                "test_scope": spec["test_scope"], "training_start": "original_full_sft_with_fresh_lora",
                "identical_input_duplicates_removed": len(duplicates),
                "replay_restored_categories": {key: sum(r["source_file"] == "v2/train.jsonl" and r["category"] == key for r in replay)
                                              for key in ("history_fill", "history_modify", "confirm_action", "cancel_action", "pending_other_task", "pending_cancel_existing")},
                "trajectory_evidence": {"file": "trajectories.jsonl", "sha256": sha(output / "trajectories.jsonl"), "count": len(trajectories)},
                "limits": spec["integration_gaps"] + [spec["not_guaranteed"]]}
    save(output / "manifest.json", manifest)
    data_report(output, manifest)
    print(json.dumps({"records": {k: v["records"] for k, v in outputs.items()}, "checks": checks}, ensure_ascii=False, indent=2))


def data_report(output, manifest, encoding=None):
    lines = ["# TrainPilot V4 数据构建记录", "", "按用户批准的覆盖规范构造；没有运行模型训练。", "",
             "模型身份为TrainPilot；完整已知字段；多任务先澄清；字段清空不训练伪执行。", "",
             "## 数据规模", "", "|划分|记录数|来源|模式|", "|---|---:|---|---|"]
    for split, info in manifest["outputs"].items():
        lines.append(f"|{split}|{info['records']}|{json.dumps(info['origins'], ensure_ascii=False)}|{json.dumps(info['modes'], ensure_ascii=False)}|")
    lines += ["", "旧回放只来自旧train；V3内部已经包含的V2回放不再收录。恢复六类有效旧场景的候选共336条，每条保留来源ID。排除理由逐条见replay_selection.json。", "",
              "检查范围：来源SHA256、结构化字段与可见依据、原Agent提示词逐条一致性、输入重复/冲突、分组与轨迹划分、已知旧留出输入排除。",
              f"新增脚本化原Agent请求数：{manifest['checks']['scripted_runtime_requests']}；多轮实际上下文和状态证据见trajectories.jsonl。", "",
              "标签是规则构造的期望答案；脚本化Agent不访问外部模型或数据库。程序通过不能证明模型效果或数据库业务成功。", "",
              "## 训练类别覆盖", "", "|类别|训练记录数|", "|---|---:|"]
    for category, n in sorted(manifest["outputs"]["train"]["categories"].items()):
        lines.append(f"|{category}|{n}|")
    lines += ["", "## 数据与评测边界", "", "- val/test按新表达族和整条轨迹划分，是合成开发留出集。旧V2/V3测试作为回归。最终独立验收题尚未制作，不能用本次生成器的成功率充当模型成绩。",
              "- 来源语义经模板检查及逐条规则检查；可见字符串证据不证明所有语义正确。仍需人工抽查具体表达和标签。",
              "- 只有用户需求理解与结构化动作选择，以及实际普通回复模式；没有训练可执行的原生function calling或隐藏思维链。",
              "- 字段清空、直接切换未取消任务、闲聊/澄清后保留待办，以及真正的数据库操作存在Agent接口或集成待办，详见spec_v4.json。", ""]
    if encoding:
        lines += ["## 编码检查", "", f"最大长度{encoding['max_tokens']} tokens；上限{encoding['max_length']}；截断{encoding['truncated']}条；无有效答案{encoding['zero_targets']}条。", "",
                  "只监督assistant模板正文及im_end结束符，用户输入和padding不监督。Qwen模板的空think前缀不含人工思维链。", "",
                  "监督token分布见encoding_check.json；样本条数与独立语义数量不同。", ""]
    lines += ["## 下一步", "", "1. 按类别抽查数据与标签，重点检查确认/放弃和编号对象。",
              "2. 提示词单独做V3对照；Agent部署时须使用intent_prompt_v4.txt同版，当前尚未修改原Agent。",
              "3. 从原Full SFT重新初始化LoRA，保留V2/V3，保存各epoch导出并根据旧/新验证联合选取。",
              "4. 使用独立验收题及真实Agent多轮测试确认能力，最后再接数据库业务验证执行结果。", ""]
    (output / "data_report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def check_encoding(tokenizer_dir, output):
    from transformers import AutoTokenizer
    from model.dataset.dataset import QwenSFTEncoder
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoder = QwenSFTEncoder(tokenizer, 1536)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    by_split, maximum = {}, 0
    for split in ("train", "val", "test"):
        assert sha(output / f"{split}.jsonl") == manifest["outputs"][split]["sha256"]
        rows = read_jsonl(output / f"{split}.jsonl")
        tokens, modes, categories = 0, Counter(), Counter()
        for row in rows:
            full = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False)
            encoded = encoder(row)
            assert full == encoded["input_ids"], ("发生截断", row["id"], len(full))
            supervised = [label for label in encoded["labels"] if label != -100]
            answer = row["messages"][-1]["content"] + "<|im_end|>"
            decoded = tokenizer.decode(supervised)
            assert supervised[-1] == im_end and decoded.endswith(answer), row["id"]
            assert decoded[:-len(answer)] in {"", "<think>\n\n</think>\n\n"}, row["id"]
            count = sum(v != -100 for v in encoded["labels"][1:])
            assert count
            maximum = max(maximum, len(full))
            tokens += count
            modes[row["mode"]] += count
            categories[row["category"]] += count
        by_split[split] = {"records": len(rows), "supervised_tokens": tokens,
                           "tokens_by_mode": dict(modes), "tokens_by_category": dict(categories)}
        print(f"编码检查完成：{split}，{len(rows)}条", flush=True)
    result = {"max_length": 1536, "max_tokens": maximum, "truncated": 0, "zero_targets": 0, "splits": by_split,
              "assistant_end_id": im_end, "inputs_sha256": {s: manifest["outputs"][s]["sha256"] for s in by_split},
              "scope": "完整聊天模板、assistant监督正文及结束符，无模型效果结论。"}
    save(output / "encoding_check.json", result)
    manifest["status"] = "encoding_checked_pending_training"
    manifest["encoding_check"] = {"file": "encoding_check.json", "sha256": sha(output / "encoding_check.json")}
    save(output / "manifest.json", manifest)
    data_report(output, manifest, result)
    print(json.dumps({"max_tokens": maximum, "truncated": 0, "zero_targets": 0, "tokens": {s: v["supervised_tokens"] for s, v in by_split.items()}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-project", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.tokenizer_dir:
        check_encoding(args.tokenizer_dir, args.output_dir)
    else:
        asyncio.run(build(args.agent_project.resolve(), args.output_dir))


if __name__ == "__main__":
    main()
