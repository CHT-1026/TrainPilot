"""构造V3：筛选V2训练样本，使用原Agent推进已批准的多轮场景。

生成需要Agent项目的Python环境；编码检查使用训练项目环境，不加载模型权重。
--agent-project PATH 生成；--tokenizer-dir PATH 检查已有数据的完整监督与长度。
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
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.dataset.build_agent_data import DATA_ROOT, FIELDS, INTENTS, canonical, read_jsonl, render_prompt

DEFAULT_OUTPUT = DATA_ROOT / "v3"
KINDS = ("collect_modify_confirm", "cancel_then_new", "pending_decision", "other_object", "side_query_resume")
AGENT_SOURCES = (
    "app/workflows/intent_protocol.py", "app/workflows/intent_protocol.txt",
    "app/workflows/chat_agent.py", "app/workflows/chat_state.py",
    "app/workflows/chat_nodes/recognition.py", "app/workflows/chat_nodes/routing.py",
    "app/workflows/chat_nodes/leave.py", "app/workflows/chat_nodes/response.py",
    "app/workflows/chat_nodes/execute.py", "app/workflows/chat_nodes/summarize.py",
    "app/routes/chat.py", "app/capabilities/policy_engine.py",
    "app/domain/leave/presenter.py", "local_runtime/chat_cli.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def select_replay(spec):
    source = DATA_ROOT / spec["v2_replay"]["source"]
    original = read_jsonl(source)
    expected_hash = json.loads((DATA_ROOT / "v2/manifest.json").read_text(encoding="utf-8"))["outputs"]["train"]["sha256"]
    assert sha256(source) == expected_hash, "V2训练文件与保存的基线不一致"
    excluded_categories = set(spec["v2_replay"]["excluded_categories"])
    excluded_groups = set(spec["v2_replay"]["excluded_template_groups"])
    kept, decisions = [], []
    for row in original:
        reason = "replaced_by_runtime_trajectories" if row["category"] in excluded_categories else (
            "ambiguous_reason_boundary" if row["group_id"] in excluded_groups else "retained_business_training")
        include = reason == "retained_business_training"
        decisions.append({"source_id": row["id"], "source_group": row["group_id"],
                          "category": row["category"], "included": include, "reason": reason})
        if include:
            record = deepcopy(row)
            record.update(id="v3-replay-" + row["id"], group_id="v2/" + row["group_id"],
                          origin="v2_train_replay", source_id=row["id"])
            kept.append(record)
    return kept, {"source": "v2/train.jsonl", "source_sha256": expected_hash,
                  "counts": dict(Counter(item["reason"] for item in decisions)), "decisions": decisions}


def parameters(seed, family_id, kind, index):
    digest = hashlib.sha256(f"{seed}/{family_id}/{kind}/{index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    # 首轮只有假别与current_date；为不同参数变体分配不相交的日期窗口。
    today = date(2027, 1, 1) + timedelta(days=index * 19 + rng.randrange(13))
    if index == 0:
        today = date(2026, 12, 28)  # 明确包含跨年上下文，日期字段仍由用户给出。
    start = today + timedelta(days=7 + rng.randrange(21))
    start += timedelta(days=(-start.weekday()) % 7)
    types = ("事假", "年假", "病假")
    leave_type = rng.choice(types)
    reasons = {
        "事假": ("家中设备维修", "办理居住证件", "处理搬家事项", "照顾家人", "办理房屋交接", "处理家庭事务"),
        "年假": ("安排休息", "返乡探亲", "陪伴家人", "安排个人旅行", "家庭出游", "集中休整"),
        "病假": ("门诊就诊", "到医院复查", "身体不适", "接受康复治疗", "进行门诊检查", "遵医嘱休养"),
    }
    decision = rng.choice(("通过", "驳回"))
    return {"today": today.isoformat(), "leave_type": leave_type,
            "new_type": rng.choice([value for value in types if value != leave_type]),
            "start_date": start.isoformat(), "end_date": (start + timedelta(days=1)).isoformat(),
            "new_end": (start + timedelta(days=2)).isoformat(), "reason": rng.choice(reasons[leave_type]),
            "pending_id": "leave_" + f"{rng.getrandbits(48):012x}",
            "other_id": "leave_" + f"{rng.getrandbits(48):012x}",
            "decision_label": decision, "decision": "approved" if decision == "通过" else "rejected",
            "comments": "必要材料已核对" if decision == "通过" else "请补充必要材料",
            "policy_topic": rng.choice(("年假申请", "请假审批", "病假证明", "调休申请", "差旅报销", "费用报销"))}


def pending_seed(kind, values):
    """仅设定模拟前置状态；不把未连接的数据库当作已查询。"""
    c = values
    if kind == "submit_leave":
        fields = {key: c[key] for key in ("leave_type", "start_date", "end_date", "reason")}
        draft = {**fields, "working_days": 2, "requires_hr": False}
        question = f"先拟一份{c['leave_type']}：{c['start_date']}至{c['end_date']}，原因“{c['reason']}”，暂不提交。"
        reply = f"请核对待确认草稿：{c['leave_type']}，{c['start_date']}至{c['end_date']}，原因“{c['reason']}”。回复确认办理，回复取消放弃。"
    elif kind == "approve":
        fields = {}
        draft = {"app_id": c["pending_id"], "decision": c["decision"],
                 "decision_label": c["decision_label"], "comments": c["comments"]}
        question = f"申请{c['pending_id']}的审批决定拟为{c['decision_label']}，意见“{c['comments']}”，先让我核对。"
        reply = f"待确认审批：{c['pending_id']}，决定{c['decision_label']}，意见“{c['comments']}”。请确认执行或放弃本次审批。"
    else:
        fields = {}
        draft = {"app_id": c["pending_id"]}
        question = f"我要撤回自己已提交的申请{c['pending_id']}，先整理撤回草稿。"
        reply = f"待确认撤回申请：{c['pending_id']}。回复确认执行撤回，回复取消放弃这项撤回操作。"
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": reply}],
            "pending_action": {"action_id": "act_synthetic_seed", "action_type": kind, "draft": draft},
            "conversation_context": {"accumulated_extracted_info": fields}}


def make_trajectory(family, kind, values, index):
    c = dict(values)
    leave = {key: c[key] for key in ("leave_type", "start_date", "end_date", "reason")}
    updated = {**leave, "end_date": c["new_end"]}
    definition = {"id": f"{family['split']}-{family['id']}-{kind}-{index:03d}",
                  "group_id": "v3/" + family["id"], "split": family["split"], "kind": kind,
                  "current_date": c["today"], "role": "employee", "turns": []}

    def turn(text, intent, fields, nodes, pending_type, draft=None, accumulated=None):
        item = {"user": text.format(**c), "expected": {"intent": intent, "extracted_info": fields},
                "expected_nodes": nodes, "expected_pending_type": pending_type}
        if draft is not None:
            item["expected_draft"] = draft
        if accumulated is not None:
            item["expected_accumulated"] = accumulated
        definition["turns"].append(item)

    if kind == "collect_modify_confirm":
        texts = family[kind]
        turn(texts[0], "submit_leave", {"leave_type": c["leave_type"]}, ["submit_leave"], None,
             accumulated={"leave_type": c["leave_type"]})
        turn(texts[1], "submit_leave", leave, ["submit_leave"], "submit_leave", draft=leave)
        turn(texts[2], "submit_leave", updated, ["submit_leave"], "submit_leave", draft=updated)
        turn(texts[3], "confirm_action", {}, ["execute_pending"], "submit_leave", accumulated={})
    elif kind == "cancel_then_new":
        texts = family[kind]
        turn(texts[0], "submit_leave", leave, ["submit_leave"], "submit_leave", draft=leave)
        turn(texts[1], "cancel_action", {}, ["clear_pending"], None, accumulated={})
        turn(texts[2], "submit_leave", {"leave_type": c["new_type"]}, ["submit_leave"], None,
             accumulated={"leave_type": c["new_type"]})
    elif kind in {"pending_decision", "other_object"}:
        pending_type = ("approve", "cancel_leave")[(index // 2) % 2]
        if kind == "other_object":
            # 未提交的请假草稿没有正式申请编号；编号干扰项仅来自已有审批或撤回待办。
            pending_type = ("approve", "cancel_leave")[index % 2]
        c["action_label"] = {"approve": "审批", "cancel_leave": "撤回", "submit_leave": "请假提交"}[pending_type]
        definition["role"] = "manager" if pending_type == "approve" else "employee"
        definition["initial_state"] = pending_seed(pending_type, c)
        definition["seed_note"] = "人工模拟前置消息和待办；未查询数据库，不代表对象存在、权限通过或执行成功。"
        if kind == "pending_decision":
            confirm = index % 2 == 0
            turn(family["pending_confirm" if confirm else "pending_cancel"],
                 "confirm_action" if confirm else "cancel_action", {},
                 ["execute_pending" if confirm else "clear_pending"], pending_type if confirm else None,
                 accumulated={})
        else:
            turn(family[kind], "delete_application", {"app_id": c["other_id"]}, ["delete"], pending_type)
    else:
        definition["initial_state"] = pending_seed("submit_leave", c)
        definition["seed_note"] = "人工模拟已有请假草稿；数据库业务不可用，临时制度查询不生成检索结果。"
        texts = family[kind]
        turn(texts[0], "policy_qa", {}, ["retrieve", "answer"], "submit_leave", accumulated={})
        turn(texts[1], "submit_leave", updated, ["submit_leave"], "submit_leave", draft=updated)
        turn(texts[2], "confirm_action", {}, ["execute_pending"], "submit_leave", accumulated={})
    return definition


class ScriptedIntent:
    def __init__(self):
        self.expected = None
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=json.dumps(self.expected, ensure_ascii=False, separators=(",", ":")))


async def run_trajectory(definition, graph_factory, run_turn, user_class):
    from app.capabilities.policy_engine import validate_leave_draft
    teacher = ScriptedIntent()
    graph = graph_factory(teacher)
    thread = definition["id"]
    if definition.get("initial_state"):
        await graph.aupdate_state({"configurable": {"thread_id": thread}}, deepcopy(definition["initial_state"]))
    user = user_class(user_id="synthetic_account", name="", role=definition["role"])
    fixed_now = datetime.fromisoformat(definition["current_date"])

    class FixedClock:
        @staticmethod
        def now():
            return fixed_now

    rows, evidence = [], deepcopy(definition)
    for index, item in enumerate(definition["turns"], 1):
        teacher.expected = item["expected"]
        calls = len(teacher.prompts)
        trace = []
        with patch("app.workflows.chat_nodes.recognition.datetime", FixedClock), patch(
            "app.workflows.chat_nodes.leave.validate_leave_draft",
            side_effect=lambda raw: validate_leave_draft(raw, today=fixed_now.date()),
        ):
            state = await run_turn(graph, thread, item["user"], user, trace.append)
        assert len(teacher.prompts) == calls + 1, "只允许脚本化识别；不得调用摘要或其他模型节点"
        prompt = teacher.prompts[-1]
        context = json.loads(prompt.split("\n给定上下文：\n", 1)[1].split("\n用户输入：\n", 1)[0])
        pending = state.get("pending_action")
        draft = (pending or {}).get("draft", {})
        checks = {
            "intent": state["intent"] == item["expected"]["intent"],
            "nodes": all(f"[节点] {node}" in trace for node in item["expected_nodes"]),
            "pending_type": (pending or {}).get("action_type") == item["expected_pending_type"],
            "fields": all(draft.get(key) == value for key, value in item.get("expected_draft", {}).items()),
        }
        accumulated = state.get("conversation_context", {}).get("accumulated_extracted_info", {})
        if "expected_accumulated" in item:
            checks["accumulated"] = accumulated == item["expected_accumulated"]
        if "execute_pending" in item["expected_nodes"] or "delete" in item["expected_nodes"] or "answer" in item["expected_nodes"]:
            checks["database_unavailable"] = "没有执行数据库操作" in state["response"]
        assert all(checks.values()), (definition["id"], index, checks)
        row_id = f"v3-{thread}-turn-{index}"
        rows.append({"id": row_id, "category": definition["kind"], "mode": "intent_parse",
                     "origin": "agent_runtime_v3", "group_id": definition["group_id"],
                     "trajectory_id": thread, "turn": index,
                     "scenario": {**context, "user_message": item["user"]}, "expected": item["expected"],
                     "messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": json.dumps(item["expected"], ensure_ascii=False, separators=(",", ":"))}]})
        evidence["turns"][index - 1].update(
            sample_id=row_id, observed_response=state["response"], runtime_checks=checks,
            observed_pending={"action_type": pending["action_type"], "draft": draft} if pending else None,
            observed_accumulated=accumulated,
            observed_nodes=[line.removeprefix("[节点] ") for line in trace if line.startswith("[节点] ")],
        )
    return rows, evidence


def protected_questions():
    questions = {row["scenario"]["user_message"] for split in ("val", "test")
                 for row in read_jsonl(DATA_ROOT / "v2" / f"{split}.jsonl")}
    baseline = ROOT / "runs/agent-baseline-20261022/agent_cases.json"
    if baseline.exists():
        questions.update(row["message"] for row in json.loads(baseline.read_text(encoding="utf-8")))
    questions.update(("我想请事假", "确认提交", "算了，取消草稿"))
    return questions


def audit(splits, trajectories):
    seen_prompts, seen_groups, seen_trajectories = {}, {}, {}
    protected = protected_questions()
    prompt_template = (DATA_ROOT / "intent_prompt_v2.txt").read_text(encoding="utf-8")
    preview_users = {row["scenario"]["user_message"] for row in json.loads((DATA_ROOT / "v3_preview/samples.json").read_text(encoding="utf-8"))["samples"]}
    for split, rows in splits.items():
        for row in rows:
            prompt_key = canonical(row["messages"][:-1])
            assert prompt_key not in seen_prompts, ("重复输入或冲突标签", row["id"])
            seen_prompts[prompt_key] = split
            group = row["group_id"]
            assert seen_groups.get(group, split) == split, ("模板族跨划分", group)
            seen_groups[group] = split
            trajectory = row.get("trajectory_id")
            if trajectory:
                assert seen_trajectories.get(trajectory, split) == split, trajectory
                seen_trajectories[trajectory] = split
            scenario = row["scenario"]
            assert scenario["user_message"] not in protected, ("旧评测问题", row["id"])
            if split != "train":
                assert scenario["user_message"] not in preview_users, ("预览进入独立评测", row["id"])
            if "expected" not in row:
                assert row["origin"] == "v2_train_replay" and row["category"] == "identity"
                continue
            expected = row["expected"]
            assert set(expected) == {"intent", "extracted_info"} and expected["intent"] in INTENTS
            assert json.loads(row["messages"][-1]["content"]) == expected
            fields = expected["extracted_info"]
            assert isinstance(fields, dict) and set(fields) <= FIELDS
            assert all(isinstance(value, str) and value for value in fields.values())
            assert row["messages"][0]["content"] == render_prompt(prompt_template, scenario), row["id"]
            evidence = canonical(scenario)
            for field in ("leave_type", "reason", "app_id", "comments"):
                assert field not in fields or fields[field] in evidence, (row["id"], field)
            for field in ("start_date", "end_date"):
                if field in fields:
                    date.fromisoformat(fields[field])
                    if row["category"] == "leave_relative":
                        offset = 1 if "明天" in scenario["user_message"] else 2
                        assert fields[field] == (date.fromisoformat(scenario["current_date"]) + timedelta(days=offset)).isoformat()
                    else:
                        assert fields[field] in evidence, (row["id"], field)
            if expected["intent"] in {"confirm_action", "cancel_action"}:
                assert scenario["pending_action"] and fields == {}
            if row["category"] == "other_object":
                assert fields["app_id"] in scenario["user_message"]
                pending_id = scenario["pending_action"]["draft"].get("app_id")
                assert fields["app_id"] != pending_id
    row_ids = {row["id"] for rows in splits.values() for row in rows if row["origin"] == "agent_runtime_v3"}
    trajectory_ids = {item["sample_id"] for trajectory in trajectories for item in trajectory["turns"]}
    assert row_ids == trajectory_ids
    assert all(all(turn["runtime_checks"].values()) for trajectory in trajectories for turn in trajectory["turns"])
    return {"unique_inputs": True, "template_families_disjoint": True, "whole_trajectories_disjoint": True,
            "v2_val_test_excluded": True, "approved_preview_excluded_from_val_test": True,
            "runtime_prompt_parity": True, "schema_and_field_evidence": True,
            "scripted_runtime_state_checks": len(row_ids), "external_model_calls": 0, "database_writes": 0}


async def build(agent_project, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录已有文件，请使用新的版本或检查已有数据")
    spec = json.loads((DATA_ROOT / "spec_v3.json").read_text(encoding="utf-8"))
    families = json.loads((DATA_ROOT / "scenarios_v3.json").read_text(encoding="utf-8"))["families"]
    logging.disable(logging.INFO)
    sys.path.insert(0, str(agent_project))
    from app.domain.user.models import UserContext
    from local_runtime.chat_cli import create_terminal_graph, run_turn
    from app.workflows.intent_protocol import RULES
    assert RULES == (DATA_ROOT / "intent_prompt_v2.txt").read_text(encoding="utf-8"), "Agent协议与训练协议不同"
    replay, selection = select_replay(spec)
    splits, trajectories = {"train": replay, "val": [], "test": []}, []
    recipe = spec["new_trajectories"]
    for family in families:
        count = recipe["variants_per_family"][family["split"]]
        for kind in KINDS:
            for index in range(count):
                values = parameters(recipe["seed"], family["id"], kind, index)
                definition = make_trajectory(family, kind, values, index)
                rows, evidence = await run_trajectory(definition, create_terminal_graph, run_turn, UserContext)
                splits[family["split"]].extend(rows)
                trajectories.append(evidence)
        print(f"完成模板族 {family['id']} ({family['split']})", flush=True)
    for index, rows in enumerate(splits.values()):
        random.Random(recipe["seed"] + index).shuffle(rows)
    checks = audit(splits, trajectories)
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
        outputs[split] = {"file": path.name, "records": len(rows), "sha256": sha256(path),
                          "origins": dict(Counter(row["origin"] for row in rows)),
                          "categories": dict(Counter(row["category"] for row in rows)),
                          "intents": dict(Counter(row["expected"]["intent"] for row in rows if "expected" in row)),
                          "groups": len({row["group_id"] for row in rows}),
                          "trajectories": len({row["trajectory_id"] for row in rows if "trajectory_id" in row})}
    path = output / "trajectories.jsonl"
    path.write_bytes("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in trajectories).encode("utf-8"))
    save_json(output / "replay_selection.json", selection)
    inputs = ("spec_v3.json", "scenarios_v3.json", "intent_prompt_v2.txt", "v2/train.jsonl",
              "v2/val.jsonl", "v2/test.jsonl", "v3_preview/spec.json", "v3_preview/samples.json")
    manifest = {"version": "v3", "status": "generated_pending_encoding_check", "assistant_name": "TrainPilot",
                "training_start": spec["training_start"], "output_policy": spec["output_policy"],
                "seed": recipe["seed"], "generator_sha256": sha256(__file__),
                "inputs_sha256": {name: sha256(DATA_ROOT / name) for name in inputs},
                "agent_sources_sha256": {name: sha256(agent_project / name) for name in AGENT_SOURCES},
                "outputs": outputs,
                "evidence": {"trajectories": {"file": "trajectories.jsonl", "sha256": sha256(path), "records": len(trajectories)},
                             "replay_selection": {"file": "replay_selection.json", "sha256": sha256(output / "replay_selection.json")}},
                "replay_selection_counts": selection["counts"], "checks": checks,
                "regression_files": {"validation": "../v2/val.jsonl", "test": "../v2/test.jsonl"},
                "validation_scope": spec["validation_scope"], "split_policy": spec["split_policy"],
                "limits": spec["limits"]}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "splits": {key: len(rows) for key, rows in splits.items()},
                      "checks": checks}, ensure_ascii=False, indent=2))


def check_encoding(output, tokenizer_dir, max_length):
    import datasets  # Windows先加载Arrow，与训练入口一致。
    from transformers import AutoTokenizer
    from model.dataset.dataset import QwenSFTEncoder
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoder = QwenSFTEncoder(tokenizer, max_length=max_length)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    splits = {split: read_jsonl(output / f"{split}.jsonl") for split in ("train", "val", "test")}
    audit(splits, read_jsonl(output / "trajectories.jsonl"))
    statistics = {}
    for split, rows in splits.items():
        assert sha256(output / f"{split}.jsonl") == manifest["outputs"][split]["sha256"]
        lengths, targets, with_history = [], 0, 0
        for row in rows:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
            length = len(tokenizer(text, add_special_tokens=False).input_ids)
            assert length <= max_length, (row["id"], length)
            encoded = encoder(row)
            supervised = [token for token in encoded["labels"] if token != -100]
            assert encoded["labels"][0] == -100 and supervised, row["id"]
            assert row["messages"][-1]["content"] in tokenizer.decode(supervised), row["id"]
            assert supervised[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>"), row["id"]
            lengths.append(length)
            targets += len(supervised)
            with_history += bool(row["scenario"].get("history"))
        statistics[split] = {"records": len(rows), "min_tokens": min(lengths), "max_tokens": max(lengths),
                             "supervised_tokens": targets, "with_history": with_history,
                             "zero_target_samples": 0, "truncated_samples": 0}
        print(split, json.dumps(statistics[split], ensure_ascii=False), flush=True)
    # 通过训练端实际使用的读取器，检查元数据的混合结构能被Arrow加载。
    raw = datasets.load_dataset("json", data_files={name: str(output / f"{name}.jsonl") for name in splits},
                                cache_dir=str(ROOT / "model/dataset/datasets/cache/v3-check"))
    assert all(len(raw[name]) == len(rows) for name, rows in splits.items())
    manifest.update(status="ready_for_server_check", generator_sha256=sha256(__file__),
                    encoding_check={"max_length": max_length, "splits": statistics,
                                    "assistant_answers_and_eos_preserved": True, "training_json_reader": "passed"})
    save_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-project", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()
    if not args.agent_project and not args.tokenizer_dir:
        parser.error("生成请指定--agent-project；只检查编码请指定--tokenizer-dir")
    if args.agent_project:
        asyncio.run(build(args.agent_project.resolve(), args.output_dir.resolve()))
    if args.tokenizer_dir:
        check_encoding(args.output_dir.resolve(), args.tokenizer_dir.resolve(), args.max_length)


if __name__ == "__main__":
    main()
