"""构造可复现的 TrainPilot 专项合成数据；不调用外部模型 API。

python -m dataset.build_agent_data
相同场景模板的所有参数变体放在同一划分；现有基线题不写入训练集。
"""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset/datasets/trainpilot_agent"
INTENTS = {"submit_leave", "delete_application", "approve", "query_status", "query_pending",
           "policy_qa", "get_suggestion", "confirm_action", "cancel_action", "general_chat", "clarify"}
FIELDS = {"leave_type", "start_date", "end_date", "reason", "app_id", "decision", "comments"}
QUOTAS = {
    "identity": (24, 4, 4), "natural": (60, 7, 7),
    "leave_full": (80, 10, 10), "leave_relative": (40, 5, 5), "policy_qa": (90, 11, 11),
    "query_status": (50, 6, 6), "query_pending": (40, 5, 5), "approve": (70, 8, 8),
    "delete_application": (50, 6, 6), "get_suggestion": (40, 5, 5),
    "general_chat": (40, 5, 5), "clarify": (40, 4, 4),
    "missing_all": (30, 4, 4), "missing_dates": (30, 4, 4), "missing_reason": (30, 4, 4),
    "missing_type": (30, 4, 4), "approve_missing_id": (30, 4, 4),
    "approve_missing_decision": (30, 4, 4), "cancel_missing_id": (30, 3, 3), "suggest_missing_id": (30, 3, 3),
    "history_fill": (56, 8, 8), "history_modify": (56, 8, 8), "confirm_action": (56, 8, 8),
    "cancel_action": (56, 8, 8), "pending_other_task": (56, 6, 6), "pending_cancel_existing": (56, 6, 6),
}
NATURAL = [
    ("请直接计算{a}+{b}，只输出数字。", lambda c: str(c["a"] + c["b"])),
    ("将数字{b}、{a}按从小到大排列，用英文逗号连接。", lambda c: ",".join(map(str, sorted([c["a"], c["b"]])))),
    ("原样复述这段文本：{reason}。不要添加其他内容或标点。", lambda c: c["reason"]),
    ("从‘申请编号为{app_id}，等待处理’中提取编号，只输出编号。", lambda c: c["app_id"]),
    ("求{a}加{b}的和，不要解释。", lambda c: str(c["a"] + c["b"])),
    ("有{a}本书，又增加{b}本。总数是多少？仅答数字。", lambda c: str(c["a"] + c["b"])),
]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_prompt(template, scenario):
    pending = scenario.get("pending_action")
    history = [{"role": m["role"], "content": m["content"][:120]} for m in scenario.get("history", [])[-6:]]
    context = {"current_date": scenario["current_date"], "history": history,
               "pending_action": {"action_type": pending["action_type"], "draft": pending["draft"]} if pending else None}
    if scenario.get("summary"):
        context["summary"] = scenario["summary"]
    return template + "\n给定上下文：\n" + json.dumps(context, ensure_ascii=False) + "\n用户输入：\n" + scenario["user_message"]


def make_context(rng):
    today = date(2026, 1, 1) + timedelta(days=rng.randrange(1000))
    start = today + timedelta(days=rng.randrange(3, 50))
    end = start + timedelta(days=rng.randrange(4))
    offset = rng.choice([1, 2])
    decision = rng.choice(["通过", "驳回"])
    leave_type = rng.choice(["年假", "病假", "事假", "婚假", "产假", "丧假"])
    reasons = {
        "年假": ["安排休息", "外出旅行", "回乡探亲"], "病假": ["门诊就诊", "身体不适", "到医院复查"],
        "事假": ["办理证件", "照顾家人", "处理个人事务"], "婚假": ["筹办婚礼", "办理结婚手续", "参加自己的婚礼"],
        "产假": ["分娩休养", "产后恢复", "安排生育休假"], "丧假": ["处理亲属丧事", "参加亲属葬礼", "办理亲属身后事"],
    }
    return {
        "today": today.isoformat(), "start_date": start.isoformat(), "end_date": end.isoformat(),
        "new_start": (start + timedelta(days=7)).isoformat(), "new_end": (end + timedelta(days=7)).isoformat(),
        "relative_day": "明天" if offset == 1 else "后天", "relative_date": (today + timedelta(days=offset)).isoformat(),
        "leave_type": leave_type, "reason": rng.choice(reasons[leave_type]),
        "app_id": f"leave_{rng.getrandbits(48):012x}",
        "pending_app_id": f"leave_{rng.getrandbits(48):012x}",
        "decision": decision, "decision_word": rng.choice(["同意", "批准", "通过"] if decision == "通过" else ["驳回", "拒绝"]),
        "comments": rng.choice(["已核对证明", "材料齐全", "已确认安排"] if decision == "通过" else ["请补充材料", "信息不完整", "需要重新说明"]),
        "policy_topic": rng.choice(["年假申请", "请假审批", "差旅报销", "加班调休", "采购审批", "信息安全", "产品服务", "员工培训", "合同归档", "费用报销"]),
        "pending_type": rng.choice(["submit_leave", "approve", "cancel_leave"]),
        "a": rng.randrange(10, 999), "b": rng.randrange(2, 100),
    }


def target_for(kind, c):
    leave = {k: c[k] for k in ("leave_type", "start_date", "end_date", "reason")}
    if kind in {"leave_full", "history_fill", "history_modify"}:
        if kind == "history_modify":
            leave.update(start_date=c["new_start"], end_date=c["new_end"])
        return "submit_leave", leave
    if kind == "leave_relative":
        return "submit_leave", {**leave, "start_date": c["relative_date"], "end_date": c["relative_date"]}
    if kind == "missing_all":
        return "submit_leave", {}
    if kind.startswith("missing_"):
        omitted = {"missing_dates": {"start_date", "end_date"}, "missing_reason": {"reason"}, "missing_type": {"leave_type"}}[kind]
        return "submit_leave", {k: v for k, v in leave.items() if k not in omitted}
    if kind in {"approve", "approve_missing_id", "approve_missing_decision"}:
        fields = {k: c[k] for k in ("app_id", "decision", "comments")}
        if kind == "approve_missing_id":
            fields.pop("app_id")
        if kind == "approve_missing_decision":
            fields = {"app_id": c["app_id"]}
        return "approve", fields
    if kind in {"delete_application", "pending_cancel_existing"}:
        return "delete_application", {"app_id": c["app_id"]}
    if kind == "get_suggestion":
        return kind, {"app_id": c["app_id"]}
    if kind in {"cancel_missing_id", "suggest_missing_id"}:
        return {"cancel_missing_id": "delete_application", "suggest_missing_id": "get_suggestion"}[kind], {}
    if kind == "pending_other_task":
        return "query_status", {}
    return kind, {}


def build_row(kind, template_id, context, user_template, prompt_template, row_id):
    question = user_template.format(**context)
    if kind in {"identity", "natural"}:
        answer = context["identity_answer"] if kind == "identity" else NATURAL[template_id][1](context)
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        if kind == "identity":
            messages.insert(0, {"role": "system", "content": "你是 TrainPilot，一名企业业务助手。帮助用户理解企业资料、整理请假信息和理解审批需求。不要编造身份、权限或尚未执行的操作结果。"})
        return {"id": row_id, "category": kind, "mode": "assistant_reply", "group_id": f"{kind}/template-{template_id}",
                "scenario": {"user_message": question}, "messages": messages}
    intent, fields = target_for(kind, context)
    scenario = {"user_message": question, "current_date": context["today"], "history": [], "summary": "", "pending_action": None}
    draft = {k: context[k] for k in ("leave_type", "start_date", "end_date", "reason")}
    if kind == "history_fill":
        scenario["history"] = [{"role": "user", "content": f"我想请{context['leave_type']}。"},
                               {"role": "assistant", "content": "请补充请假起止日期和原因。"}]
    if kind == "history_modify":
        scenario["history"] = [{"role": "user", "content": f"申请{context['leave_type']}，{context['start_date']}到{context['end_date']}，原因是{context['reason']}。"},
                               {"role": "assistant", "content": "已整理草稿，请核对后确认。"}]
    if kind in {"history_modify", "confirm_action", "cancel_action", "pending_other_task", "pending_cancel_existing"}:
        scenario["pending_action"] = {"action_type": "submit_leave", "draft": draft}
        if kind != "history_modify" and context["pending_type"] == "approve":
            scenario["pending_action"] = {"action_type": "approve", "draft": {
                "app_id": context["pending_app_id"],
                "decision": context["decision"], "comments": context["comments"]}}
        elif kind != "history_modify" and context["pending_type"] == "cancel_leave":
            scenario["pending_action"] = {"action_type": "cancel_leave", "draft": {
                "app_id": context["pending_app_id"]}}
    target = {"intent": intent, "extracted_info": fields}
    return {"id": row_id, "category": kind, "mode": "intent_parse", "group_id": f"{kind}/template-{template_id}",
            "scenario": scenario, "expected": target,
            "messages": [{"role": "user", "content": render_prompt(prompt_template, scenario)},
                         {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}]}


def validate_rows(splits):
    seen_messages, seen_groups = {}, {}
    for split, rows in splits.items():
        for row in rows:
            assert "澜图" not in canonical(row) and "{person}" not in canonical(row)
            key = canonical(row["messages"])
            if key in seen_messages:
                raise ValueError(f"重复对话: {row['id']}")
            seen_messages[key] = split
            group = row["group_id"]
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError(f"场景模板跨划分: {group}")
            seen_groups[group] = split
            if "expected" not in row:
                continue
            answer = json.loads(row["messages"][-1]["content"])
            assert answer == row["expected"] and set(answer) == {"intent", "extracted_info"}
            assert answer["intent"] in INTENTS
            fields = answer["extracted_info"]
            assert set(fields) <= FIELDS
            assert all(v not in (None, "", "未知", "请假原因", "员工姓名") for v in fields.values())
            evidence = row["scenario"]["user_message"] + canonical(row["scenario"].get("history", []))
            pending = row["scenario"].get("pending_action")
            if pending:
                evidence += canonical(pending["draft"])
            for field in ("reason", "app_id", "comments"):
                assert field not in fields or fields[field] in evidence, (row["id"], field)
            for field in ("start_date", "end_date"):
                if field in fields:
                    date.fromisoformat(fields[field])
            if row["category"] == "leave_relative":
                offset = 1 if "明天" in row["scenario"]["user_message"] else 2
                expected_date = date.fromisoformat(row["scenario"]["current_date"]) + timedelta(days=offset)
                assert fields["start_date"] == fields["end_date"] == expected_date.isoformat()
    return {"exact_dialogues_unique": True, "template_groups_disjoint": True, "schema_and_literal_fields_checked": True}


def build_dataset(seed=20261022):
    templates = json.loads((DATA_ROOT / "scenarios_v2.json").read_text(encoding="utf-8"))["templates"]
    identities = json.loads((DATA_ROOT / "identity_v2.json").read_text(encoding="utf-8"))
    prompt_template = (DATA_ROOT / "intent_prompt_v2.txt").read_text(encoding="utf-8")
    rng, splits, seen = random.Random(seed), {}, set()
    for split_index, split in enumerate(("train", "val", "test")):
        rows = []
        for kind, counts in QUOTAS.items():
            for index in range(counts[split_index]):
                template_id = index % 4 if split == "train" else split_index + 3
                identity = None
                if kind == "identity":
                    group = identities[template_id]
                    item = index // 4 if split == "train" else index
                    identity = group["items"][item] if "items" in group else {"question": group["questions"][item], "answer": group["answer"]}
                    user_template = identity["question"]
                else:
                    user_template = NATURAL[template_id][0] if kind == "natural" else templates[kind][template_id]
                for attempt in range(100):
                    context = make_context(rng)
                    if identity:
                        context["identity_answer"] = identity["answer"]
                    row = build_row(kind, template_id, context, user_template, prompt_template,
                                    f"{split}-{kind}-{index:04d}")
                    fingerprint = canonical(row["messages"])
                    if fingerprint not in seen:
                        seen.add(fingerprint)
                        rows.append(row)
                        break
                else:
                    raise ValueError(f"场景表达不足以生成不重复样本: {kind}")
        rng.shuffle(rows)
        splits[split] = rows
    checks = validate_rows(splits)
    return splits, checks


def check_encoding(output_dir, tokenizer_dir, max_length):
    from transformers import AutoTokenizer
    from dataset.dataset import QwenSFTEncoder
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoder = QwenSFTEncoder(tokenizer, max_length=max_length)
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = {name: read_jsonl(output_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    validate_rows(splits)
    statistics = {}
    for name, rows in splits.items():
        data_path = output_dir / f"{name}.jsonl"
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest["outputs"][name]["sha256"]
        lengths, supervised = [], 0
        for row in rows:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
            length = len(tokenizer(text, add_special_tokens=False).input_ids)
            assert length <= max_length, (row["id"], length, max_length)
            encoded = encoder(row)
            targets = [token for token in encoded["labels"] if token != -100]
            assert targets and row["messages"][-1]["content"] in tokenizer.decode(targets), row["id"]
            assert targets[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>"), row["id"]
            assert encoded["labels"][0] == -100
            lengths.append(length)
            supervised += len(targets)
        statistics[name] = {"records": len(rows), "min_tokens": min(lengths), "max_tokens": max(lengths),
                            "supervised_tokens": supervised, "zero_target_samples": 0, "truncated_samples": 0}
    manifest["encoding_check"] = {"max_length": max_length, "splits": statistics, "assistant_answers_preserved": True}
    manifest["status"] = "ready_for_server_check"
    manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["encoding_check"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "v2")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--tokenizer-dir", type=Path, help="只检查已有数据编码；指定本地Qwen tokenizer目录")
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()
    if args.tokenizer_dir:
        check_encoding(args.output_dir, args.tokenizer_dir, args.max_length)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("输出目录非空，请使用新版本目录")
    splits, checks = build_dataset(args.seed)
    baseline_path = ROOT / "runs/agent-baseline-20261022/agent_cases.json"
    if baseline_path.exists():
        excluded = {r["message"] for r in json.loads(baseline_path.read_text(encoding="utf-8"))}
        assert all(row["scenario"]["user_message"] not in excluded for rows in splits.values() for row in rows)
        checks["existing_12_baseline_questions_excluded"] = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        path = args.output_dir / f"{split}.jsonl"
        path.write_bytes(payload)
        files[split] = {"file": path.name, "records": len(rows), "sha256": hashlib.sha256(payload).hexdigest(),
                        "categories": dict(Counter(r["category"] for r in rows)),
                        "groups": len({r["group_id"] for r in rows})}
    manifest = {"version": "v2", "status": "generated_pending_checks", "seed": args.seed, "assistant_name": "TrainPilot", "source": "按用户确认规范编写场景与标签，程序确定性组合必要上下文和字段；未调用外部模型API。",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prompt_source": json.loads((DATA_ROOT / "prompt_source_v2.json").read_text(encoding="utf-8")),
                "inputs_sha256": {name: hashlib.sha256((DATA_ROOT / name).read_bytes()).hexdigest() for name in ("scenarios_v2.json", "intent_prompt_v2.txt", "identity_v2.json", "spec_v2.json")},
                "outputs": files, "checks": checks,
                "scope": "身份、简单指令保持、11种业务意图、缺失字段和多轮状态；不训练自由工具调用或广泛推理。",
                "limits": ["由有限人工编写模板构造，1500条不等于1500种独立自然表达；需用新收集问题复测。",
                           "同一表达模板及其参数变体不跨划分，但场景概念和协议仍共享，未证明语义完全不重合。",
                           "标签由构造规则核验，未经过独立人工逐条标注或教师模型复核。",
                           "工作日、权限和申请有效性由业务程序校验；模型只提取用户表达，不假定操作已执行。",
                           "test.jsonl禁止传入训练入口；保留旧12题基线作额外回归。"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "counts": {k: len(v) for k, v in splits.items()}, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
