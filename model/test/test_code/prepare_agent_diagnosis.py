"""从已有训练/开发数据冻结诊断面板；不生成新SFT标签，不使用最终测试题推理。"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from model.trainer.agent_rewards import intent_reward


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "model/dataset/datasets/trainpilot_agent"
OUT = ROOT / "runs/agent-diagnosis-20261030"


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized(row):
    if "gold" not in row:
        return deepcopy(row)
    return {"id": row["id"], "group_id": row["group_id"], "category": row["category"],
            "mode": "intent_parse", "expected": row["gold"], "origin": row["origin"],
            "scenario": {**row["context"], "user_message": row["user"]},
            "messages": row["prompt"] + [{"role": "assistant", "content": canonical(row["gold"])}]}


def item(row, source, split, reason):
    return {"id": "diag/" + row["id"], "source_file": source, "source_split": split,
            "selection_reason": reason, "case": normalized(row), "prompt_variant": "stored"}


def select_panel():
    paths = {"v6/train.jsonl": DATA / "v6/train.jsonl", "v6/val.jsonl": DATA / "v6/val.jsonl",
             "grpo_v2/train.jsonl": DATA / "grpo_v2/train.jsonl"}
    sources = {name: read_rows(path) for name, path in paths.items()}
    selected = []
    # 完整取一个训练家族及两个开发家族，保留同话术、不同状态/证据的对照。
    for source, family_prefixes, split in (("v6/train.jsonl", ("v6-train-f00-",), "train"),
                                          ("v6/val.jsonl", ("v6-val-f10-", "v6-val-f11-"), "development")):
        selected += [item(r, source, split, "complete_state_and_evidence_family")
                     for r in sorted(sources[source], key=lambda r: r["id"]) if r["id"].startswith(family_prefixes)]
    replay = [r for r in sources["v6/train.jsonl"] if r["origin"].startswith("v5_train_replay")]
    buckets = defaultdict(list)
    for row in replay:
        key = (row["mode"], row["expected"]["intent"] if row["mode"] == "intent_parse" else row["category"])
        buckets[key].append(row)
    controls = []
    for key in sorted(buckets):
        if key[0] == "intent_parse":
            controls += sorted(buckets[key], key=lambda r: r["id"])[:2]
    for mode in ("enterprise_rag", "oa_rag"):
        groups = {k: sorted(v, key=lambda r: r["id"]) for k, v in buckets.items() if k[0] == mode}
        # 分层轮转，避免只取字典序靠前的单一RAG类型。
        count = 0
        while groups and count < 6:
            for key in list(sorted(groups)):
                controls.append(groups[key].pop(0)); count += 1
                if not groups[key]: del groups[key]
                if count == 6: break
    natural = [r for r in replay if r["mode"] in ("direct_reply", "agent_reply")]
    for mode in ("direct_reply", "agent_reply"):
        controls += sorted([r for r in natural if r["mode"] == mode], key=lambda r: r["id"])[:3]
    selected += [item(r, "v6/train.jsonl", "train", "old_task_replay_control") for r in controls]
    buckets = defaultdict(list)
    for row in sources["grpo_v2/train.jsonl"]:
        buckets[(row["category"], row["group_id"])].append(row)
    for category in sorted({key[0] for key in buckets}):
        keys = sorted(k for k in buckets if k[0] == category)[:2]
        for key in keys:
            values = sorted(buckets[key], key=lambda r: r["id"])
            selected += [item(values[i], "grpo_v2/train.jsonl", "train", "rl_training_prompt_without_model_score_selection")
                         for i in (0, 3, 7)]
    assert len(selected) == 214 and len({r["id"] for r in selected}) == len(selected)
    return selected, paths


def audit_data():
    sources = {}
    by_prompt = defaultdict(list)
    violations = []
    for version in ("v5", "v6", "grpo_v2"):
        for split in ("train", "val"):
            path = DATA / version / (split + ".jsonl")
            records = read_rows(path)
            sources[f"{version}/{split}"] = {"records": len(records), "sha256": sha(path)}
            for original in records:
                row = normalized(original)
                prompt = row["messages"][:-1]
                target = canonical(row["expected"]) if row["mode"] == "intent_parse" else row["messages"][-1]["content"]
                by_prompt[canonical(prompt)].append({"source": f"{version}/{split}", "id": row["id"], "target": target})
                if row["mode"] != "intent_parse": continue
                if intent_reward(canonical(row["expected"]), row["expected"])["reward"] != 1:
                    violations.append({"id": row["id"], "issue": "invalid_gold"})
                scenario = row.get("scenario") or {}
                text = prompt[-1]["content"]
                if "\n给定上下文：\n" in text:
                    context, user = text.split("\n给定上下文：\n", 1)[1].rsplit("\n用户输入：\n", 1)
                    expected_context = {k: v for k, v in scenario.items()
                                        if k != "user_message" and not (k == "summary" and not v)}
                    expected_context["history"] = [{"role": m["role"], "content": str(m["content"])[:120]}
                                                   for m in scenario.get("history", [])[-6:]]
                    pending = scenario.get("pending_action")
                    expected_context["pending_action"] = {"action_type": pending["action_type"], "draft": pending["draft"]} if pending else None
                    if json.loads(context) != expected_context or user != scenario["user_message"]:
                        violations.append({"id": row["id"], "issue": "rendered_context_differs_from_metadata"})
                if row["expected"]["intent"] in ("confirm_action", "cancel_action") and not scenario.get("pending_action"):
                    violations.append({"id": row["id"], "issue": "confirm_cancel_gold_without_pending"})
    conflicts = [values for values in by_prompt.values() if len({v["target"] for v in values}) > 1]
    alpaca = ROOT / "model/dataset/datasets/alpaca_gpt4_zh/alpaca_gpt4_data_zh.json"
    identity = {"checked": alpaca.exists()}
    if alpaca.exists():
        values = json.loads(alpaca.read_text(encoding="utf-8"))
        examples = [r for r in values if re.search(r"(?:我(?:是|叫)|我的名字|我被称为|我由).{0,60}(?:OpenAI|ChatGPT|Panda|Qwen|MiniMind)", r["output"], re.I)]
        identity = {"checked": True, "sha256": sha(alpaca), "records": len(values),
                    "possible_conflicting_self_identity": len(examples), "examples": examples[:10],
                    "caveat": "正则只作线索，引用、角色扮演可能误命中；不能据此直接证明V6身份错误的因果。"}
    return {"sources": sources, "input_and_state_violations": violations,
            "same_exact_prompt_different_target": conflicts,
            "identity_source_scan": identity,
            "caveat": "精确冲突检查不能证明语义标签全部正确；同提示词的RAG参考改写仍需人工判断。"}


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    selected, sources = select_panel()
    blocked = {}
    for version in ("v2", "v3", "v4", "v5", "v6", "grpo_v1", "grpo_v2"):
        path = DATA / version / "test.jsonl"
        if not path.exists(): continue
        for original in read_rows(path):
            prompt = original["prompt"] if "prompt" in original else original["messages"][:-1]
            blocked[canonical(prompt)] = original["id"]
    for value in selected:
        assert canonical(value["case"]["messages"][:-1]) not in blocked, value["id"]
    path = output / "cases.jsonl"
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    if path.exists(): assert path.read_text(encoding="utf-8") == content
    else: path.write_text(content, encoding="utf-8")
    save(output / "data_audit.json", audit_data())
    save(output / "plan.json", {"selection": "frozen_before_sampling_without_model_score_filtering",
         "cases": len(selected), "cases_sha256": sha(path),
         "modes": dict(Counter(r["case"]["mode"] for r in selected)),
         "source_splits": dict(Counter(r["source_split"] for r in selected)),
         "sources_sha256": {name: sha(path) for name, path in sources.items()},
         "sampling": {"group_size": 4, "seeds": [42, 43], "greedy_per_case": 1,
                      "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
         "expected_primary_completions": len(selected) * 9 * 2,
         "final_test_prompt_overlap": 0,
         "interpretation": "train为拟合/回放诊断，development为已使用的开发集；都不能当作最终泛化分数。",
         "training_performed": False})
    print("DIAGNOSIS_PANEL_READY", len(selected), "cases; 3852 primary completions; no training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    prepare(parser.parse_args().output_dir)
