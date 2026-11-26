"""从已审阅 V5 的各自划分抽出业务协议题；不生成新标签、不混入 RAG 歧义题。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from model.trainer.agent_rewards import intent_reward


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"source_version": "v5", "task": "intent_and_complete_fields", "splits": {}}
    groups, prompts = {}, {}
    preview = []
    for split in ("train", "val", "test"):
        path = source / f"{split}.jsonl"
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["mode"] != "intent_parse":
                    continue
                gold = row.get("expected") or json.loads(row["messages"][-1]["content"])
                assert intent_reward(json.dumps(gold, ensure_ascii=False), gold)["reward"] == 1
                record = {"id": row["id"], "group_id": row["group_id"], "category": row["category"],
                          "prompt": row["messages"][:-1], "gold": gold}
                assert record["prompt"][-1]["role"] == "user"
                key = json.dumps(record["prompt"], ensure_ascii=False, sort_keys=True)
                for cache, value in ((groups, record["group_id"]), (prompts, key)):
                    assert cache.get(value, split) == split, "划分存在重叠"
                    cache[value] = split
                records.append(record)
        target = output / f"{split}.jsonl"
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        manifest["splits"][split] = {"records": len(records), "categories": dict(Counter(r["category"] for r in records)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        if split == "train":
            seen = set()
            for r in records:
                if r["category"] not in seen:
                    seen.add(r["category"])
                    preview.append({"id": r["id"], "category": r["category"],
                                    "user_prompt": r["prompt"][-1]["content"], "gold": r["gold"]})
    manifest["group_overlap"] = 0
    manifest["prompt_overlap"] = 0
    manifest["gold_never_passed_to_model"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "preview.json").write_text(json.dumps(preview[:12], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({s: m["records"] for s, m in manifest["splits"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("model/dataset/datasets/trainpilot_agent/v5"))
    parser.add_argument("--output", type=Path, default=Path("model/dataset/datasets/trainpilot_agent/grpo_v1"))
    args = parser.parse_args()
    prepare(args.source, args.output)
