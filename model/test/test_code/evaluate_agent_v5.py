"""V5只读评测：冻结留出题、固定子集对照和本机Agent API；不训练、不联网下载。"""
import argparse
from collections import Counter, defaultdict
from http.server import HTTPServer
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "model/dataset/datasets/trainpilot_agent"
OUT = ROOT / "runs/lora-v5-eval-20261024"


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text):
    return re.sub(r"[\s。；;，,：:]", "", text).lower()


def grade(case, text):
    if case["mode"] == "intent_parse":
        from model.test.test_code.evaluate_agent_lora import score_business
        return {"kind": "business", **score_business(case, text)}
    if not case["mode"].endswith("_rag"):
        return {"kind": "natural", "passed": None, "mentions_trainpilot": "trainpilot" in text.lower()}
    expected = case["expected_claims"]
    schema, refused, claims = False, False, []
    if case["mode"] == "enterprise_rag":
        try:
            payload = json.loads(text)
            refused, claims = payload["refused"], payload["claims"]
            schema = set(payload) == {"refused", "claims"} and type(refused) is bool and isinstance(claims, list)
            schema = schema and ((refused and not claims) or (not refused and bool(claims)))
            schema = schema and all(isinstance(c, dict) and set(c) == {"text", "sources"}
                and isinstance(c["text"], str) and bool(c["text"].strip()) and isinstance(c["sources"], list)
                and bool(c["sources"]) and all(type(i) is int and 1 <= i <= len(case["evidence"]) for i in c["sources"]) for c in claims)
            actual = sorted((normalized(c["text"]), tuple(sorted(c["sources"]))) for c in claims) if schema else []
            target = sorted((normalized(c["text"]), tuple(sorted(c["sources"]))) for c in expected)
            exact = schema and refused == (not bool(expected)) and actual == target
        except (ValueError, KeyError, TypeError, AttributeError):
            schema, exact = False, False
    else:
        markers = re.findall(r"\[(\d+)\]", text)
        refusal = "现有企业资料未覆盖该问题，建议联系对应业务部门确认。"
        refused = text.strip() == refusal
        sentences = [s for s in text.splitlines() if s.strip()]
        schema = refused or bool(markers) and all(1 <= int(i) <= len(case["evidence"]) for i in markers)
        schema = schema and (refused or all(re.search(r"\[\d+\]", s) for s in sentences))
        exact = schema and normalized(text) == normalized(case["messages"][-1]["content"])
    return {"kind": "rag", "schema_and_refs": bool(schema), "refused": bool(refused),
            "refusal_decision_correct": bool(schema and refused == (not bool(expected))),
            "reference_exact": bool(exact), "passed": bool(exact),
            "grading_note": "reference_exact是参考文本及引用的一致率；释义改写需人工核验，不能直接等同语义准确率。"}


def summary(items):
    output = {}
    for mode in sorted({r["case"]["mode"] for r in items}):
        selected = [r for r in items if r["case"]["mode"] == mode]
        output[mode] = {"total": len(selected), "truncated": sum(r["stop_reason"] != "eos" for r in selected),
                        "generation_seconds": sum(r["seconds"] for r in selected)}
        keys = ("passed", "strict_json", "intent_ok", "fields_exact") if mode == "intent_parse" else (
            ("schema_and_refs", "refusal_decision_correct", "reference_exact") if mode.endswith("_rag") else ("mentions_trainpilot",))
        for key in keys:
            output[mode][key] = sum(bool(r.get(key)) for r in selected)
        categories = {}
        for c in sorted({r["case"]["category"] for r in selected}):
            part = [r for r in selected if r["case"]["category"] == c]
            categories[c] = {"total": len(part), "passed": sum(bool(r.get("passed")) for r in part),
                             "schema_and_refs": sum(bool(r.get("schema_and_refs")) for r in part)}
        output[mode]["categories"] = categories
    return output


def freeze(output):
    from model.test.test_code.evaluate_qwen_sft import sha256
    output.mkdir(parents=True, exist_ok=False)
    cases = rows(DATA / "v5/test.jsonl")
    groups = defaultdict(list)
    for c in cases:
        groups[(c["mode"], c["category"])].append(c)
    panel = []
    rng = random.Random(20261024)
    for key in sorted(groups):
        pool = sorted(groups[key], key=lambda r: r["id"])
        rng.shuffle(pool)
        panel.extend(r["id"] for r in pool[:4 if key[0].endswith("_rag") else 2])
    save(output / "cases.json", cases)
    save(output / "plan.json", {"frozen_before_generation": True, "date": "2026-10-24",
        "cases_sha256": sha256(output / "cases.json"), "source_test_sha256": sha256(DATA / "v5/test.jsonl"),
        "full_final_cases": len(cases), "matched_panel_cases": len(panel), "panel_ids": panel,
        "panel_selection": "固定seed，按mode/category分层，每层RAG最多4条，其他最多2条；不依据模型输出选题",
        "variants": ["v5_epoch2", "v5_epoch1", "v4_epoch2"],
        "limits": {"intent_and_identity": 256, "rag": 512},
        "decode": "BF16 greedy same Full SFT, same tokenizer/prompt, no training",
        "limits_of_claims": ["V4旧业务题是开发回归，非独立真实用户验收", "V5题为合成/摘录题；参考逐字一致率不等于语义正确率",
                             "业务仿真不验证MySQL并发锁、HTTP认证或Redis持久化"]})
    print("FROZEN", len(cases), "full final;", len(panel), "matched panel", flush=True)


def models(args):
    import datasets
    import torch
    from safetensors.torch import load_file
    from model.test.test_code.qwen_agent_api import load_pair, make_handler
    from model.test.test_code.evaluate_qwen_sft import generate, sha256
    output = args.output_dir
    torch.set_num_threads(2)
    cases, plan = read(output / "cases.json"), read(output / "plan.json")
    assert sha256(output / "cases.json") == plan["cases_sha256"]
    base = ROOT / "runs/trainpilot_agent_lora_v5"
    state = torch.load(base / "checkpoint.pt", map_location="cpu", weights_only=True)
    exported = load_file(str(base / "adapter/adapter_model.safetensors"))
    epoch2 = load_file(str(base / "epoch_02/adapter/adapter_model.safetensors"))
    assert state["progress"]["completed_epochs"] == 2 and state["progress"]["global_step"] == 2068
    assert exported.keys() == epoch2.keys() == state["weights"].keys()
    assert all(torch.equal(t, epoch2[k]) and torch.equal(t, state["weights"][k]) for k, t in exported.items())
    training = read(base / "run_manifest.json")
    assert training["data"]["source_sha256"] == sha256(DATA / "v5/train.jsonl")
    assert training["data"]["validation_source_sha256"] == sha256(DATA / "v5/val.jsonl")
    save(output / "artifact_check.json", {"progress": state["progress"], "tensors": len(exported),
         "checkpoint_final_epoch2_identical": True, "training_manifest": training,
         "epoch_metrics": {s: read(base / s / "metrics.json") for s in ("epoch_01", "epoch_02")}})
    del state, exported, epoch2
    directories = {"v5_epoch2": base / "adapter", "v5_epoch1": base / "epoch_01/adapter",
                   "v4_epoch2": ROOT / "runs/trainpilot_agent_lora_v4/adapter"}
    print("Loading original Full SFT + three read-only adapters", flush=True)
    model, tokenizer, metadata = load_pair(ROOT / "runs/qwen-fullsft/checkpoint.pt", ROOT / "runs/qwen-base-reference", directories["v5_epoch2"])
    for variant in ("v5_epoch1", "v4_epoch2"):
        model.load_adapter(str(directories[variant]), adapter_name=variant, is_trainable=False, autocast_adapter_dtype=False)
    names = {"v5_epoch2": "default", "v5_epoch1": "v5_epoch1", "v4_epoch2": "v4_epoch2"}
    metadata.update(adapters={n: {"path": str(p), "sha256": sha256(p / "adapter_model.safetensors")} for n, p in directories.items()},
                    cases_sha256=plan["cases_sha256"], limits=plan["limits"])
    save(output / "model_metadata.json", metadata)
    summaries = {}
    for variant in plan["variants"]:
        model.set_adapter(names[variant])
        file = output / f"{variant}.jsonl"
        completed = rows(file) if file.exists() else []
        done = {r["case"]["id"] for r in completed}
        selected = cases if variant == "v5_epoch2" else [c for c in cases if c["id"] in set(plan["panel_ids"])]
        with file.open("a", encoding="utf-8") as stream:
            for index, case in enumerate(selected, 1):
                if case["id"] in done:
                    continue
                ids = tokenizer.apply_chat_template(case["messages"][:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                limit = 512 if case["mode"].endswith("_rag") else 256
                result = generate(model, tokenizer, ids, "cuda", limit)
                row = {"variant": variant, "case": case, **result, **grade(case, result["text"])}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                completed.append(row)
                if index == 1 or index % 20 == 0 or index == len(selected):
                    print(variant, index, "/", len(selected), "truncated=", sum(r["stop_reason"] != "eos" for r in completed), flush=True)
        panel = [r for r in completed if r["case"]["id"] in set(plan["panel_ids"])]
        summaries[variant] = {"all": summary(completed), "matched_panel": summary(panel)}
        save(output / "model_summary.json", summaries)
    metadata["weights_unchanged"] = sha256(ROOT / "runs/qwen-fullsft/checkpoint.pt") == metadata["checkpoint_sha256"]
    assert metadata["weights_unchanged"]
    for variant, path in directories.items():
        assert sha256(path / "adapter_model.safetensors") == metadata["adapters"][variant]["sha256"]
    save(output / "model_metadata.json", metadata)
    if args.serve:
        def infer(messages, limit, variant):
            model.set_adapter(names[variant])
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            result = generate(model, tokenizer, ids, "cuda", min(limit, 512))
            with (output / "api_requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"variant": variant, "messages": messages, **result}, ensure_ascii=False) + "\n")
            return result
        handler = make_handler(lambda m, n: infer(m, n, "v5_epoch2"), "trainpilot-lora", {
            "trainpilot-v5-epoch1": lambda m, n: infer(m, n, "v5_epoch1"),
            "trainpilot-v4": lambda m, n: infer(m, n, "v4_epoch2")})
        print("READY", args.port, flush=True)
        HTTPServer(("127.0.0.1", args.port), handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "models"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8015)
    args = parser.parse_args()
    freeze(args.output_dir) if args.mode == "freeze" else models(args)


if __name__ == "__main__":
    main()
