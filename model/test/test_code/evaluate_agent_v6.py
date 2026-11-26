"""V5、GRPO50、V6 的配对只读评测；原始生成及评分只写入 runs。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from model.test.test_code.evaluate_grpo_regression import digest, grade, greedy, load_cases, rows, save
from model.test.test_code.evaluate_agent_v5 import summary


ROOT = Path(__file__).resolve().parents[3]


def freeze_cases():
    data = ROOT / "model/dataset/datasets/trainpilot_agent/v6"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    path = data / "test.jsonl"
    assert digest(path) == manifest["outputs"]["test"]["sha256"]
    fresh = rows(path)
    assert len(fresh) == 156
    old_path = ROOT / "model/dataset/datasets/trainpilot_agent/grpo_v2_regression/regression_panel.jsonl"
    old, _ = load_cases(old_path)
    cases = [{"suite": "v6_holdout", "case": c} for c in fresh]
    cases += [{"suite": "old_regression", "case": c} for c in old]
    assert len({r["case"]["id"] for r in cases}) == len(cases)
    return cases, {"v6_test": digest(path), "old_panel": digest(old_path)}


def summarize(completed):
    return {suite: summary([r for r in completed if r["suite"] == suite])
            for suite in ("v6_holdout", "old_regression")}


def compare(left, right):
    assert [r["case"] for r in left] == [r["case"] for r in right]
    changes = []
    for before, after in zip(left, right):
        if before["text"] != after["text"] or before["stop_reason"] != after["stop_reason"]:
            changes.append({"suite": after["suite"], "id": after["case"]["id"],
                "mode": after["case"]["mode"], "category": after["case"]["category"],
                "before": before["text"], "after": after["text"],
                "before_passed": before["passed"], "after_passed": after["passed"],
                "before_stop": before["stop_reason"], "after_stop": after["stop_reason"]})
    return {"improved": sum(c["before_passed"] is False and c["after_passed"] is True for c in changes),
            "regressed": sum(c["before_passed"] is True and c["after_passed"] is False for c in changes),
            "changed_outputs": changes}


def run(args):
    cases, data_hashes = freeze_cases()
    adapters = {
        "v5": ROOT / "runs/trainpilot_agent_lora_v5/adapter",
        "grpo50": ROOT / "runs/grpo-v2-pilot-review-20261029/received/trainpilot_grpo_v2_probe200/adapter",
        "v6": ROOT / "runs/trainpilot_agent_lora_v6/adapter",
    }
    expected = {
        "v5": "2c6d865c84c8b1f432250e69ab9d72217263fda4a8811e15ca341abc0d830eb1",
        "grpo50": "ab5bee4c225b17a357fc7fadb20570f4cdbe5d7cb28bde59c805348b355ee83a",
        "v6": "4d89275137bee224b724f06b2ff9990ab784d19e3ded9e570499b9abfd6407b8",
    }
    actual = {name: digest(path / "adapter_model.safetensors") for name, path in adapters.items()}
    assert actual == expected, actual
    print("INPUTS_OK", len(cases), dict(Counter(c["case"]["mode"] for c in cases)), flush=True)
    if args.check_only:
        return
    import datasets  # Windows 先加载 Arrow，再加载 torch。
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model:
        assert digest(args.model / "model.config.json") == "60ec90f5c9b44d9e5c4f2d95a33b871fd55a6e4ccba1d356a54ee51769b02a8e"
        weights = sorted(args.model.glob("*.safetensors"))
        assert weights, "Full SFT model目录缺少safetensors权重"
        base_signature = {"model": str(args.model.resolve()),
            "config_sha256": digest(args.model / "model.config.json"),
            "weight_files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in weights}}
    else:
        base_signature = {"checkpoint_sha256": digest(args.checkpoint),
                          "config_sha256": digest(args.config_dir / "model.config.json")}
    signature = {"data_sha256": data_hashes, "adapters": actual,
                 "base": base_signature,
                 "evaluator_sha256": digest(__file__),
                 "grader_sha256": digest(ROOT / "model/test/test_code/evaluate_agent_v5.py"),
                 "reward_sha256": digest(ROOT / "model/trainer/agent_rewards.py"),
                 "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
                 "dtype": "base BF16, adapter FP32, BF16 autocast", "decode": "greedy",
                 "limits": {"rag": 512, "other": 256}, "cases_per_variant": len(cases)}
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] == signature
    else:
        save(manifest_path, {"signature": signature, "train": False,
             "scope": "156题新家族留出+165题开发回归；固定证据，不执行检索、工具或数据库。"})
        save(args.output_dir / "cases.json", cases)
    tokenizer = AutoTokenizer.from_pretrained(adapters["v5"], local_files_only=True)
    for path in adapters.values():
        other = AutoTokenizer.from_pretrained(path, local_files_only=True)
        assert tokenizer.get_vocab() == other.get_vocab() and tokenizer.chat_template == other.chat_template
    if args.model:
        base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                   dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        metadata = {"source": str(args.model), "config": base.config.to_dict()}
        base_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        assert tokenizer.get_vocab() == base_tokenizer.get_vocab() and tokenizer.chat_template == base_tokenizer.chat_template
    else:
        from model.test.test_code.evaluate_qwen_sft import load_model
        base, metadata = load_model(args.checkpoint, args.config_dir, "cuda", torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapters["v5"]), is_trainable=False,
                                    autocast_adapter_dtype=True)
    for name in ("grpo50", "v6"):
        model.load_adapter(str(adapters[name]), adapter_name=name, is_trainable=False,
                           autocast_adapter_dtype=True)
    save(args.output_dir / "base_metadata.json", metadata)
    print("MODEL_LOADED", "allocated MiB", torch.cuda.memory_allocated() / 2**20, flush=True)
    started = time.monotonic()
    results = {}
    with torch.inference_mode():
        for variant in args.variants:
            model.set_adapter("default" if variant == "v5" else variant)
            model.requires_grad_(False)
            model.eval()
            target = args.output_dir / f"{variant}.jsonl"
            completed = rows(target) if target.exists() else []
            assert [{"suite": r["suite"], "case": r["case"]} for r in completed] == cases[:len(completed)]
            with target.open("a", encoding="utf-8") as stream:
                for index, item in enumerate(cases[len(completed):], len(completed) + 1):
                    case = item["case"]
                    ids = tokenizer.apply_chat_template(case["messages"][:-1], tokenize=True,
                              add_generation_prompt=True, enable_thinking=False)
                    limit = 512 if case["mode"].endswith("_rag") else 256
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = greedy(model, tokenizer, ids, limit, "cuda")
                    result = {**item, "variant": variant, **output,
                              **grade(case, output["text"], output["stop_reason"] == "eos")}
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed.append(result)
                    if index == 1 or index % 10 == 0 or index == len(cases):
                        print(variant, f"{index}/{len(cases)}", item["suite"],
                              "elapsed", round(time.monotonic() - started, 1), flush=True)
            results[variant] = completed
            save(args.output_dir / f"{variant}_summary.json", summarize(completed))
    for name, path in adapters.items():
        assert digest(path / "adapter_model.safetensors") == actual[name]
    if args.model:
        assert {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in weights} == base_signature["weight_files"]
    for name in ("v5", "grpo50", "v6"):
        path = args.output_dir / f"{name}.jsonl"
        if path.exists():
            results[name] = rows(path)
    if all(len(results.get(name, [])) == len(cases) for name in adapters):
        save(args.output_dir / "comparison.json", {
            "v5_to_v6": compare(results["v5"], results["v6"]),
            "grpo50_to_v6": compare(results["grpo50"], results["v6"])})
        save(args.output_dir / "completed.json", {"cases_per_variant": len(cases),
            "variants": list(adapters), "adapter_files_unchanged": True, "training_performed": False})
    print("EVAL_FINISHED", args.variants, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/qwen-fullsft/checkpoint.pt")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "runs/qwen-base-reference")
    parser.add_argument("--model", type=Path, help="服务器已有Full SFT导出的model目录；提供后不读取本地checkpoint")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/lora-v6-eval-20261030")
    parser.add_argument("--variants", nargs="+", choices=("v5", "grpo50", "v6"), default=["v6", "v5", "grpo50"])
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
