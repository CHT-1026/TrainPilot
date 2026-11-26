"""V5/V6只读采样诊断；无优化器、无反向传播，不实现新的GRPO训练。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from model.trainer.agent_rewards import intent_reward
from model.test.test_code.evaluate_grpo_regression import digest, greedy, rows, save
from model.test.test_code.evaluate_agent_v5 import grade as reference_diagnostic


ROOT = Path(__file__).resolve().parents[3]


def case_seed(case_id, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{case_id}".encode()).digest()[:4], "little")


def score_answer(case, text, ended):
    if case["mode"] == "intent_parse":
        score = intent_reward(text, case["expected"])
        return {**score, "task_reward": score["reward"] if ended else 0.0}
    details = reference_diagnostic(case, text)
    details.pop("passed", None)
    return {**details, "task_reward": None,
            "scoring_scope": "RAG参考/引用诊断或自然回复人工阅读；不提供语义RL奖励"}


def classify_groups(groups):
    scores = [a["task_reward"] for group in groups for a in group["answers"]]
    if not scores:
        return {"route": "greedy_only_prompt_ablation", "sampled_rewards": None}
    if any(s is None for s in scores):
        return {"route": "manual_reward_design", "sampled_rewards": None}
    truncated = sum(a["stop_reason"] != "eos" for group in groups for a in group["answers"])
    if truncated == len(scores):
        route = "fix_completion_limit_before_training"
    elif all(s == 1 for s in scores):
        route = "stable_under_observed_sampling"
    elif all(s == 0 for s in scores):
        route = "no_success_observed_review_prompt_labels_or_sft"
    else:
        route = "sampled_correct_and_incorrect_rl_candidate"
    return {"route": route, "sampled_rewards": scores, "truncated": truncated,
            "mixed_groups": sum(len(set(a["task_reward"] for a in g["answers"])) > 1 for g in groups),
            "observed_at_least_one_correct": any(s == 1 for s in scores),
            "caveat": "有限采样诊断不是自动训练决策；全错不证明模型永远不会，混合组也不保证RL有效。"}


def aggregate(records):
    result = {}
    for key in sorted({(r["source_split"], r["mode"], r["prompt_variant"]) for r in records}):
        selected = [r for r in records if (r["source_split"], r["mode"], r["prompt_variant"]) == key]
        groups = [g for r in selected for g in r["groups"]]
        answers = [a for r in selected for a in [r["greedy"]] + [a for g in r["groups"] for a in g["answers"]]]
        entry = {"cases": len(selected), "completions": len(answers),
                 "truncated": sum(a["stop_reason"] != "eos" for a in answers),
                 "routes": dict(Counter(r["diagnosis"]["route"] for r in selected)),
                 "generation_seconds": sum(a["seconds"] for a in answers)}
        if key[1] == "intent_parse":
            entry.update(greedy_correct=sum(r["greedy"]["task_reward"] for r in selected),
                         observed_any_correct=sum(bool(r["diagnosis"].get("observed_at_least_one_correct")) for r in selected),
                         mixed_groups=sum(len(set(a["task_reward"] for a in g["answers"])) > 1 for g in groups),
                         groups=len(groups), format_errors=sum(not a["format_ok"] for a in answers))
        result["/".join(key)] = entry
    return result


def sample_one(model, tokenizer, ids, limit, seed):
    import torch
    from transformers import GenerationConfig
    from model.trainer.qwen_grpo import eos_ids
    torch.manual_seed(seed)
    prompt = torch.tensor([ids], device="cuda")
    config = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
        repetition_penalty=1., max_new_tokens=limit, eos_token_id=eos_ids(model, tokenizer),
        pad_token_id=tokenizer.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emitted = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                 generation_config=config)[0, len(ids):].tolist()
    torch.cuda.synchronize()
    return {"text": tokenizer.decode(emitted, skip_special_tokens=True),
            "raw_text": tokenizer.decode(emitted, skip_special_tokens=False), "generated_ids": emitted,
            "input_tokens": len(ids), "generated_tokens": len(emitted),
            "stop_reason": "eos" if emitted and emitted[-1] in eos_ids(model, tokenizer) else "max_new_tokens",
            "seconds": time.monotonic() - started, "sample_seed": seed}


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = ROOT / config["data_directory"]
    capture = json.loads((data_root / "runtime_capture.json").read_text(encoding="utf-8"))
    case_path = data_root / "runtime_cases.jsonl"
    assert digest(case_path) == capture["runtime_cases_sha256"]
    cases = rows(case_path)
    assert len(cases) == 238
    if args.smoke:
        cases = [next(r for r in cases if r["case"]["mode"] == mode)
                 for mode in ("intent_parse", "enterprise_rag", "oa_rag")]
    adapters = {name: ROOT / value["path"] for name, value in config["adapters"].items()}
    for name, path in adapters.items():
        assert digest(path / "adapter_model.safetensors") == config["adapters"][name]["sha256"]
    print("DIAGNOSIS_INPUTS_OK", len(cases), "cases; no training", flush=True)
    if args.check_only: return
    assert digest(args.model / "model.config.json") == config["base_config_sha256"]
    import datasets  # Windows兼容；服务器沿用既有依赖。
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    weight_paths = sorted(args.model.glob("*.safetensors"))
    weight_hashes = {p.name: digest(p) for p in weight_paths}
    assert weight_hashes == config["base_weight_sha256"], "底座不是此前评测的Full SFT权重"
    signature = {"config_sha256": digest(args.config), "cases_sha256": digest(case_path),
        "capture_sha256": digest(data_root / "runtime_capture.json"), "base_weight_sha256": weight_hashes,
        "sampler_sha256": digest(__file__), "reward_sha256": digest(ROOT / "model/trainer/agent_rewards.py"),
        "greedy_helper_sha256": digest(ROOT / "model/test/test_code/evaluate_grpo_regression.py"),
        "reference_grader_sha256": digest(ROOT / "model/test/test_code/evaluate_agent_v5.py"),
        "eos_helper_sha256": digest(ROOT / "model/trainer/qwen_grpo.py"),
        "smoke": args.smoke, "case_ids": [r["id"] for r in cases], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "dtype": "base BF16; adapters FP32; BF16 autocast"}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] == signature
    else: save(manifest_path, {"signature": signature, "training_performed": False})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for path in adapters.values():
        other = AutoTokenizer.from_pretrained(path, local_files_only=True)
        assert tokenizer.get_vocab() == other.get_vocab() and tokenizer.chat_template == other.chat_template
    encoded = {}
    for row in cases:
        ids = tokenizer.apply_chat_template(row["inference_messages"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False)
        if len(ids) > config["max_prompt_tokens"]:
            raise ValueError(f"{row['id']} prompt过长：{len(ids)}，未截断")
        encoded[row["id"]] = ids
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                 dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, str(adapters["v5"]), is_trainable=False, autocast_adapter_dtype=True)
    model.load_adapter(str(adapters["v6"]), adapter_name="v6", is_trainable=False, autocast_adapter_dtype=True)
    results = {}
    with torch.inference_mode():
        for variant in ("v5", "v6"):
            model.set_adapter("default" if variant == "v5" else "v6")
            model.requires_grad_(False)
            model.eval()
            assert not any(p.requires_grad for p in model.parameters())
            path = output / (variant + ".jsonl")
            records = rows(path) if path.exists() else []
            assert [r["id"] for r in records] == [r["id"] for r in cases[:len(records)]]
            with path.open("a", encoding="utf-8") as stream:
                for i, row in enumerate(cases[len(records):], len(records) + 1):
                    case, ids = row["case"], encoded[row["id"]]
                    limit = config["rag_completion_tokens"] if case["mode"].endswith("_rag") else config["other_completion_tokens"]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        greedy_output = greedy(model, tokenizer, ids, limit, "cuda")
                    greedy_output.update(score_answer(case, greedy_output["text"], greedy_output["stop_reason"] == "eos"))
                    groups = []
                    if row["sampling_enabled"]:
                        for seed in config["seeds"]:
                            answers = []
                            for j in range(config["group_size"]):
                                answer = sample_one(model, tokenizer, ids, limit, case_seed(row["id"], seed * 100 + j))
                                answer.update(score_answer(case, answer["text"], answer["stop_reason"] == "eos"))
                                answers.append(answer)
                            groups.append({"seed": seed, "answers": answers,
                                           "unique_texts": len({a["text"] for a in answers})})
                    result = {"id": row["id"], "case_id": case["id"], "variant": variant,
                        "mode": case["mode"], "category": case["category"], "source_split": row["source_split"],
                        "source_file": row["source_file"], "prompt_variant": row["prompt_variant"],
                        "greedy": greedy_output, "groups": groups, "diagnosis": classify_groups(groups)}
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n"); stream.flush()
                    records.append(result)
                    print(variant, f"{i}/{len(cases)}", case["category"], result["diagnosis"]["route"], flush=True)
            results[variant] = records
            save(output / (variant + "_summary.json"), aggregate(records))
    for name, path in adapters.items():
        assert digest(path / "adapter_model.safetensors") == config["adapters"][name]["sha256"]
    assert {p.name: digest(p) for p in weight_paths} == weight_hashes
    changes = [{"id": b["id"], "mode": b["mode"], "source_split": b["source_split"],
                "v5": a["diagnosis"], "v6": b["diagnosis"]}
               for a, b in zip(results["v5"], results["v6"]) if a["diagnosis"] != b["diagnosis"]]
    save(output / "comparison.json", {"changes": changes,
        "scope": "业务有限采样诊断；RAG/自然回答保留人工核验，不能据参考措辞差异自动决定训练。"})
    save(output / "completed.json", {"cases_per_variant": len(cases), "variants": ["v5", "v6"],
         "training_performed": False, "base_and_adapter_files_unchanged": True})
    print("DIAGNOSIS_COMPLETE", len(cases), "cases per variant", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "model/config/agent_diagnosis.json")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())
