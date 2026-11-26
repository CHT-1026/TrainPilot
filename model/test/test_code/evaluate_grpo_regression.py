"""V5 与 GRPO 的冻结业务/RAG面板对照：只推理，结果写入 runs。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from model.trainer.agent_rewards import intent_reward
from model.test.test_code.evaluate_agent_v5 import grade as grade_rag, summary


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def grade(case, text, ended=True):
    if case["mode"] == "intent_parse":
        score = intent_reward(text, case["expected"])
        result = {"kind": "business", "strict_json": score["format_ok"],
                  "intent_ok": score["intent_ok"], "fields_exact": score["fields_exact"],
                  "passed": bool(score["reward"])}
    else:
        result = grade_rag(case, text)
    # 内容看似完整，但未正常结束的回答不算通过；同时保留细分诊断。
    if not ended:
        if result.get("passed") is not None:
            result["passed"] = False
        if "reference_exact" in result:
            result["reference_exact"] = False
    return result


def compare(before, after):
    left, right = ({r["case"]["id"]: r for r in side} for side in (before, after))
    if len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys():
        raise ValueError("前后回答ID必须唯一且一一对应")
    changes = []
    for key, b in left.items():
        a = right[key]
        if b["case"] != a["case"]:
            raise ValueError("同一ID的评测题目发生变化")
        if b["text"] != a["text"] or b["stop_reason"] != a["stop_reason"]:
            changes.append({"id": key, "mode": a["case"]["mode"], "category": a["case"]["category"],
                            "before": b["text"], "after": a["text"],
                            "before_passed": b.get("passed"), "after_passed": a.get("passed"),
                            "before_stop": b["stop_reason"], "after_stop": a["stop_reason"]})
    return {"cases_per_variant": len(before), "v5": summary(before), "grpo50": summary(after),
            "improved": sum(r["before_passed"] is False and r["after_passed"] is True for r in changes),
            "regressed": sum(r["before_passed"] is True and r["after_passed"] is False for r in changes),
            "changed_outputs": changes,
            "scope": "固定证据下的模型回归；不执行检索、工具或数据库。RAG逐字参考一致率不等于语义准确率；自然回复保留人工评阅。"}


def load_cases(path):
    manifest = read(path.with_suffix(".manifest.json"))
    cases = rows(path)
    if digest(path) != manifest["cases_sha256"]:
        raise ValueError("冻结评测题SHA256不匹配")
    if [c["id"] for c in cases] != manifest["case_ids"] or len(cases) != 165:
        raise ValueError("冻结面板应包含原165条题且顺序一致")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("冻结面板出现重复ID")
    return cases, manifest


def check_inputs(args):
    cases, manifest = load_cases(args.cases)
    adapters = {"v5": args.baseline_adapter, "grpo50": args.policy_adapter}
    for name, path in adapters.items():
        if digest(path / "adapter_model.safetensors") != manifest["adapters"][name]:
            raise ValueError(f"{name}权重不是本轮冻结的adapter")
    if digest(args.model / "model.config.json") != manifest["base_config_sha256"]:
        raise ValueError("底座config与本轮Full SFT底座不一致")
    signature = {"cases_sha256": manifest["cases_sha256"], "adapters": manifest["adapters"],
                 "model": str(args.model.resolve()), "base_config_sha256": manifest["base_config_sha256"],
                 "base_weight_files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                                       for p in sorted(args.model.glob("*.safetensors"))},
                 "dtype": "base BF16, adapters FP32, BF16 autocast", "device": args.device, "decode": "greedy",
                 "limits": {"rag": 512, "other": 256}, "evaluator_sha256": digest(__file__),
                 "reward_sha256": digest(Path(__file__).resolve().parents[3] / "model/trainer/agent_rewards.py"),
                 "rag_grader_sha256": digest(Path(__file__).with_name("evaluate_agent_v5.py"))}
    if not signature["base_weight_files"]:
        raise ValueError("模型目录没有safetensors底座权重")
    return cases, adapters, signature


def greedy(model, tokenizer, ids, limit, device):
    import torch
    eos = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    configured = model.config.eos_token_id
    eos.update(configured if isinstance(configured, list) else [configured])
    eos.discard(None)
    tokens = torch.tensor([ids], device=device)
    cache, generated = None, []
    started = time.monotonic()
    for _ in range(limit):
        output = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
        scores = output.logits[0, -1].float()
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("生成logits非有限")
        chosen = int(scores.argmax().item())
        cache = output.past_key_values
        generated.append(chosen)
        del output, scores
        if chosen in eos:
            break
        tokens = torch.tensor([[chosen]], device=device)
    return {"text": tokenizer.decode(generated, skip_special_tokens=True),
            "raw_text": tokenizer.decode(generated, skip_special_tokens=False), "generated_ids": generated,
            "input_tokens": len(ids), "generated_tokens": len(generated),
            "stop_reason": "eos" if generated[-1] in eos else "max_new_tokens",
            "seconds": time.monotonic() - started}


def run(args):
    cases, adapters, signature = check_inputs(args)
    print("FROZEN_PANEL", dict(Counter(c["mode"] for c in cases)), flush=True)
    if args.check_only:
        print("CHECK_OK: 165题、两份adapter摘要与Full SFT配置匹配；未加载模型。", flush=True)
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "run_manifest.json"
    if record_path.exists():
        if read(record_path)["signature"] != signature:
            raise ValueError("恢复评测的模型、题目、代码或精度不同，请用新结果目录")
    else:
        save(record_path, {"signature": signature, "train": False,
                          "scope": "165题旧业务/RAG固定回归，未使用GRPO最终150题"})
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
    model = PeftModel.from_pretrained(base, str(adapters["v5"]), is_trainable=False, autocast_adapter_dtype=True)
    model.load_adapter(str(adapters["grpo50"]), adapter_name="grpo50", is_trainable=False, autocast_adapter_dtype=True)
    tokenizer = AutoTokenizer.from_pretrained(adapters["v5"], local_files_only=True)
    candidate_tokenizer = AutoTokenizer.from_pretrained(adapters["grpo50"], local_files_only=True)
    if tokenizer.get_vocab() != candidate_tokenizer.get_vocab() or tokenizer.chat_template != candidate_tokenizer.chat_template:
        raise ValueError("前后adapter的词表或聊天模板不同")
    results = {}
    with torch.inference_mode():
        for variant, name in (("v5", "default"), ("grpo50", "grpo50")):
            model.set_adapter(name)
            model.requires_grad_(False)
            model.eval()
            path = args.output_dir / f"{variant}.jsonl"
            completed = rows(path) if path.exists() else []
            expected_prefix = cases[:len(completed)]
            if [r["case"] for r in completed] != expected_prefix:
                raise ValueError("已有结果与冻结题目顺序不一致")
            with path.open("a", encoding="utf-8") as f:
                for i, case in enumerate(cases[len(completed):], len(completed)+1):
                    ids = tokenizer.apply_chat_template(case["messages"][:-1], tokenize=True,
                              add_generation_prompt=True, enable_thinking=False)
                    limit = 512 if case["mode"].endswith("_rag") else 256
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                        output = greedy(model, tokenizer, ids, limit, args.device)
                    item = {"variant": variant, "case": case, **output,
                            **grade(case, output["text"], output["stop_reason"] == "eos")}
                    f.write(json.dumps(item, ensure_ascii=False)+"\n")
                    f.flush()
                    completed.append(item)
                    print(variant, f"{i}/165", case["mode"], "passed=", item.get("passed"), flush=True)
            results[variant] = completed
            save(args.output_dir / f"{variant}_summary.json", summary(completed))
    result = compare(results["v5"], results["grpo50"])
    # 检查输入文件在评测期间没有换成别的版本。
    _, _, current_signature = check_inputs(args)
    if current_signature != signature:
        raise ValueError("评测期间输入文件发生变化")
    save(args.output_dir / "comparison.json", result)
    save(args.output_dir / "completed.json", {"cases_per_variant": 165, "weights_unchanged": True,
                                              "train": False, "comparison": "comparison.json"})
    print("REGRESSION_COMPLETE", json.dumps({k:v for k,v in result.items() if k!="changed_outputs"}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-adapter", type=Path, required=True)
    parser.add_argument("--policy-adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("model/dataset/datasets/trainpilot_agent/grpo_v2_regression/regression_panel.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/grpo_v2_regression_20261029"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
