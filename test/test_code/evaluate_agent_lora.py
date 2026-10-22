"""本地对照评测：独立业务测试集、旧通用题及可选的 Agent 模型接口。

结果仅写入 runs；不训练模型，不生成 Markdown，不调用外部模型服务。
"""
import argparse
from contextlib import nullcontext
from http.server import HTTPServer
import json
import math
from pathlib import Path
import time

import datasets  # Windows 先加载 Arrow。
import torch

from dataset.dataset import QwenSFTEncoder
from test.test_code.evaluate_qwen_sft import (
    answer_text, build_cases, generate, grade, sha256, supervised_stats,
)
from test.test_code.qwen_agent_api import ROOT, load_pair, make_handler


def score_business(case, text):
    if case["mode"] == "intent_parse":
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            payload = None
        valid = isinstance(payload, dict)
        info = payload.get("extracted_info") if valid else None
        expected = case["expected"]
        return {
            "strict_json": valid,
            "intent_ok": valid and payload.get("intent") == expected["intent"],
            "fields_exact": isinstance(info, dict) and info == expected["extracted_info"],
            "extra_field_keys": sorted(set(info) - set(expected["extracted_info"])) if isinstance(info, dict) else [],
            "passed": payload == expected,
        }
    if case["category"] == "identity":
        return {"passed": None, "mentions_trainpilot": "trainpilot" in answer_text(text).lower(),
                "grading": "身份与职责题保留人工评阅；缺失信息或确认边界问题不要求重复自称。"}
    return {"passed": answer_text(text) == case["messages"][-1]["content"].strip()}


def summarize(rows):
    business = [r for r in rows if r["suite"] == "business"]
    structured = [r for r in business if r["case"]["mode"] == "intent_parse"]
    general = [r for r in rows if r["suite"] == "general" and r["passed"] is not None]
    logic = [r for r in general if r["case"]["group"] == "logic"]
    categories = {}
    for row in business:
        item = categories.setdefault(row["case"]["category"], {"passed": 0, "total": 0, "auto_scored": 0, "manual_review": 0})
        item["total"] += 1
        item["passed"] += bool(row["passed"])
        item["auto_scored"] += row["passed"] is not None
        item["manual_review"] += row["passed"] is None
    groups = {}
    for row in general:
        item = groups.setdefault(row["case"]["group"], {"passed": 0, "total": 0})
        item["total"] += 1
        item["passed"] += bool(row["passed"])
    return {
        "business": {"passed": sum(bool(r["passed"]) for r in business), "total": len(business),
                     "auto_scored": sum(r["passed"] is not None for r in business),
                     "manual_review": sum(r["passed"] is None for r in business),
                     "structured_total": len(structured),
                     "strict_json": sum(r["strict_json"] for r in structured),
                     "intent_correct": sum(r["intent_ok"] for r in structured),
                     "fields_exact": sum(r["fields_exact"] for r in structured),
                     "structured_exact": sum(r["passed"] for r in structured),
                     "samples_with_extra_fields": sum(bool(r["extra_field_keys"]) for r in structured),
                     "categories": categories},
        "general": {"passed": sum(r["passed"] for r in general), "total": len(general), "groups": groups,
                    "logic_content_correct_ignoring_terminal_punctuation": sum(
                        answer_text(r["raw_text"]).rstrip("。.!！") == r["case"]["expected"] for r in logic),
                    "logic_content_total": len(logic)},
        "truncated": sum(r["stop_reason"] == "max_new_tokens" for r in rows),
        "generation_seconds": sum(r["seconds"] for r in rows),
        "note": "有限模板测试与小规模旧题；不等于公开基准，开放题保留原文供阅读，不自动算通过。",
    }


@torch.inference_mode()
def evaluate(model, tokenizer, args, output):
    data_dir = ROOT / "dataset/datasets/trainpilot_agent/v2"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, item in manifest["outputs"].items():
        if sha256(data_dir / item["file"]) != item["sha256"]:
            raise ValueError(f"数据校验失败: {name}")
    business = [json.loads(line) for line in (data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    general = [case for case in build_cases() if case["group"] != "tool_diagnostic"]
    (output / "cases.json").write_text(json.dumps({"business": business, "general": general}, ensure_ascii=False, indent=2), encoding="utf-8")
    encoder = QwenSFTEncoder(tokenizer, 1536)
    summaries = {}
    for variant in ("sft", "lora"):
        rows = []
        context = model.disable_adapter() if variant == "sft" else nullcontext()
        with context, (output / f"{variant}_generations.jsonl").open("w", encoding="utf-8") as stream:
            loss_sum = correct = targets = 0
            started = time.perf_counter()
            for index, case in enumerate(business, 1):
                messages = case["messages"][:-1]
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                result = generate(model, tokenizer, ids, "cuda", args.max_new_tokens)
                row = {"suite": "business", "variant": variant, "case": case, **result,
                       **score_business(case, result["text"])}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                encoded = encoder(case)
                inputs = torch.tensor([encoded["input_ids"]], device="cuda")
                labels = torch.tensor([encoded["labels"]], device="cuda")
                logits = model(input_ids=inputs, use_cache=False).logits
                loss, matched, count = supervised_stats(logits, labels)
                loss_sum += loss
                correct += matched
                targets += count
                del inputs, labels, logits
                if index == 1 or index % 10 == 0:
                    print(f"{variant} business={index}/{len(business)} passed={sum(bool(r['passed']) for r in rows)}", flush=True)
            numeric = {"samples": len(business), "supervised_tokens": targets,
                       "loss": loss_sum / targets, "ppl": math.exp(loss_sum / targets),
                       "teacher_forced_token_accuracy": correct / targets}
            for index, case in enumerate(general, 1):
                messages = case.get("history", []) + [{"role": "user", "content": case["prompt"]}]
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                result = generate(model, tokenizer, ids, "cuda", args.max_new_tokens)
                row = {"suite": "general", "variant": variant, "case": case, **result,
                       "passed": grade(case, result["raw_text"])}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if index == 1 or index % 10 == 0:
                    print(f"{variant} general={index}/{len(general)}", flush=True)
            summaries[variant] = {**summarize(rows), "test_numeric": numeric,
                                  "evaluation_seconds": time.perf_counter() - started}
        (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SUMMARY " + json.dumps({variant: summaries[variant]}, ensure_ascii=False), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/qwen-fullsft/checkpoint.pt")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "runs/qwen-base-reference")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "runs/trainpilot_agent_lora_v2/adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/lora-eval-20261022")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--serve", action="store_true", help="评测完成后开放两组模型供真实 Agent 测试")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("已有评测结果，请使用新的 output-dir")
    model, tokenizer, metadata = load_pair(args.checkpoint, args.config_dir, args.adapter_dir)
    metadata.update(max_new_tokens=args.max_new_tokens, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    test_sha256=sha256(ROOT / "dataset/datasets/trainpilot_agent/v2/test.jsonl"),
                    control="同一内存中的BF16底座，分别disable_adapter与启用适配器；不合并权重。")
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluate(model, tokenizer, args, args.output_dir)
    metadata["checkpoint_sha256_after"] = sha256(args.checkpoint)
    metadata["adapter_sha256_after"] = sha256(args.adapter_dir / "adapter_model.safetensors")
    if metadata["checkpoint_sha256_after"] != metadata["checkpoint_sha256"] or metadata["adapter_sha256_after"] != metadata["adapter_sha256"]:
        raise ValueError("评测期间权重文件发生变化")
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.serve:
        def infer(messages, limit, enabled):
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            context = nullcontext() if enabled else model.disable_adapter()
            with context:
                result = generate(model, tokenizer, ids, "cuda", min(limit, args.max_new_tokens))
            with (args.output_dir / "agent_model_requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"variant": "lora" if enabled else "sft", "messages": messages, **result}, ensure_ascii=False) + "\n")
            return result
        handler = make_handler(lambda m, n: infer(m, n, False), "trainpilot-sft",
                               {"trainpilot-lora": lambda m, n: infer(m, n, True)})
        print(f"READY http://127.0.0.1:{args.port}/v1 models=trainpilot-sft,trainpilot-lora", flush=True)
        with HTTPServer(("127.0.0.1", args.port), handler) as server:
            server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止评测接口。")
