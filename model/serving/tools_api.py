"""冻结适配器的工具接口；与原文本API兼容，不训练、不切换权重。"""
import argparse
from contextlib import nullcontext
from http.server import HTTPServer
import json
from pathlib import Path

from model.serving.api import ROOT, load_pair, make_handler
from model.serving.tool_protocol import template_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/qwen-fullsft/checkpoint.pt")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "runs/qwen-base-reference")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "runs/tool_v7_eval_20261105/snapshot/runs/trainpilot_tool_lora_v7/adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/local-agent-v7")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--expected-adapter-sha256", help="启动前核对固定适配器")
    args = parser.parse_args()
    if args.expected_adapter_sha256:
        import hashlib
        actual = hashlib.sha256((args.adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
        if actual != args.expected_adapter_sha256:
            raise ValueError("适配器 SHA256 与固定部署配置不符")
    import torch
    from model.serving.weights import generate
    torch.set_num_threads(2)
    model, tokenizer, metadata = load_pair(args.checkpoint, args.config_dir, args.adapter_dir)
    if "tools" not in tokenizer.chat_template or "<tool_call>" not in tokenizer.chat_template:
        raise ValueError("当前tokenizer模板不支持Qwen工具调用")
    metadata.update(max_new_tokens=args.max_new_tokens, tool_calls=True, tool_protocol="qwen3-hermes+bare-json-v1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def infer(messages, max_tokens, tools=None, enabled=True):
        ids = tokenizer.apply_chat_template(template_messages(messages), tools=tools,
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        limit = min(max_tokens, args.max_new_tokens)
        if len(ids) + limit > model.config.max_position_embeddings:
            raise ValueError("工具对话超过模型上下文上限")
        context = nullcontext() if enabled else model.disable_adapter()
        with context, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = generate(model, tokenizer, ids, "cuda", limit)
        with (args.output_dir / "model_requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"variant": "lora" if enabled else "sft", "messages": messages,
                                     "tools": tools, **result}, ensure_ascii=False) + "\n")
        print(f"input={len(ids)} output={result['generated_tokens']} tools={bool(tools)} stop={result['stop_reason']}", flush=True)
        return result

    normal = {"trainpilot-lora": lambda messages, limit: infer(messages, limit)}
    tool = {"trainpilot-lora": lambda messages, limit, tools: infer(messages, limit, tools)}
    print(f"READY http://127.0.0.1:{args.port}/v1 adapter={metadata['adapter_sha256']}", flush=True)
    with HTTPServer(("127.0.0.1", args.port), make_handler(
            normal["trainpilot-lora"], "trainpilot-lora", metadata=metadata, tool_inference=tool)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
