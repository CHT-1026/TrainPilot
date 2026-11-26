"""本机 Qwen 评测接口：只读加载已有权重，供原 Agent 调用。

仅监听 127.0.0.1；固定贪心解码，stream 为生成结束后一次发送，不用于测首 token 延迟。
python -m model.serving.api
"""
import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def make_handler(infer, model_name, alternatives=None, metadata=None, tool_inference=None):
    inference_by_model = {model_name: infer, **(alternatives or {})}
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/local/model-info" and metadata is not None:
                info = {"adapter_sha256": metadata.get("adapter_sha256"),
                        "adapter": metadata.get("adapter"), "decode": "greedy"}
                if tool_inference:
                    info.update(tool_calls=True, tool_protocol="qwen3-hermes+bare-json-v1")
                self.send_json(200, info)
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [
                    {"id": name, "object": "model"} for name in inference_by_model]})
            else:
                self.send_json(404, {"error": {"message": "Not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "Not found"}})
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = request.get("model")
            if name not in inference_by_model:
                self.send_json(400, {"error": {"message": "Unknown local model"}})
                return
            try:
                tools = request.get("tools") or []
                if tools and not tool_inference:
                    self.send_json(400, {"error": {"message": "This local API does not support tools"}})
                    return
                if tools and request.get("tool_choice", "auto") != "auto":
                    self.send_json(400, {"error": {"message": "This tool endpoint supports tool_choice=auto"}})
                    return
                if tools:
                    result = tool_inference[name](request["messages"], request.get("max_tokens", 256), tools)
                else:
                    result = inference_by_model[name](request["messages"], request.get("max_tokens", 256))
            except Exception as error:
                print(f"生成失败: {type(error).__name__}: {error}", flush=True)
                self.send_json(500, {"error": {"message": type(error).__name__}})
                return
            base = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": name}
            finish = "stop" if result["stop_reason"] == "eos" else "length"
            message = {"role": "assistant", "content": result["text"]}
            parse_error = None
            protocol_used = None
            if tools and finish != "length":
                from model.serving.tool_protocol import parse_tool_output
                try:
                    message = parse_tool_output(result["text"], bare_names={t["function"]["name"] for t in tools})
                except (ValueError, json.JSONDecodeError) as error:
                    parse_error = str(error)
                if message.get("tool_calls"):
                    finish = "tool_calls"
                    protocol_used = "hermes" if "<tool_call>" in result["text"] else "bare_json_compat"
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                delta = dict(message)
                if delta.get("tool_calls"):
                    delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
                for delta, reason in [(delta, None), ({}, finish)]:
                    event = {**base, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": delta, "finish_reason": reason}]}
                    self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.send_json(200, {**base, "object": "chat.completion", "choices": [
                    {"index": 0, "message": message, "finish_reason": finish}],
                    **({"local_trace": {"raw_output": result.get("raw_text", result["text"]),
                                        "tool_parse_error": parse_error,
                                        "tool_protocol_used": protocol_used}} if tools else {}),
                    "usage": {"prompt_tokens": result["input_tokens"], "completion_tokens": result["generated_tokens"],
                              "total_tokens": result["input_tokens"] + result["generated_tokens"]}})

    return Handler


def load_pair(checkpoint, config_dir, adapter_dir):
    """同一底座加载适配器；disable_adapter 提供数值精度一致的对照组。"""
    from model.serving.weights import load_model, sha256
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
    model, metadata = load_model(checkpoint, config_dir, "cuda", torch.bfloat16)
    metadata.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                    decode="greedy", dtype="bfloat16", gpu=torch.cuda.get_device_name(0))
    if adapter_dir is not None:
        adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
        if (tokenizer.get_vocab() != adapter_tokenizer.get_vocab()
                or tokenizer.chat_template != adapter_tokenizer.chat_template):
            raise ValueError("适配器 tokenizer/模板与本地 Full SFT 底座不一致")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False,
                                          autocast_adapter_dtype=True)
        metadata.update(adapter=str(adapter_dir.resolve()),
                        adapter_sha256=sha256(adapter_dir / "adapter_model.safetensors"),
                        adapter_config=json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8")))
    model.eval()
    return model, tokenizer, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/qwen-fullsft/checkpoint.pt")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "runs/qwen-base-reference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/agent-baseline-20261022")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--adapter-dir", type=Path, help="可选 LoRA 目录；同时开放 trainpilot-sft 与 trainpilot-lora")
    args = parser.parse_args()

    from model.serving.weights import generate
    import torch

    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("加载已有 SFT 权重；不恢复优化器，不下载新模型。", flush=True)
    model, tokenizer, metadata = load_pair(args.checkpoint, args.config_dir, args.adapter_dir)
    metadata.update(max_new_tokens=args.max_new_tokens, note="本机缓冲式 API，不测流式首 token 延迟")
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def infer(messages, max_tokens, enabled=False):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        context = model.disable_adapter() if args.adapter_dir is not None and not enabled else nullcontext()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            result = generate(model, tokenizer, ids, "cuda", min(max_tokens, args.max_new_tokens))
        with (args.output_dir / "model_requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"variant": "lora" if enabled else "sft", "messages": messages, **result}, ensure_ascii=False) + "\n")
        print(f"input={result['input_tokens']} output={result['generated_tokens']} stop={result['stop_reason']}", flush=True)
        return result

    name = "trainpilot-sft"
    alternatives = {"trainpilot-lora": lambda messages, limit: infer(messages, limit, True)} if args.adapter_dir else None
    print(f"READY http://127.0.0.1:{args.port}/v1 model={name}", flush=True)
    with HTTPServer(("127.0.0.1", args.port), make_handler(infer, name, alternatives, metadata)) as server:
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止本机模型接口。")
