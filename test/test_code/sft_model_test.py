"""加载 SFT 权重，循环输入问题并显示回答；不写入测试日志。"""
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import datasets  # Windows 下先加载 pyarrow，再加载 torch。
import torch
from transformers import AutoTokenizer

from base_model_test import load_base_model, resolve_path


# 来自已使用的 MiniMind SFT 语料。这里只给模型描述工具，不执行图片生成。
IMAGE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "根据文本描述生成图片",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "图片描述，详细描述想要生成的图片内容"}},
            "required": ["prompt"],
        },
    },
}]


@torch.inference_mode()
def generate_reply(
    model, tokenizer, question, device, max_new_tokens=256, temperature=0.8, top_k=50,
    *, tools=None, thinking=None, show_prompt=False, return_details=False,
):
    if max_new_tokens <= 0 or top_k < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("生成长度必须为正；temperature、top_k 必须非负")
    # 工具定义交给训练配套模板，按其规定生成 system 中的工具说明。
    messages = [{"role": "user", "content": question}]
    if thinking is None:
        # 模板默认会预填 think；自由生成时仅补 assistant 头，
        # 之后是正文、思考还是工具调用，由模型生成的 token 决定。
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        prompt += tokenizer.bos_token + "assistant\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        # 显式开关仅用于对照实验，默认不进入这个分支。
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, open_thinking=thinking, tools=tools,
        )
    if show_prompt:
        print("\n模型输入（含模板前缀）：\n" + tokenizer.decode(prompt_ids, skip_special_tokens=False))
    budget = min(max_new_tokens, model.config.max_seq_len - len(prompt_ids))
    if budget <= 0:
        raise ValueError(f"问题和聊天模板已占满 {model.config.max_seq_len} token，请缩短问题")

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id, bos_id, pad_id = tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id
    generated = []
    stop_reason = "context_limit" if budget == model.config.max_seq_len - len(prompt_ids) else "max_new_tokens"
    past_key_values = None
    model.eval()
    for _ in range(budget):
        # 首次处理整个问题，随后只传新 token，并复用之前的 K/V。
        hidden, past_key_values = model.model(tokens, past_key_values=past_key_values, use_cache=True)
        scores = model.lm_head(hidden[:, -1, :])[0].float().clone()
        if not torch.isfinite(scores).all().item():
            raise RuntimeError("模型输出包含 NaN 或 Inf")
        scores[[bos_id, pad_id]] = float("-inf")
        if temperature == 0:
            next_id = int(scores.argmax().item())
        else:
            scores /= temperature
            if top_k:
                cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                scores[scores < cutoff] = float("-inf")
            next_id = int(torch.multinomial(torch.softmax(scores, dim=-1), 1).item())
        generated.append(next_id)
        if next_id == eos_id:
            stop_reason = "eos"
            break
        tokens = torch.tensor([[next_id]], dtype=torch.long, device=device)
    # 保留思考、工具调用和 EOS；不从模型输出中提取或清洗“最终回答”。
    text = tokenizer.decode(generated, skip_special_tokens=False)
    if return_details:
        return {"text": text, "input_tokens": len(prompt_ids), "generated_tokens": len(generated),
                "stop_reason": stop_reason, "context_limit": model.config.max_seq_len}
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description="SFT 模型交互问答测试")
    parser.add_argument("--checkpoint", default="runs/sfttrain1/checkpoint .pt")
    parser.add_argument("--tokenizer", default="dataset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 表示贪心生成")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tools", choices=("none", "image"), default="none", help="image 提供 generate_image 测试工具")
    parser.add_argument("--tools-json", help="自定义工具定义 JSON 文件，内容为工具列表，覆盖 --tools")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="默认自由生成；--thinking 预填思考前缀，--no-thinking 预填空 think")
    parser.add_argument("--show-prompt", action="store_true", help="同时显示送入模型的完整聊天模板")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(args.tokenizer), local_files_only=True)
    model = load_base_model(resolve_path(args.checkpoint), device)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer 词表大小与模型不一致")
    tools = IMAGE_TOOLS if args.tools == "image" else None
    if args.tools_json:
        tools = json.loads(resolve_path(args.tools_json).read_text(encoding="utf-8"))
    print(f"SFT 权重已加载：{resolve_path(args.checkpoint)}")
    print(f"设备={device}；每次独立问答；最多生成 {args.max_new_tokens} token。")
    if args.thinking is None:
        print("自由生成：不预填思考内容，不强制调用工具。")
    print("输入 /exit 退出。")
    if tools:
        print("提供给模型的工具：", ", ".join(tool["function"]["name"] for tool in tools))
        print("这里只展示模型生成的调用，不执行工具。")
    try:
        while True:
            question = input("\n你> ").strip()
            if question.lower() in ("/exit", "exit", "quit"):
                break
            if not question:
                continue
            try:
                result = generate_reply(
                    model, tokenizer, question, device,
                    args.max_new_tokens, args.temperature, args.top_k,
                    tools=tools, thinking=args.thinking, show_prompt=args.show_prompt, return_details=True,
                )
                print("模型原始输出>\n" + (result["text"] or "[未生成 token]"))
                reason = {"eos": "模型生成结束标记", "max_new_tokens": "达到生成 token 上限",
                          "context_limit": "达到上下文长度上限"}[result["stop_reason"]]
                print(f"[结束原因：{reason}；输入 {result['input_tokens']} token；"
                      f"生成 {result['generated_tokens']} token；上下文上限 {result['context_limit']}]")
            except (ValueError, RuntimeError) as error:
                print("本次生成失败：", error)
    except (EOFError, KeyboardInterrupt):
        pass
    print("测试结束。")


if __name__ == "__main__":
    main()
