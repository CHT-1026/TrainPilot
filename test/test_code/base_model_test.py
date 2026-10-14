"""基础语言模型的交互式续写测试：输入文本，显示模型生成的后续内容。"""
import argparse
import math
from pathlib import Path
import sys

# 支持直接运行 python test/test_code/base_model_test.py。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer

from config.config import TrainPilotconfig
from model.model import TrainPilotForCausalLM


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_base_model(checkpoint_path: Path, device: torch.device) -> TrainPilotForCausalLM:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到训练权重：{checkpoint_path}\n"
            "请先从服务器下载 checkpoint.pt，或通过 --checkpoint 指定文件。"
        )
    # mmap 延迟读取张量；交互测试只使用模型权重，不加载优化器到模型中。
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("需要本项目保存的、包含 model_config 和 model_state_dict 的 checkpoint")
    config = TrainPilotconfig(**checkpoint["model_config"])
    model = TrainPilotForCausalLM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def generate_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须是正整数")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须为非负有限数值；0 表示贪心选择")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k 必须为非负整数；0 表示不过滤候选")
    bos_id = tokenizer.token_to_id("<|im_start|>")
    eos_id = tokenizer.token_to_id("<|im_end|>")
    pad_id = tokenizer.token_to_id("<|endoftext|>")
    if any(token_id is None for token_id in (bos_id, eos_id, pad_id)):
        raise ValueError("tokenizer 缺少 BOS、EOS 或 PAD")

    # 推理前缀只加 BOS，不加 EOS 或 padding；模型从正文末尾继续预测。
    prompt_ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
    if len(prompt_ids) >= model.config.max_seq_len:
        raise ValueError(f"输入过长，请缩短到少于 {model.config.max_seq_len - 1} 个正文 token")
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = []
    model.eval()

    for _ in range(max_new_tokens):
        # 先使用完整前缀重算，便于阅读；此版本没有启用 KV cache。
        if tokens.shape[1] >= model.config.max_seq_len:
            break
        scores = model(tokens)[0, -1, :].float().clone()
        if not torch.isfinite(scores).all().item():
            raise RuntimeError("模型输出出现 NaN 或 Inf")
        scores[pad_id] = float("-inf")
        scores[bos_id] = float("-inf")
        if temperature == 0:
            next_id = int(scores.argmax().item())
        else:
            scores /= temperature
            if top_k > 0:
                cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                scores = scores.masked_fill(scores < cutoff, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
        if next_id == eos_id:
            break
        generated_ids.append(next_id)
        tokens = torch.cat((tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="循环输入文本，测试预训练基础模型的续写")
    parser.add_argument("--checkpoint", default="runs/pre_train1-20261012/checkpoint.pt")
    parser.add_argument("--tokenizer", default="dataset/tokenizer.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用 CUDA，可以指定 --device cpu")
    torch.set_num_threads(2)
    model = load_base_model(resolve_path(args.checkpoint), device)
    tokenizer = Tokenizer.from_file(str(resolve_path(args.tokenizer)))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("tokenizer 词表大小与模型不一致")
    torch.manual_seed(args.seed)
    print(f"加载完成：设备={device}，参数数目={sum(p.numel() for p in model.parameters())}")
    print("每次输入作为新的文本开头。输入 /exit 退出，空输入跳过。")
    print("可以试试：人工智能是一种、太阳系中、机器学习的基本原理是")
    try:
        while True:
            prompt = input("\n输入> ")
            if prompt.strip().lower() in ("/exit", "exit", "quit"):
                break
            if not prompt.strip():
                continue
            try:
                continuation = generate_text(
                    model, tokenizer, prompt, device,
                    args.max_new_tokens, args.temperature, args.top_k,
                )
                print("续写>", continuation or "[模型直接结束，没有生成正文]")
            except (ValueError, RuntimeError) as error:
                print("本次生成失败：", error)
    except (EOFError, KeyboardInterrupt):
        pass
    print("测试结束。")


if __name__ == "__main__":
    main()
