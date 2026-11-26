"""V6 SFT 交互体验：自然回复、业务解析及固定证据样例；不保存日志或执行业务。"""
import argparse
from contextlib import nullcontext
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
V6_SHA256 = "4d89275137bee224b724f06b2ff9990ab784d19e3ded9e570499b9abfd6407b8"
PROMPT = ROOT / "model/dataset/datasets/trainpilot_agent/intent_prompt_v4.txt"


def make_messages(mode, text, context, history):
    if mode == "intent":
        prompt = PROMPT.read_text(encoding="utf-8").strip()
        content = (prompt + "\n\n给定上下文：\n"
                   + json.dumps(context, ensure_ascii=False)
                   + "\n用户输入：\n" + text)
        return [{"role": "user", "content": content}]
    return ([{"role": "system", "content": "你是 TrainPilot，一个企业业务助手。请根据用户的问题作答。"}]
            + history + [{"role": "user", "content": text}])


def load_examples(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pair(args):
    import datasets  # Windows 下先加载 Arrow。
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    actual = hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    if actual != V6_SHA256:
        raise ValueError("适配器不是已确认的 V6 SFT 权重，请检查 --adapter 路径")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    if args.model:
        base = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=dtype,
            device_map=args.device, low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        base_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    else:
        from model.test.test_code.evaluate_qwen_sft import load_model
        base, _ = load_model(args.checkpoint, args.config_dir, args.device, dtype)
        base_tokenizer = AutoTokenizer.from_pretrained(args.config_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    if (tokenizer.get_vocab() != base_tokenizer.get_vocab()
            or tokenizer.chat_template != base_tokenizer.chat_template):
        raise ValueError("底座与 V6 的词表或聊天模板不同")
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False,
                                    autocast_adapter_dtype=True)
    model.requires_grad_(False)
    model.eval()
    print(f"底座：{args.model or args.checkpoint}\n适配器：{args.adapter}\nV6 SHA256：{actual}")
    print("精度：底座 BF16、适配器 FP32、BF16 autocast" if args.device == "cuda" else "精度：FP32")
    return model, tokenizer


def reply(model, tokenizer, messages, args):
    import torch
    from transformers import TextStreamer

    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False)
    if len(ids) > args.max_input_tokens:
        raise ValueError(f"输入共 {len(ids)} token，超过 {args.max_input_tokens}；请 /reset 或缩短输入")
    tokens = torch.tensor([ids], device=args.device)
    eos = tokenizer.eos_token_id
    stops = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
    if "<|im_end|>" in tokenizer.get_vocab():
        stops.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    stops = sorted(set(stops))
    options = dict(max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                   eos_token_id=stops, pad_token_id=tokenizer.pad_token_id or stops[0],
                   use_cache=True, streamer=TextStreamer(tokenizer, skip_prompt=True,
                                                        skip_special_tokens=False))
    if args.temperature > 0:
        options.update(temperature=args.temperature, top_p=1.0, top_k=0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
    if args.device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    print("模型原始输出>", flush=True)
    with torch.inference_mode(), amp:
        output = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens), **options)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output[0, len(ids):].tolist()
    reason = "eos" if generated and generated[-1] in stops else "达到生成长度上限"
    print(f"[输入 {len(ids)} token；输出 {len(generated)} token；{elapsed:.2f}s；结束：{reason}]\n")
    return tokenizer.decode(generated, skip_special_tokens=True)


HELP = """命令：
  /mode intent   业务解析（默认）：查看 intent/extracted_info，不执行操作
  /mode chat     自然对话：支持最近三轮历史；身份由 system 提示词提供
  /context JSON  设置业务上下文，例如日期、pending_action、history
  /context       显示当前业务上下文
  /reset         清空聊天历史与业务上下文
  /cases         列出固定输入样例（包括 RAG）
  /case 编号     原样推理该样例；不展示参考答案、不改变当前上下文
  /quit          退出；/help 查看帮助
业务上下文由你显式设置，不会根据模型输出自动产生待确认动作。
RAG 样例提供固定检索证据；此入口不连接检索库、工具或数据库。
"""


def interact(infer, args, examples, *, read=input):
    mode, history = args.mode, []
    context = {"current_date": date.today().isoformat(), "history": [], "pending_action": None}
    print(HELP)
    print("当前模式：", mode, "；当前日期：", context["current_date"])
    while True:
        try:
            text = read(f"你[{mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出，权重保持不变。")
            return
        if not text:
            continue
        if text == "/quit":
            return
        if text == "/help":
            print(HELP)
        elif text.startswith("/mode "):
            selected = text.split(maxsplit=1)[1]
            if selected not in ("intent", "chat"):
                print("可选：intent、chat")
                continue
            mode, history = selected, []
            print("当前模式：", mode)
        elif text == "/context":
            print(json.dumps(context, ensure_ascii=False, indent=2))
        elif text.startswith("/context "):
            try:
                updated = json.loads(text.split(maxsplit=1)[1])
                if not isinstance(updated, dict):
                    raise ValueError("上下文应为 JSON 对象")
                context = {"current_date": date.today().isoformat(), "history": [],
                           "pending_action": None, **updated}
                print(json.dumps(context, ensure_ascii=False, indent=2))
            except ValueError as error:
                print("上下文未修改：", error)
        elif text == "/reset":
            history = []
            context = {"current_date": date.today().isoformat(), "history": [], "pending_action": None}
            print("聊天历史与业务上下文已清空。")
        elif text == "/cases":
            for number, item in enumerate(examples, 1):
                print(f"{number:02d}  {item['mode']}  {item['id']}")
        elif text.startswith("/case "):
            try:
                number = int(text.split(maxsplit=1)[1])
                if not 1 <= number <= len(examples):
                    raise ValueError("编号不在样例列表中，请 /cases 查看")
                item = examples[number - 1]
                print("固定模型输入：\n" + json.dumps(item["messages"], ensure_ascii=False, indent=2))
                infer(item["messages"])
            except ValueError as error:
                print(error)
        elif text.startswith("/"):
            print("未知命令，输入 /help 查看。")
        else:
            try:
                answer = infer(make_messages(mode, text, context, history))
                if mode == "chat":
                    history = (history + [{"role": "user", "content": text},
                                          {"role": "assistant", "content": answer}])[-6:]
            except ValueError as error:
                print(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=os.environ.get("TRAINPILOT_SFT_MODEL"))
    parser.add_argument("--adapter", type=Path, default=ROOT / "runs/trainpilot_agent_lora_v6/adapter")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/qwen-fullsft/checkpoint.pt")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "runs/qwen-base-reference")
    parser.add_argument("--cases", type=Path, default=ROOT / "model/test/manual_v6_cases.jsonl")
    parser.add_argument("--mode", choices=("intent", "chat"), default="intent")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="默认贪心，便于复查；可选 0.7 等值观察采样")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0 or args.temperature < 0:
        parser.error("token 上限应为正，temperature 不得为负")
    model, tokenizer = load_pair(args)
    interact(lambda messages: reply(model, tokenizer, messages, args), args, load_examples(args.cases))


if __name__ == "__main__":
    main()
