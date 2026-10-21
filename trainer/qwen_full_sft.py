"""Qwen 全参数 SFT 入口，以及两个 Qwen 入口共用的数据与训练流程。

运行：python -m trainer.qwen_full_sft
数据、模型、训练循环均可通过命令行覆盖；默认首次对照训练一轮。
损失、单步训练、有效 token 加权梯度累积和验证均复用本项目实现。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from tempfile import NamedTemporaryFile

# 同时支持 python -m trainer.qwen_full_sft 和直接运行该文件。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets  # 与现有入口一致，Windows 先加载 Arrow。
import torch
from torch import nn
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from dataset.dataset import QwenSFTEncoder
from trainer.SFTtrain import next_token_loss, train_step
from trainer.trainer import evaluate, train_one_epoch
from trainer.plot_loss import LossWindow, plot_loss


MODEL = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DATASET = "llamafactory/alpaca_gpt4_zh"
DATA_REVISION = "065394ee242c43928298de8f43a8748ffd16f3e3"


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QwenLogitsModel(nn.Module):
    """让 HF/PEFT 模型返回 Tensor，接通现有 train_step 与 evaluate。"""

    def __init__(self, causal_lm):
        super().__init__()
        self.causal_lm = causal_lm

    def forward(self, input_ids, attention_mask=None):
        return self.causal_lm(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        ).logits.float()


def build_parser(method="full"):
    parser = argparse.ArgumentParser(description=f"Qwen {method} SFT：原生 PyTorch 训练循环")
    parser.set_defaults(method=method)
    parser.add_argument("--model", default=MODEL, help="HF 模型 ID 或本地模型目录")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--data-path", help="本地 JSON/JSONL；省略时下载固定版本中文 Alpaca-GPT4")
    parser.add_argument("--output-dir", default=f"runs/qwen_{method}_sft")
    parser.add_argument("--model-cache", default="runs/model-cache")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5 if method == "full" else 1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"),
                        default="adamw8bit" if method == "full" else "adamw",
                        help="24GB 全参数训练使用 adamw8bit；模型参数仍为 FP32")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=43)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--log-steps", type=int, default=20)
    # 全参数 AdamW 断点较大；默认只在结束/短试跑停止时保存一次。
    parser.add_argument("--save-steps", type=int, default=0 if method == "full" else 200)
    parser.add_argument("--max-updates", type=int, help="本次最多更新几次；保留完整余弦调度计划")
    parser.add_argument("--max-samples", type=int, default=0, help="固定抽样做试跑；0 表示使用全部数据")
    parser.add_argument("--resume", help="本入口保存的 checkpoint.pt，不是 MiniMind 断点")
    parser.add_argument("--save-training-state", action=argparse.BooleanOptionalAction,
                        default=method == "lora", help="保存优化器和续训断点；全参数默认关闭，节省磁盘")
    parser.add_argument("--prepare-only", action="store_true", help="只下载 tokenizer 并准备数据，不加载模型权重")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser


def build_optimizer(model, args):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw8bit":
        if parameters[0].device.type != "cuda":
            raise ValueError("adamw8bit 需要 CUDA；CPU 检查请指定 --optimizer adamw")
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.AdamW(parameters, lr=args.learning_rate,
                             weight_decay=args.weight_decay, foreach=False)


def prepare_data(args, tokenizer):
    if args.data_path:
        source = project_path(args.data_path)
    else:
        source = Path(hf_hub_download(
            DATASET, "alpaca_gpt4_data_zh.json", repo_type="dataset", revision=DATA_REVISION,
            local_dir=ROOT / "dataset/datasets/alpaca_gpt4_zh",
        ))
    raw = datasets.load_dataset(
        "json", data_files=str(source), split="train",
        cache_dir=str(ROOT / "dataset/datasets/cache/qwen"),
    )
    source_count = len(raw)
    if args.max_samples:
        raw = raw.shuffle(seed=args.seed).select(range(min(args.max_samples, len(raw))))
    encoded = raw.map(
        QwenSFTEncoder(tokenizer, args.max_length, args.split_seed, args.val_fraction),
        remove_columns=raw.column_names, desc="Qwen 模板编码与监督对齐",
    )
    valid = encoded.filter(lambda row: any(x != -100 for x in row["labels"][1:]), desc="过滤无有效答案样本")
    train = valid.filter(lambda row: not row["validation"]).remove_columns("validation")
    val = valid.filter(lambda row: row["validation"]).remove_columns("validation")
    if not len(train) or not len(val):
        raise ValueError("训练或验证划分为空，请增加样本数或调整验证比例")
    info = {
        "source": str(source), "source_sha256": file_sha256(source),
        "repository": None if args.data_path else DATASET,
        "revision": None if args.data_path else DATA_REVISION,
        "source_records": source_count, "selected_records": len(raw),
        "rejected_no_targets": len(encoded) - len(valid), "train": len(train), "val": len(val),
        "train_fingerprint": train._fingerprint, "val_fingerprint": val._fingerprint,
    }
    print("数据摘要:", json.dumps(info, ensure_ascii=False), flush=True)
    return train, val, info


def save_training_state(path, causal_lm, optimizer, progress, signature, method):
    if method == "lora":
        from peft import get_peft_model_state_dict
        weights = get_peft_model_state_dict(causal_lm, save_embedding_layers=False)
    else:
        weights = causal_lm.state_dict()
    state = {
        "weights": {k: v.detach().cpu() for k, v in weights.items()},
        "optimizer": optimizer.state_dict(), "progress": progress, "signature": signature,
        "random_state": random.getstate(), "torch_state": torch.get_rng_state(),
        "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".qwen-", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            torch.save(state, f)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_training(args, lora_config=None):
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.max_length, args.log_steps) <= 0:
        raise ValueError("epochs、batch、累积步数、长度和日志间隔必须为正")
    if not 0 < args.val_fraction < 1 or args.max_samples < 0 or args.save_steps < 0:
        raise ValueError("检查 val_fraction、max_samples 和 save_steps")
    if args.resume and not args.save_training_state:
        raise ValueError("恢复训练请同时使用 --save-training-state")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("max-updates 必须为正整数")
    device = torch.device(args.device)
    if args.dtype == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 模式需要支持 BF16 的 CUDA GPU；CPU 测试请使用 --dtype float32 --device cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    export = output / ("adapter" if args.method == "lora" else "model")
    if (checkpoint_path.exists() or export.exists()) and not args.resume and not args.prepare_only:
        raise FileExistsError("输出目录已有权重，请指定 --resume 或更换 --output-dir")

    cache = str(project_path(args.model_cache))
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, cache_dir=cache, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    train_data, val_data, data_info = prepare_data(args, tokenizer)
    if args.prepare_only:
        return
    causal_lm = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, cache_dir=cache,
        dtype=torch.float32, attn_implementation="sdpa",
    )
    causal_lm.config.use_cache = False
    if lora_config is not None:
        from peft import get_peft_model
        causal_lm = get_peft_model(causal_lm, lora_config)
    if not args.no_gradient_checkpointing:
        causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = QwenLogitsModel(causal_lm).to(device)
    optimizer = build_optimizer(model, args)
    signature = {k: v for k, v in vars(args).items() if k not in {
        "output_dir", "model_cache", "data_path", "resume", "prepare_only", "max_updates", "log_steps", "save_steps",
    }}
    signature.update(source_sha256=data_info["source_sha256"],
                     train_fingerprint=data_info["train_fingerprint"], val_fingerprint=data_info["val_fingerprint"])
    completed, global_step, epoch_state = 0, 0, None
    if args.resume:
        state = torch.load(project_path(args.resume), map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("恢复训练需要保持模型、数据及训练超参数一致")
        if args.method == "lora":
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(causal_lm, state["weights"])
        else:
            causal_lm.load_state_dict(state["weights"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        completed, global_step = state["progress"]["completed_epochs"], state["progress"]["global_step"]
        epoch_state = state["progress"].get("epoch_state")
        random.setstate(state["random_state"])
        torch.set_rng_state(state["torch_state"])
        if device.type == "cuda" and state["cuda_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_state"])
        del state
        print(f"恢复训练: completed_epochs={completed}, global_step={global_step}")

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    train_generator = torch.Generator()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=train_generator, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(0), collate_fn=collator, num_workers=0)
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    manifest = {"arguments": vars(args), "data": data_info,
                "model_commit": getattr(causal_lm.config, "_commit_hash", None),
                "trainable_parameters": trainable, "total_parameters": total,
                "torch": str(torch.__version__)}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"model={args.model}; method={args.method}; trainable={trainable}/{total}; device={device}", flush=True)
    print(f"optimizer={args.optimizer}; parameters=FP32; compute={args.dtype}; gradient_checkpointing={not args.no_gradient_checkpointing}")
    print(f"batch={args.batch_size}; accumulation={args.accumulation_steps}; max_length={args.max_length}; epochs={args.epochs}")
    print("结果目录:", output, flush=True)
    metrics = output / "metrics.jsonl"

    def record(value):
        with metrics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    if not args.resume:
        baseline = evaluate(model, val_loader, device, amp_dtype=amp_dtype, loss_fn=next_token_loss)
        record({"event": "initial_validation", **baseline})
        print(f"训练前验证 loss={baseline['loss']:.6f}", flush=True)
    if completed >= args.epochs:
        print("断点已完成目标训练轮数。")
        return

    started = time.perf_counter()
    processed_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(completed, args.epochs):
        train_generator.manual_seed(args.seed + epoch)
        prior_steps = epoch_state["steps"] if epoch_state else 0
        prior_tokens = epoch_state["valid_tokens"] if epoch_state else 0
        epoch_start_global = global_step - prior_steps
        loss_window = LossWindow(epoch_state)

        def on_update(state):
            step = epoch_start_global + state["steps"]
            last_trial_update = args.max_updates is not None and state["steps"] - prior_steps >= args.max_updates
            if step == 1 or step % args.log_steps == 0 or state["micro_steps"] == len(train_loader) or last_trial_update:
                mean_loss = state["loss_sum"] / state["valid_tokens"]
                window_loss = loss_window.update(state)
                print(f"epoch={epoch + 1} batch={state['micro_steps']}/{len(train_loader)} "
                      f"global_step={step} loss={mean_loss:.6f} window_loss={window_loss:.6f} "
                      f"lr={optimizer.param_groups[0]['lr']:.8f}", flush=True)
                record({"event": "update", "epoch": epoch + 1, "global_step": step,
                        "mean_loss": mean_loss, "window_loss": window_loss,
                        "valid_tokens": state["valid_tokens"], "loss_sum":state["loss_sum"],
                        "learning_rate": optimizer.param_groups[0]["lr"]})
            if args.save_training_state and args.save_steps and step % args.save_steps == 0 and state["micro_steps"] < len(train_loader):
                save_training_state(checkpoint_path, causal_lm, optimizer,
                    {"completed_epochs": epoch, "global_step": step, "epoch_state": state}, signature, args.method)

        result = train_one_epoch(
            model, optimizer, train_loader, device, accumulation_steps=args.accumulation_steps,
            amp_dtype=amp_dtype, grad_clip=args.grad_clip, base_lr=args.learning_rate,
            total_micro_steps=args.epochs * len(train_loader), epoch_index=epoch,
            start_state=epoch_state, on_update=on_update, limit_updates=args.max_updates,
            train_step_fn=train_step,
        )
        global_step = epoch_start_global + result["steps"]
        processed_tokens += result["valid_tokens"] - prior_tokens
        if args.max_updates is not None:
            args.max_updates -= result["steps"] - prior_steps
        epoch_state = None if result["finished"] else {
            k: result[k] for k in ("loss_sum", "valid_tokens", "steps", "micro_steps")
        }
        completed = epoch + 1 if result["finished"] else epoch
        if result["finished"]:
            validation = evaluate(model, val_loader, device, amp_dtype=amp_dtype, loss_fn=next_token_loss)
            record({"event": "epoch", "completed_epochs": completed,
                    "global_step": global_step, "train": result, "validation": validation})
            print(f"epoch={completed}/{args.epochs} global_step={global_step} train_loss={result['loss']:.6f} "
                  f"val_loss={validation['loss']:.6f}", flush=True)
        if not result["finished"] or (args.max_updates is not None and args.max_updates <= 0):
            break
    if args.save_training_state:
        save_training_state(checkpoint_path, causal_lm, optimizer,
            {"completed_epochs": completed, "global_step": global_step, "epoch_state": epoch_state}, signature, args.method)
    if args.method == "lora":
        causal_lm.save_pretrained(export, safe_serialization=True, save_embedding_layers=False)
    else:
        causal_lm.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_record = {"event": "run_end", "completed_epochs": completed, "global_step": global_step,
                    "finished": completed >= args.epochs, "train_tokens_this_run": processed_tokens,
                    "loop_validation_export_seconds": elapsed,
                    "supervised_tokens_per_second": processed_tokens / elapsed}
    if device.type == "cuda":
        final_record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"峰值显存 MiB={final_record['peak_allocated_mib']:.1f}")
    record(final_record)
    print(f"本次耗时（含验证与导出）={elapsed:.1f}s; completed_epochs={completed}; global_step={global_step}")
    print("模型/adapter:", export, flush=True)
    if args.save_training_state:
        print("训练断点:", checkpoint_path, flush=True)
    else:
        print("本次未保存优化器断点；导出权重可推理，但不能精确恢复中途训练。", flush=True)
    plot_loss(metrics, title=f"Qwen {args.method.upper()} SFT")


def train(args=None):
    run_training(build_parser("full").parse_args(args))


if __name__ == "__main__":
    train()
