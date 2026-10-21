import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import datasets  # Windows 下先加载 pyarrow，再加载 torch。
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.config import TrainPilotconfig
from dataset.dataset import PretrainDataset, SFTDataset, TrainPilotTokenizer, prepare_pretrain_data, prepare_sft_data
from model.model import TrainPilotForCausalLM
from trainer import pretrain, SFTtrain
from trainer.trainer import evaluate, load_checkpoint, save_checkpoint, train_one_epoch
from trainer.plot_loss import LossWindow, plot_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TrainPilot Dense 单卡预训练 / SFT")
    parser.add_argument("--config", default="config/smoke.json", help="JSON 配置文件")
    parser.add_argument("--stage", choices=("pretrain", "sft"), help="覆盖配置中的训练阶段")
    parser.add_argument("--base-checkpoint", help="首次 SFT 的基础权重，和 --resume 区分")
    parser.add_argument("--resume", help="从参数更新后的 checkpoint 恢复")
    parser.add_argument("--epochs", type=int, help="目标总轮数，包含 checkpoint 已完成的轮数")
    parser.add_argument("--device", help="覆盖配置的设备，例如 cpu、cuda 或 cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), help="覆盖训练精度")
    parser.add_argument("--output-dir", help="覆盖结果目录；相对路径基于仓库根目录")
    parser.add_argument("--max-updates", type=int, help="本次最多更新多少次，用于服务器短试跑；保留原调度总长度")
    parser.add_argument("--batch-size", type=int, help="覆盖 micro-batch 大小")
    parser.add_argument("--prepare-data", help="只划分原始 JSONL 语料，不启动训练")
    parser.add_argument("--data-output-dir", help="划分结果目录")
    parser.add_argument("--val-fraction", type=float, default=0.01, help="分组划分的验证比例")
    parser.add_argument("--split-seed", type=int, default=43)
    parser.add_argument("--source-sha256", help="下载文件应具有的 SHA256")
    args = parser.parse_args(argv)
    config_path = resolve_path(args.config)
    with config_path.open(encoding="utf-8") as source:
        config = json.load(source)
    stage = args.stage or config.get("stage", "pretrain")
    config["stage"] = stage
    if stage not in ("pretrain", "sft"):
        raise ValueError("stage 必须为 pretrain 或 sft")
    if args.base_checkpoint is not None:
        config["base_checkpoint"] = args.base_checkpoint
    functions = SFTtrain if stage == "sft" else pretrain

    if args.prepare_data is not None:
        source_path = resolve_path(args.prepare_data)
        output_dir = resolve_path(args.data_output_dir or (
            "dataset/datasets/processed/sft" if stage == "sft" else "dataset/datasets/processed"
        ))
        if stage == "sft":
            tokenizer = AutoTokenizer.from_pretrained(resolve_path(config["data"]["tokenizer_path"]), local_files_only=True)
            if args.source_sha256 and file_sha256(source_path) != args.source_sha256.lower():
                raise ValueError("SFT 原始语料 SHA256 与预期不一致")
            manifest = prepare_sft_data(
                source_path, output_dir, tokenizer, config["model"]["max_seq_len"],
                args.val_fraction, args.split_seed,
            )
        else:
            manifest = prepare_pretrain_data(source_path, output_dir, args.val_fraction, args.split_seed, args.source_sha256)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    # 1. 读取训练配置与命令行覆盖项。
    model_config = TrainPilotconfig(**config["model"])
    training = config["training"]
    for name in ("epochs", "device", "dtype", "output_dir", "batch_size"):
        value = getattr(args, name)
        if value is not None:
            training[name] = value
    training.setdefault("cpu_threads", 2)
    training.setdefault("accumulation_steps", 1)
    training.setdefault("grad_clip", None)
    training.setdefault("lr_schedule", "constant")
    training.setdefault("log_interval", 100)
    training.setdefault("save_interval", 1000)
    if training["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("只支持 float32 或 bfloat16 autocast")
    if training["lr_schedule"] not in ("constant", "cosine"):
        raise ValueError("lr_schedule 必须为 constant 或 cosine")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("max-updates 必须为正整数")

    device = torch.device(training["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("第一版只支持 CPU 或单卡 CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 无法使用 CUDA")
    amp_dtype = torch.bfloat16 if training["dtype"] == "bfloat16" else None
    if amp_dtype is not None and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("此入口的 bfloat16 模式需要支持 BF16 的 CUDA GPU")
    torch.set_num_threads(training["cpu_threads"])
    output_dir = resolve_path(training["output_dir"])
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists() and args.resume is None:
        raise FileExistsError("结果目录已有 checkpoint；请使用 --resume 或更换 --output-dir")

    # 2. 固定初始随机状态，并通过文件摘要记录实际使用的数据。
    random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["seed"])
    paths = {name: resolve_path(value) for name, value in config["data"].items()}
    fingerprints = {name: file_sha256(paths[name]) for name in ("train_path", "val_path")}
    if stage == "sft":
        for name in ("tokenizer.json", "tokenizer_config.json"):
            fingerprints[name] = file_sha256(paths["tokenizer_path"] / name)
    else:
        fingerprints["tokenizer_path"] = file_sha256(paths["tokenizer_path"])
    if paths["train_path"] == paths["val_path"] or fingerprints["train_path"] == fingerprints["val_path"]:
        raise ValueError("训练集与验证集不能是同一个文件或内容完全相同的文件")

    # 3. DataLoader 负责组 batch；只打乱训练集，保留最后一个不足整批的 batch。
    if stage == "sft":
        tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_path"], local_files_only=True)
        train_data = SFTDataset(paths["train_path"], tokenizer, max_length=model_config.max_seq_len)
        val_data = SFTDataset(paths["val_path"], tokenizer, max_length=model_config.max_seq_len)
        vocab_size = len(tokenizer)
    else:
        tokenizer = TrainPilotTokenizer(paths["tokenizer_path"], model_config.max_seq_len)
        train_data = PretrainDataset(paths["train_path"], tokenizer)
        val_data = PretrainDataset(paths["val_path"], tokenizer)
        vocab_size = tokenizer.vocab_size
    if vocab_size != model_config.vocab_size:
        raise ValueError("模型 vocab_size 与实际 tokenizer 词表长度不一致")
    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    # 独立 generator 使重建/跳过 batch 不消耗模型 Dropout 的全局随机状态。
    train_generator = torch.Generator()
    train_loader = DataLoader(train_data, shuffle=True, generator=train_generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, generator=torch.Generator().manual_seed(0), **loader_options)

    # 4. 先将模型放到设备，再创建优化器。恢复必须在模型初始化之后。
    model = TrainPilotForCausalLM(model_config).to(device=device, dtype=torch.float32)
    if stage == "sft" and args.resume is None:
        # 只载入基础模型权重；SFT 新建优化器，不继承预训练进度。
        base = torch.load(resolve_path(config["base_checkpoint"]), map_location="cpu", weights_only=True)
        base_config = dict(base["model_config"])
        base_config["max_seq_len"] = model_config.max_seq_len
        if TrainPilotconfig(**base_config) != model_config:
            raise ValueError("SFT 模型配置与基础权重不一致（允许更改 max_seq_len）")
        model.load_state_dict(base["model_state_dict"], strict=True)
        del base
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"],
    )
    completed_epochs = 0
    global_step = 0
    epoch_state = None
    if args.resume is not None:
        if training["num_workers"] != 0:
            raise ValueError("续训只支持 num_workers=0，确保数据端和模型随机状态可恢复")
        resume_path = resolve_path(args.resume)
        # 路径可以从本机迁移到服务器；核对内容摘要而不是绝对路径。
        with (resume_path.parent / "run_manifest.json").open(encoding="utf-8") as source:
            previous = json.load(source)
        if previous["config"].get("stage", "pretrain") != stage:
            raise ValueError("--resume 必须恢复同一训练阶段；首次 SFT 使用 base_checkpoint")
        if previous["data_sha256"] != fingerprints:
            raise ValueError("续训所用 tokenizer 或训练/验证数据与原运行不一致")
        for name in ("batch_size", "num_workers", "dtype", "seed", "cpu_threads", "accumulation_steps", "grad_clip", "lr_schedule"):
            if previous["config"]["training"][name] != training[name]:
                raise ValueError(f"续训时 {name} 必须与原运行一致")
        if training["lr_schedule"] == "cosine" and previous["config"]["training"]["epochs"] != training["epochs"]:
            raise ValueError("余弦调度续训需要保持原计划总 epochs 不变")
        progress = load_checkpoint(resume_path, model, optimizer)
        completed_epochs = progress["completed_epochs"]
        global_step = progress["global_step"]
        epoch_state = progress.get("epoch_state")
        # optimizer 状态会恢复原超参数，记录应以恢复后的实际值为准。
        training["learning_rate"] = (
            previous["config"]["training"]["learning_rate"]
            if training["lr_schedule"] == "cosine" else optimizer.param_groups[0]["lr"]
        )
        training["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
        print(f"恢复完成：已训练 {completed_epochs} 个 epoch，global_step={global_step}")

    print(f"阶段={stage}; 设备={device}; dtype={training['dtype']}; 参数数目={sum(p.numel() for p in model.parameters())}")
    print(f"训练样本={len(train_data)}; 验证样本={len(val_data)}; batch_size={training['batch_size']}")
    print(f"CPU threads={torch.get_num_threads()}; 目标总 epoch={training['epochs']}")
    print(f"梯度累积={training['accumulation_steps']}; 有效 batch≈{training['batch_size'] * training['accumulation_steps']}")
    print(f"结果目录：{output_dir}")
    if completed_epochs >= training["epochs"]:
        print("已达到目标总 epoch，无需重复训练；增加 --epochs 可以继续。")
        return

    # 5. 保存实际配置与数据来源。运行产物位于被 Git 忽略的 runs/ 下。
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": {"stage": stage, "model": asdict(model_config), "data": {k: str(v) for k, v in paths.items()}, "training": training},
        "data_sha256": fingerprints,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda},
        "resume_from": str(resolve_path(args.resume)) if args.resume else None,
        "base_checkpoint": str(resolve_path(config["base_checkpoint"])) if stage == "sft" else None,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    # 续训时不额外跑初始验证：DataLoader 迭代本身会消耗随机数。
    if args.resume is None:
        initial_val = evaluate(model, val_loader, device, amp_dtype=amp_dtype, loss_fn=functions.next_token_loss)
        print(f"训练前验证 loss={initial_val['loss']:.6f}")
        with metrics_path.open("w", encoding="utf-8") as log:
            log.write(json.dumps({"event": "initial_validation", "global_step": 0, "validation": initial_val}) + "\n")

    # 6. 每轮训练 -> 验证 -> 保存，epoch 和 global_step 从恢复值继续累计。
    for epoch in range(completed_epochs, training["epochs"]):
        train_generator.manual_seed(training["seed"] + epoch)
        prior_steps = epoch_state["steps"] if epoch_state is not None else 0
        epoch_global_base = global_step - prior_steps
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        loss_window = LossWindow(epoch_state)

        def on_update(state):
            current_step = epoch_global_base + state["steps"]
            last_trial_update = args.max_updates is not None and state["steps"] - prior_steps >= args.max_updates
            if current_step == 1 or current_step % training["log_interval"] == 0 or state["micro_steps"] == len(train_loader) or last_trial_update:
                window_loss = loss_window.update(state)
                print(
                    f"epoch={epoch + 1} batch={state['micro_steps']}/{len(train_loader)} "
                    f"global_step={current_step} mean_loss={state['loss_sum'] / state['valid_tokens']:.6f} "
                    f"window_loss={window_loss:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}", flush=True,
                )
                with metrics_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"event": "update", "epoch": epoch + 1,
                        "global_step": current_step, "micro_steps": state["micro_steps"],
                        "mean_loss": state["loss_sum"] / state["valid_tokens"],
                        "window_loss": window_loss, "loss_sum":state["loss_sum"],
                        "valid_tokens":state["valid_tokens"],
                        "learning_rate": optimizer.param_groups[0]["lr"]}) + "\n")
            if current_step % training["save_interval"] == 0 and state["micro_steps"] < len(train_loader):
                save_checkpoint(checkpoint_path, model, optimizer, epoch, current_step, epoch_state=state)

        train_record = train_one_epoch(
            model, optimizer, train_loader, device,
            accumulation_steps=training["accumulation_steps"], amp_dtype=amp_dtype,
            grad_clip=training["grad_clip"], epoch_index=epoch,
            base_lr=training["learning_rate"] if training["lr_schedule"] == "cosine" else None,
            total_micro_steps=training["epochs"] * len(train_loader),
            start_state=epoch_state, on_update=on_update, limit_updates=args.max_updates,
            train_step_fn=functions.train_step,
        )
        updates_this_call = train_record["steps"] - prior_steps
        global_step = epoch_global_base + train_record["steps"]
        if not train_record["finished"]:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            processed_tokens = train_record["valid_tokens"] - (epoch_state["valid_tokens"] if epoch_state else 0)
            partial_state = {name: train_record[name] for name in ("loss_sum", "valid_tokens", "steps", "micro_steps")}
            save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, epoch_state=partial_state)
            print(f"短试跑结束，global_step={global_step}；未完成的 epoch 可从 {checkpoint_path} 继续。")
            print(f"本次训练循环耗时={elapsed:.3f}s；有效监督 token/s={processed_tokens / elapsed:.2f}（含定期保存开销）")
            if device.type == "cuda":
                print(f"峰值已分配显存 MiB={torch.cuda.max_memory_allocated(device) / 1024**2:.2f}")
            plot_loss(metrics_path, title=f"TrainPilot {stage}")
            return
        val_record = evaluate(model, val_loader, device, amp_dtype=amp_dtype, loss_fn=functions.next_token_loss)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, global_step)
        record = {
            "event": "epoch", "completed_epochs": epoch + 1, "global_step": global_step,
            "train": train_record, "validation": val_record,
            "train_and_validation_seconds": elapsed,
        }
        if device.type == "cuda":
            record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        with metrics_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch + 1}/{training['epochs']} global_step={global_step} "
            f"train_loss={train_record['loss']:.6f} val_loss={val_record['loss']:.6f} "
            f"train_tokens={train_record['valid_tokens']} elapsed={elapsed:.3f}s",
            flush=True,
        )
        epoch_state = None
        plot_loss(metrics_path, title=f"TrainPilot {stage}")
        if args.max_updates is not None:
            args.max_updates -= updates_this_call
            if args.max_updates <= 0:
                print("已达到本次试跑的更新次数上限；checkpoint 已保存。")
                return
    print(f"训练完成，checkpoint：{checkpoint_path}")


if __name__ == "__main__":
    main()
