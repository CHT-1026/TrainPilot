from dataclasses import asdict
import os
from contextlib import nullcontext
from itertools import islice
import math
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import torch
from torch import nn
from torch.utils.data import DataLoader

from trainpilot.pretrain import next_token_loss, train_step


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: str | torch.device,
    *,
    accumulation_steps: int = 1,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float | None = None,
    base_lr: float | None = None,
    total_micro_steps: int | None = None,
    epoch_index: int = 0,
    start_state: dict | None = None,
    on_update=None,
    limit_updates: int | None = None,
) -> dict:
    """按有效 token 加权累积梯度，更新后可记录日志或保存 checkpoint。

    steps 是本 epoch 累计 optimizer 更新数，micro_steps 是累计 batch 数。
    start_state 来自同一数据顺序的 checkpoint；回调仅在完整梯度更新后执行。
    """
    if type(accumulation_steps) is not int or accumulation_steps <= 0:
        raise ValueError("accumulation_steps 必须是正整数")
    if limit_updates is not None and (type(limit_updates) is not int or limit_updates <= 0):
        raise ValueError("limit_updates 必须是正整数")
    state = dict(start_state or {"loss_sum": 0.0, "valid_tokens": 0, "steps": 0, "micro_steps": 0})
    if not 0 <= state["micro_steps"] < len(loader):
        raise ValueError("恢复的 batch 位置不在当前 DataLoader 范围内")
    if base_lr is not None and (total_micro_steps is None or total_micro_steps <= 0):
        raise ValueError("余弦调度需要正的 total_micro_steps")
    iterator = iter(loader)
    # 数据端无随机变换；入口用独立 generator 重建同一 epoch 的数据顺序。
    for _ in range(state["micro_steps"]):
        next(iterator)
    updates_this_call = 0
    while True:
        group = list(islice(iterator, accumulation_steps))
        if not group:
            break
        counts = [int((batch["labels"][:, 1:] != -100).sum().item()) for batch in group]
        if any(count == 0 for count in counts):
            raise ValueError("micro-batch 没有有效监督 token")
        group_tokens = sum(counts)
        if base_lr is not None:
            schedule_step = epoch_index * len(loader) + state["micro_steps"] + len(group)
            ratio = min(schedule_step / total_micro_steps, 1.0)
            lr = base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * ratio)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        for index, (batch, count) in enumerate(zip(group, counts)):
            record = train_step(
                model, optimizer, batch, device,
                zero_grad=index == 0, update_parameters=index == len(group) - 1,
                loss_weight=count / group_tokens, amp_dtype=amp_dtype, grad_clip=grad_clip,
            )
            state["loss_sum"] += record["loss"] * record["valid_tokens"]
            state["valid_tokens"] += record["valid_tokens"]
            state["micro_steps"] += 1
        state["steps"] += 1
        updates_this_call += 1
        if on_update is not None:
            on_update(dict(state))
        if limit_updates is not None and updates_this_call >= limit_updates:
            break

    if state["valid_tokens"] == 0:
        raise ValueError("训练 DataLoader 为空，没有有效监督 token")

    return {
        "loss": state["loss_sum"] / state["valid_tokens"],
        "valid_tokens": state["valid_tokens"], "steps": state["steps"],
        "micro_steps": state["micro_steps"], "loss_sum": state["loss_sum"],
        "finished": state["micro_steps"] == len(loader),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    *,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float | int]:
    was_training = model.training
    total_loss_sum = 0.0
    total_valid_tokens = 0
    steps = 0

    try:
        # eval 控制 Dropout；no_grad 控制计算图，二者职责不同。
        model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                autocast = (
                    torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype)
                    if amp_dtype is not None else nullcontext()
                )
                with autocast:
                    logits = model(input_ids, attention_mask=attention_mask)
                    loss, valid_tokens = next_token_loss(logits, labels)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("验证损失出现 NaN 或 Inf")

                total_loss_sum += loss.item() * valid_tokens
                total_valid_tokens += valid_tokens
                steps += 1

        if total_valid_tokens == 0:
            raise ValueError("验证 DataLoader 为空，没有有效监督 token")

        return {
            "loss": total_loss_sum / total_valid_tokens,
            "valid_tokens": total_valid_tokens,
            "steps": steps,
        }
    finally:
        # 正常返回或中途报错，均恢复进入验证函数时的模式。
        model.train(was_training)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    global_step: int,
    *,
    epoch_state: dict | None = None,
) -> None:
    for name, value in (("completed_epochs", completed_epochs), ("global_step", global_step)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} 必须是非负整数")

    checkpoint = {
        "format_version": 1,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        # AdamW 的动量统计与步数也在这里；只有权重不足以恢复训练。
        "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "epoch_state": epoch_state,
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        },
    }

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(checkpoint, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        # Windows 下必须先关闭临时文件，才能替换目标。
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=True,
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("checkpoint 格式或版本不支持")
    if checkpoint["model_config"] != asdict(model.config):
        raise ValueError("checkpoint 模型配置与当前模型不一致")
    optimizer_class = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    if checkpoint["optimizer_class"] != optimizer_class:
        raise ValueError("checkpoint 优化器类型与当前优化器不一致")

    progress = {
        "completed_epochs": checkpoint["completed_epochs"],
        "global_step": checkpoint["global_step"],
    }
    for name, value in progress.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"checkpoint 的 {name} 必须是非负整数")

    rng_state = checkpoint["rng_state"]
    cuda_state = rng_state["cuda"]
    restore_cuda = cuda_state is not None and torch.cuda.is_initialized()
    if restore_cuda and len(cuda_state) != torch.cuda.device_count():
        raise ValueError("checkpoint 的 CUDA 随机状态与当前设备数不一致")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 恢复必须在新模型初始化之后；初始化本身会消耗随机数。
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch"])
    if restore_cuda:
        torch.cuda.set_rng_state_all(cuda_state)
    if checkpoint.get("epoch_state") is not None:
        progress["epoch_state"] = checkpoint["epoch_state"]
    return progress
