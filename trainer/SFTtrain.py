"""SFT trainer：下一 token 损失与单步训练，仅包含两个函数。
"""
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from dataset.dataset import IGNORE_INDEX


def next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3:
        raise ValueError("logits 必须为 [B,T,V]")
    batch_size, seq_len, vocab_size = logits.shape
    if labels.shape != (batch_size, seq_len):
        raise ValueError("labels 必须为 [B,T]，与 logits 对齐")
    if batch_size == 0 or vocab_size == 0 or seq_len < 2:
        raise ValueError("batch、词表不能为空，序列至少有两个位置")
    if labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("labels 必须为 torch.long，且与 logits 在同一设备")

    # 1. 只在这里移位：位置 t 的 logits 预测位置 t+1 的 label。
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    # 2. 统计移位后不等于 IGNORE_INDEX 的标签，转换为 Python int。
    valid_tokens = int((shifted_labels != IGNORE_INDEX).sum().item())
    if valid_tokens == 0:
        raise ValueError("移位后没有有效监督目标，不能计算平均交叉熵")

    # 3. 每个位置视为一个 V 类样本，计算平均交叉熵；忽略 IGNORE_INDEX。
    loss = F.cross_entropy(
        shifted_logits.reshape(-1, vocab_size),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    return loss, valid_tokens


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: str | torch.device,
    *,
    zero_grad: bool = True,
    update_parameters: bool = True,
    loss_weight: float = 1.0,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float | None = None,
) -> dict[str, float | int]:
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    autocast = (
        torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype)
        if amp_dtype is not None else nullcontext()
    )
    with autocast:
        # 4. 模型前向，直接返回 logits。
        logits = model(input_ids, attention_mask=attention_mask)
        loss, valid_tokens = next_token_loss(logits, labels)

    if not torch.isfinite(loss).item():
        raise FloatingPointError("SFT loss 出现 NaN 或 Inf，不执行更新")

    # 5. 对 loss * loss_weight 发起反向传播。
    (loss * loss_weight).backward()

    if update_parameters:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True,
            )
        # 6. 优化器更新参数。
        optimizer.step()

    # 7. 返回更新前的 loss 和有效 token 数，避免返回带计算图的 Tensor。
    return {"loss": loss.item(), "valid_tokens": valid_tokens}