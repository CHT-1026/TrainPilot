"""预训练基础函数：下一 token 损失与一次参数更新。

本文件只提供这两个函数，没有 main 或命令行入口。
数据端 labels 尚未移位；本文件在计算损失时统一移位一次。
第一版使用普通 float32 训练，不包含混合精度、完整 epoch 或验证循环。
"""
import torch
from torch import nn
import torch.nn.functional as F

from trainpilot.data import IGNORE_INDEX


def next_token_loss(
    logits: torch.Tensor, labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """返回有效监督位置上的平均交叉熵，以及这些位置的数量。

    logits: [B,T,V]，模型输出的原始分数，不需要先做 softmax。
    labels: [B,T]，与输入同位置的整数标签，padding 位置为 -100。
    第 t 个 logit 预测第 t+1 个标签；最后一个 logit 不参与本次损失。
    返回的 loss 保留计算图，供调用者执行 backward。
    """
    if logits.ndim != 3:
        raise ValueError("logits 必须为 [B,T,V] 三维张量")
    batch_size, seq_len, vocab_size = logits.shape
    if labels.shape != (batch_size, seq_len):
        raise ValueError("labels 必须为 [B,T]，与 logits 的前两维对应")
    if batch_size == 0 or vocab_size == 0 or seq_len < 2:
        raise ValueError("batch 和词表不能为空，序列长度必须至少为 2")
    if labels.dtype != torch.long:
        raise TypeError("labels 必须使用 torch.long 整数类型")
    if logits.device != labels.device:
        raise ValueError("logits 与 labels 必须位于同一设备")

    # 1. 只在这里移位：位置 t 的输出，监督目标是位置 t+1 的 token。
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    # 2. 计数必须在移位之后进行：首个标签不参与监督，padding 也不参与。
    valid_tokens = int((shifted_labels != IGNORE_INDEX).sum().item())
    if valid_tokens == 0:
        raise ValueError("移位后没有有效监督标签，无法计算平均损失")

    # 3. 每个位置看作一个 V 类分类样本。
    # reshape 能处理切片后的非连续张量；mean 只对未被忽略的标签求平均。
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
) -> dict[str, float | int]:
    """执行一次训练，返回更新前的 loss 数值与有效监督 token 数。

    batch 必须包含 input_ids、attention_mask、labels，形状均为 [B,T]。
    调用者负责事先把模型放到 device，并创建对应的优化器。
    本函数只迁移 batch，不修改原 batch 字典，也不保存计算图到记录中。
    """
    # 1. 开启训练模式；它控制 Dropout 等模块的行为。
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # 2. 清除上一步梯度，防止本次 backward 与旧梯度累加。
    optimizer.zero_grad(set_to_none=True)

    # 3. 前向传播：模型返回 logits，损失由上面的函数计算。
    logits = model(input_ids, attention_mask=attention_mask)
    loss, valid_tokens = next_token_loss(logits, labels)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("训练损失出现 NaN 或 Inf，本次不执行参数更新")

    # 4. backward 计算梯度；step 才真正修改可训练参数。
    loss.backward()
    optimizer.step()

    # 5. 返回 Python 数值，避免日志记录意外持有整个计算图。
    # 注意：这是刚才前向得到的损失，不是更新参数后重新计算的损失。
    return {"loss": loss.item(), "valid_tokens": valid_tokens}
