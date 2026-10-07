"""学习练习：为已对齐的下一 token 标签计算交叉熵。

输入 logits [B, T, V]，整数 targets [B, T]；此处不再移位标签。
运行入口包含两组受控分数，用于核对损失和可微性。
"""

import math

import torch
from torch import nn


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    if targets.shape != (B, T):
        raise ValueError("targets 必须与 logits 的样本、位置维对应")

    # TODO：把每个 token 位置看作一个 V 类分类样本。
    # 使用 nn.CrossEntropyLoss()，整理输入与整数标签的形状。
    # 直接传入原始 logits；标签已经对齐，不在本函数中再次移位。
    # 返回一个对所有位置取平均的标量损失。
    # 1. 展平：[B, T, V] -> [B*T, V]
    logits_flat = logits.reshape(B * T, V)
    # 2. 展平标签：[B, T] -> [B*T]
    targets_flat = targets.reshape(B * T)
    # 3. 交叉熵，默认 reduction="mean"，返回标量
    loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)
    return loss


def check_loss():
    device = "cuda"
    targets = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]], device=device)
    logits = torch.zeros(2, 4, 3, device=device, requires_grad=True)
    loss = next_token_loss(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss), "损失应为有限标量"
    assert abs(loss.item() - math.log(3)) < 1e-6, "均匀预测的交叉熵不正确"
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"均匀预测 loss: {loss.item():.6f}；logits 梯度检查通过")

    # 抬高每个位置正确类别的分数，损失应降低。
    better_logits = torch.zeros(2, 4, 3, device=device)
    better_logits.scatter_(-1, targets.unsqueeze(-1), 5.0)
    better_loss = next_token_loss(better_logits, targets)
    assert torch.isfinite(better_loss) and better_loss < loss.detach()
    print(f"提高正确类别分数后的 loss: {better_loss.item():.6f}")
    print("受控交叉熵检查通过；尚未进行语言模型训练。")


if __name__ == "__main__":
    check_loss()
