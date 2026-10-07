"""下一阶段学习骨架：填写 Pre-LN Transformer Block。

与 transformer.py 放在同一目录。当前含有未完成的 TODO。
"""

import torch
from torch import nn

from transformer import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads)

        # TODO 1：两个独立的 LayerNorm，只归一化每个 token 的通道维。
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # TODO 2：用 nn.Sequential 组织前馈网络：
        # C -> 4*C -> GELU -> C，每个位置独立应用同一个网络。
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3：完成两次 Pre-LN 残差更新。
        # 第一段：归一化 -> 注意力 -> 与该段输入相加。
        u = x + self.attn(self.ln1(x))
        y = u + self.mlp(self.ln2(u))

        # 第二段：归一化 -> MLP -> 与该段输入相加。
        return y

def check_block():
    """填写模型后检查前向、梯度连通性及一次未来扰动。"""
    torch.manual_seed(0)
    model = TransformerBlock(embed_dim=128, num_heads=4).to("cuda")
    x = torch.randn(2, 7, 128, device="cuda")

    out = model(x)
    assert out.shape == x.shape, "Block 输出形状不正确"
    assert torch.isfinite(out).all(), "Block 输出包含非有限值"
    print("Block 输出形状与有限值检查通过:", tuple(out.shape))

    model.zero_grad(set_to_none=True)
    probe_loss = out.square().mean()
    probe_loss.backward()
    checked = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(param.grad).all(), f"{name} 梯度包含非有限值"
            checked += 1
    print(f"梯度检查通过: {checked} 个可训练参数张量")

    model.eval()
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x2)
    diff = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
    print(f"Block 前四个位置的最大绝对差值: {diff:.3e}")
    assert diff < 1e-5, "Block 因果性检查失败"
    print("Block 因果性检查通过。")


if __name__ == "__main__":
    check_block()