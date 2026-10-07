"""学习模板：先完成 Q/K/V 投影与拆头，再实现因果自注意力。"""

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim 和 num_heads 必须为正，且前者可被后者整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def project_qkv(self, x: torch.Tensor):
        """输入 [B, T, C]，返回形状均为 [B, T, C] 的 Q、K、V。"""
        if self.q_proj is None or self.k_proj is None or self.v_proj is None:
            raise NotImplementedError("请先完成 TODO 1：初始化 Q/K/V 投影")
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """把 [B, T, C] 转换为 [B, H, T, D]，其中 C = H * D。"""
        B, T, C = tensor.shape
        tensor = tensor.reshape(B, T, self.num_heads, self.head_dim)
        tensor = tensor.transpose(1, 2)
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.project_qkv(x)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn = q @ k.transpose(-2, -1)
        attn = attn / (self.head_dim ** 0.5)

        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, C)

        # 记得过输出投影
        out = self.out_proj(out)

        return out


def check_projection_and_heads():
    """只核对第一阶段形状；不代表完整 Attention 已通过验证。"""
    torch.manual_seed(42)
    model = CausalSelfAttention(embed_dim=128, num_heads=4).to("cuda")
    x = torch.randn(2, 7, 128, device="cuda")

    q, k, v = model.project_qkv(x)
    for name, tensor in zip(("Q", "K", "V"), (q, k, v)):
        assert tensor.shape == x.shape, f"{name} 投影形状不正确"
        heads = model.split_heads(tensor)
        expected = (x.shape[0], model.num_heads, x.shape[1], model.head_dim)
        assert tuple(heads.shape) == expected, f"{name} 拆头形状不正确"
        print(f"{name}: {tuple(tensor.shape)} -> {tuple(heads.shape)}")

    print("Q/K/V 投影和拆头形状检查通过。")

    out = model(x)
    print("attention output:", tuple(out.shape))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    print("完整注意力输出形状与有限值检查通过。")


def check_causality():
    """因果性检查：只改未来位置，过去位置的输出不应变化。"""
    torch.manual_seed(0)
    model = CausalSelfAttention(embed_dim=128, num_heads=4).to("cuda")
    model.eval()

    # 1. 创建输入 x1
    x1 = torch.randn(2, 7, 128, device="cuda")

    # 2. 复制得到 x2，只修改最后三个位置
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])

    # 3. 同一模型、eval + no_grad 下分别前向
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # 4. 比较前四个位置
    diff = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
    print(f"前四个位置的最大绝对差值: {diff:.3e}")

    # 浮点容差内应一致
    assert diff < 1e-5, "因果性检查失败：过去位置受到了未来位置的影响"
    print("因果性检查通过：过去位置不受未来位置影响。")


if __name__ == "__main__":
    check_projection_and_heads()
    check_causality()