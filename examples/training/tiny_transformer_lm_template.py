"""小型语言模型学习骨架：词嵌入、可学习位置嵌入、Block 和词表输出。

与 transformer.py、已完成的 transformer_block_template.py 放在同一目录。
直接运行检查完整模型结构；随机 token 仅用于检查，不作为真实训练数据。
"""

import torch
from torch import nn

from transformer_block_template import TransformerBlock


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # TODO 1：两个 nn.Embedding。
        # token 编号查表得到 C 维向量；位置编号也查表得到 C 维向量。
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # TODO 2：使用 nn.ModuleList 存放 num_layers 个独立的 Block。
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        # TODO 3：最终 LayerNorm，以及 C -> vocab_size 的线性输出层。
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """输入整数 token [B, T]，返回原始 logits [B, T, V]。"""
        B, T = tokens.shape
        if T > self.max_seq_len:
            raise ValueError("序列长度超过位置嵌入支持的上限")

        # TODO 4：构造 0 到 T-1 的位置编号，设备与 tokens 相同。
        # token 向量形状 [B, T, C]；位置向量可用 [T, C] 广播相加。
        # 将相加后的表示依次传过各个 Block、ln_final 和 lm_head。
        # 输出保留原始分数，训练损失在模型外部计算。
        # 1. token 嵌入：[B, T] -> [B, T, C]
        tok_emb = self.token_embedding(tokens)
        # 2. 位置编号：[T]，设备与 tokens 一致
        positions = torch.arange(T, device=tokens.device)
        # 3. 位置嵌入：[T] -> [T, C]，广播到 [B, T, C]
        pos_emb = self.position_embedding(positions)
        # 4. 相加
        x = tok_emb + pos_emb
        # 5. 依次过每个 Transformer Block
        for block in self.blocks:
            x = block(x)
        # 6. 最终 LayerNorm
        x = self.ln_final(x)
        # 7. 输出 logits：[B, T, C] -> [B, T, V]
        logits = self.lm_head(x)

        return logits
        


def check_model():
    """检查形状、有限值、梯度连通性和一次未来扰动。"""
    torch.manual_seed(0)
    vocab_size = 32  # 检查使用的假词表，不是已选定的项目 tokenizer。
    model = TinyTransformerLM(vocab_size=vocab_size).to("cuda")
    tokens = torch.randint(vocab_size, (2, 7), device="cuda")

    logits = model(tokens)
    assert logits.shape == (2, 7, vocab_size), "词表输出形状不正确"
    assert torch.isfinite(logits).all(), "logits 包含非有限值"
    print("语言模型 logits 形状与有限值检查通过:", tuple(logits.shape))

    model.zero_grad(set_to_none=True)
    # 只验证反向传播连通性；这不是下一 token 训练目标。
    probe_loss = logits.square().mean()
    probe_loss.backward()
    checked = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(param.grad).all(), f"{name} 梯度包含非有限值"
            checked += 1
    print(f"语言模型梯度检查通过: {checked} 个可训练参数张量")

    model.eval()
    changed = tokens.clone()
    # 确保未来 token 确实变化，保持编号仍在词表范围内。
    changed[:, 4:] = (changed[:, 4:] + 1) % vocab_size
    with torch.no_grad():
        logits1 = model(tokens)
        logits2 = model(changed)
    diff = (logits1[:, :4, :] - logits2[:, :4, :]).abs().max().item()
    print(f"语言模型前四个位置的最大绝对差值: {diff:.3e}")
    assert diff < 1e-5, "语言模型因果性检查失败"
    print("语言模型因果性检查通过。")


if __name__ == "__main__":
    check_model()
