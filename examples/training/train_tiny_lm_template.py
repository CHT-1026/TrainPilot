"""固定合成小批数据上的短训练练习：由学习者填写 train_step。

同目录须已有完成的 TinyTransformerLM、TransformerBlock 与 next_token_loss。
结果只代表固定训练样本上的学习行为，不作为泛化或性能优化结论。
"""

import math
import sys
import time

import torch
from pathlib import Path
from checkpoint_template import save_and_verify
from next_token_loss_template import next_token_loss
from tiny_transformer_lm_template import TinyTransformerLM


def train_step(model, optimizer, inputs, targets) -> float:
    """执行一次训练，返回更新前的 loss 数值。"""
    # TODO：清除梯度 -> 模型前向 -> next_token_loss -> 检查 loss 有限
    # -> 反向传播 -> 优化器更新。
    # 最后返回 loss 的 Python 数值，用于记录，不保留计算图。
    optimizer.zero_grad()
    logits = model(inputs)
    loss = next_token_loss(logits, targets)
    assert torch.isfinite(loss).item(), "loss 非有限"
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    torch.manual_seed(42)
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("PyTorch 报告的显存 GiB:", round(props.total_memory / 1024**3, 2))
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__, "构建 CUDA:", torch.version.cuda)

    config = dict(vocab_size=8, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=64)
    model = TinyTransformerLM(**config).to("cuda")
    # 这是本次学习试跑的起点配置，不代表最佳超参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    steps = 200
    print("模型配置:", config)
    print("优化器: AdamW, lr=0.001, weight_decay=0.01; seed=42; steps=200")
    print("可训练参数数目:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 人工循环规则：0->1->...->7->0，不需要下载数据。
    tokens = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0],
            [4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
        device="cuda",
    )
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    print("输入与目标形状:", tuple(inputs.shape), tuple(targets.shape))
    print("首行输入:", inputs[0].tolist())
    print("首行目标:", targets[0].tolist())

    model.eval()
    with torch.no_grad():
        initial_logits = model(inputs)
        initial_loss = next_token_loss(initial_logits, targets).item()
        initial_accuracy = (initial_logits.argmax(-1) == targets).float().mean().item()
    print(f"训练前 loss={initial_loss:.6f}, 训练样本 token 准确率={initial_accuracy:.2%}")

    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        loss_value = train_step(model, optimizer, inputs, targets)
        assert math.isfinite(loss_value), "训练 loss 非有限"
        if step == 1 or step % 20 == 0:
            print(f"step={step:03d} loss_before_update={loss_value:.6f}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated_bytes = torch.cuda.max_memory_allocated()

    model.eval()
    with torch.no_grad():
        final_logits = model(inputs)
        final_loss = next_token_loss(final_logits, targets).item()
        accuracy = (final_logits.argmax(-1) == targets).float().mean().item()
    print(f"训练后 loss={final_loss:.6f}, 训练样本 token 准确率={accuracy:.2%}")
    print("首行训练后预测:", final_logits.argmax(-1)[0].tolist())
    print(f"训练循环墙钟秒（含日志及同步开销）: {elapsed:.3f}")
    print(f"训练循环峰值已分配显存 MiB: {peak_allocated_bytes / 1024**2:.2f}")
    save_and_verify(
        model,
        config,
        inputs,
        Path("artifacts/toy_lm/checkpoint.pt"),
    )


if __name__ == "__main__":
    main()
