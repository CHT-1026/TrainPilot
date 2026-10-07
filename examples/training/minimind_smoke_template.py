"""MiniMind 数据到参数更新的冒烟练习：学习者填写 train_step。

使用固定上游代码与两条自写文本，只验证训练链路，不复现上游完整训练。
先 --device cpu --steps 1；通过后再制定 GPU 短试跑。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


EXPECTED_UPSTREAM_SHA = "f659b55761b754d306bd140573493a6543cafd7f"


def train_step(model, optimizer, input_ids, labels) -> float:
    # 1. 清除上一轮梯度
    optimizer.zero_grad(set_to_none=True)

    # 2. 前向传播，同时让 MiniMind 内部计算损失
    outputs = model(input_ids, labels=labels)

    # 3. 从返回对象中取出标量损失
    loss = outputs.loss

    if loss is None:
        raise RuntimeError("模型没有返回损失")

    if not torch.isfinite(loss).item():
        raise RuntimeError("损失出现 NaN 或 Inf")

    # 4. TODO：从 loss 发起反向传播
    loss.backward()
    # 5. TODO：让 optimizer 更新参数
    optimizer.step()
    # 6. TODO：返回 loss 的 Python 数值，注意方法需要调用
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        raise ValueError("本练习仅允许 1 到 200 步")
    reference_dir = args.reference_dir.resolve()
    upstream_sha = subprocess.check_output(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_sha != EXPECTED_UPSTREAM_SHA:
        raise ValueError(f"上游版本不一致：{upstream_sha}")
    sys.path.insert(0, str(reference_dir))
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    # 自写合成夹具，不下载外部语料。数据文件写入指定的忽略目录。
    args.work_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.work_dir / "smoke_texts.jsonl"
    texts = (
        "模型通过训练学习预测下一个 token。",
        "我们记录训练配置，并检查输入与标签的对齐。",
    )
    data_bytes = "".join(
        json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts
    ).encode("utf-8")
    data_path.write_bytes(data_bytes)
    tokenizer = AutoTokenizer.from_pretrained(reference_dir / "model", local_files_only=True)
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=32)
    pairs = [dataset[i] for i in range(len(dataset))]
    input_ids = torch.stack([pair[0] for pair in pairs]).to(device)
    labels = torch.stack([pair[1] for pair in pairs]).to(device)
    valid_targets = labels[:, 1:].ne(-100).sum().item()
    assert valid_targets > 0

    config_values = dict(
        vocab_size=len(tokenizer), hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        intermediate_size=512, max_position_embeddings=64,
        use_moe=False, dropout=0.0, flash_attn=False,
    )
    model = MiniMindForCausalLM(MiniMindConfig(**config_values)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print("上游 SHA:", upstream_sha)
    print("设备 / dtype:", device, next(model.parameters()).dtype)
    print("seed=42; CPU threads=2; AdamW lr=0.001 weight_decay=0.01")
    print("模型配置:", config_values)
    print("参数数目:", sum(p.numel() for p in model.parameters()))
    print("数据 SHA256:", hashlib.sha256(data_bytes).hexdigest())
    print("输入 / 标签形状:", tuple(input_ids.shape), tuple(labels.shape))
    print("每步有效监督目标数:", valid_targets)

    model.eval()
    with torch.no_grad():
        initial = model(input_ids, labels=labels)
        assert torch.isfinite(initial.logits).all() and torch.isfinite(initial.loss)
        # 独立核对移位与 ignore_index，避免把已移位标签再次交给模型。
        manual_loss = F.cross_entropy(
            initial.logits[:, :-1, :].reshape(-1, len(tokenizer)),
            labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        assert torch.allclose(initial.loss, manual_loss, atol=1e-6, rtol=1e-5)
        print("logits 形状:", tuple(initial.logits.shape))
        print(f"训练前 loss={initial.loss.item():.6f}；独立损失核对通过")
    weight_before = model.model.embed_tokens.weight.detach().clone()
    del initial, manual_loss

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        loss_value = train_step(model, optimizer, input_ids, labels)
        assert math.isfinite(loss_value), "训练损失非有限"
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"step={step:03d} loss_before_update={loss_value:.6f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() if device.type == "cuda" else None

    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(parameter.grad).all(), f"{name} 梯度非有限"
            checked += 1
    weight_diff = (model.model.embed_tokens.weight.detach() - weight_before).abs().max().item()
    assert weight_diff > 0, "嵌入权重未更新"
    model.eval()
    with torch.no_grad():
        final = model(input_ids, labels=labels)
        assert torch.isfinite(final.loss) and torch.isfinite(final.logits).all()
        print(f"训练后 loss={final.loss.item():.6f}")
    print(f"梯度检查通过：{checked} 个参数张量；嵌入权重最大变化={weight_diff:.6e}")
    print(f"训练循环墙钟秒（含日志及同步开销）: {elapsed:.3f}")
    if device.type == "cuda":
        print(f"训练循环峰值已分配显存 MiB（含验证权重快照）: {peak_allocated / 1024**2:.2f}")
    print("参考 Dataset、模型损失、反向传播与参数更新检查通过；不代表泛化效果。")


if __name__ == "__main__":
    main()
