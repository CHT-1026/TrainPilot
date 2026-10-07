"""学习练习：保存模型配置和权重，重建模型并验证完整 logits。

从训练脚本的 main 末尾调用 save_and_verify；不要另建随机模型代替已训练模型。
本练习验证权重读写，不保存优化器状态或保证断点续训。
"""

from pathlib import Path

import torch

from tiny_transformer_lm_template import TinyTransformerLM


def save_and_verify(model, model_config: dict, inputs: torch.Tensor,
                    checkpoint_path: Path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        reference_logits = model(inputs).detach().clone()

    # TODO 1：建立包含两个键的字典：
    # "model_config" 保存构建模型所需的配置；
    # "model_state_dict" 保存模型的 state_dict（记得调用方法）。
    payload = {
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)

    # 先加载到 CPU，随后将重建的模型移动到输入所在设备。
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # TODO 2：从 loaded["model_config"] 构建一个新的 TinyTransformerLM。
    # 构造函数接收关键字参数，可回顾 Python 的 **字典 解包语法。
    restored_model = TinyTransformerLM(**loaded["model_config"])
    if restored_model is None:
        raise NotImplementedError("请完成 TODO 2：重建模型")

    # TODO 3：调用 restored_model.load_state_dict，加载已保存的权重。
    # 保留默认 strict=True，让缺失或多余的权重键导致报错。
    restored_model.load_state_dict(loaded["model_state_dict"])

    restored_model = restored_model.to(inputs.device)
    restored_model.eval()
    with torch.no_grad():
        restored_logits = restored_model(inputs)

    assert torch.isfinite(restored_logits).all(), "加载后的输出包含非有限值"
    diff = (reference_logits - restored_logits).abs().max().item()
    assert torch.allclose(reference_logits, restored_logits, atol=1e-6, rtol=1e-5), (
        f"加载前后的 logits 不一致，最大绝对差值={diff:.6e}"
    )
    print("checkpoint:", checkpoint_path.resolve())
    print(f"checkpoint 大小 MiB: {checkpoint_path.stat().st_size / 1024**2:.3f}")
    print(f"加载前后 logits 最大绝对差值: {diff:.6e}")
    print("新模型加载权重后的输出一致性检查通过。")
