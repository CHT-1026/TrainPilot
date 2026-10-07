"""学习练习：用本地 MiniMind tokenizer 构造一个预训练样本。

自行填写 build_sample。输入与标签均保持长度 max_length；此处不移位。
三个自写文本案例只用于数据路径检查，不作为真实语料或效果评测。
"""

import argparse

import torch
from transformers import AutoTokenizer


def build_sample(text: str, tokenizer, max_length: int):
    """返回两个 torch.long 张量 input_ids、labels，形状均为 [max_length]。"""
    if max_length < 2:
        raise ValueError("max_length 至少为 2，需给 BOS 和 EOS 留位置")

    # TODO 1：编码 text，禁用自动特殊 token，启用截断。
    # 原文本的 token 数上限是多少？留出 BOS、EOS 的位置。
    # tokenizer(...) 的结果中，取 input_ids，而不是向量。
    text_ids = tokenizer(
        text,
        add_special_tokens=False,
        max_length=max_length - 2,
        truncation=True,
    ).input_ids

    # TODO 2：在编号列表前后手动添加 BOS、EOS，然后在右侧补齐 PAD。

    ids = [tokenizer.bos_token_id] + text_ids + [tokenizer.eos_token_id]
    ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))

    # TODO 3：构造 torch.long 输入张量；克隆它得到 labels。
    # 将 PAD 对应的标签改为 -100，保持 input_ids 中的 PAD 编号。
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100

    # TODO 4：返回 input_ids、labels。这一层不做下一 token 移位。
    return input_ids, labels


def check_samples(tokenizer):
    max_length = 32
    cases = (
        ("短文本", "模型通过训练学习预测下一个 token。"),
        ("空文本", ""),
        ("需截断的长文本", "我们记录训练配置，并检查输入与标签的对齐。" * 20),
    )
    for name, text in cases:
        input_ids, labels = build_sample(text, tokenizer, max_length)
        assert input_ids.shape == labels.shape == (max_length,), "长度错误"
        assert input_ids.dtype == labels.dtype == torch.long, "编号与标签须为整数"
        assert ((input_ids >= 0) & (input_ids < len(tokenizer))).all(), "输入编号越界"
        is_pad = input_ids.eq(tokenizer.pad_token_id)
        non_pad = input_ids[~is_pad]
        assert len(non_pad) >= 2, "至少应包含 BOS 和 EOS"
        assert non_pad[0].item() == tokenizer.bos_token_id, "缺少开头 BOS"
        assert non_pad[-1].item() == tokenizer.eos_token_id, "缺少结尾 EOS"
        assert not is_pad[:len(non_pad)].any(), "PAD 应仅出现在右侧"
        assert torch.equal(labels[~is_pad], input_ids[~is_pad]), "有效标签错误或已移位"
        assert labels[is_pad].eq(-100).all(), "PAD 标签未忽略"
        if not text:
            assert len(non_pad) == 2, "空文本只应保留 BOS、EOS"
        if name == "需截断的长文本":
            assert len(non_pad) == max_length, "长文本没有按预期截断填满"

        print(f"\n{name}:")
        print("input_ids:", input_ids.tolist())
        print("labels:", labels.tolist())
        print("解码有效输入:", tokenizer.decode(non_pad.tolist()))
        # MiniMind 模型中的损失会对齐到 labels[1:]；这里只统计有效目标。
        print("下一 token 有效监督目标数:", labels[1:].ne(-100).sum().item())
    print("\n三个样本的结构检查通过；尚未执行 MiniMind 训练。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", required=True, help="本地 tokenizer 目录")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    check_samples(tokenizer)


if __name__ == "__main__":
    main()
