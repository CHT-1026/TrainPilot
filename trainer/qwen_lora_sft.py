"""Qwen LoRA SFT：复用全参数入口的数据流程与本项目原生训练循环。

运行：python -m trainer.qwen_lora_sft
输出 adapter 可通过 PeftModel.from_pretrained 加载到同一版本基座上。
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.qwen_full_sft import build_parser, run_training


def train(args=None):
    parser = build_parser("lora")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    args = parser.parse_args(args)
    from peft import LoraConfig, TaskType
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias="none",
    )
    run_training(args, lora_config=config)


if __name__ == "__main__":
    train()
