"""Qwen LoRA SFT：复用全参数入口的数据流程与本项目原生训练循环。

运行：python -m model.trainer.qwen_lora_sft
输出 adapter 可通过 PeftModel.from_pretrained 加载到同一版本基座上。
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.trainer.qwen_full_sft import build_parser, project_path, run_training


def train(args=None):
    parser = build_parser("lora")
    parser.add_argument("--config", help="JSON训练配置；命令行参数覆盖配置文件")
    parser.add_argument("--init-adapter", help="加载已有LoRA权重开始新一轮SFT；不恢复旧优化器。--resume只用于本轮断点")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    config_path = parser.parse_known_args(args)[0].config
    if config_path:
        defaults = json.loads(project_path(config_path).read_text(encoding="utf-8"))
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            parser.error("未知配置项：" + ", ".join(sorted(unknown)))
        parser.set_defaults(**defaults)
    args = parser.parse_args(args)
    from peft import LoraConfig, TaskType
    if args.init_adapter:
        config = LoraConfig.from_pretrained(project_path(args.init_adapter))
        # 接续时继承已有adapter结构；不按命令行默认值重新创建一份LoRA。
        args.lora_rank, args.lora_alpha = config.r, config.lora_alpha
        args.lora_dropout = config.lora_dropout
        args.target_modules = sorted(config.target_modules) if not isinstance(config.target_modules, str) else config.target_modules
    else:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias="none",
        )
    run_training(args, lora_config=config)


if __name__ == "__main__":
    train()
