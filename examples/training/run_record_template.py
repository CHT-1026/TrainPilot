"""学习练习：从已记录的实测值生成 Agent 可读取的实验摘要。

输入是人工转录的 GPU 短训练记录，不重新训练，也不读取权重。
学习者填写两个派生指标；其他字段保留数据来源及统计口径。
"""

import argparse
import json
from pathlib import Path


def derive_metrics(measurements: dict) -> dict:
    steps = measurements["steps"]
    valid_targets_per_step = measurements["valid_targets_per_step"]
    elapsed_seconds = measurements["loop_elapsed_seconds"]
    if steps <= 0 or valid_targets_per_step <= 0 or elapsed_seconds <= 0:
        raise ValueError("步数、每步有效目标数、耗时必须为正")

    # TODO 1：整个训练循环处理的有效监督目标总数
    # 每一步都处理了 valid_targets_per_step 个有效目标，
    # 重复训练也要累计，所以直接乘步数。
    total_supervised_tokens = steps * valid_targets_per_step

    # TODO 2：每秒处理的有效监督目标数
    # 总量除以总耗时，得到吞吐。
    supervised_tokens_per_second = total_supervised_tokens / elapsed_seconds

    return {
        "initial_train_loss": measurements["initial_train_loss"],
        "final_train_loss": measurements["final_train_loss"],
        "validation_loss": None,
        "steps": steps,
        "valid_targets_per_step": valid_targets_per_step,
        "total_supervised_tokens": total_supervised_tokens,
        "supervised_tokens_per_second": supervised_tokens_per_second,
        "loop_elapsed_seconds": elapsed_seconds,
        "loop_peak_allocated_mib": measurements["loop_peak_allocated_mib"],
        "peak_includes_verification_weight_snapshot": measurements[
            "peak_includes_verification_weight_snapshot"
        ],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.measurements.resolve() == args.output.resolve():
        raise ValueError("输出路径不能覆盖输入记录")
    record = json.loads(args.measurements.read_text(encoding="utf-8"))
    record["metrics"] = derive_metrics(record["measurements"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("实验摘要:", args.output.resolve())
    print(json.dumps(record["metrics"], ensure_ascii=False, indent=2, allow_nan=False))
    print("数据来源和未知字段已保留；该吞吐仅描述这次固定批次运行。")


if __name__ == "__main__":
    main()
