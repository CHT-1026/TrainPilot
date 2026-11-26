#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
MODEL="${TRAINPILOT_SFT_MODEL:-$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model}"
ROOT_RUN=runs/grpo_reward_ab_20261030
mkdir -p "$ROOT_RUN"
exec 9> "$ROOT_RUN/.gpu.lock"
flock -n 9 || { echo "已有对照任务运行"; exit 1; }
python verify_packet.py
python -m unittest model.test.test_code.test_reward_fields_v2 test.test_code.test_graded_reward test.test_code.test_reward_ab -v
python -u -m model.trainer.grpo_reward_ab train --model "$MODEL" --method graded --seed 42 --smoke --output-dir "$ROOT_RUN/smoke"
python -c 'import json; from pathlib import Path; r=json.loads(Path("runs/grpo_reward_ab_20261030/smoke/completed.json").read_text()); assert r["finished"], "小试没有达到有效更新预算，请先查看rollouts，未启动正式对照"'
for seed in 42 43; do
  for method in strict graded; do
    python -u -m model.trainer.grpo_reward_ab train --model "$MODEL" --method "$method" --seed "$seed" --output-dir "$ROOT_RUN/trials/${method}_${seed}"
  done
done
python -u -m model.test.test_code.evaluate_reward_ab --model "$MODEL"
