#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
MODEL="${TRAINPILOT_SFT_MODEL:-$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model}"
OUTPUT="${TRAINPILOT_V6_EVAL_RUN:-runs/lora-v6-eval-20261030}"
mkdir -p "$OUTPUT"
exec 9> "$OUTPUT/.eval.lock"
flock -n 9 || { echo "同一结果目录已有评测任务运行"; exit 1; }
python verify_packet.py
python -m unittest model.test.test_code.test_agent_v6_eval test.test_code.test_grpo_regression test.test_code.test_agent_v5_eval -v
python -u -m model.test.test_code.evaluate_agent_v6 --model "$MODEL" --output-dir "$OUTPUT" --check-only
python -u -m model.test.test_code.evaluate_agent_v6 --model "$MODEL" --output-dir "$OUTPUT" "$@"
