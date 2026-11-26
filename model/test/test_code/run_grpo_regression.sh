#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
MODEL="${TRAINPILOT_SFT_MODEL:-$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model}"
V5="${TRAINPILOT_V5_ADAPTER:-$HOME/TrainPilot-GRPO-V2/runs/init_v5/adapter}"
POLICY="${TRAINPILOT_GRPO_ADAPTER:-$HOME/TrainPilot-GRPO-V2/runs/trainpilot_grpo_v2_probe200/adapter}"
OUTPUT="${TRAINPILOT_REGRESSION_RUN:-runs/grpo_v2_regression_20261029}"
python -m unittest model.test.test_code.test_grpo_regression -v
args=(--model "$MODEL" --baseline-adapter "$V5" --policy-adapter "$POLICY" --output-dir "$OUTPUT")
python -u -m model.test.test_code.evaluate_grpo_regression "${args[@]}" --check-only
python -u -m model.test.test_code.evaluate_grpo_regression "${args[@]}"
