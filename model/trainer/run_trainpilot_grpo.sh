#!/usr/bin/env bash
# 使用现有学校环境与 Full SFT 目录，完全离线；probe 不训练，smoke 与 pilot 分目录。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
MODEL="${TRAINPILOT_SFT_MODEL:-$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model}"
ADAPTER="${TRAINPILOT_V5_ADAPTER:-runs/init_v5/adapter}"
RUN="${TRAINPILOT_GRPO_RUN:-runs/trainpilot_grpo_pilot}"
CONFIG="${TRAINPILOT_GRPO_CONFIG:-model/config/grpo_pilot.json}"
DATA="${TRAINPILOT_GRPO_DATA:-model/dataset/datasets/trainpilot_agent/grpo_v1}"
common=(--config "$CONFIG" --model "$MODEL" --adapter "$ADAPTER"
        --data-dir "$DATA" --device cuda)
case "${1:-help}" in
  check)
    python -m pip check
    tests=(test.test_code.test_grpo test.test_code.test_agent_rewards)
    if [[ -n "${TRAINPILOT_GRPO_EXTRA_TEST:-}" ]]; then tests+=("$TRAINPILOT_GRPO_EXTRA_TEST"); fi
    python -m unittest "${tests[@]}" -q
    python -u -m model.trainer.qwen_grpo check "${common[@]}" --output-dir "$RUN"
    ;;
  probe)
    python -u -m model.trainer.qwen_grpo probe "${common[@]}" --output-dir "$RUN" "${@:2}"
    ;;
  smoke|train)
    extra=()
    output="$RUN"
    if [[ "$1" == smoke ]]; then
      output="$RUN/smoke"
      extra+=(--max-updates 2 --eval-limit 8)
    fi
    if [[ -f "$output/checkpoint.pt" ]]; then
      extra+=(--resume "$output/checkpoint.pt")
    fi
    mkdir -p "$output"
    python -u -m model.trainer.qwen_grpo train "${common[@]}" --output-dir "$output" \
      --probe-file "$RUN/probe_summary.json" "${extra[@]}" "${@:2}"
    ;;
  eval)
    python -u -m model.trainer.qwen_grpo eval "${common[@]}" --output-dir "$RUN/evaluation" \
      --policy-adapter "$RUN/adapter" "${@:2}"
    ;;
  *) echo "用法：bash model/trainer/run_trainpilot_grpo.sh check|probe|smoke|train|eval [额外参数]" ;;
esac
