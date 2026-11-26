#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
MODEL="${TRAINPILOT_SFT_MODEL:-$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model}"
INITIAL="${TRAINPILOT_V6_ADAPTER:-runs/init_v6/adapter}"
RUN="${TRAINPILOT_TOOL_RUN:-runs/trainpilot_tool_lora_v7}"
common=(--config model/config/lora_tool_v7.json --model "$MODEL" --init-adapter "$INITIAL" --output-dir "$RUN")
case "${1:-help}" in
  check)
    python -m pip check
    python verify_packet.py
    python -m unittest model.test.test_code.test_tool_sft test.test_code.test_lora_continue -v
    python -m model.trainer.check_tool_run --model "$MODEL" --init-adapter "$INITIAL"
    python -m model.dataset.check_tool_sft --tokenizer-dir "$MODEL"
    python -m model.trainer.qwen_lora_sft "${common[@]}" --prepare-only
    echo "V7_CHECK_OK: 尚未更新权重"
    ;;
  smoke|train)
    mkdir -p "$RUN"
    exec 9>"$RUN/.train.lock"
    flock -n 9 || { echo '同一输出目录已有训练在运行' >&2; exit 1; }
    python -m model.trainer.check_tool_run --model "$MODEL" --init-adapter "$INITIAL"
    extra=()
    if [[ -f "$RUN/checkpoint.pt" ]]; then extra+=(--resume "$RUN/checkpoint.pt"); fi
    if [[ "$1" == smoke ]]; then extra+=(--max-updates 10); fi
    python -u -m model.trainer.qwen_lora_sft "${common[@]}" "${extra[@]}" "${@:2}"
    ;;
  eval-before|eval-after)
    adapter="$INITIAL"; label=before
    if [[ "$1" == eval-after ]]; then adapter="$RUN/adapter"; label=after; fi
    python -u -m model.test.test_code.evaluate_tool_v7 --model "$MODEL" --adapter "$adapter" \
      --output "$RUN/eval_$label.jsonl" "${@:2}"
    ;;
  all)
    bash model/trainer/run_tool_sft_v7.sh check
    bash model/trainer/run_tool_sft_v7.sh smoke
    bash model/trainer/run_tool_sft_v7.sh train
    ;;
  *) echo '用法：bash model/trainer/run_tool_sft_v7.sh check|smoke|train|all|eval-before|eval-after' ;;
esac
