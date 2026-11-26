#!/usr/bin/env bash
# V5现有adapter接续V6；smoke的10次更新包含在完整1轮的130次更新内。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
MODEL="${TRAINPILOT_SFT_MODEL:-$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model}"
INITIAL="${TRAINPILOT_V5_ADAPTER:-$HOME/TrainPilot-GRPO-V2/runs/init_v5/adapter}"
RUN="${TRAINPILOT_LORA_RUN:-runs/trainpilot_agent_lora_v6}"
common=(--config model/config/lora_v6.json --model "$MODEL" --init-adapter "$INITIAL" --output-dir "$RUN")
case "${1:-help}" in
  check)
    python -m pip check
    python -m model.trainer.check_trainpilot_lora_v6 --model "$MODEL" --init-adapter "$INITIAL"
    python -m unittest model.test.test_code.test_lora_continue -v
    python -m model.dataset.check_agent_v6 --tokenizer-dir "$MODEL"
    python -u -m model.trainer.qwen_lora_sft "${common[@]}" --prepare-only
    echo "V6_CHECK_OK：底座/V5、数据和入口检查完成，尚未更新真实模型。"
    ;;
  smoke|train)
    mkdir -p "$RUN"
    command -v flock >/dev/null || { echo "需要服务器上的flock工具来避免重复写同一训练目录。" >&2; exit 1; }
    exec 9>"$RUN/.train.lock"
    flock -n 9 || { echo "该结果目录已有训练在运行，请勿重复启动。" >&2; exit 1; }
    python -m model.trainer.check_trainpilot_lora_v6 --model "$MODEL" --init-adapter "$INITIAL"
    extra=()
    if [[ -f "$RUN/checkpoint.pt" ]]; then
      extra+=(--resume "$RUN/checkpoint.pt")
      echo "恢复本轮V6断点：$RUN/checkpoint.pt"
    else
      echo "从V5现有adapter开始V6；新优化器，目标1轮/130次更新。"
    fi
    if [[ "$1" == smoke ]]; then extra+=(--max-updates 10); fi
    python -u -m model.trainer.qwen_lora_sft "${common[@]}" "${extra[@]}" "${@:2}"
    ;;
  *) echo "用法：bash model/trainer/run_trainpilot_lora_v6.sh check|smoke|train [训练参数]" ;;
esac
