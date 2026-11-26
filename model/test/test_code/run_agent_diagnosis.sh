#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
MODE="${1:-run}"
MODEL="${TRAINPILOT_SFT_MODEL:-$HOME/TrainPilot-Qwen/runs/qwen_full_sft_24gb/model}"
case "$MODE" in
  smoke) OUTPUT="runs/agent_diagnosis_results_20261030/smoke"; EXTRA=(--smoke) ;;
  run) OUTPUT="runs/agent_diagnosis_results_20261030/full"; EXTRA=() ;;
  *) echo "用法：bash model/test/test_code/run_agent_diagnosis.sh smoke|run"; exit 2 ;;
esac
mkdir -p "$OUTPUT"
# 两种模式共享一把GPU任务锁，防止同时启动smoke和正式诊断。
exec 9> runs/agent_diagnosis_results_20261030/.gpu.lock
flock -n 9 || { echo "本目录已有诊断任务运行"; exit 1; }
python verify_packet.py
python -m unittest model.test.test_code.test_agent_diagnosis -v
python -u -m model.test.test_code.probe_agent_diagnosis --model "$MODEL" --output-dir "$OUTPUT" "${EXTRA[@]}" --check-only
python -u -m model.test.test_code.probe_agent_diagnosis --model "$MODEL" --output-dir "$OUTPUT" "${EXTRA[@]}"
