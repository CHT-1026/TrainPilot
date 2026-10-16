#!/usr/bin/env bash
# 在已激活的 Python 环境运行；只启动 Qwen Full SFT。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${TRAINPILOT_MODEL_DIR:-runs/models/Qwen3-1.7B-Base}"
DATA_FILE="dataset/datasets/alpaca_gpt4_zh/alpaca_gpt4_data_zh.json"
RUN_DIR="${TRAINPILOT_RUN_DIR:-runs/qwen_full_sft_24gb}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/runs/hf-cache}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

common=(
  --model "$MODEL_DIR" --data-path "$DATA_FILE" --output-dir "$RUN_DIR"
  --optimizer adamw8bit --dtype bfloat16 --batch-size 1
  --accumulation-steps 8 --max-length 512 --epochs 1
  --learning-rate 1e-5 --cpu-threads 4 --log-steps 10
  --save-training-state --save-steps 200
)

case "${1:-help}" in
  prepare)
    hf download Qwen/Qwen3-1.7B-Base \
      --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
      --local-dir "$MODEL_DIR" --max-workers 2 \
      --include '*.json' '*.safetensors' '*.txt' '*.jinja'
    hf download llamafactory/alpaca_gpt4_zh alpaca_gpt4_data_zh.json \
      --repo-type dataset --revision 065394ee242c43928298de8f43a8748ffd16f3e3 \
      --local-dir dataset/datasets/alpaca_gpt4_zh
    echo "628c3b100e32af85ec8d5338c42745f6b82900f97cf4deeda307b98f94ab34ab  $DATA_FILE" | sha256sum -c -
    python -u -m trainer.qwen_full_sft "${common[@]}" --prepare-only
    ;;
  smoke|train)
    # 使用已下载的本地模型与语料，训练不再依赖外网。
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    extra=()
    if [[ -f "$RUN_DIR/checkpoint.pt" ]]; then
      extra+=(--resume "$RUN_DIR/checkpoint.pt")
    fi
    if [[ "$1" == smoke ]]; then
      extra+=(--max-updates 20)
    fi
    mkdir -p "$RUN_DIR"
    python -u -m trainer.qwen_full_sft "${common[@]}" "${extra[@]}"
    ;;
  *)
    echo "用法：bash trainer/run_qwen_full_sft.sh prepare|smoke|train"
    echo "prepare 下载并编码数据；smoke 更新 20 次并保存；train 从断点继续完成 1 轮。"
    ;;
esac
