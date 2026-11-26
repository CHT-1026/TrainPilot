#!/usr/bin/env bash
# 从已有 Full SFT 导出模型继续 LoRA；先 check，再 smoke，最后 train。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_DIR="${TRAINPILOT_SFT_MODEL:?请将 TRAINPILOT_SFT_MODEL 设为已完成Full SFT的model目录}"
VERSION="${2:-v2}"
case "$VERSION" in
  v2) DATA_CHECK_MODULE="model.dataset.build_agent_data" ;;
  v3) DATA_CHECK_MODULE="model.dataset.build_agent_v3" ;;
  v4) DATA_CHECK_MODULE="model.dataset.build_agent_v4" ;;
  v5) DATA_CHECK_MODULE="model.dataset.build_agent_v5" ;;
  *) echo "数据版本仅支持 v2、v3、v4 或 v5。" >&2; exit 2 ;;
esac
DATA_DIR="model/dataset/datasets/trainpilot_agent/$VERSION"
RUN_DIR="${TRAINPILOT_LORA_RUN:-runs/trainpilot_agent_lora_$VERSION}"
export TRAINPILOT_SFT_MODEL="$MODEL_DIR"
export TRAINPILOT_AGENT_DATA="$DATA_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
MAX_LENGTH=1536
if [[ "$VERSION" == v5 ]]; then
  MAX_LENGTH=2048
fi

common=(
  --model "$MODEL_DIR" --data-path "$DATA_DIR/train.jsonl"
  --validation-data-path "$DATA_DIR/val.jsonl" --output-dir "$RUN_DIR"
  --device cuda --dtype bfloat16 --optimizer adamw
  --batch-size 1 --accumulation-steps 8 --max-length "$MAX_LENGTH"
  --epochs 2 --learning-rate 5e-5 --cpu-threads 4 --log-steps 10
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --target-modules q_proj k_proj v_proj o_proj
  --save-training-state --save-steps 50
)
if [[ "$VERSION" == v4 || "$VERSION" == v5 ]]; then
  common+=(--save-epoch-adapters)
fi

case "${1:-help}" in
  check)
    python -m pip check
    python - <<'PY'
import json, os, shutil, hashlib
from pathlib import Path
import datasets
import torch
from transformers import AutoConfig, AutoTokenizer
model_dir = Path(os.environ['TRAINPILOT_SFT_MODEL']).expanduser().resolve(strict=True)
config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
weights = list(model_dir.glob('*.safetensors')) + list(model_dir.glob('pytorch_model*.bin'))
assert weights, '指定目录没有HF模型权重；checkpoint.pt需先导出，不能作为--model目录'
index_path = model_dir/'model.safetensors.index.json'
if index_path.exists():
    shards = set(json.loads(index_path.read_text(encoding='utf-8'))['weight_map'].values())
    missing = [name for name in shards if not (model_dir/name).is_file()]
    assert not missing, f'模型分片缺失: {missing}'
print('Full SFT模型目录:', model_dir)
print('架构:', config.model_type, '层数:', config.num_hidden_layers, '隐藏维度:', config.hidden_size)
print('权重文件:', [p.name for p in weights])
print('Tokenizer长度:', len(tokenizer), 'EOS:', tokenizer.eos_token_id)
print('PyTorch:', torch.__version__, '构建CUDA:', torch.version.cuda)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '训练需要可用CUDA与BF16'
free, total = torch.cuda.mem_get_info()
print('GPU:', torch.cuda.get_device_name(0), '显存总量/空闲GiB:', round(total/1024**3,2), round(free/1024**3,2))
x = torch.ones(32, 32, device='cuda', dtype=torch.bfloat16)
assert (x @ x)[0,0].item() == 32
print('CUDA实际运算: 通过')
root = Path(os.environ['TRAINPILOT_AGENT_DATA'])
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for name, item in manifest['outputs'].items():
    assert hashlib.sha256((root/item['file']).read_bytes()).hexdigest() == item['sha256'], name
print('数据SHA256核对:', {k:v['records'] for k,v in manifest['outputs'].items()})
print('数据版本:', manifest['version'])
if manifest['version'] in ('v3', 'v4', 'v5'):
    assert manifest['training_start'] == 'original_full_sft_with_fresh_lora'
    import math
    updates = math.ceil(manifest['outputs']['train']['records']/8)*2
    print('训练计划: 2轮，batch=1，累积8次，总更新步数=', updates)
print('项目所在磁盘可用GiB:', round(shutil.disk_usage('.').free/1024**3,2))
PY
    if [[ "$VERSION" == v4 ]]; then
      python -m "$DATA_CHECK_MODULE" --output-dir "$DATA_DIR" --tokenizer-dir "$MODEL_DIR"
    else
      python -m "$DATA_CHECK_MODULE" --output-dir "$DATA_DIR" --tokenizer-dir "$MODEL_DIR" --max-length "$MAX_LENGTH"
    fi
    python -u -m model.trainer.qwen_lora_sft "${common[@]}" --device cpu --dtype float32 --prepare-only
    echo "检查完成；尚未启动训练。请核对模型目录确实对应上次Full SFT结果。"
    ;;
  smoke|train)
    extra=()
    if [[ -f "$RUN_DIR/checkpoint.pt" ]]; then
      extra+=(--resume "$RUN_DIR/checkpoint.pt")
      echo "恢复本轮 $VERSION 断点：$RUN_DIR/checkpoint.pt"
    else
      echo "从原 Full SFT 新建 $VERSION LoRA；结果目录：$RUN_DIR"
    fi
    if [[ "$1" == smoke ]]; then
      extra+=(--max-updates 10)
    fi
    mkdir -p "$RUN_DIR"
    python -u -m model.trainer.qwen_lora_sft "${common[@]}" "${extra[@]}"
    ;;
  *)
    echo "用法：bash model/trainer/run_trainpilot_lora.sh check|smoke|train [v2|v3|v4|v5]"
    echo "check 仅检查环境、模型与数据；smoke 更新10次；train 从断点继续完成总计2轮。"
    ;;
esac
