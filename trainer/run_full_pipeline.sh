#!/usr/bin/env bash
# 完整语料：分阶段下载，Dense 预训练完成后清理其语料，再执行 SFT。
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_DISABLE_XET=1
export HF_HOME="$repo_root/dataset/datasets/cache/huggingface"
export TMPDIR="$repo_root/dataset/datasets/cache/tmp"
export TRAINPILOT_FULL_RUN="runs/full_dense_20261015"
revision="312afb4f76391145c6902f765bb51691c09a12f5"
mkdir -p "$HF_HOME" "$TMPDIR" "$TRAINPILOT_FULL_RUN/pretrain" "$TRAINPILOT_FULL_RUN/sft"

echo "代码版本：$(git rev-parse HEAD)"
df -h .
python - <<'PY'
import torch
assert torch.cuda.is_available(), '当前环境不能使用 CUDA'
assert torch.cuda.is_bf16_supported(), '当前 GPU 不支持 BF16'
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY

# 单独生成完整版配置，保留原来的 mini 实验配置。
python - <<'PY'
import json
import os
from pathlib import Path

run = os.environ['TRAINPILOT_FULL_RUN']
for stage, template, length, batch, accumulation in (
    ('pretrain', 'config/pretrain.json', 380, 32, 8),
    ('sft', 'config/sft_train.json', 768, 16, 1),
):
    cfg = json.loads(Path(template).read_text(encoding='utf-8'))
    cfg['stage'] = stage
    cfg['model']['max_seq_len'] = length
    cfg['data'] = {
        'tokenizer_path': 'dataset/tokenizer.json' if stage == 'pretrain' else 'dataset',
        'train_path': f'dataset/datasets/processed/full/{stage}/train.jsonl',
        'val_path': f'dataset/datasets/processed/full/{stage}/val.jsonl',
    }
    cfg['training'].update(
        device='cuda', dtype='bfloat16', epochs=2, batch_size=batch,
        accumulation_steps=accumulation, num_workers=0, cpu_threads=2,
        output_dir=f'{run}/{stage}', save_interval=500 if stage == 'pretrain' else 1000,
    )
    if stage == 'sft':
        cfg['base_checkpoint'] = f'{run}/pretrain/checkpoint.pt'
    path = Path(f'config/{stage}_full.json')
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('训练配置:', path)
PY

# set -e + pipefail：任一准备/训练失败，后续阶段停止。
for stage in pretrain sft; do
    config="config/${stage}_full.json"
    split_dir="dataset/datasets/processed/full/$stage"
    output="$TRAINPILOT_FULL_RUN/$stage"
    raw="dataset/datasets/${stage}_t2t.jsonl"
    echo "===== $stage：$(date -Iseconds) ====="
    if [[ -f "$output/.complete" && -f "$output/checkpoint.pt" ]]; then
        echo "$stage 已完成，跳过。"
        if [[ "$stage" == pretrain ]]; then
            rm -f -- "$split_dir/train.jsonl" "$split_dir/val.jsonl"
        fi
        continue
    fi
    if [[ ! -f "$split_dir/manifest.json" ]]; then
        hf download jingyaogong/minimind_dataset "${stage}_t2t.jsonl" \
            --repo-type dataset --revision "$revision" --local-dir dataset/datasets
        if [[ "$stage" == pretrain ]]; then
            source_sha=31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d
        else
            source_sha=b5dab3d590b3b4ae3bfeec891ba0c76b0de5acbec4f4b078dcd3912eb1735c79
        fi
        python -u -m trainer.train --config "$config" \
            --prepare-data "$raw" --source-sha256 "$source_sha" \
            --data-output-dir "$split_dir" --val-fraction 0.01 --split-seed 43 \
            2>&1 | tee -a "$output/prepare.log"
    fi
    cp "$split_dir/manifest.json" "$output/data_manifest.json"
    # 划分已发布；训练只读取划分结果，不再保留完整原始副本。
    rm -f -- "$raw"
    df -h .
    resume_args=()
    if [[ -f "$output/checkpoint.pt" ]]; then
        resume_args=(--resume "$output/checkpoint.pt")
    fi
    python -u -m trainer.train --config "$config" "${resume_args[@]}" \
        2>&1 | tee -a "$output/train.log"

    # 只有完整训练结束且权重能被严格加载，才清理预训练的划分结果。
    python - "$config" "$output/checkpoint.pt" <<'PY'
import json
import sys
import torch
from config.config import TrainPilotconfig
from model.model import TrainPilotForCausalLM

torch.set_num_threads(2)
config = json.load(open(sys.argv[1], encoding='utf-8'))
checkpoint = torch.load(sys.argv[2], map_location='cpu', weights_only=True)
assert checkpoint['completed_epochs'] == config['training']['epochs'], '训练尚未完成'
model = TrainPilotForCausalLM(TrainPilotconfig(**checkpoint['model_config']))
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
with torch.no_grad():
    logits = model(torch.tensor([[1, 2]]))
assert torch.isfinite(logits).all(), '权重前向输出含 NaN/Inf'
print('完整训练及权重加载检查通过:', sys.argv[2])
PY
    touch "$output/.complete"
    if [[ "$stage" == pretrain ]]; then
        rm -f -- "$split_dir/train.jsonl" "$split_dir/val.jsonl"
        echo "已清理预训练语料；checkpoint、数据摘要和训练日志已保留。"
    fi
    df -h .
done

echo "两个训练阶段已完成。"
echo "基础权重：$repo_root/$TRAINPILOT_FULL_RUN/pretrain/checkpoint.pt"
echo "SFT 权重：$repo_root/$TRAINPILOT_FULL_RUN/sft/checkpoint.pt"
