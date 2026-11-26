"""校验V6起点、完整Full-SFT分片与训练硬件；不更新参数。"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--init-adapter',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'model/dataset/datasets/trainpilot_agent/tool_v7/manifest.json').read_text(encoding='utf-8'))
    actual=hashlib.sha256((args.init_adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert actual==manifest['initial_adapter_sha256'], '不是冻结V6 adapter，请勿使用GRPO或其他LoRA起点'
    assert hashlib.sha256((args.model/'model.config.json').read_bytes()).hexdigest()=='60ec90f5c9b44d9e5c4f2d95a33b871fd55a6e4ccba1d356a54ee51769b02a8e', 'Full SFT配置与既定底座不一致'
    index=args.model/'model.safetensors.index.json'
    names=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    assert all((args.model/name).is_file() for name in names), '缺少底座分片'
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '需要CUDA与BF16'
    x=torch.ones(8,8,device='cuda',dtype=torch.bfloat16)
    assert (x@x)[0,0].item()==8
    free,total=torch.cuda.mem_get_info()
    print('GPU',torch.cuda.get_device_name(0),'free/total GiB',round(free/1024**3,2),round(total/1024**3,2))
    print('V6',actual,'底座',args.model.resolve()); print('V7_MODEL_CHECK_OK')


if __name__=='__main__': main()
