"""V6接续训练前核对实际Full SFT底座、V5 adapter和CUDA。"""
import argparse
import hashlib
import json
from pathlib import Path

BASE_CONFIG_SHA = "60ec90f5c9b44d9e5c4f2d95a33b871fd55a6e4ccba1d356a54ee51769b02a8e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--init-adapter',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('model/dataset/datasets/trainpilot_agent/v6'))
    args=p.parse_args()
    manifest=json.loads((args.data_dir/'manifest.json').read_text(encoding='utf-8'))
    if sha(args.init_adapter/'adapter_model.safetensors')!=manifest['initial_adapter_sha256']:
        raise ValueError('起始权重不是已确认的V5最终adapter，不能误用GRPO50或其他版本')
    if sha(args.init_adapter/'adapter_config.json')!=manifest['initial_adapter_config_sha256']:
        raise ValueError('起始V5 adapter配置已改变，请保持原rank、alpha和目标层配置')
    if sha(args.model/'model.config.json')!=BASE_CONFIG_SHA:
        raise ValueError('底座配置与已确认的Full SFT导出版本不同，请先核对路径')
    index=args.model/'model.safetensors.index.json'
    weights=[args.model/n for n in set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())] if index.exists() else list(args.model.glob('*.safetensors'))
    if not weights or any(not f.is_file() for f in weights):
        raise ValueError('Full SFT模型分片缺失')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('这份服务器配置需要CUDA与BF16支持')
    x=torch.ones(32,32,device='cuda',dtype=torch.bfloat16)
    assert (x@x)[0,0].item()==32
    free,total=torch.cuda.mem_get_info()
    print('Full SFT:',args.model.resolve())
    print('V5 adapter:',args.init_adapter.resolve())
    print('GPU:',torch.cuda.get_device_name(0),'显存空闲/总GiB:',round(free/1024**3,2),round(total/1024**3,2))
    print('SOURCE_AND_CUDA_OK；本轮建立新优化器，不加载V5旧训练断点。')


if __name__=='__main__':main()
