"""Load/unload only the user-named Qwen3.8 Q6/Q4 variants through the local server."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'C': 'qwen/qwen3.8-27b@q6_k', 'D': 'qwen/qwen3.8-27b@q4_k_m'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('load', 'unload', 'list'))
    parser.add_argument('--label', choices=('C', 'D'))
    parser.add_argument('--instance')
    parser.add_argument('--run', help='加载与卸载证据所属的冻结运行')
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    inventory = None
    if args.operation == 'list':
        with opener.open('http://127.0.0.1:1235/api/v1/models', timeout=15) as response:
            result = json.load(response)
    else:
        if not args.run:
            parser.error('load/unload需要--run，以保留独立运行的加载证据')
        receipt_dir = ROOT/'eval_bench/runs'/args.run/'private/local_runtime'
        receipt_dir.mkdir(parents=True, exist_ok=True)
        if args.operation == 'load':
            with opener.open('http://127.0.0.1:1235/api/v1/models', timeout=15) as response:
                inventory = json.load(response)
            target = next(model for model in inventory['models'] if model['key'] == 'qwen/qwen3.8-27b')
            if target['selected_variant'] != VARIANTS[args.label]:
                raise ValueError('请先在本地模型资源库选择对应量化版本，未加载其他版本')
            if target['loaded_instances']:
                raise ValueError('该模型已有运行实例，先核对其归属和量化版本，不覆盖已有实例')
            payload = {'model': target['key'], 'context_length': 16384,
                       'flash_attention': True, 'echo_load_config': True}
        else:
            payload = {'instance_id': args.instance}
        request = urllib.request.Request('http://127.0.0.1:1235/api/v1/models/' + args.operation,
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        try:
            with opener.open(request, timeout=600) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            result = {'http_status': exc.code, 'body': exc.read().decode('utf-8', errors='replace')}
        (receipt_dir/f'{args.label}_{args.operation}.json').write_text(json.dumps(
            {'request': payload, 'inventory_before': inventory, 'response': result}, ensure_ascii=False, indent=2), encoding='utf-8')
        if 'http_status' in result:
            raise RuntimeError(f"本地模型{args.operation}失败：HTTP {result['http_status']}；详情已保留在回执中")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
