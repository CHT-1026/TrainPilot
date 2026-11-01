"""Explicit evaluation CLI. No command is run by import or server startup.

prepare: freeze inputs without model calls.
generate: explicitly enabled model calls, one question at a time.
score: offline comparison and a model-anonymous review queue.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='仅冻结题目/答案/提示词，零模型调用')
    generation = commands.add_parser('generate', help='后续启用后逐题作答，支持中断续跑')
    generation.add_argument('--run', required=True)
    generation.add_argument('--model', choices=('A', 'B', 'C', 'D'), required=True)
    grading = commands.add_parser('score', help='只在本地读取标准答案并判分，不调用模型')
    grading.add_argument('--run', required=True)
    grading.add_argument('--models', choices=('A', 'B', 'C', 'D'), nargs='+', default=['A', 'B'])
    grading.add_argument('--reviews', type=Path, help='已记录理由与复核者的JSONL；不传时导出待复核队列')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local', override=False)
    load_dotenv(ROOT/'.env', override=False)
    from app.evaluation.config import load_config
    config = load_config()
    if args.command == 'prepare':
        from app.evaluation.dataset import prepare_run
        result = prepare_run(config)
    elif args.command == 'generate':
        from app.evaluation.runner import generate
        result = generate(config, args.run, args.model)
    else:
        from app.evaluation.grading import score
        if len(set(args.models)) != len(args.models):
            parser.error('--models不能重复')
        result = score(config, args.run, args.models, args.reviews)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
