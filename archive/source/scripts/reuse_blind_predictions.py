"""Import identical approved-model requests without changing historical predictions."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--target',required=True)
    p.add_argument('--model',choices=('A','B'),required=True)
    args=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    from app.evaluation.config import load_config
    from app.evaluation.reuse import reuse_completed
    print(json.dumps(reuse_completed(load_config(),args.target,args.source,args.model),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
