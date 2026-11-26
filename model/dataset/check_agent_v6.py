"""核对V6文件摘要、分组与可选的Qwen编码；不加载模型、不训练。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'datasets/trainpilot_agent/v6')
    p.add_argument('--tokenizer-dir',type=Path)
    p.add_argument('--max-length',type=int,default=2048)
    args=p.parse_args()
    manifest=json.loads((args.data_dir/'manifest.json').read_text(encoding='utf-8'))
    splits={};owners={};ids=set()
    for split,meta in manifest['outputs'].items():
        path=args.data_dir/meta['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'],split
        records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        assert len(records)==meta['records'],split
        for r in records:
            assert r['id'] not in ids,r['id'];ids.add(r['id'])
            assert owners.setdefault(r['group_id'],split)==split,r['group_id']
        splits[split]=records
    print('DATA_OK', {s:len(r) for s,r in splits.items()},flush=True)
    if not args.tokenizer_dir:return
    import datasets  # Windows先加载Arrow。
    from transformers import AutoTokenizer
    from model.dataset.dataset import QwenSFTEncoder
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_dir,local_files_only=True)
    encoder=QwenSFTEncoder(tokenizer,max_length=args.max_length)
    details=[];maximum={};bad=[]
    total=sum(len(r) for r in splits.values());i=0
    for split,records in splits.items():
        maximum[split]=0
        for row in records:
            full=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False,enable_thinking=False)
            encoded=encoder(row)
            count=sum(x!=-100 for x in encoded['labels'][1:])
            truncated=len(encoded['input_ids'])!=len(full)
            if truncated or count==0:bad.append(row['id'])
            maximum[split]=max(maximum[split],len(full))
            details.append({'id':row['id'],'split':split,'total_tokens':len(full),'supervised_targets':count,'truncated':truncated})
            i+=1
            if i==1 or i%200==0 or i==total:print('ENCODING',i,'/',total,flush=True)
    result={'tokenizer':str(args.tokenizer_dir.resolve()),'tokenizer_sha256':hashlib.sha256((args.tokenizer_dir/'tokenizer.json').read_bytes()).hexdigest(),
            'records':total,'max_length_checked':args.max_length,'max_tokens_per_split':maximum,
            'min_supervised_targets':min(r['supervised_targets'] for r in details),'invalid_records':bad,
            'no_truncation':not any(r['truncated'] for r in details),'training_performed':False,'cases':details}
    (args.data_dir/'encoding_check.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    if bad:raise ValueError(f'发现{len(bad)}条截断或无监督目标，详见encoding_check.json')
    print('ENCODING_OK',json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
