"""V7发包/训练前检查：真实工具重放、引用、分组、完整编码与tool返回屏蔽。"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from model.dataset.tool_sft import DATA, SandboxTools


def check(data_dir, tokenizer_dir, max_length=2048):
    from transformers import AutoTokenizer
    from model.dataset.dataset import QwenSFTEncoder
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    encoder=QwenSFTEncoder(tokenizer,max_length)
    manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    contract=json.loads((data_dir/'contract.json').read_text(encoding='utf-8'))
    schemas={t['function']['name']:t['function']['parameters'] for t in contract['tools']}
    groups,doc_ids,summary={},{},{}
    for split in ['train','val','test']:
        path=data_dir/f'{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['outputs'][split]['sha256'], f'{split} hash changed'
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        groups[split]={r['group_id'] for r in rows}; doc_ids[split]=set()
        lengths=[]; tool_steps=0; supervised=0
        for row in rows:
            messages=row['messages']; tools=json.loads(row['tools']) if row['tools'] else None
            if row['mode']=='tool_trajectory':
                assert messages[0]['content']==contract['system'] and tools==contract['tools']
                documents=json.loads(row['sandbox']); backend=SandboxTools(documents,schemas)
                doc_ids[split].update(d['doc_id'] for d in documents)
                delivered={}
                for i,message in enumerate(messages[:-1]):
                    if message.get('tool_calls'):
                        calls=json.loads(message['tool_calls']); assert len(calls)==1
                        call=calls[0]; result=backend.run(call['function']['name'],call['function']['arguments'])
                        returned=messages[i+1]
                        assert returned['role']=='tool' and returned['tool_call_id']==call['id']
                        assert json.loads(returned['content'])==result, row['id']
                        for s in result.get('hits',[])+result.get('segments',[]):
                            delivered[(s['doc_id'],s['locator'])]=s['text']
                        tool_steps+=1
                final=json.loads(messages[-1]['content']); expected=json.loads(row['expected'])
                assert final['status']==expected['status']
                assert all(f in final['answer'] for f in expected['facts'])
                if final['status']=='answerable': assert final['citations']
                for c in final['citations']:
                    assert c['quote'] in delivered[(c['doc_id'],c['locator'])]
            normalized=[]
            for message in messages:
                m=dict(message)
                if m.get('tool_calls'): m['tool_calls']=json.loads(m['tool_calls'])
                normalized.append(m)
            text=tokenizer.apply_chat_template(normalized,tools=tools,tokenize=False,add_generation_prompt=False,enable_thinking=False)
            full=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            encoded=encoder(row)
            assert len(full['input_ids'])<=max_length, f"{row['id']} requires {len(full['input_ids'])} tokens; no silent final-answer truncation allowed"
            assert encoded['input_ids']==full['input_ids']
            supervised+=sum(x!=-100 for x in encoded['labels'])
            assert any(x!=-100 for x in encoded['labels'][1:])
            # Tool response tokens must never become targets, even when they resemble assistant text.
            spans=[(m.start(1),m.end()) for m in re.finditer(r'<\|im_start\|>tool\n(.*?)<\|im_end\|>',text,re.S)]
            for label,(start,end) in zip(encoded['labels'],full['offset_mapping']):
                if end>start and any(left<=start and end<=right for left,right in spans): assert label==-100
            if row['mode']=='tool_trajectory' and tool_steps:
                assistant_spans=[(m.start(1),m.end(1)) for m in re.finditer(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',text,re.S)]
                for match in re.finditer(r'<tool_call>.*?</tool_call>',text,re.S):
                    if not any(left<=match.start()<right for left,right in assistant_spans): continue
                    assert any(label!=-100 and match.start()<=s<match.end() for label,(s,e) in zip(encoded['labels'],full['offset_mapping']))
            lengths.append(len(full['input_ids']))
        summary[split]={'records':len(rows),'max_tokens':max(lengths),'tool_steps':tool_steps,'supervised_tokens':supervised}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not groups[a]&groups[b], 'replay/scenario group overlap'
        assert not doc_ids[a]&doc_ids[b], 'sandbox document overlap'
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--tokenizer-dir',type=Path,required=True)
    parser.add_argument('--max-length',type=int,default=2048)
    args=parser.parse_args()
    result=check(args.data_dir,args.tokenizer_dir,args.max_length)
    print(json.dumps(result,ensure_ascii=False,indent=2)); print('TOOL_DATA_OK')


if __name__=='__main__': main()
