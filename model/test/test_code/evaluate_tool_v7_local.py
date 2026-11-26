"""同一本机BF16底座串行评测V7/V6；逐题保存，可校验后续跑。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/tool_v7_eval_20261105')
    args=p.parse_args(); out=args.run_dir; snapshot=out/'snapshot'
    import datasets
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from model.dataset.tool_sft import DATA,SandboxTools
    from model.trainer.tool_rollout import run_tool_chat
    from model.test.test_code.qwen_tool_protocol import parse_tool_output,template_messages
    from model.test.test_code.evaluate_qwen_sft import load_model
    torch.set_num_threads(4)
    data=snapshot/'model/dataset/datasets/trainpilot_agent/tool_v7'
    assert (data/'test.jsonl').read_bytes()==(DATA/'test.jsonl').read_bytes()
    assert (data/'contract.json').read_bytes()==(DATA/'contract.json').read_bytes()
    contract=json.loads((data/'contract.json').read_text(encoding='utf-8'))
    adapters={'v7':snapshot/'runs/trainpilot_tool_lora_v7/adapter','v6':ROOT/'runs/trainpilot_agent_lora_v6/adapter'}
    checkpoint=ROOT/'runs/qwen-fullsft/checkpoint.pt'; config=ROOT/'runs/qwen-base-reference'
    tokenizer=AutoTokenizer.from_pretrained(config,local_files_only=True)
    for path in adapters.values():
        other=AutoTokenizer.from_pretrained(path,local_files_only=True)
        assert other.get_vocab()==tokenizer.get_vocab() and other.chat_template==tokenizer.chat_template
    signature={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
               'adapters':{k:sha(v/'adapter_model.safetensors') for k,v in adapters.items()},
               'test_sha256':sha(data/'test.jsonl'),'contract_sha256':sha(data/'contract.json'),
               'runner_sha256':sha(Path(__file__)),'torch':str(torch.__version__),
               'transformers':__import__('transformers').__version__,
               'gpu':torch.cuda.get_device_name(0),'dtype':'base BF16, adapter FP32, BF16 autocast',
               'decode':'greedy; max_new_tokens=512; max_rounds=8; max_calls=12; timeout=300',
               'loader':'stream original Full-SFT FP32 checkpoint to BF16; no quantization'}
    signature_path=out/'local_signature.json'
    if signature_path.exists(): assert json.loads(signature_path.read_text(encoding='utf-8'))==signature
    else: signature_path.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SIGNATURE_OK',flush=True)
    print('GPU_FREE',torch.cuda.mem_get_info(),flush=True)
    model,_=load_model(checkpoint,config,'cuda',torch.bfloat16,reserve_bytes=256*1024**2)
    model=PeftModel.from_pretrained(model,str(adapters['v7']),adapter_name='v7',is_trainable=False)
    model.load_adapter(str(adapters['v6']),adapter_name='v6',is_trainable=False)
    model.eval()
    print('MODELS_LOADED',flush=True)
    schemas={t['function']['name']:t['function']['parameters'] for t in contract['tools']}
    rows=[json.loads(s) for s in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['mode']=='tool_trajectory']
    rows.sort(key=lambda r:(int(r['id'].rsplit('-',1)[1])//8,r['category']))
    expected_ids=[r['id'] for r in rows]
    def complete(messages,tools,max_tokens,timeout):
        ids=tokenizer.apply_chat_template(template_messages(messages),tools=tools,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
        if ids.shape[1]+max_tokens>model.config.max_position_embeddings: raise RuntimeError('context_budget_exhausted')
        eos=list({tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')})
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=max_tokens,
                do_sample=False,eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,logits_to_keep=1)
        generated=result[0,ids.shape[1]:].tolist()
        text=tokenizer.decode(generated,skip_special_tokens=True)
        truncated=bool(generated and generated[-1] not in eos and len(generated)>=max_tokens)
        message={'role':'assistant','content':text}; error=None; protocol=None
        if not truncated:
            try: message=parse_tool_output(text,bare_names=schemas)
            except ValueError as exc: error=str(exc)
            if message.get('tool_calls'): protocol='hermes' if '<tool_call>' in text else 'bare_json_compat'
        return {'choices':[{'message':message,'finish_reason':'length' if truncated else 'tool_calls' if message.get('tool_calls') else 'stop'}],
                'local_trace':{'raw_output':tokenizer.decode(generated,skip_special_tokens=False),
                               'tool_protocol_used':protocol,'tool_parse_error':error}}
    for variant in ['v7','v6']:
        model.set_adapter(variant); model.eval()
        path=out/f'{variant}_local.jsonl'
        done=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        assert [r['id'] for r in done]==expected_ids[:len(done)]
        with path.open('a',encoding='utf-8') as stream:
            for row in rows[len(done):]:
                outcome=run_tool_chat(row['messages'][1]['content'],SandboxTools(json.loads(row['sandbox']),schemas),complete,timeout=300)
                final=outcome['final']; expected=json.loads(row['expected'])
                ok=bool(final and final['status']==expected['status'] and all(f in final['answer'] for f in expected['facts']))
                n=sum(len(e.get('tools',[])) for e in outcome['trace'])
                if row['category']=='missing': ok=ok and n>0
                if row['category']=='conflict': ok=ok and len({c['doc_id'] for c in (final or {}).get('citations',[])})>=2
                if row['category']=='clarify': ok=ok and n==0
                h=sum(len(e.get('tools',[])) for e in outcome['trace'] if e.get('response',{}).get('local_trace',{}).get('tool_protocol_used')=='hermes')
                record={'id':row['id'],'category':row['category'],'rule_pass':ok,'expected':expected,
                        'variant':variant,'adapter_sha256':signature['adapters'][variant],
                        'hermes_calls':h,**outcome}
                stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush(); done.append(record)
                print(variant,len(done),len(rows),row['id'],outcome['status'],ok,round(outcome['seconds'],2),flush=True)
        summary={'total':len(done),'complete':[r['id'] for r in done]==expected_ids,
                 'rule_pass':sum(bool(r['rule_pass']) for r in done),'tool_calls':sum(r['tool_calls'] for r in done),
                 'hermes_calls':sum(r['hermes_calls'] for r in done),'statuses':dict(Counter(r['status'] for r in done)),
                 'per_category':{c:{'total':sum(r['category']==c for r in done),
                                   'rule_pass':sum(r['category']==c and r['rule_pass'] for r in done)} for c in sorted({r['category'] for r in done})}}
        path.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('VARIANT_DONE',variant,json.dumps(summary,ensure_ascii=False),flush=True)
    print('LOCAL_TOOL_EVAL_DONE 120 120',flush=True)


if __name__=='__main__': main()
