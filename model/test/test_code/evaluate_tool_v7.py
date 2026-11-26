"""冻结小型工具环境的生成评测；工具真实执行，不向模型泄漏teacher轨迹/答案。"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--split',choices=['val','test'],default='test')
    args=parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from model.dataset.tool_sft import DATA,SandboxTools
    from model.trainer.tool_rollout import run_tool_chat
    from model.test.test_code.qwen_tool_protocol import parse_tool_output,template_messages
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    model=PeftModel.from_pretrained(base,args.adapter,is_trainable=False).to('cuda').eval()
    contract=json.loads((DATA/'contract.json').read_text(encoding='utf-8'))
    schemas={t['function']['name']:t['function']['parameters'] for t in contract['tools']}
    def complete(messages,tools,max_tokens,timeout):
        inputs=tokenizer.apply_chat_template(template_messages(messages),tools=tools,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
        if inputs.shape[1]+max_tokens>model.config.max_position_embeddings:
            raise RuntimeError('context_budget_exhausted')
        eos=list({tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')})
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.generate(inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=max_tokens,
                               do_sample=False,eos_token_id=eos,pad_token_id=tokenizer.pad_token_id)
        generated=out[0,inputs.shape[1]:].tolist()
        text=tokenizer.decode(generated,skip_special_tokens=True)
        truncated=bool(generated and generated[-1] not in eos and len(generated)>=max_tokens)
        message={'role':'assistant','content':text}; error=None; protocol=None
        if not truncated:
            try: message=parse_tool_output(text,bare_names=schemas)
            except ValueError as exc: error=str(exc)
            if message.get('tool_calls'): protocol='hermes' if '<tool_call>' in text else 'bare_json_compat'
        finish='length' if truncated else 'tool_calls' if message.get('tool_calls') else 'stop'
        return {'choices':[{'message':message,'finish_reason':finish}],
                'local_trace':{'raw_output':tokenizer.decode(generated,skip_special_tokens=False),
                               'tool_protocol_used':protocol,'tool_parse_error':error}}
    rows=[json.loads(s) for s in (DATA/f'{args.split}.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r['mode']=='tool_trajectory']
    # Interleave families for a useful small smoke subset.
    rows.sort(key=lambda r:(int(r['id'].rsplit('-',1)[1])//8,r['category']))
    if args.limit: rows=rows[:args.limit]
    if args.output.exists(): raise FileExistsError('评测结果已存在，请换输出路径；不覆盖旧结果')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    total,passed,calls,standard=0,0,0,0
    with args.output.open('w',encoding='utf-8') as stream:
        for row in rows:
            backend=SandboxTools(json.loads(row['sandbox']),schemas)
            outcome=run_tool_chat(row['messages'][1]['content'],backend,complete,timeout=300)
            final=outcome['final']; expected=json.loads(row['expected'])
            ok=bool(final and final['status']==expected['status'] and all(f in final['answer'] for f in expected['facts']))
            # Missing/conflict/clarify use status-based grading, NOT an independently measured semantic accuracy.
            n=sum(len(e.get('tools',[])) for e in outcome['trace'])
            if row['category']=='missing': ok=ok and n>0
            if row['category']=='conflict': ok=ok and len({c['doc_id'] for c in (final or {}).get('citations',[])})>=2
            if row['category']=='clarify': ok=ok and n==0
            h=sum(len(e.get('tools',[])) for e in outcome['trace'] if e.get('response',{}).get('local_trace',{}).get('tool_protocol_used')=='hermes')
            total+=1; passed+=ok; calls+=n; standard+=h
            stream.write(json.dumps({'id':row['id'],'category':row['category'],'rule_pass':ok,
                'expected':expected,'model':args.model,'adapter':args.adapter,**outcome},ensure_ascii=False)+'\n'); stream.flush()
            print(row['id'],outcome['status'],'rule_pass=',ok,flush=True)
    summary={'total':total,'rule_pass':passed,'tool_calls':calls,'hermes_calls':standard,
             'scope':'synthetic sandbox rule checks; not peer-RAG or independent semantic accuracy'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary)


if __name__=='__main__': main()
