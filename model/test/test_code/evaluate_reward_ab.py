"""对照试验用严格业务验收及固定证据RAG回归；不以训练奖励判胜。"""
import argparse
import json
from pathlib import Path

from model.trainer.reward_v2 import graded_intent_reward
from model.test.test_code.evaluate_grpo_regression import digest, greedy, grade, rows, save
from model.test.test_code.evaluate_agent_v5 import summary

ROOT=Path(__file__).resolve().parents[3]


def business_stats(records,suite):
    selected=[r for r in records if r['suite']==suite and r['case']['mode']=='intent_parse']
    return {"total":len(selected),"strict_correct":sum(r['passed'] for r in selected),
            "severe_errors":sum(r['business_diagnostics']['severe_error'] for r in selected),
            "invalid_protocol":sum(not r['business_diagnostics']['format_ok'] for r in selected)}


def decision(results,finished):
    baseline={s:business_stats(results['v6'],s) for s in ('development','old_regression')}
    checks={}
    for seed in (42,43):
        strict=results[f'strict_{seed}'];graded=results[f'graded_{seed}']
        current={s:business_stats(graded,s) for s in baseline}
        control={s:business_stats(strict,s) for s in baseline}
        checks[str(seed)]={
            'finished':finished[f'strict_{seed}'] and finished[f'graded_{seed}'],
            'development_improves':current['development']['strict_correct']>control['development']['strict_correct']
                and current['development']['strict_correct']>=baseline['development']['strict_correct'],
            'old_business_preserved':current['old_regression']['strict_correct']>=max(baseline['old_regression']['strict_correct'],control['old_regression']['strict_correct']),
            'no_extra_severe_errors':all(current[s]['severe_errors']<=min(baseline[s]['severe_errors'],control[s]['severe_errors']) for s in baseline),
            'protocol_preserved':all(current[s]['invalid_protocol']<=min(baseline[s]['invalid_protocol'],control[s]['invalid_protocol']) for s in baseline)}
    passed=all(all(values.values()) for values in checks.values())
    return {"recommendation":"business_candidate_pending_manual_rag_review" if passed else "keep_strict_reward",
            "checks":checks,"deployed_model_changed":False,
            "note":"不自动部署；即使业务门槛通过，仍需逐条检查RAG与自然回答退化。两seed是小试，不证明统计显著或普遍泛化。"}


def run(args):
    import datasets
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4)
    cfg=json.loads((ROOT/'model/config/grpo_reward_ab.json').read_text(encoding='utf-8'))
    base_hashes={p.name:digest(p) for p in sorted(args.model.glob('*.safetensors'))}
    assert base_hashes==json.loads((ROOT/'model/config/agent_diagnosis.json').read_text(encoding='utf-8'))['base_weight_sha256']
    panel=rows(ROOT/'runs/reward-ab-data-20261030/panel.jsonl')
    assert len(panel)==269
    adapters={'v6':ROOT/cfg['initial_adapter']}
    finished={}
    for seed in cfg['seeds']:
        for method in ('strict','graded'):
            name=f'{method}_{seed}';directory=args.trial_dir/name
            metadata=json.loads((directory/'completed.json').read_text(encoding='utf-8'))
            finished[name]=metadata['finished']
            adapters[name]=directory/'adapter'
    signature={"base_weights":base_hashes,"panel":digest(ROOT/'runs/reward-ab-data-20261030/panel.jsonl'),
               "adapters":{k:digest(p/'adapter_model.safetensors') for k,p in adapters.items()},
               "evaluator":digest(__file__),"reward":digest(ROOT/'model/trainer/reward_v2.py'),
               "base_config":digest(args.model/'model.config.json'),"decode":"BF16 greedy, adapter FP32; 512 rag/256 other"}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=args.output_dir/'run_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text(encoding='utf-8'))==signature
    else:save(manifest,signature)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,str(adapters['v6']),is_trainable=False,autocast_adapter_dtype=True)
    for name,path in adapters.items():
        if name!='v6':model.load_adapter(str(path),adapter_name=name,is_trainable=False,autocast_adapter_dtype=True)
    results={}
    with torch.inference_mode():
        for name in adapters:
            model.set_adapter('default' if name=='v6' else name);model.requires_grad_(False);model.eval()
            file=args.output_dir/(name+'.jsonl');completed=rows(file) if file.exists() else []
            assert [r['case'] for r in completed]==[r['case'] for r in panel[:len(completed)]]
            with file.open('a',encoding='utf-8') as stream:
                for i,item in enumerate(panel[len(completed):],len(completed)+1):
                    case=item['case'];ids=tokenizer.apply_chat_template(item['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=greedy(model,tokenizer,ids,512 if case['mode'].endswith('_rag') else 256,'cuda')
                    scored=grade(case,output['text'],output['stop_reason']=='eos')
                    result={**item,"variant":name,**output,**scored}
                    if case['mode']=='intent_parse':result['business_diagnostics']=graded_intent_reward(output['text'],case['expected'],ended=output['stop_reason']=='eos')
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();completed.append(result)
                    if i%10==0 or i==len(panel):print(name,i,'/',len(panel),flush=True)
            results[name]=completed
            save(args.output_dir/(name+'_summary.json'),{s:summary([r for r in completed if r['suite']==s]) for s in ('development','old_regression')})
    result=decision(results,finished)
    save(args.output_dir/'decision.json',result)
    assert all(digest(p/'adapter_model.safetensors')==signature['adapters'][k] for k,p in adapters.items())
    save(args.output_dir/'completed.json',{"variants":list(adapters),"cases_per_variant":269,"training_performed":False})
    print('AB_EVALUATION_COMPLETE',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--trial-dir',type=Path,default=ROOT/'runs/grpo_reward_ab_20261030/trials')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/grpo_reward_ab_20261030/evaluation')
    run(p.parse_args())
