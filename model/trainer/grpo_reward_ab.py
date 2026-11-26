"""从同一V6起点比较strict/graded奖励，复用既有GRPO数学与模型加载。"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from model.trainer.reward_v2 import graded_intent_reward

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs/reward-ab-data-20261030"


def prepare():
    from model.test.test_code.prepare_agent_diagnosis import read_rows, save, sha
    DATA.mkdir(exist_ok=False)
    runtime = read_rows(ROOT / "runs/agent-diagnosis-20261030/runtime_cases.jsonl")
    primary = [r for r in runtime if r["sampling_enabled"]]
    def convert(r):
        return {"id":r["id"], "category":r["case"]["category"],
                "prompt":r["inference_messages"], "gold":r["case"]["expected"]}
    train = [convert(r) for r in primary if r["source_split"] == "train" and r["case"]["mode"] == "intent_parse"]
    val = [convert(r) for r in primary if r["source_split"] == "development" and r["case"]["mode"] == "intent_parse"]
    panel = [{"suite":"development", "case":r["case"], "messages":r["inference_messages"]}
             for r in primary if r["source_split"] == "development"]
    old = read_rows(ROOT / "model/dataset/datasets/trainpilot_agent/grpo_v2_regression/regression_panel.jsonl")
    panel += [{"suite":"old_regression", "case":r, "messages":r["messages"][:-1]} for r in old]
    assert len(train)==75 and len(val)==70 and len(panel)==269
    assert not ({r['id'] for r in train} & {r['id'] for r in val})
    for name, rows in (("train",train),("val",val),("panel",panel)):
        (DATA/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    save(DATA/'manifest.json', {"counts":{"train":75,"val":70,"panel":269},
        "sha256":{s:sha(DATA/(s+'.jsonl')) for s in ('train','val','panel')},
        "selection":"全部75条训练侧业务提示，不按新奖励筛选；70条开发业务，104开发+165旧回归面板",
        "final_grpo_test_used":False,"train_intents":dict(Counter(r['gold']['intent'] for r in train))})
    print('AB_DATA_READY',75,70,269)


def score(text, gold, ended, method):
    result = graded_intent_reward(text, gold, ended=ended)
    return {**result, "training_reward":result['strict_reward'] if method=='strict' else result['reward']}


def validate(model, tokenizer, rows, args, output, label):
    from model.trainer import qwen_grpo as core
    records=[]
    for row in rows:
        answer=core.generate_one(model,tokenizer,row,args,False)
        records.append({"id":row['id'],"category":row['category'],**answer,
                        **score(answer['text'],row['gold'],answer['ended'],'strict')})
    core.write_json(output/(label+'_answers.json'),records)
    result={"total":len(records),"strict_correct":sum(r['strict_reward'] for r in records),
            "severe_errors":sum(r['severe_error'] for r in records),"invalid_protocol":sum(not r['format_ok'] for r in records)}
    core.write_json(output/(label+'.json'),result)
    print(label,result,flush=True)


def trial(args):
    import torch
    from peft import set_peft_model_state_dict
    from model.trainer import qwen_grpo as core
    cfg=json.loads((ROOT/'model/config/grpo_reward_ab.json').read_text(encoding='utf-8'))
    params=SimpleNamespace(**cfg,model=str(args.model),adapter=str(ROOT/cfg['initial_adapter']),device=args.device,seed=args.seed)
    if args.smoke: params.max_updates,params.max_sampled_groups=2,16
    assert core.digest(args.model/'model.config.json')==cfg['base_config_sha256']
    assert core.digest(Path(params.adapter)/'adapter_model.safetensors')==cfg['initial_adapter_sha256']
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    assert all(core.digest(DATA/(s+'.jsonl'))==h for s,h in manifest['sha256'].items())
    signature={"config":core.digest(ROOT/'model/config/grpo_reward_ab.json'),"data":manifest['sha256'],
               "method":args.method,"seed":args.seed,"smoke":args.smoke,
               "initial_adapter":cfg['initial_adapter_sha256'],"trainer":core.digest(__file__),
               "core":core.digest(Path(core.__file__)),"reward":core.digest(ROOT/'model/trainer/reward_v2.py'),
               "base_weights":{p.name:core.digest(p) for p in sorted(args.model.glob('*.safetensors'))}}
    expected=json.loads((ROOT/'model/config/agent_diagnosis.json').read_text(encoding='utf-8'))['base_weight_sha256']
    assert signature['base_weights']==expected
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    model,tokenizer=core.load_policy(params)
    train=core.encode_rows(core.read_rows(DATA/'train.jsonl'),tokenizer,params.max_prompt_tokens)
    val=core.encode_rows(core.read_rows(DATA/'val.jsonl'),tokenizer,params.max_prompt_tokens)
    core.activate(model)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=params.learning_rate,weight_decay=0.,foreach=False)
    progress={"global_step":0,"groups":0,"mixed_groups":0,"cycle":0,"cursor":0,"skipped_groups":0}
    checkpoint=output/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['signature']!=signature: raise ValueError('已有断点配置不同，请保留原结果并使用新目录')
        set_peft_model_state_dict(model,state['weights'],adapter_name='default')
        optimizer.load_state_dict(state['optimizer']);progress=state['progress']
        torch.set_rng_state(state['torch_rng'])
        if params.device=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        # 初始policy/reference应完全对应同一V6，而不是原Full SFT。
        target=tokenizer.encode(json.dumps(train[0]['gold'],ensure_ascii=False),add_special_tokens=False)
        with torch.no_grad(),core.autocast(params):
            a=core.completion_logps(model,train[0]['input_ids'],target,params.device)
            core.activate(model,'reference')
            b=core.completion_logps(model,train[0]['input_ids'],target,params.device)
        assert (a-b).abs().max().item()<1e-5
        core.activate(model)
        validate(model,tokenizer,val,params,output,'before_val')
    core.write_json(output/'run_config.json',{"signature":signature,"parameters":vars(params),
                    "limits":"相同50次有效更新上限/100组采样上限；未达到更新预算不可判胜"})
    while progress['global_step']<params.max_updates and progress['groups']<params.max_sampled_groups:
        order=core.balanced_order(train,params.seed+progress['cycle'])
        if progress['cursor']==len(order):progress['cycle']+=1;progress['cursor']=0;continue
        row=train[order[progress['cursor']]]
        torch.manual_seed(params.seed*100000+progress['groups'])
        answers,old,ref=core.rollout(model,tokenizer,row,params)
        scored=[score(a['text'],row['gold'],a['ended'],args.method) for a in answers]
        rewards=[r['training_reward'] for r in scored]
        eligible=any(r['eligible'] for r in scored)
        mixed=len(set(rewards))>1
        core.append_json(output/'rollouts.jsonl',{"group":progress['groups'],"id":row['id'],
            "gold":row['gold'],"answers":[{**a,**s} for a,s in zip(answers,scored)],
            "training_rewards":rewards,"eligible_anchor":eligible,"mixed":mixed})
        if eligible and mixed:
            advantages=core.group_advantages(rewards)
            for inner in range(params.updates_per_group):
                core.activate(model,training=True);optimizer.zero_grad(set_to_none=True)
                loss_sum=kl=clip_fraction=0.
                for i,answer in enumerate(answers):
                    with core.autocast(params):
                        logp=core.completion_logps(model,row['input_ids'],answer['tokens'],params.device)
                        loss,stats=core.grpo_loss(logp,old[i].to(params.device),ref[i].to(params.device),advantages[i].to(params.device),params.beta,params.clip)
                        (loss/params.group_size).backward()
                    loss_sum+=loss.item()/params.group_size;kl+=stats['kl']/params.group_size
                    clip_fraction+=stats['clip_fraction']/params.group_size
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True).item()
                optimizer.step();progress['global_step']+=1
                record={"step":progress['global_step'],"group":progress['groups'],"inner":inner,
                        "reward_mean":sum(rewards)/len(rewards),"strict_mean":sum(s['strict_reward'] for s in scored)/len(scored),
                        "severe_errors":sum(s['severe_error'] for s in scored),"loss":loss_sum,
                        "kl":kl,"clip_fraction":clip_fraction,"grad_norm":norm}
                core.append_json(output/'metrics.jsonl',record);print(record,flush=True)
            progress['mixed_groups']+=1
        else: progress['skipped_groups']+=1
        progress['groups']+=1;progress['cursor']+=1
        if progress['groups']%5==0:core.save_checkpoint(model,tokenizer,optimizer,output,signature,progress)
    core.save_checkpoint(model,tokenizer,optimizer,output,signature,progress)
    validate(model,tokenizer,val,params,output,'after_val')
    assert core.digest(Path(params.adapter)/'adapter_model.safetensors')==cfg['initial_adapter_sha256']
    assert {p.name:core.digest(p) for p in sorted(args.model.glob('*.safetensors'))}==signature['base_weights']
    core.write_json(output/'completed.json',{**progress,"finished":progress['global_step']==params.max_updates,
        "method":args.method,"seed":args.seed,"scope":"只训练75条训练侧业务提示；没有RAG RL或最终测试训练"})
    print('AB_TRIAL_COMPLETE',args.method,args.seed,progress,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','train'])
    p.add_argument('--model',type=Path);p.add_argument('--method',choices=['strict','graded'])
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--seed',type=int,default=42);p.add_argument('--output-dir',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    prepare() if a.mode=='prepare' else trial(a)
