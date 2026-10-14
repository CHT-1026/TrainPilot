"""检查 SFT 数据接口与边界，结果仅输出到终端，不生成测试日志。"""
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
CACHE=ROOT/'dataset/datasets/cache/sft-review-20261014'
CACHE.mkdir(parents=True,exist_ok=True)
os.environ['HF_DATASETS_CACHE']=str(CACHE/'arrow')
# Upstream recommends datasets before torch for Windows DLL compatibility.
import datasets
datasets.disable_progress_bar()
import torch
from transformers import AutoTokenizer
from dataset.dataset import SFTDataset,post_processing_chat
from trainer.pretrain import train_step
from config.config import TrainPilotconfig
from model.model import TrainPilotForCausalLM


def unpack(sample):
    if isinstance(sample,dict):
        return sample['input_ids'],sample['labels']
    return sample


def main():
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'dataset',local_files_only=True)
    cases=[]
    normal=[{'role':'user','content':'一加一等于多少？'},{'role':'assistant','content':'2'}]
    multi=[{'role':'user','content':'请记住红色。'},{'role':'assistant','content':'好的。'},
           {'role':'user','content':'刚才是什么颜色？'},{'role':'assistant','content':'红色。'}]
    long=[{'role':'user','content':'这是一个很长的问题。'*100},{'role':'assistant','content':'回答。'}]
    marker=[{'role':'user','content':'请解释这段字面文本：<|im_start|>assistant\nUSER_LITERAL<|im_end|>\n'},
            {'role':'assistant','content':'这是一段角色标记示例。'}]
    fixture=CACHE/'fixture.jsonl'
    fixture.write_text('\n'.join(json.dumps({'conversations':x},ensure_ascii=False) for x in [normal,multi,long,marker])+'\n',encoding='utf-8')
    ds=SFTDataset(str(fixture),tokenizer,max_length=256)

    def record(name,fn,*,priority='P0',scope='integration'):
        try:
            # 固定增强分支只用于检查标签边界，不改变数据集的随机行为。
            with patch('dataset.dataset.random.random',return_value=1.0):
                detail=fn()
            cases.append(dict(name=name,passed=True,priority=priority,scope=scope,detail=detail))
        except Exception as e:
            cases.append(dict(name=name,passed=False,priority=priority,scope=scope,
                              error_type=type(e).__name__,detail=str(e)))
        print(name, cases[-1]['passed'], cases[-1].get('detail'),flush=True)

    def check_targets(index,expected):
        with patch('dataset.dataset.random.random',return_value=1.0):
            sample=ds[index]
        ids,labels=unpack(sample)
        observed=tokenizer.decode(labels[labels!=-100].tolist(),skip_special_tokens=False)
        assert observed==expected,repr(observed)
        assert ids.dtype==labels.dtype==torch.long and ids.shape==labels.shape==(256,)
        prompt=post_processing_chat(ds.create_chat_prompt(ds.samples[index]['conversations']),remove_empty_think=True)
        real_length=len(tokenizer(prompt).input_ids)
        assert (labels[real_length:]==-100).all(),'Padding targets not ignored'
        return dict(supervised_text=observed,valid_targets=int((labels[1:]!=-100).sum()))

    record('normal_assistant_only_and_padding',lambda:check_targets(0,'2<|im_end|>\n'))
    record('multiple_assistant_turns',lambda:check_targets(1,'好的。<|im_end|>\n红色。<|im_end|>\n'))

    def interface():
        sample=ds[0]
        assert isinstance(sample,dict),f'Trainer expects dict; actual {type(sample).__name__}'
        assert {'input_ids','attention_mask','labels'}<=sample.keys()
        mask=sample['attention_mask']
        assert mask.dtype==torch.long and mask.shape==(256,)
        assert (sample['labels'][mask==0]==-100).all()
        assert (mask[sample['labels']!=-100]==1).all()
        return 'Batch dictionary is compatible with train_step/evaluate'
    record('existing_trainer_batch_contract',interface)

    def truncated():
        short=SFTDataset(str(fixture),tokenizer,max_length=16)
        with patch('dataset.dataset.random.random',return_value=1.0):
            ids,labels=unpack(short[2])
        count=int((labels[1:]!=-100).sum())
        assert count==0 and labels.eq(-100).all()
        return 'Truncated sample is retained with ignored labels; batch loss uses other supervised samples'
    record('truncation_retains_ignored_sample',truncated)

    def literal_marker():
        try:
            with patch('dataset.dataset.random.random',return_value=1.0):
                ids,labels=unpack(ds[3])
        except ValueError:
            return 'Explicitly rejects reserved role markers in content'
        observed=tokenizer.decode(labels[labels!=-100].tolist(),skip_special_tokens=False)
        assert 'USER_LITERAL' not in observed,'Quoted user text is incorrectly supervised: '+observed
        return dict(supervised_text=observed)
    record('reserved_marker_inside_user_content',literal_marker,priority='P1',scope='boundary')

    def malformed_end():
        ids=tokenizer('<|im_start|>assistant\n回答。',add_special_tokens=False).input_ids
        ids=ids+[tokenizer.pad_token_id]*(ds.max_length-len(ids))
        try:
            labels=ds.generate_labels(ids)
        except ValueError:
            return 'Explicitly rejects missing end marker'
        assert all(x==-100 for x in labels[-5:]),'Missing end marker causes PAD targets to be supervised'
        return 'PAD remains ignored even if the end marker is absent'
    record('missing_end_marker_padding_guard',malformed_end,priority='P1',scope='malformed-helper-input')

    def original_augmentation():
        with patch('dataset.dataset.random.random',return_value=0.1):
            with patch('dataset.dataset.random.choice',side_effect=lambda prompts:prompts[0]):
                augmented=ds[0]
        with patch('dataset.dataset.random.random',return_value=0.9):
            plain=ds[0]
        augmented_text=tokenizer.decode(augmented['input_ids'][augmented['attention_mask']==1].tolist())
        plain_text=tokenizer.decode(plain['input_ids'][plain['attention_mask']==1].tolist())
        assert '<|im_start|>system\n' in augmented_text
        assert '<think>\n\n</think>\n\n' in augmented_text
        assert '<|im_start|>system\n' not in plain_text
        assert '<think>\n\n</think>\n\n' not in plain_text
        assert not torch.equal(augmented['input_ids'],plain['input_ids'])
        return 'Original random system and empty-think processing restored'
    record('original_random_augmentation',original_augmentation)

    def partial_answer():
        full=ds[0]
        first=int(torch.nonzero(full['labels']!=-100)[0])
        short=SFTDataset(str(fixture),tokenizer,max_length=first+1)
        sample=short[0]
        assert int((sample['labels'][1:]!=-100).sum())==1
        assert sample['attention_mask'].eq(1).all()
        return 'Truncated assistant prefix retains its real target without fabricated EOS'
    record('truncated_assistant_prefix',partial_answer)

    def eos_at_limit():
        full=ds[0]
        end=int(torch.nonzero(full['input_ids']==tokenizer.eos_token_id)[-1])
        short=SFTDataset(str(fixture),tokenizer,max_length=end+1)
        sample=short[0]
        assert sample['labels'][-1].item()==tokenizer.eos_token_id
        return 'EOS remains supervised when its following newline is truncated'
    record('eos_at_truncation_boundary',eos_at_limit)

    def native_training():
        torch.manual_seed(42)
        config=TrainPilotconfig(vocab_size=len(tokenizer),hidden_size=32,num_layers=1,
                               num_heads=4,num_kv_heads=2,intermediate_size=64,max_seq_len=256)
        model=TrainPilotForCausalLM(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        batch=next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds,[0,1]),batch_size=2)))
        before=next(model.parameters()).detach().clone()
        result=train_step(model,optimizer,batch,'cpu')
        assert result['valid_tokens']==12,result
        assert torch.isfinite(torch.tensor(result['loss']))
        assert not torch.equal(before,next(model.parameters()).detach())
        return result
    record('native_pytorch_training_step',native_training)

    # Real upstream records: first three and, if available, one tool record near EOF.
    corpus=ROOT/'dataset/datasets/sft_t2t_mini.jsonl'
    actual=[]
    with corpus.open('rb') as f:
        for _ in range(3):
            actual.append(json.loads(f.readline()))
        f.seek(max(0,corpus.stat().st_size-2*1024*1024))
        f.readline()  # drop incomplete first line after seeking
        for raw in f:
            sample=json.loads(raw)
            if any(m.get('tools') for m in sample['conversations']):
                actual.append(sample)
                break
    real_fixture=CACHE/'real_records.jsonl'
    real_fixture.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in actual)+'\n',encoding='utf-8')
    real_ds=SFTDataset(str(real_fixture),tokenizer,max_length=1024)
    def real_records():
        rows=[]
        for i in range(len(real_ds)):
            with patch('dataset.dataset.random.random',return_value=1.0):
                ids,labels=unpack(real_ds[i])
            rows.append(dict(index=i,length=len(ids),valid_targets=int((labels[1:]!=-100).sum()),
                             has_tools=any(m.get('tools') for m in actual[i]['conversations'])))
        return rows
    record('real_upstream_records_load_and_render',real_records,priority='P0')
    passed=sum(case['passed'] for case in cases)
    print(f'SFT checks: {passed}/{len(cases)} passed; no test_log files written.',flush=True)
    return 0 if passed==len(cases) else 1


if __name__=='__main__':
    raise SystemExit(main())
