"""SFT 本地评测：实际问答、合成题、模拟工具、数值一致性与性能。

权重只读；中间记录在 runs，Markdown 与基础模型日志放在一起。
python test/test_code/evaluate_sft.py --phase generate
python test/test_code/evaluate_sft.py --phase numeric
python test/test_code/evaluate_sft.py --phase supplement
python test/test_code/evaluate_sft.py --phase units
python test/test_code/evaluate_sft.py --phase report
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import sft_model_test as entry  # 同时保证 Windows 的 pyarrow/torch 加载顺序。
import torch
from dataset.dataset import SFTDataset, post_processing_chat
from trainer.pretrain import next_token_loss

ROOT = entry.PROJECT_ROOT
OUT = ROOT / 'runs/sft-eval-20261015'
REPORT = ROOT / 'test/test_log/test_sft_model_2026-10-15.md'
CHECKPOINT = ROOT / 'runs/sfttrain1/checkpoint .pt'
BASE = ROOT / 'runs/pre_train1-20261012/checkpoint.pt'
DEVICE = torch.device('cuda')
MODELS = {'free': None, 'empty_think': False}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sync():
    torch.cuda.synchronize()


def prompt_ids(tokenizer, messages, tools=None, thinking=None):
    if thinking is None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        text += tokenizer.bos_token + 'assistant\n'
        return tokenizer(text, add_special_tokens=False).input_ids
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         open_thinking=thinking, tools=tools)


@torch.inference_mode()
def generate(model, tokenizer, messages, *, tools=None, thinking=None, temperature=0, seed=42, limit=512):
    """与交互入口相同的采样规则；额外接受多轮消息并记录同步计时。"""
    ids = prompt_ids(tokenizer, messages, tools, thinking)
    budget = min(limit, model.config.max_seq_len - len(ids))
    if budget <= 0:
        return dict(text='', input_tokens=len(ids), generated_tokens=0, stop_reason='input_too_long',
                    seconds=0, first_token_seconds=None, tokens_per_second=None)
    torch.manual_seed(seed)
    tokens = torch.tensor([ids], device=DEVICE)
    cache, emitted, first = None, [], None
    reason = 'context_limit' if budget == model.config.max_seq_len - len(ids) else 'max_new_tokens'
    sync()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(budget):
        hidden, cache = model.model(tokens, past_key_values=cache, use_cache=True)
        scores = model.lm_head(hidden[:, -1, :])[0].float().clone()
        assert torch.isfinite(scores).all().item()
        scores[[tokenizer.bos_token_id, tokenizer.pad_token_id]] = float('-inf')
        if temperature:
            scores /= temperature
            cutoff = torch.topk(scores, 50).values[-1]
            scores[scores < cutoff] = float('-inf')
            chosen = int(torch.multinomial(scores.softmax(-1), 1).item())
        else:
            chosen = int(scores.argmax().item())
        emitted.append(chosen)
        if first is None:
            sync()
            first = time.perf_counter() - start
        if chosen == tokenizer.eos_token_id:
            reason = 'eos'
            break
        tokens = torch.tensor([[chosen]], device=DEVICE)
    sync()
    seconds = time.perf_counter() - start
    grams = [tuple(emitted[i:i+4]) for i in range(max(0, len(emitted)-3))]
    return dict(text=tokenizer.decode(emitted, skip_special_tokens=False), input_tokens=len(ids),
                generated_tokens=len(emitted), stop_reason=reason, seconds=seconds,
                first_token_seconds=first, tokens_per_second=len(emitted)/seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                repeated_4gram_fraction=1-len(set(grams))/len(grams) if grams else 0,
                temperature=temperature, seed=seed, max_new_tokens=limit)


def final_text(text):
    text = text.replace('<|im_end|>', '').strip()
    # 未闭合 think 不视为已给出最终回答；不能从推理中随便捡一个正确数字。
    if '<think>' in text:
        if '</think>' not in text:
            return ''
        text = text.rsplit('</think>', 1)[1].strip()
    return text


def grade(text, expected):
    final = final_text(text)
    normalized = re.sub(r'[\s*`$]', '', final)
    claims = re.findall(r'答案[:：]([ABC](?![A-Za-z0-9])|[+-]?\d+(?=$|[。；，！？元个只件]|\.$))', normalized)
    answer = claims[-1] if claims else (normalized if normalized == str(expected) else None)
    return dict(expected=str(expected), extracted=answer, correct=answer == str(expected),
                exact_format=normalized == '答案：'+str(expected) or normalized == '答案:'+str(expected))


def synthetic_cases():
    rng = random.Random(20261015)
    cases = []
    for operation in ('+', '-', '*', '/'):
        for _ in range(4):
            a, b = rng.randint(2, 19), rng.randint(2, 9)
            if operation == '/':
                a *= b
            value = {'+': lambda: a+b, '-': lambda: a-b, '*': lambda: a*b, '/': lambda: a//b}[operation]()
            symbol = {'*': '×', '/': '÷'}.get(operation, operation)
            cases.append(('算术', f'{a}{symbol}{b}等于多少？', str(value)))
    for _ in range(4):
        a, b, c = rng.randint(3, 9), rng.randint(2, 6), rng.randint(1, 3)
        cases.append(('应用题', f'盒子里原有{a}个球，又放入{b}个，再拿走{c}个，剩下几个球？', str(a+b-c)))
    for price, discount, reduction in ((100,8,10),(200,5,20),(50,8,5),(80,5,10)):
        cases.append(('应用题', f'商品原价{price}元，先打{discount}折，再减{reduction}元，最终多少钱？', str(price*discount//10-reduction)))
    cases += [
        ('逻辑', '所有猫都是动物，小花是一只猫。小花是动物吗？A.是 B.不是 C.无法确定', 'A'),
        ('逻辑', '小明比小红高，小刚也比小红高。谁更高？A.小明 B.小刚 C.无法确定', 'C'),
        ('逻辑', '如果下雨，地面就湿。今天下雨了。地面湿吗？A.湿 B.不湿 C.无法确定', 'A'),
        ('逻辑', '只有学生可以进入。小林不是学生。小林可以进入吗？A.可以 B.不可以 C.无法确定', 'B'),
        ('信息提取', '记录：红球=3个，蓝球=7个。蓝球有几个？', '7'),
        ('信息提取', '名单依次是：小林、小王、小张。小王排第几？', '2'),
        ('信息提取', '密码是4826。请输出密码。', '4826'),
        ('信息提取', '甲盒有5个球，乙盒有9个球。只问甲盒，甲盒有几个？', '5'),
    ]
    return [dict(id=f'S{i:02}', category=cat, question=q+('。\n' if cat=='逻辑' else '')+'只输出“答案：'+('选项字母' if ans in ('A','B','C') else '数字')+'”，不要写过程。', expected=ans)
            for i,(cat,q,ans) in enumerate(cases,1)]


INTUITIVE = [
    ('D01','你好。请用一句中文回应。','简短、相关的中文问候'),
    ('D02','1加1等于多少？','2'),
    ('D03','小明有3个苹果，又买了2个，吃掉1个，还剩几个？','4个；3+2-1=4'),
    ('D04','一件商品原价80元，先打八折，再减10元，最终多少钱？','54元；80×0.8-10=54'),
    ('D05','所有猫都是动物，小花是一只猫。小花是动物吗？请说明理由。','是；根据给定集合关系推导'),
    ('D06','小明比小红高，小刚也比小红高。能确定小明和小刚谁更高吗？','无法确定，缺少两人之间的比较'),
    ('D07','把“猫、狗、鸟”按原顺序用英文逗号连接，只输出结果。','猫,狗,鸟'),
    ('D08','小林把红色笔记本放进蓝色背包。笔记本是什么颜色，放在哪里？','红色；蓝色背包里'),
]
TOOLS = [
    dict(id='T01',question='你好',required=False),
    dict(id='T02',question='1加1等于多少？',required=False),
    dict(id='T03',question='不要调用工具，请用一句话描述窗边的一只猫。',required=False),
    dict(id='T04',question='解释“帮我画一只猫”这句话的意思，不要实际画图。',required=False),
    dict(id='T05',question='帮我查询这台服务器当前的GPU利用率。',required=False),
    dict(id='T06',question='请生成一张图片：一只猫坐在窗边，窗外正在下雨。',required=True,
         concepts=[['猫','cat'],['窗','window'],['雨','rain']]),
    dict(id='T07',question='请生成一张图片：白色桌子上有两个红苹果。',required=True,
         concepts=[['苹果','apple'],['红','red'],['桌','table'],['白','white'],['两个','两只','two','2']]),
    dict(id='T08',question='请生成一张图片：夕阳下的蓝色海面上有一艘小船。',required=True,
         concepts=[['海','ocean','sea'],['蓝','blue'],['船','boat','ship'],['夕阳','日落','sunset']]),
]


def tool_grade(text, case):
    chunks = re.findall(r'<tool_call>(.*?)</tool_call>', text, flags=re.S)
    attempted = '<tool_call>' in text
    calls, errors = [], []
    for chunk in chunks:
        try:
            call = json.loads(chunk)
            if not (isinstance(call,dict) and call.get('name') == 'generate_image'
                    and isinstance(call.get('arguments'),dict)
                    and isinstance(call['arguments'].get('prompt'),str)
                    and call['arguments']['prompt'].strip()):
                errors.append('JSON 可解析，但工具名称或必需参数不符合 schema')
            else:
                calls.append(call)
        except json.JSONDecodeError as error:
            errors.append(str(error))
    if text.count('<tool_call>') != len(chunks) or text.count('</tool_call>') != len(chunks):
        errors.append('工具标签缺失或不配对')
    schema_ok = bool(calls) and not errors and len(calls) == len(chunks)
    coverage = bool(schema_ok and any(all(any(alias.lower() in call['arguments']['prompt'].lower() for alias in group)
                                        for group in case.get('concepts',[])) for call in calls))
    return dict(attempted=attempted, valid_calls=calls, errors=errors, schema_ok=schema_ok,
                keyword_coverage=coverage if case['required'] else None,
                selection_ok=attempted == case['required'],
                joint_ok=(schema_ok and coverage) if case['required'] else not attempted)


def run_generations():
    tokenizer = entry.AutoTokenizer.from_pretrained(ROOT/'dataset',local_files_only=True)
    model = entry.load_base_model(CHECKPOINT, DEVICE)
    before = digest(CHECKPOINT)
    result = dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), checkpoint=str(CHECKPOINT),
                  checkpoint_sha256=before, model_config=asdict(model.config), intuitive=[], synthetic=[],
                  sampling=[], tools=[], roundtrips=[])
    # 核对带计时的评测器与当前交互入口，避免两个测试端悄悄使用不同前缀。
    for mode, thinking in MODELS.items():
        actual = generate(model, tokenizer,[dict(role='user',content='你好')],thinking=thinking,limit=16)
        expected = entry.generate_reply(model,tokenizer,'你好',DEVICE,max_new_tokens=16,temperature=0,thinking=thinking)
        assert actual['text'] == expected
    result['interactive_entry_matches'] = True
    def run(question, **kwargs):
        return generate(model,tokenizer,[dict(role='user',content=question)],**kwargs)
    for case_id, question, expectation in INTUITIVE:
        for mode, thinking in MODELS.items():
            r = dict(id=case_id,mode=mode,question=question,expectation=expectation,**run(question,thinking=thinking))
            result['intuitive'].append(r)
            print('直观',case_id,mode,r['stop_reason'],flush=True)
    save('generation.json',result)
    cases = synthetic_cases()
    for case in cases:
        for mode, thinking in MODELS.items():
            r = dict(**case,mode=mode,**run(case['question'],thinking=thinking))
            r['score'] = grade(r['text'],case['expected'])
            result['synthetic'].append(r)
            print('合成',case['id'],mode,r['score']['correct'],r['stop_reason'],flush=True)
        save('generation.json',result)
    for case in [cases[i] for i in (0,4,16,24)]:
        for seed in (42,43,44):
            r = dict(**case,mode='sample_empty_think',**run(case['question'],thinking=False,temperature=0.8,seed=seed))
            r['score'] = grade(r['text'],case['expected'])
            result['sampling'].append(r)
            print('采样',case['id'],seed,r['score']['correct'],flush=True)
    for case in TOOLS:
        for mode,thinking in MODELS.items():
            r = dict(**case,mode=mode,**run(case['question'],tools=entry.IMAGE_TOOLS,thinking=thinking))
            r['score'] = tool_grade(r['text'],case)
            result['tools'].append(r)
            print('工具',case['id'],mode,r['score']['joint_ok'],flush=True)
    # 真实模型调用路径与注入正确历史的控制路径分开，不伪造模型已成功选工具。
    for i,case in enumerate([c for c in TOOLS if c['required']],1):
        for path in ('model_call','provided_call_control'):
            image_id = f'SIM-{i:03}'
            question = case['question']+'完成后只报告工具返回的 image_id。'
            messages = [dict(role='user',content=question)]
            if path == 'model_call':
                first = run(question,tools=entry.IMAGE_TOOLS)
                scored = tool_grade(first['text'],case)
                calls = scored['valid_calls']
                if not scored['schema_ok']:
                    result['roundtrips'].append(dict(id=case['id'],path=path,question=question,
                        initial=first,score=scored,skipped=True,reason='模型未产生可执行的合法调用；没有注入调用代替模型决策'))
                    continue
                content = re.sub(r'<tool_call>.*?</tool_call>','',first['text'],flags=re.S).replace('<|im_end|>','').strip()
                messages.append(dict(role='assistant',content=content,tool_calls=calls))
            else:
                calls = [dict(name='generate_image',arguments=dict(prompt=case['question']))]
                messages.append(dict(role='assistant',content='',tool_calls=calls))
                first,scored = None,None
            tool_results = [dict(status='simulated_success',image_id=image_id,real_service_executed=False,
                                 received_prompt=call['arguments']['prompt']) for call in calls]
            messages += [dict(role='tool',content=json.dumps(value,ensure_ascii=False)) for value in tool_results]
            reply = generate(model,tokenizer,messages,tools=entry.IMAGE_TOOLS,thinking=False)
            final = final_text(reply['text'])
            success = image_id in final and '<tool_call>' not in reply['text'] and reply['stop_reason']=='eos'
            result['roundtrips'].append(dict(id=case['id'],path=path,question=question,initial=first,
                 score=scored,skipped=False,simulated_results=tool_results,reply=reply,
                 reports_id_and_stops=success,end_to_end_ok=success and (path=='provided_call_control' or scored['joint_ok'])))
            print('工具回环',case['id'],path,success,flush=True)
    result['checkpoint_sha256_after'] = digest(CHECKPOINT)
    assert before == result['checkpoint_sha256_after']
    save('generation.json',result)


def make_encoder(tokenizer, length):
    # 仅复用生产数据层的模板/标签函数，不调用会生成 Arrow 缓存的构造函数。
    obj = SFTDataset.__new__(SFTDataset)
    obj.tokenizer, obj.max_length = tokenizer,length
    obj.bos_id = tokenizer(tokenizer.bos_token+'assistant\n',add_special_tokens=False).input_ids
    obj.eos_id = tokenizer(tokenizer.eos_token+'\n',add_special_tokens=False).input_ids
    return obj


def supplement():
    """纠正选择题末选项与输出要求之间的分隔；原始试运行另存，不混进正式分母。"""
    g=json.loads((OUT/'generation.json').read_text(encoding='utf-8'))
    tokenizer=entry.AutoTokenizer.from_pretrained(ROOT/'dataset',local_files_only=True)
    model=entry.load_base_model(CHECKPOINT,DEVICE)
    cases={c['id']:c for c in synthetic_cases()}
    replaced=[]
    for i,r in enumerate(g['synthetic']):
        c=cases[r['id']]
        if r['question']!=c['question']:
            replaced.append(r)
            new=dict(**c,mode=r['mode'],**generate(model,tokenizer,[dict(role='user',content=c['question'])],thinking=MODELS[r['mode']]))
            new['score']=grade(new['text'],c['expected'])
            g['synthetic'][i]=new
            print('修正题目分隔后复测',r['id'],r['mode'],new['score']['correct'],flush=True)
    # 采样逻辑题也使用同一修正后的措辞和原种子。
    for i,r in enumerate(g['sampling']):
        c=cases[r['id']]
        if r['question']!=c['question']:
            replaced.append(r)
            new=dict(**c,mode=r['mode'],**generate(model,tokenizer,[dict(role='user',content=c['question'])],thinking=False,temperature=.8,seed=r['seed']))
            new['score']=grade(new['text'],c['expected'])
            g['sampling'][i]=new
            print('修正题目分隔后采样',r['id'],r['seed'],new['score']['correct'],flush=True)
    g.setdefault('protocol_initial',[]).extend(replaced)
    for r in g['synthetic']+g['sampling']:
        r['score']=grade(r['text'],r['expected'])
    assert digest(CHECKPOINT)==g['checkpoint_sha256']
    save('generation.json',g)


def encode_record(encoder, record):
    text = post_processing_chat(encoder.create_chat_prompt(record['conversations']),remove_empty_think=True)
    ids = encoder.tokenizer(text,add_special_tokens=False).input_ids[:encoder.max_length]
    valid = len(ids)
    ids += [encoder.tokenizer.pad_token_id]*(encoder.max_length-valid)
    mask = [1]*valid+[0]*(encoder.max_length-valid)
    labels = encoder.generate_labels(ids,mask)
    return dict(input_ids=ids,attention_mask=mask,labels=labels)


def reconstruct_validation(tokenizer, manifest):
    """只重建验证分桶，不复制 1.7GB 训练集；输出与原 HF features 字段顺序相同。"""
    encoder = make_encoder(tokenizer,768)
    source = ROOT/'dataset/datasets/sft_t2t_mini.jsonl'
    fields = ('role','content','reasoning_content','tools','tool_calls')
    threshold = int(.01*(1<<64))
    raw_hash, val_hash = hashlib.sha256(),hashlib.sha256()
    records, sample = [],[]
    rng = random.Random(20261015)
    n = rejected = 0
    start = time.perf_counter()
    with source.open('rb') as f:
        for raw in f:
            raw_hash.update(raw)
            n += 1
            record = json.loads(raw)
            conversations = [{key:m.get(key) for key in fields} for m in record['conversations']]
            record = dict(conversations=conversations)
            if len(sample)<256:
                sample.append(record)
            else:
                slot = rng.randrange(n)
                if slot<256:
                    sample[slot]=record
            group = json.dumps(conversations,ensure_ascii=False,sort_keys=True,separators=(',',':'))
            bucket = hashlib.sha256(f'43\0{group}'.encode('utf-8')).digest()
            if int.from_bytes(bucket[:8],'big') >= threshold:
                continue
            content_json=json.dumps([{k:v for k,v in m.items() if k!='role'} for m in conversations],ensure_ascii=False)
            if any(marker in content_json for marker in (tokenizer.bos_token,tokenizer.eos_token,tokenizer.pad_token)):
                rejected+=1
                continue
            try:
                encoded = encode_record(encoder,record)
            except (ValueError,TypeError):
                rejected+=1
                continue
            if not any(label!=-100 for label in encoded['labels'][1:]):
                rejected+=1
                continue
            val_hash.update((json.dumps(record,ensure_ascii=False)+'\n').encode('utf-8'))
            records.append(record)
    source_meta=json.loads((ROOT/'dataset/SFT_SOURCE.json').read_text(encoding='utf-8'))
    assert raw_hash.hexdigest()==source_meta['sha256'],'SFT 原文件校验失败'
    assert val_hash.hexdigest()==manifest['data_sha256']['val_path'],'重建验证集与训练记录不一致'
    save('validation_provenance.json',dict(raw_sha256=raw_hash.hexdigest(),validation_sha256=val_hash.hexdigest(),
         source_records=n,validation_records=len(records),validation_rejected=rejected,scan_seconds=time.perf_counter()-start,
         split_seed=43,val_fraction=.01,audit_reservoir_seed=20261015,audit_reservoir_size=len(sample)))
    print('验证集哈希核对通过',len(records),flush=True)
    return records,sample


@torch.inference_mode()
def validation(model,tokenizer,records,length):
    encoder=make_encoder(tokenizer,length)
    total=tokens=zero=0
    correct=0
    sync()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for offset in range(0,len(records),4):
        rows=[encode_record(encoder,r) for r in records[offset:offset+4]]
        batch={k:torch.tensor([r[k] for r in rows],device=DEVICE) for k in rows[0]}
        valid=batch['labels'][:,1:]!=-100
        zero+=int((valid.sum(-1)==0).sum().item())
        if not valid.any().item():
            continue
        logits=model(batch['input_ids'],attention_mask=batch['attention_mask'])
        loss,count=next_token_loss(logits,batch['labels'])
        total+=loss.item()*count
        tokens+=count
        correct+=int(((logits[:,:-1].argmax(-1)==batch['labels'][:,1:])&valid).sum().item())
        if offset%1000==0:
            print('验证',length,offset,'/',len(records),flush=True)
    sync()
    loss=total/tokens
    return dict(records=len(records),max_length=length,valid_tokens=tokens,zero_target_records=zero,
                loss=loss,ppl=math.exp(loss),teacher_forced_token_accuracy=correct/tokens,
                seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                view='no_system_augmentation_empty_think_removed',batch_size=4)


@torch.inference_mode()
def numeric_parity(model,tokenizer):
    path=ROOT/'runs/minimind-reference-f659b55/model_minimind.py'
    assert digest(path)=='548f1203282dccb3be43227b11463f469b976afbab614df52828bdd246c3786f'
    spec=importlib.util.spec_from_file_location('reference_eval',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c=model.config
    rc=mod.MiniMindConfig(vocab_size=c.vocab_size,hidden_size=c.hidden_size,num_hidden_layers=c.num_layers,
        num_attention_heads=c.num_heads,num_key_value_heads=c.num_kv_heads,intermediate_size=c.intermediate_size,
        max_position_embeddings=c.max_position_embeddings,flash_attn=c.flash_attn,dropout=c.dropout,
        rms_norm_eps=c.rms_norm_eps,rope_theta=c.rope_theta,tie_word_embeddings=c.tie_word_embeddings)
    ref=mod.MiniMindForCausalLM(rc).float().to(DEVICE).eval()
    ref.load_state_dict(model.state_dict(),strict=True)
    results=[]
    for text in ('1加1等于多少？','小明有3个苹果，又买了2个，吃掉1个，还剩几个？','所有猫都是动物，小花是一只猫。小花是动物吗？'):
        for tools in (None,entry.IMAGE_TOOLS):
            ids=prompt_ids(tokenizer,[dict(role='user',content=text)],tools,False)
            x=torch.tensor([ids],device=DEVICE)
            own=model(x)
            reference=ref(x,use_cache=False).logits
            diff=(own-reference).abs()
            results.append(dict(question=text,tools=bool(tools),max_abs_diff=diff.max().item(),
                mean_abs_diff=diff.mean().item(),allclose=torch.allclose(own,reference,atol=1e-4,rtol=1e-4),
                all_position_top1_equal=bool((own.argmax(-1)==reference.argmax(-1)).all().item())))
    del ref
    cache_checks=[]
    for tools in (None,entry.IMAGE_TOOLS):
        ids=prompt_ids(tokenizer,[dict(role='user',content='你好')],tools,None)
        x=torch.tensor([ids],device=DEVICE)
        hidden,cache=model.model(x,use_cache=True)
        diffs=[]
        for next_id in tokenizer('你好，答案是2。',add_special_tokens=False).input_ids:
            step=torch.tensor([[next_id]],device=DEVICE)
            x=torch.cat((x,step),1)
            hidden,cache=model.model(step,past_key_values=cache,use_cache=True)
            cached=model.lm_head(hidden[:,-1,:])
            full=model(x)[:,-1,:]
            diffs.append(dict(max_abs_diff=(full-cached).abs().max().item(),
                              allclose=torch.allclose(full,cached,atol=1e-4,rtol=1e-4),
                              top1_equal=bool((full.argmax(-1)==cached.argmax(-1)).all().item())))
        cache_checks.append(dict(tools=bool(tools),steps=diffs))
    return dict(upstream_commit='f659b55761b754d306bd140573493a6543cafd7f',reference_sha256=digest(path),
                same_weight_forward=results,cache_vs_full=cache_checks)


@torch.inference_mode()
def performance(model):
    measurements=[]
    for length in (16,64,256,512,768):
        x=torch.randint(3,model.config.vocab_size,(1,length),device=DEVICE)
        for _ in range(3):
            model(x)
        times=[]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            sync();start=time.perf_counter();model(x);sync()
            times.append((time.perf_counter()-start)*1000)
        measurements.append(dict(length=length,median_ms=statistics.median(times),mean_ms=statistics.mean(times),
            min_ms=min(times),max_ms=max(times),peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    return measurements


def run_numeric():
    tokenizer=entry.AutoTokenizer.from_pretrained(ROOT/'dataset',local_files_only=True)
    manifest=json.loads((ROOT/'runs/sfttrain1/run_manifest.json').read_text(encoding='utf-8'))
    records,sample=reconstruct_validation(tokenizer,manifest)
    rng=random.Random(20261015)
    subset=rng.sample(records,256)
    subset_digest=hashlib.sha256(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset).encode()).hexdigest()
    audit=[]
    encoder=make_encoder(tokenizer,768)
    for record in sample:
        prompt=post_processing_chat(encoder.create_chat_prompt(record['conversations']),remove_empty_think=True)
        ids=tokenizer(prompt,add_special_tokens=False).input_ids
        labels=encoder.generate_labels(ids[:768],[1]*min(768,len(ids)))
        audit.append(dict(full_length=len(ids),truncated=len(ids)>768,valid_targets=sum(x!=-100 for x in labels[1:]),
            tools=any(m.get('tools') for m in record['conversations']),
            reasoning=any(m.get('reasoning_content') for m in record['conversations'])))
    model=entry.load_base_model(CHECKPOINT,DEVICE)
    before=digest(CHECKPOINT)
    result=dict(environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),gpu_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
        platform=platform.platform(),dtype='float32',threads=2),
        training_manifest=manifest,training_final_record=json.loads((ROOT/'runs/sfttrain1/metrics.jsonl').read_text().splitlines()[-1]),
        sft_checkpoint_sha256=before,base_checkpoint_sha256=digest(BASE),
        tokenizer_sha256={f:digest(ROOT/'dataset'/f) for f in ('tokenizer.json','tokenizer_config.json')},
        source_code_sha256={f:digest(ROOT/f) for f in ('model/model.py','model/layers.py','dataset/dataset.py',
             'trainer/pretrain.py','trainer/SFTtrain.py','trainer/trainer.py','test/test_code/sft_model_test.py','test/test_code/evaluate_sft.py')},
        data_audit=audit,subset_sha256=subset_digest,subset_seed=20261015)
    result['parity']=numeric_parity(model,tokenizer)
    print('数值对齐完成',flush=True)
    result['forward_performance']=performance(model)
    result['sft_full_validation']=validation(model,tokenizer,records,768)
    save('numeric.json',result)
    result['sft_common_subset_340']=validation(model,tokenizer,subset,340)
    del model
    torch.cuda.empty_cache()
    model=entry.load_base_model(BASE,DEVICE)
    result['base_common_subset_340']=validation(model,tokenizer,subset,340)
    assert digest(CHECKPOINT)==before
    result['sft_checkpoint_sha256_after']=digest(CHECKPOINT)
    result['base_checkpoint_sha256_after']=digest(BASE)
    result['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result['git_status']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip()
    save('numeric.json',result)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                     *['| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |' for row in rows]])+'\n'


def report():
    g=json.loads((OUT/'generation.json').read_text(encoding='utf-8'))
    n=json.loads((OUT/'numeric.json').read_text(encoding='utf-8'))
    p=json.loads((OUT/'validation_provenance.json').read_text(encoding='utf-8'))
    units=json.loads((OUT/'unit_tests.json').read_text(encoding='utf-8'))
    assert g['checkpoint_sha256']==n['sft_checkpoint_sha256']==n['sft_checkpoint_sha256_after']
    summary=[]
    for mode in MODELS:
        rec=[r for r in g['synthetic'] if r['mode']==mode]
        summary.append(f"{mode} {sum(r['score']['correct'] for r in rec)}/{len(rec)}")
    lines=['# TrainPilot SFT 模型完整测试日志 — 2026-10-15','',
      '**结论：当前权重在本次项目诊断中尚未达到可靠的基础推理与工具决策要求。** 详见直观原文、固定规则评分及工具参数检查；这不是标准榜单排名，也不能据此推断所有任务表现。','',
      '- 合成题按预设答案提取规则通过：'+ '；'.join(summary)+'。这也受输出格式影响；答案碰巧正确不代表推导正确。',
      f"- 完整验证固定视图：loss={n['sft_full_validation']['loss']:.6f}，PPL={n['sft_full_validation']['ppl']:.4f}；token 预测指标不能代替推理正确率。",
      f"- 代码测试：{units['tests_run']} 项，失败 {units['failures']}，错误 {units['errors']}；同权重 MiniMind 前向对照 {sum(r['allclose'] for r in n['parity']['same_weight_forward'])}/6 通过。",
      '- 未修改训练代码、数据增强策略、README 或权重；测试文件与日志是本次新增内容。','',
      '## 1. 范围与可复现信息','',
      '本报告测试本地第一轮 SFT 权重，不进行训练或更新权重。直观测试指实际问题的原始回答；仿真测试指按已知规则生成的合成题，以及仅返回模拟数据的工具交互。没有执行真实图片生成，没有使用外部模型代答或自动修复模型 JSON。',
      '',f"- 实验时间：{g['timestamp']}（北京时间）。",f'- 权重：`{CHECKPOINT}`。',
      f"- 报告生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。",
      f"- checkpoint SHA256：`{g['checkpoint_sha256']}`；实验前后相同。",
      f"- 独立参数：63,912,192；Git HEAD：`{n['git_head']}`。测试期间有新增评测文件，具体源码哈希见附录。",
      '- 原始输出、数值中间记录：本机 `runs/sft-eval-20261015`；正式 Markdown 与基础模型报告并列。',
      '- 全部题目在本次测试前确定；未检查其与大规模训练语料的近似重合。这是项目诊断集，不是标准榜单或严格隐藏泛化测试。','',
      '## 2. 训练与测试环境','',table(['字段','实际值'],[(k,str(v)) for k,v in n['environment'].items()]),
      '训练环境与本地测试环境不同。训练使用 BF16，当前测试使用 FP32、eval/inference_mode、CPU threads=2、单条生成、KV cache。当前 Windows 构建没有 FlashAttention 内核，SDPA 会使用可用实现；没有因此跳过测试。',
      '',table(['训练项','值'],[(k,str(v)) for k,v in n['training_manifest']['config']['training'].items()]),
      '模型配置：\n```json\n'+json.dumps(g['model_config'],ensure_ascii=False,indent=2)+'\n```','',
      '## 3. 测试方法与评分口径','',
      '- 主对照：free 仅补 assistant 角色头；empty_think 预填空 think 标签，等于交互脚本 `--no-thinking`。后者是前缀干预，不保证模型不再输出推理。',
      '- 主测试 temperature=0、最多新生成 512 token；总长度按当前交互脚本限制为 768。RoPE 上限 32768 不是本次已验证上下文长度。',
      '- 采样对照 temperature=0.8、top_k=50，逐题独立重设种子 42/43/44。',
      '- 原始输出完整保留，包括 think、tool_call、EOS；不补写截断结果。EOS 计入 generated_tokens。个别截断输出的 tokenizer 解码本身含替换字符“�”，也原样保留，不是 Markdown 文件读取失败。',
      '- 合成题固定答案格式。先排除 think 中的内容；未闭合 think 不算最终答案。优先提取最终正文最后一个明确的“答案：数字/选项”，否则仅接受整段为标准答案。不从正文随便搜索正确数字。正确率和精确格式遵循率分别报告；该提取规则可能漏判等价自然语言回答，因此附录保留原文。',
      '- 选择题最初试运行时，最后一个选项与输出要求之间缺少分隔。正式结果已使用句号和换行分开后重新生成，包括对应采样对照；旧输出保留在 protocol_initial 附录，不混入正式评分分母。评分器用受控反例检查，避免把4.2误判为整数4，并接受“A.湿”这类明确选项声明。',
      '- 工具测试分开统计：是否该调用、JSON/schema 合法性、参数关键词覆盖。关键词覆盖只是一项有限的语义代理，不能证明画面正确。负例不调用工具也不等于问答内容正确。',
      '- 模拟回环先尝试实际模型调用；若无合法调用，不注入假调用替模型通过。另设 provided_call_control，明确注入正确调用历史，只测能否读取模拟返回；不计入自主调用成功率。',
      '- 性能数据是当前本机观测，不是服务器训练吞吐；生成计时包含 GPU 同步、首轮前向、逐 token 采样和 Python 循环，不含加载、编码、解码及磁盘写入。首 token 延迟不是纯 prefill 内核时间。','',
      '## 4. 合成题仿真结果','']
    rows=[]
    for mode in MODELS:
        for category in ['全部',*sorted({r['category'] for r in g['synthetic']})]:
            records=[r for r in g['synthetic'] if r['mode']==mode and (category=='全部' or r['category']==category)]
            rows.append((mode,category,len(records),sum(r['score']['correct'] for r in records),
                         f"{sum(r['score']['correct'] for r in records)/len(records):.2%}",
                         sum(r['score']['correct'] and r['stop_reason']=='eos' for r in records),
                         sum(r['score']['exact_format'] for r in records),sum(r['stop_reason']!='eos' for r in records)))
    lines += [table(['模式','题类','题数','答案提取正确','规则通过率','提取正确且EOS','精确格式通过','非EOS结束'],rows),
       '逐条审阅也发现“最终答案正确、过程错误”的情况：例如 S06 的 free 输出虽然答对 14-8=6，但借位和数位解释混乱；S20 的 empty_think 输出答对7，却反复声称取走球后仍剩7个。因此上表不能叫“推理过程正确率”。','',
       '自动评分也有漏判边界：例如 S03/free 的最终正文出现正确的“2+2等于4”，S10/empty_think 输出“答案：3×6=18”，但没有按规定直接给“答案：4/18”，因此规则不通过。这些不应被表述为“数学值一定算错”。S14/empty_think 虽包含正确等式，仍重复到长度上限。最终结论结合这些定性审阅，不把格式错误全部归因于推理错误。','',
       '### 4.1 采样稳定性','',table(['题号','期望','种子','提取结果','正确','停止原因'],
        [(r['id'],r['expected'],r['seed'],r['score']['extracted'],r['score']['correct'],r['stop_reason']) for r in g['sampling']]),
       '这 12 次生成来自 4 道固定题的多种子重复，不能按 12 道独立题解释。','',
       '## 5. 工具选择与模拟交互','']
    rows=[]
    for mode in MODELS:
        rec=[r for r in g['tools'] if r['mode']==mode]
        pos=[r for r in rec if r['required']];neg=[r for r in rec if not r['required']]
        attempted=[r for r in rec if r['score']['attempted']]
        rows.append((mode,f"{sum(r['score']['schema_ok'] for r in pos)}/{len(pos)}",
                     f"{sum(r['score']['joint_ok'] for r in pos)}/{len(pos)}",
                     f"{sum(r['score']['attempted'] for r in neg)}/{len(neg)}",
                     f"{sum(r['score']['schema_ok'] for r in attempted)}/{len(attempted)}"))
    lines += [table(['模式','正例合法调用','正例参数覆盖通过','负例误调用','尝试调用中schema合法'],rows),
       table(['题号','模式','需要工具','尝试调用','schema合法','关键词覆盖','标签/JSON问题'],
        [(r['id'],r['mode'],r['required'],r['score']['attempted'],r['score']['schema_ok'],r['score']['keyword_coverage'],
          '; '.join(r['score']['errors'])) for r in g['tools']]),
       '逐条核对工具参数也支持上述失败判断：T06 正文复述“猫”，实际 prompt 却是 anime girl；T07 丢失白桌和两个的约束，重复 apple；T08 把船改成 flying cars。不能只看调用前的中文解释来判工具参数正确。','',
       '### 5.1 模拟工具返回回环','',table(['题号','路径','跳过后续','返回ID并正常结束','该路径通过'],
        [(r['id'],r['path'],r['skipped'],r.get('reports_id_and_stops',False),r.get('end_to_end_ok',False)) for r in g['roundtrips']]),
       '模拟工具只返回 `simulated_success`、SIM 编号和接收到的 prompt。没有生成图片。提供正确调用历史的控制组通过，也不能替代实际模型自主调用通过。若原始记录中出现 input_too_long，表示测试脚本的768长度限制阻止了后续前向，不能单独归因于模型读工具结果失败。','',
       '本次回环定性观察：已获得模拟返回的各条路径，模型仍只复述“已生成图片”及用户描述，没有按要求报告实际 SIM 编号；T07 的自主调用路径因调用标签不完整而未进入模拟执行。此项验证的是读取并遵循工具返回，不是实际图片生成质量。','',
       '## 6. 数值质量、数据与实现检查','',
       f"完整 SFT 原文件 SHA256：`{p['raw_sha256']}`；扫描 {p['source_records']} 条。按原 HF features 补齐字段、使用 seed=43 和 1% 哈希分桶，仅重建验证部分，得到 {p['validation_records']} 条，其 SHA256 `{p['validation_sha256']}` **与服务器训练 manifest 完全一致**。没有复制完整训练文件。",'',
       '复测使用固定视图：不随机添加 system、删除空 think，保留非空推理，右侧截断/padding，仅监督 assistant，损失处移位一次。没有修改生产数据增强或训练代码。服务器历史验证有随机增强，故历史 loss 与本次不同视图不能要求逐位一致。','',
       table(['来源/范围','样本','长度','有效目标token','loss','PPL','teacher-forced token准确率','耗时秒'],[
        ('服务器epoch2历史',8714,768,n['training_final_record']['validation']['valid_tokens'],
         f"{n['training_final_record']['validation']['loss']:.6f}",f"{math.exp(n['training_final_record']['validation']['loss']):.4f}",'未记录','历史指标'),
        *[(name,v['records'],v['max_length'],v['valid_tokens'],f"{v['loss']:.6f}",f"{v['ppl']:.4f}",
           f"{v['teacher_forced_token_accuracy']:.2%}",f"{v['seconds']:.3f}") for name,v in
          [('SFT完整验证固定视图',n['sft_full_validation']),('SFT共同子集',n['sft_common_subset_340']),('预训练共同子集',n['base_common_subset_340'])]]]),
       f"共同子集为同一验证分桶中固定随机抽样 256 条，种子 {n['subset_seed']}，序列化 SHA256 `{n['subset_sha256']}`；两份模型使用完全相同输入、标签和 340 截断，才可直接比较这里的 loss。teacher-forced 准确率给定真实历史 token，不能当作自主问答正确率。",'',
       f"共同子集在340截断后，两份模型均有 {n['sft_common_subset_340']['zero_target_records']} 条样本没有有效监督目标，不贡献 loss 或 token 准确率；完整768视图零目标样本为 {n['sft_full_validation']['zero_target_records']} 条。",'',
       '### 6.1 语料抽查','',
       table(['固定源语料 reservoir 样本','条数'],[
        ('抽查总数',len(n['data_audit'])),('完整编码超过768',sum(r['truncated'] for r in n['data_audit'])),
        ('截断后零监督目标',sum(r['valid_targets']==0 for r in n['data_audit'])),
        ('包含工具定义',sum(r['tools'] for r in n['data_audit'])),('包含非空reasoning',sum(r['reasoning'] for r in n['data_audit']))]),
       '抽样种子 20261015；抽查对象来自原始语料，不是独立验证质量评级。截断不等于最终答案一定丢失；本项不把长度统计直接解释成性能差的原因。','',
       '### 6.2 同权重原实现与 KV cache','',
       table(['输入','提供工具','logits最大绝对差','allclose','全部位置top1相同'],
        [(r['question'],r['tools'],f"{r['max_abs_diff']:.3e}",r['allclose'],r['all_position_top1_equal']) for r in n['parity']['same_weight_forward']]),
       table(['KV测试提供工具','比较步数','最大绝对差','每步allclose','每步top1相同'],
        [(r['tools'],len(r['steps']),f"{max(s['max_abs_diff'] for s in r['steps']):.3e}",
          all(s['allclose'] for s in r['steps']),all(s['top1_equal'] for s in r['steps'])) for r in n['parity']['cache_vs_full']]),
       '参考 MiniMind 提交 `f659b55761b754d306bd140573493a6543cafd7f`，先严格加载当前同一份 SFT 参数，再比较 FP32 输出；不是与另外训练出来的官方权重比质量。容差 atol=rtol=1e-4。有限输入通过不能证明整个训练流程或所有输入都无误。','',
       '### 6.3 现有代码测试','',
       f"命令：`{units['command']}`；退出码：{units['exit_code']}。unittest runner 输出：\n```text\n{units['output']}\n```",'',
       '## 7. 本地推理性能','',table(['输入token','前向中位ms','均值ms','最小ms','最大ms','峰值allocated MiB'],
        [(r['length'],f"{r['median_ms']:.3f}",f"{r['mean_ms']:.3f}",f"{r['min_ms']:.3f}",f"{r['max_ms']:.3f}",
          f"{r['peak_allocated_mib']:.2f}") for r in n['forward_performance']]),
       '前向性能：batch=1，固定随机合法 token，每长度预热3次、同步测量10次，输出完整 logits；全部 token 有效且不传 padding mask。不含采样，不是 decode 吞吐；计算路径也不同于带 padding 的验证批次。','']
    perf=[]
    for mode in MODELS:
        rec=[r for r in g['intuitive'] if r['mode']==mode]
        perf.append((mode,len(rec),sum(r['generated_tokens'] for r in rec),f"{sum(r['seconds'] for r in rec):.3f}",
          f"{sum(r['generated_tokens'] for r in rec)/sum(r['seconds'] for r in rec):.2f}",
          f"{statistics.median(r['first_token_seconds'] for r in rec)*1000:.3f}",
          f"{max(r['peak_allocated_mib'] for r in rec):.2f}"))
    review = [
        ('D01','未给最终回答，进入重复并截断','有相关问候，但违反一句话要求，并产生孤立的 `</think>`'),
        ('D02','明确回答3，错误且正常EOS结束','回答2，正确'),
        ('D03','最终回答1，买入被误作减法，错误','回答0，计算及叙述错误'),
        ('D04','最终回答20元，错误','错误公式并重复；达到生成上限'),
        ('D05','转为反复讨论猫和狗，没有完成所问推导','分类学跑题，并重复“属”到生成上限'),
        ('D06','擅自添加身高关系，给出错误确定结论','有不确定性表述，但又引入“小刚比小明高”的无依据前提'),
        ('D07','输出无关Python代码并重复','同样输出无关代码，未按要求连接字符串'),
        ('D08','最终复述包含正确颜色与位置，但推理混淆笔记本与背包颜色','开头包含正确信息，随后进入重复；不能算稳定完成'),
    ]
    lines += [table(['直观组','生成次数','输出token','总循环秒','总token/总秒','首token中位ms','峰值allocated MiB'],perf),
       '每题每配置一次测量。长输出、EOS时机、机器负载会影响结果；此处和基础模型报告中的无 cache 全前缀重算速度口径不同，不能直接归因于 SFT 让推理加速。','',
       '## 8. 直观测试完整输出','',
       '以下内容审阅由本次助手根据原文完成，属于定性判断，未组织独立人工标注，未调用额外评测模型。区分“内容局部正确”和“完整、稳定地遵循任务”，不与合成题的自动规则通过率混算。','',
       table(['题号','free 内容审阅','empty_think 内容审阅'],review),'']
    for r in g['intuitive']:
        lines += [f"### {r['id']} / {r['mode']}",'',f"输入：{r['question']}",'',f"期望：{r['expectation']}",'',
          f"停止原因 `{r['stop_reason']}`；输入 {r['input_tokens']} token；生成 {r['generated_tokens']} token。",'',
          '````text\n'+r['text']+'\n````','']
    lines += ['## 9. 自动题目与工具测试原始输出','',
              '以下折叠块包含每次生成的完整原文；没有只保留成功案例。','']
    for group in ('synthetic','sampling','tools'):
        lines += [f'### {group}','']
        for r in g[group]:
            lines += [f"<details><summary>{r['id']} / {r['mode']} / seed={r['seed']} / {r['stop_reason']}</summary>",'',
             f"输入：{r['question']}",'','评分：`'+json.dumps(r['score'],ensure_ascii=False)+'`','',
             '````text\n'+r['text']+'\n````','','</details>','']
    lines += ['### 模拟回环原始记录','','````json\n'+json.dumps(g['roundtrips'],ensure_ascii=False,indent=2)+'\n````','',
       '### 选择题分隔修正前的试运行（不计入正式分母）','',
       '<details><summary>展开初始协议的完整输出</summary>\n\n````json\n'+json.dumps(g.get('protocol_initial',[]),ensure_ascii=False,indent=2)+'\n````\n\n</details>','',
       '## 10. 结论与下一步','',
       '本节结论应结合上面的直观原文与固定规则评分阅读。代码一致性、token loss、回答正确率和工具成功率是不同指标，不能互相代替。',
       '', '1. 优先修复基础问答与短推理：检查实际监督答案与推理内容，再设计少量经过核对、推导简短的数据进行对照实验，保留不同数字/条件的独立评测。',
       '2. 当前测试集作为冻结回归集，不直接加入训练；小样本训练集记住答案只能验证训练链路，不证明推理泛化。',
       '3. 工具选择、JSON、参数语义和读回工具结果分别验收；在基础推理和工具误调用未改善前，不把当前小模型直接作为 Agent 的可靠决策器。',
       '4. 是否增加训练轮数、修改学习率或换基础模型，需后续受控实验。当前结果本身不能唯一归因于模型参数量、某一训练超参或语料质量。','',
       '### 复现命令','',
       '在项目根目录、同一环境与权重下依次执行。runs 中的 JSON 是复现中间记录，正式交付为本 Markdown。若首次运行时题目已含修正分隔，supplement 不会重复生成。','',
       '```powershell\n'+ '\n'.join('.\\.venv\\Scripts\\python.exe test/test_code/evaluate_sft.py --phase '+phase for phase in ('generate','numeric','supplement','units','report'))+'\n```','',
       '## 附录：源码与词表哈希','',
       '下表是数值阶段记录时的版本。评测过程中修正了选择题分隔、评分边界并补充报告，模型/数据/训练实现没有更改；受影响的生成已重跑，所有自动答案均用最终评分器重新判分。',
       table(['文件','SHA256'],list(n['source_code_sha256'].items())+list(n['tokenizer_sha256'].items())),
       f"最终评测脚本 SHA256：`{digest(Path(__file__))}`。",'',
       '权重前后哈希：','',table(['对象','实验前','实验后'],[
          ('SFT',n['sft_checkpoint_sha256'],n['sft_checkpoint_sha256_after']),
          ('预训练',n['base_checkpoint_sha256'],n['base_checkpoint_sha256_after'])])]
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Markdown:',REPORT,flush=True)


def run_units():
    from contextlib import redirect_stdout,redirect_stderr
    import io
    import unittest
    stream=io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):
        suite=unittest.defaultTestLoader.discover(str(ROOT/'test/test_code'),pattern='test_*.py')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record=dict(command='python test/test_code/evaluate_sft.py --phase units',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        exit_code=0 if result.wasSuccessful() else 1,output=stream.getvalue())
    save('unit_tests.json',record)
    print({k:v for k,v in record.items() if k!='output'},flush=True)
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('generate','numeric','supplement','units','report'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    {'generate':run_generations,'numeric':run_numeric,'supplement':supplement,'units':run_units,'report':report}[args.phase]()


if __name__=='__main__':
    main()
