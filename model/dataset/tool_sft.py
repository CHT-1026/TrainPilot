"""V7工具SFT：可执行的合成资料环境与分组轨迹生成；不读取真实评测答案。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'model/dataset/datasets/trainpilot_agent/tool_v7'
FAMILIES = ['lookup', 'two_hop', 'empty_retry', 'pagination', 'missing', 'conflict', 'time', 'clarify']


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class SandboxTools:
    """仅操作样本自己的虚构文档；固定参数规格，返回真实计算结果。"""
    def __init__(self, documents, schemas):
        self.documents, self.schemas = documents, schemas

    def run(self, name, args):
        import jsonschema
        if name not in self.schemas:
            return {'error': 'unknown_tool'}
        try:
            jsonschema.validate(args, self.schemas[name])
        except jsonschema.ValidationError as error:
            return {'error': 'invalid_arguments', 'detail': error.message}
        if name == 'list':
            key, start = args.get('keyword', ''), args.get('offset', 0)
            docs = [{k: d[k] for k in ['doc_id', 'title']} for d in self.documents
                    if key in d['doc_id'] + d['title']]
            return {'documents': docs[start:start+20], 'next_offset': start+20 if start+20 < len(docs) else None}
        if name == 'read':
            doc = next((d for d in self.documents if d['doc_id'] == args['doc_id']), None)
            if not doc:
                return {'error': 'unknown_document'}
            start, count = args.get('offset', 0), args.get('limit', 40)
            rows = [{**s, 'doc_id': doc['doc_id']} for s in doc['segments']]
            return {'segments': rows[start:start+count], 'next_offset': start+count if start+count < len(rows) else None,
                    'total_segments': len(rows)}
        query = args['query']
        hits = []
        for doc in self.documents:
            # Exact named keys model a small searchable index; no gold answers influence retrieval.
            if any(re.search(re.escape(key) + r'(?![-A-Za-z0-9])', query) for key in doc['keys']):
                hits.append({**doc['segments'][0], 'doc_id': doc['doc_id'], 'offset': 0})
        return {'hits': hits[:args.get('top_k', 5)]}


def episode(split, index, family, contract):
    rng = random.Random(f'{split}:{index}:{family}:7')
    prefix = {'train':'TR', 'val':'VA', 'test':'TE'}[split]
    code = f'TP-{prefix}{index:04d}'
    surname = {'train':'陈林赵许周吴郑何', 'val':'欧董梁', 'test':'邱章蒋'}[split][index % len({'train':'陈林赵许周吴郑何', 'val':'欧董梁', 'test':'邱章蒋'}[split])]
    person = surname + '文思明安'[index % 4] + '远宁辰悦'[(index//4) % 4]
    boss = surname + '景星向'[index % 3] + '川航阳'[(index//3) % 3]
    metric, older = rng.randint(12, 90), rng.randint(2, 11)
    docs, actions, citations, facts = [], [], [], []
    def doc(name, title, keys, texts):
        value = {'doc_id': f'{prefix.lower()}_{index}_{name}', 'title': title, 'keys': keys,
                 'segments': [{'locator': f'line:{j+1}@0', 'text': text} for j,text in enumerate(texts)]}
        docs.append(value)
        return value
    def cite(d, n=0):
        value = {'doc_id':d['doc_id'], **d['segments'][n]}
        citations.append({'doc_id':value['doc_id'], 'locator':value['locator'], 'quote':value['text']})
        return value['text']
    questions = {
        'train': {
            'lookup':f'{code}的额定负载是多少？', 'two_hop':f'{person}的直接上级是谁，该上级负责什么岗位？',
            'empty_retry':f'我记得旧代号是{code}-OLD，请核实这台设备的额定负载。',
            'pagination':f'请从目录找到{code}操作规范，分别说明启动条件和停机条件。',
            'missing':f'资料有没有{code}未来十年的实际维修总支出？',
            'conflict':f'{code}同一生效日的两份记录给了不同负载，该采信哪一个？',
            'time':f'我只关心2024年，{code}当年的标准值是多少？', 'clarify':'帮我查一下那个标准。'},
        'val': {
            'lookup':f'查阅资料后告诉我，型号{code}规定可承载多少kg？',
            'two_hop':f'请核实{person}向谁直接汇报，并查明此人的任职。',
            'empty_retry':f'只记得{code}-OLD这个旧称，能否从资料核实它的负载？',
            'pagination':f'请查找题为{code}操作规范的资料，我需要开机与关机两项要求。',
            'missing':f'能查到{code}未来十年已经发生的维修费用合计吗？',
            'conflict':f'同一天生效的两条{code}负载规定相互矛盾，请判断是否能选定一个数值。',
            'time':f'不要用新版替代：请找{code}在2024年的标准值。', 'clarify':'想找规定，但具体哪一份我还没想好。'},
        'test': {
            'lookup':f'{code}的载重上限按参数表应填什么数值和单位？',
            'two_hop':f'我需要{person}直属主管的姓名与职务，两项都请提供出处。',
            'empty_retry':f'资料中旧标识{code}-OLD对应哪款设备，额定承重是多少？',
            'pagination':f'定位{code}操作规范，再把开始运行和结束运行的条件找齐。',
            'missing':f'请给出{code}今后十年维修实际支出的精确总数。',
            'conflict':f'对{code}，两份同日有效资料数值打架且无优先级，能得出确定负载吗？',
            'time':f'我要回填2024年的{code}标准，请排除2025年数据。', 'clarify':'那个要求到底是什么，你帮我找找。'},
    }
    question, status = questions[split][family], 'answerable'
    if family in {'lookup','empty_retry'}:
        d = doc('spec', code+'参数表', [code], [f'{code}额定负载为{metric}kg。'])
        if family == 'empty_retry':
            d['segments'][0]['text']=f'{code}（旧代号{code}-OLD）的额定负载为{metric}kg。'
            actions.append(('search', {'query':code+'-OLD'}))
        actions.append(('search', {'query':code+' 额定负载', 'top_k':5}))
        answer = cite(d); facts = [code,str(metric)+'kg']
    elif family == 'two_hop':
        d = doc('report', person+'汇报关系', [person], [f'{person}的直接上级是{boss}。'])
        e = doc('position', boss+'任职资料', [boss], [f'{boss}担任研发主管。'])
        actions = [('search', {'query':person}), ('search', {'query':boss}), ('read', {'doc_id':e['doc_id'],'offset':0,'limit':2})]
        answer = cite(d) + cite(e); facts = [boss,'研发主管']
    elif family == 'pagination':
        d = doc('guide',code+'操作规范',[code],[f'{code}启动前必须完成传感器自检。',f'{code}停机前必须保存作业状态。'])
        actions=[('list',{'keyword':code}),('read',{'doc_id':d['doc_id'],'offset':0,'limit':1}),('read',{'doc_id':d['doc_id'],'offset':1,'limit':1})]
        answer=cite(d)+cite(d,1); facts=['传感器自检','保存作业状态']
    elif family == 'missing':
        doc('spec', code+'参数表',[code],[f'{code}额定负载为{metric}kg。'])
        actions=[('search',{'query':code+' 未来十年 实际维修支出'})]
        status='not_in_corpus'; answer='当前返回的资料只包含设备参数，没有未来十年的实际维修支出，无法给出总数。'
    elif family == 'conflict':
        d=doc('a',code+'负载记录甲',[code],[f'{code}自2025-01-01生效的记录甲规定额定负载{metric}kg。'])
        e=doc('b',code+'负载记录乙',[code],[f'{code}自2025-01-01生效的记录乙规定额定负载{metric+5}kg；未注明与记录甲的优先级。'])
        actions=[('search',{'query':code})]; cite(d); cite(e)
        status='needs_clarification'; answer=f'同日生效的记录甲为{metric}kg，记录乙为{metric+5}kg，数值冲突且无优先级，需确认适用版本，不能擅自选一个。'
        facts=[str(metric)+'kg',str(metric+5)+'kg']
    elif family == 'time':
        d=doc('history',code+'历年标准',[code],[f'{code}在2025年的标准值是{metric}元。',f'{code}在2024年的标准值是{older}元。'])
        actions=[('search',{'query':code}),('read',{'doc_id':d['doc_id'],'offset':1,'limit':1})]
        answer=cite(d,1); facts=['2024',str(older)+'元']
    else:
        status='needs_clarification'; answer='请说明要查询的制度主题、设备型号或资料名称，我再检索对应内容。'
    # Short irrelevant documents are included to prevent the returned context always being the answer.
    doc('other','日常办公资料',['办公用品'],['办公用品每月集中申领一次。'])
    schemas={t['function']['name']:t['function']['parameters'] for t in contract['tools']}
    backend=SandboxTools(docs,schemas)
    messages=[{'role':'system','content':contract['system']},{'role':'user','content':question}]
    for n,(name,args) in enumerate(actions):
        call={'id':f'call_{n}','type':'function','function':{'name':name,'arguments':args}}
        messages.append({'role':'assistant','content':'','tool_calls':dumps([call])})
        messages.append({'role':'tool','tool_call_id':call['id'],'content':dumps(backend.run(name,args))})
    final={'status':status,'answer':answer,'citations':citations}
    messages.append({'role':'assistant','content':dumps(final)})
    return {'id':f'v7-{split}-{index:04d}','group_id':f'v7-{split}-{index:04d}', 'category':family,
            'mode':'tool_trajectory','messages':messages,'tools':dumps(contract['tools']),
            'sandbox':dumps(docs),'expected':dumps({'status':status,'facts':facts}), 'origin':'synthetic_executable_sandbox_v1'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    contract=json.loads((args.output/'contract.json').read_text(encoding='utf-8'))
    info={'version':'tool_v7','initial_adapter_sha256':'4d89275137bee224b724f06b2ff9990ab784d19e3ded9e570499b9abfd6407b8',
          'split':'separate synthetic entities/documents/wordings; existing V6 split retained for replay',
          'limitations':'templated synthetic sandbox, not proof of real-corpus generalization; peer benchmark excluded', 'outputs':{}}
    for split,tool_count,replay_count in [('train',960,320),('val',120,40),('test',120,40)]:
        rows=[episode(split,i,FAMILIES[i%len(FAMILIES)],contract) for i in range(tool_count)]
        old=[json.loads(s) for s in (ROOT/f'model/dataset/datasets/trainpilot_agent/v6/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        rng=random.Random('v7-replay-'+split); rng.shuffle(old)
        for r in old[:replay_count]:
            rows.append({'id':'replay-'+r['id'],'group_id':'replay-'+r['group_id'],'category':r['category'],
                         'mode':'v6_replay','messages':r['messages'],'tools':None,'sandbox':None,
                         'expected':None,'origin':'existing_v6_same_split_replay'})
        rng.shuffle(rows)
        data=''.join(dumps(row)+'\n' for row in rows).encode('utf-8')
        (args.output/f'{split}.jsonl').write_bytes(data)
        info['outputs'][split]={'records':len(rows),'tool_trajectories':tool_count,'v6_replay':replay_count,
                                'sha256':hashlib.sha256(data).hexdigest(),'categories':dict(Counter(r['category'] for r in rows if r['mode']=='tool_trajectory'))}
    (args.output/'manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
