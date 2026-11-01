"""Publish source-review tables and a complete cross-document entity concordance."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'research/benchmark_audit_20261030'
BANK=ROOT/'eval_bench/versions/20261030_v2'


def rows(path):
    return [json.loads(t) for t in path.read_text(encoding='utf-8').splitlines()]


def cell(text):
    return text.replace('|','\\|').replace('\n','<br>')


def main():
    ledger=rows(AUDIT/'audit_ledger.jsonl')
    changes=[r for r in ledger if r['question_changed'] or r['answer_changed']]
    sources={r['doc_id']:r for r in rows(AUDIT/'sources.jsonl')}
    table=['# 题目与答案修改说明表','',
      '全部1022题已进行本次来源核对；下表为273道涉及题目或答案文本修订的条目，其中100道改题干、225道改参考答案（两者有交集）。引文定位修复另在逐题JSONL中保存。独立AI复查尚未执行。','',
      '| 编号 | 原题与原答案 | 新题与新答案 | 为什么修改 | 来源位置 |',
      '| --- | --- | --- | --- | --- |']
    for row in changes:
        locations=[]
        for evidence in row['revised_evidence']:
            did=evidence['doc_id']
            locators=[s['locator'] for s in evidence['segments']] if 'segments' in evidence else evidence['locators']
            doc=sources[did]
            locations.append(f"{did} {', '.join(locators)}；生效{doc['metadata']['effective_date']}，更新{doc['metadata']['updated_at']}")
        columns=[row['qid'],'题：'+row['original_question']+'\n答：'+row['original_answer'],
                 '题：'+row['revised_question']+'\n答：'+row['revised_answer'],row['reason'],'\n'.join(locations)]
        table.append('| '+' | '.join(cell(x) for x in columns)+' |')
    (AUDIT/'修改说明表.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    labels={
      'cross_doc_composition':('不足','跨来源集合去重、三跳链、相对年份和实体归属'),
      'unit_conversion':('不足','亿元/万元、速度单位、周期倒数与二进制容量'),
      'rates_and_denominators':('不足','百分点/相对增长、利用率倒数、精确率/召回率与加权平均'),
      'temporal_version':('不足','发布/生效边界、工作日、先到质保及历史可知性'),
      'entity_deduplication':('不足','别名、跨仓SKU、同名人员、设备BOM包含'),
      'evidence_sufficiency':('不足','召回批次、申请/存量集合、部分年度和实际审批；含可确定与不可确定对照'),
      'conflict_resolution':('不足','显式新版替代、不同事件、不等价计时、核定来源和训练用途权限'),
      'structured_output':('原题未直接覆盖','仅JSON、字段/类型、排序、零值/false与未知值'),
      'dialogue_state':('原题未直接覆盖','最新用户修正、引用旧邮件、局部更新与撤销'),
      'instruction_isolation':('原题未直接覆盖','文档/检索/工具输出中的越权指令不改变任务'),
      'tool_planning':('原题未直接覆盖','本地六类工具的选用及参数、MCP规划、已知正常状态下最少调用'),
      'tool_result_reasoning':('原题未直接覆盖','截断、403、旧缓存、幂等重放、并发冲突、失败与取消'),
      'context_selection':('原题未直接覆盖','32段干扰、目标实体和有效版本，改变证据位置')}
    gold=rows(BANK/'answer_bank.jsonl')
    counts=Counter(r['capability'] for r in gold if r['evaluation_track']=='context_reasoning')
    coverage=['# 能力覆盖表','','原11种题型标签混合了题目形式和业务领域，不能当作能力覆盖已经充分。新增题与原题分轨，避免重复事实题数量主导结论。','',
      '| 能力 | 原覆盖结论 | 新增考点 | 新题编号 | 数量 | 判分 |','| --- | --- | --- | --- | --- | --- |']
    for i,(key,(state,desc)) in enumerate(labels.items(),1):
        coverage.append(f'| {key} | {state} | {desc} | L{i:02d}01–L{i:02d}08 | {counts[key]} | JSON对象字段/类型/值；推导保存在私有答案 |')
    coverage+=['','这104题提供必要材料，不给推导后的标准答案。对象、单位、事件与时点明确；既有证据不足题，也有输入充分应直接完成的正例。',
      '', '尚未通过这些文本题考察的系统能力：真实检索召回/重排、实时工具执行和权限、MCP握手与重连、会话外记忆、流式一致性、并发/取消落盘、生产缓存命中和真实成本计费、百万token上下文极限。这些仍须单独真实集成测试，不能计入本次模型JSON题通过率。']
    (AUDIT/'能力覆盖表.md').write_text('\n'.join(coverage)+'\n',encoding='utf-8')

    # Search all primary segments for concrete entities, not answer-keyword matches.
    org='\n'.join(s['text'] for s in sources['lantu_org_reporting_lines']['segments'])
    people={n for edge in re.findall(r'([\u4e00-\u9fff]{2,3})是([\u4e00-\u9fff]{2,3})的上级',org) for n in edge}
    customers={s['values'][0] for s in sources['lantu_sales_customer_list']['segments'][1:]}
    products=set()
    for source in sources.values():
        for segment in source['segments']:
            products.update(re.findall(r'LT-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*',segment['text']))
    concordance=[]
    for term in sorted(people|customers|products):
        matches=[]
        for source in sources.values():
            for segment in source['segments']:
                if term in segment['text']:
                    matches.append({'doc_id':source['doc_id'],'path':source['path'],
                        'locator':segment['locator'],'quote':segment['text'],
                        'effective_date':source['metadata']['effective_date'],'updated_at':source['metadata']['updated_at']})
        concordance.append({'entity':term,'matches':matches,'note':'完整字面索引，仅供寻找交叉证据，不据命中自动认定同事件/同配置/上下级。'})
    (AUDIT/'cross_document_entity_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in concordance),encoding='utf-8')
    print(json.dumps({'changes':len(changes),'original_reviewed':len(ledger),
                      'expanded':sum(counts.values()),'concordance_entities':len(concordance),'model_calls':0},ensure_ascii=False))


if __name__=='__main__':
    main()
