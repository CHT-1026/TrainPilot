"""Independent source arithmetic, bank integrity and coverage checks. Zero provider calls."""
from collections import Counter,defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.grading import grade_json

AUDIT=ROOT/'research/benchmark_audit_20261030'
BANK=ROOT/'eval_bench/versions/20261030_v2'


def rows(path):
    return [json.loads(t) for t in path.read_text(encoding='utf-8').splitlines()]


def number(value):
    return Decimal(str(value).replace(',','').split()[0])


def table(source,sheet):
    return [s['values'] for s in source['segments'] if s['section']==sheet][1:]


def main():
    sources={r['doc_id']:r for r in rows(AUDIT/'sources.jsonl')}
    questions=rows(BANK/'question_bank.jsonl')
    gold=rows(BANK/'answer_bank.jsonl')
    by_id={r['qid']:r for r in gold}
    checks={}
    def check(name,value):
        checks[name]=bool(value)

    check('1126_unique_ids',len(by_id)==len(questions)==len(gold)==1126)
    check('question_gold_exactly_aligned',all(q['qid']==a['qid'] and q['question']==a['question'] for q,a in zip(questions,gold)))
    original_ids={r['qid'] for r in rows(ROOT/'eval_bench/answer_bank.jsonl')}
    check('all_1022_original_ids_retained',original_ids<=by_id.keys() and len(original_ids)==1022)
    check('all_136_sources_unchanged',len(sources)==136 and all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in sources.values()))
    check('all_original_questions_have_review_ledger',len(rows(AUDIT/'audit_ledger.jsonl'))==1022)
    check('no_unresolved_original_quote_locations',not (AUDIT/'unresolved_original_quotes.jsonl').read_text(encoding='utf-8'))
    new=[r for r in gold if r['evaluation_track']=='context_reasoning']
    counts=Counter(r['capability'] for r in new)
    check('104_cases_13_capabilities_8_each',len(new)==104 and len(counts)==13 and set(counts.values())=={8})
    check('fixture_gold_is_valid_exact_json',all(grade_json(r['answer'],r['rubric']['expected'])[0] for r in new))
    check('question_texts_not_duplicated',len({r['question'] for r in questions})==len(questions))

    monthly=table(sources['lantu_finance_monthly_2025'],'月度经营数据')
    quarterly=table(sources['lantu_finance_monthly_2025'],'季度汇总')
    monthly_sum=[sum(number(r[i]) for r in monthly) for i in (1,2,3)]
    quarter_sum=[sum(number(r[i]) for r in quarterly) for i in (1,2,3)]
    check('monthly_independent_sum',monthly_sum==[Decimal('44.63'),Decimal('48.92'),Decimal('22400')])
    check('quarterly_independent_sum',quarter_sum==[Decimal('42.63'),Decimal('48.90'),Decimal('21400')])
    quarter_deltas=[]
    for i,q in enumerate(quarterly):
        subtotal=[sum(number(row[c]) for row in monthly[i*3:i*3+3]) for c in (1,2,3)]
        quarter_deltas.append({'quarter':q[0],'monthly_sum':subtotal,'quarter_table':[number(x) for x in q[1:]],
                               'monthly_minus_quarter':[subtotal[c-1]-number(q[c]) for c in (1,2,3)]})

    customer=[s['values'] for s in sources['lantu_sales_customer_list']['segments']][1:]
    amounts=[number(r[4]) for r in customer]
    grades=Counter('S' if amount>5000 else 'A' if amount>1000 else 'B' if amount>300 else 'C' for amount in amounts)
    check('customer_sum_111200_wan',sum(amounts)==111200)
    check('customer_conditional_classes_S10_A2',grades=={'S':10,'A':2})
    check('4500_is_A_not_B',1000<4500<5000 and 'A 级' in by_id['D0088']['answer'])

    bases=table(sources['lantu_org_headcount'],'基地分布')
    headcounts=[sum(number(r[c]) for r in bases) for c in range(1,7)]
    check('headcounts_3860_and_research_1286',headcounts==list(map(Decimal,['3860','1286','620','980','412','562'])))
    org='\n'.join(s['text'] for s in sources['lantu_org_reporting_lines']['segments'])
    edges=re.findall(r'([\u4e00-\u9fff]{2,3})是([\u4e00-\u9fff]{2,3})的上级',org)
    parents={child:parent for parent,child in edges}
    check('functional_branches_have_correct_parents',parents['蒋雯']=='陈依然' and parents['陈依然']=='顾承澜' and parents['邵岚']=='顾承澜')

    inventory=table(sources['lantu_support_spare_inventory'],'备件仓库存')
    stock_amount=sum(number(r[2]) for r in inventory)
    sku_row_sum=sum(number(r[3]) for r in inventory)
    check('inventory_sums',stock_amount==18600 and sku_row_sum==13900)
    check('rate_denominator_and_time_fixture_examples',json.loads(by_id['L0305']['answer'])=={'precision_percent':99.2,'recall_percent':99.2}
          and json.loads(by_id['L0306']['answer'])=={'precision_percent':50.0,'recall_percent':99.2}
          and json.loads(by_id['L0403']['answer'])=={'deadline':'2026-11-04'}
          and json.loads(by_id['L0404']['answer'])=={'deadline':'2026-11-03'})
    check('subset_unknown_and_positive_both_present',json.loads(by_id['L0603']['answer'])=={'is_subset':None}
          and json.loads(by_id['L0604']['answer'])=={'is_subset':True})
    check('no_source_or_gold_payload_side_fields',all(set({'qid','question'})<=q.keys() and 'answer' not in q and 'rubric' not in q for q in questions))

    families=defaultdict(list)
    for d in sources.values():
        m=d['metadata']
        families[m['doc_family']].append({'doc_id':d['doc_id'],'version':m['version'],
            'effective_date':m['effective_date'],'updated_at':m['updated_at'],'status_in_manifest':m['status'],
            'explicit_retirement_statements':[s for s in d['segments'] if '废止' in s['text'] or '取代' in s['text']]})
    versions={key:sorted(items,key=lambda m:(m['effective_date'],m['updated_at'])) for key,items in families.items() if len(items)>1}
    (AUDIT/'version_families.json').write_text(json.dumps(versions,ensure_ascii=False,indent=2),encoding='utf-8')
    evidence={'monthly_totals':monthly_sum,'quarterly_totals':quarter_sum,'quarter_deltas':quarter_deltas,
              'customer_contract_total_wan':sum(amounts),'customer_conditional_grades':dict(grades),
              'headcount_column_totals':headcounts,'inventory_amount_wan':stock_amount,'sku_row_count_sum':sku_row_sum,
              'machine_derived_not_model_generated':True}
    (AUDIT/'independent_arithmetic.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    report={'checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'failed':[k for k,v in checks.items() if not v],'model_calls':0,
            'capability_counts':dict(counts),'independent_ai_review':'not_run',
            'limits':'程序检查证明对应结构/公式；不替代所有自由文本参考的独立语义复查。'}
    (AUDIT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if report['failed']: raise SystemExit(1)


if __name__=='__main__':
    main()
