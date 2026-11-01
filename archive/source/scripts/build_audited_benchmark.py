"""Build a versioned, source-audited bank. No candidate answers or model clients."""
import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'eval_bench/_work'
sys.path.insert(0,str(WORK))
from audit_common import ORIGINAL, SOURCES, AUDIT

VERSION='20261030_v2'
TARGET=ROOT/'eval_bench/versions'/VERSION
MODULES=['review_a_org','review_b_hr','review_c_product','review_d_sales','review_e_policy',
         'review_f_finance','review_g_tech','review_h_support_legal','review_i_company',
         'review_j_meeting','review_k_cross']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def norm(value):
    text=unicodedata.normalize('NFKC',value)
    return re.sub(r'\s+','',text).translate(str.maketrans({'。':'.','，':',','：':':'}))


def resolve_quote(evidence):
    doc=SOURCES[evidence['doc_id']]
    segments=doc['segments']
    legacy=evidence['quote']
    if legacy.startswith('### category='):
        meta=doc['metadata']
        original_header=f"### category={meta['category']} | dept={meta['department']} | author={meta['author']}"
        if legacy!=original_header:
            raise ValueError('旧抽取标题与清单不符')
        primary=f"所属部门：{meta['department']}\n作者：{meta['author']}"
        return {**resolve_quote({**evidence,'quote':primary}),
                'legacy_extraction_quote':legacy,
                'evidence_issue':'旧 quote 为抽取器生成标题，已替换为原文件的部门/作者原句。'}
    if legacy.startswith('### created='):
        manifest=ROOT/'enterprise_rag/corpus/manifest.jsonl'
        number,line=next((i,t) for i,t in enumerate(manifest.read_text(encoding='utf-8').splitlines(),1)
                         if json.loads(t)['doc_id']==evidence['doc_id'])
        return {**evidence,'quote':line,'legacy_extraction_quote':legacy,
                'evidence_issue':'旧 quote 为抽取器合成标题；更新时间/status 依据实际清单行，不能当作正文内容。',
                'source_file':manifest.relative_to(ROOT).as_posix(),'source_sha256':sha(manifest),
                'locators':[f'jsonl_line:{number}'],'evidence_kind':'manifest_metadata'}
    needle=norm(evidence['quote'])
    texts=[]
    for s in segments:
        if doc['path'].endswith('.csv'):
            import io
            buf=io.StringIO()
            csv.writer(buf,lineterminator='').writerow(s['values'])
            text=buf.getvalue()
        elif doc['path'].endswith('.xlsx'):
            text=' | '.join('' if v is None else str(v) for v in s['values'])
        else:
            text=s['text']
        texts.append(norm(text))
    body=''.join(texts)
    start=body.find(needle)
    if start<0:
        return None
    end=start+len(needle)
    locators=[]
    offset=0
    for segment,text in zip(segments,texts):
        if offset<end and offset+len(text)>start:
            locators.append(segment['locator'])
        offset+=len(text)
    return {**evidence,'source_file':doc['path'],'source_sha256':doc['sha256'],'locators':locators}


def substantive(doc):
    segments=[]
    for s in doc['segments']:
        locator=s['locator']
        if not s['text'].strip():
            continue
        if locator.startswith('line:') and int(locator.split(':')[1])<12:
            continue
        if locator.startswith('paragraph:') and int(locator.split(':')[1])<10:
            continue
        if locator.startswith('sheet:文档信息/'):
            continue
        segments.append({'locator':locator,'section':s['section'],'text':s['text']})
    return segments


def extra_sources(qid):
    if qid[0]=='A' and int(qid[1:])<=16:
        return ['lantu_org_reporting_lines','lantu_org_executives']
    return {
        'A0041':['lantu_org_chart_2025','lantu_org_department_duties','lantu_org_reporting_lines','lantu_org_executives'],
        'A0043':['lantu_org_chart_2025','lantu_org_department_duties','lantu_org_executives'],
        'A0056':['lantu_org_chart_2024','lantu_org_chart_2025','lantu_company_milestones_2025'],
        'A0084':['lantu_org_chart_2024','lantu_org_chart_2025'],
        'A0085':['lantu_org_chart_2024','lantu_org_chart_2025'],
        'A0121':['lantu_org_executives'],
        'C0074':['lantu_product_painting','lantu_case_05_hailing'],
        'D0081':['lantu_sales_customer_list','lantu_sales_policy'],
        'D0082':['lantu_sales_customer_list','lantu_sales_policy'],
        'D0088':['lantu_sales_policy'],'D0089':['lantu_sales_policy'],
        'F0083':['lantu_finance_budget_2026','lantu_finance_annual_2025','lantu_finance_investor_qa'],
        'H0080':['lantu_legal_sales_contract_template'],
        'H0087':['lantu_legal_purchase_contract_template','lantu_policy_purchase_2025'],
        'I0045':['lantu_strategy_2026','lantu_finance_budget_2026'],
        'J0031':['lantu_meeting_strategy','lantu_strategy_2026','lantu_finance_budget_2026'],
        'K0017':['lantu_case_03_yaoyang','lantu_product_s_series','lantu_solution_photovoltaic'],
        'K0035':['lantu_legal_data_compliance','lantu_tech_vision']
    }.get(qid,[])


def main():
    if (TARGET/'freeze_receipt.json').exists():
        raise ValueError('本版本已冻结；参考答案修正请加独立勘误，改题请建立新版本，不能重建覆盖。')
    receipt=json.loads((AUDIT/'source_snapshot_receipt.json').read_text(encoding='utf-8'))
    for name,expected in receipt['original_bank_sha256'].items():
        if sha(ROOT/'eval_bench'/name)!=expected:
            raise ValueError(f'原始基线变化：{name}；不能混用旧审计记录')
    for doc in SOURCES.values():
        if sha(ROOT/doc['path'])!=doc['sha256']:
            raise ValueError(f"原始资料变化：{doc['path']}")
    reviews=[importlib.import_module(name).r for name in MODULES]
    changes={}
    owners={}
    for review in reviews:
        review.save()
        for qid in ORIGINAL:
            if review.first<=qid<=review.last:
                if qid in owners:
                    raise ValueError(f'重复审核范围：{qid}')
                owners[qid]=review
        for change in review.changes:
            if change['qid'] in changes:
                raise ValueError(f"同题重复修改：{change['qid']}")
            changes[change['qid']]=change
    if owners.keys()!=ORIGINAL.keys():
        raise ValueError('原题库尚有未完成源文核对的题')
    gold,questions,ledger,misses=[],[],[],[]
    for qid,original in ORIGINAL.items():
        row=copy.deepcopy(original)
        resolved=[resolve_quote(e) for e in original['evidence']]
        for ev,found in zip(original['evidence'],resolved):
            if found is None:
                misses.append({'qid':qid,**ev})
        change=changes.get(qid)
        if change:
            row.update(question=change['question'],answer=change['answer'])
            dids=list(dict.fromkeys(change['sources']+extra_sources(qid)))
            row['evidence']=[{'doc_id':did,'doc_title':SOURCES[did]['title'],
                'source_file':SOURCES[did]['path'],'source_sha256':SOURCES[did]['sha256'],
                'version':SOURCES[did]['metadata']['version'],
                'effective_date':SOURCES[did]['metadata']['effective_date'],
                'updated_at':SOURCES[did]['metadata']['updated_at'],
                'segments':substantive(SOURCES[did])} for did in dids]
            row['expected_doc_ids']=dids
        else:
            row['evidence']=resolved
        row['benchmark_version']=VERSION
        row['evaluation_track']='domain_closed_book'
        row['source_review']='Codex-source-review; independent review pending'
        row['rubric']={'method':'source_based_semantic_review',
            'required':'完整回答题目所问关键事实、关系、单位和条件；参考中的额外背景不自动成为必答点。',
            'allowed':'同义表达、等值单位和合理舍入；开放设计题允许有证据且标为建议的其他方案。',
            'not_allowed':'否定/对象/时间/范围/因果相反，或将明确禁止事项猜为已获豁免。关键词命中不代表正确。',
            'note':change['rubric_note'] if change else '按所问题意审阅，不能只机械匹配整段参考字面。'}
        if qid=='C0070':
            row['external_sources']=[
                {'title':'GB 15578-2008 电阻焊机的安全要求','url':'https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=11D843BACB058DD5AB4F4C9238C6B835'},
                {'title':'ISO 10218-1:2025 Industrial robots','url':'https://www.iso.org/standard/73933.html'}]
        gold.append(row)
        questions.append({**{k:row[k] for k in ('qid','question','question_type','question_type_label',
            'difficulty','difficulty_label','answer_type','multi_hop_depth','expected_doc_ids')},
            'answer_ref':qid,'benchmark_version':VERSION,'evaluation_track':row['evaluation_track']})
        ledger.append({'qid':qid,'review_status':'source_reviewed', 'independent_review_status':'not_run',
            'question_changed':row['question']!=original['question'],'answer_changed':row['answer']!=original['answer'],
            'original_question':original['question'],'original_answer':original['answer'],
            'revised_question':row['question'],'revised_answer':row['answer'],
            'reason':change['reason'] if change else '按原始资料复核，保留题目和参考答案；判分以问题关键要求为准。',
            'domain_review':owners[qid].checked,'original_evidence_with_locators':resolved,
            'revised_evidence':row['evidence'],
            'question_sha256':hashlib.sha256(row['question'].encode('utf-8')).hexdigest()})
    write_jsonl(AUDIT/'unresolved_original_quotes.jsonl',misses)
    if misses:
        raise ValueError(f'{len(misses)}条原引文尚未对应新抽取位置，见 unresolved_original_quotes.jsonl')
    from gen_l_capabilities import build as build_capabilities
    additions=build_capabilities()
    for row in additions:
        row['benchmark_version']=VERSION
        row['source_review']='deterministic fixture derivation; independent review pending'
        row['evidence']=[{'kind':'question_fixture','question_sha256':hashlib.sha256(row['question'].encode('utf-8')).hexdigest()}]
        gold.append(row)
        questions.append({k:row[k] for k in ('qid','question','question_type','question_type_label',
            'difficulty','difficulty_label','answer_type','multi_hop_depth','expected_doc_ids',
            'answer_ref','benchmark_version','evaluation_track','capability')})
    TARGET.mkdir(parents=True,exist_ok=True)
    write_jsonl(TARGET/'question_bank.jsonl',questions)
    write_jsonl(TARGET/'answer_bank.jsonl',gold)
    write_jsonl(TARGET/'grading_rules.jsonl',[])
    write_jsonl(AUDIT/'audit_ledger.jsonl',ledger)
    write_jsonl(AUDIT/'changes.jsonl',[row for row in ledger if row['question_changed'] or row['answer_changed']])
    manifest={'benchmark_version':VERSION,'status':'audited_draft_validation_pending',
        'original_questions':len(ORIGINAL),'source_documents':len(SOURCES),
        'added_context_questions':len(additions),'total_questions':len(questions),
        'reviewed_questions':len(ledger),'modified_questions_or_answers':len(changes),
        'question_text_changed':sum(r['question_changed'] for r in ledger),
        'answer_text_changed':sum(r['answer_changed'] for r in ledger),
        'independent_review_status':'not_run','raw_model_requests_during_audit':0,
        'files':{f:sha(TARGET/f) for f in ('question_bank.jsonl','answer_bank.jsonl','grading_rules.jsonl')},
        'source_snapshot_sha256':receipt['source_snapshot_sha256']}
    (TARGET/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    main()
