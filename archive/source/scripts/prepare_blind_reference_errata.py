"""Historical v1 correction recipe for its original run; never applies to the current v2 bank."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/llm_blind_eval_20261029_execution'
LEGACY_RUN = 'blind_20261029T143831Z_12819393'
ORG = ROOT/'enterprise_rag/corpus/docs/org/lantu_org_reporting_lines.md'
EXEC = ROOT/'enterprise_rag/corpus/docs/org/lantu_org_executives.md'
CHART24 = ROOT/'enterprise_rag/corpus/docs/org/lantu_org_chart_2024.md'
CHART25 = ROOT/'enterprise_rag/corpus/docs/org/lantu_org_chart_2025.md'
DUTIES = ROOT/'enterprise_rag/corpus/docs/org/lantu_org_department_duties.md'
EXPORT = ROOT/'enterprise_rag/corpus/docs/legal/lantu_legal_export_control.md'
GLOBAL = ROOT/'enterprise_rag/corpus/docs/company/lantu_company_global.md'
ROAD = ROOT/'enterprise_rag/corpus/docs/tech/lantu_tech_roadmap.md'
IPD = ROOT/'enterprise_rag/corpus/docs/tech/lantu_tech_rd_process.md'
COST = ROOT/'enterprise_rag/corpus/docs/finance/lantu_finance_cost.md'
PLAN = ROOT/'enterprise_rag/corpus/docs/strategy/lantu_strategy_2025.md'
MILE = ROOT/'enterprise_rag/corpus/docs/company/lantu_company_milestones_2025.md'
REVIEW = ROOT/'enterprise_rag/corpus/docs/meeting/lantu_meeting_case_review.md'
SERVICE = ROOT/'enterprise_rag/corpus/docs/support/lantu_support_case_hengxin.md'


def source(path, quote):
    line = next(i for i, value in enumerate(path.read_text(encoding='utf-8').splitlines(), 1) if quote in value)
    return {'file': str(path), 'line': line, 'quote': quote}


def temporal_and_process_errata():
    return [
        {'qid': 'G0072', 'kind': 'reference_correction',
         'corrected_answer': '路线图给出2026年VLA模型在3个场景试点的技术目标；IPD规定概念、计划、开发、验证、发布、生命周期管理六阶段及DCP门禁。可将试点用于验证阶段并把反馈用于后续迭代，这是合理实施衔接建议；原文没有给出一一映射，也没有规定试点必须在验证阶段之后。',
         'reason': '原参考把未写入文档的阶段先后关系当作确定事实。允许合理的流程衔接方案，但不能冒充原文已规定的唯一映射。',
         'sources': [source(ROAD, '视觉-语言-动作模型在 3 个场景试点'), source(IPD, '概念、计划、开发、验证、发布、生命周期管理六个阶段'), source(IPD, '每个阶段设置决策评审点（DCP）')]},
        {'qid': 'I0040', 'kind': 'reference_correction',
         'corrected_answer': '营收实际42.63亿元达到41.5亿元目标及42.63亿元力争值；新签订单48.90亿元较46亿元目标高2.90亿元；出货21,400台较20,000台目标高1,400台。降本目标1.8亿元，前三季度完成1.48亿元、约82%，截至前三季度还差0.32亿元；所引资料没有全年降本实际值，不能据前三季度数据认定全年未达标。',
         'reason': '原参考把前三季度进度错误外推为全年未达标，必须按统计时段比较。',
         'sources': [source(PLAN, '2025 年营业收入目标 41.5 亿元'), source(MILE, '2025 年营业收入 42.63 亿元'), source(COST, '前三季度完成 1.48 亿元')]},
        {'qid': 'J0035', 'kind': 'reference_correction',
         'corrected_answer': '两份材料讨论同一客户项目，但不能认定是同一次故障。复盘纪要创建及更新于2025-03-18，服务事件发生于2025-04-08，早期纪要不能是后来故障的事后复盘。纪要中的停机控制4小时内，与案例的远程12分钟定位、临时降载、备件6小时送达、到场2小时更换、48小时内恢复满产属于不同边界，不能硬串成同一事件时间线。',
         'reason': '原参考仅凭同一客户就合并事件，忽略纪要早于故障的明确时序，也混合停机、降载与恢复满产的计时边界。',
         'sources': [source(REVIEW, '更新时间：2025-03-18 16:00'), source(REVIEW, '停机时间控制在 4 小时内'), source(SERVICE, '2025 年 4 月 8 日凌晨'), source(SERVICE, '指导客户临时降载运行'), source(SERVICE, '产线在 48 小时内恢复满产')]},
    ]


def main():
    run = LEGACY_RUN
    directory = ROOT/'eval_bench/runs'/run
    answers = {row['qid']: row for row in [json.loads(line) for line in
               (directory/'private/answers.jsonl').read_text(encoding='utf-8').splitlines()]}
    spec = importlib.util.spec_from_file_location('reference_facts', ROOT/'eval_bench/_work/bench_facts.py')
    facts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(facts)
    edges = [edge for group in facts.REPORTING.values() for edge in group]
    roles = {'顾承澜': '董事长兼总经理', **{down: role for up, down, role in edges}}
    children = defaultdict(list)
    for up, down, role in edges:
        children[up].append(down)
    corrections = []

    def add(qid, corrected, reason, sources, kind='reference_correction'):
        corrections.append({'qid': qid, 'kind': kind, 'corrected_answer': corrected,
                            'reason': reason, 'sources': sources})

    def edge_source(up, down):
        return source(ORG, f'{up}是{down}的上级。')

    for number in range(1, 17):
        qid = f'A{number:04d}'
        down = answers[qid]['question'].split('的直接上级')[0]
        up = next(parent for parent, child, role in edges if child == down)
        role = roles[up]
        if up == '顾承澜':
            role_source = source(EXEC, '顾承澜，董事长兼总经理')
        else:
            text = ORG.read_text(encoding='utf-8')
            quote = re.search(re.escape(up)+r'担任\s*'+re.escape(role), text).group(0)
            role_source = source(ORG, quote)
        add(qid, f'{down}的直接上级是{up}。{up}的职务是{role}。',
            '生成器把REPORTING三元组中下级的职务写给了上级；按原文的人物与职务绑定纠正。',
            [edge_source(up, down), role_source])
    for number, name in ((26, '陈依然'), (27, '邵岚')):
        add(f'A{number:04d}', '', '生成器将分支职能结构拼成线性链，虚构了上级的上级。原文只披露该人的直接上级顾承澜，没有披露顾承澜的上级；原确定答案不成立，本轮单列不计能力分数。',
            [edge_source('顾承澜', name), source(ORG, '本文件中的“上级”均指行政汇报关系中的直接上级')],
            kind='invalid_question')
    add('A0028', '蒋雯的上级的上级是顾承澜，完整汇报路径为顾承澜 → 陈依然 → 蒋雯。',
        '职能结构存在分支，邵岚不是蒋雯的上级；按两条实际行政边重建路径。',
        [edge_source('顾承澜', '陈依然'), edge_source('陈依然', '蒋雯')])
    for number in range(29, 39):
        qid = f'A{number:04d}'
        match = re.fullmatch(r'(.+)的直接下属有哪些？(.+)是否在其中？', answers[qid]['question'])
        up, named = match.groups()
        subs = children[up]
        add(qid, f"{up}的直接下属有{'、'.join(subs)}。{named}在其中。",
            '生成器检索了所问上级之下再下一层的人，且遗漏跨分支直接下属；改用整份汇报关系文档的真实出边集合。',
            [edge_source(up, down) for down in subs])
    add('A0049', '林晓在LT-C900项目中是项目成员；行政直接上级是周启航（控制器研究所所长），周启航的上级是程知远。',
        '原答案把林晓的伺服算法组组长职务写给了周启航；行政关系姓名正确，职务说明需纠正。',
        [source(ORG, '周启航担任控制器研究所所长'), source(ORG, '项目成员包括林晓与赵一鸣'), edge_source('周启航', '林晓')])
    version_sources = [source(CHART24, '2024 年公司设 11 个一级部门'), source(CHART25, '2025 年 4 月起公司设 12 个一级部门')]
    add('A0084', '2024版有11个一级部门，2025版有12个。2025版新增客户成功部，售后服务部由销售中心下属部门升级为一级部门；战略投资部不再列入2025版一级部门名单。两份名单的集合变化为11+2-1=12。',
        '澄清两个进入一级名单的部门与一个不再列入的部门，避免把11到12误写成仅新增一项；不把未列入推断为组织已撤销。', version_sources)
    add('A0085', '新增客户成功部；售后服务部由销售中心下属部门升级为一级部门。另外，2024版的战略投资部未列入2025版一级部门名单，不能称其余名单完全不变；文档未解释该部门的后续归属。',
        '原参考称其余部门名称保持一致，但2024名单的战略投资部不在2025名单中。', version_sources)
    add('K0002', '', '题干写唐婧是罗嘉树的上级，但所引汇报原文明确罗嘉树是唐婧的上级。不得向不同候选模型临时替换题干，本轮保留原回答并单列排除，未来新题库修正题干后另测。',
        [source(directory/'blind/questions.jsonl', '汇报关系文档又写明唐婧是罗嘉树的上级'), edge_source('罗嘉树', '唐婧')],
        kind='invalid_question')
    add('A0043', '质量部负责人是潘文博。2025版组织架构规定部门负责人向总经理汇报，故其行政直接上级是总经理顾承澜；质量部由COO韩立恒分管，研发中心由CTO沈亦舟分管，二者分管高管不同。',
        '原参考只回答分管高管而漏掉题目所问行政直接上级；必须区分分管与直接汇报关系。',
        [source(DUTIES, '| 质量部 | 潘文博 |'), source(CHART25, '部门负责人向总经理汇报'), source(EXEC, '顾承澜，董事长兼总经理')])
    add('H0067', '两份文档对加密、视觉算法项目需审批及交易筛查的要求基本一致。出口管制指引还明确列出高精度伺服，全球化文档另要求欧盟项目满足CE和GDPR。原文未说明专项审批比法务合规部审批层级更高，不能仅凭审批名称推断层级。',
        '原参考无原文依据地增加了审批层级更高的断言；保留有依据的共同要求与范围差异。',
        [source(EXPORT, '涉及加密、视觉算法与高精度伺服的项目须专项审批'), source(GLOBAL, '涉及加密、视觉算法等敏感技术的项目须经法务合规部审批')])
    corrections.extend(temporal_and_process_errata())
    path = directory/'private/reference_errata.jsonl'
    with path.open('x', encoding='utf-8') as output:
        for correction in corrections:
            output.write(json.dumps(correction, ensure_ascii=False)+'\n')
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    with (OUT/f'{run}_manifest_before_errata.json').open('x', encoding='utf-8') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    manifest['reference_errata'] = 'private/reference_errata.jsonl'
    manifest['sha256'][manifest['reference_errata']] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['grading_revision'] = 'source-errata-v3'
    manifest['grading_revision_at'] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    files = [ORG, EXEC, CHART24, CHART25, DUTIES, EXPORT, GLOBAL, ROAD, IPD, COST, PLAN, MILE, REVIEW, SERVICE]
    receipt = {'run_id': run, 'errata_path': str(path), 'errata_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'corrections': sum(row['kind'] == 'reference_correction' for row in corrections),
               'excluded': [row['qid'] for row in corrections if row['kind'] == 'invalid_question'],
               'original_candidate_questions_changed': False, 'original_frozen_answers_changed': False,
               'source_sha256': {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
    (OUT/'reference_errata_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
