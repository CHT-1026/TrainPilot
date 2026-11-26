"""从已批准预览扩展V6修复集：新规则合成对照 + V5训练侧回放。"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import re

from model.dataset.build_agent_data import render_prompt
from model.dataset.build_agent_v5 import canonical, corpus_pairs, rag_row, synthetic_chunk
from model.dataset.preview_agent_v6 import questions
from model.trainer.agent_rewards import intent_reward

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'model/dataset/datasets/trainpilot_agent'
SEED = 20261030
INITIAL_ADAPTER_SHA = '2c6d865c84c8b1f432250e69ab9d72217263fda4a8811e15ca341abc0d830eb1'
SPLITS = {'train':range(10), 'val':range(10,12), 'test':range(12,15)}
REPLAY_QUOTAS = {'intent_parse':280, 'enterprise_rag':140, 'oa_rag':70, 'direct_reply':22, 'agent_reply':8}

CONFIRM = [
 '我核过你准备的内容了，就让那一份现在生效。',
 '前面那一步我已经检查过，现在正式授权办理。',
 '当前预览无需再改，我同意按这一版执行。',
 '请落实你眼下保留的那项待确认动作，我已核准。',
 '就以目前待办的内容为准，授权你继续到执行阶段。',
 '我认可刚才准备好的操作，本轮给出正式确认。',
 '当前这项的内容我都看过了，开始办理即可。',
 '请让现在这一份预案落地，我确认不再修改。',
 '我已复核现有草稿，执行范围就限定这一项。',
 '目前等待我确认的步骤可以照案处理，现予确认。',
 '这次确认针对你现存的那份预览，现在可以办理。',
 '请把已经准备并等待确认的内容付诸执行，我同意。',
 '我在这一轮给出授权：按当前待办内容正式处理。',
 '你现有待确认草稿我接受，接着办这一项吧。',
 '以尚未执行的当前预览为唯一范围，我现在确认执行。',
]
CANCEL = [
 '先把手头这一步停掉，我不继续办理这一项了。',
 '眼下待确认的操作不要执行了，放弃这一步。',
 '我不批准当前草稿进入执行，先把这项预案作废。',
 '这轮准备好的动作到此为止，不要继续落实。',
 '当前等待确认的那一步取消，我不再授权它。',
 '先撤销本次操作计划，不要把预览变成实际办理。',
 '我改变主意了，放弃现在这一项待确认流程。',
 '现有待办不用办下去，终止本轮准备的动作。',
 '暂存的这项操作我不确认，直接放弃它。',
 '这一步就不做了，请停留在未执行的状态。',
 '我明确拒绝执行当前预案，请结束这项等待确认。',
 '现在准备的办理动作不再需要，别执行它。',
 '本轮待确认内容作废，我取消的是这一操作计划。',
 '请停止当前尚未执行的步骤，我不再办理它。',
 '现存那项预览不要落实，终止这一项的执行安排。',
]
QUOTED = [
 '培训材料把“{phrase}”当作例句，我想弄懂这一步的含义，不要动当前草稿。',
 '我在读操作手册，请解释“{phrase}”这段措辞，不是要求现在办理。',
 '这里只讨论流程：资料中的“{phrase}”表示什么？请保持当前事项不变。',
 '我想了解“{phrase}”的制度含义，这句是引用，不是执行授权。',
 '说明书举了“{phrase}”这个说法，请讲解规则，别据此处理实际申请。',
 '请把“{phrase}”当作流程示例解释，我尚未决定做任何变更。',
 '我们只做业务用语咨询：“{phrase}”涉及哪一步，不要替我操作。',
 '我问的是教程里“{phrase}”的含义，当前草稿仍保持原样。',
 '先讲清楚制度中的“{phrase}”，这不是对现有待办的确认。',
 '我只是研究操作说明中“{phrase}”这句话，当前没有新的执行指令。',
 '请解释文档示例“{phrase}”的流程意义，此刻不实际办理。',
 '引用手册里的“{phrase}”问一下是什么意思，先不要改动业务状态。',
 '这里的“{phrase}”来自课程案例，我要咨询规则，不是授权动作。',
 '请说明制度文案“{phrase}”怎样理解，我并未要求执行当前事项。',
 '我在讨论“{phrase}”这段例示指令的意思，请勿把讨论当作确认。',
]
ACTUAL = [
 '现在不是解释用语，我已经复核当前草稿，明确授权执行这个待办。',
 '我对现存预览给出实际确认，请办理它而非讲解规则。',
 '这一轮是执行授权，当前待确认事项按现有内容办。',
 '不用说明流程了，我确认落实你当前保留的那项预案。',
 '本条消息是正式同意，请执行眼下等待确认的动作。',
 '我并非引用教程，而是在确认当前操作，现在办理。',
 '请从现有待办进入实际处理阶段，这是我的明确授权。',
 '我要求真正落实当前那份预览，不是在询问它的含义。',
 '现在对已准备的这一项给出执行确认，内容照旧。',
 '我已检查当前待办，授权立即按照这份内容办理。',
 '此刻是业务确认而非咨询，落实你保留的现有预案。',
 '我现在同意执行等待确认的那一项，请照现有内容处理。',
 '这条消息用来授权办理当前待办，不是示例讨论。',
 '按你此刻准备的预览正式实施，我在这里确认。',
 '我认可当前待确认事项并要求执行，不用再解释步骤。',
]
CAPABILITY = [
 '你整理业务资料时，会把我没写出的必要条件当成既定事实吗？',
 '如果输入中缺少关键内容，你会自己补出一个值吗？',
 '你能把尚未得到执行结果的事项说成已经办成吗？',
 '你会不会为了完成表格而替我编出未知信息？',
 '助手遇到未说明的条件时，是先询问还是自行认定？',
 '我没有提供的申请细节，你是否会直接猜好填进去？',
 '你可以把未获业务系统确认的结果宣称为成功吗？',
 '介绍一下你的工作边界，未知的必要字段会自行补齐吗？',
 '你在整理需求时会擅自假设我未表达的日期吗？',
 '如果资料和我的输入都没写某个值，你会凭经验补上吗？',
 '你的职责允许替用户虚构没有提供的办理条件吗？',
 '你能不能把只是准备好的草稿当作已经实际提交？',
 '作为助手，你会把尚未核实的业务信息写成确定事实吗？',
 '缺少申请信息时，你自身会猜测还是向用户核实？',
 '想确认你的能力边界：没有实际回执也能声称办理成功吗？',
]
POLICY = [
 '企业的请假制度对申请材料缺项规定了怎样的补交步骤？',
 '请查企业规则，审批资料没有交齐时流程应该怎样处理？',
 '公司制度中申请缺少必要字段时允许进入哪个办理环节？',
 '我想了解单位规定：申请材料不完整的补正程序是什么？',
 '企业对未提供必要证明的请假申请有哪些流程要求？',
 '根据公司的办理制度，缺项申请应怎样退回补充？',
 '请解释企业流程中提交后补交材料的规则，不是办理新申请。',
 '公司的制度对申请资料缺失的核验与补录怎样规定？',
 '我只咨询单位政策，证明材料不足时应走什么流程？',
 '帮我查一下企业制度如何处理尚未填写完整的申请。',
 '企业规则对缺少申请证明的补正步骤有哪些要求？',
 '我在了解公司审批制度，材料未齐的流程节点怎么安排？',
 '单位的业务规范里，申请出现必要信息空缺时如何补正？',
 '请依据企业制度讲讲补充未提交申请材料的办理步骤。',
 '公司请假流程对必填信息缺项的处理办法是什么？',
]
ABANDON = [
 '撤单这一步作废，原来已经提交的申请继续保留。',
 '不要执行当前撤回计划，已提交的那张单不动。',
 '我放弃撤回这项操作，保留原申请的现有状态。',
 '这次不撤单了，取消的是撤回预案而不是原申请。',
 '撤回流程现在终止，原先提交的申请仍然有效。',
 '请停止本次撤单动作，我决定保留已提交申请。',
 '放下当前撤回待办，别去取消已经提交的那一单。',
 '我不打算撤回了，把本轮撤单计划结束掉。',
 '原申请我要继续保留，当前撤回步骤不要落实。',
 '取消眼下准备的撤回操作，已提交记录保持不变。',
 '这次撤单预案不用执行，原申请照常保留。',
 '我撤销的是撤回意向，请别动那张已提交申请。',
 '当前撤回办理先作罢，已经提交的单子仍保留。',
 '不要落实本轮撤单待办，我要保持原申请有效。',
 '终止这一项撤回安排，不要撤掉原来的申请。',
]
EDIT = [
 '请更新当前草稿：', '这份预览调整如下：', '沿用当前任务，仅作这些修改：',
 '先别执行，改一下现有草稿：', '请整理当前预案的新版本：', '当前待办按下面内容更新：',
 '我补充本轮草稿的修改要求：', '请以这些信息修订现有预览：', '继续整理这份草稿：',
 '把当前任务的草稿改成这样：', '我想修订待确认内容：', '现有预案需要做以下变更：',
 '请生成当前草稿的修订预览：', '本轮只更新既有草稿：', '先保存这些草稿调整供我核对：',
]
RAG_QUESTIONS = [
 '参数样机 {device} 的额定负载和采购价格分别是多少？',
 '请按给定资料说明 {device} 的负载上限与购置价格。',
 '我需要 {device} 的额定承载和采购价，资料没有的请明确说。',
 '帮我查参数样机 {device}：额定负载是多少，采购价格是多少？',
 '依据这些记录，{device} 的额定负载、采购价格各是什么？',
 '请分别回答 {device} 的负载参数及采购金额。',
 '现有证据能说明 {device} 的额定负载和采购价格吗？',
 '只依据所给资料，报告 {device} 的负载与采购报价。',
 '需要核实参数样机 {device} 的额定承载值和采购价。',
 '请提取 {device} 的两项信息：额定负载、采购价格。',
 '我在核对 {device} 的额定负载以及采购价格，请依资料答复。',
 '给定记录里 {device} 的额定负载与采购价分别写了多少？',
 '请根据证据查明 {device} 的额定承载量及采购价格。',
 '我想同时了解 {device} 的负载指标和采购价格，缺项不要猜。',
 '按目前提供的记录，{device} 的额定负载和采购价各能确定什么？',
]
ROLE = [
 '请用两句话界定你的业务职责，并说明谁来真正执行变更。',
 '介绍一下你的助手身份，以及业务写操作由谁完成。',
 '你负责理解哪些业务需求，实际办理又由谁执行？',
 '请简述你在这个项目里的角色和执行权限边界。',
 '说明你的名称与本项目职责，不要列出未接入的业务。',
 '你在企业业务流程中做什么，能自行完成写操作吗？',
 '请讲清当前助手的名称、业务范围和执行分工。',
 '作为这个项目的助手，你能协助哪些事，谁负责真正落地？',
 '请给出你的身份介绍，并区分整理信息和实际执行。',
 '你在本项目中承担什么职责？业务变更怎样才算执行？',
 '概括当前助手在企业业务中的职责和实际操作边界。',
 '请介绍你的业务定位，并说明你是否直接修改业务数据。',
 '说一下你的助手名称、能协助的事务和实际执行者。',
 '我想知道当前助手负责什么，以及谁来落实业务操作。',
 '请介绍你在本项目的身份，避免把建议说成已执行。',
]
NEW_TASK = [
 '现在换一个独立任务：', '旧草稿先不处理，我另办一件事：', '本轮开始新的业务，不继承旧任务：',
 '请建立另一项草稿：', '接下来办理新事项，与前一份分开：', '不要沿用旧任务，我新提一件事：',
 '当前改为一个新业务目标：', '前面的预案不参与这次，请另起任务：', '重新整理另一件事项：',
 '本条提出新任务，旧字段不要带入：', '请为独立的新事项准备预览：', '我现在另办一项业务：',
 '这次改成新的申请事项：', '先把当前目标换为新任务：', '新建这一项业务草稿，旧任务信息不混用：',
]
TABLE_PREFIX = [
 '请核对演练表：', '根据给定表格查明：', '我想查询演练表：', '请只依据这张表回答：',
 '核验表中的这一项：', '请从演练表提取：', '阅读所给表格后回答：', '帮我确认表格内容：',
 '请按表内记录说明：', '依据这份演练表查一下：', '需要核实以下表格信息：',
 '请查给定表中的内容：', '表格信息请这样核对：', '我需要你依据表格回答：', '只从这张表取值：',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def save(path, obj):
    Path(path).write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def repair_rows(contract):
    template=(DATA/'intent_prompt_v4.txt').read_text(encoding='utf-8')
    for split, families in SPLITS.items():
        for family in families:
            group=f'v6/repair-family-{family:02d}'
            start=date(2027,4,1)+timedelta(days=family*3)
            leave={'action_type':'submit_leave','draft':{'leave_type':'事假','start_date':start.isoformat(),
                   'end_date':(start+timedelta(days=1)).isoformat(),'reason':f'核对第{family+1}批设备交付清单'}}
            app=f'leave_v6_f{family:02d}_a'
            approval={'action_type':'approve','draft':{'app_id':app,'decision':'approved',
                       'decision_label':'通过','comments':f'第{family+1}批书面材料已收齐'}}
            withdrawal={'action_type':'cancel_leave','draft':{'app_id':f'leave_v6_f{family:02d}_w'}}
            histories=[[],[{'role':'assistant','content':'上一项已经办理完成，目前无待确认动作。'}],
                       [{'role':'assistant','content':'前一份草稿已放弃，目前没有等待确认的预案。'}],
                       [{'role':'assistant','content':'刚才仅查询过申请列表，没有创建新的待办。'}]]
            states=[(None,h) for h in histories]+[(leave,[]),(approval,[]),(withdrawal,[]),
                    (leave,[{'role':'user','content':'这份请假暂只做预览。'},{'role':'assistant','content':'已准备当前请假草稿，等待确认。'}])]

            def intent(suffix, category, user, target, pending=None, fields=None, history=None):
                scenario={'current_date':'2027-04-01','history':history or [],'pending_action':deepcopy(pending),'user_message':user}
                expected={'intent':target,'extracted_info':fields or {}}
                return {'id':f'v6-{split}-f{family:02d}-{suffix}','group_id':group,'mode':'intent_parse',
                    'category':category,'origin':'approved_rule_synthetic_repair','scenario':scenario,'expected':expected,
                    'messages':[{'role':'user','content':render_prompt(template,scenario)},
                                {'role':'assistant','content':canonical(expected)}]}

            for j,(pending,history) in enumerate(states):
                yield split,intent(f'confirm-{j}','confirmation_state',CONFIRM[family],
                    'confirm_action' if pending else 'clarify',pending,history=history)
                yield split,intent(f'cancel-{j}','cancellation_state',CANCEL[family],
                    'cancel_action' if pending else 'clarify',pending,history=history)
            for j,pending in enumerate((None,leave,approval,withdrawal)):
                phrase=('确认送出申请','确认执行这次审批','撤掉已提交申请','放弃撤单计划')[j]
                yield split,intent(f'quoted-{j}','quoted_policy',QUOTED[family].format(phrase=phrase),'policy_qa',pending)
                yield split,intent(f'actual-{j}','actual_confirmation',ACTUAL[family],
                    'confirm_action' if pending else 'clarify',pending)
            yield split,intent('capability','capability_boundary',CAPABILITY[family],'general_chat')
            yield split,intent('policy','company_policy',POLICY[family],'policy_qa')
            yield split,intent('abandon-withdraw','withdraw_abandon',ABANDON[family],'cancel_action',withdrawal)
            yield split,intent('confirm-withdraw','withdraw_confirm',ACTUAL[family], 'confirm_action',withdrawal,
                               history=[{'role':'assistant','content':'目前预览的是撤回已提交申请；放弃此动作则保留原申请。'}])
            other=f'leave_v6_f{family:02d}_other'
            yield split,intent('withdraw-other','explicit_withdrawal',
                EDIT[family]+f'我要撤回另一个已提交编号{other}，不是当前草稿里的那张，请先给撤回预览。',
                'delete_application',withdrawal,{'app_id':other})
            end=(start+timedelta(days=3)).isoformat()
            comment=f'第{family+1}批预算口径尚未统一，请保留原始明细。'
            reason=f'补核第{family+1}批交付附件'
            updates=[
                ('leave-end','full_leave_fields',f'只把结束日期延到{end}，其他已知信息保留。','submit_leave',leave,{**leave['draft'],'end_date':end}),
                ('leave-reason','full_leave_fields',f'事由改为“{reason}”，原有假别和日期不动。','submit_leave',leave,{**leave['draft'],'reason':reason}),
                ('new-leave','new_task_no_inheritance','换成一个新请假任务，假别是年假，事由为安排休息，起止日期先不确定。','submit_leave',leave,{'leave_type':'年假','reason':'安排休息'}),
                ('approval-decision','full_approval_fields',f'当前结论改为驳回，意见逐字写“{comment}”，只做预览。','approve',approval,{'app_id':app,'decision':'驳回','comments':comment}),
                ('approval-comment','full_approval_fields',f'审批结论保持不变，意见原文换成“{comment}”。','approve',approval,{'app_id':app,'decision':'通过','comments':comment}),
                ('new-approval','new_task_no_inheritance',f'新审批仅针对{other}，结论驳回，先不填写意见，也不沿用之前的意见。','approve',approval,{'app_id':other,'decision':'驳回'}),
            ]
            for suffix,category,text,target,pending,fields in updates:
                prefix=NEW_TASK[family] if category=='new_task_no_inheritance' else EDIT[family]
                yield split,intent(suffix,category,prefix+text,target,pending,fields)

            device=f'TP-V6-{family:02d}';load=51+family*7;price=180000+family*9300
            question=RAG_QUESTIONS[family].format(device=device)
            load_text=f'参数样机{device}的额定负载为{load} kg。'
            price_text=f'参数样机{device}的采购价格为{price}元。'
            for j,kind in enumerate(('load','price','both','none','load_distractor','price_distractor')):
                text=load_text if kind.startswith('load') else price_text if kind.startswith('price') else load_text+'\n'+price_text if kind=='both' else f'参数样机{device}运输时应固定底座。'
                evidence=[synthetic_chunk(f'v6-f{family:02d}-{kind}',text,title='TrainPilot演练参数（合成资料）')]
                index=1
                if kind.endswith('distractor'):
                    evidence.insert(0,synthetic_chunk(f'v6-f{family:02d}-other-{kind}',f'其他参数样机TP-OTHER-{family:02d}的负载为999 kg，采购价格为999999元。',title='TrainPilot干扰参数（合成资料）'))
                    index=2
                if kind.startswith('load'):
                    claims=[{'text':load_text,'sources':[index]},
                            {'text':'当前资料未提供该样机的采购价格。','sources':[index]}]
                elif kind.startswith('price'):
                    claims=[{'text':price_text,'sources':[index]},
                            {'text':'当前资料未提供该样机的额定负载。','sources':[index]}]
                elif kind=='both':
                    claims=[{'text':load_text,'sources':[1]},{'text':price_text,'sources':[1]}]
                else:
                    claims=[]
                for mode in ('enterprise','oa'):
                    row=rag_row(contract,f'v6-{split}-f{family:02d}-{mode}-{kind}',group,mode,question,
                        deepcopy(evidence),deepcopy(claims),'full_evidence' if kind=='both' else 'unsupported' if kind=='none' else 'partial',synthetic=True)
                    row['origin']='approved_rule_synthetic_repair'
                    yield split,row
            item_a=f'资料补录{family+1}';item_b=f'信息更正{family+1}';days=3+family%4
            table=synthetic_chunk(f'v6-f{family:02d}-table',f'| 类别 | 事项 | 办理时限 |\n| 申请 | {item_a} | 2个工作日 |\n| 申请 | {item_b} | {days}个工作日 |',title='TrainPilot流程演练表（合成制度）')
            for kind,q,claims in (
                ('ambiguous',TABLE_PREFIX[family]+'“申请”这一类的办理时限是多少？只给唯一时限，不自行挑事项。',[]),
                ('specific',TABLE_PREFIX[family]+f'类别“申请”、事项“{item_b}”的办理时限是多少？',[{'text':f'类别为“申请”、事项为“{item_b}”的办理时限为{days}个工作日。','sources':[1]}]),
            ):
                row=rag_row(contract,f'v6-{split}-f{family:02d}-table-{kind}',group,'enterprise',q,[deepcopy(table)],claims,
                            'table_ambiguity' if kind=='ambiguous' else 'table_unique_key',synthetic=True)
                row['origin']='approved_rule_synthetic_repair'
                yield split,row
            answer='我是 TrainPilot，负责理解业务需求、整理请假和审批信息，并依据给定资料协助查询制度。实际业务变更由系统在确认和权限校验后执行，我不会把未执行的操作说成已完成。'
            q=ROLE[family]
            for mode,system in (('direct_plain',False),('direct_system',True),('agent',False)):
                messages=([{'role':'system','content':'你是 TrainPilot，一名企业业务助手。'}] if system else [])
                messages += [{'role':'user','content':contract['agent_reply_template'].replace('@@USER@@',q) if mode=='agent' else q},
                             {'role':'assistant','content':answer}]
                yield split,{'id':f'v6-{split}-f{family:02d}-role-{mode}','group_id':group,'mode':'agent_reply' if mode=='agent' else 'direct_reply',
                    'category':'identity_scope','origin':'approved_rule_synthetic_repair','question':q,'messages':messages}


def user_question(row):
    if row.get('scenario'):
        return row['scenario']['user_message']
    if row.get('question') or row.get('user'):
        return row.get('question') or row['user']
    messages=row.get('prompt') or row['messages'][:-1]
    text=messages[-1]['content']
    if '\n用户输入：\n' in text:
        return text.rsplit('\n用户输入：\n',1)[1]
    if '\n用户消息：' in text:
        return text.split('\n用户消息：',1)[1].split('\n系统已知结果：',1)[0].strip()
    return text


def rewrite(row, contract):
    """只变更已批准的提示词身份/部分回答规则，不改业务目标或证据。"""
    if row['mode'].endswith('_rag'):
        mode=row['mode'].removesuffix('_rag')
        updated=rag_row(contract,row['id'],row['group_id'],mode,row['question'],deepcopy(row['evidence']),
                        deepcopy(row['expected_claims']),row['category'])
        return {**row,'messages':updated['messages']}
    updated=deepcopy(row)
    if row['mode']=='agent_reply':
        updated['messages'][0]['content']=contract['agent_reply_template'].replace('@@USER@@',user_question(row))
    return updated


def prompt_key(row):
    return canonical(row.get('prompt') or row['messages'][:-1])


def heldout_sets(contract):
    prompts, users, source_hashes=set(),set(),{}
    groups=set()
    for version in ('v2','v3','v4','v5','grpo_v1','grpo_v2'):
        for split in ('val','test'):
            path=DATA/version/(split+'.jsonl')
            if not path.exists():
                continue
            source_hashes[path.relative_to(DATA).as_posix()]=sha(path)
            for row in read_rows(path):
                prompts.add(prompt_key(row));users.update(questions(row));users.add(user_question(row))
                if version=='v5':
                    groups.add(row['group_id'])
                if row.get('mode','').endswith('_rag') or row.get('mode')=='agent_reply':
                    prompts.add(prompt_key(rewrite(row,contract)))
    return prompts,users,groups,source_hashes


def valid_table_replay(row):
    if row['category']!='table_lookup':
        return True
    for claim in row['expected_claims']:
        if len(claim['sources'])!=1:
            return False
        evidence=row['evidence'][claim['sources'][0]-1]
        allowed={text for _,texts,_ in corpus_pairs(evidence,row['mode'].removesuffix('_rag')) for text in texts}
        if claim['text'] not in allowed:
            return False
    return True


def replay_rows(contract, held_prompts, held_groups):
    buckets=defaultdict(list);filtered=Counter();seen=set()
    for row in read_rows(DATA/'v5/train.jsonl'):
        if row['group_id'] in held_groups:
            filtered['heldout_group']+=1;continue
        if row['mode'].endswith('_rag') and not valid_table_replay(row):
            filtered['ambiguous_or_unverifiable_table']+=1;continue
        updated=rewrite(row,contract)
        key=prompt_key(updated)
        if prompt_key(row) in held_prompts or key in held_prompts:
            filtered['heldout_prompt']+=1;continue
        if key in seen:
            filtered['duplicate_prompt']+=1;continue
        seen.add(key)
        subtype=(row['expected']['intent']+'/'+row['category']) if row['mode']=='intent_parse' else row['category']
        buckets[(row['mode'],subtype)].append(updated)
    rng=random.Random(SEED)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen=[]
    def take(keys,quota):
        keys=list(keys)
        mode_rows=[]
        while len(mode_rows)<quota and keys:
            for k in list(keys):
                mode_rows.append(buckets[k].pop())
                if not buckets[k]:keys.remove(k)
                if len(mode_rows)==quota:break
        if len(mode_rows)!=quota:
            raise ValueError(f'回放候选不足: {len(mode_rows)}/{quota}')
        return mode_rows
    for mode,quota in REPLAY_QUOTAS.items():
        keys=sorted(k for k in buckets if k[0]==mode)
        if mode=='intent_parse':
            intents=sorted({k[1].split('/',1)[0] for k in keys})
            for i,intent in enumerate(intents):
                count=quota//len(intents)+(i<quota%len(intents))
                chosen.extend(take([k for k in keys if k[1].split('/',1)[0]==intent],count))
        else:
            chosen.extend(take(keys,quota))
    for row in chosen:
        row['source_id']=row['id'];row['source_file']='v5/train.jsonl'
        row['id']='v6-replay-'+row['id'];row['origin']='v5_train_replay_with_approved_prompt_sync'
    return chosen,dict(filtered)


def validate(splits, contract):
    held_prompts,held_users,held_groups,_=heldout_sets(contract)
    ids=set();owners={};prompts={};new_questions={};counts={}
    for split,records in splits.items():
        for row in records:
            assert row['id'] not in ids,row['id'];ids.add(row['id'])
            group=row['group_id']
            assert owners.setdefault(group,split)==split,group
            key=prompt_key(row)
            assert key not in held_prompts,('old heldout prompt',row['id'])
            assert key not in prompts,('duplicate prompt',row['id'],prompts.get(key))
            prompts[key]=row['id']
            if row['origin']=='approved_rule_synthetic_repair':
                q=user_question(row)
                assert q not in held_users,('old heldout user question',row['id'],q)
                assert new_questions.setdefault(q,split)==split,('cross split question',q)
            else:
                assert split=='train' and row['source_file']=='v5/train.jsonl'
                assert row['group_id'] not in held_groups
            if row['mode']=='intent_parse':
                expected=row['expected'];answer=json.loads(row['messages'][-1]['content'])
                assert intent_reward(canonical(answer),expected)['reward']==1,row['id']
                if row['category'] in ('confirmation_state','actual_confirmation','withdraw_confirm'):
                    assert expected['intent']==('confirm_action' if row['scenario']['pending_action'] else 'clarify')
                if row['category']=='cancellation_state':
                    assert expected['intent']==('cancel_action' if row['scenario']['pending_action'] else 'clarify')
            elif row['mode'].endswith('_rag'):
                assert all(c['sources'] and all(1<=i<=len(row['evidence']) for i in c['sources']) for c in row['expected_claims'])
                if row['mode']=='enterprise_rag':
                    target=json.loads(row['messages'][-1]['content'])
                    assert target=={'refused':not bool(row['expected_claims']),'claims':row['expected_claims']}
            if row['mode']!='direct_reply':
                assert 'TrainPilot' in row['messages'][0]['content'],row['id']
            elif row['messages'][0]['role']=='system':
                assert 'TrainPilot' in row['messages'][0]['content'],row['id']
        counts[split]={'records':len(records),'repair':sum(r['origin']=='approved_rule_synthetic_repair' for r in records),
            'replay':sum(r['origin'].startswith('v5_train_replay') for r in records),
            'modes':dict(Counter(r['mode'] for r in records)),'categories':dict(Counter(r['category'] for r in records)),
            'intents':dict(Counter(r['expected']['intent'] for r in records if r['mode']=='intent_parse')),
            'groups':len({r['group_id'] for r in records})}
    assert set(counts['train']['intents'])==set(('submit_leave','delete_application','approve','query_status','query_pending',
        'policy_qa','get_suggestion','confirm_action','cancel_action','general_chat','clarify'))
    return counts


def build(output,contract_path):
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    held_prompts,_,held_groups,sources=heldout_sets(contract)
    splits={s:[] for s in SPLITS}
    for split,row in repair_rows(contract):splits[split].append(row)
    replay,filtered=replay_rows(contract,held_prompts,held_groups)
    splits['train'].extend(replay)
    counts=validate(splits,contract)
    assert [counts[s]['records'] for s in SPLITS]==[1040,104,156]
    output.mkdir(exist_ok=False)
    for split,records in splits.items():
        random.Random(SEED+len(split)).shuffle(records)
        (output/(split+'.jsonl')).write_bytes((''.join(canonical(r)+'\n' for r in records)).encode('utf-8'))
    save(output/'agent_contract.json',contract)
    manifest={'version':'v6','assistant_name':'TrainPilot','approval':'2026-10-30 approved 26 preview cases, TrainPilot identity and OA partial-answer policy',
        'seed':SEED,'training_start':'existing_v5_adapter_on_original_full_sft','initial_adapter_sha256':INITIAL_ADAPTER_SHA,
        'new_optimizer':True,'resume_old_training_state':False,'model_evaluated':False,
        'data_size_rationale':'520 new repair records plus 520 stratified V5 training replay; first bridge experiment, not a claim of full coverage',
        'outputs':{s:{'file':s+'.jsonl',**counts[s],'sha256':sha(output/(s+'.jsonl'))} for s in SPLITS},
        'sources_sha256':{**sources,'v5/train.jsonl':sha(DATA/'v5/train.jsonl'),'v6_preview/samples.jsonl':sha(DATA/'v6_preview/samples.jsonl')},
        'agent_contract_sha256':sha(output/'agent_contract.json'),'agent_prompt_versions':contract['prompt_versions'],
        'replay_filters':filtered,'old_eval_prompt_overlap':0,'new_repair_old_eval_question_overlap':0,
        'group_overlap_between_splits':0,'new_repair_question_overlap_between_splits':0,
        'family_split':{s:list(v) for s,v in SPLITS.items()},
        'limitations':['15 synthetic wording/evidence families, each with 52 state/task variants; records are not independent user semantics',
            'new train/val/test repair families separated 10/2/3; all old replay only in train',
            'exact overlap checks do not prove semantic or fact-level independence',
            'RAG supports supplied evidence only; not a live retrieval/tool execution evaluation',
            'ambiguous table input is refused under current RAG protocol; no new clarification JSON field',
            'no formal training started; existing launcher does not yet load V5 as initialization']}
    save(output/'manifest.json',manifest)
    md=['# V6修复型SFT正式数据','', '2026-10-30。用户已确认预览与OA部分回答规则。训练起点：原Full SFT + 已有V5 LoRA；建立新优化器，保留V5/GRPO50对照。','',
        '| 划分 | 新修复 | 旧任务回放 | 合计 |','|---|---:|---:|---:|']
    for s in SPLITS:md.append(f"| {s} | {counts[s]['repair']} | {counts[s]['replay']} | {counts[s]['records']} |")
    md+=['','新样本包括可信待办确认/取消、引用咨询与执行、能力边界与制度咨询、撤回方向、完整字段更新、新任务不继承旧字段、部分/全量/无证据RAG、表格唯一行定位、身份职责。旧任务回放按模式与意图/类别分层，保留全部11种业务意图。','',
        '## 规模与划分','',
        '第一轮桥接采用520条新修复与520条旧任务回放的1:1配比。新修复共15组表述与证据家族，每组52条状态/任务变体，按10/2/3组分入训练/验证/测试；同句有无待办、同题不同证据、同证据不同输出模式不跨划分。1300条记录不等于1300种独立语义，也不承诺全部业务达标。','',
        '回放仅来自V5/train.jsonl，不读取旧验证/测试答案构造训练标签；旧评测内容仅用于排重。过滤歧义或无法直接核验的表格标签、重复prompt及与历史评测重合的prompt。已修改身份/模板的prompt也与同样规范化的历史评测比较，避免靠改称呼掩盖重合。','',
        '## 示例','']
    for category in ('confirmation_state','quoted_policy','capability_boundary','withdraw_abandon','full_leave_fields','partial','unsupported','table_unique_key','identity_scope'):
        row=next(r for r in splits['train'] if r['origin']=='approved_rule_synthetic_repair' and r['category']==category)
        md += ['### '+row['id'],'','用户：'+user_question(row)]
        if row.get('scenario'):md += ['','```json',json.dumps({k:v for k,v in row['scenario'].items() if k!='user_message'},ensure_ascii=False),'```']
        if row.get('evidence'):md += ['','证据：']+[f'[{i}] '+e['text'] for i,e in enumerate(row['evidence'],1)]
        md += ['','目标：','```text',row['messages'][-1]['content'],'```','']
    md += ['## 运行状态','',
        '数据结构与划分检查通过。编码/Agent实际提示词一致性结果另存encoding_check.json与integration_check.json；没有模型效果评测。现有训练入口尚需支持加载V5 adapter作为起始权重，不能直接用旧命令从Full SFT新建LoRA冒充继续训练。','',
        '## 限制','']+['- '+x for x in manifest['limitations']]
    (output/'review.md').write_bytes('\n'.join(md).encode('utf-8'))
    print(json.dumps({s:counts[s] for s in SPLITS},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=DATA/'v6')
    p.add_argument('--contract',type=Path,required=True)
    args=p.parse_args()
    build(args.output_dir,args.contract)


if __name__=='__main__':main()
