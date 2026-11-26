"""V6预览的状态方向、部分回答和唯一行定位检查；不加载模型。"""
from pathlib import Path
import json
import unittest

from model.dataset.preview_agent_v6 import build
from model.trainer.agent_rewards import intent_reward
from model.test.test_code.evaluate_agent_v5 import grade


class V6PreviewChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.original, cls.candidate, _ = build()
        cls.by_id = {r['id'].removeprefix('v6-preview-'):r for r in cls.records}

    def test_protocol_and_rag_target_structure(self):
        self.assertEqual(len(self.records),26)
        self.assertEqual(len({r['id'] for r in self.records}),26)
        for r in self.records:
            target=r['messages'][-1]['content']
            if r['mode']=='intent_parse':
                self.assertEqual(intent_reward(target,r['expected'])['reward'],1)
            elif r['mode'].endswith('_rag'):
                self.assertTrue(grade(r,target)['passed'],r['id'])

    def test_same_user_different_trusted_state(self):
        for action in ('confirm','cancel'):
            selected=[self.by_id[action+'-'+s] for s in ('none','leave','approval','withdrawal')]
            self.assertEqual(len({r['scenario']['user_message'] for r in selected}),1)
            self.assertEqual(selected[0]['expected']['intent'],'clarify')
            self.assertIsNone(selected[0]['scenario']['pending_action'])
            for r in selected[1:]:
                self.assertIsNotNone(r['scenario']['pending_action'])
                self.assertEqual(r['expected']['intent'],action+'_action')
                self.assertEqual(r['expected']['extracted_info'],{})
            self.assertEqual(len({r['group_id'] for r in selected}),1)

    def test_withdrawal_direction_new_object_and_full_fields(self):
        self.assertEqual(self.by_id['abandon-withdrawal']['expected']['intent'],'cancel_action')
        self.assertEqual(self.by_id['execute-withdrawal']['expected']['intent'],'confirm_action')
        other=self.by_id['actual-withdraw']
        self.assertNotEqual(other['scenario']['pending_action']['draft']['app_id'],other['expected']['extracted_info']['app_id'])
        leave=self.by_id['edit-leave']['expected']['extracted_info']
        self.assertEqual(set(leave),{'leave_type','start_date','end_date','reason'})
        self.assertEqual(leave['end_date'],'2027-03-12')
        approval=self.by_id['edit-approval']['expected']['extracted_info']
        self.assertEqual(approval['decision'],'驳回')
        self.assertEqual(approval['comments'],'预算口径尚未统一，请保留原始明细。')

    def test_partial_answer_vs_no_evidence(self):
        for mode in ('enterprise','oa'):
            partial=self.by_id[mode+'-partial'];none=self.by_id[mode+'-unsupported']
            self.assertEqual(partial['question'],none['question'])
            self.assertEqual(partial['group_id'],none['group_id'])
            self.assertTrue(partial['expected_claims'])
            self.assertIn('未提供采购价格',partial['expected_claims'][0]['text'])
            self.assertEqual(none['expected_claims'],[])
            self.assertNotEqual(partial['evidence'],none['evidence'])
        self.assertIn('资料仅覆盖部分问题时',self.candidate['oa_template'])
        self.assertNotIn('资料仅覆盖部分问题时',self.original['oa_template'])
        for key in ('enterprise_system','oa_template'):
            self.assertIn('TrainPilot',self.candidate[key].splitlines()[0])

    def test_ambiguous_table_not_given_arbitrary_answer(self):
        ambiguous=self.by_id['table-ambiguous'];specific=self.by_id['table-specific']
        self.assertEqual(ambiguous['expected_claims'],[])
        self.assertEqual(json.loads(ambiguous['messages'][-1]['content']),{'refused':True,'claims':[]})
        self.assertIn('信息更正',specific['question'])
        self.assertIn('5个工作日',specific['expected_claims'][0]['text'])
        self.assertEqual(ambiguous['group_id'],specific['group_id'])


if __name__=='__main__':unittest.main()
