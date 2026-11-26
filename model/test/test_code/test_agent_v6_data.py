"""V6数据的状态语义、证据边界、回放来源与分组检查。"""
from collections import Counter
import json
from pathlib import Path
import unittest

from model.dataset.build_agent_v6 import DATA, INITIAL_ADAPTER_SHA, read_rows, sha, validate
from model.test.test_code.evaluate_grpo_regression import grade

OUT=DATA/'v6'


class V6DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        cls.contract=json.loads((OUT/'agent_contract.json').read_text(encoding='utf-8'))
        cls.splits={s:read_rows(OUT/(s+'.jsonl')) for s in ('train','val','test')}
        cls.all=[r for part in cls.splits.values() for r in part]
        cls.new=[r for r in cls.all if r['origin']=='approved_rule_synthetic_repair']

    def test_counts_hashes_and_initialization(self):
        self.assertEqual([len(self.splits[s]) for s in ('train','val','test')],[1040,104,156])
        for s in self.splits:self.assertEqual(sha(OUT/(s+'.jsonl')),self.manifest['outputs'][s]['sha256'])
        self.assertEqual(self.manifest['initial_adapter_sha256'],INITIAL_ADAPTER_SHA)
        self.assertTrue(self.manifest['new_optimizer'])
        self.assertFalse(self.manifest['resume_old_training_state'])
        self.assertEqual(sum(r['origin'].startswith('v5_train_replay') for r in self.all),520)

    def test_historical_eval_and_new_group_isolation(self):
        validate(self.splits,self.contract)
        owner={};docs={}
        for split,records in self.splits.items():
            for r in records:
                self.assertEqual(owner.setdefault(r['group_id'],split),split)
                for e in r.get('evidence',[]):
                    self.assertEqual(docs.setdefault(e['doc_id'],split),split)
        self.assertEqual(len({r['group_id'] for r in self.new}),15)

    def test_confirm_cancel_uses_trusted_pending_state(self):
        for category,action in (('confirmation_state','confirm_action'),('cancellation_state','cancel_action')):
            selected=[r for r in self.new if r['category']==category]
            self.assertEqual(len(selected),120)
            self.assertEqual(Counter(bool(r['scenario']['pending_action']) for r in selected),{False:60,True:60})
            for r in selected:
                self.assertEqual(r['expected']['intent'],action if r['scenario']['pending_action'] else 'clarify')
                self.assertEqual(r['expected']['extracted_info'],{})

    def test_partial_full_and_unsupported_are_not_always_refused(self):
        selected=[r for r in self.new if r['category'] in ('partial','full_evidence','unsupported')]
        self.assertEqual(len(selected),180)
        for r in selected:
            claims=r['expected_claims']
            if r['category']=='partial':
                self.assertEqual(len(claims),2)
                self.assertIn('当前资料未提供',claims[1]['text'])
                self.assertNotIn('未提供',claims[0]['text'])
                self.assertEqual(claims[0]['sources'],claims[1]['sources'])
            elif r['category']=='full_evidence':
                self.assertEqual(len(claims),2)
                self.assertTrue(all('未提供' not in c['text'] for c in claims))
            else:self.assertEqual(claims,[])

    def test_new_task_never_inherits_old_fields(self):
        for r in self.new:
            if r['category']!='new_task_no_inheritance':continue
            info=r['expected']['extracted_info']
            if r['expected']['intent']=='submit_leave':
                self.assertEqual(set(info),{'leave_type','reason'})
            else:
                self.assertEqual(set(info),{'app_id','decision'})
                self.assertNotEqual(info['app_id'],r['scenario']['pending_action']['draft']['app_id'])

    def test_replay_preserves_original_targets_and_evidence(self):
        original={r['id']:r for r in read_rows(DATA/'v5/train.jsonl')}
        counts=Counter()
        for r in self.splits['train']:
            if not r['origin'].startswith('v5_train_replay'):continue
            self.assertEqual(r['source_file'],'v5/train.jsonl')
            old=original[r['source_id']]
            if r['mode']=='intent_parse':
                self.assertEqual(r['expected'],old['expected'])
                counts[r['expected']['intent']]+=1
            elif r['mode'].endswith('_rag'):
                self.assertEqual(r['expected_claims'],old['expected_claims'])
                self.assertEqual(r['evidence'],old['evidence'])
            else:self.assertEqual(r['messages'][-1]['content'],old['messages'][-1]['content'])
        self.assertEqual(len(counts),11)
        self.assertTrue(all(25<=n<=26 for n in counts.values()))

    def test_all_reference_outputs_obey_protocol(self):
        count=0
        for r in self.all:
            if r['mode'] in ('intent_parse','enterprise_rag','oa_rag'):
                self.assertTrue(grade(r,r['messages'][-1]['content'])['passed'],r['id'])
                count+=1
        self.assertEqual(count,1225)


if __name__=='__main__':unittest.main()
