import json
from pathlib import Path
import unittest

from model.dataset.tool_sft import DATA,SandboxTools,episode


class ToolDataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=json.loads((DATA/'contract.json').read_text(encoding='utf-8'))
        cls.schemas={t['function']['name']:t['function']['parameters'] for t in cls.contract['tools']}

    def test_unknown_alias_requires_retry_with_current_id(self):
        row=episode('train',3,'empty_retry',self.contract)
        backend=SandboxTools(json.loads(row['sandbox']),self.schemas)
        results=[json.loads(m['content']) for m in row['messages'] if m['role']=='tool']
        self.assertEqual(results[0]['hits'],[])
        self.assertTrue(results[1]['hits'])
        self.assertIn('error',backend.run('execute',{}))
        self.assertIn('error',backend.run('read',{'doc_id':'none','limit':3000}))

    def test_two_hop_and_pagination_deliver_all_cited_facts(self):
        for family in ['two_hop','pagination','time','conflict']:
            row=episode('test',9,family,self.contract)
            delivered={(s['doc_id'],s['locator']):s['text'] for m in row['messages'] if m['role']=='tool'
                       for s in (json.loads(m['content']).get('hits',[])+json.loads(m['content']).get('segments',[]))}
            final=json.loads(row['messages'][-1]['content'])
            for c in final['citations']:
                self.assertIn(c['quote'],delivered[(c['doc_id'],c['locator'])])

    def test_clarification_needs_no_tool_and_missing_does_not_fabricate(self):
        for family,status in [('clarify','needs_clarification'),('missing','not_in_corpus')]:
            row=episode('val',4,family,self.contract)
            final=json.loads(row['messages'][-1]['content'])
            self.assertEqual(final['status'],status)
            self.assertEqual(final['citations'],[])
        row=episode('val',4,'clarify',self.contract)
        self.assertFalse(any(m.get('tool_calls') for m in row['messages']))

    def test_split_groups_and_document_ids_are_disjoint(self):
        groups,docs={},{}
        for split in ['train','val','test']:
            rows=[json.loads(s) for s in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            groups[split]={r['group_id'] for r in rows}
            docs[split]={d['doc_id'] for r in rows if r['sandbox'] for d in json.loads(r['sandbox'])}
        for a,b in [('train','val'),('train','test'),('val','test')]:
            self.assertFalse(groups[a]&groups[b]);self.assertFalse(docs[a]&docs[b])

    def test_teacher_trajectories_complete_through_execution_loop(self):
        from model.trainer.tool_rollout import run_tool_chat
        from model.dataset.tool_sft import FAMILIES
        for family in FAMILIES:
            row=episode('test',7,family,self.contract)
            assistant=[m for m in row['messages'] if m['role']=='assistant']
            def complete(messages,tools,max_tokens,timeout):
                message=dict(assistant.pop(0))
                if message.get('tool_calls'):
                    calls=json.loads(message['tool_calls'])
                    for c in calls: c['function']['arguments']=json.dumps(c['function']['arguments'],ensure_ascii=False)
                    message['tool_calls']=calls
                return {'choices':[{'message':message,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop'}]}
            result=run_tool_chat(row['messages'][1]['content'],SandboxTools(json.loads(row['sandbox']),self.schemas),complete)
            self.assertEqual(result['status'],'answered',family)
            self.assertEqual(result['final']['status'],json.loads(row['expected'])['status'])


if __name__=='__main__': unittest.main()
