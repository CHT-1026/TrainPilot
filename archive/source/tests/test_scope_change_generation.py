"""Do not give known revoked/replaced first-pass evidence to the model."""
import unittest
import json
from unittest.mock import patch
from enterprise_rag import pipeline


class ScopeChangeGenerationTest(unittest.TestCase):
    def test_known_acl_or_snapshot_change_drops_chunks_and_graph_before_generation(self):
        old={"chunk_id":"old#0","doc_id":"old","text":"已撤权资料","score":.01}
        def retrieved(chunks,scope):
            return {"chunks":chunks,"debug":{"top_score":.01,"graph":{"paths":["旧权限路径"]},"topic_coverage":{"source_chunk_ids":{"business":["old#0"]}}},
                    "query":{"original":"材料说明","rewritten":"材料说明","matched_synonyms":[],"expansions":[],"time_label":""},
                    "as_of":None,"terms":[],"candidates":len(chunks),"scope":scope}
        for changed in ({"snapshot":"s1","acl_version":"new"},{"snapshot":"s2","acl_version":"old"}):
            with self.subTest(scope=changed):
                captured=[]
                def generate(question,chunks,**kwargs):
                    captured.append((chunks,kwargs['graph_paths']))
                    return {"answer":"无资料","citations":[],"grounded":False,"refused":True,"method":"refusal","confidence":0}
                with patch.object(pipeline,'retrieve',side_effect=[retrieved([old],{"snapshot":"s1","acl_version":"old"}),retrieved([],changed)]), \
                     patch.object(pipeline,'generate_answer',side_effect=generate):
                    result=pipeline.query('材料说明',use_llm=True)
                self.assertEqual(captured,[([],[])])
                self.assertEqual(result['evidence_scope'],changed)
                self.assertEqual(result['retrieval']['chunks'],[])
                self.assertEqual(result['observability']['stopped_reason'],'evidence_scope_changed')
                self.assertNotIn('old#0',json.dumps(result,ensure_ascii=False))
                self.assertNotIn('旧权限路径',json.dumps(result,ensure_ascii=False))
