import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from model.test.test_code.evaluate_reward_ab import decision


class DecisionChecks(unittest.TestCase):
    def fixture(self):
        def record(suite,good):
            return {"suite":suite,"case":{"mode":"intent_parse"},"passed":good,
                    "business_diagnostics":{"severe_error":False,"format_ok":True}}
        baseline=[record('development',False),record('old_regression',True)]
        results={'v6':baseline}
        for seed in (42,43):
            results[f'strict_{seed}']=copy.deepcopy(baseline)
            results[f'graded_{seed}']=[record('development',True),record('old_regression',True)]
        return results,{k:True for k in results if k!='v6'}

    def test_reward_gain_alone_cannot_pass(self):
        results,finished=self.fixture()
        results['graded_42'][0]['passed']=False
        self.assertEqual(decision(results,finished)['recommendation'],'keep_strict_reward')

    def test_severe_error_or_incomplete_budget_rejects(self):
        results,finished=self.fixture()
        self.assertEqual(decision(results,finished)['recommendation'],'business_candidate_pending_manual_rag_review')
        finished['graded_43']=False
        self.assertEqual(decision(results,finished)['recommendation'],'keep_strict_reward')
        finished['graded_43']=True
        results['graded_42'][0]['business_diagnostics']['severe_error']=True
        self.assertEqual(decision(results,finished)['recommendation'],'keep_strict_reward')


class TinyTrialChecks(unittest.TestCase):
    def test_real_two_updates_preserve_reference_and_export(self):
        import torch
        from peft import LoraConfig,get_peft_model
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from model.trainer import qwen_grpo as core
        from model.trainer import grpo_reward_ab as ab
        torch.set_num_threads(1);torch.manual_seed(7)
        cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64)
        base=Qwen3ForCausalLM(cfg)
        lora=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
        model=get_peft_model(base,lora);model.add_adapter('reference',lora)
        # Both B matrices start at zero, so the initial two adapters have identical outputs.
        original={k:p.detach().clone() for k,p in model.named_parameters()}
        tokenizer=SimpleNamespace(encode=lambda *a,**k:[4,5],save_pretrained=lambda *a:None)
        gold={"intent":"approve","extracted_info":{"app_id":"leave_A","decision":"通过","comments":"已核对"}}
        def rollout(model,*unused):
            core.activate(model)
            completions=[[4,5],[6,7]]
            with torch.no_grad():
                old=[core.completion_logps(model,[1,2],c,'cpu') for c in completions]
                core.activate(model,'reference')
                ref=[core.completion_logps(model,[1,2],c,'cpu') for c in completions]
            core.activate(model)
            texts=[json.dumps(gold),json.dumps({'intent':'approve','extracted_info':{'app_id':'leave_A','decision':'通过'}})]
            return ([{'text':t,'tokens':c,'ended':True,'reward':float(i==0)} for i,(t,c) in enumerate(zip(texts,completions))],old,ref)
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'model/config').mkdir(parents=True);data=root/'data';data.mkdir()
            config={'group_size':2,'updates_per_group':2,'max_updates':2,'max_sampled_groups':4,
                'learning_rate':1e-3,'beta':.02,'clip':.2,'max_prompt_tokens':64,'max_completion_tokens':8,
                'initial_adapter':'init','initial_adapter_sha256':'stub','base_config_sha256':'stub'}
            (root/'model/config/grpo_reward_ab.json').write_text(json.dumps(config))
            (root/'model/config/agent_diagnosis.json').write_text(json.dumps({'base_weight_sha256':{}}))
            (data/'manifest.json').write_text(json.dumps({'sha256':{}}))
            params=SimpleNamespace(model=root/'model',output_dir=root/'result',method='graded',seed=42,smoke=False,device='cpu')
            encoded=[{'id':'one','category':'approval','input_ids':[1,2],'gold':gold}]
            with patch.object(ab,'ROOT',root),patch.object(ab,'DATA',data),patch.object(core,'digest',return_value='stub'),\
                 patch.object(core,'load_policy',return_value=(model,tokenizer)),\
                 patch.object(core,'read_rows',return_value=[]),patch.object(core,'encode_rows',return_value=encoded),\
                 patch.object(core,'rollout',side_effect=rollout),patch.object(ab,'validate'):
                ab.trial(params)
            done=json.loads((params.output_dir/'completed.json').read_text(encoding='utf-8'))
            self.assertTrue(done['finished']);self.assertEqual(done['global_step'],2)
            self.assertTrue((params.output_dir/'adapter/adapter_model.safetensors').exists())
            changed=[k for k,p in model.named_parameters() if not torch.equal(p,original[k])]
            self.assertTrue(changed)
            self.assertTrue(all('.default.' in k and 'lora_' in k for k in changed),changed)


if __name__=='__main__':unittest.main()
