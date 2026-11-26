"""微型Qwen集成测试：已有adapter接续、只更新LoRA、导出和中途恢复。"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import datasets
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from model.trainer.qwen_lora_sft import train
from model.trainer.qwen_full_sft import QwenLogitsModel, train_one_epoch
from model.trainer.SFTtrain import train_step


def fixture(root):
    vocab={word:i for i,word in enumerate(['<pad>','<|im_start|>','<|im_end|>','[UNK]',
        'user','assistant','system','input','answer','alpha','beta','gamma','delta'])}
    raw=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    raw.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='<pad>',
        eos_token='<|im_end|>',additional_special_tokens=['<|im_start|>'])
    tokenizer.chat_template=("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' "
        "+ message['content'] + '<|im_end|>\\n' }}{% endfor %}")
    torch.manual_seed(9)
    base=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
        attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0))
    base.save_pretrained(root/'base');tokenizer.save_pretrained(root/'base')
    source=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,
        lora_dropout=.05,target_modules=['q_proj','v_proj']))
    optimizer=torch.optim.AdamW([p for p in source.parameters() if p.requires_grad],lr=.01)
    tokens=torch.tensor([[1,7,9,8,10,2]])
    batch={'input_ids':tokens,'attention_mask':torch.ones_like(tokens),
           'labels':torch.tensor([[-100,-100,-100,8,10,2]])}
    for _ in range(2):train_step(QwenLogitsModel(source),optimizer,batch,'cpu')
    source.save_pretrained(root/'v5');tokenizer.save_pretrained(root/'v5')
    for split,n in [('train',3),('val',1)]:
        records=[{'group_id':split+str(i),'messages':[{'role':'user','content':'input '+['alpha','beta','gamma'][i]},
            {'role':'assistant','content':'answer beta'}]} for i in range(n)]
        (root/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    return tokens


def argv(root,output,*extra):
    return ['--model',str(root/'base'),'--init-adapter',str(root/'v5'),
        '--data-path',str(root/'train.jsonl'),'--validation-data-path',str(root/'val.jsonl'),
        '--output-dir',str(output),'--device','cpu','--dtype','float32','--optimizer','adamw',
        '--epochs','1','--learning-rate','0.001','--batch-size','1','--accumulation-steps','2',
        '--max-length','64','--cpu-threads','2','--log-steps','1','--save-steps','1',
        '--save-training-state','--save-epoch-adapters',*extra]


class LoRAContinuationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_json_config_and_explicit_command_line_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'train.json'
            config.write_text(json.dumps({'epochs':1,'learning_rate':1e-5,'lora_rank':2,'lora_alpha':4}),encoding='utf-8')
            with patch('model.trainer.qwen_lora_sft.run_training') as run:
                train(['--config',str(config),'--learning-rate','0.002'])
            args=run.call_args.args[0]
            self.assertEqual(args.learning_rate,.002)
            self.assertEqual(args.epochs,1)
            self.assertEqual(run.call_args.kwargs['lora_config'].r,2)

    def test_existing_weights_new_optimizer_and_exact_mid_epoch_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokens=fixture(root)
            initial_bytes=(root/'v5/adapter_model.safetensors').read_bytes()
            expected=PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(root/'base'),root/'v5').eval()
            with torch.no_grad():expected_logits=expected(tokens,use_cache=False).logits
            initial_base=Qwen3ForCausalLM.from_pretrained(root/'base').state_dict()
            observed=[]

            def inspect_initial(model,optimizer,*args,**kwargs):
                self.assertEqual(len(optimizer.state),0,'新一轮SFT不得继承V5优化器')
                model.eval()
                with torch.no_grad():actual=model.causal_lm(tokens,use_cache=False).logits
                torch.testing.assert_close(actual,expected_logits,rtol=0,atol=0)
                self.assertTrue(all('lora_' in name for name,p in model.named_parameters() if p.requires_grad))
                observed.append(True)
                return train_one_epoch(model,optimizer,*args,**kwargs)

            with contextlib.redirect_stdout(io.StringIO()):
                with patch('model.trainer.qwen_full_sft.train_one_epoch',side_effect=inspect_initial):
                    train(argv(root,root/'continuous'))
                train(argv(root,root/'resumed','--max-updates','1'))
                partial=torch.load(root/'resumed/checkpoint.pt',map_location='cpu',weights_only=True)
                self.assertEqual(partial['progress']['global_step'],1)
                self.assertEqual(partial['progress']['completed_epochs'],0)
                train(argv(root,root/'resumed','--resume',str(root/'resumed/checkpoint.pt')))
            self.assertEqual(observed,[True])
            first=torch.load(root/'continuous/checkpoint.pt',map_location='cpu',weights_only=True)
            second=torch.load(root/'resumed/checkpoint.pt',map_location='cpu',weights_only=True)
            self.assertEqual(first['progress'],second['progress'])
            self.assertEqual(first['progress']['global_step'],2)
            for name,value in first['weights'].items():
                torch.testing.assert_close(value,second['weights'][name],rtol=0,atol=0)
            self.assertEqual(first['signature']['initial_adapter']['weights_sha256'],hashlib.sha256(initial_bytes).hexdigest())
            self.assertEqual(first['signature']['lora_rank'],2)
            self.assertEqual(first['signature']['target_modules'],['q_proj','v_proj'])
            self.assertEqual(initial_bytes,(root/'v5/adapter_model.safetensors').read_bytes())
            restored=PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(root/'base'),root/'resumed/adapter')
            final=restored.state_dict()
            for name,value in initial_base.items():
                key='base_model.model.'+name
                if any('.'+target+'.' in name for target in ('q_proj','v_proj')):
                    key=key.replace('.weight','.base_layer.weight').replace('.bias','.base_layer.bias')
                torch.testing.assert_close(final[key],value,rtol=0,atol=0)
            from safetensors.torch import load_file
            initial=load_file(str(root/'v5/adapter_model.safetensors'))
            exported=load_file(str(root/'resumed/adapter/adapter_model.safetensors'))
            self.assertTrue(any(not torch.equal(t,exported[k]) for k,t in initial.items()))
            self.assertTrue(all(torch.equal(t,second['weights'][k]) for k,t in exported.items()))

    def test_prepare_only_does_not_load_model_and_tokenizer_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            with patch('model.trainer.qwen_full_sft.AutoModelForCausalLM.from_pretrained',side_effect=AssertionError('不应加载模型')):
                with contextlib.redirect_stdout(io.StringIO()):train(argv(root,root/'prepare','--prepare-only'))
            self.assertFalse((root/'prepare/adapter').exists())
            from transformers import AutoTokenizer
            other=AutoTokenizer.from_pretrained(root/'v5',local_files_only=True)
            other.chat_template=other.chat_template+' '
            other.save_pretrained(root/'v5')
            with self.assertRaisesRegex(ValueError,'词表或聊天模板'):
                train(argv(root,root/'bad'))

    def test_resume_rejects_a_changed_initial_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):train(argv(root,root/'run','--max-updates','1'))
            from safetensors.torch import load_file,save_file
            path=root/'v5/adapter_model.safetensors'
            weights=load_file(str(path));weights[next(iter(weights))].add_(.001)
            save_file(weights,str(path))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'initial_adapter'):
                    train(argv(root,root/'run','--resume',str(root/'run/checkpoint.pt')))


if __name__=='__main__':unittest.main()
