"""评测器的关键行为检查：严格加载、评分边界、聊天结束标记与生成一致性。"""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import datasets
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

PATH = Path(__file__).with_name("evaluate_qwen_sft.py")
spec = importlib.util.spec_from_file_location("qwen_eval_entry", PATH)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class TokenizerFixture:
    eos_token_id = 2
    def convert_tokens_to_ids(self, text):
        return 3  # 模拟 Base EOS 与聊天 im_end 不同。
    def decode(self, ids, **kwargs):
        return str(ids)


class EvalChecks(unittest.TestCase):
    def test_chunked_statistics_match_training_loss_and_ignore_padding(self):
        torch.manual_seed(0)
        scores = torch.randn(2, 90, 17)
        labels = torch.randint(0, 17, (2,90))
        labels[:, :8] = -100
        labels[1, 70:] = -100
        loss, count = entry.next_token_loss(scores, labels)
        total, correct, checked = entry.supervised_stats(scores, labels)
        self.assertEqual(count, checked)
        self.assertAlmostEqual(total/checked, loss.item(), places=6)
        expected = ((scores[:, :-1].argmax(-1)==labels[:, 1:]) & (labels[:, 1:]!=-100)).sum().item()
        self.assertEqual(correct, expected)

    def test_scoring_does_not_accept_wrong_number_or_repair_json(self):
        case = {"kind":"number", "expected":4}
        self.assertTrue(entry.grade(case, "<think>先检查</think>\n答案：4<|im_end|>"))
        self.assertFalse(entry.grade(case, "计算过程出现4，但最后结果是5"))
        self.assertFalse(entry.grade(case, "4或5"))
        self.assertTrue(entry.grade({"kind":"json", "expected":{"x":2}}, '{"x":2}'))
        self.assertFalse(entry.grade({"kind":"json", "expected":{"x":2}}, '```json\n{"x":2}\n```'))
        self.assertFalse(entry.grade({"kind":"json", "expected":{"ok":True}}, '{"ok":1}'))
        self.assertFalse(entry.grade({"kind":"no_tool"}, "你好<tool_call>"))
        tool = {"kind":"tool", "expected":{"name":"get_weather", "arguments":{"city":"北京"}}}
        self.assertTrue(entry.grade(tool, '<tool_call>{"name":"get_weather","arguments":{"city":"北京"}}</tool_call>'))
        self.assertFalse(entry.grade(tool, '<tool_call>{"name":"get_weather","arguments":{"city":"上海"}}</tool_call>'))

    def test_chat_end_stops_generation_even_with_different_base_eos(self):
        class Model(nn.Module):
            config = SimpleNamespace(eos_token_id=2)
            def forward(self, **kwargs):
                return SimpleNamespace(logits=torch.tensor([[[0., 0., 0., 10.]]]), past_key_values=None)
        result = entry.generate(Model(), TokenizerFixture(), [1], "cpu", 8)
        self.assertEqual(result["generated_tokens"], 1)
        self.assertEqual(result["generated_ids"], [3])
        self.assertEqual(result["stop_reason"], "eos")

    def test_checkpoint_load_preserves_tied_weights_and_hf_greedy_generation(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
                             bos_token_id=1, eos_token_id=2, pad_token_id=0)
        original = Qwen3ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.save_pretrained(root)
            path = root/"checkpoint.pt"
            torch.save({"weights":original.state_dict(), "optimizer":{"not_for_inference":True},
                        "progress":{"completed_epochs":1}, "signature":{}}, path)
            loaded, metadata = entry.load_model(path, root, "cpu", torch.float32)
            self.assertEqual(loaded.lm_head.weight.data_ptr(), loaded.model.embed_tokens.weight.data_ptr())
            ids = torch.tensor([[1,5,6,7]])
            with torch.no_grad():
                torch.testing.assert_close(original(ids).logits, loaded(ids).logits)
                hf_ids = loaded.generate(ids, max_new_tokens=5, do_sample=False,
                                         eos_token_id=[2,3], pad_token_id=0)[0,4:].tolist()
            result = entry.generate(loaded, TokenizerFixture(), ids[0].tolist(), "cpu", 5)
            self.assertEqual(result["generated_ids"], hf_ids)
            self.assertEqual(metadata["unique_parameters"], sum(p.numel() for p in original.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 默认设备加载检查需要显卡")
    def test_load_accepts_cuda_without_explicit_index(self):
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
        original = Qwen3ForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.save_pretrained(root)
            path = root/"checkpoint.pt"
            torch.save({"weights":original.state_dict(), "progress":{}, "signature":{}}, path)
            loaded, _ = entry.load_model(path, root, "cuda", torch.bfloat16)
            self.assertTrue(all(p.device.type == "cuda" and p.dtype == torch.bfloat16 for p in loaded.parameters()))
            self.assertFalse(any(b.is_meta for b in loaded.buffers()))


if __name__ == "__main__":
    unittest.main()
