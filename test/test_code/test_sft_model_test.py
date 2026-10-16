"""核对 SFT 聊天前缀、EOS、KV cache 数值一致性和交互循环。"""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoTokenizer

from config.config import TrainPilotconfig
from model.model import TrainPilotForCausalLM

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("trainpilot_sft_entry", Path(__file__).with_name("sft_model_test.py"))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, tokens, past_key_values=None, use_cache=False):
        self.calls.append((tokens.clone(), past_key_values, use_cache, torch.is_grad_enabled()))
        return torch.zeros(*tokens.shape, 1, device=tokens.device), ("cached",)


class PlannedHead(nn.Module):
    def __init__(self, ids, vocab_size, eos_id):
        super().__init__()
        self.ids = iter(ids)
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def forward(self, hidden):
        scores = hidden.new_full((hidden.shape[0], self.vocab_size), -100.0)
        scores[:, next(self.ids, self.eos_id)] = 100.0
        return scores


class ControlledModel(nn.Module):
    def __init__(self, tokenizer, ids, max_seq_len=128):
        super().__init__()
        self.config = SimpleNamespace(max_seq_len=max_seq_len, vocab_size=len(tokenizer))
        self.model = RecordingBackbone()
        self.lm_head = PlannedHead(ids, len(tokenizer), tokenizer.eos_token_id)


class SFTInteractiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "dataset", local_files_only=True)

    def test_chat_prefix_cache_and_eos(self):
        answer_ids = self.tokenizer("你好", add_special_tokens=False).input_ids
        model = ControlledModel(self.tokenizer, answer_ids)
        answer = entry.generate_reply(model, self.tokenizer, "你好", "cpu", temperature=0)
        self.assertEqual(answer, "你好" + self.tokenizer.eos_token)
        prompt_ids = self.tokenizer(
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False,
        ).input_ids
        calls = model.model.calls
        self.assertEqual(calls[0][0].tolist(), [prompt_ids])
        self.assertIn("<|im_start|>assistant\n", self.tokenizer.decode(prompt_ids))
        self.assertNotIn("<think>", self.tokenizer.decode(prompt_ids))
        self.assertIsNone(calls[0][1])
        self.assertEqual(len(calls), len(answer_ids) + 1)
        for tokens, cache, use_cache, grad in calls[1:]:
            self.assertEqual(tokens.shape, (1, 1))
            self.assertEqual(cache, ("cached",))
        self.assertTrue(all(use_cache and not grad for _, _, use_cache, grad in calls))

    def test_greedy_cache_matches_native_full_prefix(self):
        torch.manual_seed(42)
        model = TrainPilotForCausalLM(TrainPilotconfig(
            vocab_size=len(self.tokenizer), hidden_size=16, num_layers=1,
            num_heads=4, num_kv_heads=2, intermediate_size=32, max_seq_len=128,
        )).eval()
        actual = entry.generate_reply(model, self.tokenizer, "一加一是多少？", "cpu", max_new_tokens=5, temperature=0)
        ids = self.tokenizer(
            "<|im_start|>user\n一加一是多少？<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False,
        ).input_ids
        tokens = torch.tensor([ids])
        expected = []
        with torch.inference_mode():
            for _ in range(5):
                scores = model(tokens)[0, -1].clone()
                scores[[self.tokenizer.bos_token_id, self.tokenizer.pad_token_id]] = float("-inf")
                next_id = int(scores.argmax())
                expected.append(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    break
                tokens = torch.cat((tokens, torch.tensor([[next_id]])), dim=1)
        self.assertEqual(actual, self.tokenizer.decode(expected, skip_special_tokens=False))

    def test_context_limit_and_sampling(self):
        answer_ids = self.tokenizer("你好世界", add_special_tokens=False).input_ids
        prompt_length = len(self.tokenizer(
            "<|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False,
        ).input_ids)
        model = ControlledModel(self.tokenizer, answer_ids, max_seq_len=prompt_length + 1)
        reply = entry.generate_reply(model, self.tokenizer, "问题", "cpu", temperature=0.8, top_k=1)
        self.assertEqual(reply, self.tokenizer.decode(answer_ids[:1], skip_special_tokens=True))
        self.assertEqual(len(model.model.calls), 1)
        with self.assertRaisesRegex(ValueError, "缩短问题"):
            entry.generate_reply(ControlledModel(self.tokenizer, [], max_seq_len=1), self.tokenizer, "问题", "cpu")

    def test_input_loop_and_default_checkpoint(self):
        model = ControlledModel(self.tokenizer, self.tokenizer("你好", add_special_tokens=False).input_ids)
        output = io.StringIO()
        with patch.object(entry, "load_base_model", return_value=model) as load, \
             patch("builtins.input", side_effect=["", "你好", "/exit"]) as read, redirect_stdout(output):
            entry.main(["--device", "cpu", "--temperature", "0"])
        self.assertEqual(read.call_count, 3)
        self.assertEqual(load.call_args.args[0], ROOT / "runs/sfttrain1/checkpoint .pt")
        self.assertIn("模型原始输出>\n你好<|im_end|>", output.getvalue())
        self.assertIn("测试结束", output.getvalue())

    def test_tool_schema_and_raw_call_are_preserved(self):
        text = '<think>需要画图。</think>\n<tool_call>\n{"name":"generate_image","arguments":{"prompt":"窗边的猫"}}\n</tool_call>'
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        model = ControlledModel(self.tokenizer, ids, max_seq_len=768)
        printed = io.StringIO()
        with redirect_stdout(printed):
            output = entry.generate_reply(
                model, self.tokenizer, "画一只猫", "cpu", temperature=0,
                tools=entry.IMAGE_TOOLS, thinking=True, show_prompt=True,
            )
        prompt = self.tokenizer.decode(model.model.calls[0][0][0].tolist(), skip_special_tokens=False)
        self.assertIn("# Tools", prompt)
        self.assertIn('"name": "generate_image"', prompt)
        self.assertIn('"required": ["prompt"]', prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertEqual(output, text + self.tokenizer.eos_token)
        self.assertIn(prompt, printed.getvalue())

    def test_tool_mode_does_not_fabricate_a_call(self):
        text = "我不能完成这个请求。"
        model = ControlledModel(self.tokenizer, self.tokenizer(text, add_special_tokens=False).input_ids, max_seq_len=768)
        output = entry.generate_reply(model, self.tokenizer, "画图", "cpu", temperature=0, tools=entry.IMAGE_TOOLS)
        self.assertEqual(output, text + self.tokenizer.eos_token)
        self.assertNotIn("<tool_call>", output)

    def test_available_tools_leave_default_reasoning_and_reply_unforced(self):
        outputs = ["你好！", "<think>先计算。</think>答案是2。",
                   '<tool_call>{"name":"generate_image","arguments":{"prompt":"猫"}}</tool_call>']
        for text in outputs:
            with self.subTest(text=text):
                model = ControlledModel(self.tokenizer, self.tokenizer(text, add_special_tokens=False).input_ids, max_seq_len=768)
                output = entry.generate_reply(model, self.tokenizer, "请求", "cpu", temperature=0, tools=entry.IMAGE_TOOLS)
                prompt = self.tokenizer.decode(model.model.calls[0][0][0].tolist(), skip_special_tokens=False)
                self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
                self.assertEqual(output, text + self.tokenizer.eos_token)

    def test_explicit_no_thinking_is_only_an_optional_comparison_mode(self):
        model = ControlledModel(self.tokenizer, [])
        entry.generate_reply(model, self.tokenizer, "你好", "cpu", temperature=0, thinking=False)
        prompt = self.tokenizer.decode(model.model.calls[0][0][0].tolist(), skip_special_tokens=False)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_stop_details_distinguish_eos_generation_and_context_limits(self):
        ids = self.tokenizer("你好世界", add_special_tokens=False).input_ids
        eos = entry.generate_reply(ControlledModel(self.tokenizer, []), self.tokenizer, "问题", "cpu",
                                   temperature=0, return_details=True)
        self.assertEqual(eos["stop_reason"], "eos")
        self.assertEqual(eos["text"], self.tokenizer.eos_token)
        self.assertEqual(eos["generated_tokens"], 1)
        capped = entry.generate_reply(ControlledModel(self.tokenizer, ids), self.tokenizer, "问题", "cpu",
                                      max_new_tokens=1, temperature=0, return_details=True)
        self.assertEqual(capped["stop_reason"], "max_new_tokens")
        self.assertEqual(capped["generated_tokens"], 1)
        context = entry.generate_reply(
            ControlledModel(self.tokenizer, ids, max_seq_len=eos["input_tokens"] + 1),
            self.tokenizer, "问题", "cpu", temperature=0, return_details=True,
        )
        self.assertEqual(context["stop_reason"], "context_limit")
        self.assertEqual(context["input_tokens"] + context["generated_tokens"], context["context_limit"])


class EvaluationScoringChecks(unittest.TestCase):
    def test_final_answer_does_not_borrow_a_number_from_thinking(self):
        from evaluate_sft import grade
        self.assertFalse(grade('<think>答案：4</think>答案：3', '4')['correct'])
        self.assertFalse(grade('<think>答案：4', '4')['correct'])
        self.assertTrue(grade('<think>先考虑3。</think>答案：4。', '4')['correct'])

    def test_integer_scoring_rejects_decimal_and_missing_answer(self):
        from evaluate_sft import grade
        for text in ('答案：4.2', '答案：42', '答案：4/2', '问题中有4个球，但我不知道答案'):
            self.assertFalse(grade(text, '4')['correct'], text)
        self.assertTrue(grade('答案：-4', '-4')['correct'])
        self.assertTrue(grade('答案：C', 'C')['correct'])
        self.assertTrue(grade('答案：A.湿', 'A')['correct'])

    def test_tool_json_is_not_repaired_and_arguments_are_checked(self):
        from evaluate_sft import tool_grade
        case = dict(required=True, concepts=[['猫', 'cat'], ['雨', 'rain']])
        good = '<tool_call>{"name":"generate_image","arguments":{"prompt":"cat in rain"}}</tool_call>'
        self.assertTrue(tool_grade(good, case)['joint_ok'])
        for text in (good.replace('cat in rain', 'forest'),
                     good.replace('"cat in rain"', '4'),
                     good.replace('"cat in rain"', '"cat in rain",'),
                     good.replace('</tool_call>', '')):
            self.assertFalse(tool_grade(text, case)['joint_ok'], text)
        self.assertTrue(tool_grade('你好', dict(required=False))['joint_ok'])
        self.assertFalse(tool_grade(good, dict(required=False))['joint_ok'])


if __name__ == "__main__":
    unittest.main()
