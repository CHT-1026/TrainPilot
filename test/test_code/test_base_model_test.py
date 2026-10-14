"""交互测试入口的加载、停止条件与输入循环检查。"""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from tokenizers import Tokenizer, models, pre_tokenizers

from config.config import TrainPilotconfig
from model.model import TrainPilotForCausalLM
from trainer.trainer import save_checkpoint

# Python 自身可能安装名为 test 的包，按文件路径加载这个交互脚本。
entry_path = Path(__file__).resolve().parents[2] / "test" / "test_code" / "base_model_test.py"
entry_spec = importlib.util.spec_from_file_location("trainpilot_interactive_entry", entry_path)
entry = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(entry)
generate_text = entry.generate_text
load_base_model = entry.load_base_model
main = entry.main


class ControlledModel(nn.Module):
    def __init__(self, max_seq_len=8, end_after_one=True):
        super().__init__()
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self.end_after_one = end_after_one
        self.inputs = []
        self.grad_flags = []

    def forward(self, tokens):
        self.inputs.append(tokens.clone())
        self.grad_flags.append(torch.is_grad_enabled())
        scores = torch.full((*tokens.shape, 7), -10.0)
        next_id = 2 if self.end_after_one and len(self.inputs) > 1 else 5
        scores[..., next_id] = 10.0
        return scores


class InteractiveModelChecks(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(models.WordLevel({
            "<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
            "[UNK]": 3, "hello": 4, "world": 5, "again": 6,
        }, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.device = torch.device("cpu")

    def test_prompt_adds_only_bos_and_eos_stops_generation(self):
        for temperature in (0.0, 0.8):
            with self.subTest(temperature=temperature):
                model = ControlledModel()
                output = generate_text(model, self.tokenizer, "hello", self.device, temperature=temperature, top_k=1)
                self.assertEqual(output, "world")
                self.assertEqual(model.inputs[0].tolist(), [[1, 4]])
                self.assertEqual(len(model.inputs), 2)
                self.assertTrue(all(flag is False for flag in model.grad_flags))

    def test_generation_respects_training_context_limit(self):
        model = ControlledModel(max_seq_len=4, end_after_one=False)
        self.assertEqual(generate_text(model, self.tokenizer, "hello", self.device, temperature=0), "world world")
        self.assertEqual(len(model.inputs), 2)
        with self.assertRaises(ValueError):
            generate_text(model, self.tokenizer, "hello hello hello", self.device)

    def test_rejects_invalid_generation_options(self):
        for options in ({"max_new_tokens": 0}, {"temperature": -1}, {"temperature": float("nan")}, {"top_k": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate_text(ControlledModel(), self.tokenizer, "hello", self.device, **options)

    def test_load_preserves_logits_and_input_loop_can_exit(self):
        torch.manual_seed(42)
        config = TrainPilotconfig(
            vocab_size=7, hidden_size=16, num_layers=1, num_heads=4,
            num_kv_heads=2, intermediate_size=32, max_seq_len=8,
        )
        model = TrainPilotForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            tokenizer_path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(str(tokenizer_path))
            save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 0, 0)
            restored = load_base_model(path, self.device)
            tokens = torch.tensor([[1, 4]])
            with torch.no_grad():
                torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
            output = io.StringIO()
            with patch("builtins.input", side_effect=["hello", "/exit"]) as read, redirect_stdout(output):
                main([
                    "--checkpoint", str(path), "--tokenizer", str(tokenizer_path),
                    "--device", "cpu", "--temperature", "0", "--max-new-tokens", "2",
                ])
            self.assertEqual(read.call_count, 2)
            self.assertIn("续写>", output.getvalue())
            self.assertIn("测试结束", output.getvalue())

    def test_missing_checkpoint_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "下载 checkpoint.pt"):
                load_base_model(Path(directory) / "missing.pt", self.device)
