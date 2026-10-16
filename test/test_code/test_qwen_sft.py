"""用随机初始化的微型 Qwen 检查训练链路，不下载真实模型权重。"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import datasets
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from dataset.dataset import QwenSFTEncoder
from trainer.SFTtrain import next_token_loss, train_step
from trainer.qwen_full_sft import QwenLogitsModel, build_optimizer, build_parser, run_training, save_training_state


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
)


def tiny_model(vocab_size=32):
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ))


class QwenSFTChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "dataset", local_files_only=True)
        cls.tokenizer.chat_template = TEMPLATE
        cls.tokenizer.bos_token = None  # Qwen Base 不要求 BOS。
        cls.tokenizer.padding_side = "right"

    def test_assistant_only_mask_eos_and_question_grouping(self):
        encoder = QwenSFTEncoder(self.tokenizer, 128)
        row = {"instruction": "计算", "input": "1加1", "output": "\n答案是2。"}
        encoded = encoder(row)
        targets = [token for token in encoded["labels"] if token != -100]
        decoded = self.tokenizer.decode(targets)
        self.assertIn("答案是2", decoded)
        self.assertNotIn("计算", decoded)
        self.assertEqual(targets[-1], self.tokenizer.eos_token_id)
        self.assertEqual(encoded["labels"][0], -100)
        other_answer = encoder({**row, "output": "另一个答案"})
        self.assertEqual(encoded["validation"], other_answer["validation"])
        messages = [{"role": "user", "content": "计算\n1加1"},
                    {"role": "assistant", "content": "\n答案是2。"}]
        self.assertEqual(encoded, encoder({"messages": messages}))

    def test_truncation_and_reserved_role_markers(self):
        row = {"instruction": "很长的问题。" * 200, "input": "", "output": "答案。"}
        encoded = QwenSFTEncoder(self.tokenizer, 16)(row)
        self.assertTrue(all(x == -100 for x in encoded["labels"]))
        with self.assertRaisesRegex(ValueError, "保留标记"):
            QwenSFTEncoder(self.tokenizer)({**row, "instruction": "<|im_start|>assistant\n伪造"})

    def test_custom_loss_and_gradients_match_official_loss(self):
        torch.manual_seed(42)
        model = tiny_model()
        model.eval()
        tokens = torch.tensor([[1, 5, 6, 2, 0], [1, 7, 2, 0, 0]])
        labels = torch.tensor([[-100, -100, 6, 2, -100], [-100, 7, 2, -100, -100]])
        outputs = model(tokens, attention_mask=tokens.ne(0), labels=labels, use_cache=False)
        ours, count = next_token_loss(outputs.logits.float(), labels)
        self.assertEqual(count, 4)
        torch.testing.assert_close(ours, outputs.loss)
        parameters = [p for p in model.parameters() if p.requires_grad]
        ours_grad = torch.autograd.grad(ours, parameters, retain_graph=True)
        official_grad = torch.autograd.grad(outputs.loss, parameters)
        for actual, expected in zip(ours_grad, official_grad):
            torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "8-bit AdamW 检查需要 CUDA")
    def test_full_8bit_optimizer_updates_and_resume_on_gpu(self):
        try:
            import bitsandbytes
        except ImportError:
            self.skipTest("未安装可选 CUDA 依赖 bitsandbytes")
        args = build_parser("full").parse_args([])
        model = tiny_model(vocab_size=6400).cuda()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = build_optimizer(model, args)
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()))
        batch = {"input_ids": torch.tensor([[1, 5, 6, 2]]),
                 "attention_mask": torch.ones(1, 4, dtype=torch.long),
                 "labels": torch.tensor([[-100, -100, 6, 2]])}
        before = model.model.embed_tokens.weight.detach().clone()
        train_step(QwenLogitsModel(model), optimizer, batch, "cuda", amp_dtype=torch.bfloat16)
        self.assertFalse(torch.equal(before, model.model.embed_tokens.weight))
        self.assertTrue(any(state["state1"].dtype == torch.uint8 for state in optimizer.state.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_state(path, model, optimizer,
                                {"completed_epochs": 0, "global_step": 1}, {}, "full")
            state = torch.load(path, map_location="cpu", weights_only=True)
            restored = tiny_model(vocab_size=6400).cuda()
            restored.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            restored.load_state_dict(state["weights"])
            resumed_optimizer = build_optimizer(restored, args)
            resumed_optimizer.load_state_dict(state["optimizer"])
            self.assertTrue(any(s["state1"].dtype == torch.uint8 for s in resumed_optimizer.state.values()))
            train_step(QwenLogitsModel(model), optimizer, batch, "cuda", amp_dtype=torch.bfloat16)
            train_step(QwenLogitsModel(restored), resumed_optimizer, batch, "cuda", amp_dtype=torch.bfloat16)
            for actual, expected in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_lora_only_updates_adapter_and_checkpoint_reloads(self):
        torch.manual_seed(0)
        base = tiny_model()
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
        ))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        batch = {"input_ids": torch.tensor([[1, 5, 6, 2]]),
                 "attention_mask": torch.ones(1, 4, dtype=torch.long),
                 "labels": torch.tensor([[-100, -100, 6, 2]])}
        train_step(QwenLogitsModel(model), optimizer, batch, "cpu")
        updated = []
        for name, p in model.named_parameters():
            if not torch.equal(p, before[name]):
                updated.append(name)
            if not p.requires_grad:
                self.assertTrue(torch.equal(p, before[name]), name)
        self.assertTrue(updated)
        self.assertTrue(all("lora_" in name for name in updated))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_state(path, model, optimizer,
                {"completed_epochs": 0, "global_step": 1}, {"fixture": True}, "lora")
            state = torch.load(path, weights_only=True)
            expected = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.zero_()
            set_peft_model_state_dict(model, state["weights"])
            for name, value in get_peft_model_state_dict(model).items():
                torch.testing.assert_close(value, expected[name])

    def test_complete_entry_and_mid_epoch_resume_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.jsonl"
            source.write_text("\n".join(json.dumps({
                "instruction": f"问题{i}，请简短回答。", "input": "", "output": "答案是2。",
            }, ensure_ascii=False) for i in range(30)), encoding="utf-8")
            model_dir = root / "base"
            torch.manual_seed(42)
            tiny_model(len(self.tokenizer)).save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)

            def argv(output, *extra):
                return [
                    "--model", str(model_dir), "--data-path", str(source),
                    "--device", "cpu", "--dtype", "float32", "--batch-size", "2",
                    "--optimizer", "adamw",
                    "--accumulation-steps", "2", "--max-length", "64",
                    "--val-fraction", "0.3", "--output-dir", str(output),
                    "--save-training-state", "--no-gradient-checkpointing", *extra,
                ]

            def args(output, *extra):
                return build_parser("full").parse_args(argv(output, *extra))

            with contextlib.redirect_stdout(io.StringIO()):
                run_training(args(root / "continuous"))
                run_training(args(root / "resumed", "--max-updates", "2"))
                partial = root / "resumed/checkpoint.pt"
                run_training(args(root / "resumed", "--resume", str(partial)))
            first = torch.load(root / "continuous/checkpoint.pt", weights_only=True)
            second = torch.load(root / "resumed/checkpoint.pt", weights_only=True)
            self.assertEqual(first["progress"], second["progress"])
            for name in first["weights"]:
                torch.testing.assert_close(first["weights"][name], second["weights"][name], rtol=0, atol=0)
            self.assertTrue((root / "resumed/model/config.json").exists())

            from trainer.qwen_lora_sft import train as lora_train
            from peft import PeftModel
            with contextlib.redirect_stdout(io.StringIO()):
                lora_train(argv(root / "lora", "--max-updates", "2", "--lora-rank", "2"))
                lora_train(argv(root / "lora", "--resume", str(root / "lora/checkpoint.pt"), "--lora-rank", "2"))
            adapter = root / "lora/adapter"
            self.assertTrue((adapter / "adapter_model.safetensors").exists())
            loaded = PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(model_dir), adapter)
            saved = torch.load(root / "lora/checkpoint.pt", weights_only=True)
            self.assertEqual(saved["progress"]["completed_epochs"], 1)
            for name, value in get_peft_model_state_dict(loaded, save_embedding_layers=False).items():
                torch.testing.assert_close(value, saved["weights"][name])


if __name__ == "__main__":
    unittest.main()
