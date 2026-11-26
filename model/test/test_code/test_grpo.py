"""GRPO 数值与小模型检查；使用随机 tiny Qwen3，不加载或训练项目权重。"""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.trainer.qwen_grpo import (
    activate, balanced_order, completion_logps, generate_one, group_advantages, grpo_loss, save_checkpoint, train,
)


class GRPOMathChecks(unittest.TestCase):
    def test_data_conversion_excludes_answer_and_preserves_splits(self):
        from model.dataset.prepare_grpo_data import prepare
        with TemporaryDirectory() as folder:
            source, output = Path(folder) / "v5", Path(folder) / "rl"
            source.mkdir()
            for split in ("train", "val", "test"):
                row = {"mode": "intent_parse", "id": split, "group_id": split, "category": "chat",
                       "expected": {"intent": "general_chat", "extracted_info": {}},
                       "messages": [{"role": "user", "content": split + "你好"},
                                    {"role": "assistant", "content": "GOLD_MUST_NOT_ENTER_PROMPT"}]}
                (source / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            prepare(source, output)
            for split in ("train", "val", "test"):
                record = json.loads((output / f"{split}.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(record["id"], split)
                self.assertNotIn("GOLD_MUST_NOT_ENTER_PROMPT", json.dumps(record["prompt"]))
                self.assertEqual(record["gold"]["intent"], "general_chat")

    def test_population_group_advantage_and_homogeneous_groups(self):
        self.assertTrue(torch.allclose(group_advantages([1, 1, 0, 0]),
                                       torch.tensor([1., 1., -1., -1.]), atol=3e-4))
        for rewards in ([0, 0, 0, 0], [1, 1, 1, 1]):
            self.assertTrue(torch.equal(group_advantages(rewards), torch.zeros(4)))

    def test_clipping_direction_for_positive_and_negative_advantage(self):
        old = torch.log(torch.tensor([.3, .3]))
        logp = torch.log(torch.tensor([.6, .15])).requires_grad_()
        positive, _ = grpo_loss(logp, old, old, 1., beta=0.)
        negative, _ = grpo_loss(logp, old, old, -1., beta=0.)
        self.assertAlmostEqual(positive.item(), -.85, places=5)
        self.assertAlmostEqual(negative.item(), 1.4, places=5)

    def test_kl_and_old_reference_are_detached(self):
        logp = torch.tensor([-.5, -1.], requires_grad=True)
        old = torch.tensor([-.7, -.8], requires_grad=True)
        ref = torch.tensor([-.6, -.9], requires_grad=True)
        loss, stats = grpo_loss(logp, old, ref, 0.)
        self.assertGreaterEqual(stats["kl"], 0.)
        loss.backward()
        self.assertIsNotNone(logp.grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)
        self.assertAlmostEqual(grpo_loss(logp, logp.detach(), logp.detach(), 0.)[0].item(), 0.)

    def test_every_training_row_is_seen_once_and_categories_interleave(self):
        rows = [{"category": "a"}] * 5 + [{"category": "b"}] * 2 + [{"category": "c"}]
        order = balanced_order(rows, 42)
        self.assertEqual(sorted(order), list(range(8)))
        self.assertEqual({rows[i]["category"] for i in order[:3]}, {"a", "b", "c"})
        self.assertEqual(order, balanced_order(rows, 42))


class TinyPolicyChecks(unittest.TestCase):
    def setUp(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=64, attention_dropout=0.)
        model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0., target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.add_adapter("reference", LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
                          target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        set_peft_model_state_dict(model, get_peft_model_state_dict(model), adapter_name="reference")
        self.model = model
        activate(model)
        self.prompt, self.completions = [3, 5, 8], [[9, 10, 2], [11, 7, 2]]

    def test_completion_alignment_matches_full_forward(self):
        p, c = self.prompt, self.completions[0]
        measured = completion_logps(self.model, p, c, "cpu")
        logits = self.model(torch.tensor([p + c]), use_cache=False).logits[0, len(p)-1:-1].float()
        gold = F.log_softmax(logits, -1).gather(-1, torch.tensor(c).unsqueeze(-1))[:, 0]
        self.assertTrue(torch.allclose(measured, gold, atol=1e-6))

    def test_actual_sampling_and_greedy_paths_respect_completion_limit(self):
        tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0,
            convert_tokens_to_ids=lambda _: 2, decode=lambda tokens, **_: str(tokens))
        args = SimpleNamespace(device="cpu", max_completion_tokens=4)
        row = {"input_ids": self.prompt, "gold": {"intent": "general_chat", "extracted_info": {}}}
        for sample in (True, False):
            result = generate_one(self.model, tokenizer, row, args, sample)
            self.assertGreater(len(result["tokens"]), 0)
            self.assertLessEqual(len(result["tokens"]), 4)
            self.assertEqual(result["reward"], 0.)

    def test_real_backward_changes_policy_but_not_base_or_reference(self):
        model = self.model
        original = {k: v.detach().clone() for k, v in model.named_parameters()}
        with torch.no_grad():
            old = [completion_logps(model, self.prompt, c, "cpu") for c in self.completions]
            activate(model, "reference")
            reference = [completion_logps(model, self.prompt, c, "cpu") for c in self.completions]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, reference)))
        activate(model, training=True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        for i, c in enumerate(self.completions):
            logp = completion_logps(model, self.prompt, c, "cpu")
            loss, _ = grpo_loss(logp, old[i], reference[i], [1., -1.][i])
            (loss / 2).backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        self.assertTrue(torch.isfinite(norm))
        optimizer.step()
        changed = [k for k, p in model.named_parameters() if not torch.equal(original[k], p)]
        self.assertTrue(changed)
        self.assertTrue(all("lora_" in k and ".default." in k for k in changed), changed)
        activate(model, "reference")
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        activate(model)
        self.assertTrue(all(".default." in k for k, p in model.named_parameters() if p.requires_grad))

    def test_checkpoint_export_and_resume_restore_policy_only(self):
        from peft import set_peft_model_state_dict
        model = self.model
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        completion_logps(model, self.prompt, self.completions[0], "cpu").sum().backward()
        optimizer.step()
        expected = completion_logps(model, self.prompt, self.completions[0], "cpu").detach()
        tokenizer = SimpleNamespace(save_pretrained=lambda *_: None)
        with TemporaryDirectory() as folder:
            path = Path(folder)
            save_checkpoint(model, tokenizer, optimizer, path, {"test": True}, {"global_step": 2})
            state = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertTrue((path / "adapter/adapter_model.safetensors").exists())
            self.assertFalse(any("reference" in k for k in state["weights"]))
            with torch.no_grad():
                for p in parameters:
                    p.add_(1.)
            set_peft_model_state_dict(model, state["weights"])
            optimizer.load_state_dict(state["optimizer"])
            self.assertTrue(torch.equal(expected, completion_logps(model, self.prompt, self.completions[0], "cpu")))

    def test_two_update_loop_and_resume_match_continuous_training(self):
        args = SimpleNamespace(device="cpu", seed=42, learning_rate=1e-3, resume=None,
                               max_updates=2, eval_limit=1, updates_per_group=2, group_size=2,
                               save_steps=2, beta=.02, clip=.2)
        rows = [{"id": "one", "category": "a", "input_ids": self.prompt}]
        tokenizer = SimpleNamespace(save_pretrained=lambda *_: None)
        continuous, resumed = copy.deepcopy(self.model), copy.deepcopy(self.model)

        def fake_rollout(model, *_):
            activate(model)
            with torch.no_grad():
                old = [completion_logps(model, self.prompt, c, "cpu") for c in self.completions]
                activate(model, "reference")
                ref = [completion_logps(model, self.prompt, c, "cpu") for c in self.completions]
            activate(model)
            return ([{"tokens": c, "reward": float(i == 0), "ended": True} for i, c in enumerate(self.completions)], old, ref)

        def fake_eval(model, tokenizer, rows, args, output, label):
            (output / f"{label}.json").write_text(json.dumps({"case_ids": ["one"]}), encoding="utf-8")

        with TemporaryDirectory() as folder, \
             patch("model.trainer.qwen_grpo.rollout", side_effect=fake_rollout), \
             patch("model.trainer.qwen_grpo.evaluate", side_effect=fake_eval):
            a, b = Path(folder) / "a", Path(folder) / "b"
            a.mkdir(); b.mkdir()
            train(resumed, tokenizer, rows, rows, args, a, {"test": True})
            args.resume, args.max_updates = str(a / "checkpoint.pt"), 4
            train(resumed, tokenizer, rows, rows, args, a, {"test": True})
            args.resume = None
            train(continuous, tokenizer, rows, rows, args, b, {"test": True})
            self.assertEqual(torch.load(a / "checkpoint.pt", weights_only=True)["progress"]["global_step"], 4)
            for (name, x), (_, y) in zip(resumed.named_parameters(), continuous.named_parameters()):
                self.assertTrue(torch.equal(x, y), name)


if __name__ == "__main__":
    unittest.main()
