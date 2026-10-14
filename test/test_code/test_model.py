"""骨干模型检查：独立层、RoPE buffer、梯度、因果性与权重重载。"""
import io
import math
import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F

from config.config import TrainPilotconfig
from model.model import TrainPilotModel, TrainPilotForCausalLM


class BackboneChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = TrainPilotconfig(
            vocab_size=32, hidden_size=16, num_layers=2,
            num_heads=4, num_kv_heads=2, intermediate_size=32,
            max_seq_len=7, dropout=0.0,
        )
        self.tokens = torch.randint(0, 32, (2, 7))

    def test_independent_blocks_and_rotary_buffers(self):
        model = TrainPilotModel(self.config)
        self.assertEqual(len(model.layers), 2)
        self.assertIsNot(model.layers[0], model.layers[1])
        self.assertIsNot(model.layers[0].self_attn.q_proj.weight,
                         model.layers[1].self_attn.q_proj.weight)
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        for name in ("freqs_cos", "freqs_sin"):
            self.assertIn(name, buffers)
            self.assertNotIn(name, parameters)
            self.assertEqual(tuple(buffers[name].shape), (7, 4))
            self.assertFalse(buffers[name].requires_grad)
            self.assertNotIn(name, model.state_dict())
        # 不分配 GPU 内存，检查 .to() 是否也迁移了非持久化 buffer。
        model.to("meta")
        self.assertEqual(model.freqs_cos.device.type, "meta")
        self.assertEqual(model.freqs_sin.device.type, "meta")

    def test_hidden_shape_and_all_parameter_gradients(self):
        model = TrainPilotModel(self.config)
        hidden, cache = model(self.tokens)
        self.assertEqual(tuple(hidden.shape), (2, 7, 16))
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertEqual(len(cache), 2)
        self.assertTrue(all(item is None for item in cache))
        hidden.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_future_tokens_do_not_change_past_hidden_states(self):
        changed = self.tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % self.config.vocab_size
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                self.config.flash_attn = sdpa
                model = TrainPilotModel(self.config).eval()
                with torch.no_grad():
                    before, _ = model(self.tokens)
                    after, _ = model(changed)
                torch.testing.assert_close(before[:, :4], after[:, :4], rtol=1e-5, atol=1e-6)

    def test_weight_reload_preserves_hidden_states(self):
        model = TrainPilotModel(self.config).eval()
        restored = TrainPilotModel(self.config).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            before, _ = model(self.tokens)
            after, _ = restored(self.tokens)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_rejects_sequence_beyond_rotary_limit(self):
        model = TrainPilotModel(self.config)
        too_long = torch.zeros(1, 8, dtype=torch.long)
        with self.assertRaises(ValueError):
            model(too_long)

    def test_rotary_limit_can_exceed_training_sample_length(self):
        model = TrainPilotModel(replace(self.config, max_position_embeddings=10)).eval()
        self.assertEqual(model.freqs_cos.shape[0], 10)
        with torch.no_grad():
            hidden, _ = model(torch.zeros(1, 9, dtype=torch.long))
        self.assertEqual(hidden.shape, (1, 9, 16))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 11, dtype=torch.long))


class LanguageModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = TrainPilotconfig(
            vocab_size=32, hidden_size=16, num_layers=2,
            num_heads=4, num_kv_heads=2, intermediate_size=32,
            max_seq_len=7, dropout=0.0,
        )
        self.tokens = torch.randint(0, 32, (2, 7))

    def test_initialization_scale_keeps_initial_loss_near_uniform_prediction(self):
        config = replace(self.config, vocab_size=6400, hidden_size=64, intermediate_size=128)
        model = TrainPilotForCausalLM(config).eval()
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertAlmostEqual(model.model.embed_tokens.weight.std().item(), 0.02, delta=0.001)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertAlmostEqual(module.weight.std().item(), 0.02, delta=0.002)
        tokens = torch.tensor([[1, 30, 50, 70, 2]])
        with torch.no_grad():
            logits = model(tokens)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 6400), tokens[:, 1:].reshape(-1))
        self.assertLess(abs(loss.item() - math.log(6400)), 0.5)

    def test_logits_and_next_token_loss_gradients(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                model = TrainPilotForCausalLM(replace(self.config, tie_word_embeddings=tied))
                logits = model(self.tokens)
                self.assertEqual(tuple(logits.shape), (2, 7, 32))
                self.assertTrue(torch.isfinite(logits).all())
                labels = self.tokens.clone()
                labels[:, -2:] = -100
                # 只在测试端计算训练目标，模型 forward 保留原始 logits。
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 32),
                    labels[:, 1:].reshape(-1), ignore_index=-100,
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_weight_sharing_and_unique_parameter_count(self):
        tied = TrainPilotForCausalLM(replace(self.config, tie_word_embeddings=True))
        untied = TrainPilotForCausalLM(replace(self.config, tie_word_embeddings=False))
        self.assertIs(tied.lm_head.weight, tied.model.embed_tokens.weight)
        self.assertIsNot(untied.lm_head.weight, untied.model.embed_tokens.weight)
        self.assertIsNone(tied.lm_head.bias)
        tied_parameters = list(tied.parameters())
        shared = tied.model.embed_tokens.weight
        self.assertEqual(sum(parameter is shared for parameter in tied_parameters), 1)
        difference = sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied_parameters)
        self.assertEqual(difference, self.config.vocab_size * self.config.hidden_size)

    def test_logits_are_causal(self):
        changed = self.tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % self.config.vocab_size
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                model = TrainPilotForCausalLM(replace(self.config, flash_attn=sdpa)).eval()
                with torch.no_grad():
                    before = model(self.tokens)
                    after = model(changed)
                torch.testing.assert_close(before[:, :4], after[:, :4], rtol=1e-5, atol=1e-6)

    def test_serialized_weights_preserve_logits_and_sharing(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                config = replace(self.config, tie_word_embeddings=tied)
                original = TrainPilotForCausalLM(config).eval()
                restored = TrainPilotForCausalLM(config).eval()
                # 使用内存文件核对实际序列化，不在仓库留下 checkpoint。
                checkpoint = io.BytesIO()
                torch.save(original.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
                self.assertEqual(
                    restored.lm_head.weight is restored.model.embed_tokens.weight, tied,
                )
                with torch.no_grad():
                    before = original(self.tokens)
                    after = restored(self.tokens)
                torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
