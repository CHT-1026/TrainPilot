"""CPU float32 检查：GQA 分组、因果性、梯度及手写/SDPA 一致性。"""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from trainpilot.config import TrainPilotconfig
from trainpilot.layers import Attention, precompute_freqs_cis, repeat_kv


class AttentionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = TrainPilotconfig(
            hidden_size=16, num_heads=4, num_kv_heads=2,
            intermediate_size=32, max_seq_len=7, dropout=0.0,
        )
        self.positions = precompute_freqs_cis(
            dim=self.config.head_dim, end=7, rope_base=self.config.rope_theta
        )
        self.x = torch.randn(2, 7, 16)

    def make_attention(self, sdpa=False, dropout=0.0):
        config = TrainPilotconfig(
            hidden_size=16, num_heads=4, num_kv_heads=2,
            intermediate_size=32, max_seq_len=7,
            dropout=dropout, flash_attn=sdpa,
        )
        return Attention(config)

    def test_repeat_kv_groups_and_gradients(self):
        x = torch.tensor([[[[10., 11.], [20., 21.]]]], requires_grad=True)
        actual = repeat_kv(x, 2)
        expected = torch.tensor([[[[10., 11.], [10., 11.], [20., 21.], [20., 21.]]]])
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))
        torch.testing.assert_close(repeat_kv(x, 1), x)

    def test_projection_shapes_and_forward_gradients(self):
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                model = self.make_attention(sdpa)
                self.assertEqual(tuple(model.q_proj.weight.shape), (16, 16))
                self.assertEqual(tuple(model.k_proj.weight.shape), (8, 16))
                self.assertEqual(tuple(model.v_proj.weight.shape), (8, 16))
                self.assertEqual(tuple(model.q_norm.weight.shape), (4,))
                self.assertEqual(tuple(model.k_norm.weight.shape), (4,))
                x = self.x.clone().requires_grad_()
                output, cache = model(x, self.positions)
                self.assertEqual(output.shape, x.shape)
                self.assertIsNone(cache)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_future_tokens_do_not_change_past_outputs(self):
        changed = self.x.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                model = self.make_attention(sdpa).eval()
                with torch.no_grad():
                    before, _ = model(self.x, self.positions)
                    after, _ = model(changed, self.positions)
                torch.testing.assert_close(before[:, :4], after[:, :4], rtol=1e-5, atol=1e-6)

    def test_manual_and_sdpa_outputs_and_gradients_match(self):
        manual = self.make_attention(False)
        sdpa = self.make_attention(True)
        sdpa.load_state_dict(manual.state_dict())
        x_manual = self.x.clone().requires_grad_()
        x_sdpa = self.x.clone().requires_grad_()
        manual_output, _ = manual(x_manual, self.positions)
        # 确认调用了 SDPA，避免两个分支都走手写路径而造成假通过。
        with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as call:
            sdpa_output, _ = sdpa(x_sdpa, self.positions)
            call.assert_called_once()
        torch.testing.assert_close(manual_output, sdpa_output, rtol=1e-4, atol=1e-6)
        manual_output.square().mean().backward()
        sdpa_output.square().mean().backward()
        torch.testing.assert_close(x_manual.grad, x_sdpa.grad, rtol=1e-4, atol=1e-6)
        sdpa_parameters = dict(sdpa.named_parameters())
        for name, parameter in manual.named_parameters():
            torch.testing.assert_close(
                parameter.grad, sdpa_parameters[name].grad, rtol=1e-4, atol=1e-6,
            )

    def test_sdpa_dropout_follows_train_eval_mode(self):
        model = self.make_attention(True, dropout=0.3)
        with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as call:
            model.train()
            model(self.x, self.positions)
            self.assertEqual(call.call_args.kwargs["dropout_p"], 0.3)
            model.eval()
            with torch.no_grad():
                first, _ = model(self.x, self.positions)
                self.assertEqual(call.call_args.kwargs["dropout_p"], 0.0)
                second, _ = model(self.x, self.positions)
            torch.testing.assert_close(first, second)

    def test_masked_keys_do_not_change_visible_outputs(self):
        # 1 表示有效位置、0 表示屏蔽位置；这里每行始终有可见 key。
        mask = torch.tensor([[1., 0., 1., 0., 1., 1., 1.]]).repeat(2, 1)
        changed = self.x.clone()
        changed[:, [1, 3]] = torch.randn_like(changed[:, [1, 3]]) * 10
        visible = [0, 2, 4, 5, 6]
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                model = self.make_attention(sdpa).eval()
                with torch.no_grad():
                    before, _ = model(self.x, self.positions, attention_mask=mask)
                    after, _ = model(changed, self.positions, attention_mask=mask)
                torch.testing.assert_close(
                    before[:, visible], after[:, visible], rtol=1e-5, atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
