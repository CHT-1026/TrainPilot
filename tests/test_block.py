"""Block 检查：形状、梯度、因果性与两次残差连接。"""
import unittest

import torch
from torch import nn

from trainpilot.config import TrainPilotconfig
from trainpilot.layers import precompute_freqs_cis
from trainpilot.model import TrainPilotBlock


class ConstantAttention(nn.Module):
    """受控子层：返回常量，便于单独验证 Block 的连接顺序。"""
    def forward(self, x, *args, **kwargs):
        return torch.ones_like(x), None


class BlockChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = TrainPilotconfig(
            hidden_size=16, num_heads=4, num_kv_heads=2,
            intermediate_size=32, max_seq_len=7, dropout=0.0,
        )
        self.positions = precompute_freqs_cis(
            dim=self.config.head_dim, end=7, rope_base=self.config.rope_theta,
        )
        self.x = torch.randn(2, 7, 16)

    def test_shape_and_all_parameter_gradients(self):
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                self.config.flash_attn = sdpa
                block = TrainPilotBlock(layer_id=0, config=self.config)
                x = self.x.clone().requires_grad_()
                output, cache = block(x, self.positions)
                self.assertEqual(output.shape, x.shape)
                self.assertIsNone(cache)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in block.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_future_tokens_do_not_change_past_outputs(self):
        changed = self.x.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                self.config.flash_attn = sdpa
                block = TrainPilotBlock(layer_id=0, config=self.config).eval()
                with torch.no_grad():
                    before, _ = block(self.x, self.positions)
                    after, _ = block(changed, self.positions)
                torch.testing.assert_close(before[:, :4], after[:, :4], rtol=1e-5, atol=1e-6)

    def test_zero_sublayers_preserve_input(self):
        block = TrainPilotBlock(layer_id=0, config=self.config)
        with torch.no_grad():
            block.self_attn.o_proj.weight.zero_()
            block.mlp.down_proj.weight.zero_()
        x = self.x.clone().requires_grad_()
        output, _ = block(x, self.positions)
        torch.testing.assert_close(output, x)
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_second_residual_uses_updated_hidden_states(self):
        block = TrainPilotBlock(layer_id=0, config=self.config)
        block.input_layernorm = nn.Identity()
        block.self_attn = ConstantAttention()
        block.post_attention_layernorm = nn.Identity()
        block.mlp = nn.Identity()
        x = torch.tensor([[[1., 2., 3., 4.]]])
        # 第一段得到 [2,3,4,5]；第二段的输入与残差都应使用这个结果。
        output, cache = block(x, self.positions)
        expected = torch.tensor([[[4., 6., 8., 10.]]])
        torch.testing.assert_close(output, expected)
        self.assertIsNone(cache)


if __name__ == "__main__":
    unittest.main()
