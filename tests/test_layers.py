"""固定数值与梯度检查；先完成 layers.py 再运行。"""
import unittest
import torch
from trainpilot.config import TrainPilotconfig
from trainpilot.layers import (
    RMSNorm, FFN_SWiGLU, precompute_freqs_cis, apply_rotary_pos_emb,
)


class LayerChecks(unittest.TestCase):
    def test_rmsnorm_known_values_and_gradient(self):
        norm = RMSNorm(2, eps=1e-6)
        x = torch.tensor([[3.0, 4.0]], requires_grad=True)
        actual = norm(x)
        expected = torch.tensor([[0.8485281, 1.1313708]])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(norm.weight.grad)

    def test_rmsnorm_zero_and_head_dimensions(self):
        norm = RMSNorm(8)
        for shape in ((2, 3, 8), (2, 3, 4, 8)):
            x = torch.zeros(shape)
            actual = norm(x)
            self.assertEqual(actual.shape, x.shape)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, x)

    def test_swiglu_controlled_weights(self):
        config = TrainPilotconfig(
            hidden_size=2, num_heads=1, num_kv_heads=1, intermediate_size=2
        )
        layer = FFN_SWiGLU(config)
        with torch.no_grad():
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(layer, name)
                self.assertIsNone(projection.bias)
                projection.weight.copy_(torch.eye(2))
        x = torch.tensor([[[-1.0, 2.0]]], requires_grad=True)
        actual = layer(x)
        expected = torch.tensor([[[0.26894143, 3.52318831]]])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in layer.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_swiglu_expansion_and_override(self):
        # 不同的输入和中间维度可以发现 down_proj 的方向错误。
        config = TrainPilotconfig(
            hidden_size=8, num_heads=2, num_kv_heads=1, intermediate_size=12
        )
        for override, expected_size in ((None, 12), (16, 16)):
            with self.subTest(intermediate_size=override):
                layer = FFN_SWiGLU(config, intermediate_size=override)
                self.assertEqual(tuple(layer.gate_proj.weight.shape), (expected_size, 8))
                self.assertEqual(tuple(layer.up_proj.weight.shape), (expected_size, 8))
                self.assertEqual(tuple(layer.down_proj.weight.shape), (8, expected_size))
                x = torch.randn(2, 3, 8, requires_grad=True)
                actual = layer(x)
                self.assertEqual(actual.shape, x.shape)
                self.assertTrue(torch.isfinite(actual).all())
                actual.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in layer.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_rope_precompute_shape_and_values(self):
        cos, sin = precompute_freqs_cis(dim=4, end=7, rope_base=100)
        self.assertEqual(tuple(cos.shape), (7, 4))
        self.assertEqual(tuple(sin.shape), (7, 4))
        torch.testing.assert_close(cos[0], torch.ones(4))
        torch.testing.assert_close(sin[0], torch.zeros(4))
        # 位置 1 的两组角度为 1 和 0.1 弧度，采用前后半维配对。
        expected_cos = torch.tensor([0.54030231, 0.99500417, 0.54030231, 0.99500417])
        expected_sin = torch.tensor([0.84147098, 0.09983342, 0.84147098, 0.09983342])
        torch.testing.assert_close(cos[1], expected_cos, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(sin[1], expected_sin, rtol=1e-5, atol=1e-6)

    def test_rope_rotation_broadcast_and_gradients(self):
        # 独立构造 90 度旋转，不依赖预计算函数；Q/K 头数不同。
        q = torch.tensor([1., 2., 3., 4.]).repeat(2, 7, 4, 1).requires_grad_()
        k = torch.tensor([5., 6., 7., 8.]).repeat(2, 7, 2, 1).requires_grad_()
        cos, sin = torch.zeros(7, 4), torch.ones(7, 4)
        rotated_q, rotated_k = apply_rotary_pos_emb(q, k, cos, sin)
        expected_q = torch.tensor([-3., -4., 1., 2.]).repeat(2, 7, 4, 1)
        expected_k = torch.tensor([-7., -8., 5., 6.]).repeat(2, 7, 2, 1)
        torch.testing.assert_close(rotated_q, expected_q)
        torch.testing.assert_close(rotated_k, expected_k)
        torch.testing.assert_close(rotated_q.square().sum(-1), q.square().sum(-1))
        torch.testing.assert_close(rotated_k.square().sum(-1), k.square().sum(-1))
        (rotated_q.square().mean() + rotated_k.square().mean()).backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())

    def test_rope_relative_position(self):
        cos, sin = precompute_freqs_cis(dim=4, end=5, rope_base=100)
        q = torch.tensor([1., 2., 3., 4.]).repeat(1, 5, 1, 1)
        k = torch.tensor([5., 6., 7., 8.]).repeat(1, 5, 1, 1)
        rotated_q, rotated_k = apply_rotary_pos_emb(q, k, cos, sin)
        # 固定内容且相对距离相同，两个旋转后的内积应相同。
        dot_02 = (rotated_q[:, 0] * rotated_k[:, 2]).sum(-1)
        dot_13 = (rotated_q[:, 1] * rotated_k[:, 3]).sum(-1)
        torch.testing.assert_close(dot_02, dot_13, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
