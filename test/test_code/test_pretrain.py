"""训练基础函数检查；通过 unittest discovery 运行，无独立入口。"""
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from config.config import TrainPilotconfig
from model.model import TrainPilotForCausalLM
from trainer.pretrain import next_token_loss, train_step


class NonFiniteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.full((1, 4, 32), float("nan")))

    def forward(self, input_ids, attention_mask=None):
        return self.scores.expand(input_ids.shape[0], -1, -1)


class PretrainChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = TrainPilotconfig(
            vocab_size=32, hidden_size=16, num_layers=1, num_heads=4,
            num_kv_heads=2, intermediate_size=32, max_seq_len=4, dropout=0.0,
        )
        self.batch = {
            "input_ids": torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "labels": torch.tensor([[1, 4, 5, 2], [1, 6, 2, -100]]),
        }

    def test_uniform_scores_ignore_padding_and_first_label(self):
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        labels = torch.tensor([[999, 0, 1, -100]])
        original_labels = labels.clone()
        loss, count = next_token_loss(logits, labels)
        self.assertEqual(count, 2)
        self.assertEqual(loss.ndim, 0)
        self.assertAlmostEqual(loss.item(), math.log(3), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # 位置 2 对应忽略标签，位置 3 没有后续标签；两者梯度均应为零。
        torch.testing.assert_close(logits.grad[:, 2:], torch.zeros(1, 2, 3))
        torch.testing.assert_close(labels, original_labels)

    def test_next_token_alignment(self):
        logits = torch.zeros(1, 4, 3)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 0] = 10.0
        # 首个标签不是监督目标，后续三个标签依次为 1、2、0。
        labels = torch.tensor([[999, 1, 2, 0]])
        loss, count = next_token_loss(logits, labels)
        self.assertEqual(count, 3)
        self.assertLess(loss.item(), 0.001)

    def test_rejects_invalid_shapes_dtype_and_empty_supervision(self):
        cases = [
            (torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.long), ValueError),
            (torch.zeros(1, 3, 4), torch.zeros(1, 2, dtype=torch.long), ValueError),
            (torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.long), ValueError),
            (torch.zeros(0, 3, 4), torch.zeros(0, 3, dtype=torch.long), ValueError),
            (torch.zeros(1, 3, 0), torch.zeros(1, 3, dtype=torch.long), ValueError),
            (torch.zeros(1, 3, 4), torch.zeros(1, 3), TypeError),
            (torch.zeros(1, 3, 4), torch.tensor([[1, -100, -100]]), ValueError),
        ]
        for logits, labels, error in cases:
            with self.subTest(logits_shape=logits.shape, labels_dtype=labels.dtype):
                with self.assertRaises(error):
                    next_token_loss(logits, labels)

    def test_train_step_updates_parameters_and_returns_plain_values(self):
        model = TrainPilotForCausalLM(self.config).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        before_weight = model.model.embed_tokens.weight.detach().clone()
        original_batch = {name: tensor.clone() for name, tensor in self.batch.items()}
        with torch.no_grad():
            initial_logits = model(self.batch["input_ids"], self.batch["attention_mask"])
            expected_loss, _ = next_token_loss(initial_logits, self.batch["labels"])
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            record = train_step(model, optimizer, self.batch, "cpu")
            step.assert_called_once()
        self.assertTrue(model.training)
        self.assertIsInstance(record["loss"], float)
        self.assertIsInstance(record["valid_tokens"], int)
        self.assertEqual(record["valid_tokens"], 5)
        self.assertAlmostEqual(record["loss"], expected_loss.item(), places=6)
        self.assertFalse(torch.equal(before_weight, model.model.embed_tokens.weight))
        for name, tensor in self.batch.items():
            torch.testing.assert_close(tensor, original_batch[name])
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_train_step_clears_stale_gradients(self):
        clean = TrainPilotForCausalLM(self.config)
        stale = TrainPilotForCausalLM(self.config)
        stale.load_state_dict(clean.state_dict())
        for parameter in stale.parameters():
            parameter.grad = torch.full_like(parameter, 10.0)
        clean_optimizer = torch.optim.SGD(clean.parameters(), lr=0.01)
        stale_optimizer = torch.optim.SGD(stale.parameters(), lr=0.01)
        train_step(clean, clean_optimizer, self.batch, "cpu")
        train_step(stale, stale_optimizer, self.batch, "cpu")
        clean_parameters = dict(clean.named_parameters())
        for name, parameter in stale.named_parameters():
            torch.testing.assert_close(parameter, clean_parameters[name], rtol=0, atol=0)

    def test_nonfinite_loss_does_not_update_parameters(self):
        model = NonFiniteModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(FloatingPointError):
                train_step(model, optimizer, self.batch, "cpu")
            step.assert_not_called()
        self.assertIsNone(model.scores.grad)
