"""检查 epoch 统计、验证边界与 checkpoint 恢复，通过 unittest discovery 运行。"""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trainpilot.config import TrainPilotconfig
from trainpilot.model import TrainPilotForCausalLM
from trainpilot.trainer import evaluate, load_checkpoint, save_checkpoint, train_one_epoch
from train import main as train_main


class FixedScoresModel(nn.Module):
    """每个位置输出相同的可学习分数，用于独立核对统计。"""

    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))
        self.forward_modes = []

    def forward(self, input_ids, attention_mask=None):
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        return self.scores.expand(*input_ids.shape, -1)


class TrainerChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.samples = [
            {
                "input_ids": torch.tensor([1, 0, 0, 0]),
                "attention_mask": torch.tensor([1, 1, 0, 0]),
                "labels": torch.tensor([1, 0, -100, -100]),
            },
            {
                "input_ids": torch.tensor([1, 1, 1, 1]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "labels": torch.tensor([1, 1, 1, 1]),
            },
        ]
        # 独立构造四个真实监督位置，不采用 batch 均值的平均。
        self.expected_loss = F.cross_entropy(
            torch.tensor([[2.0, 0.0, 0.0]]).repeat(4, 1),
            torch.tensor([0, 1, 1, 1]),
        ).item()

    def test_train_loss_is_weighted_by_valid_tokens(self):
        model = FixedScoresModel().eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        record = train_one_epoch(
            model, optimizer, DataLoader(self.samples, batch_size=1), "cpu",
        )
        self.assertAlmostEqual(record["loss"], self.expected_loss, places=6)
        self.assertEqual(record["valid_tokens"], 4)
        self.assertEqual(record["steps"], 2)
        self.assertIsInstance(record["loss"], float)
        self.assertIsInstance(record["valid_tokens"], int)
        self.assertTrue(all(mode == (True, True) for mode in model.forward_modes))

    def test_epoch_updates_model_and_keeps_last_partial_batch(self):
        config = TrainPilotconfig(
            vocab_size=3, hidden_size=16, num_layers=1, num_heads=4,
            num_kv_heads=2, intermediate_size=32, max_seq_len=4,
        )
        model = TrainPilotForCausalLM(config)
        before = model.model.embed_tokens.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(self.samples + self.samples[:1], batch_size=2)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            record = train_one_epoch(model, optimizer, loader, "cpu")
            self.assertEqual(step.call_count, 2)
        self.assertEqual(record["steps"], 2)
        self.assertEqual(record["valid_tokens"], 5)
        self.assertFalse(torch.equal(before, model.model.embed_tokens.weight))

    def test_evaluation_is_batch_size_independent_and_restores_mode(self):
        for training in (True, False):
            for batch_size in (1, 2):
                with self.subTest(training=training, batch_size=batch_size):
                    model = FixedScoresModel().train(training)
                    model.scores.grad = torch.ones_like(model.scores)
                    before = model.scores.detach().clone()
                    old_grad = model.scores.grad.clone()
                    record = evaluate(
                        model, DataLoader(self.samples, batch_size=batch_size), "cpu",
                    )
                    self.assertAlmostEqual(record["loss"], self.expected_loss, places=6)
                    self.assertEqual(record["valid_tokens"], 4)
                    self.assertEqual(record["steps"], 2 if batch_size == 1 else 1)
                    self.assertEqual(model.training, training)
                    self.assertTrue(all(mode == (False, False) for mode in model.forward_modes))
                    torch.testing.assert_close(model.scores, before, rtol=0, atol=0)
                    torch.testing.assert_close(model.scores.grad, old_grad, rtol=0, atol=0)

    def test_empty_loaders_fail_and_evaluation_restores_mode(self):
        model = FixedScoresModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaises(ValueError):
            train_one_epoch(model, optimizer, DataLoader([], batch_size=1), "cpu")
        for training in (True, False):
            model.train(training)
            with self.assertRaises(ValueError):
                evaluate(model, DataLoader([], batch_size=1), "cpu")
            self.assertEqual(model.training, training)

    def test_nonfinite_evaluation_fails_and_restores_mode(self):
        model = FixedScoresModel()
        with torch.no_grad():
            model.scores.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            evaluate(model, DataLoader(self.samples, batch_size=1), "cpu")
        self.assertTrue(model.training)
        self.assertIsNone(model.scores.grad)

    def test_invalid_batch_restores_evaluation_mode(self):
        model = FixedScoresModel().eval()
        with self.assertRaises(KeyError):
            evaluate(model, DataLoader([{}], batch_size=1), "cpu")
        self.assertFalse(model.training)

    def test_accumulation_matches_large_batch_with_unequal_valid_token_counts(self):
        config = TrainPilotconfig(
            vocab_size=3, hidden_size=16, num_layers=1, num_heads=4,
            num_kv_heads=2, intermediate_size=32, max_seq_len=4,
        )
        large = TrainPilotForCausalLM(config)
        accumulated = TrainPilotForCausalLM(config)
        accumulated.load_state_dict(large.state_dict())
        samples = self.samples + self.samples[:1]
        large_record = train_one_epoch(
            large, torch.optim.SGD(large.parameters(), lr=0.01),
            DataLoader(samples, batch_size=2), "cpu",
        )
        small_record = train_one_epoch(
            accumulated, torch.optim.SGD(accumulated.parameters(), lr=0.01),
            DataLoader(samples, batch_size=1), "cpu", accumulation_steps=2,
        )
        self.assertEqual(small_record["steps"], 2)
        self.assertEqual(small_record["valid_tokens"], 5)
        self.assertAlmostEqual(small_record["loss"], large_record["loss"], places=6)
        for name, value in large.state_dict().items():
            torch.testing.assert_close(accumulated.state_dict()[name], value, rtol=1e-5, atol=1e-6)

    def test_bfloat16_autocast_keeps_model_parameters_float32(self):
        config = TrainPilotconfig(
            vocab_size=3, hidden_size=16, num_layers=1, num_heads=4,
            num_kv_heads=2, intermediate_size=32, max_seq_len=4,
        )
        model = TrainPilotForCausalLM(config)
        seen_dtypes = []
        hook = model.lm_head.register_forward_hook(lambda module, inputs, output: seen_dtypes.append(output.dtype))
        try:
            record = train_one_epoch(
                model, torch.optim.AdamW(model.parameters(), lr=1e-3),
                DataLoader(self.samples, batch_size=1), "cpu", accumulation_steps=2,
                amp_dtype=torch.bfloat16, grad_clip=1.0,
            )
            val = evaluate(model, DataLoader(self.samples, batch_size=2), "cpu", amp_dtype=torch.bfloat16)
        finally:
            hook.remove()
        self.assertEqual(record["steps"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(val["loss"])))
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in seen_dtypes))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))

    def test_cosine_schedule_reaches_minimum_learning_rate(self):
        model = FixedScoresModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        seen = []
        train_one_epoch(
            model, optimizer, DataLoader(self.samples, batch_size=1), "cpu",
            base_lr=0.01, total_micro_steps=2,
            on_update=lambda state: seen.append(optimizer.param_groups[0]["lr"]),
        )
        self.assertAlmostEqual(seen[0], 0.0055)
        self.assertAlmostEqual(seen[1], 0.001)

    def test_nonfinite_gradient_does_not_update_parameters_when_clipping(self):
        model = FixedScoresModel()
        original = model.scores.detach().clone()
        model.scores.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(RuntimeError):
                train_one_epoch(model, optimizer, DataLoader(self.samples, batch_size=2), "cpu", grad_clip=1.0)
            step.assert_not_called()
        torch.testing.assert_close(model.scores, original, rtol=0, atol=0)


class CheckpointChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        random.seed(42)
        self.config = TrainPilotconfig(
            vocab_size=8, hidden_size=16, num_layers=1, num_heads=4,
            num_kv_heads=2, intermediate_size=32, max_seq_len=4,
        )
        self.samples = [
            {
                "input_ids": torch.tensor([1, token, 2, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "labels": torch.tensor([1, token, 2, -100]),
            }
            for token in (3, 4, 5)
        ]

    def make_model_and_optimizer(self, config=None):
        model = TrainPilotForCausalLM(config or self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        return model, optimizer

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_nested_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def test_round_trip_restores_weights_adamw_state_and_progress(self):
        model, optimizer = self.make_model_and_optimizer()
        train_one_epoch(model, optimizer, DataLoader(self.samples, batch_size=2), "cpu")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, completed_epochs=1, global_step=2)
            restored, restored_optimizer = self.make_model_and_optimizer()
            # 恢复应覆盖新优化器的超参数，而不是仅恢复模型权重。
            restored_optimizer.param_groups[0]["lr"] = 0.5
            progress = load_checkpoint(path, restored, restored_optimizer)
            self.assertEqual(progress, {"completed_epochs": 1, "global_step": 2})
            self.assert_nested_equal(restored.state_dict(), model.state_dict())
            self.assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
            model.eval()
            restored.eval()
            batch = next(iter(DataLoader(self.samples, batch_size=3)))
            with torch.no_grad():
                torch.testing.assert_close(
                    restored(batch["input_ids"], batch["attention_mask"]),
                    model(batch["input_ids"], batch["attention_mask"]),
                    rtol=0, atol=0,
                )

    def test_resumed_next_epoch_matches_continuous_training_with_dropout_and_shuffle(self):
        config = replace(self.config, dropout=0.2)
        model, optimizer = self.make_model_and_optimizer(config)
        loader = DataLoader(self.samples, batch_size=2, shuffle=True, num_workers=0)
        first = train_one_epoch(model, optimizer, loader, "cpu")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 1, first["steps"])
            expected_record = train_one_epoch(model, optimizer, loader, "cpu")
            expected_random = random.random()

            restored, restored_optimizer = self.make_model_and_optimizer(config)
            torch.rand(17)
            random.random()
            progress = load_checkpoint(path, restored, restored_optimizer)
            resumed_loader = DataLoader(self.samples, batch_size=2, shuffle=True, num_workers=0)
            actual_record = train_one_epoch(restored, restored_optimizer, resumed_loader, "cpu")
            self.assertEqual(actual_record, expected_record)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(progress["global_step"], first["steps"])
            self.assert_nested_equal(restored.state_dict(), model.state_dict())
            self.assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())

    def test_rejects_different_config_before_modifying_model(self):
        model, optimizer = self.make_model_and_optimizer()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 0, 0)
            restored, restored_optimizer = self.make_model_and_optimizer(
                replace(self.config, rope_theta=10000.0),
            )
            before = {key: value.clone() for key, value in restored.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "模型配置"):
                load_checkpoint(path, restored, restored_optimizer)
            self.assert_nested_equal(restored.state_dict(), before)

    def test_rejects_different_optimizer_type(self):
        model, optimizer = self.make_model_and_optimizer()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 0, 0)
            wrong_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            with self.assertRaisesRegex(ValueError, "优化器类型"):
                load_checkpoint(path, model, wrong_optimizer)

    def test_failed_save_preserves_previous_checkpoint_and_removes_temp_file(self):
        model, optimizer = self.make_model_and_optimizer()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 0, 0)
            before = path.read_bytes()
            with patch("trainpilot.trainer.torch.save", side_effect=OSError("模拟写入失败")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, model, optimizer, 1, 2)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_invalid_progress_does_not_create_checkpoint(self):
        model, optimizer = self.make_model_and_optimizer()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            for epochs, step in ((-1, 0), (0, -1), (True, 0), (0, 1.5)):
                with self.subTest(epochs=epochs, step=step):
                    with self.assertRaises(ValueError):
                        save_checkpoint(path, model, optimizer, epochs, step)
                    self.assertFalse(path.exists())


class TrainingEntryChecks(unittest.TestCase):
    def make_config(self, directory):
        root = Path(__file__).resolve().parents[1]
        config = {
            "model": {
                "vocab_size": 6400, "hidden_size": 16, "num_layers": 1,
                "num_heads": 4, "num_kv_heads": 2, "intermediate_size": 32,
                "max_seq_len": 8, "dropout": 0.2,
            },
            "data": {
                "tokenizer_path": str(root / "assets" / "tokenizer.json"),
                "train_path": str(root / "tests" / "fixtures" / "train.jsonl"),
                "val_path": str(root / "tests" / "fixtures" / "val.jsonl"),
            },
            "training": {
                "device": "cpu", "dtype": "float32", "epochs": 1,
                "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.01,
                "seed": 42, "num_workers": 0, "output_dir": str(Path(directory) / "run"),
            },
        }
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path, config

    def test_complete_entry_and_resumed_run_match_two_continuous_epochs(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config_path, _ = self.make_config(directory)
            continuous = Path(directory) / "continuous"
            resumed = Path(directory) / "resumed"
            train_main(["--config", str(config_path), "--epochs", "2", "--output-dir", str(continuous)])
            train_main(["--config", str(config_path), "--output-dir", str(resumed)])
            train_main([
                "--config", str(config_path), "--epochs", "2", "--output-dir", str(resumed),
                "--resume", str(resumed / "checkpoint.pt"),
            ])
            expected = torch.load(continuous / "checkpoint.pt", map_location="cpu", weights_only=True)
            actual = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(actual["completed_epochs"], 2)
            self.assertEqual(actual["global_step"], 4)
            for name, tensor in expected["model_state_dict"].items():
                torch.testing.assert_close(actual["model_state_dict"][name], tensor, rtol=0, atol=0)
            expected_metrics = [json.loads(line) for line in (continuous / "metrics.jsonl").read_text().splitlines() if json.loads(line)["event"] != "update"]
            actual_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines() if json.loads(line)["event"] != "update"]
            self.assertEqual(len(actual_metrics), 3)
            for actual_record, expected_record in zip(actual_metrics, expected_metrics):
                for key in ("train", "validation", "global_step"):
                    self.assertEqual(actual_record.get(key), expected_record.get(key))

    def test_entry_rejects_changed_resume_data_and_fresh_overwrite(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config_path, config = self.make_config(directory)
            train_main(["--config", str(config_path)])
            with self.assertRaises(FileExistsError):
                train_main(["--config", str(config_path)])
            changed = Path(directory) / "changed.jsonl"
            changed.write_text('{"text": "不同的训练语料"}\n', encoding="utf-8")
            config["data"]["train_path"] = str(changed)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "数据与原运行不一致"):
                train_main([
                    "--config", str(config_path), "--epochs", "2", "--resume",
                    str(Path(directory) / "run" / "checkpoint.pt"),
                ])

    def test_mid_epoch_resume_matches_continuous_cosine_accumulated_training(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config_path, config = self.make_config(directory)
            config["training"].update(
                epochs=2, batch_size=1, accumulation_steps=2, lr_schedule="cosine",
                grad_clip=1.0, save_interval=1,
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            continuous = Path(directory) / "continuous"
            resumed = Path(directory) / "resumed"
            train_main(["--config", str(config_path), "--output-dir", str(continuous)])
            train_main(["--config", str(config_path), "--output-dir", str(resumed), "--max-updates", "1"])
            partial = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(partial["completed_epochs"], 0)
            self.assertEqual(partial["global_step"], 1)
            self.assertEqual(partial["epoch_state"]["micro_steps"], 2)
            train_main([
                "--config", str(config_path), "--output-dir", str(resumed),
                "--resume", str(resumed / "checkpoint.pt"),
            ])
            expected = torch.load(continuous / "checkpoint.pt", map_location="cpu", weights_only=True)
            actual = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(actual["completed_epochs"], 2)
            self.assertEqual(actual["global_step"], 4)
            self.assertIsNone(actual["epoch_state"])
            for name, tensor in expected["model_state_dict"].items():
                torch.testing.assert_close(actual["model_state_dict"][name], tensor, rtol=0, atol=0)
            expected_epochs = [json.loads(line) for line in (continuous / "metrics.jsonl").read_text().splitlines() if json.loads(line)["event"] == "epoch"]
            actual_epochs = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines() if json.loads(line)["event"] == "epoch"]
            self.assertEqual(len(actual_epochs), 2)
            for a, b in zip(actual_epochs, expected_epochs):
                self.assertEqual(a["train"], b["train"])
                self.assertEqual(a["validation"], b["validation"])

    def test_prepare_data_entry_does_not_start_training(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "raw.jsonl"
            source.write_text("\n".join(json.dumps({"text": f"语料 {index}"}, ensure_ascii=False) for index in range(100)), encoding="utf-8")
            output = root / "data"
            train_main(["--prepare-data", str(source), "--data-output-dir", str(output), "--val-fraction", "0.2"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["train"] + manifest["counts"]["val"], 100)
            self.assertFalse((output / "checkpoint.pt").exists())
