"""检查日志口径、断点去重、区间 token 加权与自动绘图。"""
import json
from pathlib import Path
import tempfile
import unittest

from trainer.plot_loss import LossWindow, plot_loss, read_history


class LossPlotChecks(unittest.TestCase):
    def test_window_uses_token_weights_and_resume_baseline(self):
        window = LossWindow()
        self.assertEqual(window.update({"loss_sum":30., "valid_tokens":10}), 3.)
        self.assertEqual(window.update({"loss_sum":50., "valid_tokens":20}), 2.)
        restored = LossWindow({"loss_sum":50., "valid_tokens":20})
        self.assertEqual(restored.update({"loss_sum":110., "valid_tokens":50}), 2.)

    def test_qwen_stdout_gets_true_end_step_without_fabricating_windows(self):
        text = "\n".join([
            "batch=1; accumulation=8; max_length=512; epochs=1",
            "epoch=1 batch=240/41819 global_step=30 loss=1.669601 lr=0.00001000",
            "epoch=1 batch=41760/41819 global_step=5220 loss=1.394746 lr=0.00000100",
            "epoch=1/1 train_loss=1.394851 val_loss=1.384495",
            "本次耗时（含验证与导出）=7617.0s; completed_epochs=1; global_step=5228",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"train.log"
            path.write_text(text, encoding="utf-8")
            history = read_history(path)
        self.assertEqual(history.epochs[0]["step"],5228)
        self.assertEqual(history.validations,[{"step":5228,"loss":1.384495}])
        self.assertEqual(len(history.updates),2)
        self.assertTrue(all(row["window_loss"] is None for row in history.updates))

    def test_jsonl_duplicate_resume_and_initial_validation_formats(self):
        rows = [
            {"event":"initial_validation","loss":2.4},
            {"event":"update","epoch":1,"global_step":10,"mean_loss":2.0},
            {"event":"update","epoch":1,"global_step":10,"mean_loss":1.9,"window_loss":1.8,"learning_rate":1e-5},
            {"event":"epoch","completed_epochs":1,"global_step":11,"train":{"loss":1.85},"validation":{"loss":2.1}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metrics.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows),encoding="utf-8")
            history = read_history(path)
            figure = plot_loss(path, title="Fixture SFT")
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size,1000)
        self.assertEqual(len(history.updates),1)
        self.assertEqual(history.updates[0]["loss"],1.9)
        self.assertEqual([row["step"] for row in history.validations],[0,11])
        self.assertEqual(history.updates[0]["lr"],1e-5)

    def test_multi_epoch_stdout_supports_mean_loss_and_explicit_summary_step(self):
        text = "\n".join([
            "训练前验证 loss=2.500000",
            "epoch=1 batch=8/8 global_step=1 mean_loss=2.2 window_loss=2.2 lr=0.001",
            "epoch=1/2 global_step=1 train_loss=2.2 val_loss=2.3",
            "epoch=2 batch=8/8 global_step=2 mean_loss=2.0 window_loss=2.0 lr=0.0001",
            "epoch=2/2 global_step=2 train_loss=2.0 val_loss=2.1",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"train.log"
            path.write_text(text,encoding="utf-8")
            history=read_history(path)
        self.assertEqual([row["step"] for row in history.epochs],[1,2])
        self.assertEqual([row["loss"] for row in history.validations],[2.5,2.3,2.1])


if __name__ == "__main__":
    unittest.main()
