"""用替身Python检查真实Bash入口的版本选择与续训参数；不启动GPU训练。"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
GIT = shutil.which("git")
GIT_BASH = Path(GIT).resolve().parents[1] / "bin/bash.exe" if GIT else None
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH and GIT_BASH.exists() else shutil.which("bash")


@unittest.skipUnless(BASH, "需要Bash检查服务器启动脚本")
class LoRALauncherChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "runs").mkdir(exist_ok=True)

    def launch(self, directory, mode, version="v3", resume=False):
        root = Path(directory)
        output = root / "run"
        if resume:
            output.mkdir()
            (output / "checkpoint.pt").touch()
        arguments = root / "arguments.txt"
        env = {**os.environ, "TRAINPILOT_SFT_MODEL": "/models/original-full-sft",
               "TRAINPILOT_LORA_RUN": output.as_posix(), "LAUNCH_ARGS": arguments.as_posix(),
               "SCRIPT": (ROOT / "model/trainer/run_trainpilot_lora.sh").as_posix(),
               "MODE": mode, "VERSION": version}
        command = 'python() { printf "%s\\n" "$@" > "$LAUNCH_ARGS"; }; export -f python; bash "$SCRIPT" "$MODE" "$VERSION"'
        result = subprocess.run([BASH, "-c", command], env=env, cwd=ROOT,
                                capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return arguments.read_text(encoding="utf-8").splitlines(), result.stdout

    def test_v3_smoke_starts_fresh_with_new_data_and_ten_updates(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            args, message = self.launch(directory, "smoke")
        self.assertEqual(args[args.index("--model") + 1], "/models/original-full-sft")
        self.assertEqual(args[args.index("--data-path") + 1], "model/dataset/datasets/trainpilot_agent/v3/train.jsonl")
        self.assertEqual(args[args.index("--validation-data-path") + 1], "model/dataset/datasets/trainpilot_agent/v3/val.jsonl")
        self.assertEqual(args[args.index("--max-updates") + 1], "10")
        self.assertEqual(args[args.index("--epochs") + 1], "2")
        self.assertNotIn("--resume", args)

    def test_v3_train_resumes_its_own_checkpoint_without_smoke_limit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            args, message = self.launch(directory, "train", resume=True)
            self.assertEqual(args[args.index("--resume") + 1], (Path(directory) / "run/checkpoint.pt").as_posix())
        self.assertNotIn("--max-updates", args)

    def test_v2_still_selects_v2_data(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            args, message = self.launch(directory, "smoke", version="v2")
        self.assertEqual(args[args.index("--data-path") + 1], "model/dataset/datasets/trainpilot_agent/v2/train.jsonl")

    def test_v4_smoke_uses_v4_and_preserves_each_epoch_adapter(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            args, _ = self.launch(directory, "smoke", version="v4")
        self.assertEqual(args[args.index("--data-path") + 1], "model/dataset/datasets/trainpilot_agent/v4/train.jsonl")
        self.assertEqual(args[args.index("--validation-data-path") + 1], "model/dataset/datasets/trainpilot_agent/v4/val.jsonl")
        self.assertEqual(args[args.index("--model") + 1], "/models/original-full-sft")
        self.assertIn("--save-epoch-adapters", args)
        self.assertEqual(args[args.index("--max-updates") + 1], "10")
        self.assertNotIn("--resume", args)

    def test_v4_train_only_resumes_its_own_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            args, _ = self.launch(directory, "train", version="v4", resume=True)
            self.assertEqual(args[args.index("--resume") + 1], (Path(directory) / "run/checkpoint.pt").as_posix())
        self.assertIn("--save-epoch-adapters", args)
        self.assertNotIn("--max-updates", args)

    def test_v5_fresh_mixed_training_and_own_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            args, _ = self.launch(directory, "smoke", version="v5")
        self.assertEqual(args[args.index("--data-path") + 1], "model/dataset/datasets/trainpilot_agent/v5/train.jsonl")
        self.assertEqual(args[args.index("--max-length") + 1], "2048")
        self.assertIn("--save-epoch-adapters", args)
        self.assertNotIn("--resume", args)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            args, _ = self.launch(directory, "train", version="v5", resume=True)
            self.assertEqual(args[args.index("--resume") + 1], (Path(directory) / "run/checkpoint.pt").as_posix())
        self.assertNotIn("--max-updates", args)


if __name__ == "__main__":
    unittest.main()
