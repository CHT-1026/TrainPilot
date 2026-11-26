"""The unified launcher must never stop another project's processes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent import launch as launcher


class LauncherStopTest(unittest.TestCase):
    def run_stop(self, command, *, pid=100, role="backend", children=None):
        process = Mock()
        process.cmdline.return_value = command
        process.children.return_value = children or []
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "processes.json"
            state.write_text(json.dumps({role+"_pid": pid}), encoding="utf-8")
            with patch.object(launcher,"STATE",state), patch("psutil.Process", return_value=process), \
                    contextlib.redirect_stdout(io.StringIO()):
                launcher.stop_owned()
            self.assertFalse(state.exists())
        return process

    def test_stops_owned_backend_and_python_redirector_child(self):
        command = ["python.exe","-m","uvicorn","agent.local_runtime.web_app:app",
                   "--env-file",str(launcher.AGENT_ROOT/".env.local")]
        child=Mock(); child.cmdline.return_value=command
        parent=self.run_stop(command,children=[child])
        child.terminate.assert_called_once()
        parent.terminate.assert_called_once()

    def test_reused_pid_with_other_command_is_not_stopped(self):
        parent=self.run_stop(["python.exe","-m","uvicorn","other.main:app"])
        parent.terminate.assert_not_called()

    def test_other_project_and_suffix_path_are_not_stopped(self):
        for env in [r"C:\Other\.env.local",str(launcher.AGENT_ROOT/".env.local")+".backup"]:
            parent=self.run_stop(["python.exe","-m","uvicorn","agent.local_runtime.web_app:app","--env-file",env])
            parent.terminate.assert_not_called()

    def test_foreign_wsl_keeper_is_not_stopped(self):
        parent=self.run_stop(["wsl.exe","--exec","sleep","infinity"],role="wsl")
        parent.terminate.assert_not_called()

    def test_pinned_adapter_process_can_be_stopped(self):
        config=launcher.load_config()
        parent=self.run_stop(["python.exe","-m","model.serving.tools_api",
            "--expected-adapter-sha256",config["adapter_sha256"],
            "--adapter-dir",str(launcher.project_path(config["adapter_dir"]))],role="model")
        parent.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
