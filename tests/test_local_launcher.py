"""Evaluate launcher stop predicates with mocked PowerShell process cmdlets."""
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lantu_launcher_test", ROOT / "启动本地服务.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class FakeState:
    def __init__(self, data=None):
        self.data = data
        self.unlinked = False

    def exists(self): return self.data is not None
    def read_text(self, **kwargs): return json.dumps(self.data)
    def unlink(self, **kwargs): self.unlinked = True


class LauncherStopTest(unittest.TestCase):
    def capture(self, state=None, root=ROOT):
        scripts = []
        fake_state = FakeState(state)
        with patch.object(launcher, "STATE", fake_state), patch.object(launcher, "ROOT", root), \
             patch.object(launcher.subprocess, "run", side_effect=lambda args, **kw: scripts.append(args[-1])), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.stop_owned()
        self.assertTrue(fake_state.unlinked)
        return scripts

    def evaluate_backend(self, script, root=ROOT):
        env = str(root / ".env.local")
        own = f'"{root / ".venv312/Scripts/python.exe"}" -u -m uvicorn local_runtime.web_app:app --env-file "{env}" --port 8080'
        base = f'"C:\\Runtime\\python.exe" -u -m uvicorn local_runtime.web_app:app --env-file "{env}" --port 8080'
        processes = [
            {"ProcessId": 101, "Name": "python.exe", "CommandLine": own},
            {"ProcessId": 102, "Name": "python.exe", "CommandLine": base},
            {"ProcessId": 103, "Name": "PYTHONW.EXE", "CommandLine": base.replace("-u -m", "-m")},
            {"ProcessId": 104, "Name": "powershell.exe", "CommandLine": own},
            {"ProcessId": 105, "Name": "python.exe", "CommandLine": base.replace(env, r"C:\Other\.env.local")},
            {"ProcessId": 106, "Name": "python.exe", "CommandLine": base.replace(env, str(root) + "-copy\\.env.local")},
            {"ProcessId": 107, "Name": "python.exe", "CommandLine": 'python.exe -c "print(\' -m uvicorn local_runtime.web_app:app --env-file ' + env + '\')"'},
            {"ProcessId": 108, "Name": "python.exe", "CommandLine": base.replace("web_app:app ", "web_app:app_extra ")},
            {"ProcessId": 109, "Name": "python.exe", "CommandLine": base.replace(env, env + ".backup")},
        ]
        payload = json.dumps(processes, ensure_ascii=False).replace("'", "''")
        # These functions shadow both OS process cmdlets. This test never calls
        # the real Stop-Process and never enumerates the host's actual processes.
        prefix = (
            "$ErrorActionPreference = 'Stop'; "
            "$script:oaMockProcesses = ConvertFrom-Json '" + payload + "'; "
            "$script:oaMockStopped = @(); "
            "function Get-CimInstance { param($ClassName, $Filter) "
            "if ($Filter -match '^ProcessId=(\\d+)$') { "
            "return @($script:oaMockProcesses | Where-Object { $_.ProcessId -eq [int]$Matches[1] }) } "
            "return $script:oaMockProcesses }; "
            "function Stop-Process { param($Id, $ErrorAction) $script:oaMockStopped += [int]$Id }; "
        )
        result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", prefix + script +
                                 "; ConvertTo-Json -Compress @($script:oaMockStopped)"],
                                capture_output=True, text=True, encoding="utf-8", check=True,
                                creationflags=launcher.HIDDEN)
        return json.loads(result.stdout.strip())

    def test_stale_recorded_pid_finds_both_wrapper_and_base_interpreter(self):
        script = self.capture({"backend_pid": 999})[0]
        self.assertEqual(sorted(self.evaluate_backend(script)), [101, 102, 103])

    def test_reused_pid_of_foreign_command_is_not_stopped(self):
        script = self.capture({"backend_pid": 109})[0]
        self.assertEqual(sorted(self.evaluate_backend(script)), [101, 102, 103])

    def test_no_state_file_still_finds_orphaned_owned_backend(self):
        script = self.capture()[0]
        self.assertEqual(sorted(self.evaluate_backend(script)), [101, 102, 103])

    def test_root_with_spaces_and_apostrophe_is_exactly_quoted(self):
        root = pathlib.Path(r"C:\Owned Root\O'Brien")
        script = self.capture({"backend_pid": 999}, root=root)[0]
        self.assertEqual(sorted(self.evaluate_backend(script, root=root)), [101, 102, 103])

    def test_untrusted_wsl_marker_is_not_used_to_stop_any_keeper(self):
        scripts = self.capture({"backend_pid": 999, "wsl_pid": 200, "wsl_marker": "exec sleep infinity"})
        self.assertEqual(len(scripts), 1)


if __name__ == "__main__":
    unittest.main()
