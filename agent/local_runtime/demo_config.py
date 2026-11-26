"""Local deployment paths, relative to the unified project root."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = PROJECT_ROOT / "agent"
CONFIG_PATH = AGENT_ROOT / "config/local_demo.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
