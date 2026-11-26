"""Compatibility entry for the unified project launcher."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.launch import main
if __name__ == "__main__":
    raise SystemExit(main())
