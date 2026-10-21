"""Run service-free regressions; block Redis even during legacy-test teardown."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

names = [
    "tests.test_phase0", "tests.test_phase1", "tests.test_summary",
    "tests.test_enterprise_rag", "tests.test_chat_features",
    "tests.test_knowledge_flow",
    "tests.test_rate_limit.RateLimitTest.test_memory_fallback",
    "tests.test_rate_limit.RateLimitTest.test_disabled_limit",
]
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
with patch("app.core.rate_limit.get_redis_client", side_effect=RuntimeError("service-free test")):
    result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
