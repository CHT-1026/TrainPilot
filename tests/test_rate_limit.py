"""限流单元测试：Redis 路径与内存降级。"""
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

# 单元测试固定使用本机 Redis，避免默认 .env 指向远程实例导致连接等待。
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")

from app.core.rate_limit import check_rate_limit, reset_rate_limit


class RateLimitTest(unittest.TestCase):
    def test_redis_path(self):
        key = "test:" + uuid4().hex
        reset_rate_limit(key)
        try:
            for _ in range(3):
                allowed, remaining, _ = check_rate_limit(key, 3, 60)
                self.assertTrue(allowed)
            allowed, remaining, retry_after = check_rate_limit(key, 3, 60)
            self.assertFalse(allowed)
            self.assertEqual(remaining, 0)
            self.assertGreater(retry_after, 0)
        finally:
            reset_rate_limit(key)

    def test_memory_fallback(self):
        key = "test:" + uuid4().hex
        with patch("app.core.rate_limit.get_redis_client", side_effect=RuntimeError("no redis")):
            allowed, _, _ = check_rate_limit(key, 2, 60)
            self.assertTrue(allowed)
            allowed, _, _ = check_rate_limit(key, 2, 60)
            self.assertTrue(allowed)
            allowed, _, retry_after = check_rate_limit(key, 2, 60)
            self.assertFalse(allowed)
            self.assertGreater(retry_after, 0)
        reset_rate_limit(key)

    def test_disabled_limit(self):
        allowed, _, retry_after = check_rate_limit("test:disabled", 0, 60)
        self.assertTrue(allowed)
        self.assertEqual(retry_after, 0)


if __name__ == "__main__":
    unittest.main()
