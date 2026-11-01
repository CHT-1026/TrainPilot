"""限流单元测试：Redis 路径与内存降级。"""
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.core.rate_limit import check_rate_limit, reset_rate_limit


class RateLimitTest(unittest.TestCase):
    def test_redis_path(self):
        key = "test:" + uuid4().hex
        redis = MagicMock()
        redis.incr.side_effect = [1, 2, 3, 4]
        redis.ttl.return_value = 42
        with patch("app.core.rate_limit.get_redis_client", return_value=redis), \
             patch("app.core.rate_limit._memory_check", side_effect=AssertionError("Redis path must not fall back")):
            reset_rate_limit(key)
            for _ in range(3):
                allowed, remaining, _ = check_rate_limit(key, 3, 60)
                self.assertTrue(allowed)
            allowed, remaining, retry_after = check_rate_limit(key, 3, 60)
            self.assertFalse(allowed)
            self.assertEqual(remaining, 0)
            self.assertEqual(retry_after, 42)
            reset_rate_limit(key)
        redis.expire.assert_called_once_with(f"rate_limit:{key}", 60)
        redis.ttl.assert_called_once_with(f"rate_limit:{key}")
        self.assertEqual(redis.incr.call_count, 4)
        self.assertEqual(redis.delete.call_count, 2)

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
