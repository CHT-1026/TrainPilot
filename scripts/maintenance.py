"""轻量运维清理：过期幂等键、陈旧会话元数据。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe scripts\\maintenance.py --dry-run
    .venv312\\Scripts\\python.exe scripts\\maintenance.py
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for name in (".env.local", ".env"):
    path = ROOT / name
    if path.exists():
        for key, value in dotenv_values(path).items():
            if value is not None:
                import os

                os.environ.setdefault(key, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from app.core.config import get_settings
    from app.database.connection import get_redis_client, get_sync_db_connection

    settings = get_settings()
    result = {"idempotency_expired": 0, "sessions_stale": 0, "dry_run": args.dry_run}

    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM idempotency_keys WHERE expires_at < NOW()")
            result["idempotency_expired"] = int(cursor.fetchone()["total"])
            if not args.dry_run and result["idempotency_expired"]:
                cursor.execute("DELETE FROM idempotency_keys WHERE expires_at < NOW()")
        if not args.dry_run:
            conn.commit()
    finally:
        conn.close()

    try:
        redis_client = get_redis_client()
        cutoff = datetime.now() - timedelta(days=settings.session_ttl_days)
        for key in redis_client.scan_iter(match="session_metadata:*"):
            for field, value in redis_client.hgetall(key).items():
                try:
                    meta = json.loads(value)
                except json.JSONDecodeError:
                    continue
                updated = meta.get("updated_at", "")
                try:
                    if datetime.fromisoformat(updated) < cutoff:
                        result["sessions_stale"] += 1
                        if not args.dry_run:
                            redis_client.hdel(key, field)
                except ValueError:
                    continue
    finally:
        pass

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
