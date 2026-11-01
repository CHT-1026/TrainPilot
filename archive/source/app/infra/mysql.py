"""MySQL 事务与连接上下文。"""
from contextlib import contextmanager

from app.database.connection import get_sync_db_connection


@contextmanager
def connection():
    """只读连接：用完归还连接池。"""
    conn = get_sync_db_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction():
    """写事务：异常回滚，正常提交。"""
    conn = get_sync_db_connection()
    try:
        conn.begin()
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
