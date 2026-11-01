"""
数据库连接池管理
提供同步 MySQL 和 Redis 连接。

配置在首次建池时读取，避免模块导入顺序早于 .env 加载。
"""
import logging

import pymysql
import redis
import redis.asyncio as aioredis
from dbutils.pooled_db import PooledDB

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# MySQL 连接池全局变量
_mysql_pool = None

# Redis 连接池全局变量
_redis_client = None
_async_redis_client = None


def get_sync_db_connection():
    """获取 MySQL 连接（使用连接池）。"""
    global _mysql_pool

    if _mysql_pool is None:
        settings = get_settings()
        _mysql_pool = PooledDB(
            creator=pymysql,
            maxconnections=10,
            mincached=2,
            maxcached=5,
            blocking=True,
            maxusage=None,
            setsession=[],
            ping=1,
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
            read_timeout=15,
            write_timeout=15,
        )
        logger.info(
            "MySQL连接池初始化成功 host=%s database=%s",
            settings.mysql_host,
            settings.mysql_database,
        )

    return _mysql_pool.connection()


def get_redis_client():
    """获取全局 Redis 客户端。"""
    global _redis_client

    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
            socket_keepalive=True,
            socket_keepalive_options={},
            socket_connect_timeout=5,
            socket_timeout=15,
        )
        logger.info("Redis客户端初始化成功")

    return _redis_client


def get_async_redis_client():
    """异步 Redis 客户端：供 AsyncRedisSaver 与 astream 使用。"""
    global _async_redis_client

    if _async_redis_client is None:
        settings = get_settings()
        _async_redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=5,
            socket_timeout=15,
        )
        logger.info("异步 Redis 客户端初始化成功")

    return _async_redis_client


async def close_async_redis():
    global _async_redis_client
    if _async_redis_client is not None:
        await _async_redis_client.aclose()
        _async_redis_client = None
        logger.info("异步 Redis 连接已关闭")


def close_all_connections():
    """应用关闭时释放连接。"""
    global _mysql_pool, _redis_client

    _mysql_pool = None

    if _redis_client:
        _redis_client.close()
        logger.info("Redis连接已关闭")
        _redis_client = None
