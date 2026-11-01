"""JWT 认证工具。密钥统一来自 Settings。"""
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

from app.core.config import get_settings

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7


def _secret_key() -> str:
    return get_settings().secret_key


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, _secret_key(), algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
    except JWTError:
        return None
