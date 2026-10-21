"""
密码加密工具模块
使用bcrypt算法对用户密码进行加密存储
"""
import bcrypt


def hash_password(password: str) -> str:
    """将明文密码加密为bcrypt哈希值
    
    Args:
        password: 明文密码
        
    Returns:
        bcrypt哈希字符串
    """
    # 生成盐值并加密
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码是否正确
    
    Args:
        plain_password: 用户输入的明文密码
        hashed_password: 数据库中存储的bcrypt哈希值
        
    Returns:
        密码是否匹配
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False
