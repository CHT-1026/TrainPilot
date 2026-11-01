"""
认证路由
处理用户登录、个人信息获取和认证依赖注入
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.domain.user.models import UserContext
from app.models.requests import LoginRequest
from app.services.user_service import verify_login, get_user_by_username
from app.utils.auth import create_access_token, verify_token

# 创建认证路由
router = APIRouter(prefix="/api/auth", tags=["认证"])

# 认证安全配置
security = HTTPBearer(auto_error=False)


# 从Authorization Header获取当前用户（依赖注入函数）
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """从Authorization Header获取当前用户（依赖注入函数）
    
    Args:
        credentials: HTTP认证凭证
        
    Returns:
        dict: 用户信息字典
        
    Raises:
        HTTPException: 认证失败时抛出401异常
    """
    # 如果没有认证凭证
    if not credentials:
        # 抛出401未授权异常
        raise HTTPException(status_code=401, detail="未提供认证信息")
    
    # 获取token
    token = credentials.credentials
    # 验证token
    payload = verify_token(token)
    # 如果token无效
    if not payload:
        # 抛出401过期异常
        raise HTTPException(status_code=401, detail="登录已过期")
    
    # 获取用户名
    username = payload.get("sub")
    # 查询用户信息
    user = get_user_by_username(username)
    # 如果用户不存在
    if not user:
        # 抛出401异常
        raise HTTPException(status_code=401, detail="用户不存在")
    
    # 返回用户信息
    return user


def get_user_context(current_user: dict = Depends(get_current_user)) -> UserContext:
    """服务端用户上下文：权限判断只使用这个对象。"""
    return UserContext.from_dict(current_user)


# 用户登录接口
@router.post("/login")
async def login(request: LoginRequest):
    """用户登录接口"""
    # 验证用户名和密码
    user = verify_login(request.username, request.password)
    # 如果验证失败
    if not user:
        # 抛出401异常
        raise HTTPException(status_code=401, detail="账号或密码错误")
    # 创建访问token
    token = create_access_token(data={"sub": request.username})
    # 返回登录成功响应
    return {
        "code": 200,
        "message": "登录成功",
        "data": {**user, "token": token}
    }


# 获取当前用户信息接口
@router.get("/profile")
async def profile(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    # 返回用户信息
    return {"code": 200, "data": current_user}
