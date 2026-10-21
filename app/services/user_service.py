"""
用户认证服务
从MySQL数据库验证用户登录，支持用户管理
"""
from app.database.connection import get_sync_db_connection
from app.infra.mysql import connection, transaction
from app.utils.password import verify_password


def verify_login(username: str, password: str) -> dict | None:
    """验证登录"""
    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT username, name, role, department, position, manager_id, password "
                "FROM users WHERE username = %s",
                (username,)
            )
            user = cursor.fetchone()
            
            if not user or not verify_password(password, user["password"]):
                return None
            
            return {
                "username": user["username"],
                "name": user["name"],
                "role": user["role"],
                "department": user["department"],
                "position": user["position"],
                "manager_id": user.get("manager_id"),
            }
    finally:
        conn.close()


def list_users() -> list[dict]:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT username, name, role, department, position, manager_id "
                "FROM users ORDER BY username"
            )
            return cursor.fetchall()


def set_manager(username: str, manager_id: str | None) -> dict | None:
    """设置直属主管；manager_id 为空表示回退部门权限。"""
    with transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
            if not cursor.fetchone():
                return None
            if manager_id:
                if manager_id == username:
                    raise ValueError("不能将自己设为直属主管")
                cursor.execute(
                    "SELECT username, role FROM users WHERE username = %s", (manager_id,)
                )
                manager = cursor.fetchone()
                if not manager:
                    raise ValueError("直属主管不存在")
                if manager["role"] not in ("manager", "admin"):
                    raise ValueError("直属主管必须是 manager 或 admin")
            cursor.execute(
                "UPDATE users SET manager_id = %s WHERE username = %s",
                (manager_id, username),
            )
    return get_user_by_username(username)


def get_user_by_username(username: str) -> dict | None:
    """根据用户名获取用户信息（不含密码）"""
    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT username, name, role, department, position, manager_id "
                "FROM users WHERE username = %s",
                (username,)
            )
            user = cursor.fetchone()
            if user:
                return {
                    "username": user["username"],
                    "name": user["name"],
                    "role": user["role"],
                    "department": user["department"],
                    "position": user["position"],
                    "manager_id": user.get("manager_id"),
                }
    finally:
        conn.close()
    return None
