"""
自定义异常类
提供统一的错误码和异常处理
"""


# 业务异常基类
class BusinessException(Exception):
    """业务异常基类"""
    # 默认错误代码
    code = "BUSINESS_ERROR"
    # 默认HTTP状态码
    status_code = 400
    
    # 初始化异常
    def __init__(self, message: str, code: str = None):
        super().__init__(message)
        # 保存错误消息
        self.message = message
        # 如果提供了错误代码
        if code:
            # 更新错误代码
            self.code = code
