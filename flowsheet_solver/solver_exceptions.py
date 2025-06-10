"""
求解器异常类定义
==============

定义FlowsheetSolver使用的各种异常类型，包括收敛异常、超时异常等。
这些异常类对应原VB.NET版本中的异常处理机制。
"""

class SolverException(Exception):
    """
    求解器基础异常类
    
    所有FlowsheetSolver相关异常的基类，提供统一的异常处理接口。
    """
    
    def __init__(self, message: str = "求解器异常", inner_exception=None):
        super().__init__(message)
        self.message = message  # 添加message属性以便测试访问
        self.inner_exception = inner_exception
        self.detailed_description = ""
        self.user_action = ""
    
    def add_detail_info(self, detailed_description: str, user_action: str = ""):
        """
        添加详细错误信息和用户操作建议
        
        Args:
            detailed_description: 详细错误描述
            user_action: 建议的用户操作
        """
        self.detailed_description = detailed_description
        self.user_action = user_action


class ConvergenceException(SolverException):
    """
    收敛异常
    
    当循环求解或同步调节无法收敛时抛出此异常。
    """
    
    def __init__(self, message: str = "收敛失败", max_iterations: int = 0, 
                 current_error: float = 0.0, tolerance: float = 0.0):
        super().__init__(message)
        self.max_iterations = max_iterations
        self.current_error = current_error
        self.tolerance = tolerance
        
    def __str__(self):
        # 如果有详细信息则显示，否则只显示基本消息
        if self.max_iterations > 0 or self.current_error > 0 or self.tolerance > 0:
            return (f"{super().__str__()} - "
                    f"最大迭代次数: {self.max_iterations}, "
                    f"当前误差: {self.current_error:.6e}, "
                    f"容差: {self.tolerance:.6e}")
        else:
            return super().__str__()


class TimeoutException(SolverException):
    """
    超时异常
    
    当计算超过设定的超时时间时抛出此异常。
    """
    
    def __init__(self, message: str = "计算超时", timeout_seconds: float = 0.0, 
                 elapsed_seconds: float = 0.0, operation: str = ""):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds  # 添加测试期望的属性
        self.operation = operation  # 添加测试期望的属性
        
    def __str__(self):
        # 只有在指定了超时时间时才显示额外信息
        if self.timeout_seconds > 0:
            return f"{super().__str__()} - 超时时间: {self.timeout_seconds}秒"
        else:
            return super().__str__()


class InfiniteLoopException(SolverException):
    """
    无限循环异常
    
    当检测到无限循环依赖时抛出此异常。
    通常需要插入Recycle对象来打断循环。
    """
    
    def __init__(self, message: str = "检测到无限循环依赖"):
        super().__init__(message)
        self.add_detail_info(
            "在获取流程图对象计算顺序时检测到无限循环",
            "请在需要的位置插入Recycle循环切断对象"
        )


class CalculationException(SolverException):
    """
    计算异常
    
    单个对象计算过程中发生错误时抛出此异常。
    """
    
    def __init__(self, message: str = "计算异常", object_name: str = "", object_type: str = "",
                 calculation_object: str = "", calculation_step: str = ""):
        super().__init__(message)
        self.object_name = object_name
        self.object_type = object_type
        self.calculation_object = calculation_object or object_name  # 添加测试期望的属性
        self.calculation_step = calculation_step  # 添加测试期望的属性
        
    def __str__(self):
        # 只有在指定了对象名称时才显示额外信息
        if self.object_name:
            return f"{super().__str__()} - 对象: {self.object_name} ({self.object_type})"
        return super().__str__()


class NetworkException(SolverException):
    """
    网络异常
    
    远程计算过程中发生网络错误时抛出此异常。
    """
    
    def __init__(self, message: str = "网络异常", server_address: str = "", server_port: int = 0):
        super().__init__(message)
        self.server_address = server_address
        self.server_port = server_port  # 添加测试期望的属性
        
    def __str__(self):
        # 只有在指定了服务器地址时才显示额外信息
        if self.server_address:
            return f"{super().__str__()} - 服务器: {self.server_address}"
        return super().__str__()


class DataException(SolverException):
    """
    数据异常
    
    数据序列化、压缩或传输过程中发生错误时抛出此异常。
    """
    
    def __init__(self, message: str = "数据异常", data_size: int = 0, data_type: str = "",
                 invalid_value: str = ""):
        super().__init__(message)
        self.data_size = data_size
        self.data_type = data_type  # 添加测试期望的属性
        self.invalid_value = invalid_value  # 添加测试期望的属性
        
    def __str__(self):
        # 只有在指定了数据大小时才显示额外信息
        if self.data_size > 0:
            return f"{super().__str__()} - 数据大小: {self.data_size}字节"
        return super().__str__() 