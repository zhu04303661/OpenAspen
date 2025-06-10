"""
数值计算收敛相关异常
"""

class ConvergenceError(Exception):
    """收敛计算基础异常"""
    pass

class MaxIterationsError(ConvergenceError):
    """超过最大迭代次数异常"""
    def __init__(self, max_iterations: int, current_error: float = None):
        self.max_iterations = max_iterations
        self.current_error = current_error
        message = f"Maximum iterations ({max_iterations}) exceeded"
        if current_error is not None:
            message += f" with current error: {current_error}"
        super().__init__(message)

class SolutionNotFoundError(ConvergenceError):
    """解未找到异常"""
    def __init__(self, algorithm: str, message: str = ""):
        self.algorithm = algorithm
        super().__init__(f"Solution not found using '{algorithm}' algorithm: {message}")

class NumericalInstabilityError(ConvergenceError):
    """数值不稳定异常"""
    def __init__(self, message: str):
        super().__init__(f"Numerical instability detected: {message}")

class InvalidInitialGuessError(ConvergenceError):
    """无效初值异常"""
    def __init__(self, variable: str, value: float):
        self.variable = variable
        self.value = value
        super().__init__(f"Invalid initial guess for '{variable}': {value}")

class SingularMatrixError(ConvergenceError):
    """奇异矩阵异常"""
    def __init__(self, message: str = "Matrix is singular or near-singular"):
        super().__init__(message) 