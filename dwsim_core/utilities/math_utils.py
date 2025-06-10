"""
数学工具函数
提供热力学计算中常用的数值方法
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
from ..exceptions.convergence_errors import MaxIterationsError, SolutionNotFoundError

class NewtonRaphson:
    """牛顿-拉夫逊法求解器"""
    
    @staticmethod
    def solve(func: Callable[[float], float], 
              dfunc: Callable[[float], float],
              x0: float,
              tolerance: float = 1e-6,
              max_iterations: int = 100) -> float:
        """
        使用牛顿-拉夫逊法求解方程
        
        Args:
            func: 目标函数
            dfunc: 目标函数的导数
            x0: 初值
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
            
        Returns:
            方程的解
        """
        x = x0
        
        for i in range(max_iterations):
            f_val = func(x)
            
            if abs(f_val) < tolerance:
                return x
            
            df_val = dfunc(x)
            if abs(df_val) < 1e-15:
                raise SolutionNotFoundError("Newton-Raphson", "Derivative is zero")
            
            x_new = x - f_val / df_val
            
            if abs(x_new - x) < tolerance:
                return x_new
            
            x = x_new
        
        raise MaxIterationsError(max_iterations, abs(func(x)))

class BisectionMethod:
    """二分法求解器"""
    
    @staticmethod
    def solve(func: Callable[[float], float],
              a: float, b: float,
              tolerance: float = 1e-6,
              max_iterations: int = 100) -> float:
        """
        使用二分法求解方程
        
        Args:
            func: 目标函数
            a: 区间左端点
            b: 区间右端点
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
            
        Returns:
            方程的解
        """
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            raise SolutionNotFoundError("Bisection", "Function values at endpoints have same sign")
        
        for i in range(max_iterations):
            c = (a + b) / 2
            fc = func(c)
            
            if abs(fc) < tolerance or (b - a) / 2 < tolerance:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        raise MaxIterationsError(max_iterations, abs(func((a + b) / 2)))

class RachfordRice:
    """Rachford-Rice方程求解器"""
    
    @staticmethod
    def solve(K_values: List[float], 
              z_feed: List[float],
              tolerance: float = 1e-6,
              max_iterations: int = 100) -> float:
        """
        求解Rachford-Rice方程
        
        Args:
            K_values: K值列表 (yi/xi)
            z_feed: 进料组成
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
            
        Returns:
            汽相摩尔分数
        """
        def rachford_rice_function(V):
            """Rachford-Rice函数"""
            return sum((K_values[i] - 1) * z_feed[i] / (1 + V * (K_values[i] - 1)) 
                      for i in range(len(K_values)))
        
        def rachford_rice_derivative(V):
            """Rachford-Rice函数的导数"""
            return -sum((K_values[i] - 1)**2 * z_feed[i] / (1 + V * (K_values[i] - 1))**2 
                       for i in range(len(K_values)))
        
        # 确定求解区间
        K_min = min(K_values)
        K_max = max(K_values)
        
        if K_min >= 1:
            return 1.0  # 全部为汽相
        elif K_max <= 1:
            return 0.0  # 全部为液相
        
        # 使用二分法求解
        V_min = 1 / (1 - K_max) + 1e-10
        V_max = 1 / (1 - K_min) - 1e-10
        
        return BisectionMethod.solve(rachford_rice_function, V_min, V_max, tolerance, max_iterations)

def cubic_roots(a: float, b: float, c: float, d: float) -> List[float]:
    """
    求解三次方程 ax³ + bx² + cx + d = 0
    
    Args:
        a, b, c, d: 三次方程系数
        
    Returns:
        实根列表
    """
    if abs(a) < 1e-15:
        # 退化为二次方程
        if abs(b) < 1e-15:
            # 退化为一次方程
            if abs(c) < 1e-15:
                return []
            return [-d / c]
        else:
            # 二次方程
            discriminant = c**2 - 4*b*d
            if discriminant < 0:
                return []
            elif discriminant == 0:
                return [-c / (2*b)]
            else:
                sqrt_disc = np.sqrt(discriminant)
                return [(-c + sqrt_disc) / (2*b), (-c - sqrt_disc) / (2*b)]
    
    # 标准化系数
    b_norm = b / a
    c_norm = c / a
    d_norm = d / a
    
    # 使用NumPy求解三次方程
    coeffs = [1, b_norm, c_norm, d_norm]
    roots = np.roots(coeffs)
    
    # 返回实根
    real_roots = []
    for root in roots:
        if np.isreal(root):
            real_roots.append(float(np.real(root)))
    
    return real_roots

def linear_interpolation(x: float, x_data: List[float], y_data: List[float]) -> float:
    """
    线性插值
    
    Args:
        x: 插值点
        x_data: x数据点
        y_data: y数据点
        
    Returns:
        插值结果
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have same length")
    
    if len(x_data) < 2:
        raise ValueError("Need at least 2 data points for interpolation")
    
    # 查找插值区间
    if x <= x_data[0]:
        return y_data[0]
    elif x >= x_data[-1]:
        return y_data[-1]
    
    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            # 线性插值
            t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + t * (y_data[i + 1] - y_data[i])
    
    raise ValueError("Failed to find interpolation interval")

def numerical_derivative(func: Callable[[float], float], 
                        x: float, 
                        h: float = 1e-8) -> float:
    """
    数值求导（中心差分）
    
    Args:
        func: 目标函数
        x: 求导点
        h: 步长
        
    Returns:
        导数值
    """
    return (func(x + h) - func(x - h)) / (2 * h)

def tridiagonal_solve(a: List[float], b: List[float], c: List[float], d: List[float]) -> List[float]:
    """
    求解三对角矩阵方程组
    
    Args:
        a: 下对角线
        b: 主对角线
        c: 上对角线
        d: 右侧向量
        
    Returns:
        解向量
    """
    n = len(d)
    
    # 前向消元
    for i in range(1, n):
        m = a[i] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]
    
    # 回代
    x = [0.0] * n
    x[-1] = d[-1] / b[-1]
    
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x 