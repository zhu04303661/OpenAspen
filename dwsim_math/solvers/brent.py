"""
Brent求根方法模块
================

实现Brent方法求解单变量非线性方程的根。Brent方法结合了二分法、割线法和
反二次插值法的优点，具有较好的收敛性和数值稳定性。

这些功能是从DWSIM.Math/Brent.vb转换而来的。

主要功能:
- 单变量方程求根
- 函数最小值搜索
- 自适应区间细分

作者: DWSIM团队 (Python转换版本)
许可证: GNU General Public License v3.0
"""

import math
import numpy as np
from typing import Callable, Any, Tuple, Optional
import warnings


class BrentSolver:
    """
    Brent求根算法求解器
    
    使用Brent方法求解单变量非线性方程 f(x) = 0 的根。Brent方法是一种混合
    算法，结合了二分法的稳定性和割线法的快速收敛性。
    
    算法特点:
    - 保证收敛（如果初始区间包含根）
    - 具有超线性收敛速度
    - 数值稳定性好
    """
    
    def __init__(self):
        """初始化Brent求解器"""
        self._function = None
        self._last_result = None
    
    def set_function(self, func: Callable[[float, Any], float]):
        """
        设置要求解的函数
        
        参数:
            func: 函数对象，接受参数 (x, other_args) 并返回函数值
        """
        self._function = func
    
    def evaluate_function(self, x: float, other_args: Any = None) -> float:
        """
        计算函数值
        
        参数:
            x: 自变量值
            other_args: 其他参数
            
        返回:
            float: 函数值 f(x)
        """
        if self._function is None:
            raise ValueError("必须先设置函数")
        return self._function(x, other_args)
    
    def solve(self, 
              func: Callable[[float, Any], float],
              min_val: float, 
              max_val: float,
              tolerance: float = 1e-8,
              max_iterations: int = 100,
              subdivision_count: int = 10,
              other_args: Any = None) -> float:
        """
        使用Brent方法求解方程 f(x) = 0 的根
        
        参数:
            func: 目标函数，接受参数 (x, other_args) 并返回函数值
            min_val: 搜索区间的下界
            max_val: 搜索区间的上界
            tolerance: 求解精度，默认1e-8
            max_iterations: 最大迭代次数，默认100
            subdivision_count: 初始区间细分数，默认10
            other_args: 传递给函数的其他参数
            
        返回:
            float: 方程的根
            
        示例:
            >>> def f(x, args):
            ...     return x**3 - 2*x - 5
            >>> solver = BrentSolver()
            >>> root = solver.solve(f, 1.0, 3.0)
            >>> print(f"根: {root}")
        """
        self.set_function(func)
        
        # 第一步：找到包含根的区间
        x_lower, x_upper = self._find_bracketing_interval(
            min_val, max_val, subdivision_count, other_args
        )
        
        # 第二步：使用Brent方法精确求根
        root = self._brent_method(
            x_lower, x_upper, tolerance, max_iterations, other_args
        )
        
        self._last_result = root
        return root
    
    def _find_bracketing_interval(self, 
                                 min_val: float, 
                                 max_val: float,
                                 n_subdivisions: int,
                                 other_args: Any) -> Tuple[float, float]:
        """
        找到包含根的区间
        
        通过细分初始区间，找到一个子区间使得函数在区间端点处的值异号，
        从而保证该区间内存在根。
        
        参数:
            min_val: 初始区间下界
            max_val: 初始区间上界
            n_subdivisions: 细分数
            other_args: 函数的其他参数
            
        返回:
            tuple: (下界, 上界) 包含根的区间
            
        异常:
            ValueError: 如果找不到包含根的区间
        """
        delta_x = (max_val - min_val) / n_subdivisions
        x_lower = min_val
        
        while x_lower < max_val:
            x_upper = x_lower + delta_x
            
            if x_upper > max_val:
                x_upper = max_val
            
            f_lower = self.evaluate_function(x_lower, other_args)
            f_upper = self.evaluate_function(x_upper, other_args)
            
            # 检查是否找到了包含根的区间（函数值异号）
            if f_lower * f_upper < 0:
                return x_lower, x_upper
            
            # 检查是否某个端点就是根
            if abs(f_lower) < 1e-15:
                return x_lower, x_lower
            if abs(f_upper) < 1e-15:
                return x_upper, x_upper
            
            x_lower = x_upper
        
        # 如果没有找到包含根的区间，尝试扩展搜索范围
        extended_range = (max_val - min_val) * 0.1
        extended_min = min_val - extended_range
        extended_max = max_val + extended_range
        
        # 检查扩展范围是否有根
        try:
            f_ext_min = self.evaluate_function(extended_min, other_args)
            f_ext_max = self.evaluate_function(extended_max, other_args)
            f_min = self.evaluate_function(min_val, other_args)
            f_max = self.evaluate_function(max_val, other_args)
            
            if f_ext_min * f_min < 0:
                return extended_min, min_val
            if f_max * f_ext_max < 0:
                return max_val, extended_max
        except:
            pass
        
        raise ValueError(f"在区间 [{min_val}, {max_val}] 内找不到根，"
                        f"函数值: f({min_val}) = {self.evaluate_function(min_val, other_args)}, "
                        f"f({max_val}) = {self.evaluate_function(max_val, other_args)}")
    
    def _brent_method(self, 
                     x_lower: float, 
                     x_upper: float,
                     tolerance: float,
                     max_iterations: int,
                     other_args: Any) -> float:
        """
        Brent方法的核心实现
        
        参数:
            x_lower: 区间下界
            x_upper: 区间上界
            tolerance: 求解精度
            max_iterations: 最大迭代次数
            other_args: 函数的其他参数
            
        返回:
            float: 方程的根
        """
        # 初始化
        a = x_lower
        b = x_upper
        c = x_upper
        
        fa = self.evaluate_function(a, other_args)
        fb = self.evaluate_function(b, other_args)
        fc = fb
        
        # 主迭代循环
        for iteration in range(max_iterations):
            # 确保 |f(b)| <= |f(c)|
            if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
                c = a
                fc = fa
                d = b - a
                e = d
            
            if abs(fc) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa
            
            # 检查收敛条件
            tol1 = tolerance
            xm = 0.5 * (c - b)
            
            if abs(xm) <= tol1 or fb == 0:
                return b
            
            if abs(fb) < tol1:
                return b
            
            # 决定使用插值还是二分法
            if abs(e) >= tol1 and abs(fa) > abs(fb):
                s = fb / fa
                
                if a == c:
                    # 线性插值（割线法）
                    p = 2 * xm * s
                    q = 1 - s
                else:
                    # 反二次插值
                    q = fa / fc
                    r = fb / fc
                    p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                    q = (q - 1) * (r - 1) * (s - 1)
                
                if p > 0:
                    q = -q
                p = abs(p)
                
                min1 = 3 * xm * q - abs(tol1 * q)
                min2 = abs(e * q)
                
                if 2 * p < min(min1, min2):
                    # 接受插值步长
                    e = d
                    d = p / q
                else:
                    # 使用二分法
                    d = xm
                    e = d
            else:
                # 使用二分法
                d = xm
                e = d
            
            # 更新点a
            a = b
            fa = fb
            
            # 更新点b
            if abs(d) > tol1:
                b += d
            else:
                b += math.copysign(tol1, xm)
            
            fb = self.evaluate_function(b, other_args)
        
        warnings.warn(f"Brent方法在{max_iterations}次迭代后未收敛，当前结果: {b}")
        return b
    
    def find_minimum(self,
                    func: Callable[[float, Any], float],
                    x_min: float,
                    x_max: float,
                    tolerance: float = 1e-8,
                    max_iterations: int = 100,
                    other_args: Any = None) -> Tuple[float, float]:
        """
        使用Brent方法寻找函数的最小值
        
        通过寻找函数导数的零点来找到最小值点。
        
        参数:
            func: 目标函数
            x_min: 搜索区间下界
            x_max: 搜索区间上界
            tolerance: 求解精度
            max_iterations: 最大迭代次数
            other_args: 函数的其他参数
            
        返回:
            tuple: (最小值点, 最小值)
        """
        # 定义导数函数（使用数值微分）
        def derivative(x, args):
            h = max(abs(x) * 1e-8, 1e-8)
            return (func(x + h, args) - func(x - h, args)) / (2 * h)
        
        # 找到导数为零的点
        try:
            x_min_point = self.solve(
                derivative, x_min, x_max, tolerance, max_iterations, 10, other_args
            )
            min_value = func(x_min_point, other_args)
            return x_min_point, min_value
        except ValueError:
            # 如果找不到导数零点，使用三点搜索
            return self._golden_section_search(func, x_min, x_max, tolerance, other_args)
    
    def _golden_section_search(self,
                              func: Callable[[float, Any], float],
                              a: float,
                              b: float,
                              tolerance: float,
                              other_args: Any) -> Tuple[float, float]:
        """
        黄金分割搜索法寻找最小值
        
        参数:
            func: 目标函数
            a: 搜索区间下界
            b: 搜索区间上界
            tolerance: 求解精度
            other_args: 函数的其他参数
            
        返回:
            tuple: (最小值点, 最小值)
        """
        phi = (1 + math.sqrt(5)) / 2  # 黄金分割比
        
        # 初始化两个内部点
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi
        
        f1 = func(x1, other_args)
        f2 = func(x2, other_args)
        
        while abs(b - a) > tolerance:
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - (b - a) / phi
                f1 = func(x1, other_args)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (b - a) / phi
                f2 = func(x2, other_args)
        
        x_min = (a + b) / 2
        min_value = func(x_min, other_args)
        return x_min, min_value
    
    @property
    def last_result(self) -> Optional[float]:
        """获取最后一次求解的结果"""
        return self._last_result


# 便捷函数
def brent_solve(func: Callable[[float, Any], float],
               a: float, 
               b: float,
               tol: float = 1e-8,
               maxiter: int = 100,
               args: Any = None) -> float:
    """
    使用Brent方法求解方程根的便捷函数
    
    参数:
        func: 目标函数
        a: 搜索区间下界
        b: 搜索区间上界
        tol: 求解精度
        maxiter: 最大迭代次数
        args: 函数的其他参数
        
    返回:
        float: 方程的根
    """
    solver = BrentSolver()
    return solver.solve(func, a, b, tol, maxiter, 10, args)

def brent_minimize(func: Callable[[float, Any], float],
                  a: float,
                  b: float,
                  tol: float = 1e-8,
                  maxiter: int = 100,
                  args: Any = None) -> Tuple[float, float]:
    """
    使用Brent方法寻找函数最小值的便捷函数
    
    参数:
        func: 目标函数
        a: 搜索区间下界
        b: 搜索区间上界
        tol: 求解精度
        maxiter: 最大迭代次数
        args: 函数的其他参数
        
    返回:
        tuple: (最小值点, 最小值)
    """
    solver = BrentSolver()
    return solver.find_minimum(func, a, b, tol, maxiter, args) 