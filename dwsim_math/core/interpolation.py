"""
插值算法模块
============

实现各种插值算法，包括线性插值、样条插值、多项式插值等。
这些功能是从DWSIM.Math/Interpolation.vb转换而来的。

主要功能:
- 重心插值公式
- Floater-Hormann有理插值
- 等距多项式插值
- 线性插值

作者: DWSIM团队 (Python转换版本)
许可证: GNU General Public License v3.0
"""

import numpy as np
from typing import Union, List, Tuple, Optional
import warnings
import math


class Interpolation:
    """
    插值算法类
    
    提供各种插值方法，包括线性插值、多项式插值和有理插值。
    主要基于重心插值公式实现。
    """
    
    @staticmethod
    def interpolate(x: Union[List[float], np.ndarray], 
                   y: Union[List[float], np.ndarray], 
                   x_interp: float,
                   method: str = "linear") -> float:
        """
        通用插值接口
        
        参数:
            x: 插值节点数组
            y: 函数值数组  
            x_interp: 插值点
            method: 插值方法 ("linear", "polynomial", "rational")
            
        返回:
            float: 插值结果
            
        示例:
            >>> x = [0, 1, 2, 3, 4]
            >>> y = [1, 2, 5, 10, 17]
            >>> result = Interpolation.interpolate(x, y, 2.5)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if len(x) != len(y):
            raise ValueError("x和y数组长度必须相同")
        
        if len(x) == 0:
            return 0.0
        elif len(x) == 1:
            return float(y[0] * x_interp / x[0]) if x[0] != 0 else float(y[0])
        
        if method == "linear":
            return BarycentricInterpolation.linear_interpolation(x, y, x_interp)
        elif method == "polynomial":
            w = FloaterHormannRational.build_weights(x, len(x))
            return BarycentricInterpolation.barycentric_interpolation(x, y, w, x_interp)
        elif method == "rational":
            w = FloaterHormannRational.build_floater_hormann_weights(x, len(x), 0.5)
            return BarycentricInterpolation.barycentric_interpolation(x, y, w, x_interp)
        else:
            raise ValueError(f"未知的插值方法: {method}")


class BarycentricInterpolation:
    """
    重心插值类
    
    实现基于重心公式的插值算法。重心插值公式具有数值稳定性好、
    计算效率高的特点。
    """
    
    @staticmethod
    def barycentric_interpolation(x: np.ndarray, 
                                 f: np.ndarray, 
                                 w: np.ndarray, 
                                 t: float) -> float:
        """
        重心插值公式
        
        数学公式:
        f(t) = Σ(w_i * f_i / (t - x_i)) / Σ(w_i / (t - x_i))
        
        参数:
            x: 插值节点数组
            f: 函数值数组
            w: 重心权重数组
            t: 插值点
            
        返回:
            float: 插值结果
        """
        n = len(x)
        
        if n == 0:
            return 0.0
        
        # 检查是否在节点上
        for i in range(n):
            if abs(t - x[i]) < 1e-15:
                return float(f[i])
        
        # 寻找最接近的节点
        min_distance = abs(t - x[0])
        closest_index = 0
        
        for i in range(1, n):
            distance = abs(t - x[i])
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # 判断是否使用安全公式
        threshold = math.sqrt(np.finfo(float).tiny)
        use_safe_formula = min_distance <= threshold
        
        if use_safe_formula:
            # 使用安全公式（避免溢出）
            return BarycentricInterpolation._safe_barycentric_interpolation(
                x, f, w, t, closest_index
            )
        else:
            # 使用快速公式
            return BarycentricInterpolation._fast_barycentric_interpolation(
                x, f, w, t
            )
    
    @staticmethod
    def _fast_barycentric_interpolation(x: np.ndarray, 
                                       f: np.ndarray, 
                                       w: np.ndarray, 
                                       t: float) -> float:
        """快速重心插值公式"""
        s1 = 0.0  # 分子
        s2 = 0.0  # 分母
        
        for i in range(len(x)):
            v = w[i] / (t - x[i])
            s1 += v * f[i]
            s2 += v
        
        if abs(s2) < 1e-15:
            warnings.warn("重心插值分母接近零")
            return 0.0
        
        return s1 / s2
    
    @staticmethod
    def _safe_barycentric_interpolation(x: np.ndarray, 
                                       f: np.ndarray, 
                                       w: np.ndarray, 
                                       t: float, 
                                       closest_index: int) -> float:
        """安全重心插值公式（避免溢出）"""
        s1 = w[closest_index] * f[closest_index]
        s2 = w[closest_index]
        s = t - x[closest_index]
        
        for i in range(len(x)):
            if i != closest_index:
                v = s * w[i] / (t - x[i])
                s1 += v * f[i]
                s2 += v
        
        if abs(s2) < 1e-15:
            return float(f[closest_index])
        
        return s1 / s2
    
    @staticmethod
    def linear_interpolation(x: np.ndarray, 
                           y: np.ndarray, 
                           x_interp: float) -> float:
        """
        线性插值
        
        在最接近的两个点之间进行线性插值。
        
        参数:
            x: x坐标数组（必须有序）
            y: y坐标数组
            x_interp: 插值点
            
        返回:
            float: 插值结果
        """
        # 确保数组有序
        sort_indices = np.argsort(x)
        x_sorted = x[sort_indices]
        y_sorted = y[sort_indices]
        
        # 边界情况
        if x_interp <= x_sorted[0]:
            return float(y_sorted[0])
        if x_interp >= x_sorted[-1]:
            return float(y_sorted[-1])
        
        # 寻找插值区间
        for i in range(len(x_sorted) - 1):
            if x_sorted[i] <= x_interp <= x_sorted[i + 1]:
                # 线性插值公式
                t = (x_interp - x_sorted[i]) / (x_sorted[i + 1] - x_sorted[i])
                return float(y_sorted[i] * (1 - t) + y_sorted[i + 1] * t)
        
        return float(y_sorted[-1])
    
    @staticmethod
    def equidistant_polynomial_interpolation(a: float, 
                                           b: float, 
                                           f: np.ndarray, 
                                           t: float) -> float:
        """
        等距多项式插值
        
        在等距节点上进行多项式插值。
        
        参数:
            a: 插值区间左端点
            b: 插值区间右端点
            f: 函数值数组，f[i] = F(a + (b-a)*i/(n-1))
            t: 插值点
            
        返回:
            float: 插值结果
        """
        n = len(f)
        
        if n == 0:
            return 0.0
        elif n == 1:
            return float(f[0])
        
        # 构建等距节点
        x = np.linspace(a, b, n)
        
        # 检查是否在节点上
        for i in range(n):
            if abs(t - x[i]) < 1e-15:
                return float(f[i])
        
        # 计算等距多项式插值的权重
        w = np.zeros(n)
        for i in range(n):
            w[i] = (-1) ** i
            for j in range(n):
                if j != i:
                    w[i] /= (i - j)
        
        # 使用重心插值公式
        return BarycentricInterpolation.barycentric_interpolation(x, f, w, t)


class FloaterHormannRational:
    """
    Floater-Hormann有理插值类
    
    实现Floater-Hormann有理插值方法，该方法结合了多项式插值和
    有理插值的优点，具有良好的数值稳定性。
    """
    
    @staticmethod
    def build_floater_hormann_weights(x: np.ndarray, 
                                     n: int, 
                                     d: float = 0.5) -> np.ndarray:
        """
        构建Floater-Hormann有理插值权重
        
        参数:
            x: 插值节点数组
            n: 节点数量
            d: 插值阶数参数
            
        返回:
            np.ndarray: 权重数组
        """
        if n <= 0:
            return np.array([])
        
        w = np.zeros(n)
        
        # Floater-Hormann权重计算
        d_int = max(0, min(n - 1, int(d * (n - 1))))
        
        for i in range(n):
            w[i] = 0.0
            
            # 计算组合数的和
            for k in range(max(0, i - d_int), min(i, n - d_int - 1) + 1):
                # 计算二项式系数
                binom_coeff = FloaterHormannRational._binomial_coefficient(d_int, i - k)
                
                # 计算权重贡献
                weight_contrib = binom_coeff
                for j in range(k, k + d_int + 1):
                    if j != i:
                        weight_contrib /= abs(x[i] - x[j])
                
                w[i] += (-1) ** (i - k) * weight_contrib
        
        return w
    
    @staticmethod
    def build_weights(x: np.ndarray, n: int) -> np.ndarray:
        """
        构建标准多项式插值权重
        
        参数:
            x: 插值节点数组
            n: 节点数量
            
        返回:
            np.ndarray: 权重数组
        """
        if n <= 0:
            return np.array([])
        
        w = np.ones(n)
        
        # 计算Lagrange插值的重心权重
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i] /= (x[i] - x[j])
        
        return w
    
    @staticmethod
    def _binomial_coefficient(n: int, k: int) -> float:
        """
        计算二项式系数 C(n, k)
        
        参数:
            n: 上标
            k: 下标
            
        返回:
            float: 二项式系数
        """
        if k < 0 or k > n:
            return 0.0
        
        if k == 0 or k == n:
            return 1.0
        
        # 使用对称性优化
        k = min(k, n - k)
        
        result = 1.0
        for i in range(k):
            result = result * (n - i) / (i + 1)
        
        return result


class SplineInterpolation:
    """
    样条插值类
    
    实现三次样条插值算法。
    """
    
    @staticmethod
    def cubic_spline(x: np.ndarray, 
                    y: np.ndarray, 
                    x_interp: Union[float, np.ndarray],
                    boundary_condition: str = "natural") -> Union[float, np.ndarray]:
        """
        三次样条插值
        
        参数:
            x: 节点x坐标
            y: 节点y坐标
            x_interp: 插值点
            boundary_condition: 边界条件 ("natural", "clamped")
            
        返回:
            插值结果
        """
        from scipy.interpolate import CubicSpline
        
        # 使用scipy的实现
        cs = CubicSpline(x, y, bc_type=boundary_condition)
        return cs(x_interp)


# 便捷函数
def interpolate(x: Union[List[float], np.ndarray], 
               y: Union[List[float], np.ndarray], 
               x_interp: float,
               method: str = "linear") -> float:
    """插值的便捷函数"""
    return Interpolation.interpolate(x, y, x_interp, method)

def linear_interp(x: Union[List[float], np.ndarray], 
                 y: Union[List[float], np.ndarray], 
                 x_interp: float) -> float:
    """线性插值的便捷函数"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return BarycentricInterpolation.linear_interpolation(x, y, x_interp)

def cubic_spline_interp(x: Union[List[float], np.ndarray], 
                       y: Union[List[float], np.ndarray], 
                       x_interp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """三次样条插值的便捷函数"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return SplineInterpolation.cubic_spline(x, y, x_interp) 