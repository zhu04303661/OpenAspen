"""
复数模块
========

提供复数的数学运算功能，包括基本算术运算、三角函数、指数函数等。
这些功能是从DWSIM.Math.DotNumerics/Complex.cs转换而来的。

主要功能:
- 复数的基本运算（加减乘除）
- 复数的幅值和幅角计算
- 复数的共轭、模长等属性
- 复数的三角函数和指数函数

作者: DWSIM团队 (Python转换版本)
许可证: GNU General Public License v3.0
"""

import math
import numpy as np
from typing import Union, Tuple


class Complex:
    """
    复数类
    
    表示一个复数，支持各种复数运算。复数由实部和虚部组成，可以进行
    基本的算术运算、三角函数运算和指数函数运算。
    
    属性:
        real: 实部
        imaginary: 虚部
        modulus: 模长（绝对值）
        argument: 幅角
        conjugate: 共轭复数
    """
    
    def __init__(self, real: float = 0.0, imaginary: float = 0.0):
        """
        初始化复数
        
        参数:
            real: 实部，默认为0.0
            imaginary: 虚部，默认为0.0
            
        示例:
            >>> z1 = Complex(3, 4)  # 3 + 4i
            >>> z2 = Complex(2.5)   # 2.5 + 0i
            >>> z3 = Complex()      # 0 + 0i
        """
        self._real = float(real)
        self._imaginary = float(imaginary)
    
    @property
    def real(self) -> float:
        """获取或设置实部"""
        return self._real
    
    @real.setter
    def real(self, value: float):
        """设置实部"""
        self._real = float(value)
    
    @property
    def imaginary(self) -> float:
        """获取或设置虚部"""
        return self._imaginary
    
    @imaginary.setter
    def imaginary(self, value: float):
        """设置虚部"""
        self._imaginary = float(value)
    
    @property
    def imag(self) -> float:
        """获取虚部（imaginary的别名）"""
        return self._imaginary
    
    @imag.setter
    def imag(self, value: float):
        """设置虚部（imaginary的别名）"""
        self._imaginary = float(value)
    
    @property
    def conjugate(self) -> 'Complex':
        """
        获取复数的共轭
        
        返回:
            Complex: 共轭复数
            
        示例:
            >>> z = Complex(3, 4)
            >>> z_conj = z.conjugate
            >>> print(z_conj)  # 3 - 4i
        """
        return Complex(self._real, -self._imaginary)
    
    @property
    def modulus(self) -> float:
        """
        获取复数的模长（绝对值）
        
        返回:
            float: 模长 |z| = √(real² + imaginary²)
            
        示例:
            >>> z = Complex(3, 4)
            >>> print(z.modulus)  # 5.0
        """
        return math.sqrt(self._real * self._real + self._imaginary * self._imaginary)
    
    @property
    def argument(self) -> float:
        """
        获取复数的幅角
        
        返回:
            float: 幅角（弧度），范围为[-π, π]
            
        示例:
            >>> z = Complex(1, 1)
            >>> print(z.argument)  # π/4 ≈ 0.7854
        """
        return math.atan2(self._imaginary, self._real)
    
    @argument.setter
    def argument(self, value: float):
        """
        设置复数的幅角（保持模长不变）
        
        参数:
            value: 新的幅角（弧度）
        """
        modulus = self.modulus
        self._real = math.cos(value) * modulus
        self._imaginary = math.sin(value) * modulus
    
    def __add__(self, other: Union['Complex', float, int]) -> 'Complex':
        """
        复数加法
        
        参数:
            other: 另一个复数或实数
            
        返回:
            Complex: 相加的结果
        """
        if isinstance(other, Complex):
            return Complex(self._real + other._real, self._imaginary + other._imaginary)
        else:
            return Complex(self._real + float(other), self._imaginary)
    
    def __radd__(self, other: Union[float, int]) -> 'Complex':
        """右加法"""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Complex', float, int]) -> 'Complex':
        """
        复数减法
        
        参数:
            other: 另一个复数或实数
            
        返回:
            Complex: 相减的结果
        """
        if isinstance(other, Complex):
            return Complex(self._real - other._real, self._imaginary - other._imaginary)
        else:
            return Complex(self._real - float(other), self._imaginary)
    
    def __rsub__(self, other: Union[float, int]) -> 'Complex':
        """右减法"""
        return Complex(float(other) - self._real, -self._imaginary)
    
    def __mul__(self, other: Union['Complex', float, int]) -> 'Complex':
        """
        复数乘法
        
        参数:
            other: 另一个复数或实数
            
        返回:
            Complex: 相乘的结果
            
        公式:
            (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        """
        if isinstance(other, Complex):
            real_part = self._real * other._real - self._imaginary * other._imaginary
            imag_part = self._real * other._imaginary + self._imaginary * other._real
            return Complex(real_part, imag_part)
        else:
            return Complex(self._real * float(other), self._imaginary * float(other))
    
    def __rmul__(self, other: Union[float, int]) -> 'Complex':
        """右乘法"""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Complex', float, int]) -> 'Complex':
        """
        复数除法
        
        参数:
            other: 另一个复数或实数
            
        返回:
            Complex: 相除的结果
            
        公式:
            (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
        """
        if isinstance(other, Complex):
            if other._real == 0 and other._imaginary == 0:
                raise ZeroDivisionError("不能除以零")
            
            denominator = other._real * other._real + other._imaginary * other._imaginary
            real_part = (self._real * other._real + self._imaginary * other._imaginary) / denominator
            imag_part = (self._imaginary * other._real - self._real * other._imaginary) / denominator
            return Complex(real_part, imag_part)
        else:
            if other == 0:
                raise ZeroDivisionError("不能除以零")
            return Complex(self._real / float(other), self._imaginary / float(other))
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Complex':
        """右除法"""
        if self._real == 0 and self._imaginary == 0:
            raise ZeroDivisionError("不能除以零")
        
        other_float = float(other)
        denominator = self._real * self._real + self._imaginary * self._imaginary
        real_part = (other_float * self._real) / denominator
        imag_part = -(other_float * self._imaginary) / denominator
        return Complex(real_part, imag_part)
    
    def __pos__(self) -> 'Complex':
        """正号操作符"""
        return Complex(self._real, self._imaginary)
    
    def __neg__(self) -> 'Complex':
        """负号操作符"""
        return Complex(-self._real, -self._imaginary)
    
    def __abs__(self) -> float:
        """绝对值（模长）"""
        return self.modulus
    
    def abs(self) -> float:
        """绝对值（模长）的方法形式"""
        return self.modulus
    
    def arg(self) -> float:
        """幅角的方法形式"""
        return self.argument
    
    def __eq__(self, other: Union['Complex', float, int]) -> bool:
        """
        相等比较
        
        参数:
            other: 另一个复数或实数
            
        返回:
            bool: 是否相等
        """
        if isinstance(other, Complex):
            return (abs(self._real - other._real) < 1e-15 and 
                    abs(self._imaginary - other._imaginary) < 1e-15)
        else:
            return (abs(self._real - float(other)) < 1e-15 and 
                    abs(self._imaginary) < 1e-15)
    
    def __ne__(self, other: Union['Complex', float, int]) -> bool:
        """不等比较"""
        return not self.__eq__(other)
    
    def __str__(self) -> str:
        """
        字符串表示
        
        返回:
            str: 复数的字符串表示，如 "3 + 4i" 或 "2 - 3i"
        """
        if self._imaginary == 0:
            return f"{self._real}"
        elif self._real == 0:
            if self._imaginary == 1:
                return "i"
            elif self._imaginary == -1:
                return "-i"
            else:
                return f"{self._imaginary}i"
        else:
            if self._imaginary == 1:
                return f"{self._real} + i"
            elif self._imaginary == -1:
                return f"{self._real} - i"
            elif self._imaginary > 0:
                return f"{self._real} + {self._imaginary}i"
            else:
                return f"{self._real} - {abs(self._imaginary)}i"
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"Complex({self._real}, {self._imaginary})"
    
    def __hash__(self) -> int:
        """哈希值"""
        return hash((self._real, self._imaginary))
    
    def __complex__(self) -> complex:
        """转换为Python内置复数类型"""
        return complex(self._real, self._imaginary)
    
    def to_string(self, format_spec: str = "G") -> str:
        """
        格式化字符串表示
        
        参数:
            format_spec: 格式说明符
            
        返回:
            str: 格式化的字符串
        """
        if format_spec.upper() == "G":
            return self.__str__()
        else:
            real_str = f"{self._real:{format_spec}}"
            imag_str = f"{self._imaginary:{format_spec}}"
            
            if self._imaginary == 0:
                return real_str
            elif self._real == 0:
                return f"{imag_str}i"
            elif self._imaginary > 0:
                return f"{real_str} + {imag_str}i"
            else:
                return f"{real_str} - {abs(float(imag_str))}i"
    
    @staticmethod
    def from_polar(modulus: float, argument: float) -> 'Complex':
        """
        从极坐标形式创建复数
        
        参数:
            modulus: 模长
            argument: 幅角（弧度）
            
        返回:
            Complex: 对应的复数
            
        示例:
            >>> z = Complex.from_polar(5, math.pi/4)  # 模长为5，幅角为π/4的复数
        """
        real = modulus * math.cos(argument)
        imaginary = modulus * math.sin(argument)
        return Complex(real, imaginary)
    
    @staticmethod
    def from_string(s: str) -> 'Complex':
        """
        从字符串创建复数
        
        参数:
            s: 字符串表示，如 "3+4i", "2-3i", "5", "2i"
            
        返回:
            Complex: 对应的复数
        """
        s = s.replace(" ", "").replace("j", "i")
        
        if "i" not in s:
            # 纯实数
            return Complex(float(s), 0)
        
        if s == "i":
            return Complex(0, 1)
        elif s == "-i":
            return Complex(0, -1)
        
        # 解析复数
        if "+" in s:
            parts = s.split("+")
            if len(parts) == 2:
                real_part = float(parts[0])
                imag_str = parts[1].replace("i", "")
                imag_part = 1.0 if imag_str == "" else float(imag_str)
                return Complex(real_part, imag_part)
        elif "-" in s[1:]:  # 不考虑开头的负号
            minus_idx = s.rfind("-")
            real_part = float(s[:minus_idx])
            imag_str = s[minus_idx+1:].replace("i", "")
            imag_part = -(1.0 if imag_str == "" else float(imag_str))
            return Complex(real_part, imag_part)
        else:
            # 纯虚数
            imag_str = s.replace("i", "")
            if imag_str == "" or imag_str == "+":
                imag_part = 1.0
            elif imag_str == "-":
                imag_part = -1.0
            else:
                imag_part = float(imag_str)
            return Complex(0, imag_part)
        
        raise ValueError(f"无法解析复数字符串: {s}")
    
    def power(self, exponent: Union['Complex', float, int]) -> 'Complex':
        """
        复数的幂运算
        
        参数:
            exponent: 指数，可以是复数或实数
            
        返回:
            Complex: z^exponent
            
        公式:
            z^w = e^(w * ln(z)) = e^(w * (ln|z| + i*arg(z)))
        """
        if isinstance(exponent, (int, float)):
            if exponent == 0:
                return Complex(1, 0)
            elif exponent == 1:
                return Complex(self._real, self._imaginary)
            elif exponent == 2:
                return self * self
            else:
                # 使用极坐标形式计算
                r = self.modulus
                theta = self.argument
                
                if r == 0:
                    return Complex(0, 0)
                
                new_r = r ** exponent
                new_theta = theta * exponent
                
                return Complex.from_polar(new_r, new_theta)
        else:
            # 复数指数
            if self._real == 0 and self._imaginary == 0:
                return Complex(0, 0)
            
            ln_z = Complex(math.log(self.modulus), self.argument)
            w_ln_z = exponent * ln_z
            
            exp_real = math.exp(w_ln_z._real)
            return Complex(
                exp_real * math.cos(w_ln_z._imaginary),
                exp_real * math.sin(w_ln_z._imaginary)
            )
    
    def __pow__(self, exponent: Union['Complex', float, int]) -> 'Complex':
        """幂运算操作符"""
        return self.power(exponent)
    
    def sqrt(self) -> 'Complex':
        """
        复数的平方根
        
        返回:
            Complex: 平方根（主值）
        """
        if self._real == 0 and self._imaginary == 0:
            return Complex(0, 0)
        
        r = self.modulus
        theta = self.argument
        
        new_r = math.sqrt(r)
        new_theta = theta / 2
        
        return Complex.from_polar(new_r, new_theta)
    
    def exp(self) -> 'Complex':
        """
        复数的指数函数
        
        返回:
            Complex: e^z = e^(real + i*imaginary) = e^real * (cos(imaginary) + i*sin(imaginary))
        """
        exp_real = math.exp(self._real)
        return Complex(
            exp_real * math.cos(self._imaginary),
            exp_real * math.sin(self._imaginary)
        )
    
    def log(self) -> 'Complex':
        """
        复数的自然对数
        
        返回:
            Complex: ln(z) = ln|z| + i*arg(z)
        """
        if self._real == 0 and self._imaginary == 0:
            raise ValueError("不能计算零的对数")
        
        return Complex(math.log(self.modulus), self.argument)
    
    def sin(self) -> 'Complex':
        """
        复数的正弦函数
        
        返回:
            Complex: sin(z) = sin(real)*cosh(imaginary) + i*cos(real)*sinh(imaginary)
        """
        return Complex(
            math.sin(self._real) * math.cosh(self._imaginary),
            math.cos(self._real) * math.sinh(self._imaginary)
        )
    
    def cos(self) -> 'Complex':
        """
        复数的余弦函数
        
        返回:
            Complex: cos(z) = cos(real)*cosh(imaginary) - i*sin(real)*sinh(imaginary)
        """
        return Complex(
            math.cos(self._real) * math.cosh(self._imaginary),
            -math.sin(self._real) * math.sinh(self._imaginary)
        )
    
    def tan(self) -> 'Complex':
        """
        复数的正切函数
        
        返回:
            Complex: tan(z) = sin(z) / cos(z)
        """
        return self.sin() / self.cos()


# 便捷函数
def complex_from_string(s: str) -> Complex:
    """从字符串创建复数的便捷函数"""
    return Complex.from_string(s)

def complex_from_polar(modulus: float, argument: float) -> Complex:
    """从极坐标创建复数的便捷函数"""
    return Complex.from_polar(modulus, argument)

def cexp(z: Union[Complex, complex, float]) -> Complex:
    """复数指数函数"""
    if isinstance(z, Complex):
        return z.exp()
    elif isinstance(z, complex):
        z_complex = Complex(z.real, z.imag)
        return z_complex.exp()
    else:
        return Complex(math.exp(float(z)), 0)

def clog(z: Union[Complex, complex, float]) -> Complex:
    """复数对数函数"""
    if isinstance(z, Complex):
        return z.log()
    elif isinstance(z, complex):
        z_complex = Complex(z.real, z.imag)
        return z_complex.log()
    else:
        if z <= 0:
            raise ValueError("实数对数的参数必须为正")
        return Complex(math.log(float(z)), 0)

def csqrt(z: Union[Complex, complex, float]) -> Complex:
    """复数平方根函数"""
    if isinstance(z, Complex):
        return z.sqrt()
    elif isinstance(z, complex):
        z_complex = Complex(z.real, z.imag)
        return z_complex.sqrt()
    else:
        z_float = float(z)
        if z_float >= 0:
            return Complex(math.sqrt(z_float), 0)
        else:
            return Complex(0, math.sqrt(-z_float)) 