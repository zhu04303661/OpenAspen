"""
数值计算模块
============

包含各种数值计算功能，如复数运算、线性代数、常微分方程求解等。

子模块:
- complex_number: 复数运算
- linear_algebra: 线性代数
- ode: 常微分方程求解器
"""

from .complex_number import Complex

try:
    from . import linear_algebra
    _has_linear_algebra = True
except ImportError:
    _has_linear_algebra = False

try:
    from . import ode
    _has_ode = True
except ImportError:
    _has_ode = False

__all__ = ['Complex']
if _has_linear_algebra:
    __all__.append('linear_algebra')
if _has_ode:
    __all__.append('ode') 