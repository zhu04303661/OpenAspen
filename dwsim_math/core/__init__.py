"""
核心数学模块
============

包含基本的数学运算和工具函数，是其他模块的基础。

子模块:
- general: 通用数学函数
- matrix_ops: 矩阵操作
- interpolation: 插值算法
- extrapolation: 外推算法
- intersection: 交点计算
"""

from .general import MathCommon
from .matrix_ops import MatrixOperations, Determinant, Inverse
from .interpolation import Interpolation
 
# 尝试导入可能尚未实现的模块
try:
    from .extrapolation import Extrapolation
except ImportError:
    pass

try:
    from .intersection import Intersection
except ImportError:
    pass

__all__ = [
    'MathCommon',
    'MatrixOperations',
    'Determinant', 
    'Inverse',
    'Interpolation'
] 