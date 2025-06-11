"""
DWSIM数学计算库 (Python版本)
============================

这是DWSIM（开源过程仿真器）数学计算库的Python版本，从原来的VB.NET和C#代码完整转换而来。
该库提供了过程仿真中所需的各种数学算法和数值计算功能。

主要模块:
- core: 核心数学函数
- solvers: 方程求解器
- optimization: 优化算法
- numerics: 数值计算
- random: 随机数生成
- swarm: 群体智能优化
- special: 特殊函数

作者: DWSIM团队
许可证: GNU General Public License v3.0
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DWSIM Team"
__license__ = "GPL-3.0"

# 导入主要子模块
from . import core
from . import solvers
from . import optimization
from . import numerics
from . import random
try:
    from . import swarm
except ImportError:
    pass  # swarm模块可能尚未实现
try:
    from . import special
except ImportError:
    pass  # special模块可能尚未实现

# 导入常用类和函数
from .core.general import MathCommon
from .core.matrix_ops import MatrixOperations, Determinant, Inverse
from .core.interpolation import Interpolation
from .numerics.complex_number import Complex
from .solvers.brent import BrentSolver
from .optimization.lbfgs import LBFGS
from .random.mersenne_twister import MersenneTwister

# 尝试导入线性代数模块
try:
    from .numerics.linear_algebra.matrix import Matrix
    from .numerics.linear_algebra.vector import Vector
except ImportError:
    # 如果模块尚未创建，则不导入
    pass

__all__ = [
    # 子模块
    'core',
    'solvers', 
    'optimization',
    'numerics',
    'random',
    
    # 主要类
    'MathCommon',
    'MatrixOperations',
    'Determinant',
    'Inverse',
    'Interpolation',
    'Complex',
    'BrentSolver',
    'LBFGS',
    'MersenneTwister'
] 