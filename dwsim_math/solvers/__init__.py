"""
求解器模块
==========

包含各种数值求解算法，如方程求根、线性方程组求解、非线性方程组求解等。

子模块:
- brent: Brent方法求根
- broyden: Broyden方法求解非线性方程组
- linear_system: 线性方程组求解器
- lm: Levenberg-Marquardt算法
- lm_fit: LM拟合算法
"""

from .brent import BrentSolver

try:
    from .broyden import BroydenSolver
    __all__ = ['BrentSolver', 'BroydenSolver']
except ImportError:
    # broyden模块暂未实现
    __all__ = ['BrentSolver'] 