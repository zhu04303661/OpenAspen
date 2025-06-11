"""
优化模块
========

包含各种优化算法，如L-BFGS、Levenberg-Marquardt、Brent最小化等。

子模块:
- lbfgs: L-BFGS优化算法
- lbfgsb: L-BFGS-B有界优化算法
- lm: Levenberg-Marquardt算法
- lm_fit: LM拟合算法
- brent_minimize: Brent最小化算法
- gdem: 全局差分进化算法
"""

from .lbfgs import LBFGS

try:
    from .brent_minimize import BrentMinimizer
    __all__ = ['LBFGS', 'BrentMinimizer']
except ImportError:
    # brent_minimize模块暂未实现
    __all__ = ['LBFGS'] 