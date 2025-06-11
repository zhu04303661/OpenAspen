"""
随机数生成模块
==============

包含各种伪随机数生成器和随机分布。

子模块:
- base: 随机数生成器基类
- mersenne_twister: 梅森旋转算法
- xorshift: XorShift算法
- kiss: KISS算法
- cmwc4096: CMWC4096算法
- distributions: 各种概率分布
"""

from .mersenne_twister import MersenneTwister

__all__ = ['MersenneTwister']

try:
    from .xorshift import XorShift
    __all__.append('XorShift')
except ImportError:
    pass

try:
    from .kiss import KISS
    __all__.append('KISS')
except ImportError:
    pass 