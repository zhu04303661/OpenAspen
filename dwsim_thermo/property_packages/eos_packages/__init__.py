"""
状态方程包模块 (Equation of State Packages)

该模块包含各种状态方程的实现，用于计算气液平衡和热力学性质。

包含的状态方程:
- PC-SAFT: 统计缔合流体理论 (Perturbed-Chain Statistical Associating Fluid Theory)
- Lee-Kesler-Plocker: 基于对应态原理的状态方程
- Peng-Robinson-Stryjek-Vera: 改进的PR状态方程
- SRK: Soave-Redlich-Kwong (已在上级目录实现)
- PR: Peng-Robinson (已在上级目录实现)

作者: OpenAspen项目组
版本: 1.0.0
"""

from .pc_saft import PCSAFT

# 尝试导入新增的状态方程
try:
    from .lee_kesler_plocker import (
        LeeKeslerPlocknerEOS, 
        LeeKeslerPlocknerParameters, 
        LeeKeslerPlocknerPropertyPackage
    )
    LEE_KESLER_AVAILABLE = True
except ImportError:
    LEE_KESLER_AVAILABLE = False

try:
    from .peng_robinson_stryjek_vera import (
        PengRobinsonStryjekVeraEOS,
        PRSVParameters,
        PRSVPropertyPackage
    )
    PRSV_AVAILABLE = True
except ImportError:
    PRSV_AVAILABLE = False

__all__ = ['PCSAFT']

if LEE_KESLER_AVAILABLE:
    __all__.extend([
        'LeeKeslerPlocknerEOS',
        'LeeKeslerPlocknerParameters',
        'LeeKeslerPlocknerPropertyPackage'
    ])

if PRSV_AVAILABLE:
    __all__.extend([
        'PengRobinsonStryjekVeraEOS',
        'PRSVParameters',
        'PRSVPropertyPackage'
    ]) 