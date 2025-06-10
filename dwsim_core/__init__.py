"""
DWSIM Python - 核心架构层
基于DWSIM5的Python重新实现

提供基础接口定义、异常处理和工具类
"""

__version__ = "1.0.0"
__author__ = "DWSIM Python Team"

from . import interfaces
from . import exceptions  
from . import utilities

__all__ = ["interfaces", "exceptions", "utilities"] 