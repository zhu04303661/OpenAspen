"""异常处理模块"""

from .thermodynamic_errors import *
from .convergence_errors import *

__all__ = [
    "ThermodynamicError",
    "ComponentNotFoundError", 
    "PropertyCalculationError",
    "FlashCalculationError",
    "ConvergenceError",
    "MaxIterationsError",
    "SolutionNotFoundError"
] 