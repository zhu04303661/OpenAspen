"""
DWSIM5 Python重新实现 - 流程图求解器包
====================================

这个包提供了DWSIM5流程图求解器的Python重新实现，包括：
- 流程图求解算法
- 收敛求解器
- 远程求解器客户端
- 任务调度器
- 异常处理

主要组件：
- FlowsheetSolver: 主要的流程图求解器类
- CalculationArgs: 计算参数数据类
- ConvergenceSolver: 收敛算法实现
- RemoteSolverClient: 远程求解器客户端
"""

__version__ = "1.0.0"
__author__ = "DWSIM Python Team"
__email__ = "dwsim-python@example.com"

# 延迟导入以避免循环导入
def __getattr__(name):
    """延迟导入机制"""
    if name == "FlowsheetSolver":
        from .solver import FlowsheetSolver
        return FlowsheetSolver
    elif name == "SolverSettings":
        from .solver import SolverSettings
        return SolverSettings
    elif name == "CalculationArgs":
        from .calculation_args import CalculationArgs
        return CalculationArgs
    elif name == "ObjectType":
        from .calculation_args import ObjectType
        return ObjectType
    elif name == "CalculationStatus":
        from .calculation_args import CalculationStatus
        return CalculationStatus
    elif name == "BroydenSolver":
        from .convergence_solver import BroydenSolver
        return BroydenSolver
    elif name == "NewtonRaphsonSolver":
        from .convergence_solver import NewtonRaphsonSolver
        return NewtonRaphsonSolver
    elif name == "RecycleConvergenceSolver":
        from .convergence_solver import RecycleConvergenceSolver
        return RecycleConvergenceSolver
    elif name == "SimultaneousAdjustSolver":
        from .convergence_solver import SimultaneousAdjustSolver
        return SimultaneousAdjustSolver
    elif name == "AzureSolverClient":
        from .remote_solvers import AzureSolverClient
        return AzureSolverClient
    elif name == "TCPSolverClient":
        from .remote_solvers import TCPSolverClient
        return TCPSolverClient
    elif name == "SolverException":
        from .solver_exceptions import SolverException
        return SolverException
    elif name == "ConvergenceException":
        from .solver_exceptions import ConvergenceException
        return ConvergenceException
    elif name == "TimeoutException":
        from .solver_exceptions import TimeoutException
        return TimeoutException
    elif name == "CalculationException":
        from .solver_exceptions import CalculationException
        return CalculationException
    elif name == "NetworkException":
        from .solver_exceptions import NetworkException
        return NetworkException
    elif name == "DataException":
        from .solver_exceptions import DataException
        return DataException
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 定义公共API
__all__ = [
    "FlowsheetSolver",
    "SolverSettings", 
    "CalculationArgs",
    "ObjectType",
    "CalculationStatus",
    "BroydenSolver",
    "NewtonRaphsonSolver", 
    "RecycleConvergenceSolver",
    "SimultaneousAdjustSolver",
    "AzureSolverClient",
    "TCPSolverClient",
    "SolverException",
    "ConvergenceException",
    "TimeoutException",
    "CalculationException",
    "NetworkException",
    "DataException"
] 