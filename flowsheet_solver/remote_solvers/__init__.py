"""
远程求解器包
===========

提供Azure云计算和TCP网络分布式计算的客户端实现。
"""

from .azure_solver_client import AzureSolverClient
from .tcp_solver_client import TCPSolverClient

__all__ = [
    "AzureSolverClient",
    "TCPSolverClient"
] 