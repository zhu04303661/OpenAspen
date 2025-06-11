"""
远程求解器包
===========

提供Azure云计算和TCP网络分布式计算的客户端实现。
"""

from .azure_solver_client import AzureSolverClient
from .tcp_solver_client import TCPSolverClient

class RemoteSolverManager:
    """远程求解器管理器，用于管理多个远程求解器服务器"""
    
    def __init__(self):
        self.servers = {}
        
    def add_solver_server(self, name: str, address: str, port: int):
        """添加求解器服务器"""
        self.servers[name] = {"address": address, "port": port, "available": False}
        
    def get_available_servers(self):
        """获取可用的服务器列表"""
        return [name for name, info in self.servers.items() if info.get("available", False)]

__all__ = [
    "AzureSolverClient",
    "TCPSolverClient",
    "RemoteSolverManager"
] 