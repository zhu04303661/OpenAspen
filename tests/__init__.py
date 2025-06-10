"""
DWSIM FlowsheetSolver 测试套件
=============================

完整的测试覆盖，包括：
- 单元测试 (unit/)
- 集成测试 (integration/)  
- 性能测试 (performance/)
- 测试夹具 (fixtures/)

测试目标：
1. FlowsheetSolver主求解器功能
2. 收敛算法实现
3. 远程求解器客户端
4. 异常处理系统
5. 数据结构和枚举
6. 事件系统
7. 性能基准测试
"""

import sys
import os

# 添加源码路径到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flowsheet_solver'))

__version__ = "1.0.0"
__author__ = "DWSIM Python Team" 