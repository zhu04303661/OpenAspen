"""
DWSIM 单元操作模块
================

包含DWSIM单元操作的Python实现，包括：
- 基础单元操作类
- 混合器、分离器等基本操作
- 热交换器、反应器等复杂操作
- CAPE-OPEN兼容接口
- 与FlowsheetSolver的完整集成

从原VB.NET版本1:1转换而来，保持功能完全一致。
"""

# 导入基础类
from .base_classes import (
    UnitOpBaseClass, 
    SpecialOpBaseClass,
    SimulationObjectClass,
    ConnectionPoint,
    GraphicObject,
    SpecialOpObjectInfo
)

# 导入具体单元操作
from .unit_operations import (
    Mixer, Splitter, Heater, Cooler, 
    HeatExchanger, Pump, Compressor, Valve,
    ComponentSeparator, Filter, Vessel, Tank,
    PressureBehavior, PhaseProperties, CompoundData
)

# 导入集成功能
from .integration import (
    UnitOperationRegistry,
    IntegratedFlowsheetSolver,
    create_integrated_solver,
    register_custom_operation,
    global_unit_operation_registry
)

# 公开的API
__all__ = [
    # 基础类
    'UnitOpBaseClass', 'SpecialOpBaseClass', 'SimulationObjectClass',
    'ConnectionPoint', 'GraphicObject', 'SpecialOpObjectInfo',
    
    # 具体单元操作
    'Mixer', 'Splitter', 'Heater', 'Cooler',
    'HeatExchanger', 'Pump', 'Compressor', 'Valve',
    'ComponentSeparator', 'Filter', 'Vessel', 'Tank',
    
    # 数据类和枚举
    'PressureBehavior', 'PhaseProperties', 'CompoundData',
    
    # 集成功能
    'UnitOperationRegistry', 'IntegratedFlowsheetSolver',
    'create_integrated_solver', 'register_custom_operation',
    'global_unit_operation_registry'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "DWSIM Team"
__description__ = "DWSIM单元操作Python实现，与FlowsheetSolver完美集成"

# 快捷函数
def get_available_operations():
    """
    获取所有可用的单元操作类型
    
    Returns:
        List[str]: 操作类型列表
    """
    return global_unit_operation_registry.get_available_operations()


def create_operation(operation_type: str, name: str = "", description: str = ""):
    """
    创建单元操作实例
    
    Args:
        operation_type: 操作类型
        name: 操作名称
        description: 操作描述
        
    Returns:
        UnitOpBaseClass: 创建的单元操作实例
    """
    return global_unit_operation_registry.create_operation(operation_type, name, description)


def create_solver_with_operations(*operations):
    """
    创建带有指定操作的集成求解器
    
    Args:
        *operations: 操作配置元组列表 (type, name, description)
        
    Returns:
        IntegratedFlowsheetSolver: 配置好的求解器
        
    Example:
        solver = create_solver_with_operations(
            ("Mixer", "MIX-001", "混合器"),
            ("Heater", "HX-001", "加热器"),
            ("Pump", "P-001", "泵")
        )
    """
    solver = create_integrated_solver()
    
    for operation_config in operations:
        if len(operation_config) >= 2:
            op_type = operation_config[0]
            name = operation_config[1]
            description = operation_config[2] if len(operation_config) > 2 else ""
            
            solver.create_and_add_operation(op_type, name, description)
    
    return solver


# 模块信息
def get_module_info():
    """
    获取模块信息
    
    Returns:
        dict: 模块信息字典
    """
    return {
        'name': 'dwsim_operations',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'available_operations': get_available_operations(),
        'total_operations': len(get_available_operations())
    } 