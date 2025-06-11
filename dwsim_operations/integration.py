"""
DWSIM 单元操作与FlowsheetSolver集成模块
====================================

将单元操作模块与现有的FlowsheetSolver进行完美衔接，实现：
- 单元操作注册和管理
- 与flowsheet_solver的无缝集成
- 计算参数传递和状态同步
- 事件处理和错误管理

从原VB.NET版本1:1转换的Python实现。
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass

# 导入现有的flowsheet_solver模块
try:
    from ..flowsheet_solver.solver import FlowsheetSolver
    from ..flowsheet_solver.calculation_args import CalculationArgs, ObjectType
    from ..flowsheet_solver.solver_exceptions import *
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from flowsheet_solver.solver import FlowsheetSolver
    from flowsheet_solver.calculation_args import CalculationArgs, ObjectType
    from flowsheet_solver.solver_exceptions import *

# 导入单元操作模块
from .base_classes import UnitOpBaseClass, SpecialOpBaseClass
from .unit_operations import (
    Mixer, Splitter, Heater, Cooler, HeatExchanger, 
    Pump, Compressor, Valve, ComponentSeparator, 
    Filter, Vessel, Tank
)


@dataclass
class UnitOperationRegistry:
    """
    单元操作注册表
    
    管理所有可用的单元操作类型，提供创建和查找功能。
    """
    
    def __init__(self):
        """初始化注册表"""
        self._registry: Dict[str, Type[UnitOpBaseClass]] = {}
        self.logger = logging.getLogger(__name__)
        self._register_default_operations()
    
    def _register_default_operations(self):
        """注册默认的单元操作"""
        default_operations = {
            'Mixer': Mixer,
            'Splitter': Splitter,
            'Heater': Heater,
            'Cooler': Cooler,
            'HeatExchanger': HeatExchanger,
            'Pump': Pump,
            'Compressor': Compressor,
            'Valve': Valve,
            'ComponentSeparator': ComponentSeparator,
            'Filter': Filter,
            'Vessel': Vessel,
            'Tank': Tank
        }
        
        for name, cls in default_operations.items():
            self.register_operation(name, cls)
    
    def register_operation(self, name: str, operation_class: Type[UnitOpBaseClass]):
        """
        注册单元操作类
        
        Args:
            name: 操作名称
            operation_class: 操作类
        """
        if not issubclass(operation_class, UnitOpBaseClass):
            raise ValueError(f"操作类 {operation_class} 必须继承自 UnitOpBaseClass")
        
        self._registry[name] = operation_class
        self.logger.info(f"已注册单元操作: {name}")
    
    def create_operation(self, operation_type: str, name: str = "", description: str = "") -> UnitOpBaseClass:
        """
        创建单元操作实例
        
        Args:
            operation_type: 操作类型
            name: 操作名称
            description: 操作描述
            
        Returns:
            UnitOpBaseClass: 创建的单元操作实例
        """
        if operation_type not in self._registry:
            raise ValueError(f"未知的单元操作类型: {operation_type}")
        
        operation_class = self._registry[operation_type]
        return operation_class(name, description)
    
    def get_available_operations(self) -> List[str]:
        """
        获取所有可用的单元操作类型
        
        Returns:
            List[str]: 操作类型列表
        """
        return list(self._registry.keys())
    
    def is_registered(self, operation_type: str) -> bool:
        """
        检查操作类型是否已注册
        
        Args:
            operation_type: 操作类型
            
        Returns:
            bool: 是否已注册
        """
        return operation_type in self._registry


class IntegratedFlowsheetSolver(FlowsheetSolver):
    """
    集成的FlowsheetSolver
    
    扩展原有的FlowsheetSolver，添加对DWSIM单元操作的支持。
    """
    
    def __init__(self, settings=None):
        """
        初始化集成求解器
        
        Args:
            settings: 求解器设置
        """
        super().__init__(settings)
        
        # 单元操作注册表
        self.unit_operation_registry = UnitOperationRegistry()
        
        # 单元操作实例字典
        self.unit_operations: Dict[str, UnitOpBaseClass] = {}
        
        # 设置事件处理器
        self._setup_unit_operation_events()
        
        self.logger.info("已初始化集成FlowsheetSolver")
    
    def _setup_unit_operation_events(self):
        """设置单元操作相关的事件处理器"""
        
        def on_unit_op_calculation_started(obj_name: str):
            """单元操作计算开始事件处理"""
            self.logger.debug(f"单元操作 {obj_name} 开始计算")
        
        def on_unit_op_calculation_finished(obj_name: str, success: bool):
            """单元操作计算完成事件处理"""
            status = "成功" if success else "失败"
            self.logger.debug(f"单元操作 {obj_name} 计算{status}")
        
        # 注册事件处理器
        self.add_event_handler('unit_op_calculation_started', on_unit_op_calculation_started)
        self.add_event_handler('unit_op_calculation_finished', on_unit_op_calculation_finished)
    
    def add_unit_operation(self, operation: UnitOpBaseClass, flowsheet: Any = None):
        """
        添加单元操作到求解器
        
        Args:
            operation: 单元操作实例
            flowsheet: 流程图对象（可选）
        """
        if not isinstance(operation, UnitOpBaseClass):
            raise TypeError("operation必须是UnitOpBaseClass的实例")
        
        # 设置流程图引用
        if flowsheet:
            operation.flowsheet = flowsheet
        
        # 添加到单元操作字典
        self.unit_operations[operation.name] = operation
        
        self.logger.info(f"已添加单元操作: {operation.name} ({operation.__class__.__name__})")
    
    def remove_unit_operation(self, operation_name: str):
        """
        移除单元操作
        
        Args:
            operation_name: 操作名称
        """
        if operation_name in self.unit_operations:
            del self.unit_operations[operation_name]
            self.logger.info(f"已移除单元操作: {operation_name}")
        else:
            self.logger.warning(f"单元操作 {operation_name} 不存在")
    
    def create_and_add_operation(self, operation_type: str, name: str, 
                                description: str = "", flowsheet: Any = None) -> UnitOpBaseClass:
        """
        创建并添加单元操作
        
        Args:
            operation_type: 操作类型
            name: 操作名称
            description: 操作描述
            flowsheet: 流程图对象
            
        Returns:
            UnitOpBaseClass: 创建的单元操作
        """
        operation = self.unit_operation_registry.create_operation(operation_type, name, description)
        self.add_unit_operation(operation, flowsheet)
        return operation
    
    def calculate_unit_operation(self, operation_name: str, args: Optional[CalculationArgs] = None) -> bool:
        """
        计算指定的单元操作
        
        Args:
            operation_name: 操作名称
            args: 计算参数
            
        Returns:
            bool: 计算是否成功
        """
        if operation_name not in self.unit_operations:
            raise ValueError(f"单元操作 {operation_name} 不存在")
        
        operation = self.unit_operations[operation_name]
        
        try:
            # 触发计算开始事件
            self.fire_event('unit_op_calculation_started', operation_name)
            
            # 执行计算
            operation.solve()
            
            # 触发计算完成事件
            self.fire_event('unit_op_calculation_finished', operation_name, True)
            
            return True
            
        except Exception as e:
            operation.error_message = str(e)
            self.logger.error(f"单元操作 {operation_name} 计算失败: {e}")
            
            # 触发计算完成事件（失败）
            self.fire_event('unit_op_calculation_finished', operation_name, False)
            
            return False
    
    def calculate_all_operations(self, in_dependency_order: bool = True) -> Dict[str, bool]:
        """
        计算所有单元操作
        
        Args:
            in_dependency_order: 是否按依赖顺序计算
            
        Returns:
            Dict[str, bool]: 每个操作的计算结果
        """
        results = {}
        
        operations_to_calculate = list(self.unit_operations.keys())
        
        if in_dependency_order:
            # 这里应该实现依赖关系排序
            # 暂时使用简单顺序
            pass
        
        for op_name in operations_to_calculate:
            results[op_name] = self.calculate_unit_operation(op_name)
        
        return results
    
    def get_operation_by_name(self, name: str) -> Optional[UnitOpBaseClass]:
        """
        按名称获取单元操作
        
        Args:
            name: 操作名称
            
        Returns:
            Optional[UnitOpBaseClass]: 单元操作实例
        """
        return self.unit_operations.get(name)
    
    def get_operations_by_type(self, operation_type: Type[UnitOpBaseClass]) -> List[UnitOpBaseClass]:
        """
        按类型获取单元操作列表
        
        Args:
            operation_type: 操作类型
            
        Returns:
            List[UnitOpBaseClass]: 操作列表
        """
        return [op for op in self.unit_operations.values() if isinstance(op, operation_type)]
    
    def validate_all_operations(self) -> Dict[str, List[str]]:
        """
        验证所有单元操作
        
        Returns:
            Dict[str, List[str]]: 验证错误字典，键为操作名称，值为错误消息列表
        """
        validation_errors = {}
        
        for name, operation in self.unit_operations.items():
            errors = []
            try:
                operation.validate()
            except Exception as e:
                errors.append(str(e))
            
            if errors:
                validation_errors[name] = errors
        
        return validation_errors
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """
        获取计算摘要
        
        Returns:
            Dict[str, Any]: 计算摘要信息
        """
        total_operations = len(self.unit_operations)
        calculated_operations = sum(1 for op in self.unit_operations.values() if op.calculated)
        error_operations = sum(1 for op in self.unit_operations.values() if op.error_message)
        
        return {
            'total_operations': total_operations,
            'calculated_operations': calculated_operations,
            'error_operations': error_operations,
            'calculation_rate': calculated_operations / total_operations if total_operations > 0 else 0,
            'operations_by_type': self._get_operations_by_type_summary()
        }
    
    def _get_operations_by_type_summary(self) -> Dict[str, int]:
        """获取按类型分组的操作数量摘要"""
        type_summary = {}
        for operation in self.unit_operations.values():
            op_type = operation.__class__.__name__
            type_summary[op_type] = type_summary.get(op_type, 0) + 1
        return type_summary
    
    def reset_all_calculations(self):
        """重置所有计算状态"""
        for operation in self.unit_operations.values():
            operation.de_calculate()
        
        self.logger.info("已重置所有单元操作的计算状态")
    
    def export_operations_config(self) -> Dict[str, Any]:
        """
        导出单元操作配置
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        config = {
            'operations': {},
            'metadata': {
                'total_count': len(self.unit_operations),
                'export_time': str(self.performance_stats.get('total_time', 0))
            }
        }
        
        for name, operation in self.unit_operations.items():
            config['operations'][name] = {
                'type': operation.__class__.__name__,
                'tag': operation.tag,
                'description': operation.description,
                'calculated': operation.calculated,
                'error_message': operation.error_message
            }
        
        return config
    
    def import_operations_config(self, config: Dict[str, Any]):
        """
        导入单元操作配置
        
        Args:
            config: 配置字典
        """
        if 'operations' not in config:
            raise ValueError("无效的配置格式")
        
        for name, op_config in config['operations'].items():
            operation_type = op_config.get('type')
            if not operation_type:
                continue
            
            try:
                operation = self.unit_operation_registry.create_operation(
                    operation_type,
                    name,
                    op_config.get('description', '')
                )
                
                operation.tag = op_config.get('tag', '')
                self.add_unit_operation(operation)
                
            except Exception as e:
                self.logger.error(f"导入操作 {name} 失败: {e}")


# 全局注册表实例
global_unit_operation_registry = UnitOperationRegistry()


def create_integrated_solver(settings=None) -> IntegratedFlowsheetSolver:
    """
    创建集成求解器的便捷函数
    
    Args:
        settings: 求解器设置
        
    Returns:
        IntegratedFlowsheetSolver: 集成求解器实例
    """
    return IntegratedFlowsheetSolver(settings)


def register_custom_operation(name: str, operation_class: Type[UnitOpBaseClass]):
    """
    注册自定义单元操作
    
    Args:
        name: 操作名称
        operation_class: 操作类
    """
    global_unit_operation_registry.register_operation(name, operation_class) 