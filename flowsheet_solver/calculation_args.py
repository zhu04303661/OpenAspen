"""
计算参数类定义
============

定义计算过程中传递的参数对象，包含对象信息、计算状态等。
对应原VB.NET版本中的CalculationArgs类。
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ObjectType(Enum):
    """
    对象类型枚举
    
    定义流程图中各种对象的类型，用于计算调度和分派。
    """
    # 大写名称（测试期望的主要名称）
    MATERIAL_STREAM = "MaterialStream"
    ENERGY_STREAM = "EnergyStream"
    UNITOPERATION = "UnitOperation"
    RECYCLE = "Recycle"
    ENERGYRECYCLE = "EnergyRecycle"
    SPECIFICATION = "Specification"
    ADJUST = "Adjust"
    SPEC = "Spec"
    UNKNOWN = "Unknown"
    
    # 小写别名（向后兼容）
    MaterialStream = "MaterialStream"
    EnergyStream = "EnergyStream"
    UnitOperation = "UnitOperation"
    Recycle = "Recycle"
    EnergyRecycle = "EnergyRecycle"
    Specification = "Specification"
    Adjust = "Adjust"
    Spec = "Spec"
    Unknown = "Unknown"


@dataclass
class CalculationArgs:
    """
    计算参数类
    
    封装计算过程中需要传递的参数信息，包括对象标识、计算状态、
    发送者信息等。用于在计算队列中传递对象信息。
    
    Attributes:
        name: 对象名称（唯一标识符）
        tag: 对象标签（显示名称）
        object_type: 对象类型
        sender: 发送者标识（如"FlowsheetSolver", "Adjust"等）
        calculated: 计算状态标志
        error_message: 错误信息
        calculation_time: 计算耗时（秒）
        iteration_count: 迭代次数
        priority: 计算优先级
    """
    
    name: str = ""
    tag: str = ""
    object_type: ObjectType = ObjectType.Unknown
    sender: str = ""
    calculated: bool = False
    error_message: str = ""
    calculation_time: float = 0.0
    iteration_count: int = 0
    priority: int = 0
    
    def __post_init__(self):
        """
        初始化后处理
        
        确保对象类型为ObjectType枚举，如果传入字符串则自动转换。
        """
        if isinstance(self.object_type, str):
            try:
                self.object_type = ObjectType(self.object_type)
            except ValueError:
                self.object_type = ObjectType.Unknown
    
    def reset_calculation_state(self):
        """
        重置计算状态
        
        清除计算相关的状态信息，准备进行新的计算。
        """
        self.calculated = False
        self.error_message = ""
        self.calculation_time = 0.0
        self.iteration_count = 0
    
    def set_error(self, error_message: str):
        """
        设置错误信息
        
        Args:
            error_message: 错误消息
        """
        self.error_message = error_message
        self.calculated = False
    
    def set_success(self, calculation_time: float = 0.0, iteration_count: int = 0):
        """
        设置计算成功状态
        
        Args:
            calculation_time: 计算耗时
            iteration_count: 迭代次数
        """
        self.calculated = True
        self.error_message = ""
        self.calculation_time = calculation_time
        self.iteration_count = iteration_count
    
    def copy(self) -> 'CalculationArgs':
        """
        创建副本
        
        Returns:
            CalculationArgs: 当前对象的副本
        """
        return CalculationArgs(
            name=self.name,
            tag=self.tag,
            object_type=self.object_type,
            sender=self.sender,
            calculated=self.calculated,
            error_message=self.error_message,
            calculation_time=self.calculation_time,
            iteration_count=self.iteration_count,
            priority=self.priority
        )
    
    def is_valid(self) -> bool:
        """
        检查参数有效性
        
        Returns:
            bool: 参数是否有效
        """
        return bool(self.name and self.object_type != ObjectType.Unknown)
    
    def __str__(self) -> str:
        """
        字符串表示
        
        Returns:
            str: 对象的字符串描述
        """
        status = "已计算" if self.calculated else "未计算"
        if self.error_message:
            status = f"错误: {self.error_message}"
            
        return (f"CalculationArgs(name='{self.name}', tag='{self.tag}', "
                f"type={self.object_type.value}, sender='{self.sender}', "
                f"status={status})")
    
    def __repr__(self) -> str:
        """
        详细表示
        
        Returns:
            str: 对象的详细字符串描述
        """
        return self.__str__()


class CalculationStatus(Enum):
    """
    计算状态枚举
    
    定义计算过程中的各种状态。
    """
    NOTSTARTED = "NotStarted"      # 未开始
    RUNNING = "Running"            # 运行中
    COMPLETED = "Completed"        # 已完成
    ERROR = "Error"                # 错误
    CANCELLED = "Cancelled"        # 已取消
    TIMEOUT = "Timeout"            # 超时
    
    # 为了向后兼容，保留原来的名称
    NotStarted = "NotStarted"
    Running = "Running"
    Completed = "Completed"
    Error = "Error"
    Cancelled = "Cancelled"
    Timeout = "Timeout" 