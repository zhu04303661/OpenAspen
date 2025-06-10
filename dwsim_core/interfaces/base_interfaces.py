"""
基础接口定义
定义系统中所有核心组件的接口契约
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import numpy as np

class PhaseType(Enum):
    """相态类型枚举"""
    VAPOR = "vapor"
    LIQUID = "liquid"
    SOLID = "solid"
    AQUEOUS = "aqueous"

class PropertyType(Enum):
    """物性类型枚举"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    ENTHALPY = "enthalpy"
    ENTROPY = "entropy"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    VAPOR_PRESSURE = "vapor_pressure"
    FUGACITY_COEFFICIENT = "fugacity_coefficient"
    ACTIVITY_COEFFICIENT = "activity_coefficient"
    COMPRESSIBILITY_FACTOR = "compressibility_factor"
    HEAT_CAPACITY = "heat_capacity"
    MOLECULAR_WEIGHT = "molecular_weight"

class IComponent(ABC):
    """组分接口"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """组分名称"""
        pass
    
    @property
    @abstractmethod
    def cas_number(self) -> str:
        """CAS号"""
        pass
    
    @property
    @abstractmethod
    def molecular_weight(self) -> float:
        """分子量 (g/mol)"""
        pass
    
    @property
    @abstractmethod
    def critical_temperature(self) -> float:
        """临界温度 (K)"""
        pass
    
    @property
    @abstractmethod
    def critical_pressure(self) -> float:
        """临界压力 (Pa)"""
        pass
    
    @property
    @abstractmethod
    def acentric_factor(self) -> float:
        """偏心因子"""
        pass

class IStream(ABC):
    """物流接口"""
    
    @abstractmethod
    def get_property(self, property_type: PropertyType, phase: Optional[PhaseType] = None) -> float:
        """获取物性值"""
        pass
    
    @abstractmethod
    def set_property(self, property_type: PropertyType, value: float, phase: Optional[PhaseType] = None):
        """设置物性值"""
        pass
    
    @abstractmethod
    def get_composition(self, phase: Optional[PhaseType] = None) -> Dict[str, float]:
        """获取组成"""
        pass
    
    @abstractmethod
    def set_composition(self, composition: Dict[str, float], phase: Optional[PhaseType] = None):
        """设置组成"""
        pass
    
    @abstractmethod
    def get_flow_rate(self) -> float:
        """获取流量 (mol/s)"""
        pass
    
    @abstractmethod
    def set_flow_rate(self, flow_rate: float):
        """设置流量 (mol/s)"""
        pass

class IPropertyPackage(ABC):
    """物性包接口"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """物性包名称"""
        pass
    
    @abstractmethod
    def calculate_properties(self, stream: IStream, properties: List[PropertyType]) -> Dict[PropertyType, float]:
        """计算物性"""
        pass
    
    @abstractmethod
    def flash_calculation(self, stream: IStream, spec1: PropertyType, spec2: PropertyType) -> IStream:
        """闪蒸计算"""
        pass
    
    @abstractmethod
    def add_component(self, component_name: str):
        """添加组分"""
        pass

class IUnitOperation(ABC):
    """单元操作接口"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """单元操作名称"""
        pass
    
    @abstractmethod
    def calculate(self) -> bool:
        """执行计算"""
        pass
    
    @abstractmethod
    def get_inlet_streams(self) -> List[IStream]:
        """获取进料流"""
        pass
    
    @abstractmethod
    def get_outlet_streams(self) -> List[IStream]:
        """获取出料流"""
        pass
    
    @abstractmethod
    def is_calculated(self) -> bool:
        """检查是否已计算"""
        pass

class IFlashAlgorithm(ABC):
    """闪蒸算法接口"""
    
    @abstractmethod
    def calculate(self, stream: IStream, spec1_type: PropertyType, spec1_value: float,
                 spec2_type: PropertyType, spec2_value: float, property_package: IPropertyPackage) -> IStream:
        """执行闪蒸计算"""
        pass

class ISolver(ABC):
    """求解器接口"""
    
    @abstractmethod
    def solve(self, equations: List[callable], variables: List[float], 
              tolerance: float = 1e-6, max_iterations: int = 100) -> List[float]:
        """求解方程组"""
        pass 