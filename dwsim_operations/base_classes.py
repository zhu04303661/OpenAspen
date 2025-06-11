"""
DWSIM 单元操作基础类
==================

实现单元操作的基础类，包含：
- UnitOpBaseClass: 主要的单元操作基础类
- SpecialOpBaseClass: 特殊操作基础类
- CAPE-OPEN兼容接口

从原VB.NET版本1:1转换的Python实现。
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET


class SimulationObjectClass(Enum):
    """
    仿真对象类别枚举
    
    定义不同类型的仿真对象分类。
    """
    # 物料流和能量流
    Streams = "Streams"
    
    # 混合器和分离器
    MixersSplitters = "MixersSplitters"
    
    # 传热设备
    HeatExchangers = "HeatExchangers"
    
    # 分离设备
    SeparationEquipment = "SeparationEquipment"
    
    # 反应器
    Reactors = "Reactors"
    
    # 流体流动设备
    PressureChangers = "PressureChangers"
    
    # 逻辑操作
    Logical = "Logical"
    
    # 能量设备
    EnergyStreams = "EnergyStreams"
    
    # 其他设备
    Other = "Other"


@dataclass
class ConnectionPoint:
    """
    连接点类
    
    表示单元操作的输入或输出连接点。
    """
    is_attached: bool = False
    attached_connector_name: str = ""
    attached_to_name: str = ""
    connector_type: str = "ConOut"  # ConOut, ConIn, ConEn
    position: tuple = (0, 0)
    
    def attach(self, connector_name: str, attached_to: str):
        """
        连接到另一个对象
        
        Args:
            connector_name: 连接器名称
            attached_to: 连接到的对象名称
        """
        self.is_attached = True
        self.attached_connector_name = connector_name
        self.attached_to_name = attached_to
    
    def detach(self):
        """断开连接"""
        self.is_attached = False
        self.attached_connector_name = ""
        self.attached_to_name = ""


@dataclass
class GraphicObject:
    """
    图形对象类
    
    表示单元操作在流程图中的图形表示。
    """
    tag: str = ""
    name: str = ""
    object_type: str = "UnitOperation"
    calculated: bool = False
    active: bool = True
    position: tuple = (0, 0)
    size: tuple = (50, 50)
    
    # 连接点列表
    input_connectors: List[ConnectionPoint] = field(default_factory=list)
    output_connectors: List[ConnectionPoint] = field(default_factory=list)
    energy_connectors: List[ConnectionPoint] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.input_connectors:
            self.input_connectors = []
        if not self.output_connectors:
            self.output_connectors = []
        if not self.energy_connectors:
            self.energy_connectors = []


class UnitOpBaseClass(ABC):
    """
    单元操作基础类
    
    所有单元操作的基础类，包含：
    - 基本属性和状态管理
    - 计算接口定义
    - 数据保存和加载
    - 属性包管理
    - 连接管理
    - 调试功能
    """
    
    def __init__(self):
        """初始化单元操作基础类"""
        
        # 基本属性
        self.name: str = ""
        self.tag: str = ""
        self.description: str = ""
        self.component_name: str = ""
        self.component_description: str = ""
        
        # 计算状态
        self.calculated: bool = False
        self.error_message: str = ""
        self.last_updated: Optional[float] = None
        
        # 对象分类
        self.object_class: SimulationObjectClass = SimulationObjectClass.Other
        
        # 图形对象
        self.graphic_object: Optional[GraphicObject] = None
        
        # 属性包相关
        self._property_package: Optional[Any] = None
        self._property_package_id: str = ""
        
        # 流程图引用
        self.flowsheet: Optional[Any] = None
        
        # 调试相关
        self.debug_mode: bool = False
        self.debug_text: str = ""
        
        # 关联的实用工具
        self.attached_utilities: List[Any] = []
        
        # 规格和调节相关
        self.is_spec_attached: bool = False
        self.attached_spec_id: str = ""
        self.spec_var_type: str = ""
        
        # CAPE-OPEN模式
        self._cape_open_mode: bool = False
        
        # 日志记录器
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # 创建默认图形对象
        self.create_graphic_object()
    
    def create_graphic_object(self):
        """创建默认的图形对象"""
        self.graphic_object = GraphicObject(
            tag=self.tag or self.name,
            name=self.name,
            object_type=self.__class__.__name__
        )
    
    @property
    def property_package(self) -> Optional[Any]:
        """
        获取或设置关联的属性包
        
        Returns:
            属性包对象
        """
        if self._property_package is not None:
            return self._property_package
            
        if self._property_package_id and self.flowsheet:
            # 尝试从流程图中获取属性包
            if hasattr(self.flowsheet, 'property_packages'):
                property_packages = self.flowsheet.property_packages
                if self._property_package_id in property_packages:
                    return property_packages[self._property_package_id]
                
                # 如果指定的属性包不存在，返回第一个可用的
                for pp in property_packages.values():
                    self._property_package_id = getattr(pp, 'unique_id', '')
                    return pp
        
        return None
    
    @property_package.setter
    def property_package(self, value: Optional[Any]):
        """设置属性包"""
        if value is not None:
            self._property_package_id = getattr(value, 'unique_id', '')
            self._property_package = value
        else:
            self._property_package = None
    
    @abstractmethod
    def calculate(self, args: Optional[Any] = None):
        """
        执行单元操作计算
        
        Args:
            args: 计算参数
            
        这是抽象方法，必须在子类中实现具体的计算逻辑。
        """
        pass
    
    def solve(self):
        """
        求解单元操作
        
        这是对calculate方法的包装，添加了状态管理和异常处理。
        """
        try:
            self.logger.info(f"开始计算单元操作: {self.name}")
            start_time = time.time()
            
            # 重置错误状态
            self.error_message = ""
            
            # 调用具体的计算方法
            self.calculate()
            
            # 更新状态
            self.calculated = True
            self.last_updated = time.time()
            
            calculation_time = time.time() - start_time
            self.logger.info(f"单元操作 {self.name} 计算完成，耗时: {calculation_time:.3f}秒")
            
        except Exception as e:
            self.calculated = False
            self.error_message = str(e)
            self.logger.error(f"单元操作 {self.name} 计算失败: {e}")
            raise
    
    def de_calculate(self):
        """
        取消计算状态
        
        将单元操作标记为未计算状态。
        """
        self.calculated = False
        self.error_message = ""
        if self.graphic_object:
            self.graphic_object.calculated = False
    
    def unsolve(self):
        """
        取消求解
        
        这是de_calculate的别名。
        """
        self.de_calculate()
    
    def validate(self) -> bool:
        """
        验证单元操作的有效性
        
        Returns:
            bool: 是否有效
        """
        # 基本验证
        if not self.name:
            raise ValueError("单元操作名称不能为空")
        
        if not self.property_package:
            raise ValueError(f"单元操作 {self.name} 缺少属性包")
        
        # 检查必要的连接
        if self.graphic_object:
            return self._validate_connections()
        
        return True
    
    def _validate_connections(self) -> bool:
        """
        验证连接的有效性
        
        Returns:
            bool: 连接是否有效
        """
        # 子类可以重写此方法来添加特定的连接验证逻辑
        return True
    
    def get_debug_report(self) -> str:
        """
        获取调试报告
        
        Returns:
            str: 调试信息
        """
        self.debug_mode = True
        self.debug_text = ""
        
        try:
            self.append_debug_line(f"=== {self.name} 调试报告 ===")
            self.append_debug_line(f"对象类型: {self.__class__.__name__}")
            self.append_debug_line(f"计算状态: {'已计算' if self.calculated else '未计算'}")
            self.append_debug_line(f"属性包: {self.property_package.__class__.__name__ if self.property_package else '无'}")
            
            # 尝试执行计算
            self.calculate()
            self.append_debug_line("计算成功完成")
            
        except Exception as e:
            self.append_debug_line(f"计算失败: {str(e)}")
            
        finally:
            self.debug_mode = False
        
        return self.debug_text
    
    def append_debug_line(self, text: str):
        """
        添加调试信息行
        
        Args:
            text: 调试文本
        """
        if self.debug_mode:
            self.debug_text += f"{text}\n"
    
    def load_data(self, data: List[ET.Element]) -> bool:
        """
        从XML数据加载对象状态
        
        Args:
            data: XML元素列表
            
        Returns:
            bool: 加载是否成功
        """
        try:
            for element in data:
                if element.tag == "PropertyPackage":
                    self._property_package_id = element.text or ""
                elif element.tag == "Name":
                    self.name = element.text or ""
                elif element.tag == "Tag":
                    self.tag = element.text or ""
                elif element.tag == "Description":
                    self.description = element.text or ""
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return False
    
    def save_data(self) -> List[ET.Element]:
        """
        保存对象状态到XML数据
        
        Returns:
            List[ET.Element]: XML元素列表
        """
        elements = []
        
        # 基本信息
        if self.name:
            name_elem = ET.Element("Name")
            name_elem.text = self.name
            elements.append(name_elem)
        
        if self.tag:
            tag_elem = ET.Element("Tag")
            tag_elem.text = self.tag
            elements.append(tag_elem)
        
        if self.description:
            desc_elem = ET.Element("Description")
            desc_elem.text = self.description
            elements.append(desc_elem)
        
        # 属性包ID
        pp_id = self._property_package_id
        if not pp_id and self._property_package:
            pp_id = getattr(self._property_package, 'name', '')
        
        if pp_id:
            pp_elem = ET.Element("PropertyPackage")
            pp_elem.text = pp_id
            elements.append(pp_elem)
        
        return elements
    
    def clone_xml(self) -> 'UnitOpBaseClass':
        """
        通过XML方式克隆对象
        
        Returns:
            UnitOpBaseClass: 克隆的对象
        """
        # 创建新实例
        clone = self.__class__()
        
        # 保存当前数据并加载到新实例
        data = self.save_data()
        clone.load_data(data)
        
        return clone
    
    def clone_json(self) -> 'UnitOpBaseClass':
        """
        通过JSON方式克隆对象
        
        Returns:
            UnitOpBaseClass: 克隆的对象
        """
        # 序列化为JSON
        json_data = self.to_json()
        
        # 创建新实例并反序列化
        clone = self.__class__()
        clone.from_json(json_data)
        
        return clone
    
    def to_json(self) -> str:
        """
        转换为JSON字符串
        
        Returns:
            str: JSON字符串
        """
        data = {
            'name': self.name,
            'tag': self.tag,
            'description': self.description,
            'component_name': self.component_name,
            'component_description': self.component_description,
            'calculated': self.calculated,
            'error_message': self.error_message,
            'property_package_id': self._property_package_id,
            'object_class': self.object_class.value if self.object_class else None
        }
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def from_json(self, json_str: str):
        """
        从JSON字符串加载
        
        Args:
            json_str: JSON字符串
        """
        try:
            data = json.loads(json_str)
            
            self.name = data.get('name', '')
            self.tag = data.get('tag', '')
            self.description = data.get('description', '')
            self.component_name = data.get('component_name', '')
            self.component_description = data.get('component_description', '')
            self.calculated = data.get('calculated', False)
            self.error_message = data.get('error_message', '')
            self._property_package_id = data.get('property_package_id', '')
            
            # 对象分类
            object_class_str = data.get('object_class')
            if object_class_str:
                try:
                    self.object_class = SimulationObjectClass(object_class_str)
                except ValueError:
                    pass
                    
        except Exception as e:
            self.logger.error(f"从JSON加载失败: {e}")
            raise
    
    def get_display_name(self) -> str:
        """
        获取显示名称
        
        Returns:
            str: 显示名称
        """
        return self.tag or self.name or self.__class__.__name__
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(name='{self.name}', tag='{self.tag}')"
    
    def __repr__(self) -> str:
        """详细表示"""
        return self.__str__()


class SpecialOpBaseClass(UnitOpBaseClass):
    """
    特殊操作基础类
    
    用于逻辑操作等特殊类型的单元操作。
    """
    
    def __init__(self):
        """初始化特殊操作基础类"""
        super().__init__()
        self.object_class = SimulationObjectClass.Logical
    
    def calculate(self, args: Optional[Any] = None):
        """
        特殊操作的默认计算方法
        
        Args:
            args: 计算参数
        """
        # 特殊操作通常不需要复杂的计算
        self.calculated = True


@dataclass
class SpecialOpObjectInfo:
    """
    特殊操作对象信息
    
    用于存储特殊操作的相关信息。
    """
    name: str = ""
    description: str = ""
    object_type: str = ""
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'object_type': self.object_type,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpecialOpObjectInfo':
        """从字典创建"""
        return cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            object_type=data.get('object_type', ''),
            enabled=data.get('enabled', True)
        ) 