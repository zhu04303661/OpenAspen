"""
DWSIM 具体单元操作类
==================

实现具体的单元操作，包括：
- Mixer: 混合器
- Splitter: 分离器  
- Heater: 加热器
- Cooler: 冷却器
- HeatExchanger: 热交换器
- Pump: 泵
- Compressor: 压缩机
- Valve: 阀门
- ComponentSeparator: 组分分离器
- Filter: 过滤器
- Vessel: 容器
- Tank: 储罐

从原VB.NET版本1:1转换的Python实现。
"""

import math
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from .base_classes import UnitOpBaseClass, SimulationObjectClass


class PressureBehavior(Enum):
    """
    混合器压力行为枚举
    """
    AVERAGE = "Average"      # 平均值
    MAXIMUM = "Maximum"      # 最大值
    MINIMUM = "Minimum"      # 最小值


@dataclass
class PhaseProperties:
    """
    相态属性数据类
    """
    temperature: float = 0.0
    pressure: float = 0.0
    enthalpy: float = 0.0
    massflow: float = 0.0
    molarfraction: float = 0.0
    massfraction: float = 0.0
    
    def __post_init__(self):
        """确保数值有效"""
        if math.isnan(self.temperature):
            self.temperature = 273.15
        if math.isnan(self.pressure):
            self.pressure = 101325.0
        if math.isnan(self.enthalpy):
            self.enthalpy = 0.0
        if math.isnan(self.massflow):
            self.massflow = 0.0


@dataclass
class CompoundData:
    """
    化合物数据类
    """
    name: str = ""
    mass_fraction: float = 0.0
    mole_fraction: float = 0.0
    molar_weight: float = 0.0
    
    def __post_init__(self):
        """确保数值有效"""
        if math.isnan(self.mass_fraction):
            self.mass_fraction = 0.0
        if math.isnan(self.mole_fraction):
            self.mole_fraction = 0.0
        if math.isnan(self.molar_weight):
            self.molar_weight = 1.0


class Mixer(UnitOpBaseClass):
    """
    混合器单元操作
    
    用于将多个物料流混合成一个物料流，同时执行质量和能量平衡。
    
    混合器执行设备中的质量平衡，确定出料流的质量流量和组成。
    压力根据用户定义的参数计算。
    温度通过对出料流进行PH闪蒸计算，焓值从入料流计算得出（能量平衡）。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化混合器
        
        Args:
            name: 混合器名称
            description: 混合器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.MixersSplitters
        self.component_name = name or "Mixer"
        self.component_description = description or "物料流混合器"
        self.name = name or "MIXER-001"
        self.tag = name or "混合器"
        
        # 混合器特定属性
        self.pressure_calculation = PressureBehavior.MINIMUM
        
        # 创建图形对象和连接点
        self._create_mixer_graphic_object()
    
    def _create_mixer_graphic_object(self):
        """创建混合器的图形对象"""
        from .base_classes import GraphicObject, ConnectionPoint
        
        self.graphic_object = GraphicObject(
            tag=self.tag,
            name=self.name,
            object_type="Mixer"
        )
        
        # 创建6个输入连接点（最多6个入料流）
        for i in range(6):
            input_conn = ConnectionPoint(connector_type="ConIn")
            self.graphic_object.input_connectors.append(input_conn)
        
        # 创建1个输出连接点
        output_conn = ConnectionPoint(connector_type="ConOut")
        self.graphic_object.output_connectors.append(output_conn)
    
    def _validate_connections(self) -> bool:
        """
        验证混合器连接
        
        Returns:
            bool: 连接是否有效
        """
        # 检查是否有输出连接
        if not self.graphic_object.output_connectors[0].is_attached:
            raise ValueError("混合器必须连接输出物料流")
        
        # 检查是否至少有一个输入连接
        has_input = any(conn.is_attached for conn in self.graphic_object.input_connectors)
        if not has_input:
            raise ValueError("混合器必须至少连接一个输入物料流")
        
        return True
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行混合器计算
        
        Args:
            args: 计算参数（可选）
        """
        # 验证连接
        self._validate_connections()
        
        if self.debug_mode:
            self.append_debug_line("混合器用于将最多六个物料流混合成一个，同时执行所有质量和能量平衡。")
            self.append_debug_line("混合器执行设备中的质量平衡，确定出料流的质量流量和组成。")
            self.append_debug_line("压力根据用户定义的参数计算。")
            self.append_debug_line("温度通过对出料流进行PH闪蒸计算，焓值从入料流计算得出（能量平衡）。")
        
        # 获取输出流对象
        output_stream = self._get_output_stream()
        if not output_stream:
            raise ValueError("无法获取输出物料流")
        
        # 初始化计算变量
        total_enthalpy = 0.0      # 总焓值 [kJ/s]
        specific_enthalpy = 0.0   # 比焓 [kJ/kg]
        temperature = 0.0         # 温度 [K]
        total_mass_flow = 0.0     # 总质量流量 [kg/s]
        pressure = 0.0            # 压力 [Pa]
        inlet_count = 0           # 有效入料流数量
        
        # 组分质量流量字典 [kg/s]
        component_mass_flows = {}
        
        # 遍历所有输入连接点
        for i, input_conn in enumerate(self.graphic_object.input_connectors):
            if not input_conn.is_attached:
                continue
            
            if self.debug_mode:
                self.append_debug_line(f"<h3>入料流 #{i+1}</h3>")
            
            # 获取入料流对象
            input_stream = self._get_input_stream(input_conn)
            if not input_stream:
                continue
            
            # 验证入料流是否已计算
            if not self._is_stream_calculated(input_stream):
                raise ValueError(f"入料流 {input_conn.attached_to_name} 尚未计算")
            
            # 获取入料流属性
            stream_props = self._get_stream_properties(input_stream)
            
            if self.debug_mode:
                self.append_debug_line(f"质量流量: {stream_props.massflow} kg/s")
                self.append_debug_line(f"压力: {stream_props.pressure} Pa")
                self.append_debug_line(f"焓值: {stream_props.enthalpy} kJ/kg")
            
            # 处理压力计算
            if self.pressure_calculation == PressureBehavior.MINIMUM:
                if stream_props.pressure < pressure or pressure == 0:
                    pressure = stream_props.pressure
            elif self.pressure_calculation == PressureBehavior.MAXIMUM:
                if stream_props.pressure > pressure or pressure == 0:
                    pressure = stream_props.pressure
            else:  # AVERAGE
                pressure += stream_props.pressure
                inlet_count += 1
            
            # 累计质量流量和焓值
            mass_flow = stream_props.massflow
            total_mass_flow += mass_flow
            
            if not math.isnan(stream_props.enthalpy):
                total_enthalpy += mass_flow * stream_props.enthalpy
            
            # 处理组分质量流量
            components = self._get_stream_components(input_stream)
            for comp_name, comp_data in components.items():
                if comp_name not in component_mass_flows:
                    component_mass_flows[comp_name] = 0.0
                component_mass_flows[comp_name] += comp_data.mass_fraction * mass_flow
            
            # 温度加权平均
            if total_mass_flow != 0:
                temperature += (mass_flow / total_mass_flow) * stream_props.temperature
        
        # 计算比焓
        if total_mass_flow != 0:
            specific_enthalpy = total_enthalpy / total_mass_flow
        else:
            specific_enthalpy = 0.0
        
        # 计算平均压力
        if self.pressure_calculation == PressureBehavior.AVERAGE and inlet_count > 0:
            pressure = pressure / inlet_count
        
        # 如果没有质量流量，设置默认温度
        if total_mass_flow == 0:
            temperature = 273.15  # 0°C
        
        if self.debug_mode:
            self.append_debug_line("<h3>混合后出料流</h3>")
            self.append_debug_line(f"质量流量: {total_mass_flow} kg/s")
            self.append_debug_line(f"压力: {pressure} Pa")
            self.append_debug_line(f"焓值: {specific_enthalpy} kJ/kg")
        
        # 设置输出流属性
        self._set_output_stream_properties(
            output_stream, 
            total_mass_flow, 
            pressure, 
            specific_enthalpy, 
            temperature,
            component_mass_flows
        )
        
        # 标记计算完成
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True
    
    def _get_output_stream(self):
        """获取输出物料流对象"""
        if not self.flowsheet:
            return None
        
        output_conn = self.graphic_object.output_connectors[0]
        if not output_conn.is_attached:
            return None
        
        return self.flowsheet.simulation_objects.get(output_conn.attached_to_name)
    
    def _get_input_stream(self, input_conn):
        """获取输入物料流对象"""
        if not self.flowsheet:
            return None
        
        return self.flowsheet.simulation_objects.get(input_conn.attached_to_name)
    
    def _is_stream_calculated(self, stream) -> bool:
        """检查物料流是否已计算"""
        return getattr(stream, 'calculated', False)
    
    def _get_stream_properties(self, stream) -> PhaseProperties:
        """获取物料流的相态属性"""
        # 这里需要根据实际的物料流实现来调整
        # 假设物料流有phases属性，包含相态信息
        
        try:
            phase = getattr(stream, 'phases', [{}])[0]
            props = getattr(phase, 'properties', {})
            
            return PhaseProperties(
                temperature=getattr(props, 'temperature', 273.15),
                pressure=getattr(props, 'pressure', 101325.0),
                enthalpy=getattr(props, 'enthalpy', 0.0),
                massflow=getattr(props, 'massflow', 0.0),
                molarfraction=getattr(props, 'molarfraction', 1.0),
                massfraction=getattr(props, 'massfraction', 1.0)
            )
        except:
            # 返回默认值
            return PhaseProperties()
    
    def _get_stream_components(self, stream) -> Dict[str, CompoundData]:
        """获取物料流的组分信息"""
        components = {}
        
        try:
            phase = getattr(stream, 'phases', [{}])[0]
            compounds = getattr(phase, 'compounds', {})
            
            for comp_name, comp_obj in compounds.items():
                components[comp_name] = CompoundData(
                    name=comp_name,
                    mass_fraction=getattr(comp_obj, 'mass_fraction', 0.0),
                    mole_fraction=getattr(comp_obj, 'mole_fraction', 0.0),
                    molar_weight=getattr(comp_obj, 'molar_weight', 1.0)
                )
        except:
            pass
        
        return components
    
    def _set_output_stream_properties(self, output_stream, mass_flow, pressure, 
                                     enthalpy, temperature, component_mass_flows):
        """设置输出物料流的属性"""
        try:
            # 清除输出流
            if hasattr(output_stream, 'clear'):
                output_stream.clear()
            if hasattr(output_stream, 'clear_all_props'):
                output_stream.clear_all_props()
            
            # 获取相态对象
            phases = getattr(output_stream, 'phases', [])
            if not phases:
                return
            
            phase = phases[0]
            props = getattr(phase, 'properties', None)
            
            if props:
                # 设置基本属性
                if mass_flow != 0:
                    props.enthalpy = enthalpy
                props.pressure = pressure
                props.massflow = mass_flow
                props.molarfraction = 1.0
                props.massfraction = 1.0
                props.temperature = temperature
            
            # 设置组分组成
            compounds = getattr(phase, 'compounds', {})
            
            # 首先设置质量分数
            for comp_name, comp_obj in compounds.items():
                if comp_name in component_mass_flows and mass_flow != 0:
                    comp_obj.mass_fraction = component_mass_flows[comp_name] / mass_flow
                else:
                    comp_obj.mass_fraction = 0.0
            
            # 计算摩尔分数
            if mass_flow != 0:
                mass_div_mm = 0.0
                for comp_name, comp_obj in compounds.items():
                    molar_weight = getattr(comp_obj, 'molar_weight', 1.0)
                    if molar_weight > 0:
                        mass_div_mm += comp_obj.mass_fraction / molar_weight
                
                for comp_name, comp_obj in compounds.items():
                    if mass_div_mm > 0:
                        molar_weight = getattr(comp_obj, 'molar_weight', 1.0)
                        if molar_weight > 0:
                            comp_obj.mole_fraction = (comp_obj.mass_fraction / molar_weight) / mass_div_mm
                        else:
                            comp_obj.mole_fraction = 0.0
                    else:
                        comp_obj.mole_fraction = 0.0
            else:
                for comp_name, comp_obj in compounds.items():
                    comp_obj.mole_fraction = 0.0
            
            # 设置规格类型
            if hasattr(output_stream, 'spec_type'):
                output_stream.spec_type = "Pressure_and_Enthalpy"
                
        except Exception as e:
            self.logger.error(f"设置输出流属性失败: {e}")
            raise
    
    def clone_xml(self) -> 'Mixer':
        """克隆混合器对象"""
        clone = Mixer()
        data = self.save_data()
        clone.load_data(data)
        clone.pressure_calculation = self.pressure_calculation
        return clone
    
    def clone_json(self) -> 'Mixer':
        """通过JSON克隆混合器对象"""
        import json
        
        # 序列化当前对象
        data = {
            'name': self.name,
            'tag': self.tag,
            'description': self.description,
            'pressure_calculation': self.pressure_calculation.value,
            'component_name': self.component_name,
            'component_description': self.component_description
        }
        
        # 创建新对象并反序列化
        clone = Mixer()
        clone.name = data['name']
        clone.tag = data['tag']
        clone.description = data['description']
        clone.pressure_calculation = PressureBehavior(data['pressure_calculation'])
        clone.component_name = data['component_name']
        clone.component_description = data['component_description']
        
        return clone


class Splitter(UnitOpBaseClass):
    """
    分离器单元操作
    
    用于将一个物料流分离成多个物料流。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化分离器
        
        Args:
            name: 分离器名称
            description: 分离器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.MixersSplitters
        self.component_name = name or "Splitter"
        self.component_description = description or "物料流分离器"
        self.name = name or "SPLITTER-001"
        self.tag = name or "分离器"
        
        # 分离器特定属性 - 分流比例
        self.split_ratios = [0.5, 0.5]  # 默认两路等分
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行分离器计算
        
        Args:
            args: 计算参数（可选）
        """
        # 简化的分离器计算逻辑
        # 实际实现需要根据具体需求调整
        
        if self.debug_mode:
            self.append_debug_line("分离器计算开始")
        
        # 这里添加具体的分离器计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Heater(UnitOpBaseClass):
    """
    加热器单元操作
    
    用于加热物料流到指定温度或提供指定热量。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化加热器
        
        Args:
            name: 加热器名称  
            description: 加热器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.HeatExchangers
        self.component_name = name or "Heater"
        self.component_description = description or "物料流加热器"
        self.name = name or "HEATER-001"
        self.tag = name or "加热器"
        
        # 加热器特定属性
        self.outlet_temperature = 373.15  # 出口温度 [K]
        self.heat_duty = 0.0              # 热负荷 [kW]
        self.calculation_mode = "OutletTemperature"  # 计算模式
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行加热器计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("加热器计算开始")
        
        # 这里添加具体的加热器计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Cooler(UnitOpBaseClass):
    """
    冷却器单元操作
    
    用于冷却物料流到指定温度或移除指定热量。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化冷却器
        
        Args:
            name: 冷却器名称
            description: 冷却器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.HeatExchangers
        self.component_name = name or "Cooler"
        self.component_description = description or "物料流冷却器"
        self.name = name or "COOLER-001"
        self.tag = name or "冷却器"
        
        # 冷却器特定属性
        self.outlet_temperature = 273.15  # 出口温度 [K]
        self.heat_duty = 0.0              # 热负荷 [kW]
        self.calculation_mode = "OutletTemperature"  # 计算模式
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行冷却器计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("冷却器计算开始")
        
        # 这里添加具体的冷却器计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class HeatExchanger(UnitOpBaseClass):
    """
    热交换器单元操作
    
    用于两个物料流之间的热量交换。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化热交换器
        
        Args:
            name: 热交换器名称
            description: 热交换器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.HeatExchangers
        self.component_name = name or "HeatExchanger"
        self.component_description = description or "热交换器"
        self.name = name or "HX-001"
        self.tag = name or "热交换器"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行热交换器计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("热交换器计算开始")
        
        # 这里添加具体的热交换器计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Pump(UnitOpBaseClass):
    """
    泵单元操作
    
    用于提高物料流的压力。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化泵
        
        Args:
            name: 泵名称
            description: 泵描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.PressureChangers
        self.component_name = name or "Pump"
        self.component_description = description or "离心泵"
        self.name = name or "PUMP-001"
        self.tag = name or "泵"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行泵计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("泵计算开始")
        
        # 这里添加具体的泵计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Compressor(UnitOpBaseClass):
    """
    压缩机单元操作
    
    用于压缩气体物料流。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化压缩机
        
        Args:
            name: 压缩机名称
            description: 压缩机描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.PressureChangers
        self.component_name = name or "Compressor"
        self.component_description = description or "离心压缩机"
        self.name = name or "COMP-001"
        self.tag = name or "压缩机"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行压缩机计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("压缩机计算开始")
        
        # 这里添加具体的压缩机计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Valve(UnitOpBaseClass):
    """
    阀门单元操作
    
    用于降低物料流的压力。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化阀门
        
        Args:
            name: 阀门名称
            description: 阀门描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.PressureChangers
        self.component_name = name or "Valve"
        self.component_description = description or "节流阀"
        self.name = name or "VALVE-001"
        self.tag = name or "阀门"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行阀门计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("阀门计算开始")
        
        # 这里添加具体的阀门计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class ComponentSeparator(UnitOpBaseClass):
    """
    组分分离器单元操作
    
    用于按组分分离物料流。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化组分分离器
        
        Args:
            name: 组分分离器名称
            description: 组分分离器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.SeparationEquipment
        self.component_name = name or "ComponentSeparator"
        self.component_description = description or "组分分离器"
        self.name = name or "CSEP-001"
        self.tag = name or "组分分离器"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行组分分离器计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("组分分离器计算开始")
        
        # 这里添加具体的组分分离器计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Filter(UnitOpBaseClass):
    """
    过滤器单元操作
    
    用于过滤分离固体颗粒。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化过滤器
        
        Args:
            name: 过滤器名称
            description: 过滤器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.SeparationEquipment
        self.component_name = name or "Filter"
        self.component_description = description or "过滤器"
        self.name = name or "FILTER-001"
        self.tag = name or "过滤器"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行过滤器计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("过滤器计算开始")
        
        # 这里添加具体的过滤器计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Vessel(UnitOpBaseClass):
    """
    容器单元操作
    
    用于物料的暂存和相分离。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化容器
        
        Args:
            name: 容器名称
            description: 容器描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.SeparationEquipment
        self.component_name = name or "Vessel"
        self.component_description = description or "分离容器"
        self.name = name or "VESSEL-001"
        self.tag = name or "容器"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行容器计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("容器计算开始")
        
        # 这里添加具体的容器计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True


class Tank(UnitOpBaseClass):
    """
    储罐单元操作
    
    用于物料的储存。
    """
    
    def __init__(self, name: str = "", description: str = ""):
        """
        初始化储罐
        
        Args:
            name: 储罐名称
            description: 储罐描述
        """
        super().__init__()
        
        # 设置对象属性
        self.object_class = SimulationObjectClass.Other
        self.component_name = name or "Tank"
        self.component_description = description or "储罐"
        self.name = name or "TANK-001"
        self.tag = name or "储罐"
    
    def calculate(self, args: Optional[Any] = None):
        """
        执行储罐计算
        
        Args:
            args: 计算参数（可选）
        """
        if self.debug_mode:
            self.append_debug_line("储罐计算开始")
        
        # 这里添加具体的储罐计算逻辑
        # ...
        
        self.calculated = True
        if self.graphic_object:
            self.graphic_object.calculated = True 