"""
DWSIM 单元操作测试专用 pytest fixtures
=======================================

为DWSIM单元操作测试提供共享的测试设备和数据。
"""

import pytest
import logging
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from dwsim_operations import *
    from dwsim_operations.integration import *
except ImportError as e:
    print(f"警告：无法导入dwsim_operations模块: {e}")


@pytest.fixture(scope="session")
def disable_logging():
    """会话级别禁用日志以减少噪音"""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def integrated_solver():
    """
    创建集成求解器实例
    
    Returns:
        IntegratedFlowsheetSolver: 求解器实例
    """
    try:
        return create_integrated_solver()
    except Exception as e:
        pytest.skip(f"无法创建集成求解器: {e}")


@pytest.fixture
def sample_mixer(integrated_solver):
    """
    创建测试用混合器
    
    Args:
        integrated_solver: 集成求解器fixture
        
    Returns:
        Mixer: 混合器实例
    """
    return integrated_solver.create_and_add_operation("Mixer", "MIX-001", "测试混合器")


@pytest.fixture
def sample_heater(integrated_solver):
    """
    创建测试用加热器
    
    Args:
        integrated_solver: 集成求解器fixture
        
    Returns:
        Heater: 加热器实例
    """
    return integrated_solver.create_and_add_operation("Heater", "HX-001", "测试加热器")


@pytest.fixture
def sample_pump(integrated_solver):
    """
    创建测试用泵
    
    Args:
        integrated_solver: 集成求解器fixture
        
    Returns:
        Pump: 泵实例
    """
    return integrated_solver.create_and_add_operation("Pump", "P-001", "测试泵")


@pytest.fixture
def sample_heat_exchanger(integrated_solver):
    """
    创建测试用热交换器
    
    Args:
        integrated_solver: 集成求解器fixture
        
    Returns:
        HeatExchanger: 热交换器实例
    """
    return integrated_solver.create_and_add_operation("HeatExchanger", "HX-001", "测试热交换器")


@pytest.fixture
def sample_valve(integrated_solver):
    """
    创建测试用阀门
    
    Args:
        integrated_solver: 集成求解器fixture
        
    Returns:
        Valve: 阀门实例
    """
    return integrated_solver.create_and_add_operation("Valve", "V-001", "测试阀门")


@pytest.fixture
def sample_splitter(integrated_solver):
    """
    创建测试用分离器
    
    Args:
        integrated_solver: 集成求解器fixture
        
    Returns:
        Splitter: 分离器实例
    """
    return integrated_solver.create_and_add_operation("Splitter", "SPL-001", "测试分离器")


@pytest.fixture
def sample_mixer_data():
    """
    混合器测试数据
    
    Returns:
        Dict: 测试数据
    """
    return {
        "pressures": [300000, 250000, 280000],  # Pa
        "mass_flows": [100.0, 150.0, 80.0],    # kg/s
        "temperatures": [298.15, 308.15, 318.15],  # K
        "inlet_data": [
            {"mass_flow": 100.0, "enthalpy": 2500.0, "temperature": 298.15},
            {"mass_flow": 150.0, "enthalpy": 2300.0, "temperature": 308.15},
            {"mass_flow": 80.0, "enthalpy": 2700.0, "temperature": 318.15}
        ],
        "component_data": [
            {"H2O": 0.9, "NaCl": 0.1},
            {"H2O": 0.8, "NaCl": 0.2}
        ]
    }


@pytest.fixture
def sample_heater_data():
    """
    加热器测试数据
    
    Returns:
        Dict: 测试数据
    """
    return {
        "inlet_conditions": {
            "mass_flow": 100.0,      # kg/s
            "enthalpy": 2500.0,      # kJ/kg
            "temperature": 298.15,   # K
            "pressure": 300000.0     # Pa
        },
        "outlet_conditions": {
            "temperature": 373.15,   # K
            "enthalpy": 2700.0       # kJ/kg
        },
        "heat_duty": 1000.0,        # kW
        "efficiency": 0.85,         # 85%
        "pressure_drop": 5000.0     # Pa
    }


@pytest.fixture
def sample_pump_data():
    """
    泵测试数据
    
    Returns:
        Dict: 测试数据
    """
    return {
        "inlet_pressure": 100000.0,     # Pa
        "outlet_pressure": 500000.0,    # Pa
        "flow_rate": 0.1,               # m³/s
        "fluid_density": 1000.0,        # kg/m³
        "efficiency": 0.75,             # 75%
        "gravity": 9.81,                # m/s²
        "npsh_data": {
            "suction_pressure": 200000.0,  # Pa
            "vapor_pressure": 3169.0,      # Pa
            "suction_head": 5.0             # m
        }
    }


@pytest.fixture
def sample_heat_exchanger_data():
    """
    热交换器测试数据
    
    Returns:
        Dict: 测试数据
    """
    return {
        "hot_side": {
            "inlet_temp": 373.15,      # K (100°C)
            "outlet_temp": 333.15,     # K (60°C)
            "mass_flow": 2.0,          # kg/s
            "cp": 4180.0               # J/kg·K
        },
        "cold_side": {
            "inlet_temp": 298.15,      # K (25°C)
            "outlet_temp": 318.15,     # K (45°C)
            "mass_flow": 3.0,          # kg/s
            "cp": 4180.0               # J/kg·K
        },
        "heat_transfer": {
            "overall_u": 1000.0,       # W/m²·K
            "area": 10.0,              # m²
            "lmtd": 44.3               # K
        }
    }


@pytest.fixture
def sample_valve_data():
    """
    阀门测试数据
    
    Returns:
        Dict: 测试数据
    """
    return {
        "inlet_pressure": 1000000.0,    # Pa
        "pressure_drop_ratio": 0.2,     # 20%
        "flow_rate": 100.0,             # m³/h
        "fluid_density": 1000.0,        # kg/m³
        "cv_calculation": {
            "pressure_drop": 100000.0,   # Pa
            "expected_cv": 31.62         # 预期Cv值
        }
    }


@pytest.fixture
def sample_splitter_data():
    """
    分离器测试数据
    
    Returns:
        Dict: 测试数据
    """
    return {
        "inlet_flow": 1000.0,        # kg/s
        "split_ratios": [0.6, 0.4],  # 60%, 40%
        "outlet_flows": [600.0, 400.0]  # kg/s
    }


@pytest.fixture
def mock_property_package():
    """
    模拟属性包
    
    Returns:
        Mock: 模拟的属性包对象
    """
    prop_pack = Mock()
    prop_pack.name = "MockPropertyPackage"
    prop_pack.components = ["H2O", "NaCl", "CO2"]
    prop_pack.calculate_properties = Mock(return_value=True)
    return prop_pack


@pytest.fixture
def performance_timer():
    """
    性能计时器fixture
    
    Returns:
        function: 计时器函数
    """
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture(params=[
    "Mixer", "Splitter", "Heater", "Cooler", 
    "Pump", "Compressor", "Valve", "HeatExchanger"
])
def unit_operation_type(request):
    """
    单元操作类型参数化fixture
    
    Returns:
        str: 单元操作类型
    """
    return request.param


@pytest.fixture(params=[
    PressureBehavior.MINIMUM,
    PressureBehavior.MAXIMUM, 
    PressureBehavior.AVERAGE
])
def pressure_behavior(request):
    """
    压力计算行为参数化fixture
    
    Returns:
        PressureBehavior: 压力计算行为
    """
    return request.param


@pytest.fixture
def calculation_error_threshold():
    """
    计算误差阈值
    
    Returns:
        Dict: 各种计算的误差阈值
    """
    return {
        "default": 1e-6,
        "temperature": 1e-3,  # K
        "pressure": 1e-2,     # Pa
        "flow": 1e-6,         # kg/s
        "enthalpy": 1e-3,     # kJ/kg
        "percentage": 1e-4    # 百分比
    }


# 用于性能测试的数据生成器
@pytest.fixture
def large_flowsheet_data():
    """
    大型流程图测试数据生成器
    
    Returns:
        Dict: 大型流程图数据
    """
    def generate_operations(count: int):
        operations = []
        for i in range(count):
            op_type = ["Mixer", "Heater", "Pump", "Cooler"][i % 4]
            operations.append({
                "type": op_type,
                "name": f"OP-{i:03d}",
                "description": f"操作{i}"
            })
        return operations
    
    return {
        "small": generate_operations(10),
        "medium": generate_operations(50),
        "large": generate_operations(100),
        "xlarge": generate_operations(200)
    } 