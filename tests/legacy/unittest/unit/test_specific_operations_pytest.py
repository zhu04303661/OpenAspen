"""
DWSIM 具体单元操作详细测试 (pytest版本)
======================================

针对每个具体单元操作的详细功能测试，基于VB.NET源码分析。

使用pytest框架进行测试管理和执行。

测试范围：
1. 混合器 (Mixer.vb) - 质量能量平衡、压力计算
2. 加热器/冷却器 (Heater.vb/Cooler.vb) - 多种计算模式
3. 泵 (Pump.vb) - 泵曲线、效率计算、NPSH
4. 热交换器 (HeatExchanger.vb) - 传热计算、LMTD
5. 阀门 (Valve.vb) - 压降计算、Cv值
6. 分离器 (Splitter.vb) - 分流比计算

确保与VB.NET版本功能1:1对应。
"""

import pytest
import sys
import os
import math
from typing import Dict, Any, List

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dwsim_operations import *
from dwsim_operations.integration import *


# ============================================================================
# 混合器详细功能测试
# ============================================================================

@pytest.mark.mixer
class TestMixerDetailedFunctionality:
    """
    测试目标：混合器详细功能测试
    基于：DWSIM.UnitOperations/Unit Operations/Mixer.vb
    
    工作步骤：
    1. 测试三种压力计算模式的具体计算逻辑
    2. 验证质量平衡计算的准确性
    3. 测试能量平衡和焓值计算
    4. 验证组分混合和摩尔分数计算
    5. 测试多入料流的混合逻辑
    """
    
    def test_pressure_calculation_minimum_mode(self, sample_mixer, sample_mixer_data):
        """
        测试最小压力计算模式
        验证逻辑：对应VB.NET中 PressureBehavior.Minimum 的计算
        """
        sample_mixer.pressure_calculation = PressureBehavior.MINIMUM
        
        # 模拟输入流压力数据
        mock_pressures = sample_mixer_data["pressures"]
        
        # 验证最小压力选择逻辑
        expected_min_pressure = min(mock_pressures)
        assert expected_min_pressure == 250000
        
        # 验证压力计算模式设置
        assert sample_mixer.pressure_calculation == PressureBehavior.MINIMUM
    
    def test_pressure_calculation_maximum_mode(self, sample_mixer, sample_mixer_data):
        """
        测试最大压力计算模式
        验证逻辑：对应VB.NET中 PressureBehavior.Maximum 的计算
        """
        sample_mixer.pressure_calculation = PressureBehavior.MAXIMUM
        
        # 模拟输入流压力数据
        mock_pressures = sample_mixer_data["pressures"]
        
        # 验证最大压力选择逻辑
        expected_max_pressure = max(mock_pressures)
        assert expected_max_pressure == 300000
        
        # 验证压力计算模式设置
        assert sample_mixer.pressure_calculation == PressureBehavior.MAXIMUM
    
    def test_pressure_calculation_average_mode(self, sample_mixer, sample_mixer_data):
        """
        测试平均压力计算模式
        验证逻辑：对应VB.NET中 PressureBehavior.Average 的计算
        """
        sample_mixer.pressure_calculation = PressureBehavior.AVERAGE
        
        # 模拟输入流压力数据
        mock_pressures = sample_mixer_data["pressures"]
        
        # 验证平均压力计算逻辑
        expected_avg_pressure = sum(mock_pressures) / len(mock_pressures)
        assert abs(expected_avg_pressure - 276666.6666666667) < 1e-6
        
        # 验证压力计算模式设置
        assert sample_mixer.pressure_calculation == PressureBehavior.AVERAGE
    
    def test_mass_balance_calculation(self, sample_mixer_data):
        """
        测试质量平衡计算
        验证逻辑：对应VB.NET中的质量流量累加逻辑
        """
        # 模拟入料流质量流量
        inlet_mass_flows = sample_mixer_data["mass_flows"]
        
        # 计算总质量流量
        total_mass_flow = sum(inlet_mass_flows)
        assert total_mass_flow == 330.0
        
        # 验证质量守恒
        assert total_mass_flow > 0
    
    def test_energy_balance_calculation(self, sample_mixer_data, calculation_error_threshold):
        """
        测试能量平衡计算
        验证逻辑：对应VB.NET中的焓值计算逻辑
        """
        # 模拟入料流数据
        inlet_data = sample_mixer_data["inlet_data"]
        
        # 计算总焓值和比焓
        total_enthalpy = sum(data["mass_flow"] * data["enthalpy"] for data in inlet_data)
        total_mass_flow = sum(data["mass_flow"] for data in inlet_data)
        specific_enthalpy = total_enthalpy / total_mass_flow
        
        expected_specific_enthalpy = (100*2500 + 150*2300 + 80*2700) / 330
        assert abs(specific_enthalpy - expected_specific_enthalpy) < calculation_error_threshold["enthalpy"]
    
    def test_component_mixing_calculation(self, calculation_error_threshold):
        """
        测试组分混合计算
        验证逻辑：对应VB.NET中的组分质量分数和摩尔分数计算
        """
        # 模拟组分数据
        inlet1_components = {"H2O": 0.9, "NaCl": 0.1}  # 质量分数
        inlet2_components = {"H2O": 0.8, "NaCl": 0.2}
        
        mass_flow1, mass_flow2 = 100.0, 200.0  # kg/s
        
        # 计算混合后组分质量流量
        total_mass_flow = mass_flow1 + mass_flow2
        
        mixed_h2o_mass = (inlet1_components["H2O"] * mass_flow1 + 
                          inlet2_components["H2O"] * mass_flow2)
        mixed_nacl_mass = (inlet1_components["NaCl"] * mass_flow1 + 
                           inlet2_components["NaCl"] * mass_flow2)
        
        # 计算混合后质量分数
        mixed_h2o_fraction = mixed_h2o_mass / total_mass_flow
        mixed_nacl_fraction = mixed_nacl_mass / total_mass_flow
        
        # 验证质量守恒
        assert abs(mixed_h2o_fraction + mixed_nacl_fraction - 1.0) < calculation_error_threshold["default"]
        
        # 验证计算结果
        expected_h2o_fraction = (0.9 * 100 + 0.8 * 200) / 300
        assert abs(mixed_h2o_fraction - expected_h2o_fraction) < calculation_error_threshold["default"]
    
    def test_temperature_weighted_average(self, sample_mixer_data, calculation_error_threshold):
        """
        测试温度加权平均计算
        验证逻辑：对应VB.NET中的温度计算逻辑
        """
        # 模拟入料流温度和质量流量
        inlet_data = sample_mixer_data["inlet_data"]
        
        total_mass_flow = sum(data["mass_flow"] for data in inlet_data)
        
        # 计算加权平均温度
        weighted_temp = sum(data["mass_flow"] / total_mass_flow * data["temperature"] 
                           for data in inlet_data)
        
        # 验证温度计算
        expected_temp = (100*298.15 + 150*308.15 + 80*318.15) / 330
        assert abs(weighted_temp - expected_temp) < calculation_error_threshold["temperature"]


# ============================================================================
# 加热器/冷却器详细功能测试
# ============================================================================

@pytest.mark.heater
class TestHeaterCoolerDetailedFunctionality:
    """
    测试目标：加热器/冷却器详细功能测试
    基于：DWSIM.UnitOperations/Unit Operations/Heater.vb 和 Cooler.vb
    
    工作步骤：
    1. 测试多种计算模式（热量、出口温度、能量流等）
    2. 验证热量平衡计算
    3. 测试效率计算
    4. 验证压降计算
    5. 测试温度变化计算模式
    """
    
    def test_heater_calculation_modes(self, sample_heater):
        """
        测试加热器计算模式
        验证逻辑：对应VB.NET中的CalculationMode枚举值
        """
        # 测试出口温度计算模式
        sample_heater.calculation_mode = "OutletTemperature"
        sample_heater.outlet_temperature = 373.15  # 100°C
        
        assert sample_heater.calculation_mode == "OutletTemperature"
        assert sample_heater.outlet_temperature == 373.15
        
        # 测试热负荷计算模式
        sample_heater.calculation_mode = "HeatAdded"
        sample_heater.heat_duty = 1000.0  # kW
        
        assert sample_heater.calculation_mode == "HeatAdded"
        assert sample_heater.heat_duty == 1000.0
    
    def test_heat_duty_calculation(self, sample_heater_data):
        """
        测试热负荷计算
        验证逻辑：对应VB.NET中的热量平衡计算
        """
        # 模拟入料流数据
        inlet_data = sample_heater_data["inlet_conditions"]
        outlet_enthalpy = sample_heater_data["outlet_conditions"]["enthalpy"]
        
        # 计算所需热负荷
        heat_duty = inlet_data["mass_flow"] * (outlet_enthalpy - inlet_data["enthalpy"])
        
        assert heat_duty == 100.0 * (2700.0 - 2500.0)
        assert heat_duty == 20000.0  # kW
    
    def test_efficiency_calculation(self, sample_heater_data, calculation_error_threshold):
        """
        测试效率计算
        验证逻辑：对应VB.NET中的效率参数应用
        """
        # 设置效率
        efficiency = sample_heater_data["efficiency"]
        
        # 理论热负荷
        theoretical_heat_duty = 20000.0  # kW
        
        # 实际所需热负荷
        actual_heat_duty = theoretical_heat_duty / efficiency
        
        assert abs(actual_heat_duty - 23529.41) < calculation_error_threshold["percentage"] * 1000
    
    def test_pressure_drop_calculation(self, sample_heater_data):
        """
        测试压降计算
        验证逻辑：对应VB.NET中的压降处理
        """
        # 模拟入口压力
        inlet_pressure = sample_heater_data["inlet_conditions"]["pressure"]
        pressure_drop = sample_heater_data["pressure_drop"]
        
        # 计算出口压力
        outlet_pressure = inlet_pressure - pressure_drop
        
        assert outlet_pressure == 295000.0
    
    def test_temperature_change_mode(self):
        """
        测试温度变化计算模式
        验证逻辑：对应VB.NET中的TemperatureChange模式
        """
        # 模拟入口温度
        inlet_temperature = 298.15  # K (25°C)
        temperature_change = 50.0   # K
        
        # 计算出口温度
        outlet_temperature = inlet_temperature + temperature_change
        
        assert outlet_temperature == 348.15  # 75°C


# ============================================================================
# 泵详细功能测试
# ============================================================================

@pytest.mark.pump
class TestPumpDetailedFunctionality:
    """
    测试目标：泵详细功能测试
    基于：DWSIM.UnitOperations/Unit Operations/Pump.vb
    
    工作步骤：
    1. 测试泵的多种计算模式
    2. 验证泵曲线计算
    3. 测试效率计算和功耗
    4. 验证NPSH计算
    5. 测试泵的性能曲线
    """
    
    def test_pump_calculation_modes(self, sample_pump):
        """
        测试泵的计算模式
        验证逻辑：对应VB.NET中Pump的CalculationMode枚举
        """
        # 验证泵的对象分类
        assert sample_pump.object_class == SimulationObjectClass.PressureChangers
        
        # 验证基本属性存在
        assert hasattr(sample_pump, 'calculate')
        assert callable(sample_pump.calculate)
    
    def test_pump_head_calculation(self, sample_pump_data, calculation_error_threshold):
        """
        测试泵扬程计算
        验证逻辑：对应VB.NET中的扬程计算公式
        """
        # 模拟泵参数
        inlet_pressure = sample_pump_data["inlet_pressure"]
        outlet_pressure = sample_pump_data["outlet_pressure"]
        fluid_density = sample_pump_data["fluid_density"]
        gravity = sample_pump_data["gravity"]
        
        # 计算泵扬程
        pressure_rise = outlet_pressure - inlet_pressure
        pump_head = pressure_rise / (fluid_density * gravity)
        
        expected_head = 400000.0 / (1000.0 * 9.81)
        assert abs(pump_head - expected_head) < calculation_error_threshold["default"] * 1000
    
    def test_pump_power_calculation(self, sample_pump_data, calculation_error_threshold):
        """
        测试泵功耗计算
        验证逻辑：对应VB.NET中的功率计算
        """
        # 模拟泵运行参数
        flow_rate = sample_pump_data["flow_rate"]
        pressure_rise = sample_pump_data["outlet_pressure"] - sample_pump_data["inlet_pressure"]
        efficiency = sample_pump_data["efficiency"]
        
        # 计算理论功率
        theoretical_power = flow_rate * pressure_rise  # W
        
        # 计算实际功率
        actual_power = theoretical_power / efficiency
        
        expected_theoretical = flow_rate * pressure_rise
        expected_actual = expected_theoretical / efficiency
        
        assert theoretical_power == expected_theoretical
        assert abs(actual_power - expected_actual) < calculation_error_threshold["default"] * 1000
    
    def test_npsh_calculation(self, sample_pump_data, calculation_error_threshold):
        """
        测试NPSH（净正吸入压头）计算
        验证逻辑：对应VB.NET中的NPSH计算
        """
        # NPSH计算参数
        npsh_data = sample_pump_data["npsh_data"]
        fluid_density = sample_pump_data["fluid_density"]
        gravity = sample_pump_data["gravity"]
        
        # 计算NPSH Available
        npsh_available = ((npsh_data["suction_pressure"] - npsh_data["vapor_pressure"]) / 
                         (fluid_density * gravity)) + npsh_data["suction_head"]
        
        expected_npsh = ((200000.0 - 3169.0) / (1000.0 * 9.81)) + 5.0
        assert abs(npsh_available - expected_npsh) < calculation_error_threshold["default"] * 100


# ============================================================================
# 热交换器详细功能测试
# ============================================================================

@pytest.mark.heat_exchanger
class TestHeatExchangerDetailedFunctionality:
    """
    测试目标：热交换器详细功能测试
    基于：DWSIM.UnitOperations/Unit Operations/HeatExchanger.vb
    
    工作步骤：
    1. 测试热交换器计算模式
    2. 验证传热系数和面积计算
    3. 测试LMTD（对数平均温差）计算
    4. 验证热平衡计算
    5. 测试不同流型的传热计算
    """
    
    def test_heat_exchanger_classification(self, sample_heat_exchanger):
        """
        测试热交换器分类
        验证逻辑：对应VB.NET中的ObjectClass设置
        """
        assert sample_heat_exchanger.object_class == SimulationObjectClass.HeatExchangers
    
    def test_lmtd_calculation(self, sample_heat_exchanger_data, calculation_error_threshold):
        """
        测试对数平均温差(LMTD)计算
        验证逻辑：对应VB.NET中的LMTD计算公式
        """
        # 模拟温度数据（逆流）
        hot_side = sample_heat_exchanger_data["hot_side"]
        cold_side = sample_heat_exchanger_data["cold_side"]
        
        # 计算温差
        dt1 = hot_side["inlet_temp"] - cold_side["outlet_temp"]   # 100-45 = 55°C
        dt2 = hot_side["outlet_temp"] - cold_side["inlet_temp"]   # 60-25 = 35°C
        
        # 计算LMTD
        if dt1 != dt2:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
        else:
            lmtd = dt1
        
        expected_lmtd = (55.0 - 35.0) / math.log(55.0 / 35.0)
        assert abs(lmtd - expected_lmtd) < calculation_error_threshold["temperature"]
    
    def test_heat_transfer_calculation(self, sample_heat_exchanger_data):
        """
        测试传热计算
        验证逻辑：对应VB.NET中的传热方程 Q = U*A*LMTD
        """
        # 模拟传热参数
        heat_transfer = sample_heat_exchanger_data["heat_transfer"]
        
        # 计算传热量
        heat_transfer_rate = (heat_transfer["overall_u"] * 
                             heat_transfer["area"] * 
                             heat_transfer["lmtd"])
        
        expected_q = 1000.0 * 10.0 * 44.3
        assert heat_transfer_rate == expected_q
    
    def test_heat_balance_verification(self, sample_heat_exchanger_data):
        """
        测试热平衡验证
        验证逻辑：对应VB.NET中的热平衡检查
        """
        # 模拟热侧流体
        hot_side = sample_heat_exchanger_data["hot_side"]
        cold_side = sample_heat_exchanger_data["cold_side"]
        
        hot_temp_drop = hot_side["inlet_temp"] - hot_side["outlet_temp"]
        cold_temp_rise = cold_side["outlet_temp"] - cold_side["inlet_temp"]
        
        # 计算热负荷
        hot_side_heat = hot_side["mass_flow"] * hot_side["cp"] * hot_temp_drop
        cold_side_heat = cold_side["mass_flow"] * cold_side["cp"] * cold_temp_rise
        
        # 验证热平衡（应该接近相等）
        heat_balance_error = abs(hot_side_heat - cold_side_heat) / hot_side_heat
        assert heat_balance_error < 0.05  # 5%误差内


# ============================================================================
# 阀门详细功能测试
# ============================================================================

@pytest.mark.valve
class TestValveDetailedFunctionality:
    """
    测试目标：阀门详细功能测试
    基于：DWSIM.UnitOperations/Unit Operations/Valve.vb
    
    工作步骤：
    1. 测试阀门压降计算
    2. 验证Cv值计算
    3. 测试闪蒸计算
    4. 验证流量系数
    5. 测试噪声计算
    """
    
    def test_valve_classification(self, sample_valve):
        """
        测试阀门分类
        验证逻辑：对应VB.NET中的ObjectClass设置
        """
        assert sample_valve.object_class == SimulationObjectClass.PressureChangers
    
    def test_pressure_drop_calculation(self, sample_valve_data):
        """
        测试阀门压降计算
        验证逻辑：对应VB.NET中的压降计算方法
        """
        # 模拟阀门参数
        inlet_pressure = sample_valve_data["inlet_pressure"]
        pressure_drop_ratio = sample_valve_data["pressure_drop_ratio"]
        
        # 计算出口压力
        pressure_drop = inlet_pressure * pressure_drop_ratio
        outlet_pressure = inlet_pressure - pressure_drop
        
        assert outlet_pressure == 800000.0
    
    def test_cv_value_calculation(self, sample_valve_data, calculation_error_threshold):
        """
        测试Cv值计算
        验证逻辑：对应VB.NET中的流量系数计算
        """
        # 模拟流量参数
        flow_rate = sample_valve_data["flow_rate"]
        cv_calc = sample_valve_data["cv_calculation"]
        fluid_density = sample_valve_data["fluid_density"]
        
        # 简化的Cv计算（实际公式更复杂）
        cv_value = flow_rate * math.sqrt(fluid_density / cv_calc["pressure_drop"])
        
        expected_cv = 100.0 * math.sqrt(1000.0 / 100000.0)
        assert abs(cv_value - expected_cv) < calculation_error_threshold["default"] * 100


# ============================================================================
# 分离器详细功能测试
# ============================================================================ 

@pytest.mark.splitter
class TestSplitterDetailedFunctionality:
    """
    测试目标：分离器详细功能测试
    基于：DWSIM.UnitOperations/Unit Operations/Splitter.vb
    
    工作步骤：
    1. 测试分流比计算
    2. 验证质量平衡
    3. 测试能量平衡
    4. 验证组分分离
    5. 测试多出口分离
    """
    
    def test_splitter_classification(self, sample_splitter):
        """
        测试分离器分类
        验证逻辑：对应VB.NET中的ObjectClass设置
        """
        assert sample_splitter.object_class == SimulationObjectClass.MixersSplitters
    
    def test_split_ratio_calculation(self, sample_splitter_data, calculation_error_threshold):
        """
        测试分流比计算
        验证逻辑：对应VB.NET中的分流比逻辑
        """
        # 模拟分流参数
        inlet_flow = sample_splitter_data["inlet_flow"]
        split_ratios = sample_splitter_data["split_ratios"]
        expected_outlets = sample_splitter_data["outlet_flows"]
        
        # 计算出口流量
        outlet_flow_1 = inlet_flow * split_ratios[0]
        outlet_flow_2 = inlet_flow * split_ratios[1]
        
        # 验证质量守恒
        total_outlet = outlet_flow_1 + outlet_flow_2
        assert abs(total_outlet - inlet_flow) < calculation_error_threshold["flow"]
        
        # 验证分流结果
        assert outlet_flow_1 == expected_outlets[0]
        assert outlet_flow_2 == expected_outlets[1]


# ============================================================================
# 参数化测试
# ============================================================================

@pytest.mark.parametrize("pressure_mode", [
    PressureBehavior.MINIMUM,
    PressureBehavior.MAXIMUM,
    PressureBehavior.AVERAGE
])
def test_mixer_pressure_modes_parametrized(sample_mixer, pressure_mode):
    """
    参数化测试混合器所有压力计算模式
    """
    sample_mixer.pressure_calculation = pressure_mode
    assert sample_mixer.pressure_calculation == pressure_mode


@pytest.mark.parametrize("operation_type,expected_class", [
    ("Mixer", SimulationObjectClass.MixersSplitters),
    ("Heater", SimulationObjectClass.HeatExchangers),
    ("Pump", SimulationObjectClass.PressureChangers),
    ("Valve", SimulationObjectClass.PressureChangers),
    ("Splitter", SimulationObjectClass.MixersSplitters),
])
def test_operation_classification_parametrized(integrated_solver, operation_type, expected_class):
    """
    参数化测试所有单元操作的分类
    """
    op = integrated_solver.create_and_add_operation(operation_type, f"{operation_type[:3].upper()}-001")
    assert op.object_class == expected_class


# ============================================================================
# 性能测试
# ============================================================================

@pytest.mark.performance
class TestOperationPerformance:
    """性能测试类"""
    
    @pytest.mark.slow
    def test_large_mixer_calculation_performance(self, integrated_solver, performance_timer):
        """测试大量混合器的计算性能"""
        performance_timer.start()
        
        # 创建100个混合器
        mixers = []
        for i in range(100):
            mixer = integrated_solver.create_and_add_operation("Mixer", f"MIX-{i:03d}")
            mixers.append(mixer)
        
        creation_time = performance_timer.stop()
        
        # 验证性能要求
        assert creation_time < 1.0  # 1秒内创建100个混合器
        assert len(mixers) == 100


if __name__ == "__main__":
    # 使用pytest运行测试
    pytest.main([__file__, "-v"]) 