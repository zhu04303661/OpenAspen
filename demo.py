#!/usr/bin/env python3
"""
DWSIM Python 演示脚本
展示核心功能的使用方法
"""

import sys
import os
import json

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dwsim_core.interfaces.base_interfaces import PhaseType, PropertyType
from dwsim_core.utilities.constants import PhysicalConstants
from dwsim_core.utilities.unit_converter import UnitConverter

def demo_constants():
    """演示物理常数使用"""
    print("=== 物理常数演示 ===")
    print(f"通用气体常数 R = {PhysicalConstants.R} J/(mol·K)")
    print(f"标准大气压 = {PhysicalConstants.ATM_PRESSURE} Pa")
    print(f"水的沸点 = {PhysicalConstants.WATER_BOILING_POINT} K")
    
    # 获取不同单位的气体常数
    R_bar = PhysicalConstants.get_R_in_units("bar", "L")
    print(f"气体常数 (bar·L/mol·K) = {R_bar}")
    print()

def demo_unit_converter():
    """演示单位转换"""
    print("=== 单位转换演示 ===")
    converter = UnitConverter()
    
    # 压力转换
    pressure_pa = 101325  # 1 atm in Pa
    pressure_bar = converter.convert_pressure(pressure_pa, "pa", "bar")
    pressure_psi = converter.convert_pressure(pressure_pa, "pa", "psi")
    
    print(f"压力转换:")
    print(f"  {pressure_pa} Pa = {pressure_bar:.3f} bar = {pressure_psi:.2f} psi")
    
    # 温度转换
    temp_k = 298.15  # 25°C in K
    temp_c = converter.convert_temperature(temp_k, "k", "c")
    temp_f = converter.convert_temperature(temp_k, "k", "f")
    
    print(f"温度转换:")
    print(f"  {temp_k} K = {temp_c:.2f} °C = {temp_f:.2f} °F")
    
    # 检查单位兼容性
    compatible = converter.is_compatible("pa", "bar")
    print(f"Pa 和 bar 是否兼容: {compatible}")
    print()

def demo_component_database():
    """演示组分数据库功能"""
    print("=== 组分数据库演示 ===")
    
    # 读取组分数据
    try:
        with open("dwsim_data/assets/databases/dwsim_components.json", "r") as f:
            database = json.load(f)
        
        print(f"数据库版本: {database['database_info']['version']}")
        print(f"组分数量: {len(database['components'])}")
        print()
        
        # 显示甲烷的物性数据
        methane = None
        for comp in database['components']:
            if comp['name'] == 'Methane':
                methane = comp
                break
        
        if methane:
            print("甲烷 (CH4) 物性数据:")
            print(f"  CAS号: {methane['cas_number']}")
            print(f"  分子量: {methane['molecular_weight']} g/mol")
            print(f"  临界温度: {methane['critical_temperature']} K")
            print(f"  临界压力: {methane['critical_pressure']} Pa")
            print(f"  偏心因子: {methane['acentric_factor']}")
            print(f"  Antoine系数: {methane['antoine_coefficients']}")
            print()
            
    except FileNotFoundError:
        print("组分数据库文件未找到")
        print()

def demo_property_types():
    """演示属性类型枚举"""
    print("=== 物性类型演示 ===")
    print("支持的物性类型:")
    for prop in PropertyType:
        print(f"  - {prop.value}")
    print()
    
    print("支持的相态类型:")
    for phase in PhaseType:
        print(f"  - {phase.value}")
    print()

def demo_calculations():
    """演示基本计算功能"""
    print("=== 基本计算演示 ===")
    
    # Antoine方程计算蒸汽压
    def antoine_vapor_pressure(A, B, C, T):
        """使用Antoine方程计算蒸汽压"""
        return 10**(A - B/(T + C))
    
    # 甲烷在298.15K的蒸汽压
    T = 298.15  # K
    A, B, C = 15.2243, 897.84, -7.16  # 甲烷的Antoine系数
    vapor_pressure = antoine_vapor_pressure(A, B, C, T)
    
    print(f"甲烷在 {T} K 时的蒸汽压:")
    print(f"  蒸汽压 = {vapor_pressure:.0f} Pa")
    
    # 转换为其他单位
    converter = UnitConverter()
    vp_bar = converter.convert_pressure(vapor_pressure, "pa", "bar")
    vp_atm = converter.convert_pressure(vapor_pressure, "pa", "atm")
    
    print(f"  蒸汽压 = {vp_bar:.3f} bar")
    print(f"  蒸汽压 = {vp_atm:.6f} atm")
    print()

def demo_error_handling():
    """演示异常处理"""
    print("=== 异常处理演示 ===")
    
    from dwsim_core.exceptions.thermodynamic_errors import ComponentNotFoundError, ParameterOutOfRangeError
    from dwsim_core.exceptions.convergence_errors import MaxIterationsError
    
    # 演示组分未找到异常
    try:
        raise ComponentNotFoundError("Unknown_Component")
    except ComponentNotFoundError as e:
        print(f"捕获异常: {e}")
    
    # 演示参数超出范围异常
    try:
        raise ParameterOutOfRangeError("temperature", -100, (0, 1000))
    except ParameterOutOfRangeError as e:
        print(f"捕获异常: {e}")
    
    # 演示收敛异常
    try:
        raise MaxIterationsError(100, 0.001)
    except MaxIterationsError as e:
        print(f"捕获异常: {e}")
    
    print()

def main():
    """主函数"""
    print("DWSIM Python 核心功能演示")
    print("=" * 50)
    print()
    
    # 运行各个演示模块
    demo_constants()
    demo_unit_converter()
    demo_component_database()
    demo_property_types()
    demo_calculations()
    demo_error_handling()
    
    print("演示完成！")
    print()
    print("下一步:")
    print("1. 运行完整的测试套件: pytest tests/")
    print("2. 启动API服务器: cd dwsim_api && python main.py")
    print("3. 访问API文档: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 