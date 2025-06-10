"""
物理常数定义
包含热力学计算中常用的物理常数
"""

import math

class PhysicalConstants:
    """物理常数类"""
    
    # 通用气体常数
    R = 8.314472  # J/(mol·K)
    R_ATM = 0.08205736  # atm·L/(mol·K) 
    R_BAR = 0.08314472  # bar·L/(mol·K)
    
    # 阿伏伽德罗常数
    AVOGADRO = 6.0221415e23  # mol^-1
    
    # 玻尔兹曼常数
    BOLTZMANN = 1.3806505e-23  # J/K
    
    # 普朗克常数
    PLANCK = 6.6260693e-34  # J·s
    
    # 光速
    LIGHT_SPEED = 2.99792458e8  # m/s
    
    # 标准重力加速度
    GRAVITY = 9.80665  # m/s²
    
    # 标准大气压
    ATM_PRESSURE = 101325  # Pa
    ATM_PRESSURE_BAR = 1.01325  # bar
    ATM_PRESSURE_ATM = 1.0  # atm
    
    # 标准温度
    STD_TEMPERATURE = 273.15  # K (0°C)
    
    # 冰点和沸点 (水, 1 atm)
    WATER_FREEZING_POINT = 273.15  # K
    WATER_BOILING_POINT = 373.15  # K
    
    # 数学常数
    PI = math.pi
    E = math.e
    
    # 单位转换因子
    class Units:
        """单位转换因子"""
        
        # 压力转换因子 (转换为Pa)
        PA_TO_PA = 1.0
        BAR_TO_PA = 1e5
        ATM_TO_PA = 101325
        PSI_TO_PA = 6894.757
        MMHG_TO_PA = 133.322
        KPA_TO_PA = 1e3
        MPA_TO_PA = 1e6
        
        # 温度转换 (K)
        def celsius_to_kelvin(celsius: float) -> float:
            return celsius + 273.15
        
        def fahrenheit_to_kelvin(fahrenheit: float) -> float:
            return (fahrenheit - 32) * 5/9 + 273.15
        
        def kelvin_to_celsius(kelvin: float) -> float:
            return kelvin - 273.15
        
        def kelvin_to_fahrenheit(kelvin: float) -> float:
            return (kelvin - 273.15) * 9/5 + 32
        
        # 能量转换因子 (转换为J)
        J_TO_J = 1.0
        KJ_TO_J = 1e3
        CAL_TO_J = 4.184
        KCAL_TO_J = 4184
        BTU_TO_J = 1055.056
        
        # 体积转换因子 (转换为m³)
        M3_TO_M3 = 1.0
        L_TO_M3 = 1e-3
        ML_TO_M3 = 1e-6
        FT3_TO_M3 = 0.028316846592
        
        # 质量转换因子 (转换为kg)
        KG_TO_KG = 1.0
        G_TO_KG = 1e-3
        LB_TO_KG = 0.45359237
        
    @classmethod
    def get_R_in_units(cls, pressure_unit: str = "Pa", volume_unit: str = "m3") -> float:
        """
        获取指定单位下的气体常数
        
        Args:
            pressure_unit: 压力单位 (Pa, bar, atm)
            volume_unit: 体积单位 (m3, L)
            
        Returns:
            对应单位的气体常数值
        """
        base_R = cls.R  # J/(mol·K) = Pa·m³/(mol·K)
        
        # 压力转换
        if pressure_unit.lower() == "bar":
            base_R /= cls.Units.BAR_TO_PA
        elif pressure_unit.lower() == "atm":
            base_R /= cls.Units.ATM_TO_PA
        
        # 体积转换
        if volume_unit.lower() == "l":
            base_R /= cls.Units.L_TO_M3
            
        return base_R 