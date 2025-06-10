"""
单位转换器
提供各种物理量的单位转换功能
"""

from typing import Dict, Union
from .constants import PhysicalConstants

class UnitConverter:
    """单位转换器类"""
    
    def __init__(self):
        self.pressure_units = {
            'pa': 1.0,
            'kpa': 1000.0,
            'mpa': 1000000.0,
            'bar': 100000.0,
            'atm': 101325.0,
            'psi': 6894.757,
            'mmhg': 133.322,
            'torr': 133.322
        }
        
        self.temperature_units = {
            'k': 'kelvin',
            'c': 'celsius',
            'f': 'fahrenheit',
            'r': 'rankine'
        }
        
        self.energy_units = {
            'j': 1.0,
            'kj': 1000.0,
            'cal': 4.184,
            'kcal': 4184.0,
            'btu': 1055.056,
            'kwh': 3600000.0
        }
        
        self.volume_units = {
            'm3': 1.0,
            'l': 0.001,
            'ml': 0.000001,
            'ft3': 0.028316846592,
            'gal': 0.003785411784
        }
        
        self.mass_units = {
            'kg': 1.0,
            'g': 0.001,
            'lb': 0.45359237,
            'oz': 0.028349523125,
            'ton': 1000.0
        }
    
    def convert_pressure(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        压力单位转换
        
        Args:
            value: 待转换的数值
            from_unit: 原单位
            to_unit: 目标单位
            
        Returns:
            转换后的数值
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.pressure_units:
            raise ValueError(f"Unknown pressure unit: {from_unit}")
        if to_unit not in self.pressure_units:
            raise ValueError(f"Unknown pressure unit: {to_unit}")
        
        # 转换为Pa，再转换为目标单位
        pa_value = value * self.pressure_units[from_unit]
        result = pa_value / self.pressure_units[to_unit]
        
        return result
    
    def convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        温度单位转换
        
        Args:
            value: 待转换的数值
            from_unit: 原单位 (K, C, F, R)
            to_unit: 目标单位
            
        Returns:
            转换后的数值
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # 先转换为Kelvin
        if from_unit == 'k':
            kelvin_value = value
        elif from_unit == 'c':
            kelvin_value = value + 273.15
        elif from_unit == 'f':
            kelvin_value = (value - 32) * 5/9 + 273.15
        elif from_unit == 'r':
            kelvin_value = value * 5/9
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # 从Kelvin转换为目标单位
        if to_unit == 'k':
            result = kelvin_value
        elif to_unit == 'c':
            result = kelvin_value - 273.15
        elif to_unit == 'f':
            result = (kelvin_value - 273.15) * 9/5 + 32
        elif to_unit == 'r':
            result = kelvin_value * 9/5
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
            
        return result
    
    def convert_energy(self, value: float, from_unit: str, to_unit: str) -> float:
        """能量单位转换"""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.energy_units:
            raise ValueError(f"Unknown energy unit: {from_unit}")
        if to_unit not in self.energy_units:
            raise ValueError(f"Unknown energy unit: {to_unit}")
        
        # 转换为焦耳，再转换为目标单位
        joule_value = value * self.energy_units[from_unit]
        result = joule_value / self.energy_units[to_unit]
        
        return result
    
    def convert_volume(self, value: float, from_unit: str, to_unit: str) -> float:
        """体积单位转换"""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.volume_units:
            raise ValueError(f"Unknown volume unit: {from_unit}")
        if to_unit not in self.volume_units:
            raise ValueError(f"Unknown volume unit: {to_unit}")
        
        # 转换为立方米，再转换为目标单位
        m3_value = value * self.volume_units[from_unit]
        result = m3_value / self.volume_units[to_unit]
        
        return result
    
    def convert_mass(self, value: float, from_unit: str, to_unit: str) -> float:
        """质量单位转换"""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit not in self.mass_units:
            raise ValueError(f"Unknown mass unit: {from_unit}")
        if to_unit not in self.mass_units:
            raise ValueError(f"Unknown mass unit: {to_unit}")
        
        # 转换为千克，再转换为目标单位
        kg_value = value * self.mass_units[from_unit]
        result = kg_value / self.mass_units[to_unit]
        
        return result
    
    def convert_molar_energy(self, value: float, from_unit: str, to_unit: str) -> float:
        """摩尔能量单位转换 (J/mol, kJ/mol, cal/mol等)"""
        # 提取能量部分和摩尔部分
        if '/mol' not in from_unit or '/mol' not in to_unit:
            raise ValueError("Molar energy units must contain '/mol'")
        
        from_energy = from_unit.replace('/mol', '').strip()
        to_energy = to_unit.replace('/mol', '').strip()
        
        return self.convert_energy(value, from_energy, to_energy)
    
    def get_supported_units(self, quantity: str) -> list:
        """
        获取支持的单位列表
        
        Args:
            quantity: 物理量类型 (pressure, temperature, energy, volume, mass)
            
        Returns:
            支持的单位列表
        """
        quantity = quantity.lower()
        
        if quantity == 'pressure':
            return list(self.pressure_units.keys())
        elif quantity == 'temperature':
            return list(self.temperature_units.keys())
        elif quantity == 'energy':
            return list(self.energy_units.keys())
        elif quantity == 'volume':
            return list(self.volume_units.keys())
        elif quantity == 'mass':
            return list(self.mass_units.keys())
        else:
            raise ValueError(f"Unknown quantity type: {quantity}")
    
    def is_compatible(self, unit1: str, unit2: str) -> bool:
        """
        检查两个单位是否兼容（可以相互转换）
        
        Args:
            unit1: 单位1
            unit2: 单位2
            
        Returns:
            是否兼容
        """
        # 检查每个单位类型
        for unit_dict in [self.pressure_units, self.temperature_units, 
                         self.energy_units, self.volume_units, self.mass_units]:
            if unit1.lower() in unit_dict and unit2.lower() in unit_dict:
                return True
        
        return False 