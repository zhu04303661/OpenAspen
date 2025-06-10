"""
单位转换器测试
"""

import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dwsim_core.utilities.unit_converter import UnitConverter
from dwsim_core.exceptions.thermodynamic_errors import IncompatibleUnitsError

class TestUnitConverter:
    """单位转换器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.converter = UnitConverter()
    
    def test_pressure_conversion(self):
        """测试压力转换"""
        # 1 atm = 101325 Pa
        result = self.converter.convert_pressure(101325, "pa", "atm")
        assert abs(result - 1.0) < 1e-6
        
        # 1 bar = 100000 Pa
        result = self.converter.convert_pressure(100000, "pa", "bar")
        assert abs(result - 1.0) < 1e-6
        
        # 1 atm = 14.6959 psi
        result = self.converter.convert_pressure(1, "atm", "psi")
        assert abs(result - 14.6959) < 0.01
    
    def test_temperature_conversion(self):
        """测试温度转换"""
        # 0°C = 273.15 K
        result = self.converter.convert_temperature(0, "c", "k")
        assert abs(result - 273.15) < 1e-6
        
        # 32°F = 0°C
        result = self.converter.convert_temperature(32, "f", "c")
        assert abs(result - 0) < 1e-6
        
        # 100°C = 212°F
        result = self.converter.convert_temperature(100, "c", "f")
        assert abs(result - 212) < 1e-6
    
    def test_energy_conversion(self):
        """测试能量转换"""
        # 1 kJ = 1000 J
        result = self.converter.convert_energy(1, "kj", "j")
        assert abs(result - 1000) < 1e-6
        
        # 1 cal = 4.184 J
        result = self.converter.convert_energy(1, "cal", "j")
        assert abs(result - 4.184) < 1e-6
    
    def test_invalid_units(self):
        """测试无效单位"""
        with pytest.raises(ValueError):
            self.converter.convert_pressure(100, "invalid_unit", "pa")
    
    def test_unit_compatibility(self):
        """测试单位兼容性"""
        assert self.converter.is_compatible("pa", "bar") == True
        assert self.converter.is_compatible("k", "c") == True
        assert self.converter.is_compatible("pa", "k") == False
    
    def test_supported_units(self):
        """测试支持的单位列表"""
        pressure_units = self.converter.get_supported_units("pressure")
        assert "pa" in pressure_units
        assert "bar" in pressure_units
        assert "atm" in pressure_units
        
        temp_units = self.converter.get_supported_units("temperature")
        assert "k" in temp_units
        assert "c" in temp_units
        assert "f" in temp_units 