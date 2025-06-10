"""
热力学计算相关异常
"""

class ThermodynamicError(Exception):
    """热力学计算基础异常"""
    pass

class ComponentNotFoundError(ThermodynamicError):
    """组分未找到异常"""
    def __init__(self, component_name: str):
        self.component_name = component_name
        super().__init__(f"Component '{component_name}' not found in database")

class PropertyCalculationError(ThermodynamicError):
    """物性计算异常"""
    def __init__(self, property_name: str, message: str = ""):
        self.property_name = property_name
        super().__init__(f"Error calculating property '{property_name}': {message}")

class FlashCalculationError(ThermodynamicError):
    """闪蒸计算异常"""
    def __init__(self, message: str):
        super().__init__(f"Flash calculation error: {message}")

class ParameterOutOfRangeError(ThermodynamicError):
    """参数超出范围异常"""
    def __init__(self, parameter: str, value: float, valid_range: tuple):
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range
        super().__init__(f"Parameter '{parameter}' value {value} is outside valid range {valid_range}")

class PhaseNotFoundError(ThermodynamicError):
    """相态未找到异常"""
    def __init__(self, phase: str):
        self.phase = phase
        super().__init__(f"Phase '{phase}' not found or invalid")

class IncompatibleUnitsError(ThermodynamicError):
    """单位不兼容异常"""
    def __init__(self, from_unit: str, to_unit: str):
        self.from_unit = from_unit
        self.to_unit = to_unit
        super().__init__(f"Cannot convert from '{from_unit}' to '{to_unit}' - incompatible units") 