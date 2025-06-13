"""
DWSIM热力学计算库 - 化合物类
================================

定义了化合物类和纯组分物性数据类，用于管理化学组分的基础物性数据
和热力学关联参数。

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from .enums import ComponentType, PhaseType, DatabaseType

@dataclass
class PureComponentProperties:
    """纯组分物性数据类
    
    包含化合物的所有基础物性数据和热力学关联参数。
    数据来源包括DIPPR、NIST等权威数据库。
    """
    
    # 基础信息
    name: str = ""                          # 化合物名称
    formula: str = ""                       # 分子式
    cas_number: str = ""                    # CAS登记号
    molecular_weight: float = 0.0           # 分子量 [kg/mol]
    
    # 临界常数
    critical_temperature: float = 0.0       # 临界温度 [K]
    critical_pressure: float = 0.0          # 临界压力 [Pa]
    critical_volume: float = 0.0            # 临界体积 [m³/mol]
    critical_compressibility: float = 0.0   # 临界压缩因子 [-]
    
    # 偏心因子和其他参数
    acentric_factor: float = 0.0            # 偏心因子 [-]
    dipole_moment: float = 0.0              # 偶极矩 [Debye]
    radius_of_gyration: float = 0.0         # 回转半径 [m]
    
    # 正常沸点和熔点
    normal_boiling_point: float = 0.0       # 正常沸点 [K]
    normal_melting_point: float = 0.0       # 正常熔点 [K]
    triple_point_temperature: float = 0.0   # 三相点温度 [K]
    triple_point_pressure: float = 0.0      # 三相点压力 [Pa]
    
    # 热力学关联参数
    # Antoine方程参数 (log10(P_mmHg) = A - B/(C + T_K))
    antoine_a: float = 0.0
    antoine_b: float = 0.0
    antoine_c: float = 0.0
    antoine_tmin: float = 0.0               # Antoine方程适用温度下限 [K]
    antoine_tmax: float = 0.0               # Antoine方程适用温度上限 [K]
    
    # 理想气体热容关联 (Cp = A + B*T + C*T² + D*T³ + E*T⁴) [J/mol/K]
    cp_ig_a: float = 0.0
    cp_ig_b: float = 0.0
    cp_ig_c: float = 0.0
    cp_ig_d: float = 0.0
    cp_ig_e: float = 0.0
    cp_ig_tmin: float = 0.0                 # 热容关联适用温度下限 [K]
    cp_ig_tmax: float = 0.0                 # 热容关联适用温度上限 [K]
    
    # 液体密度关联 (Rackett方程参数)
    rackett_a: float = 0.0
    rackett_b: float = 0.0
    rackett_c: float = 0.0
    rackett_d: float = 0.0
    
    # 粘度关联参数
    viscosity_a: float = 0.0
    viscosity_b: float = 0.0
    viscosity_c: float = 0.0
    viscosity_d: float = 0.0
    
    # 导热系数关联参数
    thermal_conductivity_a: float = 0.0
    thermal_conductivity_b: float = 0.0
    thermal_conductivity_c: float = 0.0
    thermal_conductivity_d: float = 0.0
    
    # 表面张力关联参数
    surface_tension_a: float = 0.0
    surface_tension_b: float = 0.0
    surface_tension_c: float = 0.0
    surface_tension_d: float = 0.0
    
    # UNIFAC参数
    unifac_r: float = 0.0                   # UNIFAC体积参数
    unifac_q: float = 0.0                   # UNIFAC表面积参数
    unifac_groups: Dict[int, int] = field(default_factory=dict)  # UNIFAC基团组成
    
    # UNIQUAC参数
    uniquac_r: float = 0.0                  # UNIQUAC体积参数
    uniquac_q: float = 0.0                  # UNIQUAC表面积参数
    
    # PC-SAFT参数
    pc_saft_m: float = 0.0                  # 链段数
    pc_saft_sigma: float = 0.0              # 硬球直径 [Å]
    pc_saft_epsilon_k: float = 0.0          # 能量参数 [K]
    pc_saft_kappa_ab: float = 0.0           # 缔合体积参数
    pc_saft_epsilon_ab: float = 0.0         # 缔合能量参数 [K]
    
    # 环境和安全数据
    flash_point: float = 0.0                # 闪点 [K]
    auto_ignition_temperature: float = 0.0  # 自燃温度 [K]
    lower_flammability_limit: float = 0.0   # 可燃下限 [vol%]
    upper_flammability_limit: float = 0.0   # 可燃上限 [vol%]
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.unifac_groups:
            self.unifac_groups = {}
    
    def validate(self) -> List[str]:
        """验证物性数据的完整性和合理性
        
        Returns:
            List[str]: 验证错误信息列表
        """
        errors = []
        
        # 检查必需的基础数据
        if not self.name:
            errors.append("化合物名称不能为空")
        if not self.cas_number:
            errors.append("CAS号不能为空")
        if self.molecular_weight <= 0:
            errors.append("分子量必须大于0")
            
        # 检查临界常数
        if self.critical_temperature <= 0:
            errors.append("临界温度必须大于0")
        if self.critical_pressure <= 0:
            errors.append("临界压力必须大于0")
        if self.critical_volume <= 0:
            errors.append("临界体积必须大于0")
            
        # 检查物理合理性
        if self.normal_boiling_point > self.critical_temperature:
            errors.append("正常沸点不能高于临界温度")
        if self.normal_melting_point > self.normal_boiling_point:
            errors.append("熔点不能高于沸点")
            
        return errors

class Compound:
    """化合物类
    
    用于管理化学组分的完整信息，包括基础物性数据、状态变量
    和计算方法。是热力学计算的基础对象。
    """
    
    def __init__(
        self,
        name: str,
        cas: Optional[str] = None,
        formula: Optional[str] = None,
        component_type: ComponentType = ComponentType.NORMAL,
        database_source: DatabaseType = DatabaseType.BUILT_IN
    ):
        """初始化化合物对象
        
        Args:
            name: 化合物名称
            cas: CAS登记号
            formula: 分子式
            component_type: 组分类型
            database_source: 数据库来源
        """
        self.name = name
        self.cas_number = cas or ""
        self.formula = formula or ""
        self.component_type = component_type
        self.database_source = database_source
        
        # 初始化物性数据
        self.properties = PureComponentProperties(
            name=name,
            cas_number=self.cas_number,
            formula=formula or ""
        )
        
        # 状态变量
        self._temperature = 298.15          # 当前温度 [K]
        self._pressure = 101325.0           # 当前压力 [Pa]
        
        # 缓存计算结果
        self._property_cache: Dict[str, Any] = {}
        self._cache_conditions: Dict[str, tuple] = {}
        
        # 用户自定义参数
        self.user_properties: Dict[str, Any] = {}
        self.notes = ""
        
    @property
    def temperature(self) -> float:
        """当前温度 [K]"""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float):
        """设置温度并清除相关缓存"""
        if value <= 0:
            raise ValueError("温度必须大于0 K")
        if abs(value - self._temperature) > 1e-6:
            self._temperature = value
            self._clear_temperature_dependent_cache()
    
    @property
    def pressure(self) -> float:
        """当前压力 [Pa]"""
        return self._pressure
    
    @pressure.setter
    def pressure(self, value: float):
        """设置压力并清除相关缓存"""
        if value <= 0:
            raise ValueError("压力必须大于0 Pa")
        if abs(value - self._pressure) > 1e-6:
            self._pressure = value
            self._clear_pressure_dependent_cache()
    
    def set_state(self, temperature: float, pressure: float):
        """同时设置温度和压力
        
        Args:
            temperature: 温度 [K]
            pressure: 压力 [Pa]
        """
        if temperature <= 0:
            raise ValueError("温度必须大于0 K")
        if pressure <= 0:
            raise ValueError("压力必须大于0 Pa")
            
        self._temperature = temperature
        self._pressure = pressure
        self._property_cache.clear()
        self._cache_conditions.clear()
    
    def load_properties_from_database(self, database: DatabaseType = None) -> bool:
        """从数据库加载物性数据
        
        Args:
            database: 指定数据库类型，None时使用默认数据库
            
        Returns:
            bool: 是否成功加载
        """
        if database:
            self.database_source = database
            
        # 这里应该连接到实际的数据库
        # 目前返回一些示例数据
        if self.cas_number == "7732-18-5":  # 水
            self._load_water_properties()
            return True
        elif self.cas_number == "74-82-8":  # 甲烷
            self._load_methane_properties()
            return True
        elif self.cas_number == "74-84-0":  # 乙烷
            self._load_ethane_properties()
            return True
            
        return False
    
    def _load_water_properties(self):
        """加载水的物性数据"""
        props = self.properties
        props.molecular_weight = 0.018015  # kg/mol
        props.critical_temperature = 647.1  # K
        props.critical_pressure = 22064000  # Pa
        props.critical_volume = 0.000056  # m³/mol
        props.acentric_factor = 0.3449
        props.normal_boiling_point = 373.15  # K
        props.normal_melting_point = 273.15  # K
        
        # Antoine方程参数
        props.antoine_a = 8.07131
        props.antoine_b = 1730.63
        props.antoine_c = 233.426
        props.antoine_tmin = 273.15
        props.antoine_tmax = 647.1
        
        # 理想气体热容参数
        props.cp_ig_a = 33.363
        props.cp_ig_b = -0.00795
        props.cp_ig_c = 2.873e-5
        props.cp_ig_d = -1.283e-8
        props.cp_ig_e = 0.0
    
    def _load_methane_properties(self):
        """加载甲烷的物性数据"""
        props = self.properties
        props.molecular_weight = 0.016043  # kg/mol
        props.critical_temperature = 190.56  # K
        props.critical_pressure = 4599200  # Pa
        props.critical_volume = 0.000099  # m³/mol
        props.acentric_factor = 0.0115
        props.normal_boiling_point = 111.67  # K
        
        # Antoine方程参数
        props.antoine_a = 6.69561
        props.antoine_b = 405.42
        props.antoine_c = 267.777
        props.antoine_tmin = 90.69
        props.antoine_tmax = 190.56
        
        # 理想气体热容参数
        props.cp_ig_a = 19.875
        props.cp_ig_b = 0.05024
        props.cp_ig_c = 1.269e-5
        props.cp_ig_d = -1.1e-8
        props.cp_ig_e = 0.0
    
    def _load_ethane_properties(self):
        """加载乙烷的物性数据"""
        props = self.properties
        props.molecular_weight = 0.030070  # kg/mol
        props.critical_temperature = 305.32  # K
        props.critical_pressure = 4872200  # Pa
        props.critical_volume = 0.0001455  # m³/mol
        props.acentric_factor = 0.0995
        props.normal_boiling_point = 184.55  # K
        
        # Antoine方程参数
        props.antoine_a = 6.80266
        props.antoine_b = 656.4
        props.antoine_c = 256.58
        props.antoine_tmin = 89.89
        props.antoine_tmax = 305.32
        
        # 理想气体热容参数
        props.cp_ig_a = 6.900
        props.cp_ig_b = 0.17255
        props.cp_ig_c = -6.406e-5
        props.cp_ig_d = 7.285e-9
        props.cp_ig_e = 0.0
    
    def calculate_vapor_pressure(self, temperature: float = None) -> float:
        """计算饱和蒸汽压
        
        Args:
            temperature: 温度 [K]，None时使用当前温度
            
        Returns:
            float: 饱和蒸汽压 [Pa]
        """
        T = temperature if temperature is not None else self.temperature
        
        # 使用缓存
        cache_key = f"vapor_pressure_{T}"
        if cache_key in self._property_cache:
            return self._property_cache[cache_key]
        
        # Antoine方程计算
        props = self.properties
        if props.antoine_a != 0 and props.antoine_b != 0:
            if props.antoine_tmin <= T <= props.antoine_tmax:
                log_p_mmhg = props.antoine_a - props.antoine_b / (props.antoine_c + T)
                p_mmhg = 10 ** log_p_mmhg
                p_pa = p_mmhg * 133.322  # mmHg to Pa
                
                self._property_cache[cache_key] = p_pa
                return p_pa
        
        # 如果Antoine参数不可用，使用简化的Clausius-Clapeyron方程
        if props.normal_boiling_point > 0:
            if props.critical_temperature > 0:
                tr = T / props.critical_temperature
                pr = np.exp(5.373 * (1 + props.acentric_factor) * (1 - 1/tr))
                p_pa = pr * props.critical_pressure
                
                self._property_cache[cache_key] = p_pa
                return p_pa
        
        raise ValueError(f"无法计算{self.name}在{T}K下的饱和蒸汽压：缺少必要参数")
    
    def calculate_ideal_gas_cp(self, temperature: float = None) -> float:
        """计算理想气体热容
        
        Args:
            temperature: 温度 [K]，None时使用当前温度
            
        Returns:
            float: 理想气体热容 [J/mol/K]
        """
        T = temperature if temperature is not None else self.temperature
        
        # 使用缓存
        cache_key = f"cp_ig_{T}"
        if cache_key in self._property_cache:
            return self._property_cache[cache_key]
        
        props = self.properties
        if props.cp_ig_a != 0:
            cp = (props.cp_ig_a + 
                  props.cp_ig_b * T + 
                  props.cp_ig_c * T**2 + 
                  props.cp_ig_d * T**3 + 
                  props.cp_ig_e * T**4)
            
            self._property_cache[cache_key] = cp
            return cp
        
        # 如果没有关联参数，使用简单估算
        cp = 8.314 * (3.5 + 0.001 * T)  # 简单估算
        self._property_cache[cache_key] = cp
        return cp
    
    def calculate_reduced_properties(self, temperature: float = None, 
                                   pressure: float = None) -> tuple:
        """计算对比温度和对比压力
        
        Args:
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            tuple: (对比温度, 对比压力)
        """
        T = temperature if temperature is not None else self.temperature
        P = pressure if pressure is not None else self.pressure
        
        props = self.properties
        if props.critical_temperature <= 0 or props.critical_pressure <= 0:
            raise ValueError(f"缺少{self.name}的临界常数")
        
        tr = T / props.critical_temperature
        pr = P / props.critical_pressure
        
        return tr, pr
    
    def get_property_summary(self) -> Dict[str, Any]:
        """获取物性数据摘要
        
        Returns:
            Dict[str, Any]: 物性数据摘要
        """
        props = self.properties
        summary = {
            "基础信息": {
                "名称": props.name,
                "分子式": props.formula,
                "CAS号": props.cas_number,
                "分子量": f"{props.molecular_weight:.6f} kg/mol",
            },
            "临界常数": {
                "临界温度": f"{props.critical_temperature:.2f} K",
                "临界压力": f"{props.critical_pressure:.0f} Pa",
                "临界体积": f"{props.critical_volume*1000:.2f} cm³/mol",
                "偏心因子": f"{props.acentric_factor:.4f}",
            },
            "正常沸点": f"{props.normal_boiling_point:.2f} K",
            "正常熔点": f"{props.normal_melting_point:.2f} K",
            "当前状态": {
                "温度": f"{self.temperature:.2f} K",
                "压力": f"{self.pressure:.0f} Pa",
            }
        }
        
        return summary
    
    def _clear_temperature_dependent_cache(self):
        """清除温度相关的缓存"""
        keys_to_remove = [k for k in self._property_cache.keys() 
                         if any(prop in k for prop in ['vapor_pressure', 'cp_ig', 'viscosity'])]
        for key in keys_to_remove:
            del self._property_cache[key]
    
    def _clear_pressure_dependent_cache(self):
        """清除压力相关的缓存"""
        keys_to_remove = [k for k in self._property_cache.keys() 
                         if any(prop in k for prop in ['density', 'fugacity'])]
        for key in keys_to_remove:
            del self._property_cache[key]
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Compound(name='{self.name}', formula='{self.formula}', MW={self.properties.molecular_weight:.4f})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"Compound(name='{self.name}', cas='{self.cas_number}', "
                f"formula='{self.formula}', type={self.component_type.value})")

# 预定义的常用化合物
def create_common_compounds() -> Dict[str, Compound]:
    """创建常用化合物的字典
    
    Returns:
        Dict[str, Compound]: 常用化合物字典
    """
    compounds = {}
    
    # 水
    water = Compound("水", cas="7732-18-5", formula="H2O")
    water.load_properties_from_database()
    compounds["水"] = water
    compounds["H2O"] = water
    compounds["water"] = water
    
    # 甲烷
    methane = Compound("甲烷", cas="74-82-8", formula="CH4")
    methane.load_properties_from_database()
    compounds["甲烷"] = methane
    compounds["CH4"] = methane
    compounds["methane"] = methane
    
    # 乙烷
    ethane = Compound("乙烷", cas="74-84-0", formula="C2H6")
    ethane.load_properties_from_database()
    compounds["乙烷"] = ethane
    compounds["C2H6"] = ethane
    compounds["ethane"] = ethane
    
    return compounds

# 全局常用化合物实例
COMMON_COMPOUNDS = create_common_compounds()

__all__ = [
    "PureComponentProperties",
    "Compound", 
    "COMMON_COMPOUNDS",
    "create_common_compounds"
] 