"""
DWSIM热力学计算库 - 相类
========================

定义了相类，用于管理混合物中各相的组成、物性和热力学状态。
支持气相、液相、固相等多种相态。

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from .enums import PhaseType, PropertyType
from .compound import Compound

@dataclass
class PhaseProperties:
    """相物性数据类"""
    temperature: float = 298.15          # 温度 [K]
    pressure: float = 101325.0           # 压力 [Pa]
    volume: float = 0.0                  # 摩尔体积 [m³/mol]
    enthalpy: float = 0.0                # 摩尔焓 [J/mol]
    entropy: float = 0.0                 # 摩尔熵 [J/mol/K]
    gibbs_energy: float = 0.0            # 摩尔Gibbs自由能 [J/mol]
    helmholtz_energy: float = 0.0        # 摩尔Helmholtz自由能 [J/mol]
    internal_energy: float = 0.0         # 摩尔内能 [J/mol]
    heat_capacity_cp: float = 0.0        # 定压热容 [J/mol/K]
    heat_capacity_cv: float = 0.0        # 定容热容 [J/mol/K]
    density: float = 0.0                 # 密度 [kg/m³]
    compressibility_factor: float = 1.0  # 压缩因子 [-]
    
    # 输运性质
    viscosity: float = 0.0               # 粘度 [Pa·s]
    thermal_conductivity: float = 0.0    # 导热系数 [W/m/K]
    surface_tension: float = 0.0         # 表面张力 [N/m]

class Phase:
    """相类
    
    用于管理混合物中单个相的所有信息，包括组成、物性、
    逸度系数、活度系数等热力学性质。
    """
    
    def __init__(
        self,
        phase_type: PhaseType,
        compounds: List[Compound],
        mole_fractions: Optional[List[float]] = None,
        name: Optional[str] = None
    ):
        """初始化相对象
        
        Args:
            phase_type: 相态类型
            compounds: 化合物列表
            mole_fractions: 摩尔分数列表，None时自动均分
            name: 相名称，None时自动生成
        """
        self.phase_type = phase_type
        self.compounds = compounds.copy()
        self.n_components = len(compounds)
        
        # 验证组分数量
        if self.n_components == 0:
            raise ValueError("相中必须包含至少一个组分")
        
        # 初始化摩尔分数
        if mole_fractions is None:
            self.mole_fractions = np.ones(self.n_components) / self.n_components
        else:
            if len(mole_fractions) != self.n_components:
                raise ValueError("摩尔分数数量必须与组分数量一致")
            self.mole_fractions = np.array(mole_fractions, dtype=float)
            self._normalize_composition()
        
        # 设置相名称
        self.name = name or f"{phase_type.value}相"
        
        # 初始化物性
        self.properties = PhaseProperties()
        
        # 组分物性数据
        self.component_properties = {}
        self.fugacity_coefficients = np.ones(self.n_components)
        self.activity_coefficients = np.ones(self.n_components)
        
        # 计算缓存
        self._property_cache: Dict[str, Any] = {}
        self._last_calc_conditions = None
        
        # 质量相关属性
        self._molecular_weight = 0.0
        self._mass_fractions = np.zeros(self.n_components)
        self._update_mass_properties()
    
    @property
    def composition(self) -> np.ndarray:
        """获取摩尔分数组成"""
        return self.mole_fractions.copy()
    
    @composition.setter
    def composition(self, mole_fractions: Union[List[float], np.ndarray]):
        """设置摩尔分数组成"""
        if len(mole_fractions) != self.n_components:
            raise ValueError("摩尔分数数量必须与组分数量一致")
        
        self.mole_fractions = np.array(mole_fractions, dtype=float)
        self._normalize_composition()
        self._update_mass_properties()
        self._clear_composition_dependent_cache()
    
    def _normalize_composition(self):
        """标准化摩尔分数组成"""
        total = np.sum(self.mole_fractions)
        if total <= 0:
            raise ValueError("摩尔分数总和必须大于0")
        
        if abs(total - 1.0) > 1e-12:
            self.mole_fractions = self.mole_fractions / total
    
    def _update_mass_properties(self):
        """更新质量相关属性"""
        # 计算平均分子量
        mw_array = np.array([comp.properties.molecular_weight for comp in self.compounds])
        self._molecular_weight = np.sum(self.mole_fractions * mw_array)
        
        # 计算质量分数
        if self._molecular_weight > 0:
            self._mass_fractions = (self.mole_fractions * mw_array) / self._molecular_weight
        else:
            self._mass_fractions = self.mole_fractions.copy()
    
    @property
    def molecular_weight(self) -> float:
        """相的平均分子量 [kg/mol]"""
        return self._molecular_weight
    
    @property
    def mass_fractions(self) -> np.ndarray:
        """质量分数组成"""
        return self._mass_fractions.copy()
    
    def get_component_mole_fraction(self, compound_name: str) -> float:
        """获取指定组分的摩尔分数
        
        Args:
            compound_name: 化合物名称
            
        Returns:
            float: 摩尔分数
        """
        for i, comp in enumerate(self.compounds):
            if comp.name == compound_name:
                return self.mole_fractions[i]
        
        raise ValueError(f"未找到组分: {compound_name}")
    
    def set_component_mole_fraction(self, compound_name: str, mole_fraction: float):
        """设置指定组分的摩尔分数
        
        Args:
            compound_name: 化合物名称
            mole_fraction: 摩尔分数
        """
        for i, comp in enumerate(self.compounds):
            if comp.name == compound_name:
                self.mole_fractions[i] = mole_fraction
                self._normalize_composition()
                self._update_mass_properties()
                self._clear_composition_dependent_cache()
                return
        
        raise ValueError(f"未找到组分: {compound_name}")
    
    def add_component(self, compound: Compound, mole_fraction: float = 0.0):
        """添加新组分
        
        Args:
            compound: 化合物对象
            mole_fraction: 摩尔分数
        """
        # 检查是否已存在
        for comp in self.compounds:
            if comp.name == compound.name:
                raise ValueError(f"组分{compound.name}已存在")
        
        # 添加组分
        self.compounds.append(compound)
        self.n_components += 1
        
        # 更新摩尔分数数组
        new_fractions = np.zeros(self.n_components)
        new_fractions[:-1] = self.mole_fractions
        new_fractions[-1] = mole_fraction
        
        self.mole_fractions = new_fractions
        self._normalize_composition()
        self._update_mass_properties()
        
        # 更新其他数组
        self.fugacity_coefficients = np.ones(self.n_components)
        self.activity_coefficients = np.ones(self.n_components)
        
        self._clear_all_cache()
    
    def remove_component(self, compound_name: str):
        """移除组分
        
        Args:
            compound_name: 化合物名称
        """
        if self.n_components <= 1:
            raise ValueError("相中必须保留至少一个组分")
        
        # 找到要移除的组分索引
        remove_index = None
        for i, comp in enumerate(self.compounds):
            if comp.name == compound_name:
                remove_index = i
                break
        
        if remove_index is None:
            raise ValueError(f"未找到组分: {compound_name}")
        
        # 移除组分
        self.compounds.pop(remove_index)
        self.n_components -= 1
        
        # 更新摩尔分数数组
        new_fractions = np.delete(self.mole_fractions, remove_index)
        self.mole_fractions = new_fractions
        self._normalize_composition()
        self._update_mass_properties()
        
        # 更新其他数组
        self.fugacity_coefficients = np.delete(self.fugacity_coefficients, remove_index)
        self.activity_coefficients = np.delete(self.activity_coefficients, remove_index)
        
        self._clear_all_cache()
    
    def set_temperature_pressure(self, temperature: float, pressure: float):
        """设置相的温度和压力
        
        Args:
            temperature: 温度 [K]
            pressure: 压力 [Pa]
        """
        if temperature <= 0:
            raise ValueError("温度必须大于0 K")
        if pressure <= 0:
            raise ValueError("压力必须大于0 Pa")
        
        self.properties.temperature = temperature
        self.properties.pressure = pressure
        
        # 更新所有组分的状态
        for compound in self.compounds:
            compound.set_state(temperature, pressure)
        
        self._clear_state_dependent_cache()
    
    def calculate_mixing_property(self, property_name: str, 
                                 mixing_rule: str = "linear") -> float:
        """计算混合物性
        
        Args:
            property_name: 物性名称
            mixing_rule: 混合规则 ("linear", "logarithmic", "volume_weighted")
            
        Returns:
            float: 混合物性值
        """
        if self.n_components == 1:
            # 纯组分情况
            compound = self.compounds[0]
            if hasattr(compound, f"calculate_{property_name}"):
                return getattr(compound, f"calculate_{property_name}")()
        
        # 获取组分物性
        component_values = []
        for compound in self.compounds:
            if hasattr(compound, f"calculate_{property_name}"):
                value = getattr(compound, f"calculate_{property_name}")()
                component_values.append(value)
            else:
                raise ValueError(f"组分{compound.name}不支持计算{property_name}")
        
        component_values = np.array(component_values)
        
        # 应用混合规则
        if mixing_rule == "linear":
            # 线性混合规则
            return np.sum(self.mole_fractions * component_values)
        
        elif mixing_rule == "logarithmic":
            # 对数混合规则（适用于活度系数等）
            log_values = np.log(np.maximum(component_values, 1e-12))
            return np.exp(np.sum(self.mole_fractions * log_values))
        
        elif mixing_rule == "volume_weighted":
            # 体积加权混合规则
            volumes = np.array([comp.properties.critical_volume for comp in self.compounds])
            weights = self.mole_fractions * volumes
            weights = weights / np.sum(weights)
            return np.sum(weights * component_values)
        
        else:
            raise ValueError(f"不支持的混合规则: {mixing_rule}")
    
    def calculate_ideal_gas_cp(self) -> float:
        """计算相的理想气体热容
        
        Returns:
            float: 理想气体热容 [J/mol/K]
        """
        cache_key = f"cp_ig_{self.properties.temperature}"
        if cache_key in self._property_cache:
            return self._property_cache[cache_key]
        
        cp_values = []
        for compound in self.compounds:
            cp = compound.calculate_ideal_gas_cp(self.properties.temperature)
            cp_values.append(cp)
        
        cp_mix = np.sum(self.mole_fractions * np.array(cp_values))
        self._property_cache[cache_key] = cp_mix
        
        return cp_mix
    
    def calculate_average_molecular_weight(self) -> float:
        """计算平均分子量（与molecular_weight属性相同）
        
        Returns:
            float: 平均分子量 [kg/mol]
        """
        return self.molecular_weight
    
    def get_critical_properties(self) -> Dict[str, float]:
        """获取混合物的临界性质
        
        Returns:
            Dict[str, float]: 临界性质字典
        """
        # 使用线性混合规则估算临界性质
        tc_values = np.array([comp.properties.critical_temperature for comp in self.compounds])
        pc_values = np.array([comp.properties.critical_pressure for comp in self.compounds])
        vc_values = np.array([comp.properties.critical_volume for comp in self.compounds])
        omega_values = np.array([comp.properties.acentric_factor for comp in self.compounds])
        
        tc_mix = np.sum(self.mole_fractions * tc_values)
        pc_mix = np.sum(self.mole_fractions * pc_values)
        vc_mix = np.sum(self.mole_fractions * vc_values)
        omega_mix = np.sum(self.mole_fractions * omega_values)
        
        return {
            "critical_temperature": tc_mix,
            "critical_pressure": pc_mix,
            "critical_volume": vc_mix,
            "acentric_factor": omega_mix
        }
    
    def validate_composition(self) -> List[str]:
        """验证组成的有效性
        
        Returns:
            List[str]: 验证错误信息列表
        """
        errors = []
        
        # 检查摩尔分数
        if np.any(self.mole_fractions < 0):
            errors.append("摩尔分数不能为负数")
        
        if np.any(self.mole_fractions > 1):
            errors.append("摩尔分数不能大于1")
        
        total = np.sum(self.mole_fractions)
        if abs(total - 1.0) > 1e-6:
            errors.append(f"摩尔分数总和应为1，当前为{total:.6f}")
        
        # 检查组分物性
        for i, compound in enumerate(self.compounds):
            comp_errors = compound.properties.validate()
            if comp_errors:
                errors.extend([f"组分{compound.name}: {err}" for err in comp_errors])
        
        return errors
    
    def get_composition_summary(self) -> Dict[str, Any]:
        """获取组成摘要
        
        Returns:
            Dict[str, Any]: 组成摘要
        """
        components = []
        for i, compound in enumerate(self.compounds):
            components.append({
                "名称": compound.name,
                "分子式": compound.formula,
                "摩尔分数": f"{self.mole_fractions[i]:.6f}",
                "质量分数": f"{self.mass_fractions[i]:.6f}",
                "分子量": f"{compound.properties.molecular_weight:.4f} kg/mol"
            })
        
        summary = {
            "相类型": self.phase_type.value,
            "相名称": self.name,
            "组分数量": self.n_components,
            "平均分子量": f"{self.molecular_weight:.4f} kg/mol",
            "温度": f"{self.properties.temperature:.2f} K",
            "压力": f"{self.properties.pressure:.0f} Pa",
            "组分详情": components
        }
        
        return summary
    
    def copy(self) -> 'Phase':
        """创建相的深拷贝
        
        Returns:
            Phase: 新的相对象
        """
        new_phase = Phase(
            phase_type=self.phase_type,
            compounds=self.compounds,  # Compound对象会被复制引用
            mole_fractions=self.mole_fractions.copy(),
            name=f"{self.name}_copy"
        )
        
        # 复制物性
        new_phase.properties = PhaseProperties(
            temperature=self.properties.temperature,
            pressure=self.properties.pressure,
            volume=self.properties.volume,
            enthalpy=self.properties.enthalpy,
            entropy=self.properties.entropy,
            gibbs_energy=self.properties.gibbs_energy,
            helmholtz_energy=self.properties.helmholtz_energy,
            internal_energy=self.properties.internal_energy,
            heat_capacity_cp=self.properties.heat_capacity_cp,
            heat_capacity_cv=self.properties.heat_capacity_cv,
            density=self.properties.density,
            compressibility_factor=self.properties.compressibility_factor,
            viscosity=self.properties.viscosity,
            thermal_conductivity=self.properties.thermal_conductivity,
            surface_tension=self.properties.surface_tension
        )
        
        # 复制系数
        new_phase.fugacity_coefficients = self.fugacity_coefficients.copy()
        new_phase.activity_coefficients = self.activity_coefficients.copy()
        
        return new_phase
    
    def _clear_composition_dependent_cache(self):
        """清除组成相关的缓存"""
        keys_to_remove = [k for k in self._property_cache.keys() 
                         if any(prop in k for prop in ['mixing', 'average', 'cp_ig'])]
        for key in keys_to_remove:
            del self._property_cache[key]
    
    def _clear_state_dependent_cache(self):
        """清除状态相关的缓存"""
        keys_to_remove = [k for k in self._property_cache.keys() 
                         if any(prop in k for prop in ['fugacity', 'activity', 'density'])]
        for key in keys_to_remove:
            del self._property_cache[key]
    
    def _clear_all_cache(self):
        """清除所有缓存"""
        self._property_cache.clear()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Phase(type={self.phase_type.value}, n_comp={self.n_components}, MW={self.molecular_weight:.4f})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        comp_names = [comp.name for comp in self.compounds]
        return (f"Phase(type={self.phase_type.value}, compounds={comp_names}, "
                f"composition={self.mole_fractions.tolist()})")

__all__ = [
    "PhaseProperties",
    "Phase"
] 