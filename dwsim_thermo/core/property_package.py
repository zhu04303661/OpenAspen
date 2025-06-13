"""
DWSIM热力学计算库 - 物性包基类
==============================

定义了物性包的抽象基类，为所有热力学模型提供标准接口。
包含相平衡计算、热力学性质计算、输运性质计算等核心方法。

作者：OpenAspen项目组
版本：1.0.0
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from .enums import (
    PhaseType, FlashSpec, PackageType, PropertyType, 
    CalculationMode, ConvergenceStatus, MixingRule
)
from .compound import Compound
from .phase import Phase

@dataclass
class FlashResult:
    """闪蒸计算结果数据类"""
    converged: bool = False                  # 是否收敛
    iterations: int = 0                      # 迭代次数
    vapor_fraction: float = 0.0              # 汽化率
    
    # 相组成
    vapor_phase: Optional[Phase] = None      # 气相
    liquid_phase: Optional[Phase] = None     # 液相
    liquid2_phase: Optional[Phase] = None    # 液相2（液液分相）
    
    # 物性结果
    temperature: float = 298.15              # 温度 [K]
    pressure: float = 101325.0               # 压力 [Pa]
    enthalpy: float = 0.0                    # 摩尔焓 [J/mol]
    entropy: float = 0.0                     # 摩尔熵 [J/mol/K]
    gibbs_energy: float = 0.0                # 摩尔Gibbs自由能 [J/mol]
    volume: float = 0.0                      # 摩尔体积 [m³/mol]
    
    # 收敛信息
    convergence_status: ConvergenceStatus = ConvergenceStatus.NOT_STARTED
    residual: float = 1.0                    # 最终残差
    error_message: str = ""                  # 错误信息
    
    def get_phases(self) -> List[Phase]:
        """获取所有存在的相"""
        phases = []
        if self.vapor_phase is not None:
            phases.append(self.vapor_phase)
        if self.liquid_phase is not None:
            phases.append(self.liquid_phase)
        if self.liquid2_phase is not None:
            phases.append(self.liquid2_phase)
        return phases
    
    def get_phase_fractions(self) -> Dict[PhaseType, float]:
        """获取各相的摩尔分率"""
        fractions = {}
        
        if self.vapor_phase is not None:
            fractions[PhaseType.VAPOR] = self.vapor_fraction
        
        if self.liquid_phase is not None:
            fractions[PhaseType.LIQUID] = 1.0 - self.vapor_fraction
            
        if self.liquid2_phase is not None:
            # 对于液液分相，需要重新计算分配
            total_liquid = 1.0 - self.vapor_fraction
            # 这里简化处理，实际应该从计算结果中获取
            fractions[PhaseType.LIQUID] = total_liquid * 0.5
            fractions[PhaseType.LIQUID2] = total_liquid * 0.5
        
        return fractions

@dataclass
class PropertyPackageParameters:
    """物性包参数配置类"""
    
    # 数值计算参数
    tolerance: float = 1e-8                  # 收敛容差
    max_iterations: int = 100                # 最大迭代次数
    damping_factor: float = 1.0              # 阻尼因子
    
    # 闪蒸算法参数
    flash_tolerance: float = 1e-6            # 闪蒸收敛容差
    flash_max_iterations: int = 50           # 闪蒸最大迭代次数
    
    # 相稳定性参数
    stability_test: bool = True              # 是否进行相稳定性测试
    min_phase_fraction: float = 1e-8         # 最小相分率
    
    # 二元交互参数
    binary_interaction_parameters: Dict[tuple, float] = None
    
    def __post_init__(self):
        if self.binary_interaction_parameters is None:
            self.binary_interaction_parameters = {}

class PropertyPackage(ABC):
    """物性包抽象基类
    
    定义了所有热力学模型必须实现的标准接口。
    子类需要实现具体的热力学计算方法。
    """
    
    def __init__(
        self,
        package_type: PackageType,
        compounds: List[Compound],
        parameters: Optional[PropertyPackageParameters] = None
    ):
        """初始化物性包
        
        Args:
            package_type: 物性包类型
            compounds: 化合物列表
            parameters: 物性包参数，None时使用默认参数
        """
        self.package_type = package_type
        self.compounds = compounds.copy()
        self.n_components = len(compounds)
        
        if self.n_components == 0:
            raise ValueError("物性包必须包含至少一个组分")
        
        # 初始化参数
        self.parameters = parameters or PropertyPackageParameters()
        
        # 创建组分索引映射
        self.component_index = {comp.name: i for i, comp in enumerate(compounds)}
        
        # 二元交互参数矩阵
        self.binary_parameters = np.zeros((self.n_components, self.n_components))
        self._initialize_binary_parameters()
        
        # 计算缓存
        self._property_cache: Dict[str, Any] = {}
        self._last_conditions: Optional[tuple] = None
        
        # 统计信息
        self.calculation_stats = {
            "flash_calls": 0,
            "property_calls": 0,
            "cache_hits": 0,
            "convergence_failures": 0
        }
    
    @property
    def name(self) -> str:
        """物性包名称"""
        return self.package_type.value
    
    @property
    def component_names(self) -> List[str]:
        """组分名称列表"""
        return [comp.name for comp in self.compounds]
    
    def _initialize_binary_parameters(self):
        """初始化二元交互参数"""
        for (comp1, comp2), kij in self.parameters.binary_interaction_parameters.items():
            i = self.component_index.get(comp1)
            j = self.component_index.get(comp2)
            if i is not None and j is not None:
                self.binary_parameters[i, j] = kij
                self.binary_parameters[j, i] = kij
    
    def set_binary_parameter(self, comp1: str, comp2: str, kij: float):
        """设置二元交互参数
        
        Args:
            comp1: 组分1名称
            comp2: 组分2名称
            kij: 二元交互参数
        """
        i = self.component_index.get(comp1)
        j = self.component_index.get(comp2)
        
        if i is None:
            raise ValueError(f"未找到组分: {comp1}")
        if j is None:
            raise ValueError(f"未找到组分: {comp2}")
        
        self.binary_parameters[i, j] = kij
        self.binary_parameters[j, i] = kij
        
        # 清除相关缓存
        self._clear_cache()
    
    def get_binary_parameter(self, comp1: str, comp2: str) -> float:
        """获取二元交互参数
        
        Args:
            comp1: 组分1名称
            comp2: 组分2名称
            
        Returns:
            float: 二元交互参数
        """
        i = self.component_index.get(comp1)
        j = self.component_index.get(comp2)
        
        if i is None:
            raise ValueError(f"未找到组分: {comp1}")
        if j is None:
            raise ValueError(f"未找到组分: {comp2}")
        
        return self.binary_parameters[i, j]
    
    # ========== 抽象方法 - 子类必须实现 ==========
    
    @abstractmethod
    def calculate_fugacity_coefficient(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> np.ndarray:
        """计算逸度系数
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            np.ndarray: 逸度系数数组
        """
        pass
    
    @abstractmethod
    def calculate_activity_coefficient(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> np.ndarray:
        """计算活度系数
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            np.ndarray: 活度系数数组
        """
        pass
    
    @abstractmethod
    def calculate_compressibility_factor(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算压缩因子
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 压缩因子
        """
        pass
    
    @abstractmethod
    def calculate_enthalpy_departure(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算焓偏差
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 焓偏差 [J/mol]
        """
        pass
    
    @abstractmethod
    def calculate_entropy_departure(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算熵偏差
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 熵偏差 [J/mol/K]
        """
        pass
    
    # ========== 通用热力学性质计算方法 ==========
    
    def calculate_k_values(
        self,
        temperature: float,
        pressure: float,
        liquid_composition: np.ndarray,
        vapor_composition: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """计算平衡常数K值
        
        Args:
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            liquid_composition: 液相组成
            vapor_composition: 气相组成，None时使用液相组成初估
            
        Returns:
            np.ndarray: K值数组
        """
        # 创建相对象
        liquid_phase = Phase(PhaseType.LIQUID, self.compounds, liquid_composition)
        
        if vapor_composition is None:
            vapor_composition = liquid_composition
        vapor_phase = Phase(PhaseType.VAPOR, self.compounds, vapor_composition)
        
        # 设置温度压力
        liquid_phase.set_temperature_pressure(temperature, pressure)
        vapor_phase.set_temperature_pressure(temperature, pressure)
        
        # 计算逸度系数
        phi_l = self.calculate_fugacity_coefficient(liquid_phase, temperature, pressure)
        phi_v = self.calculate_fugacity_coefficient(vapor_phase, temperature, pressure)
        
        # K = (phi_l / phi_v) * (gamma_l / gamma_v)
        # 对于大多数模型，gamma_v = 1
        gamma_l = self.calculate_activity_coefficient(liquid_phase, temperature, pressure)
        
        k_values = (phi_l / phi_v) * gamma_l
        
        return k_values
    
    def calculate_enthalpy(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算焓
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 摩尔焓 [J/mol]
        """
        # 理想气体焓
        h_ig = self._calculate_ideal_gas_enthalpy(phase, temperature)
        
        # 焓偏差
        h_dep = self.calculate_enthalpy_departure(phase, temperature, pressure)
        
        return h_ig + h_dep
    
    def calculate_entropy(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算熵
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 摩尔熵 [J/mol/K]
        """
        # 理想气体熵
        s_ig = self._calculate_ideal_gas_entropy(phase, temperature, pressure)
        
        # 熵偏差
        s_dep = self.calculate_entropy_departure(phase, temperature, pressure)
        
        return s_ig + s_dep
    
    def calculate_gibbs_energy(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算Gibbs自由能
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 摩尔Gibbs自由能 [J/mol]
        """
        h = self.calculate_enthalpy(phase, temperature, pressure)
        s = self.calculate_entropy(phase, temperature, pressure)
        
        return h - temperature * s
    
    def calculate_density(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算密度
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 密度 [kg/m³]
        """
        z = self.calculate_compressibility_factor(phase, temperature, pressure)
        
        # 摩尔体积 V = ZRT/P
        R = 8.314  # J/mol/K
        v_molar = z * R * temperature / pressure  # m³/mol
        
        # 密度 = MW / V_molar
        mw = phase.molecular_weight  # kg/mol
        density = mw / v_molar  # kg/m³
        
        return density
    
    def _calculate_ideal_gas_enthalpy(self, phase: Phase, temperature: float) -> float:
        """计算理想气体焓"""
        # 参考温度
        T_ref = 298.15  # K
        
        # 各组分理想气体热容积分
        h_ig = 0.0
        for i, compound in enumerate(self.compounds):
            x_i = phase.mole_fractions[i]
            
            # 简化的焓计算：H = Cp_avg * (T - T_ref)
            cp_avg = compound.calculate_ideal_gas_cp(temperature)
            h_i = cp_avg * (temperature - T_ref)
            
            h_ig += x_i * h_i
        
        return h_ig
    
    def _calculate_ideal_gas_entropy(self, phase: Phase, temperature: float, pressure: float) -> float:
        """计算理想气体熵"""
        R = 8.314  # J/mol/K
        T_ref = 298.15  # K
        P_ref = 101325.0  # Pa
        
        s_ig = 0.0
        for i, compound in enumerate(self.compounds):
            x_i = phase.mole_fractions[i]
            
            # 简化的熵计算
            cp_avg = compound.calculate_ideal_gas_cp(temperature)
            s_i = cp_avg * np.log(temperature / T_ref) - R * np.log(pressure / P_ref)
            
            # 混合熵
            if x_i > 1e-12:
                s_i -= R * np.log(x_i)
            
            s_ig += x_i * s_i
        
        return s_ig
    
    # ========== 闪蒸计算接口 ==========
    
    def flash_pt(
        self,
        feed_composition: np.ndarray,
        pressure: float,
        temperature: float
    ) -> FlashResult:
        """PT闪蒸计算
        
        Args:
            feed_composition: 进料组成
            pressure: 压力 [Pa]
            temperature: 温度 [K]
            
        Returns:
            FlashResult: 闪蒸计算结果
        """
        self.calculation_stats["flash_calls"] += 1
        
        # 子类应该重写此方法以实现具体的闪蒸算法
        # 这里提供一个简单的默认实现
        result = FlashResult()
        result.temperature = temperature
        result.pressure = pressure
        result.converged = False
        result.error_message = "基类未实现PT闪蒸算法"
        result.convergence_status = ConvergenceStatus.ERROR
        
        return result
    
    def flash_ph(
        self,
        feed_composition: np.ndarray,
        pressure: float,
        enthalpy: float
    ) -> FlashResult:
        """PH闪蒸计算
        
        Args:
            feed_composition: 进料组成
            pressure: 压力 [Pa]
            enthalpy: 摩尔焓 [J/mol]
            
        Returns:
            FlashResult: 闪蒸计算结果
        """
        self.calculation_stats["flash_calls"] += 1
        
        # 默认实现：通过迭代求解温度
        result = FlashResult()
        result.pressure = pressure
        result.converged = False
        result.error_message = "基类未实现PH闪蒸算法"
        result.convergence_status = ConvergenceStatus.ERROR
        
        return result
    
    def flash_ps(
        self,
        feed_composition: np.ndarray,
        pressure: float,
        entropy: float
    ) -> FlashResult:
        """PS闪蒸计算
        
        Args:
            feed_composition: 进料组成
            pressure: 压力 [Pa]
            entropy: 摩尔熵 [J/mol/K]
            
        Returns:
            FlashResult: 闪蒸计算结果
        """
        self.calculation_stats["flash_calls"] += 1
        
        result = FlashResult()
        result.pressure = pressure
        result.converged = False
        result.error_message = "基类未实现PS闪蒸算法"
        result.convergence_status = ConvergenceStatus.ERROR
        
        return result
    
    # ========== 缓存管理 ==========
    
    def _get_cache_key(self, method_name: str, *args) -> str:
        """生成缓存键"""
        args_str = "_".join(str(arg) for arg in args)
        return f"{method_name}_{args_str}"
    
    def _clear_cache(self):
        """清除所有缓存"""
        self._property_cache.clear()
    
    # ========== 统计和诊断 ==========
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        stats = self.calculation_stats.copy()
        stats["cache_size"] = len(self._property_cache)
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.calculation_stats = {
            "flash_calls": 0,
            "property_calls": 0,
            "cache_hits": 0,
            "convergence_failures": 0
        }
    
    def validate_configuration(self) -> List[str]:
        """验证物性包配置
        
        Returns:
            List[str]: 验证错误信息列表
        """
        errors = []
        
        # 检查组分
        if self.n_components == 0:
            errors.append("物性包必须包含至少一个组分")
        
        # 检查组分物性数据
        for compound in self.compounds:
            comp_errors = compound.properties.validate()
            if comp_errors:
                errors.extend([f"组分{compound.name}: {err}" for err in comp_errors])
        
        # 检查参数
        if self.parameters.tolerance <= 0:
            errors.append("收敛容差必须大于0")
        
        if self.parameters.max_iterations <= 0:
            errors.append("最大迭代次数必须大于0")
        
        return errors
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"PropertyPackage(type={self.package_type.value}, n_comp={self.n_components})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        comp_names = [comp.name for comp in self.compounds]
        return (f"PropertyPackage(type={self.package_type.value}, "
                f"compounds={comp_names})")

__all__ = [
    "FlashResult",
    "PropertyPackageParameters", 
    "PropertyPackage"
] 