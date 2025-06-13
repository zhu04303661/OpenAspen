"""
闪蒸算法基类
===========

基于DWSIM FlashAlgorithmBase.vb的Python实现
提供所有闪蒸算法的抽象基类和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from enum import Enum
import logging

from ..core.enums import FlashSpec, PhaseType
from ..core.compound import Compound
from ..core.phase import Phase as PhaseObj


class FlashConvergenceError(Exception):
    """闪蒸收敛失败异常"""
    def __init__(self, message: str, iterations: int, residual: float):
        super().__init__(message)
        self.iterations = iterations
        self.residual = residual


class FlashValidationError(Exception):
    """闪蒸验证失败异常"""
    pass


@dataclass
class FlashSettings:
    """闪蒸算法设置"""
    # PT闪蒸设置
    pt_max_external_iterations: int = 100
    pt_external_tolerance: float = 1e-4
    pt_max_internal_iterations: int = 100
    pt_internal_tolerance: float = 1e-4
    
    # PH闪蒸设置
    ph_max_external_iterations: int = 100
    ph_external_tolerance: float = 1e-4
    ph_max_internal_iterations: int = 100
    ph_internal_tolerance: float = 1e-4
    
    # PS闪蒸设置
    ps_max_external_iterations: int = 100
    ps_external_tolerance: float = 1e-4
    ps_max_internal_iterations: int = 100
    ps_internal_tolerance: float = 1e-4
    
    # PV闪蒸设置
    pv_fixed_damping_factor: float = 1.0
    pv_max_temperature_change: float = 10.0
    pv_temperature_derivative_epsilon: float = 0.1
    
    # 通用设置
    validate_equilibrium_calc: bool = False
    validation_gibbs_tolerance: float = 0.01
    use_phase_identification: bool = False
    calculate_bubble_dew_points: bool = False
    
    # 稳定性测试设置
    stability_test_severity: int = 0
    stability_test_comp_ids: List[str] = None
    check_incipient_liquid_stability: bool = False
    stability_random_tries: int = 20
    
    # 快速模式
    nested_loops_fast_mode: bool = True
    inside_out_fast_mode: bool = True
    
    # Gibbs最小化设置
    gibbs_optimization_method: str = "SLSQP"
    
    def __post_init__(self):
        if self.stability_test_comp_ids is None:
            self.stability_test_comp_ids = []


@dataclass
class FlashCalculationResult:
    """闪蒸计算结果"""
    # 基本信息
    flash_specification_1: FlashSpec
    flash_specification_2: FlashSpec
    flash_algorithm_type: str
    
    # 计算结果
    calculated_pressure: float = 0.0
    calculated_temperature: float = 0.0
    calculated_enthalpy: float = 0.0
    calculated_entropy: float = 0.0
    calculated_volume: float = 0.0
    
    # 相分布
    vapor_phase_mole_fraction: float = 0.0
    liquid1_phase_mole_fraction: float = 0.0
    liquid2_phase_mole_fraction: float = 0.0
    solid_phase_mole_fraction: float = 0.0
    
    # 相组成
    vapor_phase_mole_fractions: List[float] = None
    liquid1_phase_mole_fractions: List[float] = None
    liquid2_phase_mole_fractions: List[float] = None
    solid_phase_mole_fractions: List[float] = None
    
    # K值
    k_values: List[float] = None
    
    # 收敛信息
    iterations_taken: int = 0
    converged: bool = False
    residual: float = 0.0
    
    # 错误信息
    result_exception: Optional[Exception] = None
    
    # 诊断信息
    calculation_time: float = 0.0
    
    def __post_init__(self):
        if self.vapor_phase_mole_fractions is None:
            self.vapor_phase_mole_fractions = []
        if self.liquid1_phase_mole_fractions is None:
            self.liquid1_phase_mole_fractions = []
        if self.liquid2_phase_mole_fractions is None:
            self.liquid2_phase_mole_fractions = []
        if self.solid_phase_mole_fractions is None:
            self.solid_phase_mole_fractions = []
        if self.k_values is None:
            self.k_values = []
    
    def get_phase_mole_amounts(self, phase: str, total_moles: float = 1.0) -> List[float]:
        """获取指定相的摩尔量"""
        if phase.lower() == "vapor":
            return [x * self.vapor_phase_mole_fraction * total_moles 
                   for x in self.vapor_phase_mole_fractions]
        elif phase.lower() == "liquid1":
            return [x * self.liquid1_phase_mole_fraction * total_moles 
                   for x in self.liquid1_phase_mole_fractions]
        elif phase.lower() == "liquid2":
            return [x * self.liquid2_phase_mole_fraction * total_moles 
                   for x in self.liquid2_phase_mole_fractions]
        elif phase.lower() == "solid":
            return [x * self.solid_phase_mole_fraction * total_moles 
                   for x in self.solid_phase_mole_fractions]
        else:
            raise ValueError(f"Unknown phase: {phase}")


class FlashAlgorithmBase(ABC):
    """
    闪蒸算法抽象基类
    
    所有闪蒸算法都应继承此类并实现抽象方法
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.settings = FlashSettings()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._debug_mode = False
        
    def __str__(self) -> str:
        return self.name if self.name else self.__class__.__name__
    
    @property
    def debug_mode(self) -> bool:
        return self._debug_mode
    
    @debug_mode.setter
    def debug_mode(self, value: bool):
        self._debug_mode = value
        if value:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
    
    def write_debug_info(self, text: str):
        """写入调试信息"""
        if self._debug_mode:
            self.logger.debug(text)
    
    def calculate_equilibrium(self, 
                            spec1: FlashSpec, 
                            spec2: FlashSpec,
                            val1: float, 
                            val2: float,
                            property_package,
                            mixture_mole_fractions: List[float],
                            initial_k_values: Optional[List[float]] = None,
                            initial_estimate: float = 0.0) -> FlashCalculationResult:
        """
        计算相平衡的主入口函数
        
        Args:
            spec1: 第一个闪蒸规格
            spec2: 第二个闪蒸规格
            val1: 第一个规格的值
            val2: 第二个规格的值
            property_package: 物性包实例
            mixture_mole_fractions: 混合物摩尔分数
            initial_k_values: 初始K值估算
            initial_estimate: 初始温度或压力估算
            
        Returns:
            FlashCalculationResult: 闪蒸计算结果
        """
        import time
        start_time = time.time()
        
        # 创建结果对象
        result = FlashCalculationResult(
            flash_specification_1=spec1,
            flash_specification_2=spec2,
            flash_algorithm_type=self.__class__.__name__
        )
        
        try:
            # 验证输入
            self._validate_inputs(mixture_mole_fractions, val1, val2)
            
            # 根据闪蒸规格调用相应的方法
            if spec1 == FlashSpec.P and spec2 == FlashSpec.T:
                flash_result = self.flash_pt(mixture_mole_fractions, val1, val2, 
                                           property_package, initial_k_values)
                result.calculated_pressure = val1
                result.calculated_temperature = val2
                
            elif spec1 == FlashSpec.T and spec2 == FlashSpec.P:
                flash_result = self.flash_pt(mixture_mole_fractions, val2, val1, 
                                           property_package, initial_k_values)
                result.calculated_pressure = val2
                result.calculated_temperature = val1
                
            elif spec1 == FlashSpec.P and spec2 == FlashSpec.H:
                flash_result = self.flash_ph(mixture_mole_fractions, val1, val2, 
                                           property_package, initial_estimate)
                result.calculated_pressure = val1
                result.calculated_enthalpy = val2
                
            elif spec1 == FlashSpec.H and spec2 == FlashSpec.P:
                flash_result = self.flash_ph(mixture_mole_fractions, val2, val1, 
                                           property_package, initial_estimate)
                result.calculated_pressure = val2
                result.calculated_enthalpy = val1
                
            elif spec1 == FlashSpec.P and spec2 == FlashSpec.S:
                flash_result = self.flash_ps(mixture_mole_fractions, val1, val2, 
                                           property_package, initial_estimate)
                result.calculated_pressure = val1
                result.calculated_entropy = val2
                
            elif spec1 == FlashSpec.S and spec2 == FlashSpec.P:
                flash_result = self.flash_ps(mixture_mole_fractions, val2, val1, 
                                           property_package, initial_estimate)
                result.calculated_pressure = val2
                result.calculated_entropy = val1
                
            elif spec1 == FlashSpec.T and spec2 == FlashSpec.V:
                flash_result = self.flash_tv(mixture_mole_fractions, val1, val2, 
                                           property_package, initial_estimate)
                result.calculated_temperature = val1
                result.calculated_volume = val2
                
            elif spec1 == FlashSpec.V and spec2 == FlashSpec.T:
                flash_result = self.flash_tv(mixture_mole_fractions, val2, val1, 
                                           property_package, initial_estimate)
                result.calculated_temperature = val2
                result.calculated_volume = val1
                
            elif spec1 == FlashSpec.P and spec2 == FlashSpec.V:
                flash_result = self.flash_pv(mixture_mole_fractions, val1, val2, 
                                           property_package, initial_estimate)
                result.calculated_pressure = val1
                result.calculated_volume = val2
                
            elif spec1 == FlashSpec.V and spec2 == FlashSpec.P:
                flash_result = self.flash_pv(mixture_mole_fractions, val2, val1, 
                                           property_package, initial_estimate)
                result.calculated_pressure = val2
                result.calculated_volume = val1
                
            else:
                raise ValueError(f"Unsupported flash specification combination: {spec1}, {spec2}")
            
            # 解析闪蒸结果
            self._parse_flash_result(flash_result, result)
            
            # 验证结果
            if self.settings.validate_equilibrium_calc:
                self._validate_result(result, property_package)
            
            result.converged = True
            
        except Exception as e:
            result.result_exception = e
            result.converged = False
            self.logger.error(f"Flash calculation failed: {str(e)}")
        
        finally:
            result.calculation_time = time.time() - start_time
        
        return result
    
    @abstractmethod
    def flash_pt(self, z: List[float], P: float, T: float, 
                property_package, initial_k_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """PT闪蒸计算"""
        pass
    
    @abstractmethod
    def flash_ph(self, z: List[float], P: float, H: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """PH闪蒸计算"""
        pass
    
    @abstractmethod
    def flash_ps(self, z: List[float], P: float, S: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """PS闪蒸计算"""
        pass
    
    @abstractmethod
    def flash_tv(self, z: List[float], T: float, V: float, 
                property_package, initial_pressure: float = 0.0) -> Dict[str, Any]:
        """TV闪蒸计算"""
        pass
    
    @abstractmethod
    def flash_pv(self, z: List[float], P: float, V: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """PV闪蒸计算"""
        pass
    
    def _validate_inputs(self, z: List[float], val1: float, val2: float):
        """验证输入参数"""
        if not z or len(z) == 0:
            raise ValueError("Mixture composition cannot be empty")
        
        if abs(sum(z) - 1.0) > 1e-6:
            raise ValueError(f"Mixture composition must sum to 1.0, got {sum(z)}")
        
        if any(x < 0 for x in z):
            raise ValueError("Mixture composition cannot contain negative values")
        
        if val1 <= 0 or val2 <= 0:
            raise ValueError("Flash specification values must be positive")
    
    def _parse_flash_result(self, flash_result: Dict[str, Any], result: FlashCalculationResult):
        """解析闪蒸结果到结果对象"""
        # 标准闪蒸结果格式: {L1, V, Vx1, Vy, ecount, L2, Vx2, S, Vs}
        if isinstance(flash_result, dict):
            result.liquid1_phase_mole_fraction = flash_result.get('L1', 0.0)
            result.vapor_phase_mole_fraction = flash_result.get('V', 0.0)
            result.liquid2_phase_mole_fraction = flash_result.get('L2', 0.0)
            result.solid_phase_mole_fraction = flash_result.get('S', 0.0)
            
            result.liquid1_phase_mole_fractions = flash_result.get('Vx1', [])
            result.vapor_phase_mole_fractions = flash_result.get('Vy', [])
            result.liquid2_phase_mole_fractions = flash_result.get('Vx2', [])
            result.solid_phase_mole_fractions = flash_result.get('Vs', [])
            
            result.iterations_taken = flash_result.get('ecount', 0)
            result.residual = flash_result.get('residual', 0.0)
            
            # 计算K值
            if (result.liquid1_phase_mole_fractions and 
                result.vapor_phase_mole_fractions and
                len(result.liquid1_phase_mole_fractions) == len(result.vapor_phase_mole_fractions)):
                result.k_values = [
                    y / x if x > 1e-15 else 1e15 
                    for x, y in zip(result.liquid1_phase_mole_fractions, 
                                   result.vapor_phase_mole_fractions)
                ]
    
    def _validate_result(self, result: FlashCalculationResult, property_package):
        """验证闪蒸结果的热力学一致性"""
        # 检查物料平衡
        total_mole_fraction = (result.vapor_phase_mole_fraction + 
                              result.liquid1_phase_mole_fraction + 
                              result.liquid2_phase_mole_fraction + 
                              result.solid_phase_mole_fraction)
        
        if abs(total_mole_fraction - 1.0) > 1e-6:
            raise FlashValidationError(f"Material balance error: total fraction = {total_mole_fraction}")
        
        # 检查组成归一化
        for phase_name, compositions in [
            ("vapor", result.vapor_phase_mole_fractions),
            ("liquid1", result.liquid1_phase_mole_fractions),
            ("liquid2", result.liquid2_phase_mole_fractions),
            ("solid", result.solid_phase_mole_fractions)
        ]:
            if compositions and abs(sum(compositions) - 1.0) > 1e-6:
                raise FlashValidationError(f"{phase_name} composition sum = {sum(compositions)}")
    
    def calculate_mixture_enthalpy(self, T: float, P: float, 
                                 phase_fractions: Dict[str, float],
                                 phase_compositions: Dict[str, List[float]],
                                 property_package) -> float:
        """计算混合物焓"""
        total_enthalpy = 0.0
        
        for phase_name, fraction in phase_fractions.items():
            if fraction > 1e-15 and phase_name in phase_compositions:
                composition = phase_compositions[phase_name]
                if composition:
                    phase_enthalpy = property_package.calculate_enthalpy(
                        T, P, composition, phase_name)
                    total_enthalpy += fraction * phase_enthalpy
        
        return total_enthalpy
    
    def calculate_mixture_entropy(self, T: float, P: float, 
                                phase_fractions: Dict[str, float],
                                phase_compositions: Dict[str, List[float]],
                                property_package) -> float:
        """计算混合物熵"""
        total_entropy = 0.0
        
        for phase_name, fraction in phase_fractions.items():
            if fraction > 1e-15 and phase_name in phase_compositions:
                composition = phase_compositions[phase_name]
                if composition:
                    phase_entropy = property_package.calculate_entropy(
                        T, P, composition, phase_name)
                    total_entropy += fraction * phase_entropy
        
        return total_entropy
    
    def estimate_initial_k_values(self, z: List[float], P: float, T: float, 
                                property_package) -> List[float]:
        """估算初始K值"""
        k_values = []
        
        for i, compound in enumerate(property_package.compounds):
            # Wilson方程估算
            try:
                Tc = compound.properties.critical_temperature
                Pc = compound.properties.critical_pressure
                omega = compound.properties.acentric_factor
                
                Tr = T / Tc
                Pr = P / Pc
                
                # Wilson方程
                k = (Pc / P) * np.exp(5.37 * (1 + omega) * (1 - 1/Tr))
                k_values.append(max(k, 1e-10))
                
            except (AttributeError, ZeroDivisionError):
                # 如果缺少物性数据，使用默认值
                k_values.append(1.0)
        
        return k_values
    
    def check_phase_stability(self, z: List[float], T: float, P: float, 
                            property_package, phase: str = "liquid") -> bool:
        """检查相稳定性"""
        # 简化的相稳定性检查
        # 实际实现需要更复杂的Michelsen稳定性分析
        try:
            # 计算试探相的逸度系数
            if phase.lower() == "liquid":
                test_phase = "vapor"
            else:
                test_phase = "liquid"
            
            # 这里应该实现完整的稳定性分析
            # 暂时返回True表示稳定
            return True
            
        except Exception:
            return True
    
    def solve_rachford_rice(self, z: List[float], k: List[float], 
                          beta_min: float = 0.0, beta_max: float = 1.0) -> float:
        """求解Rachford-Rice方程"""
        from scipy.optimize import brentq
        
        def rr_equation(beta):
            return sum(z[i] * (k[i] - 1) / (1 + beta * (k[i] - 1)) 
                      for i in range(len(z)))
        
        try:
            # 检查边界条件
            f_min = rr_equation(beta_min)
            f_max = rr_equation(beta_max)
            
            if f_min * f_max > 0:
                # 没有根，返回边界值
                if abs(f_min) < abs(f_max):
                    return beta_min
                else:
                    return beta_max
            
            # 使用Brent方法求解
            beta = brentq(rr_equation, beta_min, beta_max, xtol=1e-12)
            return max(beta_min, min(beta_max, beta))
            
        except Exception as e:
            self.logger.warning(f"Rachford-Rice solver failed: {e}")
            return 0.5  # 返回中间值作为备选
    
    def update_phase_compositions(self, z: List[float], k: List[float], 
                                beta: float) -> Tuple[List[float], List[float]]:
        """更新相组成"""
        n_comp = len(z)
        x = [0.0] * n_comp  # 液相组成
        y = [0.0] * n_comp  # 气相组成
        
        for i in range(n_comp):
            denominator = 1 + beta * (k[i] - 1)
            if abs(denominator) > 1e-15:
                x[i] = z[i] / denominator
                y[i] = k[i] * x[i]
            else:
                x[i] = z[i]
                y[i] = z[i]
        
        # 归一化
        sum_x = sum(x)
        sum_y = sum(y)
        
        if sum_x > 1e-15:
            x = [xi / sum_x for xi in x]
        if sum_y > 1e-15:
            y = [yi / sum_y for yi in y]
        
        return x, y
    
    def check_convergence(self, k_old: List[float], k_new: List[float], 
                        tolerance: float = 1e-6) -> bool:
        """检查K值收敛"""
        if len(k_old) != len(k_new):
            return False
        
        max_error = 0.0
        for i in range(len(k_old)):
            if k_old[i] > 1e-15:
                error = abs((k_new[i] - k_old[i]) / k_old[i])
                max_error = max(max_error, error)
        
        return max_error < tolerance 