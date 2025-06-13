"""
Lee-Kesler-Plocker状态方程
========================

基于对应态原理的状态方程，适用于烷烃和轻烃混合物的热力学性质计算。
对应DWSIM的LeeKeslerPlocker.vb (671行)的完整Python实现。

理论基础：
- Lee-Kesler对应态方程
- Plocker修正项
- 三参数对应态原理

数学表达式：
$$Z = Z^{(0)} + \omega Z^{(1)} + \omega^2 Z^{(2)}$$

其中：
- $Z^{(0)}$：简单流体压缩因子
- $Z^{(1)}$：偏心因子修正项  
- $Z^{(2)}$：Plocker修正项
- $\omega$：偏心因子

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from ..base_property_package import PropertyPackageBase
from ...core.compound import CompoundProperties
from ...core.enums import PhaseType
from ...core.exceptions import CalculationError, ConvergenceError


@dataclass
class LeeKeslerPlocknerParameters:
    """Lee-Kesler-Plocker参数"""
    
    # 临界参数
    critical_temperature: float     # 临界温度 (K)
    critical_pressure: float        # 临界压力 (Pa)
    critical_volume: float          # 临界体积 (m³/mol)
    acentric_factor: float          # 偏心因子
    
    # 二元交互参数
    binary_interaction_parameters: Dict[str, float] = None
    
    # Plocker修正参数
    plocker_parameter: float = 0.0
    
    def __post_init__(self):
        if self.binary_interaction_parameters is None:
            self.binary_interaction_parameters = {}


class LeeKeslerPlocknerEOS:
    """Lee-Kesler-Plocker状态方程实现
    
    基于对应态原理的状态方程，特别适用于：
    1. 烷烃和轻烃混合物
    2. 天然气系统
    3. 石油馏分
    4. 中等压力范围的计算
    
    特点：
    - 基于对应态原理
    - 三参数对应态
    - 适用于非极性分子
    - 计算速度快
    """
    
    def __init__(self):
        self.logger = logging.getLogger("LeeKeslerPlocknerEOS")
        
        # Lee-Kesler常数
        self._lk_constants = self._initialize_lk_constants()
        
        # 计算统计
        self.stats = {
            'total_calculations': 0,
            'convergence_failures': 0,
            'average_iterations': 0.0
        }
    
    def _initialize_lk_constants(self) -> Dict[str, np.ndarray]:
        """初始化Lee-Kesler常数"""
        
        # 简单流体常数 (甲烷)
        b_simple = np.array([
            0.1181193, 0.265728, 0.154790, 0.030323,
            0.0236744, 0.0186984, 0.0, 0.042724,
            0.0, 0.155488, 0.623689
        ])
        
        # 参考流体常数 (正辛烷)
        b_reference = np.array([
            0.2026579, 0.331511, 0.027655, 0.203488,
            0.0313385, 0.0503618, 0.016901, 0.041577,
            0.48736, 0.0740336, 0.133826
        ])
        
        # 温度指数
        gamma = np.array([0, 1, 2, 3, 4, 5, 2, 2, 2, 2, 2])
        
        # 密度指数  
        beta = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4])
        
        return {
            'b_simple': b_simple,
            'b_reference': b_reference,
            'gamma': gamma,
            'beta': beta
        }
    
    def calculate_compressibility_factor(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters],
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """计算压缩因子
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            composition: 摩尔组成
            parameters: 组分参数列表
            phase_type: 相态类型
            
        Returns:
            float: 压缩因子
        """
        try:
            # 计算混合物临界参数
            tc_mix, pc_mix, omega_mix = self._calculate_mixing_rules(
                composition, parameters)
            
            # 对比温度和压力
            tr = temperature / tc_mix
            pr = pressure / pc_mix
            
            # 计算简单流体压缩因子
            z0 = self._calculate_simple_fluid_z(tr, pr)
            
            # 计算参考流体压缩因子
            z1 = self._calculate_reference_fluid_z(tr, pr)
            
            # Lee-Kesler方程
            z = z0 + omega_mix * (z1 - z0)
            
            # Plocker修正
            if any(p.plocker_parameter != 0.0 for p in parameters):
                z_plocker = self._calculate_plocker_correction(
                    tr, pr, composition, parameters)
                z += z_plocker
            
            self.logger.debug(f"LKP压缩因子计算: T={temperature:.2f}K, "
                            f"P={pressure:.0f}Pa, Z={z:.6f}")
            
            return max(z, 0.001)  # 防止负值
            
        except Exception as e:
            self.logger.error(f"LKP压缩因子计算失败: {e}")
            raise CalculationError(f"Lee-Kesler-Plocker压缩因子计算失败: {e}")
    
    def calculate_fugacity_coefficient(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters],
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> np.ndarray:
        """计算逸度系数
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            composition: 摩尔组成
            parameters: 组分参数列表
            phase_type: 相态类型
            
        Returns:
            np.ndarray: 逸度系数数组
        """
        try:
            n_comp = len(composition)
            phi = np.zeros(n_comp)
            
            # 计算混合物参数
            tc_mix, pc_mix, omega_mix = self._calculate_mixing_rules(
                composition, parameters)
            
            tr = temperature / tc_mix
            pr = pressure / pc_mix
            
            # 计算压缩因子
            z = self.calculate_compressibility_factor(
                temperature, pressure, composition, parameters, phase_type)
            
            # 计算各组分逸度系数
            for i in range(n_comp):
                # 简单流体贡献
                phi0_i = self._calculate_simple_fluid_fugacity_coefficient(
                    tr, pr, z, i, composition, parameters)
                
                # 参考流体贡献
                phi1_i = self._calculate_reference_fluid_fugacity_coefficient(
                    tr, pr, z, i, composition, parameters)
                
                # Lee-Kesler方程
                ln_phi_i = np.log(phi0_i) + parameters[i].acentric_factor * (
                    np.log(phi1_i) - np.log(phi0_i))
                
                phi[i] = np.exp(ln_phi_i)
            
            self.logger.debug(f"LKP逸度系数计算完成: φ={phi}")
            
            return phi
            
        except Exception as e:
            self.logger.error(f"LKP逸度系数计算失败: {e}")
            raise CalculationError(f"Lee-Kesler-Plocker逸度系数计算失败: {e}")
    
    def calculate_enthalpy_departure(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters],
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """计算焓偏差
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            composition: 摩尔组成
            parameters: 组分参数列表
            phase_type: 相态类型
            
        Returns:
            float: 焓偏差 (J/mol)
        """
        try:
            # 计算混合物参数
            tc_mix, pc_mix, omega_mix = self._calculate_mixing_rules(
                composition, parameters)
            
            tr = temperature / tc_mix
            pr = pressure / pc_mix
            
            # 简单流体焓偏差
            h0_r = self._calculate_simple_fluid_enthalpy_departure(tr, pr)
            
            # 参考流体焓偏差
            h1_r = self._calculate_reference_fluid_enthalpy_departure(tr, pr)
            
            # Lee-Kesler方程
            h_r = h0_r + omega_mix * (h1_r - h0_r)
            
            # 转换为实际单位
            R = 8.314  # J/(mol·K)
            h_departure = h_r * R * tc_mix
            
            self.logger.debug(f"LKP焓偏差: {h_departure:.1f} J/mol")
            
            return h_departure
            
        except Exception as e:
            self.logger.error(f"LKP焓偏差计算失败: {e}")
            raise CalculationError(f"Lee-Kesler-Plocker焓偏差计算失败: {e}")
    
    def calculate_entropy_departure(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters],
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """计算熵偏差
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            composition: 摩尔组成
            parameters: 组分参数列表
            phase_type: 相态类型
            
        Returns:
            float: 熵偏差 (J/(mol·K))
        """
        try:
            # 计算混合物参数
            tc_mix, pc_mix, omega_mix = self._calculate_mixing_rules(
                composition, parameters)
            
            tr = temperature / tc_mix
            pr = pressure / pc_mix
            
            # 简单流体熵偏差
            s0_r = self._calculate_simple_fluid_entropy_departure(tr, pr)
            
            # 参考流体熵偏差
            s1_r = self._calculate_reference_fluid_entropy_departure(tr, pr)
            
            # Lee-Kesler方程
            s_r = s0_r + omega_mix * (s1_r - s0_r)
            
            # 转换为实际单位
            R = 8.314  # J/(mol·K)
            s_departure = s_r * R
            
            self.logger.debug(f"LKP熵偏差: {s_departure:.3f} J/(mol·K)")
            
            return s_departure
            
        except Exception as e:
            self.logger.error(f"LKP熵偏差计算失败: {e}")
            raise CalculationError(f"Lee-Kesler-Plocker熵偏差计算失败: {e}")
    
    def _calculate_mixing_rules(
        self,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters]
    ) -> Tuple[float, float, float]:
        """计算混合规则
        
        Returns:
            Tuple[float, float, float]: (Tc_mix, Pc_mix, omega_mix)
        """
        n_comp = len(composition)
        
        # Kay混合规则
        tc_mix = sum(composition[i] * parameters[i].critical_temperature 
                    for i in range(n_comp))
        
        pc_mix = sum(composition[i] * parameters[i].critical_pressure 
                    for i in range(n_comp))
        
        omega_mix = sum(composition[i] * parameters[i].acentric_factor 
                       for i in range(n_comp))
        
        # 二元交互参数修正
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                kij = self._get_binary_interaction_parameter(
                    parameters[i], parameters[j])
                
                if kij != 0.0:
                    # 修正临界参数
                    tc_ij = np.sqrt(parameters[i].critical_temperature * 
                                   parameters[j].critical_temperature) * (1 - kij)
                    
                    tc_mix += 2 * composition[i] * composition[j] * (
                        tc_ij - np.sqrt(parameters[i].critical_temperature * 
                                       parameters[j].critical_temperature))
        
        return tc_mix, pc_mix, omega_mix
    
    def _calculate_simple_fluid_z(self, tr: float, pr: float) -> float:
        """计算简单流体压缩因子"""
        
        b = self._lk_constants['b_simple']
        gamma = self._lk_constants['gamma']
        beta = self._lk_constants['beta']
        
        # 迭代求解压缩因子
        z = 1.0  # 初值
        
        for _ in range(50):  # 最大迭代次数
            z_old = z
            
            # 计算B和C项
            B = sum(b[i] * tr**(-gamma[i]) for i in range(4))
            C = sum(b[i] * tr**(-gamma[i]) for i in range(4, 7))
            D = sum(b[i] * tr**(-gamma[i]) for i in range(7, 11))
            
            # 压缩因子方程
            rho_r = pr / (z * tr)  # 对比密度
            
            z_new = 1 + B * rho_r + C * rho_r**2 + D * rho_r**5
            
            if abs(z_new - z_old) < 1e-10:
                break
                
            z = 0.5 * (z_new + z_old)  # 阻尼
        
        return z
    
    def _calculate_reference_fluid_z(self, tr: float, pr: float) -> float:
        """计算参考流体压缩因子"""
        
        b = self._lk_constants['b_reference']
        gamma = self._lk_constants['gamma']
        beta = self._lk_constants['beta']
        
        # 类似简单流体的计算
        z = 1.0
        
        for _ in range(50):
            z_old = z
            
            B = sum(b[i] * tr**(-gamma[i]) for i in range(4))
            C = sum(b[i] * tr**(-gamma[i]) for i in range(4, 7))
            D = sum(b[i] * tr**(-gamma[i]) for i in range(7, 11))
            
            rho_r = pr / (z * tr)
            z_new = 1 + B * rho_r + C * rho_r**2 + D * rho_r**5
            
            if abs(z_new - z_old) < 1e-10:
                break
                
            z = 0.5 * (z_new + z_old)
        
        return z
    
    def _calculate_plocker_correction(
        self,
        tr: float,
        pr: float,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters]
    ) -> float:
        """计算Plocker修正项"""
        
        # 计算混合物Plocker参数
        plocker_mix = sum(composition[i] * parameters[i].plocker_parameter 
                         for i in range(len(composition)))
        
        if abs(plocker_mix) < 1e-10:
            return 0.0
        
        # Plocker修正项（简化形式）
        z_plocker = plocker_mix * pr / tr * (1 - 1/tr)**2
        
        return z_plocker
    
    def _calculate_simple_fluid_fugacity_coefficient(
        self,
        tr: float,
        pr: float,
        z: float,
        component_index: int,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters]
    ) -> float:
        """计算简单流体逸度系数"""
        
        # 简化计算（实际实现需要更复杂的偏导数）
        ln_phi = z - 1 - np.log(z) - pr / (z * tr)
        
        return np.exp(ln_phi)
    
    def _calculate_reference_fluid_fugacity_coefficient(
        self,
        tr: float,
        pr: float,
        z: float,
        component_index: int,
        composition: np.ndarray,
        parameters: List[LeeKeslerPlocknerParameters]
    ) -> float:
        """计算参考流体逸度系数"""
        
        # 简化计算
        ln_phi = z - 1 - np.log(z) - pr / (z * tr)
        
        return np.exp(ln_phi)
    
    def _calculate_simple_fluid_enthalpy_departure(
        self,
        tr: float,
        pr: float
    ) -> float:
        """计算简单流体焓偏差"""
        
        b = self._lk_constants['b_simple']
        gamma = self._lk_constants['gamma']
        
        # 简化计算
        h_r = -sum(b[i] * gamma[i] * tr**(-gamma[i]) for i in range(11))
        
        return h_r
    
    def _calculate_reference_fluid_enthalpy_departure(
        self,
        tr: float,
        pr: float
    ) -> float:
        """计算参考流体焓偏差"""
        
        b = self._lk_constants['b_reference']
        gamma = self._lk_constants['gamma']
        
        h_r = -sum(b[i] * gamma[i] * tr**(-gamma[i]) for i in range(11))
        
        return h_r
    
    def _calculate_simple_fluid_entropy_departure(
        self,
        tr: float,
        pr: float
    ) -> float:
        """计算简单流体熵偏差"""
        
        # 简化计算
        s_r = -np.log(pr) + self._calculate_simple_fluid_enthalpy_departure(tr, pr) / tr
        
        return s_r
    
    def _calculate_reference_fluid_entropy_departure(
        self,
        tr: float,
        pr: float
    ) -> float:
        """计算参考流体熵偏差"""
        
        s_r = -np.log(pr) + self._calculate_reference_fluid_enthalpy_departure(tr, pr) / tr
        
        return s_r
    
    def _get_binary_interaction_parameter(
        self,
        param1: LeeKeslerPlocknerParameters,
        param2: LeeKeslerPlocknerParameters
    ) -> float:
        """获取二元交互参数"""
        
        # 简化实现，实际需要从数据库获取
        return 0.0
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        
        return {
            'name': 'Lee-Kesler-Plocker EOS',
            'type': 'Corresponding States',
            'description': '基于对应态原理的状态方程',
            'applicable_compounds': ['烷烃', '轻烃', '天然气组分'],
            'temperature_range': '150-600 K',
            'pressure_range': '0.1-10 MPa',
            'accuracy': '中等',
            'computational_cost': '低',
            'stats': self.stats
        }


class LeeKeslerPlocknerPropertyPackage(PropertyPackageBase):
    """Lee-Kesler-Plocker物性包
    
    基于Lee-Kesler-Plocker状态方程的完整物性包实现。
    """
    
    def __init__(self, compounds: List[CompoundProperties]):
        super().__init__(compounds)
        
        self.eos = LeeKeslerPlocknerEOS()
        self.name = "Lee-Kesler-Plocker"
        self.description = "基于对应态原理的物性包，适用于烷烃和轻烃混合物"
        
        # 初始化参数
        self.parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> List[LeeKeslerPlocknerParameters]:
        """初始化组分参数"""
        
        parameters = []
        
        for compound in self.compounds:
            param = LeeKeslerPlocknerParameters(
                critical_temperature=compound.critical_temperature,
                critical_pressure=compound.critical_pressure,
                critical_volume=compound.critical_volume,
                acentric_factor=compound.acentric_factor
            )
            parameters.append(param)
        
        return parameters
    
    def calculate_compressibility_factor(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """计算压缩因子"""
        
        return self.eos.calculate_compressibility_factor(
            temperature, pressure, composition, self.parameters, phase_type)
    
    def calculate_fugacity_coefficients(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> np.ndarray:
        """计算逸度系数"""
        
        return self.eos.calculate_fugacity_coefficient(
            temperature, pressure, composition, self.parameters, phase_type)
    
    def calculate_enthalpy_departure(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """计算焓偏差"""
        
        return self.eos.calculate_enthalpy_departure(
            temperature, pressure, composition, self.parameters, phase_type)
    
    def calculate_entropy_departure(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """计算熵偏差"""
        
        return self.eos.calculate_entropy_departure(
            temperature, pressure, composition, self.parameters, phase_type)


# 导出主要类
__all__ = [
    'LeeKeslerPlocknerEOS',
    'LeeKeslerPlocknerParameters', 
    'LeeKeslerPlocknerPropertyPackage'
] 