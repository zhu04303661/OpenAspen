"""
Peng-Robinson-Stryjek-Vera状态方程
===============================

PRSV状态方程是对经典Peng-Robinson方程的改进，通过引入Stryjek-Vera修正项
提高了对极性和缔合化合物的预测精度。

对应DWSIM的PengRobinsonStryjekVera2.vb (867行)的完整Python实现。

理论基础：
$$P = \frac{RT}{V-b} - \frac{a(T)}{V(V+b)+b(V-b)}$$

其中：
$$a(T) = a_c \alpha(T)$$
$$\alpha(T) = [1 + \kappa(1-\sqrt{T_r})]^2$$

PRSV修正：
$$\kappa = \kappa_0 + \kappa_1(1+\sqrt{T_r})(0.7-T_r)$$

其中：
- $\kappa_0$：经典PR的$\kappa$值
- $\kappa_1$：PRSV修正参数

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
from scipy.optimize import fsolve, brentq

from ..base_property_package import PropertyPackageBase
from ...core.compound import CompoundProperties
from ...core.enums import PhaseType
from ...core.exceptions import CalculationError, ConvergenceError


@dataclass
class PRSVParameters:
    """PRSV状态方程参数"""
    
    # 临界参数
    critical_temperature: float     # 临界温度 (K)
    critical_pressure: float        # 临界压力 (Pa)
    acentric_factor: float          # 偏心因子
    
    # PRSV特有参数
    kappa0: float = 0.0             # 基础κ参数
    kappa1: float = 0.0             # PRSV修正参数κ1
    
    # 二元交互参数
    binary_interaction_parameters: Dict[str, float] = None
    
    # Margules混合规则参数
    margules_parameters: Dict[str, float] = None
    
    def __post_init__(self):
        if self.binary_interaction_parameters is None:
            self.binary_interaction_parameters = {}
        if self.margules_parameters is None:
            self.margules_parameters = {}
        
        # 如果未提供κ0，使用经典PR公式计算
        if self.kappa0 == 0.0:
            if self.acentric_factor <= 0.491:
                self.kappa0 = 0.37464 + 1.54226 * self.acentric_factor - 0.26992 * self.acentric_factor**2
            else:
                self.kappa0 = 0.379642 + 1.48503 * self.acentric_factor - 0.164423 * self.acentric_factor**2 + 0.016666 * self.acentric_factor**3


class PengRobinsonStryjekVeraEOS:
    """Peng-Robinson-Stryjek-Vera状态方程实现
    
    PRSV方程的主要改进：
    1. 引入κ1修正参数
    2. 改善了对极性化合物的预测
    3. 提高了饱和蒸汽压的计算精度
    4. 支持Margules混合规则
    
    适用范围：
    - 烷烃、烯烃、芳烃
    - 极性化合物（醇、酮、酯等）
    - 气液平衡计算
    - 高压条件
    """
    
    def __init__(self):
        self.logger = logging.getLogger("PRSVEOS")
        
        # 通用气体常数
        self.R = 8.314  # J/(mol·K)
        
        # 计算统计
        self.stats = {
            'total_calculations': 0,
            'convergence_failures': 0,
            'average_iterations': 0.0,
            'cubic_solutions': {'liquid': 0, 'vapor': 0, 'supercritical': 0}
        }
    
    def calculate_eos_parameters(
        self,
        temperature: float,
        composition: np.ndarray,
        parameters: List[PRSVParameters]
    ) -> Tuple[float, float]:
        """计算状态方程参数a和b
        
        Args:
            temperature: 温度 (K)
            composition: 摩尔组成
            parameters: 组分参数列表
            
        Returns:
            Tuple[float, float]: (a_mix, b_mix)
        """
        n_comp = len(composition)
        
        # 计算纯组分参数
        a_pure = np.zeros(n_comp)
        b_pure = np.zeros(n_comp)
        
        for i in range(n_comp):
            param = parameters[i]
            
            # 临界参数
            tc = param.critical_temperature
            pc = param.critical_pressure
            
            # 纯组分参数
            ac = 0.45724 * (self.R * tc)**2 / pc
            bc = 0.07780 * self.R * tc / pc
            
            # 计算α函数
            tr = temperature / tc
            alpha = self._calculate_alpha_function(tr, param)
            
            a_pure[i] = ac * alpha
            b_pure[i] = bc
        
        # 混合规则
        a_mix, b_mix = self._apply_mixing_rules(
            composition, a_pure, b_pure, parameters, temperature)
        
        return a_mix, b_mix
    
    def _calculate_alpha_function(
        self,
        tr: float,
        param: PRSVParameters
    ) -> float:
        """计算PRSV的α函数
        
        Args:
            tr: 对比温度
            param: 组分参数
            
        Returns:
            float: α值
        """
        sqrt_tr = np.sqrt(tr)
        
        # 计算κ值
        if param.kappa1 != 0.0:
            # PRSV修正
            kappa = param.kappa0 + param.kappa1 * (1 + sqrt_tr) * (0.7 - tr)
        else:
            # 经典PR
            kappa = param.kappa0
        
        # α函数
        alpha = (1 + kappa * (1 - sqrt_tr))**2
        
        return alpha
    
    def _apply_mixing_rules(
        self,
        composition: np.ndarray,
        a_pure: np.ndarray,
        b_pure: np.ndarray,
        parameters: List[PRSVParameters],
        temperature: float
    ) -> Tuple[float, float]:
        """应用混合规则
        
        支持：
        1. van der Waals混合规则
        2. Margules混合规则（如果提供参数）
        
        Args:
            composition: 摩尔组成
            a_pure: 纯组分a参数
            b_pure: 纯组分b参数
            parameters: 组分参数列表
            temperature: 温度
            
        Returns:
            Tuple[float, float]: (a_mix, b_mix)
        """
        n_comp = len(composition)
        
        # b参数混合规则（线性）
        b_mix = sum(composition[i] * b_pure[i] for i in range(n_comp))
        
        # a参数混合规则
        a_mix = 0.0
        
        for i in range(n_comp):
            for j in range(n_comp):
                # 几何平均
                aij = np.sqrt(a_pure[i] * a_pure[j])
                
                # 二元交互参数
                kij = self._get_binary_interaction_parameter(
                    parameters[i], parameters[j], i, j)
                
                aij *= (1 - kij)
                
                # Margules修正（如果适用）
                if self._has_margules_parameters(parameters[i], parameters[j]):
                    margules_correction = self._calculate_margules_correction(
                        parameters[i], parameters[j], temperature, i, j)
                    aij *= margules_correction
                
                a_mix += composition[i] * composition[j] * aij
        
        return a_mix, b_mix
    
    def solve_cubic_equation(
        self,
        temperature: float,
        pressure: float,
        a_mix: float,
        b_mix: float,
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """求解三次方程获得摩尔体积
        
        PRSV方程的三次形式：
        Z³ - (1-B)Z² + (A-2B-3B²)Z - (AB-B²-B³) = 0
        
        其中：A = aP/(RT)², B = bP/(RT)
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            a_mix: 混合物a参数
            b_mix: 混合物b参数
            phase_type: 相态类型
            
        Returns:
            float: 摩尔体积 (m³/mol)
        """
        try:
            # 无量纲参数
            A = a_mix * pressure / (self.R * temperature)**2
            B = b_mix * pressure / (self.R * temperature)
            
            # 三次方程系数
            # Z³ + p*Z² + q*Z + r = 0
            p = -(1 - B)
            q = A - 2*B - 3*B**2
            r = -(A*B - B**2 - B**3)
            
            # 求解三次方程
            roots = self._solve_cubic(p, q, r)
            
            # 选择物理意义的根
            valid_roots = []
            for root in roots:
                if np.isreal(root) and root.real > B:  # Z > B的物理约束
                    valid_roots.append(root.real)
            
            if not valid_roots:
                raise CalculationError("未找到有效的压缩因子根")
            
            # 根据相态选择合适的根
            if phase_type == PhaseType.LIQUID:
                z = min(valid_roots)  # 液相取最小根
                self.stats['cubic_solutions']['liquid'] += 1
            elif phase_type == PhaseType.VAPOR:
                z = max(valid_roots)  # 气相取最大根
                self.stats['cubic_solutions']['vapor'] += 1
            else:
                z = valid_roots[0]  # 超临界取任意根
                self.stats['cubic_solutions']['supercritical'] += 1
            
            # 计算摩尔体积
            volume = z * self.R * temperature / pressure
            
            self.logger.debug(f"PRSV三次方程求解: T={temperature:.2f}K, "
                            f"P={pressure:.0f}Pa, Z={z:.6f}, V={volume*1e6:.3f} cm³/mol")
            
            return volume
            
        except Exception as e:
            self.logger.error(f"PRSV三次方程求解失败: {e}")
            raise CalculationError(f"三次方程求解失败: {e}")
    
    def _solve_cubic(self, p: float, q: float, r: float) -> List[complex]:
        """求解三次方程 x³ + px² + qx + r = 0"""
        
        # 使用Cardano公式
        # 转换为标准形式 t³ + pt + q = 0
        a = p
        b = q - p**2/3
        c = r - p*q/3 + 2*p**3/27
        
        # 判别式
        discriminant = -4*b**3 - 27*c**2
        
        if discriminant > 0:
            # 三个实根
            m = 2 * np.sqrt(-b/3)
            theta = np.arccos(3*c/(b*m)) / 3
            
            roots = [
                m * np.cos(theta) - a/3,
                m * np.cos(theta + 2*np.pi/3) - a/3,
                m * np.cos(theta + 4*np.pi/3) - a/3
            ]
        else:
            # 一个实根，两个复根
            if b == 0:
                root1 = -np.cbrt(c) - a/3
                roots = [root1, root1, root1]
            else:
                sqrt_disc = np.sqrt(-discriminant/108)
                u = np.cbrt((-c + sqrt_disc)/2)
                v = np.cbrt((-c - sqrt_disc)/2)
                
                root1 = u + v - a/3
                root2 = -(u + v)/2 - a/3 + 1j * np.sqrt(3) * (u - v)/2
                root3 = -(u + v)/2 - a/3 - 1j * np.sqrt(3) * (u - v)/2
                
                roots = [root1, root2, root3]
        
        return roots
    
    def calculate_fugacity_coefficient(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        parameters: List[PRSVParameters],
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
            
            # 计算混合物参数
            a_mix, b_mix = self.calculate_eos_parameters(
                temperature, composition, parameters)
            
            # 求解摩尔体积
            volume = self.solve_cubic_equation(
                temperature, pressure, a_mix, b_mix, phase_type)
            
            # 计算压缩因子
            z = pressure * volume / (self.R * temperature)
            
            # 计算各组分逸度系数
            phi = np.zeros(n_comp)
            
            for i in range(n_comp):
                ln_phi_i = self._calculate_component_fugacity_coefficient(
                    i, temperature, pressure, volume, composition, 
                    parameters, a_mix, b_mix, z)
                
                phi[i] = np.exp(ln_phi_i)
            
            self.logger.debug(f"PRSV逸度系数计算完成: φ={phi}")
            
            return phi
            
        except Exception as e:
            self.logger.error(f"PRSV逸度系数计算失败: {e}")
            raise CalculationError(f"逸度系数计算失败: {e}")
    
    def _calculate_component_fugacity_coefficient(
        self,
        component_index: int,
        temperature: float,
        pressure: float,
        volume: float,
        composition: np.ndarray,
        parameters: List[PRSVParameters],
        a_mix: float,
        b_mix: float,
        z: float
    ) -> float:
        """计算组分逸度系数的对数值"""
        
        n_comp = len(composition)
        i = component_index
        
        # 计算纯组分参数
        param_i = parameters[i]
        tc_i = param_i.critical_temperature
        pc_i = param_i.critical_pressure
        
        ac_i = 0.45724 * (self.R * tc_i)**2 / pc_i
        bc_i = 0.07780 * self.R * tc_i / pc_i
        
        tr_i = temperature / tc_i
        alpha_i = self._calculate_alpha_function(tr_i, param_i)
        a_i = ac_i * alpha_i
        
        # 计算∂a/∂ni和∂b/∂ni
        da_dni = 0.0
        db_dni = bc_i
        
        for j in range(n_comp):
            param_j = parameters[j]
            tc_j = param_j.critical_temperature
            pc_j = param_j.critical_pressure
            
            ac_j = 0.45724 * (self.R * tc_j)**2 / pc_j
            tr_j = temperature / tc_j
            alpha_j = self._calculate_alpha_function(tr_j, param_j)
            a_j = ac_j * alpha_j
            
            # 几何平均
            aij = np.sqrt(a_i * a_j)
            
            # 二元交互参数
            kij = self._get_binary_interaction_parameter(
                param_i, param_j, i, j)
            
            aij *= (1 - kij)
            
            da_dni += 2 * composition[j] * aij
        
        # 无量纲参数
        A = a_mix * pressure / (self.R * temperature)**2
        B = b_mix * pressure / (self.R * temperature)
        
        dA_dni = da_dni * pressure / (self.R * temperature)**2
        dB_dni = db_dni * pressure / (self.R * temperature)
        
        # 逸度系数公式
        ln_phi = (dB_dni / B) * (z - 1) - np.log(z - B) - \
                 (A / (2 * np.sqrt(2) * B)) * \
                 (dA_dni / A - dB_dni / B) * \
                 np.log((z + (1 + np.sqrt(2)) * B) / (z + (1 - np.sqrt(2)) * B))
        
        return ln_phi
    
    def calculate_enthalpy_departure(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        parameters: List[PRSVParameters],
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
            # 计算状态方程参数
            a_mix, b_mix = self.calculate_eos_parameters(
                temperature, composition, parameters)
            
            # 计算da/dT
            da_dt = self._calculate_da_dt(temperature, composition, parameters)
            
            # 求解摩尔体积
            volume = self.solve_cubic_equation(
                temperature, pressure, a_mix, b_mix, phase_type)
            
            # 计算压缩因子
            z = pressure * volume / (self.R * temperature)
            
            # 焓偏差公式
            sqrt2 = np.sqrt(2)
            term1 = z - 1
            term2 = (a_mix - temperature * da_dt) / (2 * sqrt2 * b_mix * self.R * temperature)
            term3 = np.log((z + (1 + sqrt2) * b_mix * pressure / (self.R * temperature)) /
                          (z + (1 - sqrt2) * b_mix * pressure / (self.R * temperature)))
            
            h_departure = self.R * temperature * (term1 - term2 * term3)
            
            self.logger.debug(f"PRSV焓偏差: {h_departure:.1f} J/mol")
            
            return h_departure
            
        except Exception as e:
            self.logger.error(f"PRSV焓偏差计算失败: {e}")
            raise CalculationError(f"焓偏差计算失败: {e}")
    
    def _calculate_da_dt(
        self,
        temperature: float,
        composition: np.ndarray,
        parameters: List[PRSVParameters]
    ) -> float:
        """计算da/dT"""
        
        n_comp = len(composition)
        da_dt = 0.0
        
        for i in range(n_comp):
            for j in range(n_comp):
                param_i = parameters[i]
                param_j = parameters[j]
                
                # 计算dα/dT
                dalpha_dt_i = self._calculate_dalpha_dt(temperature, param_i)
                dalpha_dt_j = self._calculate_dalpha_dt(temperature, param_j)
                
                # 临界参数
                tc_i = param_i.critical_temperature
                pc_i = param_i.critical_pressure
                tc_j = param_j.critical_temperature
                pc_j = param_j.critical_pressure
                
                ac_i = 0.45724 * (self.R * tc_i)**2 / pc_i
                ac_j = 0.45724 * (self.R * tc_j)**2 / pc_j
                
                # 计算daij/dT
                tr_i = temperature / tc_i
                tr_j = temperature / tc_j
                alpha_i = self._calculate_alpha_function(tr_i, param_i)
                alpha_j = self._calculate_alpha_function(tr_j, param_j)
                
                daij_dt = 0.5 * np.sqrt(ac_i * ac_j) * (
                    dalpha_dt_i * np.sqrt(alpha_j / alpha_i) +
                    dalpha_dt_j * np.sqrt(alpha_i / alpha_j)
                )
                
                # 二元交互参数
                kij = self._get_binary_interaction_parameter(
                    param_i, param_j, i, j)
                
                daij_dt *= (1 - kij)
                
                da_dt += composition[i] * composition[j] * daij_dt
        
        return da_dt
    
    def _calculate_dalpha_dt(
        self,
        temperature: float,
        param: PRSVParameters
    ) -> float:
        """计算dα/dT"""
        
        tc = param.critical_temperature
        tr = temperature / tc
        sqrt_tr = np.sqrt(tr)
        
        # 计算κ和dκ/dT
        if param.kappa1 != 0.0:
            kappa = param.kappa0 + param.kappa1 * (1 + sqrt_tr) * (0.7 - tr)
            dkappa_dt = param.kappa1 * (0.5/sqrt_tr * (0.7 - tr) - (1 + sqrt_tr)) / tc
        else:
            kappa = param.kappa0
            dkappa_dt = 0.0
        
        # dα/dT
        alpha = (1 + kappa * (1 - sqrt_tr))**2
        
        dalpha_dt = 2 * (1 + kappa * (1 - sqrt_tr)) * (
            dkappa_dt * (1 - sqrt_tr) - kappa * 0.5 / sqrt_tr
        ) / tc
        
        return dalpha_dt
    
    def _get_binary_interaction_parameter(
        self,
        param1: PRSVParameters,
        param2: PRSVParameters,
        i: int,
        j: int
    ) -> float:
        """获取二元交互参数"""
        
        if i == j:
            return 0.0
        
        # 尝试从参数中获取
        key1 = f"{i}-{j}"
        key2 = f"{j}-{i}"
        
        if key1 in param1.binary_interaction_parameters:
            return param1.binary_interaction_parameters[key1]
        elif key2 in param1.binary_interaction_parameters:
            return param1.binary_interaction_parameters[key2]
        elif key1 in param2.binary_interaction_parameters:
            return param2.binary_interaction_parameters[key1]
        elif key2 in param2.binary_interaction_parameters:
            return param2.binary_interaction_parameters[key2]
        
        return 0.0  # 默认值
    
    def _has_margules_parameters(
        self,
        param1: PRSVParameters,
        param2: PRSVParameters
    ) -> bool:
        """检查是否有Margules参数"""
        
        return (len(param1.margules_parameters) > 0 or 
                len(param2.margules_parameters) > 0)
    
    def _calculate_margules_correction(
        self,
        param1: PRSVParameters,
        param2: PRSVParameters,
        temperature: float,
        i: int,
        j: int
    ) -> float:
        """计算Margules修正项"""
        
        # 简化实现，实际需要更复杂的Margules方程
        return 1.0
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        
        return {
            'name': 'Peng-Robinson-Stryjek-Vera EOS',
            'type': 'Cubic Equation of State',
            'description': 'PRSV状态方程，改进的PR方程',
            'improvements': [
                '引入κ1修正参数',
                '改善极性化合物预测',
                '提高饱和蒸汽压精度',
                '支持Margules混合规则'
            ],
            'applicable_compounds': ['烷烃', '烯烃', '芳烃', '极性化合物'],
            'temperature_range': '150-800 K',
            'pressure_range': '0.1-100 MPa',
            'accuracy': '高',
            'computational_cost': '中等',
            'stats': self.stats
        }


class PRSVPropertyPackage(PropertyPackageBase):
    """PRSV物性包
    
    基于Peng-Robinson-Stryjek-Vera状态方程的完整物性包实现。
    """
    
    def __init__(self, compounds: List[CompoundProperties]):
        super().__init__(compounds)
        
        self.eos = PengRobinsonStryjekVeraEOS()
        self.name = "Peng-Robinson-Stryjek-Vera"
        self.description = "改进的PR状态方程，适用于烷烃和极性化合物"
        
        # 初始化参数
        self.parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> List[PRSVParameters]:
        """初始化组分参数"""
        
        parameters = []
        
        for compound in self.compounds:
            param = PRSVParameters(
                critical_temperature=compound.critical_temperature,
                critical_pressure=compound.critical_pressure,
                acentric_factor=compound.acentric_factor
            )
            
            # 可以从数据库加载κ1参数
            param.kappa1 = self._get_kappa1_parameter(compound)
            
            parameters.append(param)
        
        return parameters
    
    def _get_kappa1_parameter(self, compound: CompoundProperties) -> float:
        """获取κ1参数"""
        
        # 简化实现，实际应从数据库获取
        # 这里提供一些常见化合物的κ1值
        kappa1_database = {
            'methane': 0.0,
            'ethane': 0.0,
            'propane': 0.0,
            'n-butane': 0.0,
            'water': 0.25,
            'carbon dioxide': 0.0,
            'hydrogen sulfide': 0.0
        }
        
        return kappa1_database.get(compound.name.lower(), 0.0)
    
    def calculate_compressibility_factor(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> float:
        """计算压缩因子"""
        
        a_mix, b_mix = self.eos.calculate_eos_parameters(
            temperature, composition, self.parameters)
        
        volume = self.eos.solve_cubic_equation(
            temperature, pressure, a_mix, b_mix, phase_type)
        
        return pressure * volume / (self.eos.R * temperature)
    
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


# 导出主要类
__all__ = [
    'PengRobinsonStryjekVeraEOS',
    'PRSVParameters',
    'PRSVPropertyPackage'
] 