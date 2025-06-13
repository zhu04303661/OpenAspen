"""
Soave-Redlich-Kwong状态方程物性包
===============================

基于DWSIM SoaveRedlichKwong.vb的Python实现
实现SRK状态方程的完整热力学计算功能
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from ..core.property_package import PropertyPackage, PropertyPackageParameters
from ..core.compound import Compound
from ..core.phase import Phase
from ..core.enums import PhaseType


@dataclass
class SRKParameters:
    """SRK状态方程参数"""
    # 纯组分参数
    a: List[float] = None  # 吸引力参数
    b: List[float] = None  # 排斥体积参数
    m: List[float] = None  # Soave参数
    
    # 混合规则参数
    kij: np.ndarray = None  # 二元交互参数矩阵
    
    # 体积平移参数
    volume_translation: List[float] = None
    
    def __post_init__(self):
        if self.a is None:
            self.a = []
        if self.b is None:
            self.b = []
        if self.m is None:
            self.m = []
        if self.volume_translation is None:
            self.volume_translation = []


class SoaveRedlichKwongPackage(PropertyPackage):
    """
    Soave-Redlich-Kwong状态方程物性包
    
    状态方程：P = RT/(V-b) - a(T)/(V(V+b))
    其中：a(T) = a * α(T)
         α(T) = [1 + m(1-√Tr)]²
    """
    
    def __init__(self, compounds: List[Compound]):
        # 先初始化SRK参数，避免基类调用时出错
        self.srk_params = SRKParameters()
        
        from ..core.enums import PackageType
        super().__init__(PackageType.SRK, compounds)
        self.logger = logging.getLogger("SRKPackage")
        
        # SRK常数
        self.omega_a = 0.42748  # SRK常数a
        self.omega_b = 0.08664  # SRK常数b
        
        # 初始化参数
        self._initialize_parameters()
        
        # 默认二元交互参数
        self._initialize_binary_parameters()
        
        # 体积平移系数
        self._initialize_volume_translation()
    
    def _initialize_parameters(self):
        """初始化SRK参数"""
        n_comp = len(self.compounds)
        
        self.srk_params.a = []
        self.srk_params.b = []
        self.srk_params.m = []
        
        for compound in self.compounds:
            # 计算a和b参数
            Tc = compound.properties.critical_temperature
            Pc = compound.properties.critical_pressure
            omega = compound.properties.acentric_factor
            
            # SRK参数
            a = self.omega_a * (8.314**2) * (Tc**2) / Pc
            b = self.omega_b * 8.314 * Tc / Pc
            
            # Soave参数m
            if omega <= 0.49:
                m = 0.480 + 1.574 * omega - 0.176 * omega**2
            else:
                m = 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3
            
            self.srk_params.a.append(a)
            self.srk_params.b.append(b)
            self.srk_params.m.append(m)
            
            self.logger.debug(f"{compound.name}: a={a:.2e}, b={b:.2e}, m={m:.4f}")
    
    def _initialize_binary_parameters(self):
        """初始化二元交互参数"""
        n_comp = len(self.compounds)
        self.srk_params.kij = np.zeros((n_comp, n_comp))
        
        # 设置一些常见的二元交互参数
        for i in range(n_comp):
            for j in range(n_comp):
                if i != j:
                    # 这里可以根据化合物类型设置特定的kij值
                    # 暂时使用默认值0
                    self.srk_params.kij[i, j] = self._get_default_kij(
                        self.compounds[i], self.compounds[j])
    
    def _get_default_kij(self, comp1: Compound, comp2: Compound) -> float:
        """获取默认的二元交互参数"""
        # 一些常见的二元交互参数
        kij_database = {
            ("methane", "ethane"): 0.0,
            ("methane", "propane"): 0.0,
            ("methane", "n-butane"): 0.0,
            ("methane", "carbon dioxide"): 0.0919,
            ("methane", "hydrogen sulfide"): 0.08,
            ("ethane", "propane"): 0.0,
            ("carbon dioxide", "hydrogen sulfide"): 0.0974,
            ("nitrogen", "methane"): 0.0311,
            ("nitrogen", "carbon dioxide"): -0.0170,
        }
        
        # 查找kij值
        key1 = (comp1.name.lower(), comp2.name.lower())
        key2 = (comp2.name.lower(), comp1.name.lower())
        
        if key1 in kij_database:
            return kij_database[key1]
        elif key2 in kij_database:
            return kij_database[key2]
        else:
            return 0.0  # 默认值
    
    def _initialize_volume_translation(self):
        """初始化体积平移系数"""
        self.srk_params.volume_translation = []
        
        for compound in self.compounds:
            # 默认体积平移系数
            c = self._get_default_volume_translation(compound)
            self.srk_params.volume_translation.append(c)
    
    def _get_default_volume_translation(self, compound: Compound) -> float:
        """获取默认的体积平移系数"""
        # 一些常见化合物的体积平移系数
        volume_translation_db = {
            "nitrogen": -0.0079,
            "carbon dioxide": 0.0833,
            "hydrogen sulfide": 0.0466,
            "methane": 0.0234,
            "ethane": 0.0605,
            "propane": 0.0825,
            "isobutane": 0.083,
            "n-butane": 0.0975,
            "isopentane": 0.1022,
            "n-pentane": 0.1209,
            "n-hexane": 0.1467,
            "n-heptane": 0.1554,
            "n-octane": 0.1794,
            "benzene": 0.0967,
            "toluene": 0.1067,
            "water": -0.0047,
        }
        
        return volume_translation_db.get(compound.name.lower(), 0.0)
    
    def calculate_alpha(self, T: float, compound_index: int) -> float:
        """计算Soave的α函数"""
        Tc = self.compounds[compound_index].properties.critical_temperature
        m = self.srk_params.m[compound_index]
        
        Tr = T / Tc
        sqrt_Tr = np.sqrt(Tr)
        
        alpha = (1 + m * (1 - sqrt_Tr))**2
        return alpha
    
    def calculate_mixing_rules(self, T: float, composition: List[float]) -> Tuple[float, float]:
        """
        计算混合规则
        
        Args:
            T: 温度 (K)
            composition: 摩尔分数
            
        Returns:
            Tuple[float, float]: (a_mix, b_mix)
        """
        n_comp = len(composition)
        
        # 计算混合的a参数
        a_mix = 0.0
        for i in range(n_comp):
            for j in range(n_comp):
                if composition[i] > 1e-15 and composition[j] > 1e-15:
                    # 计算α函数
                    alpha_i = self.calculate_alpha(T, i)
                    alpha_j = self.calculate_alpha(T, j)
                    
                    # 几何平均混合规则
                    a_ij = np.sqrt(self.srk_params.a[i] * self.srk_params.a[j] * alpha_i * alpha_j)
                    a_ij *= (1 - self.srk_params.kij[i, j])
                    
                    a_mix += composition[i] * composition[j] * a_ij
        
        # 计算混合的b参数
        b_mix = sum(composition[i] * self.srk_params.b[i] for i in range(n_comp))
        
        return a_mix, b_mix
    
    def solve_cubic_equation(self, A: float, B: float) -> List[float]:
        """
        求解SRK三次方程
        
        Z³ - Z² + (A - B - B²)Z - AB = 0
        
        Args:
            A: 无量纲参数 aP/(RT)²
            B: 无量纲参数 bP/(RT)
            
        Returns:
            List[float]: 实数根列表
        """
        # 三次方程系数
        coeffs = [1, -1, A - B - B**2, -A * B]
        
        # 求解三次方程
        roots = np.roots(coeffs)
        
        # 提取实数根
        real_roots = []
        for root in roots:
            if np.isreal(root) and np.real(root) > B:  # Z必须大于B
                real_roots.append(np.real(root))
        
        return sorted(real_roots)
    
    def calculate_compressibility_factor(self, T: float, P: float, 
                                       composition: List[float], phase: str) -> float:
        """
        计算压缩因子
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态 ("liquid" 或 "vapor")
            
        Returns:
            float: 压缩因子
        """
        # 计算混合参数
        a_mix, b_mix = self.calculate_mixing_rules(T, composition)
        
        # 计算无量纲参数
        R = 8.314  # J/mol/K
        A = a_mix * P / (R * T)**2
        B = b_mix * P / (R * T)
        
        # 求解三次方程
        Z_roots = self.solve_cubic_equation(A, B)
        
        if not Z_roots:
            # 如果没有实数根，返回理想气体值
            return 1.0
        
        # 选择合适的根
        if phase.lower() == "vapor":
            # 气相选择最大根
            return max(Z_roots)
        else:
            # 液相选择最小根
            return min(Z_roots)
    
    def calculate_fugacity_coefficients(self, T: float, P: float, 
                                      composition: List[float], phase: str) -> List[float]:
        """
        计算逸度系数
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态
            
        Returns:
            List[float]: 逸度系数列表
        """
        n_comp = len(composition)
        
        # 计算混合参数
        a_mix, b_mix = self.calculate_mixing_rules(T, composition)
        
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, composition, phase)
        
        # 计算无量纲参数
        R = 8.314
        A = a_mix * P / (R * T)**2
        B = b_mix * P / (R * T)
        
        # 计算逸度系数
        phi = []
        for i in range(n_comp):
            # 计算∂a_mix/∂n_i
            da_dni = 0.0
            for j in range(n_comp):
                if composition[j] > 1e-15:
                    alpha_i = self.calculate_alpha(T, i)
                    alpha_j = self.calculate_alpha(T, j)
                    
                    a_ij = np.sqrt(self.srk_params.a[i] * self.srk_params.a[j] * alpha_i * alpha_j)
                    a_ij *= (1 - self.srk_params.kij[i, j])
                    
                    da_dni += 2 * composition[j] * a_ij
            
            # 计算逸度系数
            bi = self.srk_params.b[i]
            
            ln_phi = (bi / b_mix) * (Z - 1) - np.log(Z - B) - \
                     (A / B) * (da_dni / a_mix - bi / b_mix) * np.log(1 + B / Z)
            
            phi.append(np.exp(ln_phi))
        
        return phi
    
    def calculate_molar_volume(self, T: float, P: float, 
                             composition: List[float], phase: str) -> float:
        """
        计算摩尔体积
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态
            
        Returns:
            float: 摩尔体积 (m³/mol)
        """
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, composition, phase)
        
        # 计算摩尔体积
        R = 8.314
        V = Z * R * T / P
        
        # 应用体积平移
        if self.srk_params.volume_translation:
            c_mix = sum(composition[i] * self.srk_params.volume_translation[i] 
                       for i in range(len(composition)))
            V += c_mix
        
        return V
    
    def calculate_enthalpy_departure(self, T: float, P: float, 
                                   composition: List[float], phase: str) -> float:
        """
        计算焓偏差 (H - H_ideal)
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态
            
        Returns:
            float: 焓偏差 (J/mol)
        """
        # 计算混合参数
        a_mix, b_mix = self.calculate_mixing_rules(T, composition)
        
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, composition, phase)
        
        # 计算da_mix/dT
        da_dT = 0.0
        n_comp = len(composition)
        for i in range(n_comp):
            for j in range(n_comp):
                if composition[i] > 1e-15 and composition[j] > 1e-15:
                    # 计算dα/dT
                    Tc_i = self.compounds[i].critical_temperature
                    Tc_j = self.compounds[j].critical_temperature
                    m_i = self.srk_params.m[i]
                    m_j = self.srk_params.m[j]
                    
                    Tr_i = T / Tc_i
                    Tr_j = T / Tc_j
                    
                    alpha_i = self.calculate_alpha(T, i)
                    alpha_j = self.calculate_alpha(T, j)
                    
                    dalpha_dT_i = -m_i * (1 + m_i * (1 - np.sqrt(Tr_i))) / (np.sqrt(Tr_i) * Tc_i)
                    dalpha_dT_j = -m_j * (1 + m_j * (1 - np.sqrt(Tr_j))) / (np.sqrt(Tr_j) * Tc_j)
                    
                    # 计算da_ij/dT
                    a_ij = np.sqrt(self.srk_params.a[i] * self.srk_params.a[j])
                    da_ij_dT = a_ij * (1 - self.srk_params.kij[i, j]) * \
                              (dalpha_dT_i / (2 * alpha_i) + dalpha_dT_j / (2 * alpha_j))
                    
                    da_dT += composition[i] * composition[j] * da_ij_dT
        
        # 计算焓偏差
        R = 8.314
        B = b_mix * P / (R * T)
        
        H_dep = R * T * (Z - 1) + (T * da_dT - a_mix) / b_mix * np.log(1 + B / Z)
        
        return H_dep
    
    def calculate_entropy_departure(self, T: float, P: float, 
                                  composition: List[float], phase: str) -> float:
        """
        计算熵偏差 (S - S_ideal)
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态
            
        Returns:
            float: 熵偏差 (J/mol/K)
        """
        # 计算混合参数
        a_mix, b_mix = self.calculate_mixing_rules(T, composition)
        
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, composition, phase)
        
        # 计算da_mix/dT（与焓偏差计算相同）
        da_dT = 0.0
        n_comp = len(composition)
        for i in range(n_comp):
            for j in range(n_comp):
                if composition[i] > 1e-15 and composition[j] > 1e-15:
                    Tc_i = self.compounds[i].properties.critical_temperature
                    Tc_j = self.compounds[j].properties.critical_temperature
                    m_i = self.srk_params.m[i]
                    m_j = self.srk_params.m[j]
                    
                    Tr_i = T / Tc_i
                    Tr_j = T / Tc_j
                    
                    alpha_i = self.calculate_alpha(T, i)
                    alpha_j = self.calculate_alpha(T, j)
                    
                    dalpha_dT_i = -m_i * (1 + m_i * (1 - np.sqrt(Tr_i))) / (np.sqrt(Tr_i) * Tc_i)
                    dalpha_dT_j = -m_j * (1 + m_j * (1 - np.sqrt(Tr_j))) / (np.sqrt(Tr_j) * Tc_j)
                    
                    a_ij = np.sqrt(self.srk_params.a[i] * self.srk_params.a[j])
                    da_ij_dT = a_ij * (1 - self.srk_params.kij[i, j]) * \
                              (dalpha_dT_i / (2 * alpha_i) + dalpha_dT_j / (2 * alpha_j))
                    
                    da_dT += composition[i] * composition[j] * da_ij_dT
        
        # 计算熵偏差
        R = 8.314
        B = b_mix * P / (R * T)
        
        S_dep = R * np.log(Z - B) + da_dT / b_mix * np.log(1 + B / Z)
        
        return S_dep
    
    def calculate_isothermal_compressibility(self, T: float, P: float, 
                                           composition: List[float], phase: str) -> float:
        """
        计算等温压缩系数
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态
            
        Returns:
            float: 等温压缩系数 (1/Pa)
        """
        # 计算混合参数
        a_mix, b_mix = self.calculate_mixing_rules(T, composition)
        
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, composition, phase)
        
        # 计算无量纲参数
        R = 8.314
        A = a_mix * P / (R * T)**2
        B = b_mix * P / (R * T)
        
        # 计算dZ/dP
        dZ_dP_numerator = A / (R * T) - B / P
        dZ_dP_denominator = 3 * Z**2 - 2 * Z + A - B - B**2
        
        if abs(dZ_dP_denominator) > 1e-15:
            dZ_dP = dZ_dP_numerator / dZ_dP_denominator
        else:
            # 理想气体近似
            dZ_dP = -1 / P
        
        # 计算等温压缩系数
        kappa_T = -(1 / Z) * (dZ_dP + Z / P)
        
        return kappa_T
    
    def calculate_joule_thomson_coefficient(self, T: float, P: float, 
                                          composition: List[float], phase: str) -> float:
        """
        计算Joule-Thomson系数
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态
            
        Returns:
            float: Joule-Thomson系数 (K/Pa)
        """
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, composition, phase)
        
        # 计算混合参数
        a_mix, b_mix = self.calculate_mixing_rules(T, composition)
        
        # 计算da_mix/dT
        da_dT = 0.0
        n_comp = len(composition)
        for i in range(n_comp):
            for j in range(n_comp):
                if composition[i] > 1e-15 and composition[j] > 1e-15:
                    Tc_i = self.compounds[i].properties.critical_temperature
                    Tc_j = self.compounds[j].properties.critical_temperature
                    m_i = self.srk_params.m[i]
                    m_j = self.srk_params.m[j]
                    
                    Tr_i = T / Tc_i
                    Tr_j = T / Tc_j
                    
                    alpha_i = self.calculate_alpha(T, i)
                    alpha_j = self.calculate_alpha(T, j)
                    
                    dalpha_dT_i = -m_i * (1 + m_i * (1 - np.sqrt(Tr_i))) / (np.sqrt(Tr_i) * Tc_i)
                    dalpha_dT_j = -m_j * (1 + m_j * (1 - np.sqrt(Tr_j))) / (np.sqrt(Tr_j) * Tc_j)
                    
                    a_ij = np.sqrt(self.srk_params.a[i] * self.srk_params.a[j])
                    da_ij_dT = a_ij * (1 - self.srk_params.kij[i, j]) * \
                              (dalpha_dT_i / (2 * alpha_i) + dalpha_dT_j / (2 * alpha_j))
                    
                    da_dT += composition[i] * composition[j] * da_ij_dT
        
        # 计算热容（简化）
        Cp = self.calculate_heat_capacity(T, P, composition, phase)
        
        # 计算Joule-Thomson系数
        R = 8.314
        B = b_mix * P / (R * T)
        
        # 简化的JT系数计算
        mu_JT = (1 / Cp) * ((T * da_dT - a_mix) / (R * T * b_mix) * 
                           (B / (Z * (Z + B))) - (Z - 1) / P)
        
        return mu_JT
    
    def calculate_heat_capacity(self, T: float, P: float, 
                              composition: List[float], phase: str) -> float:
        """
        计算定压热容
        
        Args:
            T: 温度 (K)
            P: 压力 (Pa)
            composition: 摩尔分数
            phase: 相态
            
        Returns:
            float: 定压热容 (J/mol/K)
        """
        # 计算理想气体热容
        Cp_ideal = 0.0
        for i, compound in enumerate(self.compounds):
            if composition[i] > 1e-15:
                Cp_i = compound.calculate_ideal_gas_heat_capacity(T)
                Cp_ideal += composition[i] * Cp_i
        
        # 计算偏差热容（简化）
        # 这里使用数值微分计算偏差热容
        dT = 0.1
        H_dep_plus = self.calculate_enthalpy_departure(T + dT, P, composition, phase)
        H_dep_minus = self.calculate_enthalpy_departure(T - dT, P, composition, phase)
        
        Cp_departure = (H_dep_plus - H_dep_minus) / (2 * dT)
        
        return Cp_ideal + Cp_departure
    
    def calculate_vapor_pressure(self, compound_index: int, T: float) -> float:
        """
        计算纯组分蒸汽压
        
        Args:
            compound_index: 化合物索引
            T: 温度 (K)
            
        Returns:
            float: 蒸汽压 (Pa)
        """
        compound = self.compounds[compound_index]
        
        # 使用Antoine方程或其他关联式
        if hasattr(compound, 'antoine_a') and compound.antoine_a is not None:
            # Antoine方程
            A = compound.antoine_a
            B = compound.antoine_b
            C = compound.antoine_c
            
            log_P = A - B / (T + C)
            P_sat = 10**log_P * 133.322  # 转换为Pa
        else:
            # 使用Lee-Kesler方程
            Tc = compound.properties.critical_temperature
            Pc = compound.properties.critical_pressure
            omega = compound.properties.acentric_factor
            
            Tr = T / Tc
            
            # Lee-Kesler方程
            f0 = 5.92714 - 6.09648/Tr - 1.28862*np.log(Tr) + 0.169347*Tr**6
            f1 = 15.2518 - 15.6875/Tr - 13.4721*np.log(Tr) + 0.43577*Tr**6
            
            ln_Pr = f0 + omega * f1
            P_sat = Pc * np.exp(ln_Pr)
        
        return P_sat
    
    def calculate_activity_coefficient(
        self,
        phase,
        temperature: float,
        pressure: float
    ) -> np.ndarray:
        """计算活度系数
        
        对于状态方程模型，活度系数通过逸度系数计算。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            np.ndarray: 活度系数数组
        """
        # 对于状态方程模型，活度系数通过逸度系数关系计算
        # 这里简化为返回1（理想溶液行为）
        # 实际应该通过逸度系数和参考态计算
        return np.ones(len(self.compounds))
    
    def calculate_fugacity_coefficient(
        self,
        phase,
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
        # 获取相组成
        composition = phase.mole_fractions
        
        # 确定相态
        phase_type = "vapor" if phase.phase_type.name == "VAPOR" else "liquid"
        
        # 计算逸度系数
        fugacity_coeffs = self.calculate_fugacity_coefficients(
            temperature, pressure, composition, phase_type
        )
        
        return np.array(fugacity_coeffs)

    def flash_pt(self, z: List[float], P: float, T: float) -> Dict:
        """
        PT闪蒸计算（简化版本，实际应使用专门的闪蒸算法）
        
        Args:
            z: 进料组成
            P: 压力 (Pa)
            T: 温度 (K)
            
        Returns:
            Dict: 闪蒸结果
        """
        # 这里应该调用专门的闪蒸算法
        # 暂时返回简化结果
        n_comp = len(z)
        
        # 估算K值
        K = []
        for i in range(n_comp):
            P_sat = self.calculate_vapor_pressure(i, T)
            K.append(P_sat / P)
        
        # 简化的Rachford-Rice求解
        from scipy.optimize import brentq
        
        def rr_equation(beta):
            return sum(z[i] * (K[i] - 1) / (1 + beta * (K[i] - 1)) 
                      for i in range(n_comp))
        
        try:
            beta = brentq(rr_equation, 0.0, 1.0)
        except:
            beta = 0.5
        
        # 计算相组成
        x = [z[i] / (1 + beta * (K[i] - 1)) for i in range(n_comp)]
        y = [K[i] * x[i] for i in range(n_comp)]
        
        # 归一化
        sum_x = sum(x)
        sum_y = sum(y)
        if sum_x > 0:
            x = [xi / sum_x for xi in x]
        if sum_y > 0:
            y = [yi / sum_y for yi in y]
        
        return {
            'L1': 1.0 - beta,
            'V': beta,
            'Vx1': x,
            'Vy': y,
            'K_values': K
        } 