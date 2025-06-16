"""
Peng-Robinson-Stryjek-Vera状态方程物性包
==========================================

基于Peng-Robinson-Stryjek-Vera (PRSV)状态方程的高精度热力学物性计算包。
该方程是对经典PR方程的改进，在高温和高偏心因子条件下具有更好的精度。

作者: OpenAspen项目组
日期: 2024年12月
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass

from ..core.property_package import PropertyPackage
from ..core.enums import Phase, FlashSpec
from ..equations.cubic_eos import CubicEOS
from ..solvers.numerical_methods import solve_cubic, newton_raphson


@dataclass
class PRSVParameters:
    """PRSV方程参数类"""
    kappa0: float  # PRSV参数κ₀
    kappa1: float  # PRSV参数κ₁  
    use_prsv2: bool = False  # 是否使用PRSV2


class PengRobinsonStryjekVera(PropertyPackage):
    """
    Peng-Robinson-Stryjek-Vera (PRSV)状态方程实现
    
    PRSV方程采用改进的α函数：
    α(T,ω,κ₀,κ₁) = [1 + κ(1-√Tr)]²
    
    其中：
    - κ = κ₀ + κ₁(1 + √Tr)(0.7 - Tr)  (PRSV1)
    - κ₀ = 0.378893 + 1.4897153ω - 0.17131848ω² + 0.0196554ω³
    
    适用范围:
    - 温度: 200-1000K  
    - 压力: 0.001-500 bar
    - 特别适用于高温和重组分计算
    """
    
    def __init__(self, compounds: List[str]):
        """
        初始化PRSV物性包
        
        Parameters
        ----------
        compounds : List[str]
            化合物列表
        """
        super().__init__(compounds, "Peng-Robinson-Stryjek-Vera")
        
        # PRSV特有参数
        self._prsv_parameters = {}
        self._binary_interaction_parameters = {}
        self._version = "PRSV1"  # PRSV1 或 PRSV2
        
        # 混合规则
        self._mixing_rule = "van_der_waals"
        
        # 温度压力限制  
        self._T_min = 200.0  # K
        self._T_max = 1000.0  # K
        self._P_min = 0.001  # bar
        self._P_max = 500.0  # bar
        
        # 初始化PRSV参数
        self._initialize_prsv_parameters()
        
    def _initialize_prsv_parameters(self):
        """初始化PRSV参数"""
        for compound in self.compounds:
            props = self.get_compound_properties(compound)
            w = props.get('acentric_factor', 0.0)
            
            # 默认κ₀计算 (Stryjek and Vera, 1986)
            kappa0 = 0.378893 + 1.4897153*w - 0.17131848*w**2 + 0.0196554*w**3
            
            # 默认κ₁ (通常需要实验拟合)
            kappa1 = self._estimate_kappa1(compound, w)
            
            self._prsv_parameters[compound] = PRSVParameters(
                kappa0=kappa0,
                kappa1=kappa1,
                use_prsv2=False
            )
            
    def _estimate_kappa1(self, compound: str, w: float) -> float:
        """
        估算κ₁参数
        
        Parameters
        ----------
        compound : str
            化合物名称
        w : float
            偏心因子
            
        Returns
        -------
        float
            估算的κ₁值
        """
        # 基于化合物类型和偏心因子估算
        if w < 0.1:
            return 0.0  # 简单分子
        elif w < 0.5:
            return -0.05 + 0.1*w  # 中等复杂度
        else:
            return 0.05  # 重组分
            
    def set_prsv_parameters(self, compound: str, kappa0: float, 
                           kappa1: float = 0.0, use_prsv2: bool = False):
        """
        设置PRSV参数
        
        Parameters
        ----------
        compound : str
            化合物名称
        kappa0 : float
            PRSV参数κ₀
        kappa1 : float
            PRSV参数κ₁
        use_prsv2 : bool
            是否使用PRSV2
        """
        self._prsv_parameters[compound] = PRSVParameters(
            kappa0=kappa0,
            kappa1=kappa1,  
            use_prsv2=use_prsv2
        )
        
    def _calculate_alpha_function(self, T: float, Tc: float, w: float, 
                                compound: str) -> float:
        """
        计算PRSV方程的α函数
        
        Parameters
        ----------
        T : float
            温度 [K]
        Tc : float
            临界温度 [K]
        w : float
            偏心因子 [-]
        compound : str
            化合物名称
            
        Returns
        -------
        float
            α函数值
        """
        Tr = T / Tc
        params = self._prsv_parameters.get(compound)
        
        if params is None:
            # 使用标准PR方程
            m = 0.37464 + 1.54226*w - 0.26992*w**2
            alpha = (1 + m*(1 - np.sqrt(Tr)))**2
        else:
            # PRSV方程
            if params.use_prsv2 and Tr > 1.0:
                # PRSV2用于超临界条件
                kappa = params.kappa0 + params.kappa1 * (1 + np.sqrt(Tr)) * (0.7 - Tr)
                alpha = np.exp(2*(1 + kappa)*(1 - Tr**0.5))
            else:
                # PRSV1  
                kappa = params.kappa0
                if abs(params.kappa1) > 1e-10:  # κ₁不为零
                    kappa += params.kappa1 * (1 + np.sqrt(Tr)) * (0.7 - Tr)
                    
                alpha = (1 + kappa*(1 - np.sqrt(Tr)))**2
                
        return max(alpha, 1e-10)  # 避免负值
        
    def _calculate_eos_parameters(self, T: float, P: float, 
                                x: np.ndarray) -> Tuple[float, float]:
        """
        计算状态方程参数a和b
        
        Parameters
        ----------
        T : float
            温度 [K]
        P : float
            压力 [bar]
        x : np.ndarray
            摩尔分数数组
            
        Returns
        -------
        Tuple[float, float]
            (a, b) 状态方程参数 [J·m³/mol², m³/mol]
        """
        n_comp = len(x)
        
        # 纯组分参数
        a_pure = np.zeros(n_comp)
        b_pure = np.zeros(n_comp)
        
        for i, comp in enumerate(self.compounds):
            props = self.get_compound_properties(comp)
            
            Tc = props['critical_temperature']  # K
            Pc = props['critical_pressure']  # bar
            w = props['acentric_factor']
            
            # PR方程临界参数
            R = 8.314  # J/(mol·K)
            ac = 0.45724 * (R*Tc)**2 / (Pc*1e5)  # J·m³/mol²
            bc = 0.07780 * R*Tc / (Pc*1e5)  # m³/mol
            
            # 温度依赖项
            alpha = self._calculate_alpha_function(T, Tc, w, comp)
            
            a_pure[i] = ac * alpha
            b_pure[i] = bc
            
        # 混合物参数 - van der Waals混合规则
        a_mix = 0.0
        b_mix = 0.0
        
        for i in range(n_comp):
            b_mix += x[i] * b_pure[i]
            
            for j in range(n_comp):
                # 二元交互参数
                kij = self._get_binary_interaction_parameter(
                    self.compounds[i], self.compounds[j]
                )
                
                aij = np.sqrt(a_pure[i] * a_pure[j]) * (1 - kij)
                a_mix += x[i] * x[j] * aij
                
        return a_mix, b_mix
        
    def _get_binary_interaction_parameter(self, comp1: str, comp2: str) -> float:
        """
        获取二元交互参数kij
        
        Parameters
        ----------
        comp1, comp2 : str
            化合物名称
            
        Returns
        -------
        float
            二元交互参数kij
        """
        if comp1 == comp2:
            return 0.0
            
        # 尝试获取已设置的参数
        key1 = f"{comp1}-{comp2}"
        key2 = f"{comp2}-{comp1}"
        
        if key1 in self._binary_interaction_parameters:
            return self._binary_interaction_parameters[key1]
        elif key2 in self._binary_interaction_parameters:
            return self._binary_interaction_parameters[key2]
        else:
            # 默认值 - 可基于化合物类型估算
            return self._estimate_kij(comp1, comp2)
            
    def _estimate_kij(self, comp1: str, comp2: str) -> float:
        """
        估算二元交互参数
        
        Parameters
        ----------
        comp1, comp2 : str
            化合物名称
            
        Returns
        -------
        float
            估算的kij值
        """
        # 基于化合物类型的估算
        aromatics = ['benzene', 'toluene', 'xylene']
        alkanes = ['methane', 'ethane', 'propane', 'butane', 'pentane', 
                  'hexane', 'heptane', 'octane', 'nonane', 'decane']
        
        comp1_lower = comp1.lower()
        comp2_lower = comp2.lower()
        
        # 水-烷烃
        if 'water' in [comp1_lower, comp2_lower]:
            other = comp2_lower if comp1_lower == 'water' else comp1_lower
            if any(alkane in other for alkane in alkanes):
                return 0.5  # 水-烷烃交互参数较大
        
        # 芳烃-烷烃
        if (any(arom in comp1_lower for arom in aromatics) and 
            any(alkane in comp2_lower for alkane in alkanes)) or \
           (any(arom in comp2_lower for arom in aromatics) and 
            any(alkane in comp1_lower for alkane in alkanes)):
            return 0.02  # 芳烃-烷烃小的正值
            
        # CO2-烷烃
        if ('co2' in [comp1_lower, comp2_lower] or 
            'carbon dioxide' in [comp1_lower, comp2_lower]):
            return 0.12  # CO2特殊处理
            
        return 0.0  # 默认值
        
    def set_binary_interaction_parameter(self, comp1: str, comp2: str, kij: float):
        """
        设置二元交互参数
        
        Parameters
        ----------
        comp1, comp2 : str
            化合物名称
        kij : float
            二元交互参数值
        """
        key = f"{comp1}-{comp2}"
        self._binary_interaction_parameters[key] = kij
        
    def calculate_compressibility_factor(self, T: float, P: float, 
                                       x: np.ndarray, phase: Phase) -> float:
        """
        计算压缩因子Z
        
        Parameters
        ----------
        T : float
            温度 [K]
        P : float
            压力 [bar]
        x : np.ndarray
            摩尔分数数组
        phase : Phase
            相态
            
        Returns
        -------
        float
            压缩因子Z
        """
        # 计算状态方程参数
        a, b = self._calculate_eos_parameters(T, P, x)
        
        # 转换压力单位 bar -> Pa
        P_pa = P * 1e5
        R = 8.314  # J/(mol·K)
        
        # 计算A和B参数
        A = a * P_pa / (R*T)**2
        B = b * P_pa / (R*T)
        
        # PR立方方程: Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
        coeffs = [
            1.0,
            -(1 - B),
            A - 3*B**2 - 2*B,
            -(A*B - B**2 - B**3)
        ]
        
        # 求解立方方程
        roots = solve_cubic(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        
        if not real_roots:
            raise ValueError("未找到实数根")
            
        # 根据相态选择合适的根
        if phase == Phase.VAPOR:
            return max(real_roots)  # 气相取最大根
        else:
            valid_roots = [r for r in real_roots if r > B]  # 液相根必须大于B
            if not valid_roots:
                raise ValueError("未找到有效的液相根")
            return min(valid_roots)  # 液相取最小正根
            
    def calculate_fugacity_coefficient(self, T: float, P: float, 
                                     x: np.ndarray, phase: Phase) -> np.ndarray:
        """
        计算逸度系数
        
        Parameters
        ----------  
        T : float
            温度 [K]
        P : float
            压力 [bar]
        x : np.ndarray
            摩尔分数数组
        phase : Phase
            相态
            
        Returns
        -------
        np.ndarray
            逸度系数数组
        """
        n_comp = len(x)
        
        # 计算状态方程参数
        a, b = self._calculate_eos_parameters(T, P, x)
        
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, x, phase)
        
        # 转换压力单位
        P_pa = P * 1e5
        R = 8.314
        
        # 计算A和B
        A = a * P_pa / (R*T)**2
        B = b * P_pa / (R*T)
        
        # 计算每个组分的逸度系数
        phi = np.zeros(n_comp)
        
        for i in range(n_comp):
            # 计算 ∂a/∂ni 和 ∂b/∂ni
            dadi, dbdi = self._calculate_partial_derivatives(i, T, x)
            
            # PR方程逸度系数公式
            term1 = (Z - 1) * dbdi/b - np.log(Z - B)
            
            term2 = -A/(2*np.sqrt(2)*B) * (2*dadi/a - dbdi/b)
            
            term3 = np.log((Z + (1+np.sqrt(2))*B)/(Z + (1-np.sqrt(2))*B))
            
            ln_phi = term1 + term2 * term3
            phi[i] = np.exp(ln_phi)
            
        return phi
        
    def _calculate_partial_derivatives(self, i: int, T: float, 
                                     x: np.ndarray) -> Tuple[float, float]:
        """
        计算混合规则的偏导数 ∂a/∂ni 和 ∂b/∂ni
        
        Parameters
        ----------
        i : int
            组分索引
        T : float
            温度 [K]
        x : np.ndarray
            摩尔分数数组
            
        Returns
        -------
        Tuple[float, float]
            (∂a/∂ni, ∂b/∂ni)
        """
        n_comp = len(x)
        
        # 计算纯组分参数
        a_pure = np.zeros(n_comp)
        b_pure = np.zeros(n_comp)
        
        for j, comp in enumerate(self.compounds):
            props = self.get_compound_properties(comp)
            Tc = props['critical_temperature']
            Pc = props['critical_pressure']
            w = props['acentric_factor']
            
            R = 8.314
            ac = 0.45724 * (R*Tc)**2 / (Pc*1e5)
            bc = 0.07780 * R*Tc / (Pc*1e5)
            alpha = self._calculate_alpha_function(T, Tc, w, comp)
            
            a_pure[j] = ac * alpha
            b_pure[j] = bc
            
        # ∂a/∂ni
        dadi = 0.0
        for j in range(n_comp):
            kij = self._get_binary_interaction_parameter(
                self.compounds[i], self.compounds[j]
            )
            aij = np.sqrt(a_pure[i] * a_pure[j]) * (1 - kij)
            dadi += 2 * x[j] * aij
            
        # ∂b/∂ni  
        dbdi = b_pure[i]
        
        return dadi, dbdi
        
    def calculate_density(self, T: float, P: float, x: np.ndarray, 
                         phase: Phase) -> float:
        """
        计算密度
        
        Parameters
        ----------
        T : float
            温度 [K]
        P : float
            压力 [bar]
        x : np.ndarray
            摩尔分数数组
        phase : Phase
            相态
            
        Returns
        -------
        float
            密度 [kg/m³]
        """
        Z = self.calculate_compressibility_factor(T, P, x, phase)
        
        # 计算平均分子量
        MW = sum(x[i] * self.get_compound_properties(comp)['molecular_weight'] 
                for i, comp in enumerate(self.compounds))
        
        # 密度计算: ρ = PM/(ZRT)
        P_pa = P * 1e5
        R = 8.314
        density = P_pa * MW / (Z * R * T)  # kg/m³
        
        return density
        
    def calculate_enthalpy_departure(self, T: float, P: float, 
                                   x: np.ndarray, phase: Phase) -> float:
        """
        计算焓偏离函数 (H - H_ideal)
        
        Parameters
        ----------
        T : float
            温度 [K]
        P : float
            压力 [bar]
        x : np.ndarray
            摩尔分数数组
        phase : Phase
            相态
            
        Returns
        -------
        float
            焓偏离 [J/mol]
        """
        # 计算状态方程参数
        a, b = self._calculate_eos_parameters(T, P, x)
        
        # 计算温度对a的偏导数
        dadt = self._calculate_temperature_derivative_a(T, x)
        
        # 计算压缩因子
        Z = self.calculate_compressibility_factor(T, P, x, phase)
        
        # 转换压力单位
        P_pa = P * 1e5
        R = 8.314
        
        # 计算A和B
        A = a * P_pa / (R*T)**2
        B = b * P_pa / (R*T)
        
        # 焓偏离公式
        term1 = R*T*(Z - 1)
        term2 = -(a - T*dadt)/(2*np.sqrt(2)*b)
        term3 = np.log((Z + (1+np.sqrt(2))*B)/(Z + (1-np.sqrt(2))*B))
        
        H_dep = term1 + term2 * term3
        
        return H_dep
        
    def _calculate_temperature_derivative_a(self, T: float, x: np.ndarray) -> float:
        """
        计算 ∂a/∂T
        
        Parameters
        ----------
        T : float
            温度 [K]
        x : np.ndarray
            摩尔分数数组
            
        Returns
        -------
        float
            ∂a/∂T
        """
        n_comp = len(x)
        dadt_mix = 0.0
        
        for i in range(n_comp):
            for j in range(n_comp):
                comp_i = self.compounds[i]
                comp_j = self.compounds[j]
                
                props_i = self.get_compound_properties(comp_i)
                props_j = self.get_compound_properties(comp_j)
                
                # 计算 ∂αᵢ/∂T 和 ∂αⱼ/∂T
                dadti = self._calculate_alpha_temperature_derivative(T, comp_i, props_i)
                dadtj = self._calculate_alpha_temperature_derivative(T, comp_j, props_j)
                
                # 临界参数
                R = 8.314
                Tc_i = props_i['critical_temperature']
                Pc_i = props_i['critical_pressure']
                ac_i = 0.45724 * (R*Tc_i)**2 / (Pc_i*1e5)
                
                Tc_j = props_j['critical_temperature']
                Pc_j = props_j['critical_pressure']
                ac_j = 0.45724 * (R*Tc_j)**2 / (Pc_j*1e5)
                
                # 交互参数
                kij = self._get_binary_interaction_parameter(comp_i, comp_j)
                
                # 混合规则偏导数
                dadt_ij = np.sqrt(ac_i * ac_j) * (1 - kij) * \
                         (dadti/np.sqrt(ac_i) + dadtj/np.sqrt(ac_j)) / 2
                         
                dadt_mix += x[i] * x[j] * dadt_ij
                
        return dadt_mix
        
    def _calculate_alpha_temperature_derivative(self, T: float, compound: str, 
                                              props: Dict) -> float:
        """
        计算 ∂α/∂T
        
        Parameters
        ----------
        T : float
            温度 [K]
        compound : str
            化合物名称
        props : Dict
            化合物性质字典
            
        Returns
        -------
        float
            ∂α/∂T
        """
        Tc = props['critical_temperature']
        w = props['acentric_factor']
        Tr = T / Tc
        
        params = self._prsv_parameters.get(compound)
        
        if params is None:
            # 标准PR
            m = 0.37464 + 1.54226*w - 0.26992*w**2
            dadT = -m * (1 + m*(1 - np.sqrt(Tr))) / (np.sqrt(Tr*T*Tc))
        else:
            # PRSV
            kappa = params.kappa0
            if abs(params.kappa1) > 1e-10:
                kappa += params.kappa1 * (1 + np.sqrt(Tr)) * (0.7 - Tr)
                
            dadT = -kappa * (1 + kappa*(1 - np.sqrt(Tr))) / (np.sqrt(Tr*T*Tc))
            
        return dadT
        
    def validate_conditions(self, T: float, P: float) -> bool:
        """
        验证温度压力条件是否在适用范围内
        
        Parameters
        ----------
        T : float
            温度 [K]
        P : float  
            压力 [bar]
            
        Returns
        -------
        bool
            是否在适用范围内
        """
        valid = True
        
        if not (self._T_min <= T <= self._T_max):
            warnings.warn(f"温度 {T}K 超出推荐范围 [{self._T_min}-{self._T_max}K]")
            valid = False
            
        if not (self._P_min <= P <= self._P_max):
            warnings.warn(f"压力 {P} bar 超出推荐范围 [{self._P_min}-{self._P_max} bar]")
            valid = False
            
        return valid
        
    def get_model_info(self) -> Dict[str, Union[str, float, List[str]]]:
        """
        获取模型信息
        
        Returns
        -------
        Dict[str, Union[str, float, List[str]]]
            模型信息字典
        """
        return {
            'name': 'Peng-Robinson-Stryjek-Vera',
            'version': self._version,
            'type': 'Cubic Equation of State',
            'temperature_range': [self._T_min, self._T_max],
            'pressure_range': [self._P_min, self._P_max],
            'applicable_phases': ['vapor', 'liquid'],
            'mixing_rule': self._mixing_rule,
            'binary_parameters': len(self._binary_interaction_parameters),
            'prsv_parameters': len(self._prsv_parameters),
            'compounds': self.compounds.copy(),
            'description': 'PRSV状态方程，PR方程的高精度改进版本，适用于高温和重组分'
        } 