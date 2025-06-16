"""
Lee-Kesler-Plocker状态方程物性包
=====================================

基于Lee-Kesler-Plocker立方状态方程的热力学物性计算包。
该方程是对Peng-Robinson方程的改进，提供更好的密度计算精度。

作者: OpenAspen项目组
日期: 2024年12月
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

from ..core.property_package import PropertyPackage
from ..core.enums import Phase, FlashSpec
from ..equations.cubic_eos import CubicEOS
from ..solvers.numerical_methods import solve_cubic, newton_raphson


class LeeKeslerPlocker(PropertyPackage):
    """
    Lee-Kesler-Plocker状态方程实现
    
    该方程形式为:
    P = RT/(V-b) - a(T)/(V(V+b) + b(V-b))
    
    其中:
    - a(T) = ac × α(T,ω) 
    - α(T,ω) = [1 + m(1-Tr^0.5)]²
    - m = 0.37464 + 1.54226ω - 0.26992ω²
    
    适用范围:
    - 温度: 150-800K
    - 压力: 0.001-200 bar
    - 适用于轻烃和天然气系统
    """
    
    def __init__(self, compounds: List[str]):
        """
        初始化Lee-Kesler-Plocker物性包
        
        Parameters
        ----------
        compounds : List[str]
            化合物列表
        """
        super().__init__(compounds, "Lee-Kesler-Plocker")
        
        # LKP特有参数
        self._binary_interaction_parameters = {}
        self._alpha_function_type = "LKP"
        
        # 默认混合规则参数
        self._mixing_rule = "van_der_waals"
        
        # 温度压力限制
        self._T_min = 150.0  # K
        self._T_max = 800.0  # K
        self._P_min = 0.001  # bar
        self._P_max = 200.0  # bar
        
    def _calculate_alpha_function(self, T: float, Tc: float, w: float) -> float:
        """
        计算LKP方程的α函数
        
        Parameters
        ----------
        T : float
            温度 [K]
        Tc : float
            临界温度 [K]  
        w : float
            偏心因子 [-]
            
        Returns
        -------
        float
            α函数值
        """
        Tr = T / Tc
        
        # LKP改进的m函数
        if w <= 0.491:
            m = 0.37464 + 1.54226*w - 0.26992*w**2
        else:
            # 高偏心因子修正
            m = 0.3796 + 1.485*w - 0.1644*w**2 + 0.01667*w**3
            
        alpha = (1 + m*(1 - np.sqrt(Tr)))**2
        
        return alpha
        
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
            (a, b) 状态方程参数
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
            
            # 临界参数计算
            ac = 0.45724 * (8.314*Tc)**2 / Pc  # J·m³/mol²
            bc = 0.07780 * 8.314*Tc / Pc  # m³/mol
            
            # 温度依赖项
            alpha = self._calculate_alpha_function(T, Tc, w)
            
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
        # 基于化合物类型的简单估算
        hydrocarbon_list = ['methane', 'ethane', 'propane', 'butane', 
                           'pentane', 'hexane', 'heptane', 'octane']
        
        if comp1 in hydrocarbon_list and comp2 in hydrocarbon_list:
            return 0.0  # 烷烃间交互参数通常很小
        elif 'water' in [comp1.lower(), comp2.lower()]:
            return 0.5  # 水与烷烃交互参数较大
        else:
            return 0.05  # 默认小值
            
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
        # 转换压力单位 bar -> Pa
        P_pa = P * 1e5
        
        # 计算状态方程参数
        a, b = self._calculate_eos_parameters(T, P, x)
        
        # 计算A和B参数
        R = 8.314  # J/(mol·K)
        A = a * P_pa / (R*T)**2
        B = b * P_pa / (R*T)
        
        # 立方方程: Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
        coeffs = [
            1.0,
            -(1 - B),
            A - 3*B**2 - 2*B,
            -(A*B - B**2 - B**3)
        ]
        
        # 求解立方方程
        roots = solve_cubic(coeffs)
        
        # 根据相态选择合适的根
        if phase == Phase.VAPOR:
            return max(roots)  # 气相取最大根
        else:
            return min([r for r in roots if r > 0])  # 液相取最小正根
            
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
        
        # 计算每个组分的偏导数
        phi = np.zeros(n_comp)
        
        for i in range(n_comp):
            # 计算 ∂a/∂ni 和 ∂b/∂ni
            dadi, dbdi = self._calculate_partial_derivatives(i, T, x)
            
            # 逸度系数公式
            term1 = (Z - 1) - np.log(Z - B)
            term2 = -A/(2*np.sqrt(2)*B) * (2*dadi/a - dbdi/b)
            term3 = np.log((Z + (1+np.sqrt(2))*B)/(Z + (1-np.sqrt(2))*B))
            
            ln_phi = term1 + term2 * term3
            phi[i] = np.exp(ln_phi)
            
        return phi
        
    def _calculate_partial_derivatives(self, i: int, T: float, 
                                     x: np.ndarray) -> Tuple[float, float]:
        """
        计算混合规则的偏导数
        
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
            
            ac = 0.45724 * (8.314*Tc)**2 / Pc
            bc = 0.07780 * 8.314*Tc / Pc
            alpha = self._calculate_alpha_function(T, Tc, w)
            
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
        
        # 密度计算
        P_pa = P * 1e5
        R = 8.314
        density = P_pa * MW / (Z * R * T)  # kg/m³
        
        return density
    
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
        if not (self._T_min <= T <= self._T_max):
            warnings.warn(f"温度 {T}K 超出推荐范围 [{self._T_min}-{self._T_max}K]")
            return False
            
        if not (self._P_min <= P <= self._P_max):
            warnings.warn(f"压力 {P} bar 超出推荐范围 [{self._P_min}-{self._P_max} bar]")
            return False
            
        return True
        
    def get_model_info(self) -> Dict[str, Union[str, float, List[str]]]:
        """
        获取模型信息
        
        Returns
        -------
        Dict[str, Union[str, float, List[str]]]
            模型信息字典
        """
        return {
            'name': 'Lee-Kesler-Plocker',
            'type': 'Cubic Equation of State',
            'temperature_range': [self._T_min, self._T_max],
            'pressure_range': [self._P_min, self._P_max],
            'applicable_phases': ['vapor', 'liquid'],
            'mixing_rule': self._mixing_rule,
            'binary_parameters': len(self._binary_interaction_parameters),
            'compounds': self.compounds.copy(),
            'description': 'Lee-Kesler-Plocker立方状态方程，适用于轻烃和天然气系统'
        } 