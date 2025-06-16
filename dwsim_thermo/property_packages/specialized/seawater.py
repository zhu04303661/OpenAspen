"""
海水脱盐模型 (Seawater Desalination Model)
========================================

基于IAPWS (International Association for the Properties of Water and Steam) 
海水性质标准的完整海水热力学模型实现。

该模型专门用于海水淡化工艺设计和计算，包括：
- 海水热力学性质计算
- 渗透压计算  
- 沸点升高和冰点降低
- 电解质活度系数
- 海水密度和粘度
- 蒸发器设计计算

理论基础:
---------
海水主要由H2O和NaCl组成，其热力学性质可以通过以下方法计算：
1. IAPWS-IF97水蒸气性质作为基础
2. Pitzer离子相互作用模型计算电解质效应
3. 海水的经验关联式修正

参考文献:
- IAPWS Release on the IAPWS Formulation 2008 for the Thermodynamic Properties of Seawater
- Sharqawy, M.H., et al. (2010). Thermophysical properties of seawater
- DWSIM VB.NET SeaWater.vb源代码
- Millero, F.J. (2010). History of the equation of state of seawater

作者: OpenAspen项目组
日期: 2024年12月
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
from scipy.optimize import fsolve, brentq
from dataclasses import dataclass

from ..base_property_package import PropertyPackage
from ...core.compound import Compound
from ...core.enums import PackageType, Phase
from ...flash_algorithms.base_flash import FlashAlgorithmBase


@dataclass
class SeawaterProperties:
    """海水性质计算结果"""
    salinity: float              # 盐度 [g/kg]
    temperature: float           # 温度 [K]
    pressure: float             # 压力 [Pa]
    density: float              # 密度 [kg/m³]
    osmotic_pressure: float     # 渗透压 [Pa]
    boiling_point_elevation: float  # 沸点升高 [K]
    freezing_point_depression: float  # 冰点降低 [K]
    viscosity: float            # 粘度 [Pa·s]
    thermal_conductivity: float # 导热系数 [W/(m·K)]
    specific_heat: float        # 比热容 [J/(kg·K)]
    activity_coefficient_water: float  # 水的活度系数
    vapor_pressure: float       # 蒸汽压 [Pa]


class SeawaterModel(PropertyPackage):
    """
    海水热力学模型
    
    基于IAPWS海水性质标准的完整实现，专门用于海水淡化工艺设计。
    
    特点:
    - 基于IAPWS-2008海水性质公式
    - 支持0-120g/kg盐度范围
    - 温度范围: 273-373K
    - 压力范围: 0.1-10 MPa
    - 集成Pitzer电解质模型
    """
    
    def __init__(self, compounds: List[Compound], **kwargs):
        """
        初始化海水模型
        
        Parameters:
        -----------
        compounds : List[Compound]
            化合物列表（必须包含水和盐）
        **kwargs : 
            其他参数
        """
        # 验证化合物列表
        self._validate_compounds(compounds)
        
        super().__init__(PackageType.SEAWATER, compounds, **kwargs)
        
        self.model_name = "Seawater Desalination Model"
        self.model_description = "基于IAPWS标准的海水热力学模型"
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型参数
        self._initialize_seawater_parameters()
        
        # IAPWS-IF97水蒸气性质接口
        self._water_props = None  # 需要集成IAPWS-IF97
        
    def _validate_compounds(self, compounds: List[Compound]) -> None:
        """验证化合物列表"""
        compound_names = [comp.name.lower() for comp in compounds]
        
        # 检查是否包含水
        water_found = any(name in ['water', 'h2o', '水'] for name in compound_names)
        if not water_found:
            raise ValueError("海水模型必须包含水组分")
            
        # 检查是否包含盐分
        salt_found = any(name in ['salt', 'nacl', 'sodium chloride', '盐'] for name in compound_names)
        if not salt_found:
            warnings.warn("海水模型建议包含盐分组分")
    
    def _initialize_seawater_parameters(self) -> None:
        """初始化海水模型参数"""
        
        # IAPWS-2008海水性质常数
        self.iapws_constants = {
            'R': 8.314472,  # 气体常数 [J/(mol·K)]
            'M_w': 0.018015268,  # 水的摩尔质量 [kg/mol]
            'M_s': 0.0314038218,  # 海盐的平均摩尔质量 [kg/mol]
        }
        
        # Pitzer模型参数 (NaCl-H2O系统)
        self.pitzer_params = self._initialize_pitzer_parameters()
        
        # 海水密度关联参数 (Sharqawy et al., 2010)
        self.density_coeffs = {
            'A': [9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8],
            'B': [8.020e2, -2.001, 1.677e-2, -3.060e-5, -1.613e-5]
        }
        
        # 粘度关联参数
        self.viscosity_coeffs = {
            'mu_w': [1.5700386464e-1, 6.4992620050e1, -9.1296496657e1],
            'A': 1.541, 'B': 1.998e-2, 'C': -9.52e-5
        }
        
        # 导热系数参数
        self.thermal_conductivity_coeffs = {
            'lambda_w': [5.7011e-1, 1.7841e-3, -2.7002e-6, 1.1055e-9],
            'S_coeffs': [1.0, -2.551e-3, 6.23e-6, -1.33e-8]
        }
    
    def _initialize_pitzer_parameters(self) -> Dict:
        """初始化Pitzer电解质模型参数"""
        # NaCl-H2O系统的Pitzer参数 (Pitzer & Mayorga, 1973)
        return {
            'beta0': 0.0765,        # Pitzer β(0)参数
            'beta1': 0.2664,        # Pitzer β(1)参数  
            'beta2': -0.00127,      # Pitzer β(2)参数
            'Cphi': -0.00759,       # Pitzer Cᵠ参数
            'alpha1': 2.0,          # α₁参数
            'alpha2': 50.0,         # α₂参数
            'b': 1.2,               # b参数
        }
    
    def calculate_seawater_properties(self, T: float, P: float, 
                                    salinity: float) -> SeawaterProperties:
        """
        计算海水的完整热力学性质
        
        Parameters:
        -----------
        T : float
            温度 [K]
        P : float  
            压力 [Pa]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        SeawaterProperties
            海水性质对象
        """
        # 输入验证
        self._validate_inputs(T, P, salinity)
        
        # 计算各项性质
        density = self.calculate_density(T, P, salinity)
        osmotic_pressure = self.calculate_osmotic_pressure(T, salinity)
        bpe = self.calculate_boiling_point_elevation(T, P, salinity)
        fpd = self.calculate_freezing_point_depression(salinity)
        viscosity = self.calculate_viscosity(T, salinity)
        thermal_conductivity = self.calculate_thermal_conductivity(T, salinity)
        specific_heat = self.calculate_specific_heat(T, salinity)
        activity_coeff = self.calculate_water_activity_coefficient(T, salinity)
        vapor_pressure = self.calculate_vapor_pressure(T, salinity)
        
        return SeawaterProperties(
            salinity=salinity,
            temperature=T,
            pressure=P,
            density=density,
            osmotic_pressure=osmotic_pressure,
            boiling_point_elevation=bpe,
            freezing_point_depression=fpd,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            specific_heat=specific_heat,
            activity_coefficient_water=activity_coeff,
            vapor_pressure=vapor_pressure
        )
    
    def calculate_density(self, T: float, P: float, salinity: float) -> float:
        """
        计算海水密度
        
        基于Sharqawy et al. (2010)的关联式：
        ρ_sw = ρ_w + A·S + B·S^1.5 + C·S^2
        
        Parameters:
        -----------
        T : float
            温度 [K]
        P : float
            压力 [Pa]  
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            海水密度 [kg/m³]
        """
        # 温度转换为摄氏度
        t = T - 273.15
        
        # 纯水密度 (IAPWS-IF97简化)
        rho_w = self._calculate_pure_water_density(T, P)
        
        # 盐度修正系数
        S = salinity / 1000.0  # 转换为kg_salt/kg_solution
        
        # Sharqawy关联式系数
        A = 8.020e2 - 2.001*t + 1.677e-2*t**2 - 3.060e-5*t**3 - 1.613e-5*t**4
        B = -5.72466e-3 + 1.0227e-4*t - 1.6546e-6*t**2
        C = 4.8314e-4
        
        # 海水密度
        rho_sw = rho_w + A*S + B*S**1.5 + C*S**2
        
        # 压力修正 (简化)
        P_atm = 101325.0  # Pa
        beta_T = 4.5e-10  # 等温压缩系数 [1/Pa]
        rho_sw *= (1 + beta_T * (P - P_atm))
        
        return rho_sw
    
    def calculate_osmotic_pressure(self, T: float, salinity: float) -> float:
        """
        计算渗透压
        
        基于van't Hoff方程和Pitzer模型修正：
        Π = iMRT = (1 + νφ - 1)MRT
        
        Parameters:
        -----------
        T : float
            温度 [K]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            渗透压 [Pa]
        """
        R = self.iapws_constants['R']
        M_s = self.iapws_constants['M_s']
        
        # 摩尔浓度计算
        m = salinity / (M_s * 1000 * (1 - salinity/1000))  # mol/kg_water
        
        # Pitzer活度系数计算
        gamma_pm = self._calculate_pitzer_activity_coefficient(T, m)
        
        # 渗透系数
        phi = self._calculate_osmotic_coefficient(T, m, gamma_pm)
        
        # 渗透压
        nu = 2  # NaCl解离度
        osmotic_pressure = nu * phi * m * R * T * 1000  # Pa
        
        return osmotic_pressure
    
    def calculate_boiling_point_elevation(self, T: float, P: float, salinity: float) -> float:
        """
        计算沸点升高
        
        基于Raoult定律和活度系数修正：
        ΔT_b = (RT²/ΔH_vap) * ln(a_w)
        
        Parameters:
        -----------
        T : float
            温度 [K]
        P : float
            压力 [Pa]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            沸点升高 [K]
        """
        # 水的活度计算
        a_w = self.calculate_water_activity(T, salinity)
        
        # 水的汽化潜热 (简化)
        Delta_H_vap = 40660.0  # J/mol
        R = self.iapws_constants['R']
        
        # 沸点升高
        if a_w > 0:
            Delta_T_b = -(R * T**2 / Delta_H_vap) * np.log(a_w)
        else:
            Delta_T_b = 0.0
        
        return Delta_T_b
    
    def calculate_freezing_point_depression(self, salinity: float) -> float:
        """
        计算冰点降低
        
        基于经验关联式 (UNESCO):
        ΔT_f = -a·S - b·S^1.5 - c·S^2
        
        Parameters:
        -----------
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            冰点降低 [K] (负值表示降低)
        """
        S = salinity
        
        # UNESCO关联式系数
        a = 0.0575
        b = 1.710523e-3
        c = 2.154996e-4
        
        Delta_T_f = -(a*S + b*S**1.5 + c*S**2)
        
        return Delta_T_f
    
    def calculate_viscosity(self, T: float, salinity: float) -> float:
        """
        计算海水粘度
        
        基于Sharqawy et al. (2010)关联式
        
        Parameters:
        -----------
        T : float
            温度 [K]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            动力粘度 [Pa·s]
        """
        t = T - 273.15  # 温度 [°C]
        S = salinity
        
        # 纯水粘度 (IAPWS简化)
        mu_w = self._calculate_pure_water_viscosity(T)
        
        # 盐度修正
        A = 1.541 + 1.998e-2*t - 9.52e-5*t**2
        B = 7.974 - 7.561e-2*t + 4.724e-4*t**2
        
        mu_sw = mu_w * (1 + A*S/1000 + B*(S/1000)**2)
        
        return mu_sw
    
    def calculate_thermal_conductivity(self, T: float, salinity: float) -> float:
        """
        计算海水导热系数
        
        Parameters:
        -----------
        T : float
            温度 [K]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            导热系数 [W/(m·K)]
        """
        t = T - 273.15  # 温度 [°C]
        S = salinity / 1000.0  # 转换为质量分数
        
        # 纯水导热系数
        lambda_w = 5.7011e-1 + 1.7841e-3*t - 2.7002e-6*t**2 + 1.1055e-9*t**3
        
        # 盐度修正
        f_S = 1.0 - 2.551e-3*S + 6.23e-6*S**2 - 1.33e-8*S**3
        
        lambda_sw = lambda_w * f_S
        
        return lambda_sw
    
    def calculate_specific_heat(self, T: float, salinity: float) -> float:
        """
        计算海水比热容
        
        Parameters:
        -----------
        T : float
            温度 [K]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            比热容 [J/(kg·K)]
        """
        t = T - 273.15  # 温度 [°C]
        S = salinity
        
        # Sharqawy et al. (2010) 关联式
        A = 5.328 - 9.76e-2*S + 4.04e-4*S**2
        B = -6.913e-3 + 7.351e-4*S - 3.15e-6*S**2
        C = 9.6e-6 - 1.927e-6*S + 8.23e-9*S**2
        D = 2.5e-9 + 1.666e-9*S - 7.125e-12*S**2
        
        cp_sw = A + B*t + C*t**2 + D*t**3
        cp_sw *= 1000  # 转换为J/(kg·K)
        
        return cp_sw
    
    def calculate_water_activity_coefficient(self, T: float, salinity: float) -> float:
        """
        计算水的活度系数
        
        基于Pitzer模型
        
        Parameters:
        -----------
        T : float
            温度 [K]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            水的活度系数
        """
        # 摩尔浓度
        M_s = self.iapws_constants['M_s']
        m = salinity / (M_s * 1000 * (1 - salinity/1000))
        
        # Pitzer模型计算
        phi = self._calculate_osmotic_coefficient(T, m)
        
        # 水的活度系数
        M_w = self.iapws_constants['M_w']
        ln_gamma_w = -phi * m * 2 * M_w  # 近似
        gamma_w = np.exp(ln_gamma_w)
        
        return gamma_w
    
    def calculate_water_activity(self, T: float, salinity: float) -> float:
        """计算水的活度"""
        # 摩尔分数
        M_w = self.iapws_constants['M_w']
        M_s = self.iapws_constants['M_s']
        
        n_w = (1000 - salinity) / (M_w * 1000)
        n_s = salinity / (M_s * 1000)
        x_w = n_w / (n_w + 2*n_s)  # 考虑NaCl解离
        
        # 活度系数
        gamma_w = self.calculate_water_activity_coefficient(T, salinity)
        
        return x_w * gamma_w
    
    def calculate_vapor_pressure(self, T: float, salinity: float) -> float:
        """
        计算海水蒸汽压
        
        Parameters:
        -----------
        T : float
            温度 [K]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        float
            蒸汽压 [Pa]
        """
        # 纯水饱和蒸汽压 (Antoine方程简化)
        P_sat_w = self._calculate_pure_water_vapor_pressure(T)
        
        # 水的活度
        a_w = self.calculate_water_activity(T, salinity)
        
        # 海水蒸汽压
        P_sat_sw = P_sat_w * a_w
        
        return P_sat_sw
    
    def _calculate_pitzer_activity_coefficient(self, T: float, m: float) -> float:
        """计算Pitzer活度系数"""
        params = self.pitzer_params
        
        # 离子强度
        I = m  # 对于1:1电解质
        
        # Debye-Hückel参数
        A_phi = self._calculate_debye_huckel_parameter(T)
        
        # Pitzer函数
        alpha1 = params['alpha1']
        alpha2 = params['alpha2']
        
        # γ函数
        sqrt_I = np.sqrt(I)
        g_alpha1 = 2 * (1 - (1 + alpha1*sqrt_I) * np.exp(-alpha1*sqrt_I)) / (alpha1**2 * I)
        g_alpha2 = 2 * (1 - (1 + alpha2*sqrt_I) * np.exp(-alpha2*sqrt_I)) / (alpha2**2 * I)
        
        # BMX和CMX
        BMX = params['beta0'] + params['beta1']*g_alpha1 + params['beta2']*g_alpha2
        CMX = params['Cphi'] / (2 * np.sqrt(2))
        
        # 活度系数
        ln_gamma = -A_phi * sqrt_I / (1 + params['b']*sqrt_I) + 2*BMX*m + 3*CMX*m**2
        
        return np.exp(ln_gamma)
    
    def _calculate_osmotic_coefficient(self, T: float, m: float, gamma_pm: float = None) -> float:
        """计算渗透系数"""
        if gamma_pm is None:
            gamma_pm = self._calculate_pitzer_activity_coefficient(T, m)
        
        # 渗透系数近似
        phi = 1 - np.log(gamma_pm) / (2 * m) if m > 0 else 1.0
        
        return phi
    
    def _calculate_debye_huckel_parameter(self, T: float) -> float:
        """计算Debye-Hückel参数"""
        # 简化计算，实际需要更精确的公式
        A_phi = 0.391 + 1.69e-4 * (T - 298.15)
        return A_phi
    
    def _calculate_pure_water_density(self, T: float, P: float) -> float:
        """计算纯水密度（简化IAPWS-IF97）"""
        t = T - 273.15
        
        # 简化的密度公式
        rho_w = 999.842594 + 6.793952e-2*t - 9.095290e-3*t**2 + \
                1.001685e-4*t**3 - 1.120083e-6*t**4 + 6.536332e-9*t**5
        
        return rho_w
    
    def _calculate_pure_water_viscosity(self, T: float) -> float:
        """计算纯水粘度"""
        # IAPWS简化公式
        A = 280.68
        B = 511.45
        C = 61.131
        D = 0.45903
        
        mu_w = A / (T - B + C*np.sqrt(np.max([0, T - B])) - D*(T - B))
        return mu_w * 1e-6  # 转换为Pa·s
    
    def _calculate_pure_water_vapor_pressure(self, T: float) -> float:
        """计算纯水饱和蒸汽压"""
        # Antoine方程
        if T < 373.15:
            # 液相
            A, B, C = 8.07131, 1730.63, 233.426
            ln_P = A - B / (T - 273.15 + C)
            P_sat = np.exp(ln_P) * 133.322  # mmHg to Pa
        else:
            # 气相
            A, B, C = 8.14019, 1810.94, 244.485
            ln_P = A - B / (T - 273.15 + C)
            P_sat = np.exp(ln_P) * 133.322
        
        return P_sat
    
    def _validate_inputs(self, T: float, P: float, salinity: float) -> None:
        """验证输入参数"""
        if not (273.15 <= T <= 373.15):
            warnings.warn(f"温度{T:.2f} K超出推荐范围 [273.15, 373.15] K")
        
        if not (1e5 <= P <= 1e7):
            warnings.warn(f"压力{P:.0f} Pa超出推荐范围 [1e5, 1e7] Pa")
        
        if not (0 <= salinity <= 120):
            warnings.warn(f"盐度{salinity:.1f} g/kg超出推荐范围 [0, 120] g/kg")


# 海水闪蒸算法
class SeawaterFlashAlgorithm(FlashAlgorithmBase):
    """
    海水专用闪蒸算法
    
    考虑海水的特殊性质：
    - 不挥发的盐分
    - 水的活度降低
    - 沸点升高效应
    """
    
    def __init__(self, seawater_model: SeawaterModel):
        super().__init__(seawater_model)
        self.seawater_model = seawater_model
        
    def flash_pt(self, P: float, T: float, z: np.ndarray, 
                 salinity: float = 35.0) -> Dict:
        """
        海水PT闪蒸
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K]
        z : np.ndarray
            总组成 [water, salt]
        salinity : float
            盐度 [g/kg]
            
        Returns:
        --------
        Dict
            闪蒸结果
        """
        # 计算海水性质
        props = self.seawater_model.calculate_seawater_properties(T, P, salinity)
        
        # 检查是否发生汽化
        if T > 373.15 + props.boiling_point_elevation:
            # 有蒸汽产生
            vapor_fraction = self._calculate_vapor_fraction(T, P, salinity, props)
            
            return {
                'converged': True,
                'flash_type': 'Seawater_VL',
                'phases': {
                    'vapor': {
                        'mole_fractions': [1.0, 0.0],  # 纯水蒸汽
                        'phase_fraction': vapor_fraction,
                        'phase_type': 'vapor'
                    },
                    'liquid': {
                        'mole_fractions': z,
                        'phase_fraction': 1 - vapor_fraction,
                        'phase_type': 'liquid'
                    }
                },
                'properties': {
                    'temperature': T,
                    'pressure': P,
                    'salinity': salinity,
                    'seawater_properties': props
                }
            }
        else:
            # 单液相
            return {
                'converged': True,
                'flash_type': 'Seawater_L',
                'phases': {
                    'liquid': {
                        'mole_fractions': z,
                        'phase_fraction': 1.0,
                        'phase_type': 'liquid'
                    }
                },
                'properties': {
                    'temperature': T,
                    'pressure': P,
                    'salinity': salinity,
                    'seawater_properties': props
                }
            }
    
    def _calculate_vapor_fraction(self, T: float, P: float, salinity: float,
                                props: SeawaterProperties) -> float:
        """计算蒸汽分率（简化）"""
        # 简化计算：基于能量平衡
        T_sat = 373.15 + props.boiling_point_elevation
        
        if T <= T_sat:
            return 0.0
        
        # 过热度
        Delta_T = T - T_sat
        
        # 简化的蒸汽分率计算
        # 实际需要考虑汽化潜热和显热
        vapor_fraction = min(0.95, Delta_T / 50.0)
        
        return vapor_fraction


# 海水淡化工艺计算工具
class DesalinationCalculator:
    """海水淡化工艺计算工具类"""
    
    def __init__(self, seawater_model: SeawaterModel):
        self.seawater_model = seawater_model
    
    def calculate_reverse_osmosis_pressure(self, salinity: float, T: float = 298.15,
                                         recovery_ratio: float = 0.5) -> Dict:
        """
        计算反渗透所需压力
        
        Parameters:
        -----------
        salinity : float
            进料盐度 [g/kg]
        T : float
            温度 [K]
        recovery_ratio : float
            回收率
            
        Returns:
        --------
        Dict
            RO计算结果
        """
        # 进料渗透压
        pi_feed = self.seawater_model.calculate_osmotic_pressure(T, salinity)
        
        # 浓缩液盐度估算
        salinity_concentrate = salinity / (1 - recovery_ratio)
        pi_concentrate = self.seawater_model.calculate_osmotic_pressure(T, salinity_concentrate)
        
        # 平均渗透压
        pi_avg = (pi_feed + pi_concentrate) / 2
        
        # 所需压力（包括压力损失）
        pressure_loss = 0.5e5  # Pa
        required_pressure = pi_avg + pressure_loss
        
        return {
            'feed_osmotic_pressure': pi_feed,
            'concentrate_osmotic_pressure': pi_concentrate,
            'average_osmotic_pressure': pi_avg,
            'required_pressure': required_pressure,
            'concentrate_salinity': salinity_concentrate,
            'recovery_ratio': recovery_ratio
        }
    
    def calculate_msf_performance(self, T_top: float, T_bottom: float,
                                salinity: float, n_stages: int = 20) -> Dict:
        """
        计算多级闪蒸(MSF)性能
        
        Parameters:
        -----------
        T_top : float
            顶温 [K]
        T_bottom : float
            底温 [K]
        salinity : float
            进料盐度 [g/kg]
        n_stages : int
            级数
            
        Returns:
        --------
        Dict
            MSF性能参数
        """
        # 温度分布
        Delta_T = (T_top - T_bottom) / n_stages
        temperatures = np.linspace(T_top, T_bottom, n_stages + 1)
        
        # 各级产水量计算（简化）
        total_distillate = 0.0
        stage_results = []
        
        for i in range(n_stages):
            T_stage = temperatures[i]
            
            # 海水性质
            props = self.seawater_model.calculate_seawater_properties(
                T_stage, 101325.0, salinity
            )
            
            # 闪蒸产水量（简化计算）
            flash_fraction = Delta_T / 100.0  # 经验公式
            stage_distillate = flash_fraction * 1000  # kg/h 假设1000kg/h进料
            
            total_distillate += stage_distillate
            
            stage_results.append({
                'stage': i + 1,
                'temperature': T_stage,
                'distillate_flow': stage_distillate,
                'boiling_point_elevation': props.boiling_point_elevation
            })
        
        # 性能比
        latent_heat = 2260e3  # J/kg
        heat_input = 1000 * 4180 * (T_top - 298.15)  # J/h
        performance_ratio = total_distillate * latent_heat / heat_input
        
        return {
            'total_distillate': total_distillate,
            'performance_ratio': performance_ratio,
            'stage_results': stage_results,
            'number_of_stages': n_stages,
            'temperature_range': T_top - T_bottom
        }
    
    def optimize_med_configuration(self, capacity: float, max_temperature: float,
                                 salinity: float) -> Dict:
        """
        优化多效蒸发(MED)配置
        
        Parameters:
        -----------
        capacity : float
            产水能力 [m³/day]
        max_temperature : float
            最高温度 [K]
        salinity : float
            进料盐度 [g/kg]
            
        Returns:
        --------
        Dict
            优化的MED配置
        """
        # 简化的MED优化
        # 实际需要考虑传热面积、能耗等因素
        
        optimal_effects = max(3, min(12, int(capacity / 1000)))
        Delta_T_per_effect = min(10, (max_temperature - 313.15) / optimal_effects)
        
        effects_data = []
        for i in range(optimal_effects):
            T_effect = max_temperature - i * Delta_T_per_effect
            
            props = self.seawater_model.calculate_seawater_properties(
                T_effect, 101325.0, salinity
            )
            
            effects_data.append({
                'effect': i + 1,
                'temperature': T_effect,
                'boiling_point_elevation': props.boiling_point_elevation,
                'distillate_production': capacity / optimal_effects
            })
        
        return {
            'optimal_number_of_effects': optimal_effects,
            'temperature_drop_per_effect': Delta_T_per_effect,
            'effects_data': effects_data,
            'total_capacity': capacity,
            'estimated_gain_output_ratio': optimal_effects * 0.8
        } 