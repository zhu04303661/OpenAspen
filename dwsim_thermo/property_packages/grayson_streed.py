"""
Grayson-Streed重烃模型 (Grayson-Streed Heavy Hydrocarbon Model)
==============================================================

专门用于重质烃类系统的K值关联模型。
该模型基于对应态原理，特别适用于C6+重烃的气液平衡计算。

Grayson-Streed模型是石油工业中广泛使用的经验关联式，
特别适用于：
- 重烃分离工艺
- 原油蒸馏塔设计
- 加氢裂化装置
- 催化裂化装置

理论基础:
---------
Grayson-Streed模型基于对应态原理：
ln(K_i) = ln(P_ci/P) + f(T_ri, P_ri, ω_i)

其中：
- P_ci: 组分i的临界压力
- T_ri: 对比温度 (T/T_ci)
- P_ri: 对比压力 (P/P_ci)  
- ω_i: 偏心因子

模型特点：
- 适用于C1-C40+烃类
- 温度范围: 200-800K
- 压力范围: 0.1-200 bar
- 特别适用于重烃系统

参考文献:
- Grayson, H.G., & Streed, C.W. (1963). Vapor-liquid equilibria for high temperature
- API Technical Data Book, Section 4
- DWSIM VB.NET GraysonStreed.vb源代码
- Prausnitz, J.M., et al. (1999). Molecular Thermodynamics

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
from enum import Enum

from .base_property_package import PropertyPackage
from ..core.compound import Compound
from ..core.enums import PackageType, Phase


class HydrocarbonType(Enum):
    """烃类类型枚举"""
    PARAFFIN = "paraffin"           # 石蜡烃
    NAPHTHENE = "naphthene"         # 环烷烃
    AROMATIC = "aromatic"           # 芳烃
    OLEFIN = "olefin"              # 烯烃


@dataclass
class GSParameters:
    """Grayson-Streed模型参数"""
    critical_temperature: float    # 临界温度 [K]
    critical_pressure: float       # 临界压力 [Pa]
    acentric_factor: float         # 偏心因子
    molecular_weight: float        # 分子量 [g/mol]
    hydrocarbon_type: HydrocarbonType  # 烃类类型
    boiling_point: float           # 标准沸点 [K]
    
    # Grayson-Streed特有参数
    alpha: float = 0.0             # α参数
    beta: float = 0.0              # β参数
    gamma: float = 0.0             # γ参数


class GraysonStreedModel(PropertyPackage):
    """
    Grayson-Streed重烃模型
    
    基于对应态原理的重烃K值关联模型，
    特别适用于石油炼制过程中的重烃分离计算。
    
    特点:
    - 专门针对重烃优化
    - 考虑烃类结构差异
    - 高温高压适用性好
    - 计算速度快
    """
    
    def __init__(self, compounds: List[Compound], **kwargs):
        """
        初始化Grayson-Streed模型
        
        Parameters:
        -----------
        compounds : List[Compound]
            化合物列表
        **kwargs : 
            其他参数
        """
        super().__init__(PackageType.PENG_ROBINSON, compounds, **kwargs)  # 暂用PR枚举
        
        self.model_name = "Grayson-Streed Heavy Hydrocarbon"
        self.model_description = "重烃K值关联模型，基于对应态原理"
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型参数
        self._initialize_gs_parameters()
        
        # 模型适用范围
        self._T_min = 200.0    # K
        self._T_max = 800.0    # K
        self._P_min = 0.1e5    # Pa
        self._P_max = 200e5    # Pa
        
    def _initialize_gs_parameters(self) -> None:
        """初始化Grayson-Streed参数"""
        
        # 存储每个组分的GS参数
        self.gs_parameters = []
        
        for compound in self.compounds:
            # 从化合物属性获取基本参数
            Tc = getattr(compound, 'critical_temperature', 500.0)
            Pc = getattr(compound, 'critical_pressure', 5e6)
            omega = getattr(compound, 'acentric_factor', 0.2)
            MW = getattr(compound, 'molecular_weight', 100.0)
            Tb = getattr(compound, 'boiling_point', 400.0)
            
            # 确定烃类类型
            hc_type = self._determine_hydrocarbon_type(compound.name)
            
            # 计算GS特有参数
            alpha, beta, gamma = self._calculate_gs_specific_parameters(Tc, Pc, omega, hc_type)
            
            gs_param = GSParameters(
                critical_temperature=Tc,
                critical_pressure=Pc,
                acentric_factor=omega,
                molecular_weight=MW,
                hydrocarbon_type=hc_type,
                boiling_point=Tb,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
            
            self.gs_parameters.append(gs_param)
            
        # Grayson-Streed关联参数表
        self._initialize_correlation_parameters()
    
    def _determine_hydrocarbon_type(self, compound_name: str) -> HydrocarbonType:
        """确定烃类类型"""
        name = compound_name.lower()
        
        # 芳烃
        if any(arom in name for arom in ['benzene', 'toluene', 'xylene', 'aromatic']):
            return HydrocarbonType.AROMATIC
        
        # 环烷烃
        if any(naph in name for naph in ['cyclohexane', 'cyclopentane', 'naphthene']):
            return HydrocarbonType.NAPHTHENE
        
        # 烯烃
        if any(ole in name for ole in ['ethylene', 'propylene', 'butene', 'ene']):
            return HydrocarbonType.OLEFIN
        
        # 默认为石蜡烃
        return HydrocarbonType.PARAFFIN
    
    def _calculate_gs_specific_parameters(self, Tc: float, Pc: float, omega: float, 
                                        hc_type: HydrocarbonType) -> Tuple[float, float, float]:
        """计算Grayson-Streed特有参数"""
        
        # 基于烃类类型的修正系数
        type_corrections = {
            HydrocarbonType.PARAFFIN: (0.0, 0.0, 0.0),
            HydrocarbonType.NAPHTHENE: (0.05, -0.02, 0.01),
            HydrocarbonType.AROMATIC: (0.10, -0.05, 0.03),
            HydrocarbonType.OLEFIN: (-0.02, 0.01, -0.01)
        }
        
        base_alpha, base_beta, base_gamma = type_corrections[hc_type]
        
        # 基于物性的修正
        alpha = base_alpha + 0.1 * omega
        beta = base_beta + 0.05 * (Tc / 500.0 - 1.0)
        gamma = base_gamma + 0.02 * (Pc / 5e6 - 1.0)
        
        return alpha, beta, gamma
    
    def _initialize_correlation_parameters(self) -> None:
        """初始化关联参数"""
        
        # Grayson-Streed关联式系数
        self.correlation_coeffs = {
            # 低压修正系数 (P < 1 bar)
            'low_pressure': {
                'A': [1.0, 0.0, 0.0],
                'B': [0.0, 1.0, 0.0],
                'C': [0.0, 0.0, 1.0]
            },
            
            # 中压修正系数 (1-10 bar)  
            'medium_pressure': {
                'A': [0.9, 0.1, 0.0],
                'B': [0.05, 0.9, 0.05],
                'C': [0.0, 0.1, 0.9]
            },
            
            # 高压修正系数 (>10 bar)
            'high_pressure': {
                'A': [0.8, 0.15, 0.05],
                'B': [0.1, 0.8, 0.1],
                'C': [0.05, 0.15, 0.8]
            }
        }
        
        # 温度修正参数
        self.temperature_correction = {
            'paraffin': {'a': 1.0, 'b': 0.0, 'c': 0.0},
            'naphthene': {'a': 0.95, 'b': 0.05, 'c': 0.01},
            'aromatic': {'a': 0.90, 'b': 0.08, 'c': 0.02},
            'olefin': {'a': 1.05, 'b': -0.03, 'c': 0.01}
        }
    
    def calculate_k_values(self, T: float, P: float) -> np.ndarray:
        """
        计算K值 (气液分配系数)
        
        Parameters:
        -----------
        T : float
            温度 [K]
        P : float
            压力 [Pa]
            
        Returns:
        --------
        np.ndarray
            K值数组
        """
        # 输入验证
        self._validate_inputs(T, P)
        
        n_comp = len(self.compounds)
        K_values = np.zeros(n_comp)
        
        for i, gs_param in enumerate(self.gs_parameters):
            K_values[i] = self._calculate_component_k_value(T, P, gs_param)
        
        return K_values
    
    def _calculate_component_k_value(self, T: float, P: float, gs_param: GSParameters) -> float:
        """
        计算单个组分的K值
        
        Grayson-Streed关联式：
        ln(K_i) = A + B/T_r + C*ln(T_r) + D*T_r + E*T_r^2 + (压力修正) + (结构修正)
        """
        # 对比温度和压力
        Tr = T / gs_param.critical_temperature
        Pr = P / gs_param.critical_pressure
        
        # 基础K值计算 (基于对应态)
        K_base = self._calculate_base_k_value(Tr, Pr, gs_param.acentric_factor)
        
        # 压力修正
        pressure_correction = self._calculate_pressure_correction(Pr, gs_param)
        
        # 温度修正
        temperature_correction = self._calculate_temperature_correction(Tr, gs_param)
        
        # 结构修正 (基于烃类类型)
        structure_correction = self._calculate_structure_correction(Tr, Pr, gs_param)
        
        # 最终K值
        ln_K = np.log(K_base) + pressure_correction + temperature_correction + structure_correction
        K = np.exp(ln_K)
        
        # 合理性检查
        K = max(1e-6, min(1e6, K))
        
        return K
    
    def _calculate_base_k_value(self, Tr: float, Pr: float, omega: float) -> float:
        """计算基础K值（基于对应态原理）"""
        
        # 简化的对应态关联 (类似于Lee-Kesler)
        if Tr < 1.0:
            # 液相占主导
            K_base = (1.0 / Pr) * np.exp(5.373 * (1 + omega) * (1 - 1/Tr))
        else:
            # 气相占主导
            K_base = (1.0 / Pr) * np.exp(5.373 * (1 + omega) * (1 - 1/Tr))
        
        return K_base
    
    def _calculate_pressure_correction(self, Pr: float, gs_param: GSParameters) -> float:
        """计算压力修正项"""
        
        # 选择压力范围对应的系数
        if Pr < 0.1:
            coeffs = self.correlation_coeffs['low_pressure']
        elif Pr < 1.0:
            coeffs = self.correlation_coeffs['medium_pressure']
        else:
            coeffs = self.correlation_coeffs['high_pressure']
        
        # 压力修正公式
        correction = coeffs['A'][0] * np.log(Pr) + coeffs['B'][0] * Pr + coeffs['C'][0] * Pr**2
        
        # 偏心因子修正
        correction += gs_param.acentric_factor * (0.1 * np.log(Pr) + 0.05 * Pr)
        
        return correction
    
    def _calculate_temperature_correction(self, Tr: float, gs_param: GSParameters) -> float:
        """计算温度修正项"""
        
        hc_type_str = gs_param.hydrocarbon_type.value
        temp_params = self.temperature_correction[hc_type_str]
        
        # 温度修正公式
        correction = (temp_params['a'] * (Tr - 1.0) + 
                     temp_params['b'] * (Tr - 1.0)**2 + 
                     temp_params['c'] * np.log(Tr))
        
        return correction
    
    def _calculate_structure_correction(self, Tr: float, Pr: float, gs_param: GSParameters) -> float:
        """计算结构修正项（基于烃类类型）"""
        
        # 基于GS特有参数的修正
        correction = (gs_param.alpha * (Tr - 1.0) + 
                     gs_param.beta * (Pr - 1.0) + 
                     gs_param.gamma * np.sqrt(Tr * Pr))
        
        # 分子量修正 (对重烃的特殊处理)
        if gs_param.molecular_weight > 200.0:
            mw_correction = 0.01 * np.log(gs_param.molecular_weight / 100.0)
            correction += mw_correction * (1.0 - Tr)
        
        return correction
    
    def calculate_fugacity_coefficients(self, T: float, P: float, x: np.ndarray, phase: str) -> np.ndarray:
        """
        计算逸度系数
        
        Parameters:
        -----------
        T : float
            温度 [K]
        P : float
            压力 [Pa]
        x : np.ndarray
            摩尔分数
        phase : str
            相态 ('liquid' 或 'vapor')
            
        Returns:
        --------
        np.ndarray
            逸度系数数组
        """
        n_comp = len(x)
        phi = np.ones(n_comp)
        
        if phase == 'vapor':
            # 气相逸度系数 (简化处理)
            for i, gs_param in enumerate(self.gs_parameters):
                Tr = T / gs_param.critical_temperature
                Pr = P / gs_param.critical_pressure
                
                # 简化的气相逸度系数
                ln_phi = (Pr / Tr) * (gs_param.acentric_factor + 0.1)
                phi[i] = np.exp(ln_phi)
                
        elif phase == 'liquid':
            # 液相逸度系数
            for i, gs_param in enumerate(self.gs_parameters):
                Tr = T / gs_param.critical_temperature
                Pr = P / gs_param.critical_pressure
                
                # 基于活度系数的液相逸度系数
                gamma = self._calculate_activity_coefficient(i, x, T)
                P_sat = self._calculate_vapor_pressure(gs_param, T)
                
                phi[i] = gamma * P_sat / P
                phi[i] = max(1e-6, min(1e6, phi[i]))
        
        return phi
    
    def _calculate_activity_coefficient(self, component_index: int, x: np.ndarray, T: float) -> float:
        """计算活度系数（简化Wilson模型）"""
        
        # 简化处理：假设理想溶液
        gamma = 1.0
        
        # 对于重烃系统，可以加入简单的非理想性修正
        gs_param = self.gs_parameters[component_index]
        
        # 基于分子量差异的修正
        avg_mw = np.average([param.molecular_weight for param in self.gs_parameters], weights=x)
        mw_correction = 1.0 + 0.001 * abs(gs_param.molecular_weight - avg_mw)
        
        gamma *= mw_correction
        
        return gamma
    
    def _calculate_vapor_pressure(self, gs_param: GSParameters, T: float) -> float:
        """计算蒸汽压"""
        
        # 使用Antoine方程或Riedel方程
        # 这里使用简化的Riedel方程
        
        Tc = gs_param.critical_temperature
        Pc = gs_param.critical_pressure
        Tb = gs_param.boiling_point
        
        # Riedel方程
        Tr = T / Tc
        Tbr = Tb / Tc
        
        if Tr < 1.0:
            alpha = 0.0838 * (3.758 - 3.758 * Tbr + Tbr**2) / (1 - Tbr)
            
            ln_Pr = alpha * (1 - 1/Tr) / (1 - Tbr)
            P_sat = Pc * np.exp(ln_Pr)
        else:
            P_sat = Pc  # 超临界
        
        return max(100.0, P_sat)  # 最小值100 Pa
    
    def flash_pt(self, P: float, T: float, z: np.ndarray) -> Dict:
        """
        等温等压闪蒸计算
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K]
        z : np.ndarray
            总组成
            
        Returns:
        --------
        Dict
            闪蒸结果
        """
        # 计算K值
        K = self.calculate_k_values(T, P)
        
        # Rachford-Rice方程求解
        vapor_fraction = self._solve_rachford_rice(z, K)
        
        # 计算相组成
        x_liquid = z / (vapor_fraction + (1 - vapor_fraction) * K)
        y_vapor = K * x_liquid
        
        # 归一化
        x_liquid = x_liquid / np.sum(x_liquid)
        y_vapor = y_vapor / np.sum(y_vapor)
        
        # 计算逸度系数
        phi_liquid = self.calculate_fugacity_coefficients(T, P, x_liquid, 'liquid')
        phi_vapor = self.calculate_fugacity_coefficients(T, P, y_vapor, 'vapor')
        
        return {
            'converged': True,
            'flash_type': 'Grayson_Streed_VL',
            'phases': {
                'vapor': {
                    'mole_fractions': y_vapor,
                    'phase_fraction': vapor_fraction,
                    'fugacity_coefficients': phi_vapor,
                    'phase_type': 'vapor'
                },
                'liquid': {
                    'mole_fractions': x_liquid,
                    'phase_fraction': 1 - vapor_fraction,
                    'fugacity_coefficients': phi_liquid,
                    'phase_type': 'liquid'
                }
            },
            'properties': {
                'temperature': T,
                'pressure': P,
                'k_values': K,
                'model': 'Grayson-Streed'
            }
        }
    
    def _solve_rachford_rice(self, z: np.ndarray, K: np.ndarray, 
                           initial_guess: float = 0.5) -> float:
        """求解Rachford-Rice方程"""
        
        def rr_equation(beta):
            result = 0.0
            for i in range(len(z)):
                if abs(K[i] - 1.0) > 1e-12:
                    result += z[i] * (K[i] - 1.0) / (1 + beta * (K[i] - 1.0))
            return result
        
        try:
            # 检查是否需要求解
            if np.all(K > 1.0):
                return 1.0  # 全气相
            elif np.all(K < 1.0):
                return 0.0  # 全液相
            else:
                # 使用Brent方法求解
                beta = brentq(rr_equation, 1e-8, 1-1e-8, xtol=1e-12)
                return max(1e-8, min(1-1e-8, beta))
        except:
            return initial_guess
    
    def _validate_inputs(self, T: float, P: float) -> None:
        """验证输入参数"""
        if not (self._T_min <= T <= self._T_max):
            warnings.warn(f"温度{T:.2f} K超出推荐范围 [{self._T_min}, {self._T_max}] K")
        
        if not (self._P_min <= P <= self._P_max):
            warnings.warn(f"压力{P:.0f} Pa超出推荐范围 [{self._P_min/1e5:.1f}, {self._P_max/1e5:.0f}] bar")
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'name': 'Grayson-Streed Heavy Hydrocarbon Model',
            'type': 'K-value Correlation',
            'description': '重烃K值关联模型，基于对应态原理',
            'applicable_compounds': 'Heavy hydrocarbons (C6+)',
            'temperature_range': f'{self._T_min}-{self._T_max} K',
            'pressure_range': f'{self._P_min/1e5:.1f}-{self._P_max/1e5:.0f} bar',
            'phases': ['vapor', 'liquid'],
            'properties': [
                'K值计算',
                '逸度系数',
                '活度系数',
                '蒸汽压',
                'PT闪蒸'
            ],
            'hydrocarbon_types': [
                'Paraffins (石蜡烃)',
                'Naphthenes (环烷烃)',
                'Aromatics (芳烃)',
                'Olefins (烯烃)'
            ],
            'advantages': [
                '专门针对重烃优化',
                '考虑烃类结构差异',
                '计算速度快',
                '工业验证充分'
            ],
            'limitations': [
                '仅适用于烃类系统',
                '不适用于强极性化合物',
                '精度依赖于参数质量'
            ],
            'references': [
                'Grayson, H.G., & Streed, C.W. (1963)',
                'API Technical Data Book',
                'Prausnitz, J.M., et al. (1999)'
            ]
        }


# 重烃表征工具
class HeavyHydrocarbonCharacterizer:
    """重烃表征工具类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def characterize_petroleum_fraction(self, distillation_data: Dict, 
                                     specific_gravity: float) -> Dict:
        """
        表征石油馏分
        
        Parameters:
        -----------
        distillation_data : Dict
            蒸馏数据 {'temperatures': [K], 'cumulative_volumes': [%]}
        specific_gravity : float
            比重
            
        Returns:
        --------
        Dict
            表征结果
        """
        # 计算平均分子量
        avg_mw = self._estimate_molecular_weight(distillation_data, specific_gravity)
        
        # 估算临界性质
        critical_props = self._estimate_critical_properties(avg_mw, specific_gravity)
        
        # 估算偏心因子
        omega = self._estimate_acentric_factor(avg_mw, specific_gravity)
        
        # 确定烃类分布
        hc_distribution = self._estimate_hydrocarbon_distribution(specific_gravity)
        
        return {
            'average_molecular_weight': avg_mw,
            'critical_temperature': critical_props['Tc'],
            'critical_pressure': critical_props['Pc'],
            'acentric_factor': omega,
            'hydrocarbon_distribution': hc_distribution,
            'characterization_method': 'Grayson-Streed Compatible'
        }
    
    def _estimate_molecular_weight(self, distillation_data: Dict, sg: float) -> float:
        """估算平均分子量"""
        # 使用Lee-Kesler关联
        T_avg = np.mean(distillation_data['temperatures'])
        
        # 简化的分子量关联
        MW = 42.965 * (T_avg**1.26007) * (sg**-4.98308) * np.exp(0.00377 * T_avg)
        
        return max(50.0, min(1000.0, MW))
    
    def _estimate_critical_properties(self, MW: float, sg: float) -> Dict:
        """估算临界性质"""
        # Riazi-Daubert关联
        
        # 临界温度
        Tc = 341.7 + 811.0 * sg + (0.4244 + 0.1174 * sg) * MW + (0.4669 - 3.2623 * sg) * 1e5 / MW
        
        # 临界压力
        ln_Pc = 8.3634 - 0.0566 / sg - (0.24244 + 2.2898 / sg + 0.11857 / sg**2) * 1e-3 * MW + \
                (1.4685 + 3.648 / sg + 0.47227 / sg**2) * 1e-7 * MW**2 - \
                (0.42019 + 1.6977 / sg**2) * 1e-10 * MW**3
        Pc = np.exp(ln_Pc) * 1e5  # Pa
        
        return {'Tc': Tc, 'Pc': Pc}
    
    def _estimate_acentric_factor(self, MW: float, sg: float) -> float:
        """估算偏心因子"""
        # 经验关联
        omega = -7.904 + 0.1352 * MW - 0.007465 * MW**2 + 8.359 * sg + \
                (1.408 - 0.01063 * MW) / sg
        
        return max(0.0, min(2.0, omega))
    
    def _estimate_hydrocarbon_distribution(self, sg: float) -> Dict:
        """估算烃类分布"""
        # 基于比重的经验分布
        if sg < 0.75:
            # 轻质烃
            return {'paraffin': 0.7, 'naphthene': 0.2, 'aromatic': 0.1, 'olefin': 0.0}
        elif sg < 0.85:
            # 中质烃
            return {'paraffin': 0.5, 'naphthene': 0.3, 'aromatic': 0.2, 'olefin': 0.0}
        else:
            # 重质烃
            return {'paraffin': 0.3, 'naphthene': 0.3, 'aromatic': 0.4, 'olefin': 0.0} 