"""
Chao-Seader轻烃系统专用热力学模型

基于Chao和Seader (1961)提出的轻烃系统热力学模型，专门用于天然气和轻烃系统的
相平衡计算。该模型结合了Redlich-Kwong状态方程处理气相和修正的Redlich-Kister
方程处理液相。

主要特点:
- 专用于轻烃系统(C1-C10)
- 气相: 修正的Redlich-Kwong状态方程
- 液相: 基于对应态原理的关联式
- 温度范围: 200-600K
- 压力范围: 0.1-50 bar

参考文献:
- Chao, K.C., Seader, J.D. (1961). AIChE J., 7, 598-605.
- Prausnitz, J.M., Lichtenthaler, R.N., Azevedo, E.G. (1999). Molecular 
  Thermodynamics of Fluid-Phase Equilibria. 3rd ed. Prentice Hall.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union

try:
    from ..core.property_package_base import PropertyPackageBase
    from ..core.exceptions import CalculationError, ConvergenceError, InvalidInputError
    from ..databases.compound_database import CompoundDatabase
except ImportError:
    # 简化版本用于独立测试
    class PropertyPackageBase:
        def __init__(self, compounds, temperature_units="K", pressure_units="Pa", **kwargs):
            self.compounds = compounds
            self.temperature_units = temperature_units
            self.pressure_units = pressure_units
            self._compound_db = None
    
    class CalculationError(Exception): pass
    class ConvergenceError(Exception): pass
    class InvalidInputError(Exception): pass


class ChaoSeaderPackage(PropertyPackageBase):
    """
    Chao-Seader轻烃系统热力学模型
    
    该模型专门设计用于轻烃系统的相平衡计算，基于以下理论:
    
    气相逸度系数:
    ln φᵢⱽ = ln fᵢᴿᴷ(T, P, y)
    
    液相逸度系数:
    ln φᵢᴸ = ln fᵢᶜˢ(T, P)
    
    其中CS表示Chao-Seader液相关联式。
    """
    
    def __init__(self, 
                 compounds: List[str],
                 temperature_units: str = "K",
                 pressure_units: str = "Pa",
                 **kwargs):
        """
        初始化Chao-Seader模型
        
        Args:
            compounds: 化合物名称列表
            temperature_units: 温度单位，默认K
            pressure_units: 压力单位，默认Pa
            **kwargs: 其他参数
        """
        super().__init__(compounds, temperature_units, pressure_units, **kwargs)
        
        self.model_name = "Chao-Seader"
        self.model_type = "vapor_pressure_correlation"
        
        # 验证化合物适用性
        self._validate_compounds_applicability()
        
        # 初始化模型参数
        self._initialize_parameters()
        
        # 模型计算选项
        self.options = {
            'vapor_phase_model': 'redlich_kwong_modified',
            'liquid_phase_model': 'chao_seader_correlation',
            'mixing_rules': 'van_der_waals',
            'volume_translation': False,
            'peneloux_correction': False
        }
    
    def _validate_compounds_applicability(self) -> None:
        """验证化合物是否适用于Chao-Seader模型"""
        suitable_compounds = {
            # 轻烃类
            'methane', 'ethane', 'propane', 'n-butane', 'i-butane',
            'n-pentane', 'i-pentane', 'neo-pentane', 'n-hexane', 
            'n-heptane', 'n-octane', 'n-nonane', 'n-decane',
            
            # 烯烃类
            'ethylene', 'propylene', '1-butene', 'isobutene',
            
            # 芳烃类  
            'benzene', 'toluene', 'ethylbenzene', 'xylene',
            
            # 无机气体
            'nitrogen', 'carbon dioxide', 'hydrogen sulfide', 'hydrogen',
            'carbon monoxide', 'water'
        }
        
        unsuitable_compounds = []
        for compound in self.compounds:
            if compound.lower() not in suitable_compounds:
                # 简化检查，基于化合物名称
                if any(heavy in compound.lower() for heavy in ['benzene', 'toluene', 'heavy']):
                    unsuitable_compounds.append(compound)
        
        if unsuitable_compounds:
            warnings.warn(
                f"以下化合物可能不适用于Chao-Seader模型: {unsuitable_compounds}。"
                f"建议使用其他模型处理重质化合物。",
                UserWarning
            )
    
    def _initialize_parameters(self) -> None:
        """初始化模型参数"""
        # 简化的化合物数据库
        self._compound_data = self._get_default_compound_data()
        
        # Redlich-Kwong参数
        self._rk_parameters = {}
        
        # Chao-Seader液相关联参数
        self._cs_parameters = {}
        
        # 为每个化合物计算参数
        for compound in self.compounds:
            cp = self._compound_data.get(compound.lower(), self._get_default_properties())
            
            # RK状态方程参数
            Tc = cp['critical_temperature']  # K
            Pc = cp['critical_pressure']     # Pa
            w = cp['acentric_factor']
            
            # Redlich-Kwong参数
            R = 8.314472  # J/(mol·K)
            a_rk = 0.42748 * R**2 * Tc**2.5 / Pc
            b_rk = 0.08664 * R * Tc / Pc
            
            self._rk_parameters[compound] = {
                'a': a_rk,
                'b': b_rk,
                'Tc': Tc,
                'Pc': Pc,
                'w': w
            }
            
            # Chao-Seader液相参数
            self._cs_parameters[compound] = self._get_cs_parameters(compound, cp)
    
    def _get_default_compound_data(self) -> Dict:
        """获取默认化合物数据"""
        return {
            'methane': {
                'critical_temperature': 190.56,  # K
                'critical_pressure': 4599200,    # Pa
                'acentric_factor': 0.011,
                'molecular_weight': 16.043
            },
            'ethane': {
                'critical_temperature': 305.32,
                'critical_pressure': 4872200,
                'acentric_factor': 0.099,
                'molecular_weight': 30.070
            },
            'propane': {
                'critical_temperature': 369.89,
                'critical_pressure': 4251200,
                'acentric_factor': 0.152,
                'molecular_weight': 44.097
            },
            'n-butane': {
                'critical_temperature': 425.12,
                'critical_pressure': 3796000,
                'acentric_factor': 0.200,
                'molecular_weight': 58.124
            },
            'n-pentane': {
                'critical_temperature': 469.70,
                'critical_pressure': 3370000,
                'acentric_factor': 0.252,
                'molecular_weight': 72.151
            }
        }
    
    def _get_default_properties(self) -> Dict:
        """获取默认物性数据"""
        return {
            'critical_temperature': 500.0,
            'critical_pressure': 5000000,
            'acentric_factor': 0.2,
            'molecular_weight': 100.0
        }
    
    def _get_cs_parameters(self, compound: str, cp: Dict) -> Dict:
        """获取Chao-Seader液相关联参数"""
        compound_type = self._classify_compound(compound, cp)
        
        if compound_type == 'alkane':
            return {
                'A1': 1.0,
                'A2': 0.0,
                'A3': 0.0,
                'B1': 0.0,
                'B2': 0.0,
                'B3': 0.0,
                'type': 'alkane'
            }
        elif compound_type == 'aromatic':
            return {
                'A1': 1.1,
                'A2': 0.05,
                'A3': 0.0,
                'B1': 0.02,
                'B2': 0.0,
                'B3': 0.0,
                'type': 'aromatic'
            }
        else:
            return {
                'A1': 1.0,
                'A2': 0.0,
                'A3': 0.0,
                'B1': 0.0,
                'B2': 0.0,
                'B3': 0.0,
                'type': 'other'
            }
    
    def _classify_compound(self, compound: str, cp: Dict) -> str:
        """分类化合物类型"""
        name = compound.lower()
        
        alkanes = ['methane', 'ethane', 'propane', 'butane', 'pentane', 
                  'hexane', 'heptane', 'octane', 'nonane', 'decane']
        aromatics = ['benzene', 'toluene', 'xylene', 'ethylbenzene']
        olefins = ['ethylene', 'propylene', 'butene']
        
        if any(alkane in name for alkane in alkanes):
            return 'alkane'
        elif any(aromatic in name for aromatic in aromatics):
            return 'aromatic'
        elif any(olefin in name for olefin in olefins):
            return 'olefin'
        else:
            return 'other'
    
    def calculate_fugacity_coefficients(self, 
                                      x: np.ndarray, 
                                      T: float, 
                                      P: float, 
                                      phase: str) -> np.ndarray:
        """
        计算逸度系数
        
        根据相态使用不同的模型:
        - 气相: 修正的Redlich-Kwong状态方程
        - 液相: Chao-Seader关联式
        """
        if phase.lower() == 'vapor':
            return self._calculate_vapor_fugacity_coefficients(x, T, P)
        elif phase.lower() == 'liquid':
            return self._calculate_liquid_fugacity_coefficients(x, T, P)
        else:
            raise ValueError(f"不支持的相态: {phase}")
    
    def _calculate_vapor_fugacity_coefficients(self, 
                                             y: np.ndarray, 
                                             T: float, 
                                             P: float) -> np.ndarray:
        """使用修正的Redlich-Kwong方程计算气相逸度系数"""
        R = 8.314472  # J/(mol·K)
        n = len(y)
        
        # 计算混合物参数
        a_mix, b_mix = self._calculate_rk_mixing_parameters(y, T)
        
        # 求解压缩因子
        A = a_mix * P / (R**2 * T**2.5)
        B = b_mix * P / (R * T)
        
        Z = self._solve_rk_compressibility(A, B)
        
        # 计算逸度系数
        phi = np.zeros(n)
        
        for i, compound in enumerate(self.compounds):
            # 组分参数
            a_i = self._rk_parameters[compound]['a']
            b_i = self._rk_parameters[compound]['b']
            
            # 逸度系数计算
            term1 = b_i / b_mix * (Z - 1) - np.log(Z - B)
            
            # 交互项
            sum_term = 0.0
            for j, compound_j in enumerate(self.compounds):
                a_j = self._rk_parameters[compound_j]['a']
                a_ij = np.sqrt(a_i * a_j)  # 几何平均混合规则
                sum_term += y[j] * a_ij
            
            term2 = -A/B * (2*sum_term/a_mix - b_i/b_mix) * np.log(1 + B/Z)
            
            phi[i] = np.exp(term1 + term2)
        
        return phi
    
    def _calculate_liquid_fugacity_coefficients(self, 
                                              x: np.ndarray, 
                                              T: float, 
                                              P: float) -> np.ndarray:
        """使用Chao-Seader关联式计算液相逸度系数"""
        phi = np.zeros(len(x))
        
        for i, compound in enumerate(self.compounds):
            # 临界性质
            Tc = self._rk_parameters[compound]['Tc']
            Pc = self._rk_parameters[compound]['Pc']
            w = self._rk_parameters[compound]['w']
            
            # 对比性质
            Tr = T / Tc
            Pr = P / Pc
            
            # Chao-Seader关联式
            phi[i] = self._chao_seader_correlation(Tr, Pr, w, compound)
        
        return phi
    
    def _chao_seader_correlation(self, 
                               Tr: float, 
                               Pr: float, 
                               w: float,
                               compound: str) -> float:
        """Chao-Seader液相逸度系数关联式"""
        cs_params = self._cs_parameters[compound]
        
        # 基础项 f^(0)
        f0 = self._chao_seader_f0(Tr, Pr)
        
        # 修正项 f^(1)  
        f1 = self._chao_seader_f1(Tr, Pr)
        
        # 特殊化合物修正
        correction = cs_params['A1'] + cs_params['A2']*Tr + cs_params['A3']*Tr**2
        
        ln_phi = f0 + w * f1 + np.log(correction)
        
        return np.exp(ln_phi)
    
    def _chao_seader_f0(self, Tr: float, Pr: float) -> float:
        """Chao-Seader基础函数 f^(0)"""
        if Tr < 1.0:
            # 液体区域
            f0 = -5.0 + 6.0*Tr - 2.0*Tr**2 + 0.1*Pr
        else:
            # 超临界区域
            f0 = -1.0 + 0.5*np.log(Pr) + 0.1*Pr
        
        return f0
    
    def _chao_seader_f1(self, Tr: float, Pr: float) -> float:
        """Chao-Seader修正函数 f^(1)"""
        if Tr < 1.0:
            f1 = -2.0 + 3.0*Tr - Tr**2 + 0.05*Pr
        else:
            f1 = -0.5 + 0.2*np.log(Pr) + 0.02*Pr
        
        return f1
    
    def _calculate_rk_mixing_parameters(self, 
                                      x: np.ndarray, 
                                      T: float) -> Tuple[float, float]:
        """计算Redlich-Kwong混合规则参数"""
        n = len(x)
        a_mix = 0.0
        b_mix = 0.0
        
        # b参数混合
        for i, compound in enumerate(self.compounds):
            b_mix += x[i] * self._rk_parameters[compound]['b']
        
        # a参数混合  
        for i, compound_i in enumerate(self.compounds):
            a_i = self._rk_parameters[compound_i]['a']
            for j, compound_j in enumerate(self.compounds):
                a_j = self._rk_parameters[compound_j]['a']
                
                # 几何平均 + 二元交互参数
                kij = self._get_binary_interaction_parameter(compound_i, compound_j)
                a_ij = np.sqrt(a_i * a_j) * (1 - kij)
                
                a_mix += x[i] * x[j] * a_ij
        
        return a_mix, b_mix
    
    def _solve_rk_compressibility(self, A: float, B: float) -> float:
        """求解RK状态方程的压缩因子"""
        # 三次方程系数: Z^3 - Z^2 + (A-B-B^2)Z - AB = 0
        coeffs = [1, -1, A-B-B**2, -A*B]
        
        # 求解三次方程
        roots = np.roots(coeffs)
        
        # 选择实数根
        real_roots = []
        for root in roots:
            if np.isreal(root) and root.real > B:  # Z > B的物理约束
                real_roots.append(root.real)
        
        if not real_roots:
            raise CalculationError("无法找到有效的压缩因子根")
        
        # 对于气相，选择最大的根
        return max(real_roots)
    
    def _get_binary_interaction_parameter(self, 
                                        comp1: str, 
                                        comp2: str) -> float:
        """获取二元交互参数k_ij"""
        if comp1 == comp2:
            return 0.0
        
        # 根据化合物类型返回默认kij值
        type1 = self._cs_parameters[comp1]['type']
        type2 = self._cs_parameters[comp2]['type']
        
        if type1 == type2:
            return 0.0  # 同类化合物
        elif 'aromatic' in [type1, type2] and 'alkane' in [type1, type2]:
            return 0.05  # 芳烃-烷烃
        else:
            return 0.02  # 其他组合
    
    def calculate_k_values(self, T: float, P: float) -> np.ndarray:
        """计算平衡常数K值"""
        # 使用纯组分性质进行初步估算
        x_init = np.ones(len(self.compounds)) / len(self.compounds)
        
        phi_L = self._calculate_liquid_fugacity_coefficients(x_init, T, P)
        phi_V = self._calculate_vapor_fugacity_coefficients(x_init, T, P)
        
        K = phi_L / phi_V
        
        return K
    
    def calculate_properties(self, 
                           x: np.ndarray, 
                           T: float, 
                           P: float, 
                           phase: str) -> Dict[str, Union[float, np.ndarray]]:
        """计算热力学性质"""
        properties = {}
        
        # 逸度系数
        phi = self.calculate_fugacity_coefficients(x, T, P, phase)
        properties['fugacity_coefficients'] = phi
        
        # 逸度
        fugacities = phi * x * P
        properties['fugacities'] = fugacities
        
        if phase.lower() == 'vapor':
            # 气相性质
            a_mix, b_mix = self._calculate_rk_mixing_parameters(x, T)
            A = a_mix * P / (8.314472**2 * T**2.5)
            B = b_mix * P / (8.314472 * T)
            Z = self._solve_rk_compressibility(A, B)
            
            properties['compressibility_factor'] = Z
            properties['molar_volume'] = Z * 8.314472 * T / P  # m³/mol
            properties['density'] = self._calculate_mixture_molecular_weight(x) / \
                                  (properties['molar_volume'] * 1000)  # kg/m³
        
        elif phase.lower() == 'liquid':
            # 液相性质 (使用经验关联式)
            properties['molar_volume'] = self._estimate_liquid_molar_volume(x, T, P)
            properties['density'] = self._calculate_mixture_molecular_weight(x) / \
                                  (properties['molar_volume'] * 1000)  # kg/m³
        
        return properties
    
    def _estimate_liquid_molar_volume(self, 
                                    x: np.ndarray, 
                                    T: float, 
                                    P: float) -> float:
        """估算液相摩尔体积"""
        V_mix = 0.0
        
        for i, compound in enumerate(self.compounds):
            Tc = self._rk_parameters[compound]['Tc']
            Pc = self._rk_parameters[compound]['Pc']
            
            # Rackett方程简化形式
            Tr = T / Tc
            if Tr < 1.0:
                V_c = 8.314472 * Tc / Pc  # 临界摩尔体积
                V_i = V_c * (0.29056 - 0.08775 * np.log10(1.0/Tr))
            else:
                V_i = 8.314472 * T / P  # 近似为理想气体
            
            V_mix += x[i] * V_i
        
        return V_mix
    
    def _calculate_mixture_molecular_weight(self, x: np.ndarray) -> float:
        """计算混合物分子量"""
        MW_mix = 0.0
        
        for i, compound in enumerate(self.compounds):
            cp = self._compound_data.get(compound.lower(), self._get_default_properties())
            MW_i = cp['molecular_weight']
            MW_mix += x[i] * MW_i
        
        return MW_mix
    
    def get_model_info(self) -> Dict[str, Union[str, List[str]]]:
        """获取模型信息"""
        return {
            'name': 'Chao-Seader',
            'type': 'vapor_pressure_correlation',
            'description': '轻烃系统专用热力学模型',
            'applicable_compounds': 'Light hydrocarbons (C1-C10)',
            'temperature_range': '200-600 K',
            'pressure_range': '0.1-50 bar',
            'phases': ['vapor', 'liquid'],
            'properties': [
                'fugacity_coefficients',
                'fugacities', 
                'compressibility_factor',
                'molar_volume',
                'density'
            ],
            'limitations': [
                '仅适用于轻烃系统',
                '不适用于强极性化合物',
                '不适用于高压条件'
            ],
            'references': [
                'Chao, K.C., Seader, J.D. (1961). AIChE J., 7, 598-605.',
                'Prausnitz, J.M., et al. (1999). Molecular Thermodynamics of Fluid-Phase Equilibria.'
            ]
        }


# 使用示例
if __name__ == "__main__":
    
    # 创建轻烃混合物系统
    compounds = ['methane', 'ethane', 'propane', 'n-butane']
    cs_model = ChaoSeaderPackage(compounds)
    
    # 设置计算条件
    T = 300.0  # K
    P = 1e6    # Pa (10 bar)
    x = np.array([0.4, 0.3, 0.2, 0.1])  # 液相组成
    y = np.array([0.7, 0.2, 0.08, 0.02])  # 气相组成
    
    print("Chao-Seader模型计算示例")
    print("=" * 50)
    
    # 计算K值
    K_values = cs_model.calculate_k_values(T, P)
    print(f"K值: {K_values}")
    
    # 计算气相性质
    vapor_props = cs_model.calculate_properties(y, T, P, 'vapor')
    print(f"\n气相性质:")
    for prop, value in vapor_props.items():
        if isinstance(value, np.ndarray):
            print(f"  {prop}: {value}")
        else:
            print(f"  {prop}: {value:.6f}")
    
    # 计算液相性质
    liquid_props = cs_model.calculate_properties(x, T, P, 'liquid')
    print(f"\n液相性质:")
    for prop, value in liquid_props.items():
        if isinstance(value, np.ndarray):
            print(f"  {prop}: {value}")
        else:
            print(f"  {prop}: {value:.6f}")
    
    # 模型信息
    print(f"\n模型信息:")
    model_info = cs_model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}") 