"""
黑油模型 (Black Oil Model)
用于石油工业的拟组分模型，适用于油藏工程和石油加工

作者: OpenAspen项目组
版本: 1.0.0
"""

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from typing import List, Dict, Optional, Tuple
from ..base_property_package import PropertyPackage


class BlackOilModel(PropertyPackage):
    """
    黑油模型实现
    
    特点:
    - 基于PVT关联式
    - 适用于石油馏分
    - 处理拟组分
    - 高温高压适用性
    """
    
    def __init__(self, compounds: List[str], api_gravity: float = 35.0):
        """
        初始化黑油模型
        
        Parameters:
        -----------
        compounds : List[str]
            组分列表
        api_gravity : float
            API重度 (默认35°API)
        """
        super().__init__(compounds, "Black Oil Model")
        self.api_gravity = api_gravity
        self.specific_gravity = 141.5 / (131.5 + api_gravity)
        
        # 模型参数
        self._initialize_black_oil_parameters()
    
    def _initialize_black_oil_parameters(self):
        """初始化黑油模型参数"""
        # Standing关联式参数
        self.standing_coeffs = {
            'Rs': {'A': 0.0362, 'B': 1.0937, 'C': 25.724, 'D': 0.5},
            'Bob': {'A': 0.9759, 'B': 12e-5, 'C': 1.25, 'D': 0.5},
            'muod': {'A': 1.8e-5, 'B': 3.5, 'C': 1.8e-3, 'D': 0.43}
        }
        
        # Vazquez-Beggs关联式参数
        self.vazquez_beggs_coeffs = {
            'light': {'C1': 0.0362, 'C2': 1.0937, 'C3': 25.724},
            'heavy': {'C1': 0.0178, 'C2': 1.187, 'C3': 23.931}
        }
    
    def calculate_gas_oil_ratio(self, P: float, T: float, 
                              gamma_g: float = 0.7) -> float:
        """
        计算溶解气油比 Rs (Gas-Oil Ratio)
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K] 
        gamma_g : float
            天然气相对密度
            
        Returns:
        --------
        float
            溶解气油比 [Sm³/Sm³]
        """
        # 转换单位
        P_psia = P * 0.000145038  # Pa to psia
        T_F = (T - 273.15) * 9/5 + 32  # K to °F
        
        # Standing关联式
        coeffs = self.standing_coeffs['Rs']
        
        Rs = coeffs['A'] * gamma_g * (P_psia ** coeffs['B']) * \
             np.exp(coeffs['C'] * (self.api_gravity / (T_F + 460))**coeffs['D'])
             
        return Rs
    
    def calculate_oil_formation_volume_factor(self, P: float, T: float,
                                            Rs: float, gamma_g: float = 0.7) -> float:
        """
        计算原油体积系数 Bob
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K]
        Rs : float
            溶解气油比 [Sm³/Sm³]
        gamma_g : float
            天然气相对密度
            
        Returns:
        --------
        float
            原油体积系数 [Rm³/Sm³]
        """
        # 转换单位
        T_F = (T - 273.15) * 9/5 + 32  # K to °F
        
        # Standing关联式
        coeffs = self.standing_coeffs['Bob']
        
        Bob = coeffs['A'] + coeffs['B'] * (Rs * (gamma_g/self.specific_gravity)**coeffs['C'] + 
                                          coeffs['D'] * (T_F - 60))
        
        return Bob
    
    def calculate_dead_oil_viscosity(self, T: float) -> float:
        """
        计算脱气原油粘度
        
        Parameters:
        -----------
        T : float
            温度 [K]
            
        Returns:
        --------
        float
            脱气原油粘度 [Pa·s]
        """
        # 转换单位
        T_F = (T - 273.15) * 9/5 + 32  # K to °F
        
        # Beggs-Robinson关联式
        coeffs = self.standing_coeffs['muod']
        
        Z = 3.0324 - 0.02023 * self.api_gravity
        Y = 10**Z
        X = Y * (T_F**(-1.163))
        
        muod = 10**X - 1  # cP
        
        return muod * 1e-3  # 转换为Pa·s
    
    def calculate_saturated_oil_viscosity(self, muod: float, Rs: float) -> float:
        """
        计算饱和原油粘度
        
        Parameters:
        -----------
        muod : float
            脱气原油粘度 [Pa·s]
        Rs : float
            溶解气油比 [Sm³/Sm³]
            
        Returns:  
        --------
        float
            饱和原油粘度 [Pa·s]
        """
        # Chew-Connally关联式
        A = 10.715 * (Rs + 100)**(-0.515)
        B = 5.44 * (Rs + 150)**(-0.338)
        
        muob = A * (muod * 1000)**B  # cP
        
        return muob * 1e-3  # 转换为Pa·s
    
    def calculate_bubble_point_pressure(self, T: float, Rs: float,
                                      gamma_g: float = 0.7) -> float:
        """
        计算泡点压力
        
        Parameters:
        -----------
        T : float
            温度 [K]
        Rs : float
            溶解气油比 [Sm³/Sm³]
        gamma_g : float
            天然气相对密度
            
        Returns:
        --------
        float
            泡点压力 [Pa]
        """
        # 转换单位
        T_F = (T - 273.15) * 9/5 + 32  # K to °F
        
        # Standing关联式逆推
        coeffs = self.standing_coeffs['Rs']
        
        # 求解压力
        def pressure_equation(P_psia):
            Rs_calc = coeffs['A'] * gamma_g * (P_psia ** coeffs['B']) * \
                      np.exp(coeffs['C'] * (self.api_gravity / (T_F + 460))**coeffs['D'])
            return Rs_calc - Rs
        
        try:
            P_psia = fsolve(pressure_equation, 1000.0)[0]
            return P_psia / 0.000145038  # psia to Pa
        except:
            return 101325.0  # 默认大气压
    
    def calculate_pvt_properties(self, P: float, T: float) -> Dict[str, float]:
        """
        计算完整的PVT性质
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K]
            
        Returns:
        --------
        Dict[str, float]
            PVT性质字典
        """
        gamma_g = 0.7  # 默认天然气相对密度
        
        # 计算溶解气油比
        Rs = self.calculate_gas_oil_ratio(P, T, gamma_g)
        
        # 计算原油体积系数
        Bob = self.calculate_oil_formation_volume_factor(P, T, Rs, gamma_g)
        
        # 计算粘度
        muod = self.calculate_dead_oil_viscosity(T)
        muob = self.calculate_saturated_oil_viscosity(muod, Rs)
        
        # 计算泡点压力
        Pb = self.calculate_bubble_point_pressure(T, Rs, gamma_g)
        
        return {
            'gas_oil_ratio': Rs,
            'oil_formation_volume_factor': Bob,
            'dead_oil_viscosity': muod,
            'saturated_oil_viscosity': muob,
            'bubble_point_pressure': Pb,
            'specific_gravity': self.specific_gravity,
            'api_gravity': self.api_gravity
        }
    
    def calculate_fugacity_coefficient(self, component: str, x: np.ndarray,
                                     P: float, T: float, phase: str = 'liquid') -> float:
        """
        计算逸度系数 (简化实现)
        
        Parameters:
        -----------
        component : str
            组分名称
        x : np.ndarray
            摩尔分数数组
        P : float
            压力 [Pa]
        T : float
            温度 [K]
        phase : str
            相态 ('liquid' 或 'vapor')
            
        Returns:
        --------
        float
            逸度系数
        """
        if phase == 'liquid':
            # 液相使用活度系数模型
            return 1.0  # 简化处理
        else:
            # 气相使用状态方程
            return self._calculate_vapor_fugacity_coefficient(component, x, P, T)
    
    def _calculate_vapor_fugacity_coefficient(self, component: str, x: np.ndarray,
                                            P: float, T: float) -> float:
        """
        计算气相逸度系数
        """
        # 使用Peng-Robinson方程简化计算
        R = 8.314  # J/(mol·K)
        
        # 简化的PR方程逸度系数
        Z = P / (R * T)  # 简化压缩因子
        phi = np.exp(Z - 1 - np.log(Z))
        
        return phi
    
    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型信息
        
        Returns:
        --------
        Dict[str, any]
            模型信息字典
        """
        return {
            'name': self.name,
            'type': 'Petroleum Fluid Model',
            'compounds': self.compounds,
            'api_gravity': self.api_gravity,
            'specific_gravity': self.specific_gravity,
            'applicable_range': {
                'temperature': '273-423 K',
                'pressure': '0.1-20 MPa',
                'phases': ['liquid', 'vapor'],
                'systems': ['petroleum fluids', 'oil-gas mixtures']
            },
            'correlations': ['Standing', 'Beggs-Robinson', 'Chew-Connally'],
            'limitations': [
                '适用于常规石油流体',
                '不适用于凝析气',
                '温度范围有限'
            ]
        }


class WaterModel:
    """
    水模型类，用于黑油模型中的含水计算
    """
    
    @staticmethod
    def calculate_water_formation_volume_factor(P: float, T: float, 
                                              salinity: float = 0.0) -> float:
        """
        计算地层水体积系数
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K]
        salinity : float
            盐度 [mg/L]
            
        Returns:
        --------
        float
            地层水体积系数
        """
        # McCain关联式
        P_psia = P * 0.000145038
        T_F = (T - 273.15) * 9/5 + 32
        
        # 压力修正
        DVwp = -1.0001e-2 + 1.33391e-4 * T_F + 5.50654e-7 * T_F**2
        
        # 温度修正
        DVwT = -1.95301e-9 * P_psia * T_F - 1.72834e-13 * P_psia**2 * T_F - \
               3.58922e-7 * P_psia - 2.25341e-10 * P_psia**2
        
        # 盐度修正 (简化)
        salt_correction = 1.0 - salinity * 1e-6
        
        Bw = (1 + DVwp) * (1 + DVwT) * salt_correction
        
        return Bw
    
    @staticmethod
    def calculate_water_compressibility(P: float, T: float, 
                                      salinity: float = 0.0) -> float:
        """
        计算地层水压缩系数
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K]
        salinity : float
            盐度 [mg/L]
            
        Returns:
        --------
        float
            地层水压缩系数 [1/Pa]
        """
        # Dodson-Standing关联式
        P_psia = P * 0.000145038
        T_F = (T - 273.15) * 9/5 + 32
        
        Cw = (3.8546 - 0.000134 * P_psia) * 1e-6 + \
             (0.01052 + 4.77e-7 * P_psia) * 1e-6 * (T_F - 60) - \
             8.9e-13 * (T_F - 60)**2
        
        # 单位转换: 1/psia to 1/Pa
        return Cw / 0.000145038


# 使用示例
if __name__ == "__main__":
    # 创建黑油模型
    compounds = ['C7+', 'CH4', 'C2H6', 'C3H8', 'H2O']
    black_oil = BlackOilModel(compounds, api_gravity=32.0)
    
    # 计算PVT性质
    P = 10e6  # 10 MPa
    T = 350.0  # 350 K
    
    pvt_props = black_oil.calculate_pvt_properties(P, T)
    
    print("黑油模型PVT计算结果:")
    for prop, value in pvt_props.items():
        print(f"{prop}: {value:.6f}")
    
    # 获取模型信息
    model_info = black_oil.get_model_info()
    print(f"\n模型信息: {model_info['name']}")
    print(f"适用范围: {model_info['applicable_range']}") 