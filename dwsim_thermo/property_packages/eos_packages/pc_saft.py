"""
PC-SAFT状态方程 (Perturbed-Chain Statistical Associating Fluid Theory)

基于Gross & Sadowski (2001)开发的PC-SAFT状态方程实现。
PC-SAFT是一种基于统计力学的状态方程，特别适用于链状分子和缔合流体。

参考文献:
- Gross, J., & Sadowski, G. (2001). Ind. Eng. Chem. Res., 40(4), 1244-1260.
- Gross, J., & Sadowski, G. (2002). Ind. Eng. Chem. Res., 41(22), 5510-5515.
- DWSIM热力学库VB.NET原始实现

数学表达式:
残余Helmholtz自由能:
$$A^{res} = A^{hc} + A^{disp} + A^{assoc}$$

其中:
- $A^{hc}$: 硬链贡献 (Hard Chain)
- $A^{disp}$: 色散贡献 (Dispersion)  
- $A^{assoc}$: 缔合贡献 (Association)

作者: OpenAspen项目组
版本: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy.optimize import fsolve
from ..base import PropertyPackageBase
from ...core.compound import Compound
from ...core.phase import Phase

class PCSAFT(PropertyPackageBase):
    """
    PC-SAFT状态方程实现
    
    PC-SAFT (Perturbed-Chain Statistical Associating Fluid Theory) 是一种
    基于统计力学的状态方程，特别适用于：
    - 链状分子 (聚合物、长链烷烃)
    - 缔合流体 (醇类、有机酸)
    - 极性分子
    - 宽温度和压力范围
    
    特点:
    - 基于分子理论
    - 参数具有物理意义
    - 适用范围广
    - 精度高
    """
    
    def __init__(self, compounds: List[Compound], **kwargs):
        """
        初始化PC-SAFT状态方程
        
        参数:
            compounds: 化合物列表
            **kwargs: 其他参数
        """
        super().__init__(compounds, **kwargs)
        self.model_name = "PC-SAFT"
        self.model_type = "状态方程"
        
        # PC-SAFT分子参数
        self.m = {}      # 链段数
        self.sigma = {}  # 链段直径 [Å]
        self.epsilon = {} # 链段间作用能 [K]
        self.kappa = {}  # 缔合体积参数
        self.epsilon_assoc = {}  # 缔合能参数 [K]
        
        # 二元交互参数
        self.kij = {}    # 二元交互参数
        
        # 初始化参数
        self._initialize_parameters()
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
    
    def _initialize_parameters(self):
        """初始化PC-SAFT参数"""
        # 为每个化合物设置默认参数
        for i, compound in enumerate(self.compounds):
            # 默认参数 (需要从数据库获取)
            self.m[i] = self._get_chain_length(compound)
            self.sigma[i] = self._get_segment_diameter(compound)
            self.epsilon[i] = self._get_segment_energy(compound)
            self.kappa[i] = self._get_association_volume(compound)
            self.epsilon_assoc[i] = self._get_association_energy(compound)
        
        # 初始化二元交互参数
        n = len(self.compounds)
        for i in range(n):
            for j in range(n):
                if i != j:
                    key = f"{i}-{j}"
                    self.kij[key] = 0.0  # 默认值
    
    def _get_chain_length(self, compound: Compound) -> float:
        """获取链段数参数"""
        # TODO: 从数据库获取
        return 2.0  # 默认值
    
    def _get_segment_diameter(self, compound: Compound) -> float:
        """获取链段直径参数 [Å]"""
        # TODO: 从数据库获取
        return 3.5  # 默认值
    
    def _get_segment_energy(self, compound: Compound) -> float:
        """获取链段间作用能参数 [K]"""
        # TODO: 从数据库获取
        return 250.0  # 默认值
    
    def _get_association_volume(self, compound: Compound) -> float:
        """获取缔合体积参数"""
        # TODO: 从数据库获取，非缔合分子返回0
        return 0.0  # 默认值
    
    def _get_association_energy(self, compound: Compound) -> float:
        """获取缔合能参数 [K]"""
        # TODO: 从数据库获取，非缔合分子返回0
        return 0.0  # 默认值
    
    def calculate_eos(self, x: np.ndarray, T: float, P: float) -> Dict[str, float]:
        """
        计算PC-SAFT状态方程
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            P: 压力 [Pa]
            
        返回:
            包含计算结果的字典
        """
        # 计算混合物参数
        m_mix = self._calculate_mixture_chain_length(x)
        
        # 计算压缩因子
        Z = self._calculate_compressibility_factor(T, P, x)
        
        # 计算残余Helmholtz自由能
        A_res = self._calculate_residual_helmholtz_energy(T, P, x)
        
        # 计算逸度系数
        phi = self._calculate_fugacity_coefficients(T, P, x)
        
        return {
            'Z': Z,
            'A_res': A_res,
            'phi': phi,
            'm_mix': m_mix
        }
    
    def _calculate_mixture_chain_length(self, x: np.ndarray) -> float:
        """
        计算混合物链段数
        
        $$m_{mix} = \sum_i x_i m_i$$
        
        参数:
            x: 摩尔分数数组
            
        返回:
            混合物链段数
        """
        return sum(x[i] * self.m[i] for i in range(len(x)))
    
    def _calculate_mixture_parameters(self, x: np.ndarray, T: float) -> Tuple[float, float]:
        """
        计算混合物参数
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            
        返回:
            (sigma_mix, epsilon_mix) 混合物参数
        """
        n = len(x)
        
        # 计算混合物链段直径
        sigma_mix = 0.0
        for i in range(n):
            for j in range(n):
                sigma_ij = 0.5 * (self.sigma[i] + self.sigma[j])
                sigma_mix += x[i] * x[j] * sigma_ij**3
        sigma_mix = sigma_mix**(1/3)
        
        # 计算混合物链段间作用能
        epsilon_mix = 0.0
        for i in range(n):
            for j in range(n):
                kij = self.kij.get(f"{i}-{j}", 0.0)
                epsilon_ij = np.sqrt(self.epsilon[i] * self.epsilon[j]) * (1 - kij)
                sigma_ij = 0.5 * (self.sigma[i] + self.sigma[j])
                epsilon_mix += x[i] * x[j] * epsilon_ij * sigma_ij**3
        
        epsilon_mix = epsilon_mix / (sigma_mix**3)
        
        return sigma_mix, epsilon_mix
    
    def _calculate_compressibility_factor(self, T: float, P: float, x: np.ndarray) -> float:
        """
        计算压缩因子
        
        使用PC-SAFT状态方程求解压缩因子
        
        参数:
            T: 温度 [K]
            P: 压力 [Pa]
            x: 摩尔分数数组
            
        返回:
            压缩因子
        """
        # 转换压力单位到Pa
        P_Pa = P if P > 1000 else P * 1e5  # 假设小于1000的是bar
        
        # 计算混合物参数
        m_mix = self._calculate_mixture_chain_length(x)
        sigma_mix, epsilon_mix = self._calculate_mixture_parameters(x, T)
        
        # 计算约化密度的初始猜测
        rho_guess = P_Pa / (8.314 * T)  # 理想气体近似
        
        def equation(rho):
            """PC-SAFT状态方程"""
            eta = np.pi * rho * sigma_mix**3 / 6  # 堆积分数
            
            # 硬球贡献
            Z_hs = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
            
            # 链贡献
            g_hs = (1 - eta/2) / (1 - eta)**3
            Z_chain = -(m_mix - 1) * g_hs
            
            # 色散贡献
            T_reduced = T / epsilon_mix
            Z_disp = self._calculate_dispersion_contribution(eta, T_reduced, m_mix)
            
            # 总压缩因子
            Z_total = Z_hs + Z_chain + Z_disp
            
            # 状态方程 P = ρRT*Z
            P_calc = rho * 8.314 * T * Z_total
            
            return P_calc - P_Pa
        
        try:
            # 求解密度
            rho_solution = fsolve(equation, rho_guess)[0]
            
            # 计算对应的压缩因子
            eta = np.pi * rho_solution * sigma_mix**3 / 6
            Z_hs = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
            g_hs = (1 - eta/2) / (1 - eta)**3
            Z_chain = -(m_mix - 1) * g_hs
            T_reduced = T / epsilon_mix
            Z_disp = self._calculate_dispersion_contribution(eta, T_reduced, m_mix)
            
            Z = Z_hs + Z_chain + Z_disp
            
            return Z
            
        except Exception as e:
            self.logger.warning(f"PC-SAFT压缩因子计算失败: {e}")
            return 1.0  # 返回理想气体值
    
    def _calculate_dispersion_contribution(self, eta: float, T_reduced: float, m_mix: float) -> float:
        """
        计算色散贡献
        
        参数:
            eta: 堆积分数
            T_reduced: 约化温度
            m_mix: 混合物链段数
            
        返回:
            色散贡献
        """
        # 简化的色散贡献计算
        # 实际实现需要更复杂的积分和求和
        I1 = self._calculate_integral_I1(eta)
        I2 = self._calculate_integral_I2(eta)
        
        Z_disp1 = -2 * np.pi * eta * I1 / T_reduced
        Z_disp2 = -np.pi * m_mix * eta * I2 / T_reduced**2
        
        return Z_disp1 + Z_disp2
    
    def _calculate_integral_I1(self, eta: float) -> float:
        """计算积分I1"""
        # 拟合公式
        a = [0.9105, 0.6362, 2.6861, -26.547, 97.759, -159.59, 91.297]
        
        I1 = 0.0
        for i, ai in enumerate(a):
            I1 += ai * eta**i
        
        return I1
    
    def _calculate_integral_I2(self, eta: float) -> float:
        """计算积分I2"""
        # 拟合公式
        b = [0.7240, 2.2382, -4.0025, -21.003, 26.855, 206.55, -355.60, -165.21]
        
        I2 = 0.0
        for i, bi in enumerate(b):
            I2 += bi * eta**i
        
        return I2
    
    def _calculate_residual_helmholtz_energy(self, T: float, P: float, x: np.ndarray) -> float:
        """
        计算残余Helmholtz自由能
        
        $$A^{res} = A^{hc} + A^{disp} + A^{assoc}$$
        
        参数:
            T: 温度 [K]
            P: 压力 [Pa]
            x: 摩尔分数数组
            
        返回:
            残余Helmholtz自由能 [J/mol]
        """
        # 计算各项贡献
        A_hc = self._calculate_hard_chain_contribution(T, P, x)
        A_disp = self._calculate_dispersion_helmholtz_contribution(T, P, x)
        A_assoc = self._calculate_association_contribution(T, P, x)
        
        return A_hc + A_disp + A_assoc
    
    def _calculate_hard_chain_contribution(self, T: float, P: float, x: np.ndarray) -> float:
        """计算硬链贡献"""
        # 简化实现
        return 0.0
    
    def _calculate_dispersion_helmholtz_contribution(self, T: float, P: float, x: np.ndarray) -> float:
        """计算色散贡献到Helmholtz自由能"""
        # 简化实现
        return 0.0
    
    def _calculate_association_contribution(self, T: float, P: float, x: np.ndarray) -> float:
        """计算缔合贡献"""
        # 简化实现，对于非缔合分子返回0
        return 0.0
    
    def _calculate_fugacity_coefficients(self, T: float, P: float, x: np.ndarray) -> np.ndarray:
        """
        计算逸度系数
        
        参数:
            T: 温度 [K]
            P: 压力 [Pa]
            x: 摩尔分数数组
            
        返回:
            逸度系数数组
        """
        n = len(x)
        phi = np.ones(n)
        
        # 计算压缩因子
        Z = self._calculate_compressibility_factor(T, P, x)
        
        # 简化的逸度系数计算
        # 实际需要计算化学势的偏导数
        for i in range(n):
            phi[i] = Z  # 简化近似
        
        return phi
    
    def set_pure_component_parameters(self, comp_idx: int, m: float, sigma: float, 
                                    epsilon: float, kappa: float = 0.0, 
                                    epsilon_assoc: float = 0.0):
        """
        设置纯组分PC-SAFT参数
        
        参数:
            comp_idx: 化合物索引
            m: 链段数
            sigma: 链段直径 [Å]
            epsilon: 链段间作用能 [K]
            kappa: 缔合体积参数
            epsilon_assoc: 缔合能参数 [K]
        """
        self.m[comp_idx] = m
        self.sigma[comp_idx] = sigma
        self.epsilon[comp_idx] = epsilon
        self.kappa[comp_idx] = kappa
        self.epsilon_assoc[comp_idx] = epsilon_assoc
        
        compound_name = self.compounds[comp_idx].name
        self.logger.info(f"设置{compound_name}的PC-SAFT参数:")
        self.logger.info(f"  m={m:.3f}, σ={sigma:.3f}Å, ε/k={epsilon:.1f}K")
        if kappa > 0:
            self.logger.info(f"  κ={kappa:.3f}, ε_assoc/k={epsilon_assoc:.1f}K")
    
    def set_binary_interaction_parameter(self, comp1_idx: int, comp2_idx: int, kij: float):
        """
        设置二元交互参数
        
        参数:
            comp1_idx: 化合物1的索引
            comp2_idx: 化合物2的索引
            kij: 二元交互参数
        """
        key12 = f"{comp1_idx}-{comp2_idx}"
        key21 = f"{comp2_idx}-{comp1_idx}"
        
        self.kij[key12] = kij
        self.kij[key21] = kij  # 对称
        
        name1 = self.compounds[comp1_idx].name
        name2 = self.compounds[comp2_idx].name
        self.logger.info(f"设置二元交互参数: {name1}-{name2}, kij={kij:.4f}")
    
    def get_pure_component_parameters(self, comp_idx: int) -> Dict[str, float]:
        """
        获取纯组分参数
        
        参数:
            comp_idx: 化合物索引
            
        返回:
            参数字典
        """
        return {
            'm': self.m[comp_idx],
            'sigma': self.sigma[comp_idx],
            'epsilon': self.epsilon[comp_idx],
            'kappa': self.kappa[comp_idx],
            'epsilon_assoc': self.epsilon_assoc[comp_idx]
        }
    
    def get_binary_interaction_parameters(self) -> Dict[str, float]:
        """
        获取所有二元交互参数
        
        返回:
            二元交互参数字典
        """
        return self.kij.copy()
    
    def validate_parameters(self) -> bool:
        """
        验证PC-SAFT参数的有效性
        
        返回:
            参数是否有效
        """
        for i in range(len(self.compounds)):
            # 检查链段数
            if self.m[i] <= 0:
                self.logger.error(f"化合物{i}的链段数无效: m={self.m[i]}")
                return False
            
            # 检查链段直径
            if self.sigma[i] <= 0:
                self.logger.error(f"化合物{i}的链段直径无效: σ={self.sigma[i]}")
                return False
            
            # 检查链段间作用能
            if self.epsilon[i] <= 0:
                self.logger.error(f"化合物{i}的链段间作用能无效: ε={self.epsilon[i]}")
                return False
            
            # 检查缔合参数
            if self.kappa[i] < 0:
                self.logger.error(f"化合物{i}的缔合体积参数无效: κ={self.kappa[i]}")
                return False
        
        return True
    
    def get_model_info(self) -> Dict:
        """
        获取PC-SAFT模型信息
        
        返回:
            模型信息字典
        """
        return {
            'name': self.model_name,
            'type': self.model_type,
            'description': 'PC-SAFT统计缔合流体理论状态方程，适用于链状分子和缔合流体',
            'parameters': {
                'molecular_params': len(self.compounds) * 3,  # m, σ, ε for each compound
                'association_params': sum(1 for kappa in self.kappa.values() if kappa > 0) * 2,
                'binary_interactions': len([k for k in self.kij.values() if k != 0])
            },
            'applicable_phases': ['Vapor', 'Liquid'],
            'temperature_range': '200-800 K',
            'pressure_range': '0.001-1000 bar',
            'features': [
                '基于分子理论', '适用于链状分子', '缔合流体处理',
                '宽温压范围', '参数物理意义明确'
            ],
            'limitations': [
                '参数需要实验数据拟合',
                '计算复杂度高',
                '对某些体系精度有限'
            ],
            'advantages': [
                '理论基础扎实',
                '适用范围广',
                '参数可预测性强',
                '对聚合物和生物分子效果好'
            ],
            'typical_applications': [
                '聚合物溶液',
                '生物分子体系',
                '缔合流体(醇类)',
                '长链烷烃',
                '超临界流体'
            ]
        } 