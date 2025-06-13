"""
UNIQUAC (Universal Quasi-Chemical) 活度系数模型

基于Abrams和Prausnitz (1975)提出的UNIQUAC方程实现活度系数计算。
该模型结合了局部组成概念和拟化学方法，适用于含各种极性组分的液体混合物。

参考文献:
- Abrams, D. S., & Prausnitz, J. M. (1975). AIChE Journal, 21(1), 116-128.
- DWSIM热力学库VB.NET原始实现

作者: OpenAspen项目组  
版本: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from ..base import PropertyPackageBase  
from ...core.compound import Compound
from ...core.phase import Phase

class UNIQUAC(PropertyPackageBase):
    """
    UNIQUAC活度系数模型实现
    
    该模型包含组合项(combinatorial)和残基项(residual)两部分：
    ln(γi) = ln(γi_C) + ln(γi_R)
    
    模型参数包括:
    - 二元交互参数τij, τji  
    - 分子体积参数ri
    - 分子表面积参数qi
    """
    
    def __init__(self, compounds: List[Compound], **kwargs):
        """
        初始化UNIQUAC模型
        
        参数:
            compounds: 化合物列表
            **kwargs: 其他参数
        """
        super().__init__(compounds, **kwargs)
        self.model_name = "UNIQUAC"
        self.model_type = "活度系数模型"
        
        # UNIQUAC模型参数
        self.tau_ij = {}  # 二元交互参数τij
        self.r_i = {}     # 分子体积参数
        self.q_i = {}     # 分子表面积参数
        self.binary_params = {}  # 二元参数数据库
        
        # 初始化参数
        self._initialize_parameters()
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def _initialize_parameters(self):
        """初始化UNIQUAC模型参数"""
        n = len(self.compounds)
        
        # 初始化τij矩阵
        for i in range(n):
            for j in range(n):
                key = f"{i}-{j}"
                if i == j:
                    self.tau_ij[key] = 1.0  # τii = 1
                else:
                    # 从数据库获取参数，如果没有则使用默认值
                    self.tau_ij[key] = self._get_binary_param(i, j, 'tau', 1.0)
        
        # 初始化分子参数ri和qi
        for i, compound in enumerate(self.compounds):
            # 从化合物属性或数据库获取UNIQUAC参数
            self.r_i[i] = self._get_molecular_param(compound, 'r', 1.0)
            self.q_i[i] = self._get_molecular_param(compound, 'q', 1.0)
    
    def _get_binary_param(self, i: int, j: int, param_type: str, default: float) -> float:
        """
        获取二元交互参数
        
        参数:
            i, j: 化合物索引
            param_type: 参数类型 ('tau')
            default: 默认值
            
        返回:
            参数值
        """
        # TODO: 实现从数据库读取参数的功能
        return default
    
    def _get_molecular_param(self, compound: Compound, param_type: str, default: float) -> float:
        """
        获取分子参数
        
        参数:
            compound: 化合物对象
            param_type: 参数类型 ('r' 或 'q')
            default: 默认值
            
        返回:
            分子参数值
        """
        # TODO: 实现从化合物属性或数据库读取参数的功能
        return default
    
    def calculate_activity_coefficients(self, x: np.ndarray, T: float, P: float = None) -> np.ndarray:
        """
        计算活度系数
        
        参数:
            x: 液相摩尔分数数组
            T: 温度 [K]
            P: 压力 [Pa] (可选)
            
        返回:
            活度系数数组
        """
        n = len(x)
        gamma = np.zeros(n)
        
        # 计算面积分数θi和体积分数φi
        theta, phi = self._calculate_fractions(x)
        
        # 计算组合项
        gamma_c = self._calculate_combinatorial_term(x, theta, phi)
        
        # 计算残基项
        gamma_r = self._calculate_residual_term(x, theta, T)
        
        # 总活度系数
        for i in range(n):
            gamma[i] = gamma_c[i] * gamma_r[i]
        
        return gamma
    
    def _calculate_fractions(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算面积分数和体积分数
        
        参数:
            x: 摩尔分数数组
            
        返回:
            (面积分数θ, 体积分数φ)
        """
        n = len(x)
        
        # 计算体积分数φi
        sum_xr = sum(x[i] * self.r_i[i] for i in range(n))
        phi = np.array([x[i] * self.r_i[i] / sum_xr if sum_xr > 0 else 0 for i in range(n)])
        
        # 计算面积分数θi
        sum_xq = sum(x[i] * self.q_i[i] for i in range(n))
        theta = np.array([x[i] * self.q_i[i] / sum_xq if sum_xq > 0 else 0 for i in range(n)])
        
        return theta, phi
    
    def _calculate_combinatorial_term(self, x: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        计算组合项
        
        参数:
            x: 摩尔分数数组
            theta: 面积分数数组
            phi: 体积分数数组
            
        返回:
            组合项活度系数数组
        """
        n = len(x)
        gamma_c = np.zeros(n)
        z = 10.0  # 配位数，通常取10
        
        for i in range(n):
            if x[i] > 0 and phi[i] > 0 and theta[i] > 0:
                # ln(γi_C) = ln(φi/xi) + (z/2)*qi*ln(θi/φi) + li - (φi/xi)*Σ(xj*lj)
                term1 = np.log(phi[i] / x[i])
                term2 = (z / 2.0) * self.q_i[i] * np.log(theta[i] / phi[i])
                
                # 计算li = (z/2)*(ri - qi) - (ri - 1)
                li = (z / 2.0) * (self.r_i[i] - self.q_i[i]) - (self.r_i[i] - 1.0)
                
                # 计算Σ(xj*lj)
                sum_xl = 0.0
                for j in range(n):
                    lj = (z / 2.0) * (self.r_i[j] - self.q_i[j]) - (self.r_i[j] - 1.0)
                    sum_xl += x[j] * lj
                
                term3 = li - (phi[i] / x[i]) * sum_xl
                
                ln_gamma_c = term1 + term2 + term3
                gamma_c[i] = np.exp(ln_gamma_c)
            else:
                gamma_c[i] = 1.0
        
        return gamma_c
    
    def _calculate_residual_term(self, x: np.ndarray, theta: np.ndarray, T: float) -> np.ndarray:
        """
        计算残基项
        
        参数:
            x: 摩尔分数数组
            theta: 面积分数数组
            T: 温度 [K]
            
        返回:
            残基项活度系数数组
        """
        n = len(x)
        gamma_r = np.zeros(n)
        
        # 计算τij矩阵
        tau_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                key = f"{i}-{j}"
                tau_matrix[i, j] = self.tau_ij[key]
        
        # 计算残基项
        for i in range(n):
            # ln(γi_R) = qi * [1 - ln(Σ(θj*τji)) - Σ(θj*τij/Σ(θk*τkj))]
            
            # 第一项：Σ(θj*τji)
            sum1 = sum(theta[j] * tau_matrix[j, i] for j in range(n))
            
            if sum1 > 0:
                term1 = np.log(sum1)
            else:
                term1 = 0.0
            
            # 第二项：Σ(θj*τij/Σ(θk*τkj))
            term2 = 0.0
            for j in range(n):
                sum2 = sum(theta[k] * tau_matrix[k, j] for k in range(n))
                if sum2 > 0:
                    term2 += theta[j] * tau_matrix[i, j] / sum2
            
            ln_gamma_r = self.q_i[i] * (1.0 - term1 - term2)
            gamma_r[i] = np.exp(ln_gamma_r)
        
        return gamma_r
    
    def calculate_excess_gibbs_energy(self, x: np.ndarray, T: float) -> float:
        """
        计算超额Gibbs自由能
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            
        返回:
            超额Gibbs自由能 [J/mol]
        """
        R = 8.314  # 气体常数 [J/(mol·K)]
        
        # 计算活度系数
        gamma = self.calculate_activity_coefficients(x, T)
        
        # 计算超额Gibbs自由能
        ge = 0.0
        for i, xi in enumerate(x):
            if xi > 0 and gamma[i] > 0:
                ge += xi * np.log(gamma[i])
        
        return R * T * ge
    
    def set_binary_parameters(self, comp1_idx: int, comp2_idx: int, 
                            tau12: float, tau21: float):
        """
        设置二元交互参数
        
        参数:
            comp1_idx: 化合物1的索引
            comp2_idx: 化合物2的索引
            tau12: τ12参数
            tau21: τ21参数
        """
        key12 = f"{comp1_idx}-{comp2_idx}"
        key21 = f"{comp2_idx}-{comp1_idx}"
        
        self.tau_ij[key12] = tau12
        self.tau_ij[key21] = tau21
        
        self.logger.info(f"设置UNIQUAC参数: {self.compounds[comp1_idx].name}-{self.compounds[comp2_idx].name}")
        self.logger.info(f"τ12={tau12:.4f}, τ21={tau21:.4f}")
    
    def set_molecular_parameters(self, comp_idx: int, r: float, q: float):
        """
        设置分子参数
        
        参数:
            comp_idx: 化合物索引
            r: 体积参数
            q: 表面积参数
        """
        self.r_i[comp_idx] = r
        self.q_i[comp_idx] = q
        
        self.logger.info(f"设置{self.compounds[comp_idx].name}的UNIQUAC分子参数: r={r:.4f}, q={q:.4f}")
    
    def get_binary_parameters(self, comp1_idx: int, comp2_idx: int) -> Dict[str, float]:
        """
        获取二元交互参数
        
        参数:
            comp1_idx: 化合物1的索引
            comp2_idx: 化合物2的索引
            
        返回:
            包含参数的字典
        """
        key12 = f"{comp1_idx}-{comp2_idx}"
        key21 = f"{comp2_idx}-{comp1_idx}"
        
        return {
            'tau12': self.tau_ij[key12],
            'tau21': self.tau_ij[key21]
        }
    
    def get_molecular_parameters(self, comp_idx: int) -> Dict[str, float]:
        """
        获取分子参数
        
        参数:
            comp_idx: 化合物索引
            
        返回:
            包含分子参数的字典
        """
        return {
            'r': self.r_i[comp_idx],
            'q': self.q_i[comp_idx]
        }
    
    def validate_parameters(self) -> bool:
        """
        验证模型参数的有效性
        
        返回:
            参数是否有效
        """
        # 检查分子参数
        for i in range(len(self.compounds)):
            if self.r_i[i] <= 0 or self.q_i[i] <= 0:
                self.logger.warning(f"化合物{i}的分子参数无效: r={self.r_i[i]}, q={self.q_i[i]}")
                return False
        
        # 检查τij参数
        for key, tau in self.tau_ij.items():
            if tau <= 0:
                self.logger.warning(f"二元参数τ {key} = {tau} 必须大于0")
                return False
        
        return True
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
            模型信息字典
        """
        return {
            'name': self.model_name,
            'type': self.model_type,
            'description': 'UNIQUAC活度系数模型，结合局部组成和拟化学方法',
            'parameters': {
                'binary_interactions': len([k for k in self.tau_ij.keys() if self.tau_ij[k] != 1.0]),
                'total_compounds': len(self.compounds)
            },
            'applicable_phases': ['Liquid'],
            'temperature_range': '250-600 K (建议范围)',
            'pressure_range': '0.1-50 bar (中低压系统)',
            'features': ['组合项', '残基项', '分子体积和表面积参数']
        } 