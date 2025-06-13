"""
NRTL (Non-Random Two Liquid) 活度系数模型

基于Renon和Prausnitz (1968)提出的NRTL方程实现活度系数计算。
该模型适用于强非理想性液体混合物，特别是含极性组分的体系。

参考文献:
- Renon, H., & Prausnitz, J. M. (1968). AIChE Journal, 14(1), 135-144.
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

class NRTL(PropertyPackageBase):
    """
    NRTL活度系数模型实现
    
    该模型使用局部组成概念处理液体混合物的非理想性行为。
    模型参数包括二元交互参数τij, τji和非随机参数αij。
    """
    
    def __init__(self, compounds: List[Compound], **kwargs):
        """
        初始化NRTL模型
        
        参数:
            compounds: 化合物列表
            **kwargs: 其他参数
        """
        super().__init__(compounds, **kwargs)
        self.model_name = "NRTL"
        self.model_type = "活度系数模型"
        
        # NRTL模型参数
        self.tau_ij = {}  # 二元交互参数τij
        self.alpha_ij = {}  # 非随机参数αij
        self.binary_params = {}  # 二元参数数据库
        
        # 默认参数值
        self.default_alpha = 0.3  # 默认非随机参数
        
        # 初始化参数
        self._initialize_parameters()
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def _initialize_parameters(self):
        """初始化NRTL模型参数"""
        n = len(self.compounds)
        
        # 初始化τij和αij矩阵
        for i in range(n):
            for j in range(n):
                key = f"{i}-{j}"
                if i == j:
                    self.tau_ij[key] = 0.0
                    self.alpha_ij[key] = 0.0
                else:
                    # 从数据库获取参数，如果没有则使用默认值
                    self.tau_ij[key] = self._get_binary_param(i, j, 'tau', 0.0)
                    self.alpha_ij[key] = self._get_binary_param(i, j, 'alpha', self.default_alpha)
    
    def _get_binary_param(self, i: int, j: int, param_type: str, default: float) -> float:
        """
        获取二元交互参数
        
        参数:
            i, j: 化合物索引
            param_type: 参数类型 ('tau' 或 'alpha')
            default: 默认值
            
        返回:
            参数值
        """
        # TODO: 实现从数据库读取参数的功能
        # 目前返回默认值
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
        
        # 计算τij和Gij
        tau_matrix = np.zeros((n, n))
        G_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                key = f"{i}-{j}"
                tau_matrix[i, j] = self.tau_ij[key]
                if i != j:
                    G_matrix[i, j] = np.exp(-self.alpha_ij[key] * tau_matrix[i, j])
                else:
                    G_matrix[i, j] = 1.0
        
        # 计算活度系数
        for i in range(n):
            # 第一项
            sum1 = 0.0
            sum2 = 0.0
            for j in range(n):
                sum1 += x[j] * tau_matrix[j, i] * G_matrix[j, i]
                sum2 += x[j] * G_matrix[j, i]
            
            term1 = sum1 / sum2 if sum2 > 0 else 0.0
            
            # 第二项
            term2 = 0.0
            for j in range(n):
                sum3 = 0.0
                sum4 = 0.0
                for k in range(n):
                    sum3 += x[k] * G_matrix[k, j]
                    sum4 += x[k] * tau_matrix[k, j] * G_matrix[k, j]
                
                if sum3 > 0:
                    term2 += (x[j] * G_matrix[i, j] / sum3) * \
                            (tau_matrix[i, j] - sum4 / sum3)
            
            # 计算ln(γi)
            ln_gamma = term1 + term2
            gamma[i] = np.exp(ln_gamma)
        
        return gamma
    
    def calculate_excess_gibbs_energy(self, x: np.ndarray, T: float) -> float:
        """
        计算超额Gibbs自由能
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            
        返回:
            超额Gibbs自由能 [J/mol]
        """
        n = len(x)
        R = 8.314  # 气体常数 [J/(mol·K)]
        
        # 计算τij和Gij
        tau_matrix = np.zeros((n, n))
        G_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                key = f"{i}-{j}"
                tau_matrix[i, j] = self.tau_ij[key]
                if i != j:
                    G_matrix[i, j] = np.exp(-self.alpha_ij[key] * tau_matrix[i, j])
                else:
                    G_matrix[i, j] = 1.0
        
        # 计算超额Gibbs自由能
        ge_sum = 0.0
        for i in range(n):
            sum_term = 0.0
            for j in range(n):
                sum_term += x[j] * G_matrix[j, i]
            
            if sum_term > 0:
                numerator = 0.0
                for j in range(n):
                    numerator += x[j] * tau_matrix[j, i] * G_matrix[j, i]
                ge_sum += x[i] * numerator / sum_term
        
        return R * T * ge_sum
    
    def calculate_excess_enthalpy(self, x: np.ndarray, T: float) -> float:
        """
        计算超额焓
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            
        返回:
            超额焓 [J/mol]
        """
        # TODO: 实现超额焓计算
        # 需要温度相关的NRTL参数
        return 0.0
    
    def set_binary_parameters(self, comp1_idx: int, comp2_idx: int, 
                            tau12: float, tau21: float, alpha12: float):
        """
        设置二元交互参数
        
        参数:
            comp1_idx: 化合物1的索引
            comp2_idx: 化合物2的索引
            tau12: τ12参数
            tau21: τ21参数
            alpha12: α12参数
        """
        key12 = f"{comp1_idx}-{comp2_idx}"
        key21 = f"{comp2_idx}-{comp1_idx}"
        
        self.tau_ij[key12] = tau12
        self.tau_ij[key21] = tau21
        self.alpha_ij[key12] = alpha12
        self.alpha_ij[key21] = alpha12  # αij = αji
        
        self.logger.info(f"设置NRTL参数: {self.compounds[comp1_idx].name}-{self.compounds[comp2_idx].name}")
        self.logger.info(f"τ12={tau12:.4f}, τ21={tau21:.4f}, α12={alpha12:.4f}")
    
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
            'tau21': self.tau_ij[key21],
            'alpha12': self.alpha_ij[key12]
        }
    
    def validate_parameters(self) -> bool:
        """
        验证模型参数的有效性
        
        返回:
            参数是否有效
        """
        for key, alpha in self.alpha_ij.items():
            if alpha < 0 or alpha > 1:
                self.logger.warning(f"非随机参数α {key} = {alpha} 超出有效范围[0,1]")
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
            'description': 'NRTL活度系数模型，适用于强非理想性液体混合物',
            'parameters': {
                'binary_interactions': len([k for k in self.tau_ij.keys() if self.tau_ij[k] != 0]),
                'total_compounds': len(self.compounds)
            },
            'applicable_phases': ['Liquid'],
            'temperature_range': '273-500 K (建议范围)',
            'pressure_range': '0.1-10 bar (低压系统)'
        } 