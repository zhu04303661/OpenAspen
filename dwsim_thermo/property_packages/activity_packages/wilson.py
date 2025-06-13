"""
Wilson 活度系数模型

基于Wilson (1964)提出的局部组成模型实现活度系数计算。
Wilson模型适用于完全互溶的液体混合物，特别是含醇类和水的体系。

参考文献:
- Wilson, G. M. (1964). J. Am. Chem. Soc., 86(2), 127-130.
- Prausnitz, J. M., et al. (1999). Molecular Thermodynamics of Fluid-Phase Equilibria (3rd ed.)
- DWSIM热力学库VB.NET原始实现

数学表达式:
$$\ln \gamma_i = 1 - \ln\left(\sum_j x_j \Lambda_{ij}\right) - \sum_k \frac{x_k \Lambda_{ki}}{\sum_j x_j \Lambda_{kj}}$$

其中：
$$\Lambda_{ij} = \frac{V_j^L}{V_i^L} \exp\left(-\frac{\lambda_{ij} - \lambda_{ii}}{RT}\right)$$

作者: OpenAspen项目组  
版本: 1.0.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from ..base import PropertyPackageBase
from ...core.compound import Compound
from ...core.phase import Phase

class Wilson(PropertyPackageBase):
    """
    Wilson活度系数模型实现
    
    Wilson模型基于局部组成概念，假设分子周围的局部组成与整体组成不同。
    该模型对强非理想性体系有很好的精度，但不能处理液液分层现象。
    
    特点:
    - 仅需要2个二元参数
    - 计算速度快
    - 热力学一致性好
    - 不能预测液液平衡
    """
    
    def __init__(self, compounds: List[Compound], **kwargs):
        """
        初始化Wilson模型
        
        参数:
            compounds: 化合物列表
            **kwargs: 其他参数
        """
        super().__init__(compounds, **kwargs)
        self.model_name = "Wilson"
        self.model_type = "活度系数模型"
        
        # Wilson模型参数
        self.lambda_ij = {}  # 二元交互参数λij [J/mol]
        self.v_i = {}        # 分子体积参数Vi [cm³/mol]
        self.binary_params = {}  # 二元参数数据库
        
        # 预计算的Λij矩阵 (温度相关)
        self._lambda_matrix = None
        self._last_temperature = None
        
        # 初始化参数
        self._initialize_parameters()
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def _initialize_parameters(self):
        """初始化Wilson模型参数"""
        n = len(self.compounds)
        
        # 初始化λij矩阵
        for i in range(n):
            for j in range(n):
                key = f"{i}-{j}"
                if i == j:
                    self.lambda_ij[key] = 0.0  # λii = 0
                else:
                    # 从数据库获取参数，如果没有则使用默认值
                    self.lambda_ij[key] = self._get_binary_param(i, j, 'lambda', 0.0)
        
        # 初始化分子体积参数
        for i, compound in enumerate(self.compounds):
            # 从化合物属性或数据库获取分子体积
            self.v_i[i] = self._get_molecular_volume(compound, 50.0)  # 默认50 cm³/mol
    
    def _get_binary_param(self, i: int, j: int, param_type: str, default: float) -> float:
        """
        获取二元交互参数
        
        参数:
            i, j: 化合物索引
            param_type: 参数类型 ('lambda')
            default: 默认值
            
        返回:
            参数值
        """
        # TODO: 实现从数据库读取参数的功能
        return default
    
    def _get_molecular_volume(self, compound: Compound, default: float) -> float:
        """
        获取分子体积参数
        
        参数:
            compound: 化合物对象
            default: 默认值
            
        返回:
            分子体积 [cm³/mol]
        """
        # TODO: 实现从化合物属性获取分子体积的功能
        # 可以使用关键体积或其他方法估算
        return default
    
    def _calculate_lambda_matrix(self, T: float) -> np.ndarray:
        """
        计算Wilson参数矩阵Λij
        
        参数:
            T: 温度 [K]
            
        返回:
            Λij矩阵
        """
        n = len(self.compounds)
        R = 8.314  # 气体常数 [J/(mol·K)]
        
        lambda_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    lambda_matrix[i, j] = 1.0
                else:
                    key_ij = f"{i}-{j}"
                    # Λij = (Vj/Vi) * exp[-(λij - λii)/(RT)]
                    # 由于λii = 0，所以简化为 Λij = (Vj/Vi) * exp(-λij/(RT))
                    volume_ratio = self.v_i[j] / self.v_i[i] if self.v_i[i] > 0 else 1.0
                    energy_term = np.exp(-self.lambda_ij[key_ij] / (R * T))
                    lambda_matrix[i, j] = volume_ratio * energy_term
        
        return lambda_matrix
    
    def calculate_activity_coefficients(self, x: np.ndarray, T: float, P: float = None) -> np.ndarray:
        """
        计算活度系数
        
        Wilson方程:
        ln(γi) = 1 - ln(∑j xj·Λij) - ∑k [xk·Λki / ∑j xj·Λkj]
        
        参数:
            x: 液相摩尔分数数组
            T: 温度 [K]
            P: 压力 [Pa] (可选)
            
        返回:
            活度系数数组
        """
        n = len(x)
        
        # 检查是否需要重新计算Λij矩阵
        if self._last_temperature != T or self._lambda_matrix is None:
            self._lambda_matrix = self._calculate_lambda_matrix(T)
            self._last_temperature = T
        
        gamma = np.zeros(n)
        
        # 计算活度系数
        for i in range(n):
            # 第一项: 1 - ln(∑j xj·Λij)
            sum1 = sum(x[j] * self._lambda_matrix[i, j] for j in range(n))
            term1 = 1.0 - np.log(sum1) if sum1 > 0 else 1.0
            
            # 第二项: -∑k [xk·Λki / ∑j xj·Λkj]
            term2 = 0.0
            for k in range(n):
                sum2 = sum(x[j] * self._lambda_matrix[k, j] for j in range(n))
                if sum2 > 0:
                    term2 += x[k] * self._lambda_matrix[k, i] / sum2
            
            # 计算ln(γi)
            ln_gamma = term1 - term2
            gamma[i] = np.exp(ln_gamma)
        
        return gamma
    
    def calculate_excess_gibbs_energy(self, x: np.ndarray, T: float) -> float:
        """
        计算超额Gibbs自由能
        
        Wilson方程:
        GE/(RT) = -∑i xi·ln(∑j xj·Λij)
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            
        返回:
            超额Gibbs自由能 [J/mol]
        """
        R = 8.314  # 气体常数 [J/(mol·K)]
        
        # 确保Λij矩阵是最新的
        if self._last_temperature != T or self._lambda_matrix is None:
            self._lambda_matrix = self._calculate_lambda_matrix(T)
            self._last_temperature = T
        
        n = len(x)
        ge_sum = 0.0
        
        for i in range(n):
            sum_term = sum(x[j] * self._lambda_matrix[i, j] for j in range(n))
            if sum_term > 0 and x[i] > 0:
                ge_sum += x[i] * np.log(sum_term)
        
        return -R * T * ge_sum
    
    def calculate_excess_enthalpy(self, x: np.ndarray, T: float) -> float:
        """
        计算超额焓
        
        HE = -T²·∂(GE/T)/∂T
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            
        返回:
            超额焓 [J/mol]
        """
        # 数值微分计算∂(GE/T)/∂T
        dT = 0.1  # 温度步长
        
        ge_1 = self.calculate_excess_gibbs_energy(x, T + dT)
        ge_2 = self.calculate_excess_gibbs_energy(x, T - dT)
        
        # ∂(GE/T)/∂T ≈ [GE(T+dT)/(T+dT) - GE(T-dT)/(T-dT)] / (2·dT)
        d_ge_over_t = (ge_1/(T + dT) - ge_2/(T - dT)) / (2 * dT)
        
        return -T * T * d_ge_over_t
    
    def set_binary_parameters(self, comp1_idx: int, comp2_idx: int, 
                            lambda12: float, lambda21: float):
        """
        设置二元交互参数
        
        参数:
            comp1_idx: 化合物1的索引
            comp2_idx: 化合物2的索引
            lambda12: λ12参数 [J/mol]
            lambda21: λ21参数 [J/mol]
        """
        key12 = f"{comp1_idx}-{comp2_idx}"
        key21 = f"{comp2_idx}-{comp1_idx}"
        
        self.lambda_ij[key12] = lambda12
        self.lambda_ij[key21] = lambda21
        
        # 清除缓存的矩阵，强制重新计算
        self._lambda_matrix = None
        self._last_temperature = None
        
        self.logger.info(f"设置Wilson参数: {self.compounds[comp1_idx].name}-{self.compounds[comp2_idx].name}")
        self.logger.info(f"λ12={lambda12:.2f} J/mol, λ21={lambda21:.2f} J/mol")
    
    def set_molecular_volumes(self, comp_idx: int, volume: float):
        """
        设置分子体积参数
        
        参数:
            comp_idx: 化合物索引
            volume: 分子体积 [cm³/mol]
        """
        self.v_i[comp_idx] = volume
        
        # 清除缓存的矩阵
        self._lambda_matrix = None
        self._last_temperature = None
        
        self.logger.info(f"设置{self.compounds[comp_idx].name}的分子体积: {volume:.2f} cm³/mol")
    
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
            'lambda12': self.lambda_ij[key12],
            'lambda21': self.lambda_ij[key21]
        }
    
    def get_molecular_volumes(self) -> Dict[int, float]:
        """
        获取所有分子体积参数
        
        返回:
            分子体积字典
        """
        return self.v_i.copy()
    
    def calculate_infinite_dilution_activity_coefficient(self, solute_idx: int, solvent_idx: int, T: float) -> float:
        """
        计算无限稀释活度系数
        
        参数:
            solute_idx: 溶质索引
            solvent_idx: 溶剂索引
            T: 温度 [K]
            
        返回:
            无限稀释活度系数
        """
        # 在无限稀释条件下：x_solute → 0, x_solvent → 1
        x = np.zeros(len(self.compounds))
        x[solvent_idx] = 1.0
        x[solute_idx] = 1e-10  # 接近零但避免数值问题
        
        gamma = self.calculate_activity_coefficients(x, T)
        
        return gamma[solute_idx]
    
    def validate_parameters(self) -> bool:
        """
        验证模型参数的有效性
        
        返回:
            参数是否有效
        """
        # 检查分子体积参数
        for i, volume in self.v_i.items():
            if volume <= 0:
                self.logger.warning(f"化合物{i}的分子体积无效: V={volume} cm³/mol")
                return False
        
        # 检查对称性（Wilson参数通常不对称，这里只检查是否为数值）
        for key, lambda_val in self.lambda_ij.items():
            if not isinstance(lambda_val, (int, float)):
                self.logger.warning(f"Wilson参数{key}不是数值: {lambda_val}")
                return False
        
        return True
    
    def estimate_parameters_from_vle_data(self, vle_data: List[Tuple], T: float) -> Dict[str, float]:
        """
        从汽液平衡数据估算Wilson参数
        
        参数:
            vle_data: VLE数据列表 [(x1, y1, P), ...]
            T: 温度 [K]
            
        返回:
            优化后的参数字典
        """
        # TODO: 实现参数回归算法
        # 这里返回示例参数
        return {
            'lambda12': 1000.0,
            'lambda21': -500.0
        }
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
            模型信息字典
        """
        return {
            'name': self.model_name,
            'type': self.model_type,
            'description': 'Wilson局部组成活度系数模型，适用于完全互溶的液体混合物',
            'parameters': {
                'binary_interactions': len([k for k in self.lambda_ij.keys() if self.lambda_ij[k] != 0]),
                'total_compounds': len(self.compounds)
            },
            'applicable_phases': ['Liquid'],
            'temperature_range': '250-500 K (建议范围)',
            'pressure_range': '0.1-50 bar (低中压系统)',
            'features': [
                '局部组成模型', '二元参数', '热力学一致性',
                '快速计算', '完全互溶体系'
            ],
            'limitations': [
                '不能预测液液平衡',
                '仅适用于完全互溶体系',
                '需要实验数据回归参数'
            ],
            'advantages': [
                '参数少(仅2个二元参数)',
                '计算速度快',
                '热力学严格一致',
                '对极性体系精度高'
            ]
        } 