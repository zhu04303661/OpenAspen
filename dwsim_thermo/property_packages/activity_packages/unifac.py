"""
UNIFAC (Universal Functional-group Activity Coefficient) 基团贡献活度系数模型

基于Fredenslund等(1975)提出的UNIFAC方程实现活度系数计算。
该模型使用基团贡献法，能够预测没有实验数据的混合物的活度系数。

参考文献:
- Fredenslund, A., Jones, R. L., & Prausnitz, J. M. (1975). AIChE Journal, 21(6), 1086-1099.
- Hansen, H. K., et al. (1991). Ind. Eng. Chem. Res., 30(10), 2352-2355.
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

class UNIFAC(PropertyPackageBase):
    """
    UNIFAC基团贡献活度系数模型实现
    
    该模型包含组合项和残基项两部分：
    ln(γi) = ln(γi_C) + ln(γi_R)
    
    其中组合项与UNIQUAC相同，残基项基于基团贡献计算。
    """
    
    def __init__(self, compounds: List[Compound], **kwargs):
        """
        初始化UNIFAC模型
        
        参数:
            compounds: 化合物列表
            **kwargs: 其他参数
        """
        super().__init__(compounds, **kwargs)
        self.model_name = "UNIFAC"
        self.model_type = "活度系数模型"
        
        # UNIFAC模型参数
        self.group_assignments = {}  # 化合物的基团分配
        self.group_volumes = {}      # 基团体积参数Rk
        self.group_surfaces = {}     # 基团表面积参数Qk
        self.group_interactions = {} # 基团间交互参数amn
        
        # 分子参数(从基团计算得出)
        self.r_i = {}  # 分子体积参数
        self.q_i = {}  # 分子表面积参数
        
        # 初始化UNIFAC参数
        self._initialize_group_parameters()
        self._initialize_molecular_parameters()
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def _initialize_group_parameters(self):
        """初始化UNIFAC基团参数"""
        
        # 主基团参数 (简化版本，实际需要完整的UNIFAC参数表)
        self.main_groups = {
            1: {'name': 'CH2', 'R': 0.6744, 'Q': 0.540},      # 烷基
            2: {'name': 'C=C', 'R': 1.2680, 'Q': 0.867},      # 烯基
            5: {'name': 'ACH', 'R': 0.7081, 'Q': 0.400},      # 芳香基
            9: {'name': 'OH', 'R': 1.0000, 'Q': 1.200},       # 羟基
            18: {'name': 'H2O', 'R': 0.9200, 'Q': 1.400},     # 水
        }
        
        # 子基团参数
        self.sub_groups = {
            1: {'main': 1, 'R': 0.9011, 'Q': 0.848},   # CH3
            2: {'main': 1, 'R': 0.6744, 'Q': 0.540},   # CH2
            3: {'main': 1, 'R': 0.4469, 'Q': 0.228},   # CH
            4: {'main': 1, 'R': 0.2195, 'Q': 0.000},   # C
            # 更多子基团...
        }
        
        # 基团间交互参数amn (K)
        # 简化版本，实际需要完整的参数表
        self.group_interactions = {
            (1, 9): 986.5,    # CH2-OH
            (9, 1): 156.4,    # OH-CH2
            (1, 18): 1318.0,  # CH2-H2O
            (18, 1): 300.0,   # H2O-CH2
            (9, 18): -229.1,  # OH-H2O
            (18, 9): -14.09,  # H2O-OH
            # 更多交互参数...
        }
        
    def _initialize_molecular_parameters(self):
        """根据基团分配计算分子参数"""
        
        for i, compound in enumerate(self.compounds):
            # 获取化合物的基团分配
            groups = self._get_compound_groups(compound)
            
            # 计算ri和qi
            r_i = 0.0
            q_i = 0.0
            
            for group_id, count in groups.items():
                if group_id in self.sub_groups:
                    r_i += count * self.sub_groups[group_id]['R']
                    q_i += count * self.sub_groups[group_id]['Q']
            
            self.r_i[i] = r_i
            self.q_i[i] = q_i
            self.group_assignments[i] = groups
    
    def _get_compound_groups(self, compound: Compound) -> Dict[int, int]:
        """
        获取化合物的基团分配
        
        参数:
            compound: 化合物对象
            
        返回:
            基团分配字典 {基团ID: 数量}
        """
        # TODO: 实现从化合物结构自动识别基团的功能
        # 目前使用简化的基团分配
        
        name = compound.name.lower()
        if 'water' in name or 'h2o' in name:
            return {18: 1}  # 水分子
        elif 'ethanol' in name:
            return {1: 1, 2: 1, 9: 1}  # CH3-CH2-OH
        elif 'methanol' in name:
            return {1: 1, 9: 1}  # CH3-OH
        elif 'benzene' in name:
            return {5: 6}  # 6个芳香CH
        else:
            # 默认假设为简单烷烃
            return {1: 2, 2: 1}  # CH3-CH2-CH3 (丙烷)
    
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
        
        # 计算组合项 (与UNIQUAC相同)
        gamma_c = self._calculate_combinatorial_term(x, theta, phi)
        
        # 计算残基项 (基于基团贡献)
        gamma_r = self._calculate_residual_term(x, T)
        
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
        计算组合项 (与UNIQUAC相同)
        
        参数:
            x: 摩尔分数数组
            theta: 面积分数数组
            phi: 体积分数数组
            
        返回:
            组合项活度系数数组
        """
        n = len(x)
        gamma_c = np.zeros(n)
        z = 10.0  # 配位数
        
        for i in range(n):
            if x[i] > 0 and phi[i] > 0 and theta[i] > 0:
                term1 = np.log(phi[i] / x[i])
                term2 = (z / 2.0) * self.q_i[i] * np.log(theta[i] / phi[i])
                
                li = (z / 2.0) * (self.r_i[i] - self.q_i[i]) - (self.r_i[i] - 1.0)
                
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
    
    def _calculate_residual_term(self, x: np.ndarray, T: float) -> np.ndarray:
        """
        计算残基项 (基于基团贡献)
        
        参数:
            x: 摩尔分数数组
            T: 温度 [K]
            
        返回:
            残基项活度系数数组
        """
        n = len(x)
        gamma_r = np.zeros(n)
        
        # 计算混合物中基团的摩尔分数
        group_x = self._calculate_group_fractions(x)
        
        # 计算基团活度系数
        group_gamma = self._calculate_group_activity_coefficients(group_x, T)
        
        # 计算每个组分的残基项
        for i in range(n):
            ln_gamma_r = 0.0
            
            # 对化合物i中的每个基团k
            for group_k, count_k in self.group_assignments[i].items():
                if group_k in group_gamma:
                    # 纯组分i中基团k的活度系数
                    gamma_k_pure = self._calculate_pure_group_activity_coefficient(group_k, i, T)
                    
                    # 残基项贡献
                    if gamma_k_pure > 0 and group_gamma[group_k] > 0:
                        ln_gamma_r += count_k * (np.log(group_gamma[group_k]) - np.log(gamma_k_pure))
            
            gamma_r[i] = np.exp(ln_gamma_r)
        
        return gamma_r
    
    def _calculate_group_fractions(self, x: np.ndarray) -> Dict[int, float]:
        """
        计算混合物中基团的摩尔分数
        
        参数:
            x: 化合物摩尔分数数组
            
        返回:
            基团摩尔分数字典
        """
        group_counts = {}
        total_groups = 0.0
        
        # 统计各基团的总数
        for i, xi in enumerate(x):
            for group_k, count_k in self.group_assignments[i].items():
                if group_k not in group_counts:
                    group_counts[group_k] = 0.0
                group_counts[group_k] += xi * count_k
                total_groups += xi * count_k
        
        # 计算基团摩尔分数
        group_x = {}
        for group_k, count in group_counts.items():
            group_x[group_k] = count / total_groups if total_groups > 0 else 0.0
        
        return group_x
    
    def _calculate_group_activity_coefficients(self, group_x: Dict[int, float], T: float) -> Dict[int, float]:
        """
        计算基团活度系数
        
        参数:
            group_x: 基团摩尔分数
            T: 温度 [K]
            
        返回:
            基团活度系数字典
        """
        group_gamma = {}
        
        for group_k in group_x.keys():
            if group_k not in self.sub_groups:
                continue
                
            Qk = self.sub_groups[group_k]['Q']
            
            # 计算基团面积分数θk
            sum_xQ = sum(group_x[j] * self.sub_groups[j]['Q'] for j in group_x.keys() if j in self.sub_groups)
            theta_k = group_x[group_k] * Qk / sum_xQ if sum_xQ > 0 else 0.0
            
            # 计算基团活度系数
            # 第一项：ln(sum(θj * ψji))
            sum1 = 0.0
            for j in group_x.keys():
                if j in self.sub_groups:
                    sum_xQ_j = sum(group_x[k] * self.sub_groups[k]['Q'] for k in group_x.keys() if k in self.sub_groups)
                    theta_j = group_x[j] * self.sub_groups[j]['Q'] / sum_xQ_j if sum_xQ_j > 0 else 0.0
                    sum1 += theta_j * self._calculate_psi(j, group_k, T)
            
            # 第二项：sum(θj * ψkj / sum(θm * ψmj))
            sum2 = 0.0
            for j in group_x.keys():
                if j in self.sub_groups:
                    sum_xQ_j = sum(group_x[k] * self.sub_groups[k]['Q'] for k in group_x.keys() if k in self.sub_groups)
                    theta_j = group_x[j] * self.sub_groups[j]['Q'] / sum_xQ_j if sum_xQ_j > 0 else 0.0
                    
                    sum_denominator = 0.0
                    for m in group_x.keys():
                        if m in self.sub_groups:
                            sum_xQ_m = sum(group_x[k] * self.sub_groups[k]['Q'] for k in group_x.keys() if k in self.sub_groups)
                            theta_m = group_x[m] * self.sub_groups[m]['Q'] / sum_xQ_m if sum_xQ_m > 0 else 0.0
                            sum_denominator += theta_m * self._calculate_psi(m, j, T)
                    
                    if sum_denominator > 0:
                        sum2 += theta_j * self._calculate_psi(group_k, j, T) / sum_denominator
            
            ln_gamma_k = Qk * (1.0 - np.log(sum1) - sum2)
            
            group_gamma[group_k] = np.exp(ln_gamma_k)
        
        return group_gamma
    
    def _calculate_pure_group_activity_coefficient(self, group_k: int, compound_i: int, T: float) -> float:
        """
        计算纯组分中基团的活度系数
        
        参数:
            group_k: 基团ID
            compound_i: 化合物索引
            T: 温度 [K]
            
        返回:
            纯组分中基团的活度系数
        """
        # 在纯组分i中，只有该组分的基团存在
        groups_i = self.group_assignments[compound_i]
        
        # 计算纯组分中基团摩尔分数
        total_groups = sum(groups_i.values())
        group_x_pure = {j: count / total_groups for j, count in groups_i.items()}
        
        # 计算纯组分中的基团活度系数
        group_gamma_pure = self._calculate_group_activity_coefficients(group_x_pure, T)
        
        return group_gamma_pure.get(group_k, 1.0)
    
    def _calculate_psi(self, group_m: int, group_n: int, T: float) -> float:
        """
        计算基团间交互参数ψmn
        
        参数:
            group_m, group_n: 基团索引
            T: 温度 [K]
            
        返回:
            交互参数ψmn
        """
        if group_m == group_n:
            return 1.0
        
        # 获取基团间交互参数amn
        if (group_m, group_n) in self.group_interactions:
            amn = self.group_interactions[(group_m, group_n)]
        else:
            amn = 0.0  # 默认值
        
        # 计算ψmn = exp(-amn/T)
        return np.exp(-amn / T)
    
    def set_group_assignments(self, compound_idx: int, groups: Dict[int, int]):
        """
        设置化合物的基团分配
        
        参数:
            compound_idx: 化合物索引
            groups: 基团分配字典 {基团ID: 数量}
        """
        self.group_assignments[compound_idx] = groups
        
        # 重新计算分子参数
        r_i = 0.0
        q_i = 0.0
        for group_id, count in groups.items():
            if group_id in self.sub_groups:
                r_i += count * self.sub_groups[group_id]['R']
                q_i += count * self.sub_groups[group_id]['Q']
        
        self.r_i[compound_idx] = r_i
        self.q_i[compound_idx] = q_i
        
        self.logger.info(f"设置{self.compounds[compound_idx].name}的基团分配: {groups}")
        self.logger.info(f"计算得到分子参数: r={r_i:.4f}, q={q_i:.4f}")
    
    def get_group_assignments(self, compound_idx: int) -> Dict[int, int]:
        """
        获取化合物的基团分配
        
        参数:
            compound_idx: 化合物索引
            
        返回:
            基团分配字典
        """
        return self.group_assignments.get(compound_idx, {})
    
    def get_available_groups(self) -> Dict[int, Dict]:
        """
        获取可用的基团列表
        
        返回:
            基团参数字典
        """
        return self.sub_groups
    
    def validate_parameters(self) -> bool:
        """
        验证模型参数的有效性
        
        返回:
            参数是否有效
        """
        # 检查所有化合物是否都有基团分配
        for i in range(len(self.compounds)):
            if i not in self.group_assignments or not self.group_assignments[i]:
                self.logger.warning(f"化合物{i}缺少基团分配")
                return False
        
        # 检查分子参数
        for i in range(len(self.compounds)):
            if self.r_i[i] <= 0 or self.q_i[i] <= 0:
                self.logger.warning(f"化合物{i}的分子参数无效: r={self.r_i[i]}, q={self.q_i[i]}")
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
            'description': 'UNIFAC基团贡献活度系数模型，基于分子基团结构预测活度系数',
            'parameters': {
                'available_groups': len(self.sub_groups),
                'group_interactions': len(self.group_interactions),
                'total_compounds': len(self.compounds)
            },
            'applicable_phases': ['Liquid'],
            'temperature_range': '250-600 K (建议范围)',
            'pressure_range': '0.1-50 bar (低中压系统)',
            'features': [
                '基团贡献法', '预测性模型', '广泛适用性',
                '组合项', '残基项', '基团交互参数'
            ],
            'advantages': [
                '可预测无实验数据的体系',
                '参数数量相对较少',
                '适用于多种化合物类型'
            ]
        } 