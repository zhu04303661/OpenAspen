"""
简单液液平衡(Simple LLE)闪蒸算法
===============================

实现基于活度系数模型的简单液液平衡计算算法。
该算法适用于部分互溶的液液体系。

参考文献:
- Prausnitz, J.M., et al. "Molecular Thermodynamics of Fluid-Phase Equilibria"
- Sandler, S.I. "Chemical and Engineering Thermodynamics"

作者: OpenAspen项目组
日期: 2024年12月
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass

from .base_flash import FlashAlgorithmBase
from ..core.enums import Phase, FlashSpec
from ..core.compound import Compound
from ..solvers.numerical_methods import newton_raphson, solve_nonlinear_system


@dataclass
class LLEResult:
    """液液平衡计算结果"""
    converged: bool
    iterations: int
    x1: np.ndarray  # 液相1组成
    x2: np.ndarray  # 液相2组成
    phase1_fraction: float  # 液相1摩尔分数
    phase2_fraction: float  # 液相2摩尔分数
    activity_coeff1: np.ndarray  # 液相1活度系数
    activity_coeff2: np.ndarray  # 液相2活度系数
    gibbs_energy: float  # 混合Gibbs自由能


class SimpleLLE(FlashAlgorithmBase):
    """
    简单液液平衡闪蒸算法
    
    该算法基于以下平衡条件：
    - 化学势平衡: μᵢ¹ = μᵢ²
    - 等价条件: xᵢ¹γᵢ¹ = xᵢ²γᵢ²
    
    求解方法:
    1. 初始化两相组成估值
    2. 迭代计算活度系数
    3. 更新相组成和相分率
    4. 检查收敛性
    
    适用范围:
    - 部分互溶液液体系
    - 温度: 250-400K
    - 压力: 1-50 bar
    """
    
    def __init__(self, property_package, 
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 damping_factor: float = 0.5):
        """
        初始化简单LLE闪蒸算法
        
        Parameters
        ----------
        property_package : PropertyPackage
            物性包
        max_iterations : int
            最大迭代次数
        tolerance : float
            收敛容差
        damping_factor : float
            阻尼因子 (0 < damping_factor ≤ 1)
        """
        super().__init__(property_package)
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping_factor = damping_factor
        
        # 算法特定参数
        self._stability_check = True  # 是否进行稳定性检查
        self._trivial_solution_threshold = 1e-4  # 平凡解阈值
        
    def flash_pt(self, P: float, T: float, z: np.ndarray, 
                 initial_guess: Optional[Dict] = None) -> Dict:
        """
        等温等压液液平衡闪蒸
        
        Parameters
        ----------
        P : float
            压力 [bar]
        T : float
            温度 [K]
        z : np.ndarray
            总组成
        initial_guess : Optional[Dict]
            初始猜值
            
        Returns
        -------
        Dict
            闪蒸结果
        """
        n_comp = len(z)
        
        # 输入验证
        if not np.isclose(np.sum(z), 1.0, rtol=1e-10):
            raise ValueError("总组成之和必须等于1")
            
        if np.any(z < 0):
            raise ValueError("组成不能为负数")
            
        # 稳定性检查
        if self._stability_check:
            is_stable = self._check_liquid_stability(T, P, z)
            if is_stable:
                return self._create_single_phase_result(z, 'L1')
                
        # 初始化
        x1, x2, beta = self._initialize_compositions(z, initial_guess)
        
        # 主迭代循环
        converged = False
        iteration = 0
        
        for iteration in range(self.max_iterations):
            # 计算活度系数
            gamma1 = self.property_package.calculate_activity_coefficients(x1, T)
            gamma2 = self.property_package.calculate_activity_coefficients(x2, T)
            
            # 计算新的组成
            x1_new, x2_new, beta_new = self._solve_material_balance(
                z, x1, x2, beta, gamma1, gamma2
            )
            
            # 检查收敛性
            error1 = np.max(np.abs(x1_new - x1))
            error2 = np.max(np.abs(x2_new - x2))
            error_beta = abs(beta_new - beta)
            
            max_error = max(error1, error2, error_beta)
            
            if max_error < self.tolerance:
                converged = True
                x1, x2, beta = x1_new, x2_new, beta_new
                break
                
            # 应用阻尼更新
            x1 = self.damping_factor * x1_new + (1 - self.damping_factor) * x1
            x2 = self.damping_factor * x2_new + (1 - self.damping_factor) * x2
            beta = self.damping_factor * beta_new + (1 - self.damping_factor) * beta
            
            # 归一化组成
            x1 = x1 / np.sum(x1)
            x2 = x2 / np.sum(x2)
            
        # 最终活度系数计算
        gamma1 = self.property_package.calculate_activity_coefficients(x1, T)
        gamma2 = self.property_package.calculate_activity_coefficients(x2, T)
        
        # 检查平凡解
        if self._is_trivial_solution(x1, x2, beta):
            return self._create_single_phase_result(z, 'L1')
            
        # 计算Gibbs混合能
        G_mix = self._calculate_mixing_gibbs_energy(z, x1, x2, beta, gamma1, gamma2, T)
        
        # 创建结果
        result = {
            'converged': converged,
            'iterations': iteration + 1,
            'flash_type': 'LLE',
            'phases': {
                'L1': {
                    'mole_fractions': x1,
                    'phase_fraction': beta,
                    'activity_coefficients': gamma1,
                    'phase_type': 'liquid'
                },
                'L2': {
                    'mole_fractions': x2,
                    'phase_fraction': 1 - beta,
                    'activity_coefficients': gamma2,
                    'phase_type': 'liquid'
                }
            },
            'properties': {
                'mixing_gibbs_energy': G_mix,
                'temperature': T,
                'pressure': P
            }
        }
        
        if not converged:
            warnings.warn(f"LLE闪蒸未收敛，迭代次数: {iteration + 1}")
            
        return result
        
    def _check_liquid_stability(self, T: float, P: float, z: np.ndarray) -> bool:
        """
        检查液相稳定性
        
        Parameters
        ----------
        T : float
            温度 [K]
        P : float
            压力 [bar]
        z : np.ndarray
            总组成
            
        Returns
        -------
        bool
            True如果液相稳定
        """
        # 计算总组成的活度系数
        gamma_z = self.property_package.calculate_activity_coefficients(z, T)
        
        # 计算切线平面距离 (TPD)
        # TPD = Σ xᵢ(ln(xᵢ) + ln(γᵢ) - ln(zᵢ) - ln(γᵢᶻ))
        
        # 使用多个试验组成进行稳定性测试
        test_compositions = self._generate_test_compositions(len(z))
        
        for x_test in test_compositions:
            gamma_test = self.property_package.calculate_activity_coefficients(x_test, T)
            
            # 计算TPD
            tpd = 0.0
            for i in range(len(z)):
                if x_test[i] > 1e-12 and z[i] > 1e-12:
                    tpd += x_test[i] * (
                        np.log(x_test[i]) + np.log(gamma_test[i]) - 
                        np.log(z[i]) - np.log(gamma_z[i])
                    )
                    
            # 如果任何试验组成的TPD < 0，则不稳定
            if tpd < -1e-8:
                return False
                
        return True
        
    def _generate_test_compositions(self, n_comp: int) -> List[np.ndarray]:
        """
        生成稳定性测试用的试验组成
        
        Parameters
        ----------
        n_comp : int
            组分数
            
        Returns
        -------
        List[np.ndarray]
            试验组成列表
        """
        test_compositions = []
        
        # 纯组分组成
        for i in range(n_comp):
            x_test = np.zeros(n_comp)
            x_test[i] = 1.0
            test_compositions.append(x_test)
            
        # 等摩尔组成
        x_equal = np.ones(n_comp) / n_comp
        test_compositions.append(x_equal)
        
        # 随机组成
        np.random.seed(42)  # 确保可重现性
        for _ in range(5):
            x_random = np.random.random(n_comp)
            x_random = x_random / np.sum(x_random)
            test_compositions.append(x_random)
            
        return test_compositions
        
    def _initialize_compositions(self, z: np.ndarray, 
                               initial_guess: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        初始化两相组成
        
        Parameters
        ----------
        z : np.ndarray
            总组成
        initial_guess : Optional[Dict]
            初始猜值
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            (x1, x2, beta)
        """
        n_comp = len(z)
        
        if initial_guess is not None and 'x1' in initial_guess:
            x1 = np.array(initial_guess['x1'])
            x2 = np.array(initial_guess.get('x2', z))
            beta = initial_guess.get('beta', 0.5)
        else:
            # 默认初始化策略
            # 假设最极性组分主要在一相，最非极性组分主要在另一相
            
            # 简化的极性估算 (基于分子量和化合物名称)
            polarity_scores = self._estimate_polarity_scores()
            
            # 创建两个相偏的初始组成
            x1 = z.copy()
            x2 = z.copy()
            
            # 调整组成使其偏向不同相
            for i in range(n_comp):
                if polarity_scores[i] > 0.5:  # 极性组分偏向相1
                    x1[i] *= 1.5
                    x2[i] *= 0.5
                else:  # 非极性组分偏向相2
                    x1[i] *= 0.5
                    x2[i] *= 1.5
                    
            # 归一化
            x1 = x1 / np.sum(x1)
            x2 = x2 / np.sum(x2)
            beta = 0.5  # 初始相分率
            
        return x1, x2, beta
        
    def _estimate_polarity_scores(self) -> np.ndarray:
        """
        估算组分极性得分
        
        Returns
        -------
        np.ndarray
            极性得分数组 (0-1)
        """
        n_comp = len(self.property_package.compounds)
        scores = np.zeros(n_comp)
        
        for i, compound in enumerate(self.property_package.compounds):
            # 基于化合物名称的简单判断
            name = compound.lower()
            
            if 'water' in name or 'h2o' in name:
                scores[i] = 1.0  # 水最极性
            elif any(alcohol in name for alcohol in ['methanol', 'ethanol', 'alcohol']):
                scores[i] = 0.8  # 醇类高极性
            elif any(acid in name for acid in ['acid', 'acetic']):
                scores[i] = 0.7  # 酸类中等极性
            elif any(aromatic in name for aromatic in ['benzene', 'toluene']):
                scores[i] = 0.3  # 芳烃低极性
            elif any(alkane in name for alkane in ['methane', 'ethane', 'propane', 'butane']):
                scores[i] = 0.1  # 烷烃很低极性
            else:
                scores[i] = 0.5  # 默认中等极性
                
        return scores
        
    def _solve_material_balance(self, z: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                              beta: float, gamma1: np.ndarray, gamma2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        求解物料平衡和相平衡
        
        Parameters
        ----------
        z : np.ndarray
            总组成
        x1, x2 : np.ndarray
            当前相组成
        beta : float
            当前相分率
        gamma1, gamma2 : np.ndarray
            活度系数
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            更新的(x1, x2, beta)
        """
        n_comp = len(z)
        
        # 平衡常数 K = (x1*gamma1)/(x2*gamma2)
        K = np.zeros(n_comp)
        for i in range(n_comp):
            if x2[i] > 1e-12 and gamma2[i] > 1e-12:
                K[i] = (x1[i] * gamma1[i]) / (x2[i] * gamma2[i])
            else:
                K[i] = 1.0
                
        # Rachford-Rice方程求解beta
        def rachford_rice(beta_trial):
            result = 0.0
            for i in range(n_comp):
                if abs(K[i] - 1.0) > 1e-12:
                    result += z[i] * (K[i] - 1.0) / (1.0 + beta_trial * (K[i] - 1.0))
            return result
            
        # 使用牛顿法求解beta
        try:
            beta_new = newton_raphson(
                rachford_rice,
                beta_init=beta,
                max_iterations=20,
                tolerance=1e-10
            )
            
            # 限制beta在合理范围内
            beta_new = max(1e-8, min(1-1e-8, beta_new))
            
        except:
            # 如果牛顿法失败，使用二分法
            beta_new = self._solve_rachford_rice_bisection(rachford_rice, 1e-8, 1-1e-8)
            
        # 计算新的相组成
        x1_new = np.zeros(n_comp)
        x2_new = np.zeros(n_comp)
        
        for i in range(n_comp):
            denom = 1.0 + beta_new * (K[i] - 1.0)
            if abs(denom) > 1e-12:
                x2_new[i] = z[i] / denom
                x1_new[i] = K[i] * x2_new[i]
            else:
                x1_new[i] = z[i]
                x2_new[i] = z[i]
                
        # 归一化
        sum1 = np.sum(x1_new)
        sum2 = np.sum(x2_new)
        
        if sum1 > 1e-12:
            x1_new = x1_new / sum1
        if sum2 > 1e-12:
            x2_new = x2_new / sum2
            
        return x1_new, x2_new, beta_new
        
    def _solve_rachford_rice_bisection(self, func, a: float, b: float) -> float:
        """
        使用二分法求解Rachford-Rice方程
        
        Parameters
        ----------
        func : callable
            Rachford-Rice函数
        a, b : float
            搜索区间
            
        Returns
        -------
        float
            方程根
        """
        for _ in range(50):  # 最多50次迭代
            c = (a + b) / 2
            fc = func(c)
            
            if abs(fc) < 1e-10:
                return c
                
            if func(a) * fc < 0:
                b = c
            else:
                a = c
                
        return (a + b) / 2
        
    def _is_trivial_solution(self, x1: np.ndarray, x2: np.ndarray, beta: float) -> bool:
        """
        检查是否为平凡解
        
        Parameters
        ----------
        x1, x2 : np.ndarray
            相组成
        beta : float
            相分率
            
        Returns
        -------
        bool
            True如果是平凡解
        """
        # 检查组成是否太相似
        composition_diff = np.max(np.abs(x1 - x2))
        
        # 检查相分率是否太极端
        phase_fraction_extreme = (beta < self._trivial_solution_threshold or 
                                beta > 1 - self._trivial_solution_threshold)
        
        return composition_diff < self._trivial_solution_threshold or phase_fraction_extreme
        
    def _calculate_mixing_gibbs_energy(self, z: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                                     beta: float, gamma1: np.ndarray, gamma2: np.ndarray,
                                     T: float) -> float:
        """
        计算混合Gibbs自由能
        
        Parameters
        ----------
        z : np.ndarray
            总组成
        x1, x2 : np.ndarray
            相组成
        beta : float
            相分率
        gamma1, gamma2 : np.ndarray
            活度系数
        T : float
            温度 [K]
            
        Returns
        -------
        float
            混合Gibbs自由能 [J/mol]
        """
        R = 8.314  # J/(mol·K)
        
        # 理想混合项
        G_ideal = 0.0
        for i in range(len(z)):
            if z[i] > 1e-12:
                G_ideal += z[i] * np.log(z[i])
                
        # 实际混合项
        G_real = 0.0
        for i in range(len(z)):
            if x1[i] > 1e-12:
                G_real += beta * x1[i] * (np.log(x1[i]) + np.log(gamma1[i]))
            if x2[i] > 1e-12:
                G_real += (1-beta) * x2[i] * (np.log(x2[i]) + np.log(gamma2[i]))
                
        G_mix = R * T * (G_real - G_ideal)
        
        return G_mix
        
    def _create_single_phase_result(self, z: np.ndarray, phase_type: str) -> Dict:
        """
        创建单相结果
        
        Parameters
        ----------
        z : np.ndarray
            组成
        phase_type : str
            相类型
            
        Returns
        -------
        Dict
            单相结果
        """
        return {
            'converged': True,
            'iterations': 0,
            'flash_type': 'Single Phase',
            'phases': {
                phase_type: {
                    'mole_fractions': z,
                    'phase_fraction': 1.0,
                    'phase_type': 'liquid'
                }
            }
        }
        
    def get_algorithm_info(self) -> Dict:
        """
        获取算法信息
        
        Returns
        -------
        Dict
            算法信息
        """
        return {
            'name': 'Simple LLE',
            'type': 'Liquid-Liquid Equilibrium',
            'description': '基于活度系数模型的简单液液平衡算法',
            'applicable_systems': ['Partially miscible liquids'],
            'convergence_criteria': f'Tolerance: {self.tolerance}',
            'max_iterations': self.max_iterations,
            'damping_factor': self.damping_factor,
            'stability_check': self._stability_check
        } 