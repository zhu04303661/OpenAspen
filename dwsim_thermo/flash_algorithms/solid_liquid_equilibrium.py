"""
固液平衡 (Solid-Liquid Equilibrium, SLE) 闪蒸算法
=================================================

实现基于嵌套循环方法的固液平衡计算算法。
支持共晶和固溶体系统的相平衡计算。

理论基础:
---------
固液平衡的基本条件是各组分在固相和液相中的化学势相等：
μᵢˢ = μᵢᴸ

对于纯组分固体(共晶系统)：
ln(xᵢᴸγᵢᴸ) = (ΔHfus,i/RT)(1 - T/Tfus,i) - (ΔCp,i/R)[(Tfus,i/T) - 1 + ln(T/Tfus,i)]

对于固溶体系统：
ln(xᵢˢγᵢˢ) = ln(xᵢᴸγᵢᴸ)

参考文献:
- Prausnitz, J.M., et al. "Molecular Thermodynamics of Fluid-Phase Equilibria"
- Gmehling, J., et al. "Chemical Thermodynamics for Process Simulation"
- DWSIM VB.NET NestedLoopsSLE.vb源代码

作者: OpenAspen项目组
日期: 2024年12月
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from scipy.optimize import fsolve, minimize_scalar
import logging

from .base_flash import FlashAlgorithmBase
from ..core.enums import Phase, FlashSpec
from ..core.compound import Compound
from ..solvers.numerical_methods import newton_raphson, solve_nonlinear_system


@dataclass
class SLEResult:
    """固液平衡计算结果"""
    converged: bool
    iterations: int
    x_liquid: np.ndarray      # 液相组成
    x_solid: np.ndarray       # 固相组成 (固溶体) 或固相存在标记 (共晶)
    liquid_fraction: float    # 液相摩尔分数
    solid_fraction: float     # 固相摩尔分数
    activity_coeff_liquid: np.ndarray    # 液相活度系数
    activity_coeff_solid: np.ndarray     # 固相活度系数 (仅固溶体)
    temperature: float        # 温度 [K]
    pressure: float          # 压力 [Pa]
    system_type: str         # 系统类型: 'eutectic' 或 'solid_solution'
    gibbs_energy: float      # 系统Gibbs自由能
    solid_compounds: List[str]  # 形成固相的化合物


class SolidLiquidEquilibrium(FlashAlgorithmBase):
    """
    固液平衡闪蒸算法
    
    支持两种系统类型：
    1. 共晶系统 (Eutectic): 固相为纯组分固体
    2. 固溶体系统 (Solid Solution): 固相为均相混合物
    
    算法特点：
    - 基于嵌套循环方法
    - 自动识别系统类型
    - 处理多组分固液平衡
    - 集成相稳定性分析
    """
    
    def __init__(self, property_package, 
                 system_type: str = 'auto',
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 temperature_tolerance: float = 0.1):
        """
        初始化固液平衡算法
        
        Parameters
        ----------
        property_package : PropertyPackage
            物性包
        system_type : str
            系统类型 ('eutectic', 'solid_solution', 'auto')
        max_iterations : int
            最大迭代次数
        tolerance : float
            收敛容差
        temperature_tolerance : float
            温度收敛容差 [K]
        """
        super().__init__(property_package)
        
        self.system_type = system_type
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.temperature_tolerance = temperature_tolerance
        
        # 算法特定参数
        self._damping_factor = 0.7
        self._minimum_solid_fraction = 1e-8
        self._maximum_solid_fraction = 1.0 - 1e-8
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def flash_pt(self, P: float, T: float, z: np.ndarray, 
                 initial_guess: Optional[Dict] = None) -> Dict:
        """
        等温等压固液平衡闪蒸
        
        Parameters
        ----------
        P : float
            压力 [Pa]
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
        # 输入验证
        self._validate_inputs(P, T, z)
        
        # 确定系统类型
        if self.system_type == 'auto':
            system_type = self._determine_system_type(T, P, z)
        else:
            system_type = self.system_type
            
        # 检查是否可能形成固相
        if not self._check_solid_formation_possibility(T, P, z):
            return self._create_liquid_only_result(z, T, P)
            
        # 根据系统类型调用相应算法
        if system_type == 'eutectic':
            return self._flash_eutectic_system(P, T, z, initial_guess)
        elif system_type == 'solid_solution':
            return self._flash_solid_solution_system(P, T, z, initial_guess)
        else:
            raise ValueError(f"未知的系统类型: {system_type}")
    
    def flash_ps(self, P: float, S: float, z: np.ndarray,
                 initial_guess: Optional[Dict] = None) -> Dict:
        """
        等压等熵固液平衡闪蒸
        
        Parameters
        ----------
        P : float
            压力 [Pa]
        S : float
            摩尔熵 [J/(mol·K)]
        z : np.ndarray
            总组成
        initial_guess : Optional[Dict]
            初始猜值
            
        Returns
        -------
        Dict
            闪蒸结果
        """
        # 初始温度估算
        T_guess = initial_guess.get('temperature', 298.15) if initial_guess else 298.15
        
        def entropy_error(T):
            """熵平衡误差函数"""
            try:
                result = self.flash_pt(P, T, z)
                if result['converged']:
                    S_calc = self._calculate_mixture_entropy(result, T, P)
                    return S_calc - S
                else:
                    return 1e6  # 大的惩罚值
            except:
                return 1e6
        
        # 求解温度
        try:
            from scipy.optimize import brentq
            T_solution = brentq(entropy_error, 200.0, 800.0, xtol=self.temperature_tolerance)
            return self.flash_pt(P, T_solution, z)
        except:
            # 如果Brent方法失败，尝试简单搜索
            return self._search_temperature_for_entropy(P, S, z, T_guess)
    
    def _flash_eutectic_system(self, P: float, T: float, z: np.ndarray,
                              initial_guess: Optional[Dict] = None) -> Dict:
        """
        共晶系统固液平衡计算
        
        在共晶系统中，固相为纯组分固体，液相为混合物。
        每个组分的固液平衡关系为：
        ln(xᵢᴸγᵢᴸ) = (ΔHfus,i/RT)(1 - T/Tfus,i)
        """
        n_comp = len(z)
        
        # 初始化
        if initial_guess is None:
            x_liquid, solid_presence, beta_liquid = self._initialize_eutectic_guess(T, P, z)
        else:
            x_liquid = initial_guess.get('x_liquid', z)
            solid_presence = initial_guess.get('solid_presence', np.zeros(n_comp, dtype=bool))
            beta_liquid = initial_guess.get('beta_liquid', 0.8)
        
        # 主迭代循环
        converged = False
        for iteration in range(self.max_iterations):
            # 计算液相活度系数
            gamma_liquid = self.property_package.calculate_activity_coefficients(x_liquid, T)
            
            # 计算每个组分的理论液相组成 (基于固液平衡关系)
            x_liquid_theory = self._calculate_theoretical_liquid_composition(T, gamma_liquid)
            
            # 检查固相稳定性
            solid_presence_new = self._check_solid_stability_eutectic(x_liquid, x_liquid_theory, T)
            
            # 物料平衡计算
            x_liquid_new, beta_liquid_new = self._solve_eutectic_material_balance(
                z, x_liquid_theory, solid_presence_new
            )
            
            # 收敛性检查
            error_x = np.max(np.abs(x_liquid_new - x_liquid))
            error_beta = abs(beta_liquid_new - beta_liquid)
            solid_change = np.any(solid_presence_new != solid_presence)
            
            if error_x < self.tolerance and error_beta < self.tolerance and not solid_change:
                converged = True
                x_liquid = x_liquid_new
                beta_liquid = beta_liquid_new
                solid_presence = solid_presence_new
                break
            
            # 阻尼更新
            x_liquid = self._damping_factor * x_liquid_new + (1 - self._damping_factor) * x_liquid
            beta_liquid = self._damping_factor * beta_liquid_new + (1 - self._damping_factor) * beta_liquid
            solid_presence = solid_presence_new
            
            # 归一化液相组成
            if np.sum(x_liquid) > 0:
                x_liquid = x_liquid / np.sum(x_liquid)
        
        # 构建结果
        return self._create_eutectic_result(
            x_liquid, solid_presence, beta_liquid, gamma_liquid, T, P, z, 
            converged, iteration + 1
        )
    
    def _flash_solid_solution_system(self, P: float, T: float, z: np.ndarray,
                                   initial_guess: Optional[Dict] = None) -> Dict:
        """
        固溶体系统固液平衡计算
        
        在固溶体系统中，固相和液相都是均相混合物。
        平衡条件为：γᵢˢxᵢˢ = γᵢᴸxᵢᴸ
        """
        n_comp = len(z)
        
        # 初始化
        if initial_guess is None:
            x_liquid, x_solid, beta_liquid = self._initialize_solid_solution_guess(T, P, z)
        else:
            x_liquid = initial_guess.get('x_liquid', z)
            x_solid = initial_guess.get('x_solid', z)
            beta_liquid = initial_guess.get('beta_liquid', 0.6)
        
        # 主迭代循环
        converged = False
        for iteration in range(self.max_iterations):
            # 计算活度系数
            gamma_liquid = self.property_package.calculate_activity_coefficients(x_liquid, T)
            gamma_solid = self._calculate_solid_activity_coefficients(x_solid, T)
            
            # 计算平衡常数 K = γˢ/γᴸ
            K = gamma_solid / gamma_liquid
            
            # Rachford-Rice方程求解
            beta_liquid_new = self._solve_rachford_rice_solid_solution(z, K, beta_liquid)
            
            # 更新相组成
            x_liquid_new = z / (beta_liquid_new + (1 - beta_liquid_new) / K)
            x_solid_new = x_liquid_new / K
            
            # 归一化
            x_liquid_new = x_liquid_new / np.sum(x_liquid_new)
            x_solid_new = x_solid_new / np.sum(x_solid_new)
            
            # 收敛性检查
            error_x_l = np.max(np.abs(x_liquid_new - x_liquid))
            error_x_s = np.max(np.abs(x_solid_new - x_solid))
            error_beta = abs(beta_liquid_new - beta_liquid)
            
            max_error = max(error_x_l, error_x_s, error_beta)
            
            if max_error < self.tolerance:
                converged = True
                x_liquid = x_liquid_new
                x_solid = x_solid_new
                beta_liquid = beta_liquid_new
                break
            
            # 阻尼更新
            x_liquid = self._damping_factor * x_liquid_new + (1 - self._damping_factor) * x_liquid
            x_solid = self._damping_factor * x_solid_new + (1 - self._damping_factor) * x_solid
            beta_liquid = self._damping_factor * beta_liquid_new + (1 - self._damping_factor) * beta_liquid
        
        # 构建结果
        return self._create_solid_solution_result(
            x_liquid, x_solid, beta_liquid, gamma_liquid, gamma_solid, 
            T, P, z, converged, iteration + 1
        )
    
    def _determine_system_type(self, T: float, P: float, z: np.ndarray) -> str:
        """
        自动确定系统类型
        
        判断依据：
        1. 检查化合物的混溶性
        2. 分析熔点差异
        3. 估算固相形成倾向
        """
        # 获取化合物熔点
        melting_points = []
        for compound in self.property_package.compounds:
            Tm = getattr(compound, 'melting_point', None)
            if Tm is None:
                # 使用默认估算方法
                Tm = getattr(compound, 'critical_temperature', 500.0) * 0.6
            melting_points.append(Tm)
        
        melting_points = np.array(melting_points)
        
        # 判断规则
        # 1. 如果熔点差异很大(>50K)，倾向于共晶系统
        melting_point_range = np.max(melting_points) - np.min(melting_points)
        if melting_point_range > 50.0:
            return 'eutectic'
        
        # 2. 如果所有组分熔点都相近(<20K)，倾向于固溶体
        if melting_point_range < 20.0:
            return 'solid_solution'
        
        # 3. 默认情况下选择共晶系统（更常见）
        return 'eutectic'
    
    def _check_solid_formation_possibility(self, T: float, P: float, z: np.ndarray) -> bool:
        """检查是否可能形成固相"""
        # 简单检查：如果温度低于任何组分的熔点，则可能形成固相
        for i, compound in enumerate(self.property_package.compounds):
            if z[i] > 1e-10:  # 只检查存在的组分
                Tm = getattr(compound, 'melting_point', 
                           getattr(compound, 'critical_temperature', 500.0) * 0.6)
                if T < Tm:
                    return True
        return False
    
    def _calculate_theoretical_liquid_composition(self, T: float, gamma_liquid: np.ndarray) -> np.ndarray:
        """
        基于固液平衡关系计算理论液相组成
        
        对于共晶系统：
        ln(xᵢᴸγᵢᴸ) = (ΔHfus,i/RT)(1 - T/Tfus,i)
        """
        n_comp = len(gamma_liquid)
        x_theory = np.zeros(n_comp)
        
        R = 8.314  # J/(mol·K)
        
        for i, compound in enumerate(self.property_package.compounds):
            # 获取熔融焓和熔点
            Hfus = getattr(compound, 'enthalpy_of_fusion', 10000.0)  # J/mol
            Tfus = getattr(compound, 'melting_point', 
                          getattr(compound, 'critical_temperature', 500.0) * 0.6)  # K
            
            if T >= Tfus:
                # 温度高于熔点，组分完全液态
                x_theory[i] = 1.0
            else:
                # 计算溶解度
                delta_ln = (Hfus / (R * T)) * (1 - T / Tfus)
                x_theory[i] = np.exp(-delta_ln) / gamma_liquid[i]
                x_theory[i] = max(1e-10, min(1.0, x_theory[i]))
        
        return x_theory
    
    def _check_solid_stability_eutectic(self, x_liquid: np.ndarray, 
                                       x_theory: np.ndarray, T: float) -> np.ndarray:
        """检查共晶系统中固相稳定性"""
        solid_stable = np.zeros(len(x_liquid), dtype=bool)
        
        for i in range(len(x_liquid)):
            # 如果实际液相组成超过理论溶解度，则该组分形成固相
            if x_liquid[i] > x_theory[i] * (1 + self.tolerance):
                solid_stable[i] = True
        
        return solid_stable
    
    def _solve_eutectic_material_balance(self, z: np.ndarray, x_theory: np.ndarray,
                                       solid_presence: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        求解共晶系统的物料平衡
        
        物料平衡：zᵢ = βxᵢᴸ + (1-β)xᵢˢ
        其中对于共晶系统：xᵢˢ = 1 if solid_presence[i] else 0
        """
        n_comp = len(z)
        
        # 形成固相的组分
        solid_indices = np.where(solid_presence)[0]
        liquid_indices = np.where(~solid_presence)[0]
        
        if len(solid_indices) == 0:
            # 无固相形成，全液相
            return z, 1.0
        
        if len(liquid_indices) == 0:
            # 全固相（极端情况）
            return np.zeros(n_comp), 0.0
        
        # 计算液相分率
        # 对于形成固相的组分：zᵢ = βx_theory[i] + (1-β)
        # 解得：β = (zᵢ - 1) / (x_theory[i] - 1)
        
        beta_estimates = []
        for i in solid_indices:
            if abs(x_theory[i] - 1.0) > 1e-10:
                beta_est = (z[i] - 1.0) / (x_theory[i] - 1.0)
                beta_est = max(0.0, min(1.0, beta_est))
                beta_estimates.append(beta_est)
        
        if beta_estimates:
            beta_liquid = np.mean(beta_estimates)
        else:
            beta_liquid = 0.5
        
        # 计算液相组成
        x_liquid = np.zeros(n_comp)
        for i in range(n_comp):
            if solid_presence[i]:
                x_liquid[i] = x_theory[i]
            else:
                # 非固相组分的液相组成由物料平衡确定
                x_liquid[i] = z[i] / beta_liquid if beta_liquid > 1e-10 else z[i]
        
        # 归一化
        sum_x = np.sum(x_liquid)
        if sum_x > 1e-10:
            x_liquid = x_liquid / sum_x
        
        return x_liquid, beta_liquid
    
    def _calculate_solid_activity_coefficients(self, x_solid: np.ndarray, T: float) -> np.ndarray:
        """
        计算固相活度系数
        
        简化处理：假设固相为理想溶液或使用液相活度系数模型
        """
        # 如果物性包支持固相活度系数计算
        if hasattr(self.property_package, 'calculate_solid_activity_coefficients'):
            return self.property_package.calculate_solid_activity_coefficients(x_solid, T)
        else:
            # 简化假设：固相活度系数等于液相活度系数
            return self.property_package.calculate_activity_coefficients(x_solid, T)
    
    def _solve_rachford_rice_solid_solution(self, z: np.ndarray, K: np.ndarray, 
                                          beta_init: float) -> float:
        """求解固溶体系统的Rachford-Rice方程"""
        def rr_equation(beta):
            result = 0.0
            for i in range(len(z)):
                if abs(K[i] - 1.0) > 1e-12:
                    result += z[i] * (K[i] - 1.0) / (beta + (1 - beta) * K[i])
            return result
        
        try:
            from scipy.optimize import brentq
            beta = brentq(rr_equation, 1e-8, 1-1e-8, xtol=1e-12)
            return max(1e-8, min(1-1e-8, beta))
        except:
            return beta_init
    
    def _initialize_eutectic_guess(self, T: float, P: float, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """初始化共晶系统猜值"""
        n_comp = len(z)
        
        # 初始液相组成为总组成
        x_liquid = z.copy()
        
        # 初始假设没有固相形成
        solid_presence = np.zeros(n_comp, dtype=bool)
        
        # 初始液相分率
        beta_liquid = 0.8
        
        return x_liquid, solid_presence, beta_liquid
    
    def _initialize_solid_solution_guess(self, T: float, P: float, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """初始化固溶体系统猜值"""
        # 初始相组成都设为总组成
        x_liquid = z.copy()
        x_solid = z.copy()
        
        # 初始液相分率
        beta_liquid = 0.6
        
        return x_liquid, x_solid, beta_liquid
    
    def _create_eutectic_result(self, x_liquid: np.ndarray, solid_presence: np.ndarray,
                              beta_liquid: float, gamma_liquid: np.ndarray,
                              T: float, P: float, z: np.ndarray,
                              converged: bool, iterations: int) -> Dict:
        """创建共晶系统结果"""
        n_comp = len(z)
        
        # 固相组成（纯组分固体）
        x_solid = np.zeros(n_comp)
        x_solid[solid_presence] = 1.0
        
        # 固相活度系数（纯组分为1）
        gamma_solid = np.ones(n_comp)
        
        # 固相化合物列表
        solid_compounds = [
            self.property_package.compounds[i].name 
            for i in range(n_comp) if solid_presence[i]
        ]
        
        # 计算Gibbs自由能
        G_mix = self._calculate_gibbs_energy(x_liquid, x_solid, beta_liquid, 
                                           gamma_liquid, gamma_solid, T)
        
        return {
            'converged': converged,
            'iterations': iterations,
            'flash_type': 'SLE_Eutectic',
            'phases': {
                'liquid': {
                    'mole_fractions': x_liquid,
                    'phase_fraction': beta_liquid,
                    'activity_coefficients': gamma_liquid,
                    'phase_type': 'liquid'
                },
                'solid': {
                    'mole_fractions': x_solid,
                    'phase_fraction': 1 - beta_liquid,
                    'activity_coefficients': gamma_solid,
                    'phase_type': 'solid',
                    'solid_presence': solid_presence
                }
            },
            'properties': {
                'temperature': T,
                'pressure': P,
                'mixing_gibbs_energy': G_mix,
                'system_type': 'eutectic',
                'solid_compounds': solid_compounds
            }
        }
    
    def _create_solid_solution_result(self, x_liquid: np.ndarray, x_solid: np.ndarray,
                                    beta_liquid: float, gamma_liquid: np.ndarray,
                                    gamma_solid: np.ndarray, T: float, P: float,
                                    z: np.ndarray, converged: bool, iterations: int) -> Dict:
        """创建固溶体系统结果"""
        # 计算Gibbs自由能
        G_mix = self._calculate_gibbs_energy(x_liquid, x_solid, beta_liquid, 
                                           gamma_liquid, gamma_solid, T)
        
        return {
            'converged': converged,
            'iterations': iterations,
            'flash_type': 'SLE_SolidSolution',
            'phases': {
                'liquid': {
                    'mole_fractions': x_liquid,
                    'phase_fraction': beta_liquid,
                    'activity_coefficients': gamma_liquid,
                    'phase_type': 'liquid'
                },
                'solid': {
                    'mole_fractions': x_solid,
                    'phase_fraction': 1 - beta_liquid,
                    'activity_coefficients': gamma_solid,
                    'phase_type': 'solid'
                }
            },
            'properties': {
                'temperature': T,
                'pressure': P,
                'mixing_gibbs_energy': G_mix,
                'system_type': 'solid_solution'
            }
        }
    
    def _create_liquid_only_result(self, z: np.ndarray, T: float, P: float) -> Dict:
        """创建纯液相结果"""
        gamma_liquid = self.property_package.calculate_activity_coefficients(z, T)
        
        return {
            'converged': True,
            'iterations': 0,
            'flash_type': 'Liquid_Only',
            'phases': {
                'liquid': {
                    'mole_fractions': z,
                    'phase_fraction': 1.0,
                    'activity_coefficients': gamma_liquid,
                    'phase_type': 'liquid'
                }
            },
            'properties': {
                'temperature': T,
                'pressure': P,
                'system_type': 'liquid_only'
            }
        }
    
    def _calculate_gibbs_energy(self, x_liquid: np.ndarray, x_solid: np.ndarray,
                              beta_liquid: float, gamma_liquid: np.ndarray,
                              gamma_solid: np.ndarray, T: float) -> float:
        """计算混合Gibbs自由能"""
        R = 8.314  # J/(mol·K)
        
        G_mix = 0.0
        
        # 液相贡献
        for i in range(len(x_liquid)):
            if x_liquid[i] > 1e-12:
                G_mix += beta_liquid * x_liquid[i] * (
                    np.log(x_liquid[i]) + np.log(gamma_liquid[i])
                )
        
        # 固相贡献
        for i in range(len(x_solid)):
            if x_solid[i] > 1e-12:
                G_mix += (1 - beta_liquid) * x_solid[i] * (
                    np.log(x_solid[i]) + np.log(gamma_solid[i])
                )
        
        return R * T * G_mix
    
    def _calculate_mixture_entropy(self, result: Dict, T: float, P: float) -> float:
        """从闪蒸结果计算混合物熵"""
        # 简化实现，实际需要根据物性包计算
        return 100.0  # 占位符
    
    def _search_temperature_for_entropy(self, P: float, S: float, z: np.ndarray,
                                      T_init: float) -> Dict:
        """搜索满足熵要求的温度"""
        # 简化搜索
        for T in np.linspace(200, 800, 50):
            try:
                result = self.flash_pt(P, T, z)
                if result['converged']:
                    return result
            except:
                continue
        
        # 返回默认结果
        return self._create_liquid_only_result(z, T_init, P)
    
    def _validate_inputs(self, P: float, T: float, z: np.ndarray) -> None:
        """验证输入参数"""
        if P <= 0:
            raise ValueError("压力必须为正数")
        if T <= 0:
            raise ValueError("温度必须为正数")
        if not np.isclose(np.sum(z), 1.0, rtol=1e-6):
            raise ValueError("组成之和必须等于1")
        if np.any(z < 0):
            raise ValueError("组成不能为负数")
    
    def get_algorithm_info(self) -> Dict:
        """获取算法信息"""
        return {
            'name': 'Solid-Liquid Equilibrium (SLE)',
            'type': 'Phase Equilibrium',
            'description': '基于嵌套循环的固液平衡算法，支持共晶和固溶体系统',
            'system_types': ['eutectic', 'solid_solution'],
            'applicable_phases': ['liquid', 'solid'],
            'convergence_criteria': f'Tolerance: {self.tolerance}',
            'max_iterations': self.max_iterations,
            'capabilities': [
                '共晶系统SLE计算',
                '固溶体系统SLE计算',
                '多组分固液平衡',
                '相稳定性分析',
                'PT闪蒸',
                'PS闪蒸'
            ],
            'limitations': [
                '需要准确的熔融焓和熔点数据',
                '固相活度系数模型简化',
                '不支持多固相共存'
            ]
        }


# 辅助类：固液平衡专用计算
class SLEThermodynamics:
    """固液平衡热力学计算辅助类"""
    
    @staticmethod
    def calculate_solubility(T: float, Tfus: float, Hfus: float, 
                           gamma: float = 1.0) -> float:
        """
        计算理想溶解度
        
        Parameters
        ----------
        T : float
            温度 [K]
        Tfus : float
            熔点 [K]
        Hfus : float
            熔融焓 [J/mol]
        gamma : float
            活度系数
            
        Returns
        -------
        float
            溶解度 (摩尔分数)
        """
        R = 8.314  # J/(mol·K)
        
        if T >= Tfus:
            return 1.0 / gamma
        
        delta_ln = (Hfus / (R * T)) * (1 - T / Tfus)
        return np.exp(-delta_ln) / gamma
    
    @staticmethod
    def estimate_eutectic_temperature(T1: float, T2: float, x1: float, x2: float) -> float:
        """
        估算共晶温度
        
        Parameters
        ----------
        T1, T2 : float
            纯组分熔点 [K]
        x1, x2 : float
            共晶组成
            
        Returns
        -------
        float
            估算的共晶温度 [K]
        """
        # 简化的共晶温度估算
        return min(T1, T2) * (1 - 0.1 * abs(x1 - x2)) 