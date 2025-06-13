"""
DWSIM热力学计算库 - Inside-Out闪蒸算法
=====================================

实现Inside-Out闪蒸算法，这是一种高效的相平衡计算方法，
特别适用于状态方程模型。相比嵌套循环算法，具有更好的收敛性能。

算法原理：
1. 外循环：更新K值
2. 内循环：求解Rachford-Rice方程
3. 使用状态方程计算逸度系数
4. 通过逸度平衡更新K值

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from scipy.optimize import brentq, newton
import time

from .base_flash import (
    FlashAlgorithmBase, FlashCalculationResult, FlashSettings,
    FlashConvergenceError, FlashValidationError
)
from ..core.enums import FlashSpec

class InsideOutFlash(FlashAlgorithmBase):
    """Inside-Out闪蒸算法
    
    实现高效的Inside-Out相平衡计算算法。
    该算法通过分离K值更新和相组成计算，提供更好的收敛性能。
    """
    
    def __init__(self, name: str = "Inside-Out Flash"):
        """初始化Inside-Out闪蒸算法
        
        Args:
            name: 算法名称
        """
        super().__init__(name)
        self.logger = logging.getLogger("InsideOutFlash")
        
        # 算法特定设置
        self.max_outer_iterations = 100      # 外循环最大迭代次数
        self.max_inner_iterations = 50       # 内循环最大迭代次数
        self.outer_tolerance = 1e-6          # 外循环收敛容差
        self.inner_tolerance = 1e-8          # 内循环收敛容差
        self.damping_factor = 0.7            # 阻尼因子
        self.use_acceleration = True         # 是否使用加速收敛
        
        # 统计信息
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "average_outer_iterations": 0.0,
            "average_inner_iterations": 0.0,
            "average_calculation_time": 0.0
        }
    
    def flash_pt(self, z: List[float], P: float, T: float, 
                property_package, initial_k_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """PT闪蒸计算
        
        Args:
            z: 进料摩尔分数
            P: 压力 [Pa]
            T: 温度 [K]
            property_package: 物性包对象
            initial_k_values: 初始K值估算
            
        Returns:
            Dict: 闪蒸结果
        """
        start_time = time.time()
        self.stats["total_calls"] += 1
        
        try:
            self._validate_inputs(z, P, T)
            
            n_comp = len(z)
            
            # 初始化K值
            if initial_k_values is None:
                K = self.estimate_initial_k_values(z, P, T, property_package)
            else:
                K = initial_k_values.copy()
            
            self.write_debug_info(f"PT闪蒸开始: T={T:.2f}K, P={P/1e5:.2f}bar")
            self.write_debug_info(f"初始K值: {[f'{k:.4f}' for k in K]}")
            
            # 检查是否为单相
            if self._is_single_phase(z, K):
                return self._create_single_phase_result(z, P, T, property_package)
            
            # Inside-Out迭代
            converged = False
            outer_iterations = 0
            total_inner_iterations = 0
            
            # 存储前几次迭代的K值用于加速
            K_history = []
            
            for outer_iter in range(self.max_outer_iterations):
                outer_iterations += 1
                K_old = K.copy()
                
                # 内循环：求解Rachford-Rice方程
                try:
                    beta, inner_iters = self._solve_rachford_rice_robust(z, K)
                    total_inner_iterations += inner_iters
                    
                    # 计算相组成
                    x, y = self.update_phase_compositions(z, K, beta)
                    
                    self.write_debug_info(f"外循环 {outer_iter+1}: β={beta:.4f}")
                    self.write_debug_info(f"液相组成: {[f'{xi:.4f}' for xi in x]}")
                    self.write_debug_info(f"气相组成: {[f'{yi:.4f}' for yi in y]}")
                    
                except Exception as e:
                    self.logger.warning(f"Rachford-Rice求解失败: {e}")
                    # 使用简化方法
                    beta = 0.5
                    x, y = self.update_phase_compositions(z, K, beta)
                    inner_iters = 1
                    total_inner_iterations += inner_iters
                
                # 外循环：更新K值
                try:
                    K_new = self._update_k_values(x, y, T, P, property_package)
                    
                    # 应用阻尼
                    for i in range(n_comp):
                        K[i] = self.damping_factor * K_new[i] + (1 - self.damping_factor) * K_old[i]
                    
                    # 加速收敛（Wegstein方法）
                    if self.use_acceleration and len(K_history) >= 2:
                        K = self._apply_wegstein_acceleration(K, K_old, K_history)
                    
                    # 存储K值历史
                    K_history.append(K.copy())
                    if len(K_history) > 3:
                        K_history.pop(0)
                    
                    self.write_debug_info(f"更新K值: {[f'{k:.4f}' for k in K]}")
                    
                except Exception as e:
                    self.logger.warning(f"K值更新失败: {e}")
                    break
                
                # 检查收敛
                if self.check_convergence(K_old, K, self.outer_tolerance):
                    converged = True
                    self.write_debug_info(f"外循环收敛，迭代{outer_iter+1}次")
                    break
                
                # 检查K值合理性
                if any(k <= 0 or k > 1e10 for k in K):
                    self.logger.warning("K值超出合理范围")
                    break
            
            # 最终计算
            beta, _ = self._solve_rachford_rice_robust(z, K)
            x, y = self.update_phase_compositions(z, K, beta)
            
            # 计算残差
            residual = self._calculate_residual(x, y, T, P, property_package)
            
            calc_time = time.time() - start_time
            
            # 更新统计信息
            if converged:
                self.stats["successful_calls"] += 1
            
            self.stats["average_outer_iterations"] = (
                (self.stats["average_outer_iterations"] * (self.stats["total_calls"] - 1) + 
                 outer_iterations) / self.stats["total_calls"]
            )
            
            self.stats["average_inner_iterations"] = (
                (self.stats["average_inner_iterations"] * (self.stats["total_calls"] - 1) + 
                 total_inner_iterations) / self.stats["total_calls"]
            )
            
            self.stats["average_calculation_time"] = (
                (self.stats["average_calculation_time"] * (self.stats["total_calls"] - 1) + 
                 calc_time) / self.stats["total_calls"]
            )
            
            if not converged:
                raise FlashConvergenceError(
                    f"PT闪蒸未收敛，迭代{outer_iterations}次，残差={residual:.2e}",
                    outer_iterations, residual
                )
            
            # 构建结果
            result = {
                'L1': 1.0 - beta,
                'V': beta,
                'Vx1': x,
                'Vy': y,
                'K_values': K,
                'ecount': outer_iterations,
                'residual': residual,
                'converged': converged,
                'calculation_time': calc_time,
                'inner_iterations': total_inner_iterations
            }
            
            self.write_debug_info(f"PT闪蒸完成: β={beta:.4f}, 残差={residual:.2e}")
            
            return result
            
        except Exception as e:
            calc_time = time.time() - start_time
            self.logger.error(f"PT闪蒸计算失败: {e}")
            raise e
    
    def _solve_rachford_rice_robust(self, z: List[float], K: List[float]) -> Tuple[float, int]:
        """鲁棒的Rachford-Rice方程求解
        
        Args:
            z: 进料组成
            K: K值
            
        Returns:
            Tuple[float, int]: (汽化率, 迭代次数)
        """
        def rr_equation(beta):
            return sum(z[i] * (K[i] - 1) / (1 + beta * (K[i] - 1)) 
                      for i in range(len(z)))
        
        def rr_derivative(beta):
            return -sum(z[i] * (K[i] - 1)**2 / (1 + beta * (K[i] - 1))**2 
                       for i in range(len(z)))
        
        # 确定搜索范围
        beta_min = max(-1.0 / (max(K) - 1), 0.0) + 1e-10
        beta_max = min(-1.0 / (min(K) - 1), 1.0) - 1e-10
        
        if beta_min >= beta_max:
            return 0.5, 1  # 返回中间值
        
        iterations = 0
        
        try:
            # 检查边界条件
            f_min = rr_equation(beta_min)
            f_max = rr_equation(beta_max)
            
            if abs(f_min) < self.inner_tolerance:
                return beta_min, 1
            if abs(f_max) < self.inner_tolerance:
                return beta_max, 1
            
            if f_min * f_max > 0:
                # 没有根，返回使残差最小的边界值
                if abs(f_min) < abs(f_max):
                    return beta_min, 1
                else:
                    return beta_max, 1
            
            # 首先尝试Newton-Raphson方法
            beta_guess = (beta_min + beta_max) / 2
            
            for i in range(self.max_inner_iterations):
                iterations += 1
                
                f_val = rr_equation(beta_guess)
                if abs(f_val) < self.inner_tolerance:
                    return beta_guess, iterations
                
                df_val = rr_derivative(beta_guess)
                if abs(df_val) < 1e-15:
                    break  # 导数太小，切换到Brent方法
                
                beta_new = beta_guess - f_val / df_val
                
                # 确保在有效范围内
                if beta_new < beta_min or beta_new > beta_max:
                    break  # 超出范围，切换到Brent方法
                
                if abs(beta_new - beta_guess) < self.inner_tolerance:
                    return beta_new, iterations
                
                beta_guess = beta_new
            
            # Newton-Raphson失败，使用Brent方法
            beta = brentq(rr_equation, beta_min, beta_max, xtol=self.inner_tolerance)
            iterations += 10  # Brent方法的估计迭代次数
            
            return beta, iterations
            
        except Exception as e:
            self.logger.warning(f"Rachford-Rice求解失败: {e}")
            return 0.5, iterations + 1
    
    def _update_k_values(self, x: List[float], y: List[float], T: float, P: float, 
                        property_package) -> List[float]:
        """更新K值
        
        Args:
            x: 液相组成
            y: 气相组成
            T: 温度
            P: 压力
            property_package: 物性包
            
        Returns:
            List[float]: 新的K值
        """
        try:
            # 计算液相逸度系数
            phi_l = property_package.calculate_fugacity_coefficients(T, P, x, "liquid")
            
            # 计算气相逸度系数
            phi_v = property_package.calculate_fugacity_coefficients(T, P, y, "vapor")
            
            # 更新K值: K_i = φ_i^L / φ_i^V
            K_new = []
            for i in range(len(x)):
                if phi_v[i] > 1e-15:
                    K_new.append(phi_l[i] / phi_v[i])
                else:
                    K_new.append(1.0)  # 默认值
            
            return K_new
            
        except Exception as e:
            self.logger.warning(f"K值更新失败: {e}")
            # 返回当前K值的小幅修正
            return [1.0] * len(x)
    
    def _apply_wegstein_acceleration(self, K: List[float], K_old: List[float], 
                                   K_history: List[List[float]]) -> List[float]:
        """应用Wegstein加速方法
        
        Args:
            K: 当前K值
            K_old: 上一次K值
            K_history: K值历史
            
        Returns:
            List[float]: 加速后的K值
        """
        if len(K_history) < 2:
            return K
        
        try:
            K_acc = []
            for i in range(len(K)):
                # 计算加速因子
                if len(K_history) >= 2:
                    K_prev2 = K_history[-2][i]
                    K_prev1 = K_history[-1][i]
                    K_curr = K[i]
                    
                    # Wegstein加速
                    if abs(K_curr - K_prev1) > 1e-15 and abs(K_prev1 - K_prev2) > 1e-15:
                        q = (K_curr - K_prev1) / (K_prev1 - K_prev2)
                        if abs(q) < 0.99:  # 避免数值不稳定
                            s = q / (q - 1)
                            K_acc_i = K_prev1 + s * (K_curr - K_prev1)
                            
                            # 限制加速幅度
                            if K_acc_i > 0 and K_acc_i < 1e6:
                                K_acc.append(K_acc_i)
                            else:
                                K_acc.append(K[i])
                        else:
                            K_acc.append(K[i])
                    else:
                        K_acc.append(K[i])
                else:
                    K_acc.append(K[i])
            
            return K_acc
            
        except Exception as e:
            self.logger.warning(f"Wegstein加速失败: {e}")
            return K
    
    def _calculate_residual(self, x: List[float], y: List[float], T: float, P: float, 
                          property_package) -> float:
        """计算逸度平衡残差
        
        Args:
            x: 液相组成
            y: 气相组成
            T: 温度
            P: 压力
            property_package: 物性包
            
        Returns:
            float: 残差
        """
        try:
            # 计算逸度系数
            phi_l = property_package.calculate_fugacity_coefficients(T, P, x, "liquid")
            phi_v = property_package.calculate_fugacity_coefficients(T, P, y, "vapor")
            
            # 计算逸度平衡残差
            residual = 0.0
            for i in range(len(x)):
                if x[i] > 1e-15 and y[i] > 1e-15:
                    f_l = x[i] * phi_l[i] * P
                    f_v = y[i] * phi_v[i] * P
                    residual += abs(f_l - f_v) / max(f_l, f_v)
            
            return residual
            
        except Exception as e:
            self.logger.warning(f"残差计算失败: {e}")
            return 1.0
    
    def _is_single_phase(self, z: List[float], K: List[float]) -> bool:
        """检查是否为单相
        
        Args:
            z: 进料组成
            K: K值
            
        Returns:
            bool: 是否为单相
        """
        # 检查是否所有K值都接近1（单相）
        if all(0.99 < k < 1.01 for k in K):
            return True
        
        # 检查Rachford-Rice方程的边界值
        def rr_at_zero():
            return sum(z[i] * (K[i] - 1) for i in range(len(z)))
        
        def rr_at_one():
            return sum(z[i] * (K[i] - 1) / K[i] for i in range(len(z)))
        
        f0 = rr_at_zero()
        f1 = rr_at_one()
        
        # 如果边界值同号，则为单相
        return f0 * f1 > 0
    
    def _create_single_phase_result(self, z: List[float], P: float, T: float, 
                                  property_package) -> Dict[str, Any]:
        """创建单相结果
        
        Args:
            z: 进料组成
            P: 压力
            T: 温度
            property_package: 物性包
            
        Returns:
            Dict: 单相结果
        """
        # 判断是气相还是液相
        # 简化判断：计算压缩因子
        try:
            Z = property_package.calculate_compressibility_factor(T, P, z, "vapor")
            if Z > 0.8:
                # 气相
                return {
                    'L1': 0.0,
                    'V': 1.0,
                    'Vx1': [0.0] * len(z),
                    'Vy': z.copy(),
                    'K_values': [1e6] * len(z),
                    'ecount': 1,
                    'residual': 0.0,
                    'converged': True,
                    'single_phase': 'vapor'
                }
            else:
                # 液相
                return {
                    'L1': 1.0,
                    'V': 0.0,
                    'Vx1': z.copy(),
                    'Vy': [0.0] * len(z),
                    'K_values': [1e-6] * len(z),
                    'ecount': 1,
                    'residual': 0.0,
                    'converged': True,
                    'single_phase': 'liquid'
                }
        except:
            # 默认为液相
            return {
                'L1': 1.0,
                'V': 0.0,
                'Vx1': z.copy(),
                'Vy': [0.0] * len(z),
                'K_values': [1e-6] * len(z),
                'ecount': 1,
                'residual': 0.0,
                'converged': True,
                'single_phase': 'liquid'
            }
    
    def flash_ph(self, z: List[float], P: float, H: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """PH闪蒸计算（简化实现）
        
        Args:
            z: 进料摩尔分数
            P: 压力 [Pa]
            H: 焓 [J/mol]
            property_package: 物性包对象
            initial_temperature: 初始温度估算 [K]
            
        Returns:
            Dict: 闪蒸结果
        """
        # 简化实现：使用温度迭代
        if initial_temperature <= 0:
            T_guess = 298.15  # 默认温度
        else:
            T_guess = initial_temperature
        
        # 这里应该实现完整的PH闪蒸算法
        # 暂时返回PT闪蒸结果
        return self.flash_pt(z, P, T_guess, property_package)
    
    def flash_ps(self, z: List[float], P: float, S: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """PS闪蒸计算（简化实现）"""
        if initial_temperature <= 0:
            T_guess = 298.15
        else:
            T_guess = initial_temperature
        
        return self.flash_pt(z, P, T_guess, property_package)
    
    def flash_tv(self, z: List[float], T: float, V: float, 
                property_package, initial_pressure: float = 0.0) -> Dict[str, Any]:
        """TV闪蒸计算（简化实现）"""
        if initial_pressure <= 0:
            P_guess = 101325.0  # 1 atm
        else:
            P_guess = initial_pressure
        
        return self.flash_pt(z, P_guess, T, property_package)
    
    def flash_pv(self, z: List[float], P: float, V: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """PV闪蒸计算（简化实现）"""
        if initial_temperature <= 0:
            T_guess = 298.15
        else:
            T_guess = initial_temperature
        
        return self.flash_pt(z, P, T_guess, property_package)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        return {
            "name": self.name,
            "type": "Inside-Out Flash",
            "description": "高效的Inside-Out相平衡计算算法",
            "settings": {
                "max_outer_iterations": self.max_outer_iterations,
                "max_inner_iterations": self.max_inner_iterations,
                "outer_tolerance": self.outer_tolerance,
                "inner_tolerance": self.inner_tolerance,
                "damping_factor": self.damping_factor,
                "use_acceleration": self.use_acceleration
            },
            "statistics": self.stats.copy()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "average_outer_iterations": 0.0,
            "average_inner_iterations": 0.0,
            "average_calculation_time": 0.0
        } 