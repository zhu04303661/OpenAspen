"""
嵌套循环闪蒸算法
===============

基于DWSIM NestedLoops.vb的Python实现
实现PT、PH、PS、TV、PV等多种闪蒸规格的计算
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy.optimize import brentq, minimize_scalar
import warnings

from .base_flash import FlashAlgorithmBase, FlashConvergenceError, FlashSettings
from ..core.enums import FlashSpec, PhaseType


class NestedLoopsFlash(FlashAlgorithmBase):
    """
    嵌套循环闪蒸算法
    
    外循环：更新K值
    内循环：求解Rachford-Rice方程
    """
    
    def __init__(self, name: str = "Nested Loops Flash"):
        super().__init__(name)
        self.logger = logging.getLogger("NestedLoopsFlash")
        
    def flash_pt(self, z: List[float], P: float, T: float, 
                property_package, initial_k_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        PT闪蒸计算
        
        Args:
            z: 进料组成
            P: 压力 (Pa)
            T: 温度 (K)
            property_package: 物性包
            initial_k_values: 初始K值估算
            
        Returns:
            Dict: 闪蒸结果 {L1, V, Vx1, Vy, ecount, L2, Vx2, S, Vs, residual}
        """
        self.write_debug_info(f"开始PT闪蒸计算: P={P:.1f} Pa, T={T:.2f} K")
        
        n_comp = len(z)
        
        # 初始化K值
        if initial_k_values is None:
            K = self.estimate_initial_k_values(z, P, T, property_package)
        else:
            K = initial_k_values.copy()
        
        self.write_debug_info(f"初始K值: {[f'{k:.4f}' for k in K]}")
        
        # 检查是否为单相
        if self._is_single_phase(z, K):
            return self._single_phase_result(z, T, P, property_package)
        
        # 外循环参数
        max_external_iter = self.settings.pt_max_external_iterations
        external_tol = self.settings.pt_external_tolerance
        max_internal_iter = self.settings.pt_max_internal_iterations
        internal_tol = self.settings.pt_internal_tolerance
        
        # 初始化变量
        x = z.copy()  # 液相组成
        y = z.copy()  # 气相组成
        beta = 0.5    # 气化率
        
        converged = False
        external_iter = 0
        total_residual = 1.0
        
        # 外循环：K值迭代
        for external_iter in range(max_external_iter):
            K_old = K.copy()
            
            # 内循环：求解Rachford-Rice方程
            try:
                beta = self._solve_rachford_rice_robust(z, K, internal_tol, max_internal_iter)
                
                # 更新相组成
                x, y = self.update_phase_compositions(z, K, beta)
                
                # 计算新的K值
                K_new = self._calculate_k_values(x, y, T, P, property_package)
                
                # 检查收敛
                if self.check_convergence(K_old, K_new, external_tol):
                    converged = True
                    K = K_new
                    break
                
                # 更新K值（可选择阻尼）
                if self.settings.nested_loops_fast_mode:
                    K = K_new
                else:
                    # 使用阻尼因子
                    damping = 0.7
                    K = [damping * k_new + (1 - damping) * k_old 
                         for k_old, k_new in zip(K_old, K_new)]
                
                # 计算残差
                total_residual = sum(abs(k_new - k_old) / max(k_old, 1e-10) 
                                   for k_old, k_new in zip(K_old, K_new))
                
                self.write_debug_info(f"外循环 {external_iter + 1}: β={beta:.6f}, "
                                    f"残差={total_residual:.2e}")
                
            except Exception as e:
                self.logger.warning(f"内循环求解失败: {e}")
                # 尝试调整K值
                K = self._adjust_k_values(K, 0.1)
                continue
        
        if not converged:
            raise FlashConvergenceError(
                f"PT闪蒸未收敛，迭代{external_iter + 1}次，残差={total_residual:.2e}",
                external_iter + 1, total_residual
            )
        
        self.write_debug_info(f"PT闪蒸收敛: {external_iter + 1}次迭代, β={beta:.6f}")
        
        # 返回结果
        return {
            'L1': 1.0 - beta,      # 液相摩尔分数
            'V': beta,             # 气相摩尔分数
            'Vx1': x,              # 液相组成
            'Vy': y,               # 气相组成
            'ecount': external_iter + 1,  # 迭代次数
            'L2': 0.0,             # 第二液相摩尔分数
            'Vx2': [0.0] * n_comp, # 第二液相组成
            'S': 0.0,              # 固相摩尔分数
            'Vs': [0.0] * n_comp,  # 固相组成
            'residual': total_residual,
            'K_values': K
        }
    
    def flash_ph(self, z: List[float], P: float, H: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """
        PH闪蒸计算
        
        Args:
            z: 进料组成
            P: 压力 (Pa)
            H: 焓 (J/mol)
            property_package: 物性包
            initial_temperature: 初始温度估算 (K)
            
        Returns:
            Dict: 闪蒸结果
        """
        self.write_debug_info(f"开始PH闪蒸计算: P={P:.1f} Pa, H={H:.1f} J/mol")
        
        # 温度估算
        if initial_temperature <= 0:
            T_guess = self._estimate_temperature_from_enthalpy(z, P, H, property_package)
        else:
            T_guess = initial_temperature
        
        self.write_debug_info(f"初始温度估算: {T_guess:.2f} K")
        
        # 外循环参数
        max_external_iter = self.settings.ph_max_external_iterations
        external_tol = self.settings.ph_external_tolerance
        max_temp_change = self.settings.pv_max_temperature_change
        
        T = T_guess
        converged = False
        
        for external_iter in range(max_external_iter):
            try:
                # 在当前温度下进行PT闪蒸
                pt_result = self.flash_pt(z, P, T, property_package)
                
                # 计算当前焓
                H_calc = self._calculate_mixture_enthalpy_from_result(
                    pt_result, T, P, property_package)
                
                # 检查焓平衡
                H_error = abs(H_calc - H)
                if H_error < external_tol * abs(H):
                    converged = True
                    break
                
                # 计算温度导数
                dH_dT = self._calculate_enthalpy_derivative(
                    z, P, T, property_package, pt_result)
                
                if abs(dH_dT) < 1e-10:
                    raise FlashConvergenceError("焓对温度的导数为零", external_iter + 1, H_error)
                
                # Newton-Raphson更新温度
                dT = -(H_calc - H) / dH_dT
                
                # 限制温度变化
                dT = max(-max_temp_change, min(max_temp_change, dT))
                T_new = T + dT
                
                # 确保温度在合理范围内
                T_new = max(200.0, min(2000.0, T_new))
                
                self.write_debug_info(f"PH外循环 {external_iter + 1}: T={T:.2f} K, "
                                    f"H_calc={H_calc:.1f}, H_error={H_error:.2e}, dT={dT:.2f}")
                
                T = T_new
                
            except Exception as e:
                self.logger.warning(f"PH闪蒸迭代失败: {e}")
                T += 10.0  # 尝试提高温度
                continue
        
        if not converged:
            raise FlashConvergenceError(
                f"PH闪蒸未收敛，迭代{external_iter + 1}次",
                external_iter + 1, H_error
            )
        
        # 更新结果中的温度
        pt_result['calculated_temperature'] = T
        pt_result['calculated_enthalpy'] = H
        
        return pt_result
    
    def flash_ps(self, z: List[float], P: float, S: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """
        PS闪蒸计算
        
        Args:
            z: 进料组成
            P: 压力 (Pa)
            S: 熵 (J/mol/K)
            property_package: 物性包
            initial_temperature: 初始温度估算 (K)
            
        Returns:
            Dict: 闪蒸结果
        """
        self.write_debug_info(f"开始PS闪蒸计算: P={P:.1f} Pa, S={S:.3f} J/mol/K")
        
        # 温度估算
        if initial_temperature <= 0:
            T_guess = self._estimate_temperature_from_entropy(z, P, S, property_package)
        else:
            T_guess = initial_temperature
        
        # 外循环参数
        max_external_iter = self.settings.ps_max_external_iterations
        external_tol = self.settings.ps_external_tolerance
        max_temp_change = self.settings.pv_max_temperature_change
        
        T = T_guess
        converged = False
        
        for external_iter in range(max_external_iter):
            try:
                # 在当前温度下进行PT闪蒸
                pt_result = self.flash_pt(z, P, T, property_package)
                
                # 计算当前熵
                S_calc = self._calculate_mixture_entropy_from_result(
                    pt_result, T, P, property_package)
                
                # 检查熵平衡
                S_error = abs(S_calc - S)
                if S_error < external_tol * abs(S):
                    converged = True
                    break
                
                # 计算温度导数
                dS_dT = self._calculate_entropy_derivative(
                    z, P, T, property_package, pt_result)
                
                if abs(dS_dT) < 1e-10:
                    raise FlashConvergenceError("熵对温度的导数为零", external_iter + 1, S_error)
                
                # Newton-Raphson更新温度
                dT = -(S_calc - S) / dS_dT
                dT = max(-max_temp_change, min(max_temp_change, dT))
                T_new = T + dT
                T_new = max(200.0, min(2000.0, T_new))
                
                self.write_debug_info(f"PS外循环 {external_iter + 1}: T={T:.2f} K, "
                                    f"S_calc={S_calc:.3f}, S_error={S_error:.2e}")
                
                T = T_new
                
            except Exception as e:
                self.logger.warning(f"PS闪蒸迭代失败: {e}")
                T += 10.0
                continue
        
        if not converged:
            raise FlashConvergenceError(
                f"PS闪蒸未收敛，迭代{external_iter + 1}次",
                external_iter + 1, S_error
            )
        
        pt_result['calculated_temperature'] = T
        pt_result['calculated_entropy'] = S
        
        return pt_result
    
    def flash_tv(self, z: List[float], T: float, V: float, 
                property_package, initial_pressure: float = 0.0) -> Dict[str, Any]:
        """
        TV闪蒸计算
        
        Args:
            z: 进料组成
            T: 温度 (K)
            V: 摩尔体积 (m³/mol)
            property_package: 物性包
            initial_pressure: 初始压力估算 (Pa)
            
        Returns:
            Dict: 闪蒸结果
        """
        self.write_debug_info(f"开始TV闪蒸计算: T={T:.2f} K, V={V:.6f} m³/mol")
        
        # 压力估算
        if initial_pressure <= 0:
            P_guess = 101325.0  # 1 atm
        else:
            P_guess = initial_pressure
        
        # 使用数值方法求解
        def objective(P):
            try:
                pt_result = self.flash_pt(z, P, T, property_package)
                V_calc = self._calculate_mixture_volume_from_result(
                    pt_result, T, P, property_package)
                return (V_calc - V) ** 2
            except:
                return 1e10
        
        # 优化求解
        result = minimize_scalar(objective, bounds=(1000, 1e8), method='bounded')
        
        if not result.success:
            raise FlashConvergenceError("TV闪蒸压力求解失败", 0, result.fun)
        
        P_solution = result.x
        pt_result = self.flash_pt(z, P_solution, T, property_package)
        pt_result['calculated_pressure'] = P_solution
        pt_result['calculated_volume'] = V
        
        return pt_result
    
    def flash_pv(self, z: List[float], P: float, V: float, 
                property_package, initial_temperature: float = 0.0) -> Dict[str, Any]:
        """
        PV闪蒸计算
        
        Args:
            z: 进料组成
            P: 压力 (Pa)
            V: 摩尔体积 (m³/mol)
            property_package: 物性包
            initial_temperature: 初始温度估算 (K)
            
        Returns:
            Dict: 闪蒸结果
        """
        self.write_debug_info(f"开始PV闪蒸计算: P={P:.1f} Pa, V={V:.6f} m³/mol")
        
        # 温度估算
        if initial_temperature <= 0:
            T_guess = 300.0  # 室温
        else:
            T_guess = initial_temperature
        
        # 使用数值方法求解
        def objective(T):
            try:
                pt_result = self.flash_pt(z, P, T, property_package)
                V_calc = self._calculate_mixture_volume_from_result(
                    pt_result, T, P, property_package)
                return (V_calc - V) ** 2
            except:
                return 1e10
        
        # 优化求解
        result = minimize_scalar(objective, bounds=(200, 2000), method='bounded')
        
        if not result.success:
            raise FlashConvergenceError("PV闪蒸温度求解失败", 0, result.fun)
        
        T_solution = result.x
        pt_result = self.flash_pt(z, P, T_solution, property_package)
        pt_result['calculated_temperature'] = T_solution
        pt_result['calculated_volume'] = V
        
        return pt_result
    
    def _solve_rachford_rice_robust(self, z: List[float], K: List[float], 
                                  tolerance: float, max_iterations: int) -> float:
        """
        鲁棒的Rachford-Rice方程求解器
        """
        def rr_equation(beta):
            return sum(z[i] * (K[i] - 1) / (1 + beta * (K[i] - 1)) 
                      for i in range(len(z)))
        
        def rr_derivative(beta):
            return -sum(z[i] * (K[i] - 1)**2 / (1 + beta * (K[i] - 1))**2 
                       for i in range(len(z)))
        
        # 确定求解区间
        beta_min = max(0.0, max(-(1/(K[i] - 1)) for i in range(len(z)) if K[i] != 1.0) + 1e-10)
        beta_max = min(1.0, min(-(1/(K[i] - 1)) for i in range(len(z)) if K[i] != 1.0) - 1e-10)
        
        if beta_min >= beta_max:
            # 单相情况
            return 0.0 if abs(rr_equation(0.0)) < abs(rr_equation(1.0)) else 1.0
        
        # 检查边界值
        f_min = rr_equation(beta_min)
        f_max = rr_equation(beta_max)
        
        if abs(f_min) < tolerance:
            return beta_min
        if abs(f_max) < tolerance:
            return beta_max
        
        if f_min * f_max > 0:
            # 没有根，返回使函数值最小的边界
            return beta_min if abs(f_min) < abs(f_max) else beta_max
        
        # 使用Brent方法求解
        try:
            beta = brentq(rr_equation, beta_min, beta_max, xtol=tolerance, maxiter=max_iterations)
            return beta
        except:
            # 如果Brent方法失败，使用Newton-Raphson
            beta = (beta_min + beta_max) / 2
            for _ in range(max_iterations):
                f = rr_equation(beta)
                if abs(f) < tolerance:
                    break
                df = rr_derivative(beta)
                if abs(df) < 1e-15:
                    break
                beta_new = beta - f / df
                beta_new = max(beta_min, min(beta_max, beta_new))
                if abs(beta_new - beta) < tolerance:
                    break
                beta = beta_new
            
            return beta
    
    def _calculate_k_values(self, x: List[float], y: List[float], T: float, P: float, 
                          property_package) -> List[float]:
        """计算K值"""
        try:
            # 计算逸度系数
            phi_l = property_package.calculate_fugacity_coefficients(T, P, x, "liquid")
            phi_v = property_package.calculate_fugacity_coefficients(T, P, y, "vapor")
            
            # K = (phi_l / phi_v) * (x / y) 的理论值应该是 phi_l / phi_v
            # 但实际计算中我们直接用 y / x 来更新K值
            K = []
            for i in range(len(x)):
                if x[i] > 1e-15:
                    k_val = y[i] / x[i]
                    # 应用逸度系数校正
                    if len(phi_l) > i and len(phi_v) > i:
                        k_val *= phi_l[i] / phi_v[i]
                    K.append(max(k_val, 1e-10))
                else:
                    K.append(1.0)
            
            return K
            
        except Exception as e:
            self.logger.warning(f"K值计算失败，使用简化方法: {e}")
            # 使用简化的K值计算
            return [max(y[i] / x[i] if x[i] > 1e-15 else 1.0, 1e-10) for i in range(len(x))]
    
    def _is_single_phase(self, z: List[float], K: List[float]) -> bool:
        """检查是否为单相"""
        # 检查是否所有K值都接近1
        if all(0.99 < k < 1.01 for k in K):
            return True
        
        # 检查Rachford-Rice方程的边界值
        try:
            f_0 = sum(z[i] * (K[i] - 1) / (1 + 0 * (K[i] - 1)) for i in range(len(z)))
            f_1 = sum(z[i] * (K[i] - 1) / (1 + 1 * (K[i] - 1)) for i in range(len(z)))
            
            # 如果边界值同号，则为单相
            return f_0 * f_1 > 0
        except:
            return False
    
    def _single_phase_result(self, z: List[float], T: float, P: float, 
                           property_package) -> Dict[str, Any]:
        """返回单相结果"""
        n_comp = len(z)
        
        # 判断是液相还是气相
        # 简化判断：计算平均沸点
        try:
            avg_bp = sum(z[i] * property_package.compounds[i].normal_boiling_point 
                        for i in range(n_comp) if hasattr(property_package.compounds[i], 'normal_boiling_point'))
            is_vapor = T > avg_bp
        except:
            is_vapor = T > 373.15  # 默认以水的沸点为准
        
        if is_vapor:
            return {
                'L1': 0.0, 'V': 1.0, 'Vx1': [0.0] * n_comp, 'Vy': z,
                'ecount': 1, 'L2': 0.0, 'Vx2': [0.0] * n_comp,
                'S': 0.0, 'Vs': [0.0] * n_comp, 'residual': 0.0,
                'K_values': [1e10] * n_comp
            }
        else:
            return {
                'L1': 1.0, 'V': 0.0, 'Vx1': z, 'Vy': [0.0] * n_comp,
                'ecount': 1, 'L2': 0.0, 'Vx2': [0.0] * n_comp,
                'S': 0.0, 'Vs': [0.0] * n_comp, 'residual': 0.0,
                'K_values': [1e-10] * n_comp
            }
    
    def _adjust_k_values(self, K: List[float], factor: float) -> List[float]:
        """调整K值以改善收敛性"""
        return [k * (1 + factor * (np.random.random() - 0.5)) for k in K]
    
    def _estimate_temperature_from_enthalpy(self, z: List[float], P: float, H: float, 
                                          property_package) -> float:
        """从焓估算温度"""
        # 简化估算：使用理想气体热容
        try:
            # 假设参考温度298.15K的焓为0
            T_ref = 298.15
            Cp_avg = 30.0  # J/mol/K，平均热容估算
            
            T_guess = T_ref + H / Cp_avg
            return max(200.0, min(2000.0, T_guess))
        except:
            return 400.0  # 默认温度
    
    def _estimate_temperature_from_entropy(self, z: List[float], P: float, S: float, 
                                         property_package) -> float:
        """从熵估算温度"""
        # 简化估算
        return 400.0  # 默认温度
    
    def _calculate_mixture_enthalpy_from_result(self, result: Dict[str, Any], 
                                              T: float, P: float, property_package) -> float:
        """从闪蒸结果计算混合物焓"""
        try:
            total_enthalpy = 0.0
            
            # 液相贡献
            if result['L1'] > 1e-15:
                H_l = property_package.calculate_enthalpy(T, P, result['Vx1'], "liquid")
                total_enthalpy += result['L1'] * H_l
            
            # 气相贡献
            if result['V'] > 1e-15:
                H_v = property_package.calculate_enthalpy(T, P, result['Vy'], "vapor")
                total_enthalpy += result['V'] * H_v
            
            return total_enthalpy
        except:
            return 0.0
    
    def _calculate_mixture_entropy_from_result(self, result: Dict[str, Any], 
                                             T: float, P: float, property_package) -> float:
        """从闪蒸结果计算混合物熵"""
        try:
            total_entropy = 0.0
            
            # 液相贡献
            if result['L1'] > 1e-15:
                S_l = property_package.calculate_entropy(T, P, result['Vx1'], "liquid")
                total_entropy += result['L1'] * S_l
            
            # 气相贡献
            if result['V'] > 1e-15:
                S_v = property_package.calculate_entropy(T, P, result['Vy'], "vapor")
                total_entropy += result['V'] * S_v
            
            return total_entropy
        except:
            return 0.0
    
    def _calculate_mixture_volume_from_result(self, result: Dict[str, Any], 
                                            T: float, P: float, property_package) -> float:
        """从闪蒸结果计算混合物摩尔体积"""
        try:
            total_volume = 0.0
            
            # 液相贡献
            if result['L1'] > 1e-15:
                V_l = property_package.calculate_molar_volume(T, P, result['Vx1'], "liquid")
                total_volume += result['L1'] * V_l
            
            # 气相贡献
            if result['V'] > 1e-15:
                V_v = property_package.calculate_molar_volume(T, P, result['Vy'], "vapor")
                total_volume += result['V'] * V_v
            
            return total_volume
        except:
            return 0.024  # 默认摩尔体积
    
    def _calculate_enthalpy_derivative(self, z: List[float], P: float, T: float, 
                                     property_package, pt_result: Dict[str, Any]) -> float:
        """计算焓对温度的导数"""
        try:
            dT = self.settings.pv_temperature_derivative_epsilon
            
            # 计算T+dT时的焓
            T_plus = T + dT
            pt_result_plus = self.flash_pt(z, P, T_plus, property_package)
            H_plus = self._calculate_mixture_enthalpy_from_result(pt_result_plus, T_plus, P, property_package)
            
            # 计算T-dT时的焓
            T_minus = T - dT
            pt_result_minus = self.flash_pt(z, P, T_minus, property_package)
            H_minus = self._calculate_mixture_enthalpy_from_result(pt_result_minus, T_minus, P, property_package)
            
            # 数值导数
            dH_dT = (H_plus - H_minus) / (2 * dT)
            
            return dH_dT if abs(dH_dT) > 1e-10 else 1000.0  # 默认热容
            
        except:
            return 1000.0  # 默认dH/dT
    
    def _calculate_entropy_derivative(self, z: List[float], P: float, T: float, 
                                    property_package, pt_result: Dict[str, Any]) -> float:
        """计算熵对温度的导数"""
        try:
            dT = self.settings.pv_temperature_derivative_epsilon
            
            # 数值导数计算
            T_plus = T + dT
            pt_result_plus = self.flash_pt(z, P, T_plus, property_package)
            S_plus = self._calculate_mixture_entropy_from_result(pt_result_plus, T_plus, P, property_package)
            
            T_minus = T - dT
            pt_result_minus = self.flash_pt(z, P, T_minus, property_package)
            S_minus = self._calculate_mixture_entropy_from_result(pt_result_minus, T_minus, P, property_package)
            
            dS_dT = (S_plus - S_minus) / (2 * dT)
            
            return dS_dT if abs(dS_dT) > 1e-10 else 10.0  # 默认dS/dT
            
        except:
            return 10.0  # 默认dS/dT 