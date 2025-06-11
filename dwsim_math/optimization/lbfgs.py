"""
L-BFGS优化算法模块
==================

实现Limited-memory BFGS算法，用于求解无约束非线性优化问题。
这些功能是从DWSIM.Math/LBFGS.vb转换而来的。

L-BFGS算法是BFGS算法的内存优化版本，适用于大规模优化问题。

主要功能:
- L-BFGS算法核心实现
- 线搜索算法
- 收敛判断

作者: DWSIM团队 (Python转换版本)
许可证: GNU General Public License v3.0
"""

import numpy as np
from typing import Callable, Tuple, Optional, Any, List
import warnings
import math


class LBFGS:
    """
    L-BFGS优化算法类
    
    Limited-memory BFGS是一种拟牛顿法，用于求解无约束优化问题：
    min f(x)
    
    算法特点：
    - 存储需求：O(mn)，其中m是存储的历史信息数
    - 收敛速度：超线性收敛
    - 适用于大规模问题
    """
    
    def __init__(self, 
                 m: int = 7,
                 eps_g: float = 1e-6,
                 eps_f: float = 1e-12,
                 eps_x: float = 1e-12,
                 max_iterations: int = 1000):
        """
        初始化L-BFGS优化器
        
        参数:
            m: 存储的历史信息数量，推荐3-7
            eps_g: 梯度收敛容差
            eps_f: 函数值收敛容差
            eps_x: 变量收敛容差
            max_iterations: 最大迭代次数
        """
        self.m = max(1, min(m, 20))  # 限制m的范围
        self.eps_g = eps_g
        self.eps_f = eps_f
        self.eps_x = eps_x
        self.max_iterations = max_iterations
        
        # 存储历史信息
        self.s_history = []  # x的差分历史
        self.y_history = []  # 梯度的差分历史
        self.rho_history = []  # 1/(y^T s)的历史
        
        # 迭代信息
        self.iteration_count = 0
        self.function_evaluations = 0
        self.gradient_evaluations = 0
        
        # 回调函数
        self.func_grad_callback = None
        self.iteration_callback = None
    
    def set_function_gradient_callback(self, callback: Callable[[np.ndarray], Tuple[float, np.ndarray]]):
        """
        设置函数值和梯度计算回调
        
        参数:
            callback: 函数，输入x，返回(f(x), grad_f(x))
        """
        self.func_grad_callback = callback
    
    def set_iteration_callback(self, callback: Callable[[np.ndarray, float, np.ndarray], bool]):
        """
        设置迭代回调函数
        
        参数:
            callback: 函数，输入(x, f, g)，返回是否继续迭代
        """
        self.iteration_callback = callback
    
    def minimize(self, 
                objective: Callable[[np.ndarray], float],
                gradient: Callable[[np.ndarray], np.ndarray],
                x0: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> dict:
        """
        使用L-BFGS算法最小化函数
        
        参数:
            objective: 目标函数，输入x，返回f(x)
            gradient: 梯度函数，输入x，返回grad_f(x)
            x0: 初始点
            bounds: 变量边界约束（可选）
            
        返回:
            dict: 优化结果
                - x: 最优解
                - fun: 最优函数值
                - jac: 最优点梯度
                - nit: 迭代次数
                - nfev: 函数评估次数
                - success: 是否成功
                - message: 状态信息
        """
        # 创建组合函数
        def func_grad(x):
            return objective(x), gradient(x)
        
        self.func_grad_callback = func_grad
        
        # 初始化
        x = np.array(x0, dtype=float).copy()
        n = len(x)
        
        # 清空历史信息
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()
        
        self.iteration_count = 0
        self.function_evaluations = 0
        self.gradient_evaluations = 0
        
        # 计算初始函数值和梯度
        f, g = self._evaluate_function_gradient(x)
        
        # 检查初始梯度
        g_norm = np.linalg.norm(g)
        if g_norm <= self.eps_g:
            return self._create_result(x, f, g, 4, "梯度范数满足收敛条件")
        
        # 主迭代循环
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            
            # 调用迭代回调
            if self.iteration_callback:
                if not self.iteration_callback(x, f, g):
                    return self._create_result(x, f, g, 0, "用户中断")
            
            # 计算搜索方向
            d = self._compute_search_direction(g)
            
            # 线搜索
            alpha, f_new, g_new, ls_info = self._line_search(x, f, g, d)
            
            if alpha is None:
                return self._create_result(x, f, g, -1, "线搜索失败")
            
            # 更新变量
            x_new = x + alpha * d
            
            # 检查收敛条件
            converged, message = self._check_convergence(x, x_new, f, f_new, g_new)
            
            if converged:
                return self._create_result(x_new, f_new, g_new, 
                                         self._get_convergence_code(message), message)
            
            # 更新L-BFGS历史信息
            self._update_history(x_new - x, g_new - g)
            
            # 更新当前点
            x = x_new
            f = f_new
            g = g_new
        
        # 达到最大迭代次数
        return self._create_result(x, f, g, 5, "达到最大迭代次数")
    
    def _evaluate_function_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算函数值和梯度"""
        self.function_evaluations += 1
        self.gradient_evaluations += 1
        return self.func_grad_callback(x)
    
    def _compute_search_direction(self, g: np.ndarray) -> np.ndarray:
        """
        计算L-BFGS搜索方向
        
        使用两环递推算法计算 H_k * g，其中H_k是Hessian逆矩阵的近似
        """
        q = g.copy()
        alpha_list = []
        
        # 第一个循环：从最近的历史开始
        for i in range(len(self.s_history) - 1, -1, -1):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]
            
            alpha_i = rho_i * np.dot(s_i, q)
            alpha_list.append(alpha_i)
            q = q - alpha_i * y_i
        
        # 计算初始Hessian逆矩阵近似的乘积
        r = self._apply_initial_hessian_inverse(q)
        
        # 第二个循环：按相反顺序
        alpha_list.reverse()
        for i in range(len(self.s_history)):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]
            
            beta_i = rho_i * np.dot(y_i, r)
            r = r + (alpha_list[i] - beta_i) * s_i
        
        return -r
    
    def _apply_initial_hessian_inverse(self, q: np.ndarray) -> np.ndarray:
        """
        应用初始Hessian逆矩阵近似
        
        使用最近的s和y向量估计标量因子
        """
        if len(self.s_history) == 0:
            return q  # 使用单位矩阵
        
        # 使用最近的历史信息估计标量
        s_k = self.s_history[-1]
        y_k = self.y_history[-1]
        
        gamma_k = np.dot(s_k, y_k) / np.dot(y_k, y_k)
        
        return gamma_k * q
    
    def _line_search(self, 
                    x: np.ndarray, 
                    f: float, 
                    g: np.ndarray, 
                    d: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], str]:
        """
        Moré-Thuente线搜索算法
        
        寻找满足强Wolfe条件的步长
        """
        # 线搜索参数
        ftol = 1e-4  # 充分减少条件参数
        gtol = 0.9   # 曲率条件参数
        xtol = 1e-12 # x容差
        stpmin = 1e-12
        stpmax = 1e6
        maxfev = 20
        
        # 初始步长
        stp = 1.0
        
        # 初始搜索方向的梯度
        dginit = np.dot(g, d)
        
        if dginit >= 0:
            warnings.warn("搜索方向不是下降方向")
            return None, None, None, "搜索方向错误"
        
        # 线搜索主循环
        for nfev in range(maxfev):
            # 计算试探点
            x_trial = x + stp * d
            f_trial, g_trial = self._evaluate_function_gradient(x_trial)
            
            # 检查充分减少条件（Armijo条件）
            if f_trial > f + ftol * stp * dginit:
                # 步长太大，减小步长
                stpmax = stp
                stp = 0.5 * stp
                continue
            
            # 计算试探点的方向导数
            dg_trial = np.dot(g_trial, d)
            
            # 检查曲率条件
            if abs(dg_trial) <= -gtol * dginit:
                # 满足强Wolfe条件
                return stp, f_trial, g_trial, "成功"
            
            if dg_trial >= 0:
                # 找到了局部最小值区间
                break
            
            # 继续增大步长
            stpmin = stp
            stp = min(2.0 * stp, stpmax)
            
            if stp >= stpmax:
                break
        
        # 如果没有找到满足强Wolfe条件的点，返回当前最好的点
        x_trial = x + stp * d
        f_trial, g_trial = self._evaluate_function_gradient(x_trial)
        
        return stp, f_trial, g_trial, "近似满足"
    
    def _update_history(self, s: np.ndarray, y: np.ndarray):
        """
        更新L-BFGS历史信息
        
        参数:
            s: x的差分向量
            y: 梯度的差分向量
        """
        # 检查曲率条件
        sy = np.dot(s, y)
        if sy <= 1e-12:
            warnings.warn("跳过更新：曲率条件不满足")
            return
        
        # 计算rho
        rho = 1.0 / sy
        
        # 添加到历史
        self.s_history.append(s.copy())
        self.y_history.append(y.copy())
        self.rho_history.append(rho)
        
        # 保持历史长度
        if len(self.s_history) > self.m:
            self.s_history.pop(0)
            self.y_history.pop(0)
            self.rho_history.pop(0)
    
    def _check_convergence(self, 
                          x: np.ndarray, 
                          x_new: np.ndarray, 
                          f: float, 
                          f_new: float, 
                          g_new: np.ndarray) -> Tuple[bool, str]:
        """检查收敛条件"""
        # 梯度范数条件
        g_norm = np.linalg.norm(g_new)
        if g_norm <= self.eps_g:
            return True, "梯度范数收敛"
        
        # 函数值相对变化条件
        if abs(f_new - f) <= self.eps_f * max(abs(f), abs(f_new), 1.0):
            return True, "函数值收敛"
        
        # 变量相对变化条件
        x_diff = np.linalg.norm(x_new - x)
        if x_diff <= self.eps_x:
            return True, "变量收敛"
        
        return False, ""
    
    def _get_convergence_code(self, message: str) -> int:
        """获取收敛代码"""
        if "函数值" in message:
            return 1
        elif "变量" in message:
            return 2
        elif "梯度" in message:
            return 4
        else:
            return 0
    
    def _create_result(self, x: np.ndarray, f: float, g: np.ndarray, 
                      code: int, message: str) -> dict:
        """创建优化结果字典"""
        return {
            'x': x.copy(),
            'fun': f,
            'jac': g.copy(),
            'nit': self.iteration_count,
            'nfev': self.function_evaluations,
            'success': code > 0,
            'status': code,
            'message': message
        }


# 便捷函数
def lbfgs_minimize(func_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
                  x0: np.ndarray,
                  m: int = 7,
                  eps_g: float = 1e-6,
                  max_iter: int = 1000) -> dict:
    """
    L-BFGS优化的便捷函数
    
    参数:
        func_grad: 函数，输入x，返回(f(x), grad_f(x))
        x0: 初始点
        m: 存储的历史信息数
        eps_g: 梯度收敛容差
        max_iter: 最大迭代次数
        
    返回:
        dict: 优化结果
    """
    optimizer = LBFGS(m=m, eps_g=eps_g, max_iterations=max_iter)
    return optimizer.minimize(func_grad, x0) 