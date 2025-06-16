"""
三相闪蒸算法 (Three-Phase Flash Algorithm)
实现Boston-Fournier Inside-Out三相闪蒸方法

适用于汽液液三相平衡计算
包含稳定性分析和相分离判断

作者: OpenAspen项目组
版本: 1.0.0
"""

import numpy as np
from scipy.optimize import fsolve, minimize, root
from typing import List, Dict, Optional, Tuple
from .base_flash import FlashAlgorithmBase
from .stability_analysis import StabilityAnalyzer


class ThreePhaseFlash(FlashAlgorithmBase):
    """
    三相闪蒸算法实现
    
    特点:
    - Boston-Fournier Inside-Out方法
    - 自动稳定性分析
    - 汽液液三相处理
    - 临界点处理
    """
    
    def __init__(self, property_package):
        """
        初始化三相闪蒸算法
        
        Parameters:
        -----------
        property_package : PropertyPackage
            物性包对象
        """
        super().__init__(property_package)
        self.stability_analyzer = StabilityAnalyzer(property_package)
        
        # 算法参数
        self.max_iterations = 100
        self.tolerance = 1e-8
        self.stability_tolerance = 1e-6
        
        # 相态标识
        self.VAPOR = 'vapor'
        self.LIQUID1 = 'liquid1'
        self.LIQUID2 = 'liquid2'
        
    def flash_pt_3phase(self, P: float, T: float, z: np.ndarray,
                       initial_guess: Dict = None) -> Dict:
        """
        等温等压三相闪蒸
        
        Parameters:
        -----------
        P : float
            压力 [Pa]
        T : float
            温度 [K]
        z : np.ndarray
            总组成摩尔分数
        initial_guess : Dict, optional
            初值猜测
            
        Returns:
        --------
        Dict
            闪蒸结果字典
        """
        # 输入验证
        if not self._validate_inputs(P, T, z):
            return self._create_error_result("Invalid inputs")
        
        # 稳定性分析
        stability_result = self.stability_analyzer.analyze_stability(P, T, z)
        
        if stability_result['num_phases'] == 1:
            return self._single_phase_result(P, T, z, stability_result['phase'])
        
        elif stability_result['num_phases'] == 2:
            return self._two_phase_flash(P, T, z, initial_guess)
        
        else:  # 三相
            return self._three_phase_flash(P, T, z, initial_guess)
    
    def _three_phase_flash(self, P: float, T: float, z: np.ndarray,
                          initial_guess: Dict = None) -> Dict:
        """
        执行三相闪蒸计算
        """
        nc = len(z)  # 组分数
        
        # 初值设定
        if initial_guess is None:
            x_v, x_l1, x_l2, beta_v, beta_l1 = self._generate_initial_guess(P, T, z)
        else:
            x_v = initial_guess.get('x_vapor', np.ones(nc) / nc)
            x_l1 = initial_guess.get('x_liquid1', np.ones(nc) / nc)
            x_l2 = initial_guess.get('x_liquid2', np.ones(nc) / nc)
            beta_v = initial_guess.get('beta_vapor', 0.3)
            beta_l1 = initial_guess.get('beta_liquid1', 0.3)
        
        beta_l2 = 1.0 - beta_v - beta_l1
        
        # Inside-Out算法主循环
        for iteration in range(self.max_iterations):
            # 计算逸度系数
            phi_v = self._calculate_fugacity_coefficients(x_v, P, T, self.VAPOR)
            phi_l1 = self._calculate_fugacity_coefficients(x_l1, P, T, self.LIQUID1)
            phi_l2 = self._calculate_fugacity_coefficients(x_l2, P, T, self.LIQUID2)
            
            # 计算平衡常数
            K_vl1 = phi_l1 / phi_v  # K值: 气-液1
            K_vl2 = phi_l2 / phi_v  # K值: 气-液2
            K_l1l2 = phi_l2 / phi_l1  # K值: 液1-液2
            
            # Rachford-Rice方程组求解
            try:
                beta_v_new, beta_l1_new = self._solve_rachford_rice_3phase(
                    z, K_vl1, K_vl2, K_l1l2, beta_v, beta_l1
                )
                beta_l2_new = 1.0 - beta_v_new - beta_l1_new
            except:
                return self._create_error_result("Rachford-Rice solution failed")
            
            # 更新组成
            x_v_new = z / (beta_v_new + beta_l1_new * K_vl1 + beta_l2_new * K_vl2)
            x_l1_new = K_vl1 * x_v_new
            x_l2_new = K_vl2 * x_v_new
            
            # 归一化
            x_v_new = x_v_new / np.sum(x_v_new)
            x_l1_new = x_l1_new / np.sum(x_l1_new)
            x_l2_new = x_l2_new / np.sum(x_l2_new)
            
            # 收敛性检查
            error_v = np.max(np.abs(x_v_new - x_v))
            error_l1 = np.max(np.abs(x_l1_new - x_l1))
            error_l2 = np.max(np.abs(x_l2_new - x_l2))
            error_beta_v = abs(beta_v_new - beta_v)
            error_beta_l1 = abs(beta_l1_new - beta_l1)
            
            max_error = max(error_v, error_l1, error_l2, error_beta_v, error_beta_l1)
            
            if max_error < self.tolerance:
                # 收敛成功
                return self._create_three_phase_result(
                    P, T, z, x_v_new, x_l1_new, x_l2_new,
                    beta_v_new, beta_l1_new, beta_l2_new,
                    phi_v, phi_l1, phi_l2, iteration
                )
            
            # 更新变量
            x_v = x_v_new
            x_l1 = x_l1_new
            x_l2 = x_l2_new
            beta_v = beta_v_new
            beta_l1 = beta_l1_new
        
        return self._create_error_result(f"Maximum iterations ({self.max_iterations}) exceeded")
    
    def _solve_rachford_rice_3phase(self, z: np.ndarray, K_vl1: np.ndarray,
                                   K_vl2: np.ndarray, K_l1l2: np.ndarray,
                                   beta_v_init: float, beta_l1_init: float) -> Tuple[float, float]:
        """
        求解三相Rachford-Rice方程组
        
        Parameters:
        -----------
        z : np.ndarray
            总组成
        K_vl1, K_vl2, K_l1l2 : np.ndarray
            平衡常数
        beta_v_init, beta_l1_init : float
            初值
            
        Returns:
        --------
        Tuple[float, float]
            (beta_v, beta_l1)
        """
        def rachford_rice_equations(beta):
            beta_v, beta_l1 = beta
            beta_l2 = 1.0 - beta_v - beta_l1
            
            if beta_v < 0 or beta_l1 < 0 or beta_l2 < 0:
                return [1e6, 1e6]
            
            # 三相Rachford-Rice方程
            eq1 = np.sum(z * (K_vl1 - 1) / 
                        (beta_v + beta_l1 * K_vl1 + beta_l2 * K_vl2))
            
            eq2 = np.sum(z * (K_l1l2 - 1) * K_vl1 / 
                        (beta_v + beta_l1 * K_vl1 + beta_l2 * K_vl2))
            
            return [eq1, eq2]
        
        # 数值求解
        initial_guess = [beta_v_init, beta_l1_init]
        
        try:
            sol = root(rachford_rice_equations, initial_guess, method='hybr')
            if sol.success:
                beta_v, beta_l1 = sol.x
                # 确保物理意义
                beta_v = max(0.0, min(1.0, beta_v))
                beta_l1 = max(0.0, min(1.0 - beta_v, beta_l1))
                return beta_v, beta_l1
            else:
                raise Exception("Root finding failed")
        except:
            # 备用简化求解
            return self._simplified_beta_calculation(z, K_vl1, K_vl2)
    
    def _simplified_beta_calculation(self, z: np.ndarray, K_vl1: np.ndarray,
                                   K_vl2: np.ndarray) -> Tuple[float, float]:
        """
        简化的相分率计算
        """
        # 基于Wilson方程的初步估算
        K_avg = (K_vl1 + K_vl2) / 2
        
        def wilson_equation(beta_v):
            return np.sum(z * (K_avg - 1) / (1 + beta_v * (K_avg - 1)))
        
        try:
            beta_v = fsolve(wilson_equation, 0.3)[0]
            beta_v = max(0.0, min(0.8, beta_v))
            beta_l1 = (1.0 - beta_v) * 0.5  # 假设两液相相等
            return beta_v, beta_l1
        except:
            return 0.3, 0.35  # 默认值
    
    def _generate_initial_guess(self, P: float, T: float, 
                              z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        生成三相闪蒸初值
        """
        nc = len(z)
        
        # 基于Antoine方程的蒸汽压估算
        P_sat = self._estimate_vapor_pressures(T)
        
        # 初始K值估算
        K_est = P_sat / P
        K_est = np.clip(K_est, 0.01, 100.0)
        
        # Wilson方程初值
        beta_v_est = 0.3
        
        # 气相组成初值
        x_v_init = z * K_est / (1 + beta_v_est * (K_est - 1))
        x_v_init = x_v_init / np.sum(x_v_init)
        
        # 液相组成初值 (假设两个液相)
        x_l1_init = z.copy()
        x_l2_init = z.copy()
        
        # 基于溶解度参数的液液分离
        if hasattr(self.property_package, 'solubility_parameters'):
            delta = self.property_package.solubility_parameters
            # 极性/非极性分离
            polar_indices = np.where(delta > np.median(delta))[0]
            nonpolar_indices = np.where(delta <= np.median(delta))[0]
            
            # 调整液相组成
            x_l1_init[polar_indices] *= 1.5
            x_l1_init[nonpolar_indices] *= 0.5
            x_l2_init[polar_indices] *= 0.5
            x_l2_init[nonpolar_indices] *= 1.5
        
        # 归一化
        x_l1_init = x_l1_init / np.sum(x_l1_init)
        x_l2_init = x_l2_init / np.sum(x_l2_init)
        
        beta_v_init = 0.3
        beta_l1_init = 0.35
        
        return x_v_init, x_l1_init, x_l2_init, beta_v_init, beta_l1_init
    
    def _estimate_vapor_pressures(self, T: float) -> np.ndarray:
        """
        估算蒸汽压
        """
        if hasattr(self.property_package, 'get_vapor_pressure'):
            P_sat = np.array([
                self.property_package.get_vapor_pressure(comp, T) 
                for comp in self.property_package.compounds
            ])
        else:
            # 简化估算
            P_sat = np.ones(len(self.property_package.compounds)) * 101325.0 * \
                   np.exp(10 * (1 - 373.15 / T))
        
        return P_sat
    
    def _calculate_fugacity_coefficients(self, x: np.ndarray, P: float, T: float,
                                       phase: str) -> np.ndarray:
        """
        计算逸度系数
        """
        nc = len(x)
        phi = np.ones(nc)
        
        for i, comp in enumerate(self.property_package.compounds):
            phi[i] = self.property_package.calculate_fugacity_coefficient(
                comp, x, P, T, phase
            )
        
        return phi
    
    def _create_three_phase_result(self, P: float, T: float, z: np.ndarray,
                                 x_v: np.ndarray, x_l1: np.ndarray, x_l2: np.ndarray,
                                 beta_v: float, beta_l1: float, beta_l2: float,
                                 phi_v: np.ndarray, phi_l1: np.ndarray, phi_l2: np.ndarray,
                                 iterations: int) -> Dict:
        """
        创建三相闪蒸结果
        """
        # 计算相性质
        vapor_props = self._calculate_phase_properties(x_v, P, T, self.VAPOR)
        liquid1_props = self._calculate_phase_properties(x_l1, P, T, self.LIQUID1)
        liquid2_props = self._calculate_phase_properties(x_l2, P, T, self.LIQUID2)
        
        return {
            'success': True,
            'num_phases': 3,
            'P': P,
            'T': T,
            'z': z,
            
            # 相组成
            'vapor': {
                'x': x_v,
                'beta': beta_v,
                'phi': phi_v,
                'properties': vapor_props
            },
            'liquid1': {
                'x': x_l1,
                'beta': beta_l1,
                'phi': phi_l1,
                'properties': liquid1_props
            },
            'liquid2': {
                'x': x_l2,
                'beta': beta_l2,
                'phi': phi_l2,
                'properties': liquid2_props
            },
            
            # 算法信息
            'algorithm': 'Boston-Fournier Inside-Out 3P',
            'iterations': iterations,
            'tolerance': self.tolerance
        }
    
    def _two_phase_flash(self, P: float, T: float, z: np.ndarray,
                        initial_guess: Dict = None) -> Dict:
        """
        二相闪蒸 (转交给标准二相算法)
        """
        from .nested_loops import NestedLoopsFlash
        
        two_phase_flash = NestedLoopsFlash(self.property_package)
        return two_phase_flash.flash_pt(P, T, z, initial_guess)
    
    def _single_phase_result(self, P: float, T: float, z: np.ndarray, phase: str) -> Dict:
        """
        单相结果
        """
        phi = self._calculate_fugacity_coefficients(z, P, T, phase)
        props = self._calculate_phase_properties(z, P, T, phase)
        
        return {
            'success': True,
            'num_phases': 1,
            'P': P,
            'T': T,
            'z': z,
            phase: {
                'x': z,
                'beta': 1.0,
                'phi': phi,
                'properties': props
            },
            'algorithm': 'Single Phase',
            'iterations': 0
        }
    
    def _calculate_phase_properties(self, x: np.ndarray, P: float, T: float,
                                  phase: str) -> Dict:
        """
        计算相性质
        """
        try:
            if hasattr(self.property_package, 'calculate_phase_properties'):
                return self.property_package.calculate_phase_properties(x, P, T, phase)
            else:
                # 基本性质计算
                MW = np.sum(x * self._get_molecular_weights())
                return {
                    'molecular_weight': MW,
                    'density': None,
                    'enthalpy': None,
                    'entropy': None
                }
        except:
            return {'molecular_weight': None}
    
    def _get_molecular_weights(self) -> np.ndarray:
        """
        获取分子量
        """
        if hasattr(self.property_package, 'molecular_weights'):
            return self.property_package.molecular_weights
        else:
            # 默认分子量
            return np.ones(len(self.property_package.compounds)) * 50.0
    
    def _validate_inputs(self, P: float, T: float, z: np.ndarray) -> bool:
        """
        验证输入参数
        """
        if P <= 0 or T <= 0:
            return False
        
        if len(z) != len(self.property_package.compounds):
            return False
        
        if abs(np.sum(z) - 1.0) > 1e-6:
            return False
        
        if np.any(z < 0):
            return False
        
        return True
    
    def _create_error_result(self, message: str) -> Dict:
        """
        创建错误结果
        """
        return {
            'success': False,
            'error_message': message,
            'num_phases': 0,
            'algorithm': 'Three-Phase Flash'
        }


class LiquidLiquidExtractor:
    """
    液液萃取器类，用于液液分离的特殊处理
    """
    
    def __init__(self, property_package):
        self.property_package = property_package
    
    def extract_components(self, feed_composition: np.ndarray, 
                          solvent_composition: np.ndarray,
                          P: float, T: float) -> Dict:
        """
        液液萃取计算
        
        Parameters:
        -----------
        feed_composition : np.ndarray
            进料组成
        solvent_composition : np.ndarray
            溶剂组成
        P : float
            压力 [Pa]
        T : float
            温度 [K]
            
        Returns:
        --------
        Dict
            萃取结果
        """
        flash_algo = ThreePhaseFlash(self.property_package)
        
        # 合并进料和溶剂
        total_composition = (feed_composition + solvent_composition) / 2
        
        # 执行液液平衡计算
        result = flash_algo.flash_pt_3phase(P, T, total_composition)
        
        if result['success'] and result['num_phases'] >= 2:
            return {
                'success': True,
                'raffinate': result.get('liquid1', {}).get('x', feed_composition),
                'extract': result.get('liquid2', {}).get('x', solvent_composition),
                'separation_factor': self._calculate_separation_factor(result)
            }
        else:
            return {
                'success': False,
                'error': 'Liquid-liquid separation failed'
            }
    
    def _calculate_separation_factor(self, flash_result: Dict) -> float:
        """
        计算分离因子
        """
        if 'liquid1' not in flash_result or 'liquid2' not in flash_result:
            return 1.0
        
        x1 = flash_result['liquid1']['x']
        x2 = flash_result['liquid2']['x']
        
        # 简化的分离因子计算
        if len(x1) >= 2:
            alpha = (x1[0] / x1[1]) / (x2[0] / x2[1] + 1e-10)
            return abs(alpha)
        
        return 1.0


# 使用示例
if __name__ == "__main__":
    from ..property_packages import PengRobinson
    
    # 创建物性包和闪蒸算法
    compounds = ['methane', 'ethane', 'n-butane', 'water']
    pr = PengRobinson(compounds)
    three_phase_flash = ThreePhaseFlash(pr)
    
    # 三相闪蒸计算
    P = 5e6  # 5 MPa
    T = 300.0  # 300 K
    z = np.array([0.3, 0.2, 0.3, 0.2])  # 组成
    
    result = three_phase_flash.flash_pt_3phase(P, T, z)
    
    print("三相闪蒸计算结果:")
    print(f"成功: {result['success']}")
    print(f"相数: {result.get('num_phases', 0)}")
    
    if result['success'] and result['num_phases'] == 3:
        print(f"\n气相分率: {result['vapor']['beta']:.4f}")
        print(f"液相1分率: {result['liquid1']['beta']:.4f}")
        print(f"液相2分率: {result['liquid2']['beta']:.4f}")
        
        print(f"\n气相组成: {result['vapor']['x']}")
        print(f"液相1组成: {result['liquid1']['x']}")
        print(f"液相2组成: {result['liquid2']['x']}")
    
    # 液液萃取示例
    extractor = LiquidLiquidExtractor(pr)
    feed = np.array([0.0, 0.0, 0.8, 0.2])  # 丁烷-水混合物
    solvent = np.array([0.0, 1.0, 0.0, 0.0])  # 乙烷溶剂
    
    extraction_result = extractor.extract_components(feed, solvent, P, T)
    print(f"\n萃取结果: {extraction_result}") 