"""
相稳定性分析模块
===============

基于Michelsen算法的相稳定性分析，用于判断给定条件下相的稳定性。
对应DWSIM的MichelsenBase.vb (2,933行)中的稳定性测试功能。

理论基础：
相稳定性分析基于切线平面距离(TPD)函数：
$$TPD = \sum_i n_i [\ln f_i(T,P,\mathbf{n}) - \ln f_i^{ref}(T,P,\mathbf{z})]$$

其中：
- $n_i$：试验相中组分i的摩尔数
- $f_i$：试验相中组分i的逸度
- $f_i^{ref}$：参考相中组分i的逸度
- $\mathbf{z}$：总体组成

稳定性判据：
- TPD < 0：相不稳定，存在相分离
- TPD ≥ 0：相稳定

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
import warnings

from ..core.enums import PhaseType, ConvergenceStatus
from ..core.exceptions import CalculationError, ConvergenceError
from ..core.property_package import PropertyPackage


@dataclass
class StabilityTestSettings:
    """稳定性测试设置"""
    
    # 收敛设置
    tolerance: float = 1e-8                    # 收敛容差
    max_iterations: int = 200                  # 最大迭代次数
    
    # 试验相设置
    test_phase_count: int = 10                 # 试验相数量
    perturbation_factor: float = 1e-4          # 扰动因子
    
    # 优化设置
    use_global_optimization: bool = True       # 使用全局优化
    optimization_method: str = "SLSQP"         # 优化方法
    
    # 数值稳定性
    minimum_composition: float = 1e-12         # 最小组成
    maximum_composition: float = 1.0           # 最大组成
    
    # 初值策略
    use_wilson_estimates: bool = True          # 使用Wilson估算
    use_random_perturbation: bool = True       # 使用随机扰动
    
    # 相识别
    phase_identification: bool = True          # 相识别
    trivial_solution_tolerance: float = 1e-6   # 平凡解容差


@dataclass
class StabilityTestResult:
    """稳定性测试结果"""
    
    is_stable: bool                           # 是否稳定
    tpd_minimum: float                        # TPD最小值
    unstable_phases: List[np.ndarray]         # 不稳定相组成
    test_phases: List[np.ndarray]             # 所有试验相组成
    tpd_values: List[float]                   # 所有TPD值
    iterations: int                           # 迭代次数
    converged: bool                           # 是否收敛
    convergence_status: ConvergenceStatus     # 收敛状态
    error_message: Optional[str] = None       # 错误信息


class MichelsenStabilityAnalysis:
    """Michelsen相稳定性分析
    
    实现Michelsen的相稳定性分析算法，包括：
    1. 切线平面距离(TPD)计算
    2. 多起点优化
    3. 相识别
    4. 稳定性判断
    
    算法特点：
    - 理论严格
    - 全局收敛
    - 适用于复杂体系
    - 支持多相平衡
    """
    
    def __init__(self, settings: Optional[StabilityTestSettings] = None):
        """初始化稳定性分析器
        
        Args:
            settings: 稳定性测试设置
        """
        self.settings = settings or StabilityTestSettings()
        self.logger = logging.getLogger("MichelsenStabilityAnalysis")
        
        # 算法统计
        self.stats = {
            'total_tests': 0,
            'stable_systems': 0,
            'unstable_systems': 0,
            'convergence_failures': 0,
            'average_iterations': 0.0,
            'average_cpu_time': 0.0
        }
    
    def test_stability(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        property_package: PropertyPackage,
        reference_phase: PhaseType = PhaseType.LIQUID
    ) -> StabilityTestResult:
        """执行相稳定性测试
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            composition: 摩尔组成
            property_package: 物性包
            reference_phase: 参考相类型
            
        Returns:
            StabilityTestResult: 稳定性测试结果
        """
        import time
        start_time = time.time()
        
        self.stats['total_tests'] += 1
        
        try:
            self.logger.info(f"开始稳定性测试: T={temperature:.2f}K, P={pressure:.0f}Pa")
            
            # 验证输入
            self._validate_inputs(temperature, pressure, composition)
            
            # 计算参考相化学势
            reference_chemical_potentials = self._calculate_reference_chemical_potentials(
                temperature, pressure, composition, property_package, reference_phase)
            
            # 生成试验相初值
            test_phases = self._generate_test_phases(
                composition, temperature, pressure, property_package)
            
            # 对每个试验相进行TPD最小化
            tpd_results = []
            converged_phases = []
            
            for i, test_phase in enumerate(test_phases):
                try:
                    result = self._minimize_tpd(
                        test_phase, temperature, pressure, 
                        reference_chemical_potentials, property_package)
                    
                    tpd_results.append(result)
                    
                    if result['converged']:
                        converged_phases.append(result['composition'])
                    
                    self.logger.debug(f"试验相 {i+1}: TPD={result['tpd']:.6e}, "
                                    f"收敛={result['converged']}")
                    
                except Exception as e:
                    self.logger.warning(f"试验相 {i+1} 优化失败: {e}")
                    tpd_results.append({
                        'tpd': np.inf,
                        'composition': test_phase,
                        'converged': False,
                        'iterations': 0
                    })
            
            # 分析结果
            result = self._analyze_stability_results(
                tpd_results, composition, temperature, pressure)
            
            # 更新统计
            if result.is_stable:
                self.stats['stable_systems'] += 1
            else:
                self.stats['unstable_systems'] += 1
            
            if not result.converged:
                self.stats['convergence_failures'] += 1
            
            cpu_time = time.time() - start_time
            self.stats['average_cpu_time'] = (
                (self.stats['average_cpu_time'] * (self.stats['total_tests'] - 1) + cpu_time) /
                self.stats['total_tests']
            )
            
            self.stats['average_iterations'] = (
                (self.stats['average_iterations'] * (self.stats['total_tests'] - 1) + result.iterations) /
                self.stats['total_tests']
            )
            
            self.logger.info(f"稳定性测试完成: 稳定={result.is_stable}, "
                           f"TPD_min={result.tpd_minimum:.6e}, "
                           f"迭代={result.iterations}, CPU时间={cpu_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"稳定性测试失败: {e}")
            
            return StabilityTestResult(
                is_stable=True,  # 保守估计
                tpd_minimum=0.0,
                unstable_phases=[],
                test_phases=[],
                tpd_values=[],
                iterations=0,
                converged=False,
                convergence_status=ConvergenceStatus.FAILED,
                error_message=str(e)
            )
    
    def _validate_inputs(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray
    ):
        """验证输入参数"""
        
        if temperature <= 0:
            raise ValueError("温度必须为正值")
        
        if pressure <= 0:
            raise ValueError("压力必须为正值")
        
        if len(composition) < 2:
            raise ValueError("至少需要两个组分")
        
        if not np.isclose(composition.sum(), 1.0, rtol=1e-6):
            raise ValueError("组成之和必须等于1")
        
        if np.any(composition < 0):
            raise ValueError("组成不能为负值")
    
    def _calculate_reference_chemical_potentials(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        property_package: PropertyPackage,
        reference_phase: PhaseType
    ) -> np.ndarray:
        """计算参考相化学势
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            composition: 摩尔组成
            property_package: 物性包
            reference_phase: 参考相类型
            
        Returns:
            np.ndarray: 化学势数组 (J/mol)
        """
        try:
            # 计算逸度系数
            fugacity_coefficients = property_package.calculate_fugacity_coefficients(
                temperature, pressure, composition, reference_phase)
            
            # 计算化学势
            # μᵢ = μᵢ⁰(T,P) + RT ln(xᵢφᵢ)
            R = 8.314  # J/(mol·K)
            
            chemical_potentials = np.zeros(len(composition))
            
            for i in range(len(composition)):
                if composition[i] > self.settings.minimum_composition:
                    # 标准化学势（简化为理想气体）
                    mu_standard = R * temperature * np.log(pressure / 101325.0)
                    
                    # 活度贡献
                    activity = composition[i] * fugacity_coefficients[i]
                    mu_activity = R * temperature * np.log(max(activity, 1e-50))
                    
                    chemical_potentials[i] = mu_standard + mu_activity
                else:
                    chemical_potentials[i] = -np.inf  # 避免数值问题
            
            return chemical_potentials
            
        except Exception as e:
            self.logger.error(f"计算参考相化学势失败: {e}")
            raise CalculationError(f"参考相化学势计算失败: {e}")
    
    def _generate_test_phases(
        self,
        composition: np.ndarray,
        temperature: float,
        pressure: float,
        property_package: PropertyPackage
    ) -> List[np.ndarray]:
        """生成试验相初值
        
        Args:
            composition: 参考组成
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            property_package: 物性包
            
        Returns:
            List[np.ndarray]: 试验相组成列表
        """
        n_comp = len(composition)
        test_phases = []
        
        # 1. Wilson K值估算的试验相
        if self.settings.use_wilson_estimates:
            try:
                k_wilson = self._estimate_wilson_k_values(
                    temperature, pressure, property_package)
                
                # 生成基于K值的试验相
                for factor in [0.1, 0.5, 2.0, 10.0]:
                    test_composition = composition * k_wilson * factor
                    test_composition /= test_composition.sum()
                    test_phases.append(test_composition)
                    
            except Exception as e:
                self.logger.warning(f"Wilson K值估算失败: {e}")
        
        # 2. 纯组分试验相
        for i in range(n_comp):
            if composition[i] > self.settings.minimum_composition:
                pure_phase = np.zeros(n_comp)
                pure_phase[i] = 1.0
                test_phases.append(pure_phase)
        
        # 3. 随机扰动试验相
        if self.settings.use_random_perturbation:
            np.random.seed(42)  # 确保可重现性
            
            for _ in range(self.settings.test_phase_count // 2):
                # 随机组成
                random_composition = np.random.dirichlet(np.ones(n_comp))
                test_phases.append(random_composition)
                
                # 扰动原始组成
                perturbation = np.random.normal(0, self.settings.perturbation_factor, n_comp)
                perturbed_composition = composition + perturbation
                perturbed_composition = np.maximum(perturbed_composition, 
                                                 self.settings.minimum_composition)
                perturbed_composition /= perturbed_composition.sum()
                test_phases.append(perturbed_composition)
        
        # 4. 确保有足够的试验相
        while len(test_phases) < self.settings.test_phase_count:
            # 生成更多随机试验相
            random_composition = np.random.dirichlet(np.ones(n_comp))
            test_phases.append(random_composition)
        
        # 限制试验相数量
        test_phases = test_phases[:self.settings.test_phase_count]
        
        self.logger.debug(f"生成了 {len(test_phases)} 个试验相")
        
        return test_phases
    
    def _estimate_wilson_k_values(
        self,
        temperature: float,
        pressure: float,
        property_package: PropertyPackage
    ) -> np.ndarray:
        """估算Wilson K值
        
        Wilson方程：
        K_i = (P_sat_i / P) * exp(V_L_i * (P - P_sat_i) / (RT))
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            property_package: 物性包
            
        Returns:
            np.ndarray: K值数组
        """
        n_comp = len(property_package.compounds)
        k_values = np.ones(n_comp)
        
        R = 8.314  # J/(mol·K)
        
        for i in range(n_comp):
            compound = property_package.compounds[i]
            
            try:
                # 计算饱和蒸汽压（Antoine方程或其他方法）
                p_sat = self._calculate_vapor_pressure(temperature, compound)
                
                # 液体摩尔体积估算
                v_liquid = 0.1 / 1000  # 简化估算，实际应使用更精确的方法
                
                # Wilson K值
                if p_sat > 0:
                    poynting_correction = np.exp(v_liquid * (pressure - p_sat) / (R * temperature))
                    k_values[i] = (p_sat / pressure) * poynting_correction
                else:
                    k_values[i] = 1e-6  # 非挥发性组分
                    
            except Exception as e:
                self.logger.warning(f"组分 {i} 的K值估算失败: {e}")
                k_values[i] = 1.0
        
        return k_values
    
    def _calculate_vapor_pressure(self, temperature: float, compound) -> float:
        """计算饱和蒸汽压"""
        
        # 简化的Antoine方程
        # log10(P_sat) = A - B/(C + T)
        # 这里使用简化的估算，实际应使用更精确的方法
        
        try:
            tc = compound.critical_temperature
            pc = compound.critical_pressure
            
            if temperature >= tc:
                return pc
            
            # 简化的Riedel方程
            tr = temperature / tc
            ln_pr = 5.373 * (1 + compound.acentric_factor) * (1 - 1/tr)
            
            return pc * np.exp(ln_pr)
            
        except:
            return 101325.0  # 默认1 atm
    
    def _minimize_tpd(
        self,
        initial_composition: np.ndarray,
        temperature: float,
        pressure: float,
        reference_chemical_potentials: np.ndarray,
        property_package: PropertyPackage
    ) -> Dict[str, Any]:
        """最小化TPD函数
        
        Args:
            initial_composition: 初始试验相组成
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            reference_chemical_potentials: 参考相化学势
            property_package: 物性包
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        n_comp = len(initial_composition)
        
        # 定义目标函数
        def objective(x):
            return self._calculate_tpd(
                x, temperature, pressure, reference_chemical_potentials, property_package)
        
        # 定义约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: x.sum() - 1.0}  # 组成之和等于1
        ]
        
        # 变量边界
        bounds = [(self.settings.minimum_composition, self.settings.maximum_composition) 
                 for _ in range(n_comp)]
        
        try:
            # 使用局部优化
            result_local = minimize(
                objective,
                initial_composition,
                method=self.settings.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.settings.max_iterations,
                    'ftol': self.settings.tolerance
                }
            )
            
            best_result = result_local
            
            # 如果启用全局优化，尝试全局搜索
            if self.settings.use_global_optimization:
                try:
                    result_global = differential_evolution(
                        objective,
                        bounds,
                        constraints=constraints,
                        maxiter=self.settings.max_iterations // 4,
                        atol=self.settings.tolerance,
                        seed=42
                    )
                    
                    if result_global.fun < best_result.fun:
                        best_result = result_global
                        
                except Exception as e:
                    self.logger.debug(f"全局优化失败: {e}")
            
            # 标准化组成
            final_composition = best_result.x
            final_composition = np.maximum(final_composition, self.settings.minimum_composition)
            final_composition /= final_composition.sum()
            
            return {
                'tpd': best_result.fun,
                'composition': final_composition,
                'converged': best_result.success,
                'iterations': best_result.nit if hasattr(best_result, 'nit') else 0,
                'message': best_result.message if hasattr(best_result, 'message') else ""
            }
            
        except Exception as e:
            self.logger.error(f"TPD最小化失败: {e}")
            return {
                'tpd': np.inf,
                'composition': initial_composition,
                'converged': False,
                'iterations': 0,
                'message': str(e)
            }
    
    def _calculate_tpd(
        self,
        composition: np.ndarray,
        temperature: float,
        pressure: float,
        reference_chemical_potentials: np.ndarray,
        property_package: PropertyPackage
    ) -> float:
        """计算切线平面距离(TPD)
        
        TPD = Σᵢ nᵢ [μᵢ(T,P,n) - μᵢʳᵉᶠ(T,P,z)]
        
        Args:
            composition: 试验相组成
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            reference_chemical_potentials: 参考相化学势
            property_package: 物性包
            
        Returns:
            float: TPD值
        """
        try:
            # 标准化组成
            composition = np.maximum(composition, self.settings.minimum_composition)
            composition /= composition.sum()
            
            # 计算试验相逸度系数
            fugacity_coefficients = property_package.calculate_fugacity_coefficients(
                temperature, pressure, composition, PhaseType.LIQUID)
            
            # 计算试验相化学势
            R = 8.314  # J/(mol·K)
            tpd = 0.0
            
            for i in range(len(composition)):
                if composition[i] > self.settings.minimum_composition:
                    # 试验相化学势
                    mu_standard = R * temperature * np.log(pressure / 101325.0)
                    activity = composition[i] * fugacity_coefficients[i]
                    mu_test = mu_standard + R * temperature * np.log(max(activity, 1e-50))
                    
                    # TPD贡献
                    tpd += composition[i] * (mu_test - reference_chemical_potentials[i])
            
            return tpd
            
        except Exception as e:
            self.logger.error(f"TPD计算失败: {e}")
            return np.inf
    
    def _analyze_stability_results(
        self,
        tpd_results: List[Dict[str, Any]],
        reference_composition: np.ndarray,
        temperature: float,
        pressure: float
    ) -> StabilityTestResult:
        """分析稳定性结果
        
        Args:
            tpd_results: TPD优化结果列表
            reference_composition: 参考组成
            temperature: 温度
            pressure: 压力
            
        Returns:
            StabilityTestResult: 稳定性测试结果
        """
        # 提取有效结果
        valid_results = [r for r in tpd_results if r['converged'] and np.isfinite(r['tpd'])]
        
        if not valid_results:
            return StabilityTestResult(
                is_stable=True,  # 保守估计
                tpd_minimum=0.0,
                unstable_phases=[],
                test_phases=[r['composition'] for r in tpd_results],
                tpd_values=[r['tpd'] for r in tpd_results],
                iterations=sum(r['iterations'] for r in tpd_results),
                converged=False,
                convergence_status=ConvergenceStatus.FAILED,
                error_message="所有试验相优化失败"
            )
        
        # 找到最小TPD值
        min_tpd_result = min(valid_results, key=lambda x: x['tpd'])
        min_tpd = min_tpd_result['tpd']
        
        # 判断稳定性
        is_stable = min_tpd >= -self.settings.tolerance
        
        # 识别不稳定相
        unstable_phases = []
        if not is_stable:
            for result in valid_results:
                if result['tpd'] < -self.settings.tolerance:
                    # 检查是否为平凡解
                    if not self._is_trivial_solution(
                        result['composition'], reference_composition):
                        unstable_phases.append(result['composition'])
        
        # 去重不稳定相
        unstable_phases = self._remove_duplicate_phases(unstable_phases)
        
        # 确定收敛状态
        converged = len(valid_results) >= len(tpd_results) * 0.8  # 80%收敛率
        convergence_status = (ConvergenceStatus.CONVERGED if converged 
                            else ConvergenceStatus.PARTIALLY_CONVERGED)
        
        return StabilityTestResult(
            is_stable=is_stable,
            tpd_minimum=min_tpd,
            unstable_phases=unstable_phases,
            test_phases=[r['composition'] for r in tpd_results],
            tpd_values=[r['tpd'] for r in tpd_results],
            iterations=sum(r['iterations'] for r in tpd_results),
            converged=converged,
            convergence_status=convergence_status
        )
    
    def _is_trivial_solution(
        self,
        test_composition: np.ndarray,
        reference_composition: np.ndarray
    ) -> bool:
        """检查是否为平凡解
        
        Args:
            test_composition: 试验相组成
            reference_composition: 参考相组成
            
        Returns:
            bool: 是否为平凡解
        """
        # 计算组成差异
        composition_diff = np.abs(test_composition - reference_composition)
        max_diff = np.max(composition_diff)
        
        return max_diff < self.settings.trivial_solution_tolerance
    
    def _remove_duplicate_phases(
        self,
        phases: List[np.ndarray],
        tolerance: float = 1e-4
    ) -> List[np.ndarray]:
        """去除重复相
        
        Args:
            phases: 相组成列表
            tolerance: 判断重复的容差
            
        Returns:
            List[np.ndarray]: 去重后的相组成列表
        """
        if not phases:
            return []
        
        unique_phases = [phases[0]]
        
        for phase in phases[1:]:
            is_duplicate = False
            
            for unique_phase in unique_phases:
                if np.allclose(phase, unique_phase, atol=tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_phases.append(phase)
        
        return unique_phases
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        
        return {
            'name': 'Michelsen Stability Analysis',
            'type': 'Phase Stability Test',
            'description': '基于TPD函数的相稳定性分析',
            'theoretical_basis': 'Tangent Plane Distance (TPD) minimization',
            'advantages': [
                '理论严格',
                '全局收敛',
                '适用于复杂体系',
                '支持多相平衡'
            ],
            'limitations': [
                '计算量大',
                '需要多个初值',
                '对物性模型敏感'
            ],
            'settings': {
                'tolerance': self.settings.tolerance,
                'max_iterations': self.settings.max_iterations,
                'test_phase_count': self.settings.test_phase_count,
                'use_global_optimization': self.settings.use_global_optimization
            },
            'stats': self.stats
        }


# 导出主要类
__all__ = [
    'MichelsenStabilityAnalysis',
    'StabilityTestSettings',
    'StabilityTestResult'
] 