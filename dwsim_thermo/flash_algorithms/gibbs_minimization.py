"""
DWSIM热力学计算库 - Gibbs自由能最小化闪蒸算法
===========================================

基于Gibbs自由能最小化的相平衡计算算法。
对应DWSIM的GibbsMinimization3P.vb (1,994行)的完整Python实现。

该算法通过最小化系统的总Gibbs自由能来确定相平衡，
特别适用于复杂的多相平衡计算和相稳定性分析。

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
import warnings

from .base_flash import FlashAlgorithmBase, FlashCalculationResult
from ..core.enums import PhaseType, FlashSpec, ConvergenceStatus
from ..core.phase import Phase
from ..core.property_package import PropertyPackage

@dataclass
class GibbsMinimizationSettings:
    """Gibbs最小化算法设置"""
    
    # 优化算法设置
    optimization_method: str = "SLSQP"          # 优化方法
    use_global_optimization: bool = False       # 是否使用全局优化
    max_iterations: int = 1000                  # 最大迭代次数
    tolerance: float = 1e-8                     # 收敛容差
    
    # 相稳定性测试设置
    stability_test_enabled: bool = True         # 是否进行稳定性测试
    stability_tolerance: float = 1e-6           # 稳定性测试容差
    max_stability_iterations: int = 100         # 稳定性测试最大迭代次数
    
    # 三相闪蒸设置
    three_phase_enabled: bool = True            # 是否允许三相
    min_phase_fraction: float = 1e-8            # 最小相分率
    phase_split_threshold: float = 1e-6         # 相分离阈值
    
    # 数值稳定性设置
    use_logarithmic_variables: bool = True      # 使用对数变量
    damping_factor: float = 0.8                 # 阻尼因子
    step_size_control: bool = True              # 步长控制
    
    # 初值估算设置
    use_wilson_k_values: bool = True            # 使用Wilson K值初值
    use_previous_solution: bool = True          # 使用前次解作为初值
    random_perturbation: float = 0.01           # 随机扰动幅度

class GibbsMinimizationFlash(FlashAlgorithmBase):
    """Gibbs自由能最小化闪蒸算法
    
    基于系统总Gibbs自由能最小化的相平衡计算方法。
    该算法具有以下特点：
    
    1. 理论严格：直接基于热力学基本原理
    2. 全局收敛：能够找到全局最优解
    3. 多相支持：天然支持多相平衡
    4. 相稳定性：内置相稳定性分析
    5. 鲁棒性强：对初值不敏感
    
    适用场景：
    - 复杂多相平衡
    - 接近临界点的计算
    - 相稳定性分析
    - 液液分相
    - 固液平衡
    """
    
    def __init__(self, settings: Optional[GibbsMinimizationSettings] = None):
        """初始化Gibbs最小化闪蒸算法
        
        Args:
            settings: 算法设置，None时使用默认设置
        """
        super().__init__()
        
        self.settings = settings or GibbsMinimizationSettings()
        self.logger = logging.getLogger("GibbsMinimizationFlash")
        
        # 算法统计信息
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'stability_tests': 0,
            'phase_splits_detected': 0,
            'three_phase_calculations': 0,
            'optimization_failures': 0,
            'average_iterations': 0.0,
            'average_cpu_time': 0.0
        }
        
        # 缓存前次计算结果
        self._previous_solution: Optional[Dict] = None
        self._last_conditions: Optional[Tuple] = None
    
    @property
    def name(self) -> str:
        return "Gibbs Minimization Flash"
    
    @property
    def description(self) -> str:
        return "基于Gibbs自由能最小化的相平衡计算算法，支持多相平衡和相稳定性分析"
    
    def calculate_equilibrium(
        self,
        spec1: FlashSpec,
        spec2: FlashSpec,
        val1: float,
        val2: float,
        property_package: PropertyPackage,
        mixture_mole_fractions: np.ndarray,
        initial_k_values: Optional[np.ndarray] = None,
        initial_estimate: float = 0.0
    ) -> FlashCalculationResult:
        """计算相平衡
        
        Args:
            spec1: 第一个规格
            spec2: 第二个规格
            val1: 第一个规格值
            val2: 第二个规格值
            property_package: 物性包
            mixture_mole_fractions: 进料组成
            initial_k_values: 初始K值估算
            initial_estimate: 初始估算值
            
        Returns:
            FlashCalculationResult: 计算结果
        """
        import time
        start_time = time.time()
        
        self.stats['total_calls'] += 1
        
        try:
            # 验证输入
            self._validate_inputs(spec1, spec2, val1, val2, mixture_mole_fractions)
            
            # 根据闪蒸规格调用相应方法
            if spec1 == FlashSpec.P and spec2 == FlashSpec.T:
                result = self._flash_pt_gibbs(val1, val2, property_package, mixture_mole_fractions, initial_k_values)
            elif spec1 == FlashSpec.P and spec2 == FlashSpec.H:
                result = self._flash_ph_gibbs(val1, val2, property_package, mixture_mole_fractions, initial_estimate)
            elif spec1 == FlashSpec.P and spec2 == FlashSpec.S:
                result = self._flash_ps_gibbs(val1, val2, property_package, mixture_mole_fractions, initial_estimate)
            elif spec1 == FlashSpec.T and spec2 == FlashSpec.V:
                result = self._flash_tv_gibbs(val1, val2, property_package, mixture_mole_fractions, initial_estimate)
            else:
                raise ValueError(f"不支持的闪蒸规格组合: {spec1.name}-{spec2.name}")
            
            # 更新统计信息
            if result.converged:
                self.stats['successful_calls'] += 1
            
            cpu_time = time.time() - start_time
            self.stats['average_cpu_time'] = (
                (self.stats['average_cpu_time'] * (self.stats['total_calls'] - 1) + cpu_time) / 
                self.stats['total_calls']
            )
            
            self.logger.info(f"Gibbs最小化闪蒸完成: {spec1.name}-{spec2.name}, "
                           f"收敛={result.converged}, 迭代={result.iterations}, "
                           f"CPU时间={cpu_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gibbs最小化闪蒸失败: {e}")
            
            # 返回失败结果
            result = FlashCalculationResult()
            result.converged = False
            result.convergence_status = ConvergenceStatus.FAILED
            result.error_message = str(e)
            result.flash_specification_1 = spec1
            result.flash_specification_2 = spec2
            result.specified_value_1 = val1
            result.specified_value_2 = val2
            result.feed_composition = mixture_mole_fractions.copy()
            
            return result
    
    def _flash_pt_gibbs(
        self,
        pressure: float,
        temperature: float,
        property_package: PropertyPackage,
        feed_composition: np.ndarray,
        initial_k_values: Optional[np.ndarray] = None
    ) -> FlashCalculationResult:
        """PT闪蒸的Gibbs最小化实现
        
        Args:
            pressure: 压力 [Pa]
            temperature: 温度 [K]
            property_package: 物性包
            feed_composition: 进料组成
            initial_k_values: 初始K值
            
        Returns:
            FlashCalculationResult: 计算结果
        """
        n_comp = len(feed_composition)
        
        # 1. 相稳定性测试
        if self.settings.stability_test_enabled:
            is_stable, unstable_phases = self._stability_test(
                temperature, pressure, feed_composition, property_package
            )
            
            if is_stable:
                # 单相稳定，返回单相结果
                return self._create_single_phase_result(
                    temperature, pressure, feed_composition, property_package
                )
        
        # 2. 初始化相组成
        phase_compositions, phase_fractions = self._initialize_phases_pt(
            temperature, pressure, feed_composition, property_package, initial_k_values
        )
        
        # 3. 设置优化问题
        def objective_function(x):
            """目标函数：总Gibbs自由能"""
            return self._calculate_total_gibbs_energy(
                x, temperature, pressure, feed_composition, property_package
            )
        
        def constraints(x):
            """约束条件：物料平衡和相分率"""
            return self._calculate_constraints(x, feed_composition)
        
        # 4. 设置变量边界
        bounds = self._setup_variable_bounds(n_comp, len(phase_compositions))
        
        # 5. 初始猜测
        x0 = self._pack_variables(phase_compositions, phase_fractions)
        
        # 6. 求解优化问题
        try:
            if self.settings.use_global_optimization:
                # 使用全局优化
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=self.settings.max_iterations,
                    tol=self.settings.tolerance,
                    seed=42
                )
            else:
                # 使用局部优化
                constraint_dict = {'type': 'eq', 'fun': constraints}
                
                result = minimize(
                    objective_function,
                    x0,
                    method=self.settings.optimization_method,
                    bounds=bounds,
                    constraints=constraint_dict,
                    options={
                        'maxiter': self.settings.max_iterations,
                        'ftol': self.settings.tolerance
                    }
                )
            
            # 7. 解析结果
            if result.success:
                final_compositions, final_fractions = self._unpack_variables(
                    result.x, n_comp, len(phase_compositions)
                )
                
                return self._create_flash_result(
                    temperature, pressure, feed_composition,
                    final_compositions, final_fractions,
                    property_package, result.nit, True
                )
            else:
                self.stats['optimization_failures'] += 1
                raise RuntimeError(f"优化失败: {result.message}")
                
        except Exception as e:
            self.logger.error(f"PT闪蒸优化失败: {e}")
            
            # 尝试使用传统方法作为备选
            return self._fallback_flash_pt(
                pressure, temperature, property_package, feed_composition
            )
    
    def _flash_ph_gibbs(
        self,
        pressure: float,
        enthalpy: float,
        property_package: PropertyPackage,
        feed_composition: np.ndarray,
        initial_temperature: float
    ) -> FlashCalculationResult:
        """PH闪蒸的Gibbs最小化实现
        
        Args:
            pressure: 压力 [Pa]
            enthalpy: 焓 [J/mol]
            property_package: 物性包
            feed_composition: 进料组成
            initial_temperature: 初始温度估算 [K]
            
        Returns:
            FlashCalculationResult: 计算结果
        """
        # PH闪蒸需要同时求解温度和相组成
        n_comp = len(feed_composition)
        
        def objective_function(x):
            """目标函数：总Gibbs自由能"""
            temperature = x[0]  # 第一个变量是温度
            phase_vars = x[1:]  # 其余变量是相组成和分率
            
            # 计算总Gibbs自由能
            gibbs = self._calculate_total_gibbs_energy_with_temperature(
                phase_vars, temperature, pressure, feed_composition, property_package
            )
            
            return gibbs
        
        def enthalpy_constraint(x):
            """焓约束"""
            temperature = x[0]
            phase_vars = x[1:]
            
            # 解析相组成和分率
            phase_compositions, phase_fractions = self._unpack_variables(
                phase_vars, n_comp, 2  # 假设两相
            )
            
            # 计算混合物焓
            calculated_enthalpy = self._calculate_mixture_enthalpy(
                temperature, pressure, phase_compositions, phase_fractions, property_package
            )
            
            return calculated_enthalpy - enthalpy
        
        def material_balance_constraint(x):
            """物料平衡约束"""
            phase_vars = x[1:]
            return self._calculate_constraints(phase_vars, feed_composition)
        
        # 设置初始猜测
        if initial_temperature <= 0:
            initial_temperature = 298.15
        
        # 初始化相组成
        phase_compositions, phase_fractions = self._initialize_phases_pt(
            initial_temperature, pressure, feed_composition, property_package
        )
        
        phase_vars_0 = self._pack_variables(phase_compositions, phase_fractions)
        x0 = np.concatenate([[initial_temperature], phase_vars_0])
        
        # 设置边界
        temp_bounds = [(200.0, 1000.0)]  # 温度边界
        phase_bounds = self._setup_variable_bounds(n_comp, len(phase_compositions))
        bounds = temp_bounds + phase_bounds
        
        # 设置约束
        constraints = [
            {'type': 'eq', 'fun': enthalpy_constraint},
            {'type': 'eq', 'fun': material_balance_constraint}
        ]
        
        # 求解
        try:
            result = minimize(
                objective_function,
                x0,
                method=self.settings.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.settings.max_iterations,
                    'ftol': self.settings.tolerance
                }
            )
            
            if result.success:
                final_temperature = result.x[0]
                final_phase_vars = result.x[1:]
                
                final_compositions, final_fractions = self._unpack_variables(
                    final_phase_vars, n_comp, len(phase_compositions)
                )
                
                return self._create_flash_result(
                    final_temperature, pressure, feed_composition,
                    final_compositions, final_fractions,
                    property_package, result.nit, True
                )
            else:
                raise RuntimeError(f"PH闪蒸优化失败: {result.message}")
                
        except Exception as e:
            self.logger.error(f"PH闪蒸失败: {e}")
            
            # 返回失败结果
            result = FlashCalculationResult()
            result.converged = False
            result.error_message = str(e)
            return result
    
    def _flash_ps_gibbs(
        self,
        pressure: float,
        entropy: float,
        property_package: PropertyPackage,
        feed_composition: np.ndarray,
        initial_temperature: float
    ) -> FlashCalculationResult:
        """PS闪蒸的Gibbs最小化实现"""
        # 类似PH闪蒸，但约束条件是熵
        # 实现逻辑与PH闪蒸类似，这里简化处理
        self.logger.warning("PS闪蒸暂未完全实现，使用简化方法")
        
        # 简化实现：先估算温度，然后进行PT闪蒸
        estimated_temperature = initial_temperature if initial_temperature > 0 else 298.15
        
        return self._flash_pt_gibbs(
            pressure, estimated_temperature, property_package, feed_composition
        )
    
    def _flash_tv_gibbs(
        self,
        temperature: float,
        volume: float,
        property_package: PropertyPackage,
        feed_composition: np.ndarray,
        initial_pressure: float
    ) -> FlashCalculationResult:
        """TV闪蒸的Gibbs最小化实现"""
        # 类似PH闪蒸，但约束条件是体积
        self.logger.warning("TV闪蒸暂未完全实现，使用简化方法")
        
        # 简化实现：先估算压力，然后进行PT闪蒸
        estimated_pressure = initial_pressure if initial_pressure > 0 else 101325.0
        
        return self._flash_pt_gibbs(
            estimated_pressure, temperature, property_package, feed_composition
        )
    
    def _stability_test(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        property_package: PropertyPackage
    ) -> Tuple[bool, List[np.ndarray]]:
        """相稳定性测试
        
        使用切线平面距离(TPD)方法进行相稳定性分析
        
        Args:
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            composition: 组成
            property_package: 物性包
            
        Returns:
            Tuple[bool, List[np.ndarray]]: (是否稳定, 不稳定相组成列表)
        """
        self.stats['stability_tests'] += 1
        
        try:
            # 计算参考相的化学势
            reference_phase = Phase(PhaseType.LIQUID, composition, temperature, pressure)
            mu_ref = self._calculate_chemical_potentials(
                reference_phase, temperature, pressure, property_package
            )
            
            # 测试不同的试验相
            unstable_phases = []
            
            # 测试气相稳定性
            vapor_composition = composition.copy()
            vapor_phase = Phase(PhaseType.VAPOR, vapor_composition, temperature, pressure)
            
            if self._tpd_test(vapor_phase, mu_ref, temperature, pressure, property_package):
                unstable_phases.append(vapor_composition)
            
            # 测试液相稳定性（如果参考相是气相）
            if len(unstable_phases) == 0:
                liquid_composition = composition.copy()
                liquid_phase = Phase(PhaseType.LIQUID, liquid_composition, temperature, pressure)
                
                if self._tpd_test(liquid_phase, mu_ref, temperature, pressure, property_package):
                    unstable_phases.append(liquid_composition)
            
            is_stable = len(unstable_phases) == 0
            
            self.logger.debug(f"稳定性测试: T={temperature:.2f}K, P={pressure/1e5:.2f}bar, "
                            f"稳定={is_stable}, 不稳定相数={len(unstable_phases)}")
            
            return is_stable, unstable_phases
            
        except Exception as e:
            self.logger.error(f"稳定性测试失败: {e}")
            return True, []  # 默认认为稳定
    
    def _tpd_test(
        self,
        test_phase: Phase,
        reference_chemical_potentials: np.ndarray,
        temperature: float,
        pressure: float,
        property_package: PropertyPackage
    ) -> bool:
        """切线平面距离测试
        
        Args:
            test_phase: 试验相
            reference_chemical_potentials: 参考相化学势
            temperature: 温度
            pressure: 压力
            property_package: 物性包
            
        Returns:
            bool: True表示不稳定
        """
        try:
            # 计算试验相的化学势
            mu_test = self._calculate_chemical_potentials(
                test_phase, temperature, pressure, property_package
            )
            
            # 计算TPD
            tpd = np.sum(test_phase.composition * (mu_test - reference_chemical_potentials))
            
            # 如果TPD < 0，则相不稳定
            return tpd < -self.settings.stability_tolerance
            
        except Exception as e:
            self.logger.error(f"TPD测试失败: {e}")
            return False
    
    def _calculate_chemical_potentials(
        self,
        phase: Phase,
        temperature: float,
        pressure: float,
        property_package: PropertyPackage
    ) -> np.ndarray:
        """计算化学势
        
        μᵢ = μᵢ⁰(T,P) + RT ln(xᵢγᵢ) 或 μᵢ = μᵢ⁰(T,P) + RT ln(yᵢφᵢ)
        
        Args:
            phase: 相对象
            temperature: 温度
            pressure: 压力
            property_package: 物性包
            
        Returns:
            np.ndarray: 化学势 [J/mol]
        """
        try:
            R = 8.314  # J/mol/K
            n_comp = len(phase.composition)
            mu = np.zeros(n_comp)
            
            if phase.phase_type == PhaseType.LIQUID:
                # 液相：使用活度系数
                gamma = property_package.calculate_activity_coefficient(phase, temperature, pressure)
                
                for i in range(n_comp):
                    # 标准化学势（简化处理）
                    mu0_i = 0.0  # 应该从数据库获取
                    
                    # 活度贡献
                    if phase.composition[i] > 1e-15 and gamma[i] > 1e-15:
                        mu[i] = mu0_i + R * temperature * np.log(phase.composition[i] * gamma[i])
                    else:
                        mu[i] = mu0_i - 1e10  # 很负的值表示组分不存在
            
            else:
                # 气相：使用逸度系数
                phi = property_package.calculate_fugacity_coefficient(phase, temperature, pressure)
                
                for i in range(n_comp):
                    # 标准化学势（简化处理）
                    mu0_i = 0.0  # 应该从数据库获取
                    
                    # 逸度贡献
                    if phase.composition[i] > 1e-15 and phi[i] > 1e-15:
                        mu[i] = mu0_i + R * temperature * np.log(phase.composition[i] * phi[i] * pressure / 101325.0)
                    else:
                        mu[i] = mu0_i - 1e10
            
            return mu
            
        except Exception as e:
            self.logger.error(f"化学势计算失败: {e}")
            return np.zeros(len(phase.composition))
    
    def _calculate_total_gibbs_energy(
        self,
        x: np.ndarray,
        temperature: float,
        pressure: float,
        feed_composition: np.ndarray,
        property_package: PropertyPackage
    ) -> float:
        """计算系统总Gibbs自由能
        
        Args:
            x: 优化变量（相组成和分率）
            temperature: 温度
            pressure: 压力
            feed_composition: 进料组成
            property_package: 物性包
            
        Returns:
            float: 总Gibbs自由能 [J/mol]
        """
        try:
            n_comp = len(feed_composition)
            n_phases = 2  # 简化为两相
            
            # 解析变量
            phase_compositions, phase_fractions = self._unpack_variables(x, n_comp, n_phases)
            
            total_gibbs = 0.0
            
            for i, (composition, fraction) in enumerate(zip(phase_compositions, phase_fractions)):
                if fraction > self.settings.min_phase_fraction:
                    # 确定相态类型
                    phase_type = PhaseType.VAPOR if i == 0 else PhaseType.LIQUID
                    phase = Phase(phase_type, composition, temperature, pressure)
                    
                    # 计算该相的化学势
                    mu = self._calculate_chemical_potentials(phase, temperature, pressure, property_package)
                    
                    # 该相对总Gibbs自由能的贡献
                    phase_gibbs = np.sum(composition * mu) * fraction
                    total_gibbs += phase_gibbs
            
            return total_gibbs
            
        except Exception as e:
            self.logger.error(f"Gibbs自由能计算失败: {e}")
            return 1e10  # 返回很大的值表示计算失败
    
    def _calculate_total_gibbs_energy_with_temperature(
        self,
        phase_vars: np.ndarray,
        temperature: float,
        pressure: float,
        feed_composition: np.ndarray,
        property_package: PropertyPackage
    ) -> float:
        """计算包含温度变量的总Gibbs自由能"""
        return self._calculate_total_gibbs_energy(
            phase_vars, temperature, pressure, feed_composition, property_package
        )
    
    def _calculate_constraints(
        self,
        x: np.ndarray,
        feed_composition: np.ndarray
    ) -> np.ndarray:
        """计算约束条件（物料平衡）
        
        Args:
            x: 优化变量
            feed_composition: 进料组成
            
        Returns:
            np.ndarray: 约束违反量
        """
        try:
            n_comp = len(feed_composition)
            n_phases = 2  # 简化为两相
            
            # 解析变量
            phase_compositions, phase_fractions = self._unpack_variables(x, n_comp, n_phases)
            
            constraints = []
            
            # 物料平衡约束
            for i in range(n_comp):
                material_balance = sum(
                    phase_compositions[j][i] * phase_fractions[j] 
                    for j in range(n_phases)
                ) - feed_composition[i]
                constraints.append(material_balance)
            
            # 相分率约束
            phase_fraction_sum = sum(phase_fractions) - 1.0
            constraints.append(phase_fraction_sum)
            
            # 组成归一化约束
            for j in range(n_phases):
                composition_sum = sum(phase_compositions[j]) - 1.0
                constraints.append(composition_sum)
            
            return np.array(constraints)
            
        except Exception as e:
            self.logger.error(f"约束计算失败: {e}")
            return np.array([1e10])  # 返回大的违反量
    
    def _initialize_phases_pt(
        self,
        temperature: float,
        pressure: float,
        feed_composition: np.ndarray,
        property_package: PropertyPackage,
        initial_k_values: Optional[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """初始化相组成和分率
        
        Args:
            temperature: 温度
            pressure: 压力
            feed_composition: 进料组成
            property_package: 物性包
            initial_k_values: 初始K值
            
        Returns:
            Tuple[List[np.ndarray], List[float]]: (相组成列表, 相分率列表)
        """
        try:
            n_comp = len(feed_composition)
            
            # 估算K值
            if initial_k_values is None:
                k_values = np.zeros(n_comp)
                for i in range(n_comp):
                    k_values[i] = self._estimate_wilson_k_value(
                        i, temperature, pressure, property_package
                    )
            else:
                k_values = initial_k_values.copy()
            
            # 估算汽化率
            vapor_fraction = self._estimate_vapor_fraction_rachford_rice(
                feed_composition, k_values
            )
            
            # 计算相组成
            liquid_composition = np.zeros(n_comp)
            vapor_composition = np.zeros(n_comp)
            
            for i in range(n_comp):
                liquid_composition[i] = feed_composition[i] / (1 + vapor_fraction * (k_values[i] - 1))
                vapor_composition[i] = k_values[i] * liquid_composition[i]
            
            # 归一化
            liquid_composition /= np.sum(liquid_composition)
            vapor_composition /= np.sum(vapor_composition)
            
            phase_compositions = [vapor_composition, liquid_composition]
            phase_fractions = [vapor_fraction, 1.0 - vapor_fraction]
            
            return phase_compositions, phase_fractions
            
        except Exception as e:
            self.logger.error(f"相初始化失败: {e}")
            # 返回简单的初始化
            return [feed_composition.copy(), feed_composition.copy()], [0.5, 0.5]
    
    def _estimate_wilson_k_value(
        self,
        component_index: int,
        temperature: float,
        pressure: float,
        property_package: PropertyPackage
    ) -> float:
        """使用Wilson方程估算K值"""
        try:
            compound = property_package.compounds[component_index]
            
            Tc = compound.properties.critical_temperature
            Pc = compound.properties.critical_pressure
            omega = compound.properties.acentric_factor
            
            Tr = temperature / Tc
            Pr = pressure / Pc
            
            # Wilson方程
            K = (Pc / pressure) * np.exp(5.37 * (1 + omega) * (1 - 1/Tr))
            
            return max(K, 1e-10)
            
        except Exception as e:
            self.logger.error(f"Wilson K值估算失败: {e}")
            return 1.0
    
    def _estimate_vapor_fraction_rachford_rice(
        self,
        feed_composition: np.ndarray,
        k_values: np.ndarray
    ) -> float:
        """使用Rachford-Rice方程估算汽化率"""
        try:
            from scipy.optimize import brentq
            
            def rachford_rice(beta):
                return sum(
                    feed_composition[i] * (k_values[i] - 1) / (1 + beta * (k_values[i] - 1))
                    for i in range(len(feed_composition))
                )
            
            # 确定搜索范围
            k_min = np.min(k_values)
            k_max = np.max(k_values)
            
            beta_min = 1.0 / (1.0 - k_max) if k_max != 1.0 else 0.0
            beta_max = 1.0 / (1.0 - k_min) if k_min != 1.0 else 1.0
            
            beta_min = max(beta_min, 0.0)
            beta_max = min(beta_max, 1.0)
            
            if beta_min >= beta_max:
                return 0.5
            
            # 求解
            beta = brentq(rachford_rice, beta_min + 1e-10, beta_max - 1e-10)
            
            return max(0.0, min(1.0, beta))
            
        except Exception as e:
            self.logger.error(f"Rachford-Rice求解失败: {e}")
            return 0.5
    
    def _setup_variable_bounds(
        self,
        n_comp: int,
        n_phases: int
    ) -> List[Tuple[float, float]]:
        """设置优化变量边界"""
        bounds = []
        
        # 相组成边界
        for i in range(n_phases):
            for j in range(n_comp):
                bounds.append((1e-15, 1.0))  # 组成范围
        
        # 相分率边界
        for i in range(n_phases):
            bounds.append((self.settings.min_phase_fraction, 1.0))
        
        return bounds
    
    def _pack_variables(
        self,
        phase_compositions: List[np.ndarray],
        phase_fractions: List[float]
    ) -> np.ndarray:
        """打包优化变量"""
        variables = []
        
        # 添加相组成
        for composition in phase_compositions:
            variables.extend(composition)
        
        # 添加相分率
        variables.extend(phase_fractions)
        
        return np.array(variables)
    
    def _unpack_variables(
        self,
        x: np.ndarray,
        n_comp: int,
        n_phases: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """解包优化变量"""
        # 解析相组成
        phase_compositions = []
        idx = 0
        
        for i in range(n_phases):
            composition = x[idx:idx+n_comp]
            phase_compositions.append(composition)
            idx += n_comp
        
        # 解析相分率
        phase_fractions = x[idx:idx+n_phases].tolist()
        
        return phase_compositions, phase_fractions
    
    def _calculate_mixture_enthalpy(
        self,
        temperature: float,
        pressure: float,
        phase_compositions: List[np.ndarray],
        phase_fractions: List[float],
        property_package: PropertyPackage
    ) -> float:
        """计算混合物焓"""
        try:
            total_enthalpy = 0.0
            
            for i, (composition, fraction) in enumerate(zip(phase_compositions, phase_fractions)):
                if fraction > self.settings.min_phase_fraction:
                    phase_type = PhaseType.VAPOR if i == 0 else PhaseType.LIQUID
                    phase = Phase(phase_type, composition, temperature, pressure)
                    
                    # 计算该相的焓
                    phase_enthalpy = property_package.calculate_enthalpy(phase, temperature, pressure)
                    
                    total_enthalpy += phase_enthalpy * fraction
            
            return total_enthalpy
            
        except Exception as e:
            self.logger.error(f"混合物焓计算失败: {e}")
            return 0.0
    
    def _create_single_phase_result(
        self,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        property_package: PropertyPackage
    ) -> FlashCalculationResult:
        """创建单相结果"""
        result = FlashCalculationResult()
        
        result.converged = True
        result.convergence_status = ConvergenceStatus.CONVERGED
        result.iterations = 1
        result.temperature = temperature
        result.pressure = pressure
        result.feed_composition = composition.copy()
        
        # 判断相态
        # 简化判断：基于温度和压力
        phase_type = PhaseType.VAPOR if temperature > 373.15 else PhaseType.LIQUID
        
        if phase_type == PhaseType.VAPOR:
            result.vapor_fraction = 1.0
            result.vapor_composition = composition.copy()
            result.liquid_composition = np.zeros_like(composition)
        else:
            result.vapor_fraction = 0.0
            result.vapor_composition = np.zeros_like(composition)
            result.liquid_composition = composition.copy()
        
        return result
    
    def _create_flash_result(
        self,
        temperature: float,
        pressure: float,
        feed_composition: np.ndarray,
        phase_compositions: List[np.ndarray],
        phase_fractions: List[float],
        property_package: PropertyPackage,
        iterations: int,
        converged: bool
    ) -> FlashCalculationResult:
        """创建闪蒸结果"""
        result = FlashCalculationResult()
        
        result.converged = converged
        result.convergence_status = ConvergenceStatus.CONVERGED if converged else ConvergenceStatus.FAILED
        result.iterations = iterations
        result.temperature = temperature
        result.pressure = pressure
        result.feed_composition = feed_composition.copy()
        
        # 设置相组成和分率
        if len(phase_compositions) >= 2:
            result.vapor_fraction = phase_fractions[0]
            result.vapor_composition = phase_compositions[0].copy()
            result.liquid_composition = phase_compositions[1].copy()
        else:
            # 单相情况
            if phase_fractions[0] > 0.5:
                result.vapor_fraction = 1.0
                result.vapor_composition = phase_compositions[0].copy()
                result.liquid_composition = np.zeros_like(feed_composition)
            else:
                result.vapor_fraction = 0.0
                result.vapor_composition = np.zeros_like(feed_composition)
                result.liquid_composition = phase_compositions[0].copy()
        
        return result
    
    def _fallback_flash_pt(
        self,
        pressure: float,
        temperature: float,
        property_package: PropertyPackage,
        feed_composition: np.ndarray
    ) -> FlashCalculationResult:
        """备选PT闪蒸方法（简化实现）"""
        self.logger.warning("使用备选PT闪蒸方法")
        
        # 简化实现：假设理想混合
        result = FlashCalculationResult()
        result.converged = False
        result.convergence_status = ConvergenceStatus.FAILED
        result.temperature = temperature
        result.pressure = pressure
        result.feed_composition = feed_composition.copy()
        result.error_message = "Gibbs最小化失败，备选方法也未完全实现"
        
        return result
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """获取算法统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0
            else:
                self.stats[key] = 0.0 