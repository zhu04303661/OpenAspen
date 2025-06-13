"""
DWSIM热力学计算库 - Peng-Robinson状态方程物性包
===============================================

实现Peng-Robinson状态方程的完整热力学计算模型。
适用于气液相平衡、高压系统、天然气处理等工业应用。

状态方程：P = RT/(V-b) - a(T)/(V²+2bV-b²)

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import fsolve
from ..core.property_package import PropertyPackage, FlashResult, PropertyPackageParameters
from ..core.enums import PackageType, PhaseType, ConvergenceStatus, MixingRule
from ..core.compound import Compound
from ..core.phase import Phase

class PengRobinsonPackage(PropertyPackage):
    """Peng-Robinson状态方程物性包
    
    实现完整的Peng-Robinson状态方程热力学计算。
    适用于石油化工、炼油工业、天然气处理等领域。
    """
    
    def __init__(
        self,
        compounds: List[Compound],
        parameters: Optional[PropertyPackageParameters] = None,
        mixing_rule: MixingRule = MixingRule.VAN_DER_WAALS
    ):
        """初始化Peng-Robinson物性包
        
        Args:
            compounds: 化合物列表
            parameters: 物性包参数，None时使用默认参数
            mixing_rule: 混合规则，默认使用van der Waals混合规则
        """
        super().__init__(PackageType.PENG_ROBINSON, compounds, parameters)
        
        self.mixing_rule = mixing_rule
        self.gas_constant = 8.314  # J/mol/K
        
        # Peng-Robinson常数
        self.omega_a = 0.45724
        self.omega_b = 0.07780
        
        # 预计算纯组分PR参数
        self._calculate_pure_component_parameters()
        
        # 缓存计算参数
        self._last_calc_conditions = None
        self._cached_mixing_parameters = None
    
    def _calculate_pure_component_parameters(self):
        """计算纯组分的PR参数"""
        self.a_critical = np.zeros(self.n_components)
        self.b_critical = np.zeros(self.n_components)
        self.kappa = np.zeros(self.n_components)
        
        for i, compound in enumerate(self.compounds):
            props = compound.properties
            
            # 临界参数
            tc = props.critical_temperature  # K
            pc = props.critical_pressure     # Pa
            omega = props.acentric_factor    # 偏心因子
            
            if tc <= 0 or pc <= 0:
                raise ValueError(f"组分{compound.name}缺少有效的临界常数")
            
            # PR参数
            self.a_critical[i] = self.omega_a * (self.gas_constant * tc)**2 / pc
            self.b_critical[i] = self.omega_b * self.gas_constant * tc / pc
            
            # κ参数
            if omega <= 0.491:
                self.kappa[i] = 0.37464 + 1.54226*omega - 0.26992*omega**2
            else:
                # 修正的κ公式，适用于重组分
                self.kappa[i] = 0.379642 + 1.48503*omega - 0.164423*omega**2 + 0.016666*omega**3
    
    def _calculate_temperature_dependent_parameters(self, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算温度相关的PR参数
        
        Args:
            temperature: 温度 [K]
            
        Returns:
            tuple: (a数组, b数组)
        """
        a_i = np.zeros(self.n_components)
        b_i = self.b_critical.copy()
        
        for i, compound in enumerate(self.compounds):
            tc = compound.properties.critical_temperature
            tr = temperature / tc  # 对比温度
            
            # α函数
            alpha = (1 + self.kappa[i] * (1 - np.sqrt(tr)))**2
            
            # 温度相关的a参数
            a_i[i] = self.a_critical[i] * alpha
        
        return a_i, b_i
    
    def _calculate_mixing_parameters(self, composition: np.ndarray, temperature: float) -> Tuple[float, float]:
        """计算混合参数
        
        Args:
            composition: 摩尔分数组成
            temperature: 温度 [K]
            
        Returns:
            tuple: (a_mix, b_mix)
        """
        # 获取纯组分参数
        a_i, b_i = self._calculate_temperature_dependent_parameters(temperature)
        
        # van der Waals混合规则
        if self.mixing_rule == MixingRule.VAN_DER_WAALS:
            # b混合
            b_mix = np.sum(composition * b_i)
            
            # a混合
            a_mix = 0.0
            for i in range(self.n_components):
                for j in range(self.n_components):
                    # 几何平均 + 二元交互参数
                    kij = self.binary_parameters[i, j]
                    a_ij = np.sqrt(a_i[i] * a_i[j]) * (1 - kij)
                    a_mix += composition[i] * composition[j] * a_ij
            
            return a_mix, b_mix
        
        else:
            raise NotImplementedError(f"混合规则{self.mixing_rule}尚未实现")
    
    def calculate_compressibility_factor(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算压缩因子
        
        通过求解PR状态方程的三次方程得到压缩因子。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 压缩因子
        """
        self.calculation_stats["property_calls"] += 1
        
        # 计算混合参数
        a_mix, b_mix = self._calculate_mixing_parameters(phase.mole_fractions, temperature)
        
        # 无量纲参数
        A = a_mix * pressure / (self.gas_constant * temperature)**2
        B = b_mix * pressure / (self.gas_constant * temperature)
        
        # 三次方程系数: Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
        coeffs = [
            1,                           # Z³
            -(1 - B),                    # Z²
            A - 3*B**2 - 2*B,           # Z
            -(A*B - B**2 - B**3)        # 常数项
        ]
        
        # 求解三次方程
        roots = np.roots(coeffs)
        
        # 筛选出实数根
        real_roots = []
        for root in roots:
            if np.isreal(root) and np.real(root) > B:  # Z必须大于B
                real_roots.append(np.real(root))
        
        if not real_roots:
            raise ValueError("未找到有效的压缩因子根")
        
        # 根据相态选择合适的根
        if phase.phase_type == PhaseType.VAPOR:
            # 气相选择最大根
            return max(real_roots)
        else:
            # 液相选择最小根
            return min(real_roots)
    
    def calculate_fugacity_coefficient(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> np.ndarray:
        """计算逸度系数
        
        使用PR状态方程计算各组分的逸度系数。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            np.ndarray: 逸度系数数组
        """
        self.calculation_stats["property_calls"] += 1
        
        # 获取组成和PR参数
        x = phase.mole_fractions
        a_i, b_i = self._calculate_temperature_dependent_parameters(temperature)
        a_mix, b_mix = self._calculate_mixing_parameters(x, temperature)
        
        # 压缩因子
        Z = self.calculate_compressibility_factor(phase, temperature, pressure)
        
        # 无量纲参数
        A = a_mix * pressure / (self.gas_constant * temperature)**2
        B = b_mix * pressure / (self.gas_constant * temperature)
        
        # 逸度系数计算
        phi = np.zeros(self.n_components)
        
        sqrt2 = np.sqrt(2)
        
        for i in range(self.n_components):
            # ∂(na_mix)/∂n_i 计算
            da_dn_i = 0.0
            for j in range(self.n_components):
                kij = self.binary_parameters[i, j]
                a_ij = np.sqrt(a_i[i] * a_i[j]) * (1 - kij)
                da_dn_i += 2 * x[j] * a_ij
            
            # 逸度系数公式
            term1 = (b_i[i] / b_mix) * (Z - 1)
            term2 = -np.log(Z - B)
            term3 = -(A / (2 * sqrt2 * B)) * (
                (2 * da_dn_i / a_mix) - (b_i[i] / b_mix)
            ) * np.log((Z + B * (1 + sqrt2)) / (Z + B * (1 - sqrt2)))
            
            ln_phi_i = term1 + term2 + term3
            phi[i] = np.exp(ln_phi_i)
        
        return phi
    
    def calculate_activity_coefficient(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> np.ndarray:
        """计算活度系数
        
        对于PR状态方程，活度系数通过逸度系数关联计算。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            np.ndarray: 活度系数数组
        """
        self.calculation_stats["property_calls"] += 1
        
        # 对于EOS模型，活度系数通常设为1，逸度系数承担非理想性
        # 更严格的处理需要考虑标准态的选择
        return np.ones(self.n_components)
    
    def calculate_enthalpy_departure(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算焓偏差
        
        使用PR状态方程计算焓偏差。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 焓偏差 [J/mol]
        """
        self.calculation_stats["property_calls"] += 1
        
        # 获取PR参数
        x = phase.mole_fractions
        a_i, b_i = self._calculate_temperature_dependent_parameters(temperature)
        a_mix, b_mix = self._calculate_mixing_parameters(x, temperature)
        
        # 压缩因子
        Z = self.calculate_compressibility_factor(phase, temperature, pressure)
        
        # 计算da_mix/dT
        da_dT = 0.0
        for i in range(self.n_components):
            for j in range(self.n_components):
                kij = self.binary_parameters[i, j]
                
                # 计算各组分的da_i/dT
                tc_i = self.compounds[i].properties.critical_temperature
                tc_j = self.compounds[j].properties.critical_temperature
                
                tr_i = temperature / tc_i
                tr_j = temperature / tc_j
                
                dalpha_dT_i = -self.kappa[i] * np.sqrt(1 + self.kappa[i] * (1 - np.sqrt(tr_i))) / (np.sqrt(tr_i) * tc_i)
                dalpha_dT_j = -self.kappa[j] * np.sqrt(1 + self.kappa[j] * (1 - np.sqrt(tr_j))) / (np.sqrt(tr_j) * tc_j)
                
                da_i_dT = self.a_critical[i] * dalpha_dT_i
                da_j_dT = self.a_critical[j] * dalpha_dT_j
                
                da_ij_dT = 0.5 * (1 - kij) * (
                    da_i_dT / np.sqrt(a_i[i] * a_i[j]) * np.sqrt(a_i[i] * a_i[j]) +
                    da_j_dT / np.sqrt(a_i[i] * a_i[j]) * np.sqrt(a_i[i] * a_i[j])
                )
                
                da_dT += x[i] * x[j] * da_ij_dT
        
        # 焓偏差公式
        sqrt2 = np.sqrt(2)
        B = b_mix * pressure / (self.gas_constant * temperature)
        
        h_dep = (
            self.gas_constant * temperature * (Z - 1) -
            (temperature * da_dT - a_mix) / (2 * sqrt2 * b_mix) *
            np.log((Z + B * (1 + sqrt2)) / (Z + B * (1 - sqrt2)))
        )
        
        return h_dep
    
    def calculate_entropy_departure(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算熵偏差
        
        使用PR状态方程计算熵偏差。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 熵偏差 [J/mol/K]
        """
        self.calculation_stats["property_calls"] += 1
        
        # 获取PR参数
        x = phase.mole_fractions
        a_mix, b_mix = self._calculate_mixing_parameters(x, temperature)
        
        # 压缩因子
        Z = self.calculate_compressibility_factor(phase, temperature, pressure)
        
        # 计算da_mix/dT（与焓偏差计算相同）
        da_dT = self._calculate_da_dt(x, temperature)
        
        # 熵偏差公式
        sqrt2 = np.sqrt(2)
        B = b_mix * pressure / (self.gas_constant * temperature)
        
        s_dep = (
            self.gas_constant * np.log(Z - B) -
            da_dT / (2 * sqrt2 * b_mix) *
            np.log((Z + B * (1 + sqrt2)) / (Z + B * (1 - sqrt2)))
        )
        
        return s_dep
    
    def _calculate_da_dt(self, composition: np.ndarray, temperature: float) -> float:
        """计算da_mix/dT"""
        x = composition
        da_dT = 0.0
        
        for i in range(self.n_components):
            for j in range(self.n_components):
                kij = self.binary_parameters[i, j]
                
                # 计算各组分的da_i/dT
                tc_i = self.compounds[i].properties.critical_temperature
                tc_j = self.compounds[j].properties.critical_temperature
                
                tr_i = temperature / tc_i
                tr_j = temperature / tc_j
                
                # α函数的温度导数
                alpha_i = (1 + self.kappa[i] * (1 - np.sqrt(tr_i)))**2
                alpha_j = (1 + self.kappa[j] * (1 - np.sqrt(tr_j)))**2
                
                dalpha_dT_i = -self.kappa[i] * (1 + self.kappa[i] * (1 - np.sqrt(tr_i))) / (np.sqrt(tr_i) * tc_i)
                dalpha_dT_j = -self.kappa[j] * (1 + self.kappa[j] * (1 - np.sqrt(tr_j))) / (np.sqrt(tr_j) * tc_j)
                
                da_i_dT = self.a_critical[i] * dalpha_dT_i
                da_j_dT = self.a_critical[j] * dalpha_dT_j
                
                # 混合项的温度导数
                a_i_val = self.a_critical[i] * alpha_i
                a_j_val = self.a_critical[j] * alpha_j
                
                da_ij_dT = 0.5 * (1 - kij) * (
                    da_i_dT * np.sqrt(a_j_val / a_i_val) +
                    da_j_dT * np.sqrt(a_i_val / a_j_val)
                )
                
                da_dT += x[i] * x[j] * da_ij_dT
        
        return da_dT
    
    def flash_pt(
        self,
        feed_composition: np.ndarray,
        pressure: float,
        temperature: float
    ) -> FlashResult:
        """PT闪蒸计算
        
        使用嵌套循环算法结合PR状态方程进行PT闪蒸。
        
        Args:
            feed_composition: 进料组成
            pressure: 压力 [Pa]
            temperature: 温度 [K]
            
        Returns:
            FlashResult: 闪蒸计算结果
        """
        self.calculation_stats["flash_calls"] += 1
        
        result = FlashResult()
        result.temperature = temperature
        result.pressure = pressure
        
        try:
            # 验证输入
            if len(feed_composition) != self.n_components:
                raise ValueError("进料组成长度与组分数量不匹配")
            
            z = np.array(feed_composition) / np.sum(feed_composition)
            
            # 初始K值估算（使用Wilson方程）
            k_values = self._estimate_wilson_k_values(temperature, pressure)
            
            # 嵌套循环迭代
            for outer_iter in range(self.parameters.max_iterations):
                # 求解Rachford-Rice方程
                beta = self._solve_rachford_rice_pr(z, k_values)
                
                if beta is None:
                    result.converged = False
                    result.convergence_status = ConvergenceStatus.DIVERGENCE
                    result.error_message = "Rachford-Rice方程求解失败"
                    return result
                
                # 更新相组成
                if beta <= 1e-8:
                    # 全液相
                    x = z.copy()
                    y = k_values * x
                    beta = 0.0
                elif beta >= 1.0 - 1e-8:
                    # 全气相
                    y = z.copy()
                    x = y / k_values
                    beta = 1.0
                else:
                    # 两相
                    x = z / (1 + beta * (k_values - 1))
                    y = k_values * x
                
                # 标准化组成
                x = x / np.sum(x)
                y = y / np.sum(y)
                
                # 创建相对象计算新的K值
                liquid_phase = Phase(PhaseType.LIQUID, self.compounds, x)
                vapor_phase = Phase(PhaseType.VAPOR, self.compounds, y)
                
                liquid_phase.set_temperature_pressure(temperature, pressure)
                vapor_phase.set_temperature_pressure(temperature, pressure)
                
                # 计算逸度系数
                phi_l = self.calculate_fugacity_coefficient(liquid_phase, temperature, pressure)
                phi_v = self.calculate_fugacity_coefficient(vapor_phase, temperature, pressure)
                
                # 新的K值
                k_new = phi_l / phi_v
                
                # 检查收敛
                k_error = np.max(np.abs((k_new - k_values) / k_values))
                
                if k_error < self.parameters.flash_tolerance:
                    # 收敛
                    result.vapor_fraction = beta
                    result.converged = True
                    result.convergence_status = ConvergenceStatus.CONVERGED
                    result.iterations = outer_iter + 1
                    result.residual = k_error
                    
                    if beta > 1e-8:
                        result.vapor_phase = vapor_phase
                    if beta < 1.0 - 1e-8:
                        result.liquid_phase = liquid_phase
                    
                    # 计算物性
                    result.enthalpy = self._calculate_mixture_enthalpy(result, temperature, pressure)
                    result.entropy = self._calculate_mixture_entropy(result, temperature, pressure)
                    result.volume = self._calculate_mixture_volume(result, temperature, pressure)
                    
                    return result
                
                # 更新K值（可添加阻尼）
                damping = self.parameters.damping_factor
                k_values = k_values * (1 - damping) + k_new * damping
            
            # 未收敛
            result.converged = False
            result.convergence_status = ConvergenceStatus.MAX_ITERATIONS
            result.error_message = f"达到最大迭代次数{self.parameters.max_iterations}"
            result.iterations = self.parameters.max_iterations
            
        except Exception as e:
            result.converged = False
            result.convergence_status = ConvergenceStatus.ERROR
            result.error_message = str(e)
        
        return result
    
    def _estimate_wilson_k_values(self, temperature: float, pressure: float) -> np.ndarray:
        """使用Wilson方程估算初始K值"""
        k_values = np.zeros(self.n_components)
        
        for i, compound in enumerate(self.compounds):
            props = compound.properties
            
            if props.critical_temperature > 0 and props.critical_pressure > 0:
                tr = temperature / props.critical_temperature
                pr = pressure / props.critical_pressure
                
                # Wilson方程
                k_values[i] = (props.critical_pressure / pressure) * np.exp(
                    5.373 * (1 + props.acentric_factor) * (1 - 1/tr)
                )
            else:
                k_values[i] = 1.0
        
        return k_values
    
    def _solve_rachford_rice_pr(self, z: np.ndarray, k: np.ndarray) -> Optional[float]:
        """求解Rachford-Rice方程（PR版本）"""
        # 与理想气体版本类似，但可能需要特殊处理
        return self._solve_rachford_rice_bisection(z, k)
    
    def _solve_rachford_rice_bisection(self, z: np.ndarray, k: np.ndarray) -> Optional[float]:
        """使用二分法求解Rachford-Rice方程"""
        k_minus_1 = k - 1.0
        epsilon = 1e-12
        
        # 计算β的范围
        if np.all(np.abs(k_minus_1) < epsilon):
            return 0.5  # 所有K值都为1，任何β都可以
        
        # 计算上下界
        positive_indices = k_minus_1 > epsilon
        negative_indices = k_minus_1 < -epsilon
        
        if not np.any(positive_indices) and not np.any(negative_indices):
            return 0.5
        
        if np.any(positive_indices):
            beta_min = max(-1.0 / np.max(k_minus_1[positive_indices]), 0.0) + epsilon
        else:
            beta_min = 0.0 + epsilon
        
        if np.any(negative_indices):
            beta_max = min(-1.0 / np.min(k_minus_1[negative_indices]), 1.0) - epsilon
        else:
            beta_max = 1.0 - epsilon
        
        if beta_min >= beta_max:
            return None
        
        # 二分法求解
        for _ in range(50):  # 最大50次迭代
            beta = (beta_min + beta_max) / 2.0
            
            # Rachford-Rice函数
            rr_value = np.sum(z * k_minus_1 / (1.0 + beta * k_minus_1))
            
            if abs(rr_value) < self.parameters.flash_tolerance:
                return beta
            
            if rr_value > 0:
                beta_min = beta
            else:
                beta_max = beta
        
        return (beta_min + beta_max) / 2.0  # 返回最后的估算值
    
    def _calculate_mixture_enthalpy(self, result: FlashResult, temperature: float, pressure: float) -> float:
        """计算混合物焓"""
        if result.vapor_fraction == 0.0:
            return self.calculate_enthalpy(result.liquid_phase, temperature, pressure)
        elif result.vapor_fraction == 1.0:
            return self.calculate_enthalpy(result.vapor_phase, temperature, pressure)
        else:
            h_l = self.calculate_enthalpy(result.liquid_phase, temperature, pressure)
            h_v = self.calculate_enthalpy(result.vapor_phase, temperature, pressure)
            return (1 - result.vapor_fraction) * h_l + result.vapor_fraction * h_v
    
    def _calculate_mixture_entropy(self, result: FlashResult, temperature: float, pressure: float) -> float:
        """计算混合物熵"""
        if result.vapor_fraction == 0.0:
            return self.calculate_entropy(result.liquid_phase, temperature, pressure)
        elif result.vapor_fraction == 1.0:
            return self.calculate_entropy(result.vapor_phase, temperature, pressure)
        else:
            s_l = self.calculate_entropy(result.liquid_phase, temperature, pressure)
            s_v = self.calculate_entropy(result.vapor_phase, temperature, pressure)
            return (1 - result.vapor_fraction) * s_l + result.vapor_fraction * s_v
    
    def _calculate_mixture_volume(self, result: FlashResult, temperature: float, pressure: float) -> float:
        """计算混合物摩尔体积"""
        if result.vapor_fraction == 0.0:
            z_l = self.calculate_compressibility_factor(result.liquid_phase, temperature, pressure)
            return z_l * self.gas_constant * temperature / pressure
        elif result.vapor_fraction == 1.0:
            z_v = self.calculate_compressibility_factor(result.vapor_phase, temperature, pressure)
            return z_v * self.gas_constant * temperature / pressure
        else:
            z_l = self.calculate_compressibility_factor(result.liquid_phase, temperature, pressure)
            z_v = self.calculate_compressibility_factor(result.vapor_phase, temperature, pressure)
            v_l = z_l * self.gas_constant * temperature / pressure
            v_v = z_v * self.gas_constant * temperature / pressure
            return (1 - result.vapor_fraction) * v_l + result.vapor_fraction * v_v
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "名称": "Peng-Robinson状态方程",
            "类型": self.package_type.value,
            "适用范围": "气液相平衡，高压系统，石油化工",
            "状态方程": "P = RT/(V-b) - a(T)/(V²+2bV-b²)",
            "混合规则": self.mixing_rule.value,
            "组分数量": self.n_components,
            "组分列表": [comp.name for comp in self.compounds],
            "二元交互参数": "支持",
            "相稳定性": "支持气液相平衡"
        }

__all__ = ["PengRobinsonPackage"] 