"""
DWSIM热力学计算库 - 理想气体物性包
==================================

实现理想气体热力学模型，适用于低压气相系统。
假设分子间无相互作用，遵循理想气体状态方程 PV=nRT。

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
from typing import List, Optional
from ..core.property_package import PropertyPackage, FlashResult, PropertyPackageParameters
from ..core.enums import PackageType, PhaseType, ConvergenceStatus
from ..core.compound import Compound
from ..core.phase import Phase

class IdealPropertyPackage(PropertyPackage):
    """理想气体物性包
    
    实现理想气体热力学模型的完整物性包。
    适用于低压气相系统和组分挥发性相近的混合物。
    """
    
    def __init__(
        self,
        compounds: List[Compound],
        parameters: Optional[PropertyPackageParameters] = None
    ):
        """初始化理想气体物性包
        
        Args:
            compounds: 化合物列表
            parameters: 物性包参数，None时使用默认参数
        """
        super().__init__(PackageType.IDEAL, compounds, parameters)
        
        # 理想气体特定参数
        self.gas_constant = 8.314  # J/mol/K
        
        # 缓存标准状态的理想气体性质
        self._standard_properties = {}
        self._calculate_standard_properties()
    
    def _calculate_standard_properties(self):
        """计算标准状态下的理想气体性质"""
        T_std = 298.15  # K
        P_std = 101325.0  # Pa
        
        for i, compound in enumerate(self.compounds):
            self._standard_properties[compound.name] = {
                "cp": compound.calculate_ideal_gas_cp(T_std),
                "vapor_pressure": compound.calculate_vapor_pressure(T_std)
            }
    
    def calculate_fugacity_coefficient(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> np.ndarray:
        """计算逸度系数
        
        对于理想气体，逸度系数始终为1。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            np.ndarray: 逸度系数数组（全部为1）
        """
        self.calculation_stats["property_calls"] += 1
        
        # 理想气体的逸度系数为1
        return np.ones(self.n_components)
    
    def calculate_activity_coefficient(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> np.ndarray:
        """计算活度系数
        
        对于理想气体/理想溶液，活度系数始终为1。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            np.ndarray: 活度系数数组（全部为1）
        """
        self.calculation_stats["property_calls"] += 1
        
        # 理想溶液的活度系数为1
        return np.ones(self.n_components)
    
    def calculate_compressibility_factor(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算压缩因子
        
        对于理想气体，压缩因子始终为1。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 压缩因子（1.0）
        """
        self.calculation_stats["property_calls"] += 1
        
        # 理想气体的压缩因子为1
        return 1.0
    
    def calculate_enthalpy_departure(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算焓偏差
        
        对于理想气体，焓偏差为0。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 焓偏差（0.0）[J/mol]
        """
        self.calculation_stats["property_calls"] += 1
        
        # 理想气体的焓偏差为0
        return 0.0
    
    def calculate_entropy_departure(
        self,
        phase: Phase,
        temperature: float,
        pressure: float
    ) -> float:
        """计算熵偏差
        
        对于理想气体，熵偏差为0。
        
        Args:
            phase: 相对象
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            
        Returns:
            float: 熵偏差（0.0）[J/mol/K]
        """
        self.calculation_stats["property_calls"] += 1
        
        # 理想气体的熵偏差为0
        return 0.0
    
    def calculate_k_values(
        self,
        temperature: float,
        pressure: float,
        liquid_composition: np.ndarray,
        vapor_composition: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """计算平衡常数K值
        
        对于理想系统，K值基于Antoine方程或Raoult定律计算。
        
        Args:
            temperature: 温度 [K]
            pressure: 压力 [Pa]
            liquid_composition: 液相组成
            vapor_composition: 气相组成（理想系统中不影响K值）
            
        Returns:
            np.ndarray: K值数组
        """
        self.calculation_stats["property_calls"] += 1
        
        k_values = np.zeros(self.n_components)
        
        for i, compound in enumerate(self.compounds):
            try:
                # 使用Antoine方程计算饱和蒸汽压
                p_sat = compound.calculate_vapor_pressure(temperature)
                
                # Raoult定律：K = P_sat / P
                k_values[i] = p_sat / pressure
                
            except ValueError:
                # 如果无法计算饱和蒸汽压，使用简化估算
                # 基于对比温度的估算
                tc = compound.properties.critical_temperature
                if tc > 0:
                    tr = temperature / tc
                    # 简化的K值估算
                    k_values[i] = np.exp(5.0 * (1 - 1/tr))
                else:
                    # 默认值
                    k_values[i] = 1.0
        
        return k_values
    
    def flash_pt(
        self,
        feed_composition: np.ndarray,
        pressure: float,
        temperature: float
    ) -> FlashResult:
        """PT闪蒸计算
        
        使用理想系统的Rachford-Rice方程求解。
        
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
            
            if abs(np.sum(feed_composition) - 1.0) > 1e-6:
                raise ValueError("进料组成总和必须为1")
            
            # 标准化进料组成
            z = np.array(feed_composition) / np.sum(feed_composition)
            
            # 计算K值
            k_values = self.calculate_k_values(temperature, pressure, z)
            
            # 检查是否为单相
            k_min = np.min(k_values)
            k_max = np.max(k_values)
            
            if k_max <= 1.0:
                # 全液相
                result.vapor_fraction = 0.0
                result.liquid_phase = Phase(PhaseType.LIQUID, self.compounds, z)
                result.liquid_phase.set_temperature_pressure(temperature, pressure)
                result.converged = True
                result.convergence_status = ConvergenceStatus.CONVERGED
                
            elif k_min >= 1.0:
                # 全气相
                result.vapor_fraction = 1.0
                result.vapor_phase = Phase(PhaseType.VAPOR, self.compounds, z)
                result.vapor_phase.set_temperature_pressure(temperature, pressure)
                result.converged = True
                result.convergence_status = ConvergenceStatus.CONVERGED
                
            else:
                # 两相平衡 - 求解Rachford-Rice方程
                beta = self._solve_rachford_rice(z, k_values)
                
                if beta is not None:
                    # 计算相组成
                    x = z / (1 + beta * (k_values - 1))  # 液相组成
                    y = k_values * x                      # 气相组成
                    
                    # 创建相对象
                    result.vapor_fraction = beta
                    result.liquid_phase = Phase(PhaseType.LIQUID, self.compounds, x)
                    result.vapor_phase = Phase(PhaseType.VAPOR, self.compounds, y)
                    
                    # 设置温度压力
                    result.liquid_phase.set_temperature_pressure(temperature, pressure)
                    result.vapor_phase.set_temperature_pressure(temperature, pressure)
                    
                    result.converged = True
                    result.convergence_status = ConvergenceStatus.CONVERGED
                    
                    # 计算物性
                    result.enthalpy = self._calculate_mixture_enthalpy(result, temperature, pressure)
                    result.entropy = self._calculate_mixture_entropy(result, temperature, pressure)
                    result.volume = self._calculate_mixture_volume(result, temperature, pressure)
                    
                else:
                    result.converged = False
                    result.convergence_status = ConvergenceStatus.MAX_ITERATIONS
                    result.error_message = "Rachford-Rice方程求解失败"
            
        except Exception as e:
            result.converged = False
            result.convergence_status = ConvergenceStatus.ERROR
            result.error_message = str(e)
        
        return result
    
    def _solve_rachford_rice(self, z: np.ndarray, k: np.ndarray) -> Optional[float]:
        """求解Rachford-Rice方程
        
        方程：∑[z_i * (K_i - 1) / (1 + β * (K_i - 1))] = 0
        
        Args:
            z: 进料组成
            k: K值数组
            
        Returns:
            Optional[float]: 汽化率β，None表示求解失败
        """
        # 确定β的搜索范围
        k_minus_1 = k - 1.0
        
        # 避免除零
        epsilon = 1e-12
        
        # 计算β的上下界
        beta_min = -1.0 / np.max(k_minus_1[k_minus_1 > epsilon]) if np.any(k_minus_1 > epsilon) else -1e6
        beta_max = -1.0 / np.min(k_minus_1[k_minus_1 < -epsilon]) if np.any(k_minus_1 < -epsilon) else 1e6
        
        # 确保搜索范围合理
        beta_min = max(beta_min + epsilon, 0.0 + epsilon)
        beta_max = min(beta_max - epsilon, 1.0 - epsilon)
        
        if beta_min >= beta_max:
            return None
        
        # 使用二分法求解
        for iteration in range(self.parameters.flash_max_iterations):
            beta = (beta_min + beta_max) / 2.0
            
            # 计算Rachford-Rice函数值
            rr_function = np.sum(z * k_minus_1 / (1.0 + beta * k_minus_1))
            
            if abs(rr_function) < self.parameters.flash_tolerance:
                return beta
            
            # 更新搜索范围
            if rr_function > 0:
                beta_min = beta
            else:
                beta_max = beta
        
        return None
    
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
        """计算混合物体积"""
        if result.vapor_fraction == 0.0:
            # 液相体积（使用简化估算）
            return result.liquid_phase.molecular_weight / 1000.0  # 假设液体密度~1000 kg/m³
        elif result.vapor_fraction == 1.0:
            # 气相体积（理想气体）
            return self.gas_constant * temperature / pressure
        else:
            v_l = result.liquid_phase.molecular_weight / 1000.0
            v_v = self.gas_constant * temperature / pressure
            return (1 - result.vapor_fraction) * v_l + result.vapor_fraction * v_v
    
    def flash_ph(
        self,
        feed_composition: np.ndarray,
        pressure: float,
        enthalpy: float
    ) -> FlashResult:
        """PH闪蒸计算
        
        通过迭代求解温度，使计算的焓等于给定焓。
        
        Args:
            feed_composition: 进料组成
            pressure: 压力 [Pa]
            enthalpy: 摩尔焓 [J/mol]
            
        Returns:
            FlashResult: 闪蒸计算结果
        """
        self.calculation_stats["flash_calls"] += 1
        
        # 初始温度估算
        T_guess = 298.15  # K
        T_min = 200.0     # K
        T_max = 800.0     # K
        
        tolerance = 1000.0  # J/mol
        
        for iteration in range(self.parameters.flash_max_iterations):
            # 在当前温度下进行PT闪蒸
            pt_result = self.flash_pt(feed_composition, pressure, T_guess)
            
            if not pt_result.converged:
                break
            
            # 检查焓是否匹配
            h_error = pt_result.enthalpy - enthalpy
            
            if abs(h_error) < tolerance:
                return pt_result
            
            # 更新温度估算
            if h_error > 0:
                T_max = T_guess
            else:
                T_min = T_guess
            
            T_guess = (T_min + T_max) / 2.0
        
        # 返回失败结果
        result = FlashResult()
        result.pressure = pressure
        result.converged = False
        result.convergence_status = ConvergenceStatus.MAX_ITERATIONS
        result.error_message = "PH闪蒸温度求解失败"
        
        return result
    
    def validate_configuration(self) -> List[str]:
        """验证理想气体物性包配置"""
        errors = super().validate_configuration()
        
        # 检查理想气体适用性
        for compound in self.compounds:
            if compound.properties.critical_pressure > 0:
                # 检查是否在理想气体适用范围内
                # 这里可以添加更多的验证逻辑
                pass
        
        return errors
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "名称": "理想气体物性包",
            "类型": self.package_type.value,
            "适用范围": "低压气相系统，理想混合物",
            "状态方程": "PV = nRT",
            "逸度系数": "φᵢ = 1",
            "活度系数": "γᵢ = 1",
            "压缩因子": "Z = 1",
            "组分数量": self.n_components,
            "组分列表": [comp.name for comp in self.compounds]
        }

__all__ = ["IdealPropertyPackage"] 