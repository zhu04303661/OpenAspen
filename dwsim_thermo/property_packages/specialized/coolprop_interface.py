"""
CoolProp接口模块
===============

CoolProp是一个开源的热物性数据库和计算库，提供高精度的热力学性质计算。
本模块提供与CoolProp的接口，对应DWSIM的CoolProp.vb (1,962行)。

CoolProp特点：
1. 高精度的状态方程（REFPROP级别）
2. 广泛的化合物数据库
3. 多种状态方程支持
4. 快速的计算速度
5. 开源免费

支持的状态方程：
- Helmholtz能量方程
- 立方状态方程
- 多参数状态方程
- 对应态方程

作者：OpenAspen项目组
版本：1.0.0
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

try:
    import CoolProp.CoolProp as CP
    from CoolProp.CoolProp import PropsSI, PhaseSI, get_global_param_string
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    warnings.warn("CoolProp未安装，CoolProp接口功能将不可用")

from ..base_property_package import PropertyPackageBase
from ...core.compound import CompoundProperties
from ...core.enums import PhaseType
from ...core.exceptions import CalculationError, PropertyNotAvailableError


@dataclass
class CoolPropSettings:
    """CoolProp设置"""
    
    # 计算设置
    backend: str = "HEOS"                    # 后端类型 (HEOS, REFPROP, etc.)
    reference_state: str = "IIR"             # 参考状态
    units: str = "SI"                        # 单位系统
    
    # 数值设置
    tolerance: float = 1e-8                  # 收敛容差
    max_iterations: int = 200                # 最大迭代次数
    
    # 混合物设置
    mixing_rules: str = "Lorentz-Berthelot"  # 混合规则
    departure_functions: bool = True          # 是否使用偏差函数
    
    # 缓存设置
    enable_cache: bool = True                # 启用缓存
    cache_size: int = 1000                   # 缓存大小


class CoolPropInterface:
    """CoolProp接口类
    
    提供与CoolProp库的完整接口，支持：
    1. 纯组分性质计算
    2. 混合物性质计算
    3. 相平衡计算
    4. 传输性质计算
    5. 热力学循环分析
    """
    
    def __init__(self, settings: Optional[CoolPropSettings] = None):
        """初始化CoolProp接口
        
        Args:
            settings: CoolProp设置
        """
        if not COOLPROP_AVAILABLE:
            raise ImportError("CoolProp库未安装，请先安装CoolProp")
        
        self.settings = settings or CoolPropSettings()
        self.logger = logging.getLogger("CoolPropInterface")
        
        # 初始化CoolProp
        self._initialize_coolprop()
        
        # 支持的化合物映射
        self.compound_mapping = self._initialize_compound_mapping()
        
        # 计算统计
        self.stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'calculation_failures': 0,
            'average_calculation_time': 0.0
        }
        
        # 属性缓存
        self._property_cache = {} if self.settings.enable_cache else None
    
    def _initialize_coolprop(self):
        """初始化CoolProp设置"""
        
        try:
            # 设置全局参数
            CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, "")
            CP.set_config_bool(CP.NORMALIZE_GAS_CONSTANTS, True)
            
            # 设置参考状态
            if self.settings.reference_state:
                CP.set_reference_stateS(
                    "Water", self.settings.reference_state)
            
            self.logger.info(f"CoolProp初始化完成，版本: {CP.get_global_param_string('version')}")
            
        except Exception as e:
            self.logger.error(f"CoolProp初始化失败: {e}")
            raise CalculationError(f"CoolProp初始化失败: {e}")
    
    def _initialize_compound_mapping(self) -> Dict[str, str]:
        """初始化化合物名称映射"""
        
        # DWSIM化合物名称到CoolProp名称的映射
        mapping = {
            # 烷烃
            'methane': 'Methane',
            'ethane': 'Ethane',
            'propane': 'Propane',
            'n-butane': 'n-Butane',
            'isobutane': 'IsoButane',
            'n-pentane': 'n-Pentane',
            'isopentane': 'Isopentane',
            'n-hexane': 'n-Hexane',
            'n-heptane': 'n-Heptane',
            'n-octane': 'n-Octane',
            'n-nonane': 'n-Nonane',
            'n-decane': 'n-Decane',
            
            # 烯烃
            'ethylene': 'Ethylene',
            'propylene': 'Propylene',
            '1-butene': '1-Butene',
            
            # 芳烃
            'benzene': 'Benzene',
            'toluene': 'Toluene',
            'ethylbenzene': 'EthylBenzene',
            'o-xylene': 'o-Xylene',
            'm-xylene': 'm-Xylene',
            'p-xylene': 'p-Xylene',
            
            # 环烷烃
            'cyclopentane': 'Cyclopentane',
            'cyclohexane': 'Cyclohexane',
            
            # 醇类
            'methanol': 'Methanol',
            'ethanol': 'Ethanol',
            '1-propanol': '1-Propanol',
            '2-propanol': '2-Propanol',
            '1-butanol': '1-Butanol',
            
            # 其他有机化合物
            'acetone': 'Acetone',
            'acetonitrile': 'Acetonitrile',
            'dimethyl ether': 'DimethylEther',
            
            # 无机化合物
            'water': 'Water',
            'carbon dioxide': 'CarbonDioxide',
            'carbon monoxide': 'CarbonMonoxide',
            'hydrogen': 'Hydrogen',
            'nitrogen': 'Nitrogen',
            'oxygen': 'Oxygen',
            'argon': 'Argon',
            'helium': 'Helium',
            'hydrogen sulfide': 'HydrogenSulfide',
            'sulfur dioxide': 'SulfurDioxide',
            'ammonia': 'Ammonia',
            
            # 制冷剂
            'r134a': 'R134a',
            'r410a': 'R410A',
            'r404a': 'R404A',
            'r407c': 'R407C',
            'r22': 'R22',
            'r32': 'R32',
            'r125': 'R125',
            'r143a': 'R143a',
        }
        
        return mapping
    
    def get_coolprop_name(self, compound_name: str) -> str:
        """获取CoolProp化合物名称
        
        Args:
            compound_name: DWSIM化合物名称
            
        Returns:
            str: CoolProp化合物名称
        """
        name_lower = compound_name.lower().strip()
        
        if name_lower in self.compound_mapping:
            return self.compound_mapping[name_lower]
        
        # 尝试直接使用原名称
        try:
            # 测试是否为有效的CoolProp名称
            CP.PropsSI('Tcrit', compound_name)
            return compound_name
        except:
            pass
        
        raise PropertyNotAvailableError(
            f"化合物 '{compound_name}' 在CoolProp中不可用")
    
    def calculate_property(
        self,
        property_name: str,
        input1_name: str,
        input1_value: float,
        input2_name: str,
        input2_value: float,
        compound_name: str
    ) -> float:
        """计算纯组分性质
        
        Args:
            property_name: 性质名称
            input1_name: 第一个输入参数名称
            input1_value: 第一个输入参数值
            input2_name: 第二个输入参数名称
            input2_value: 第二个输入参数值
            compound_name: 化合物名称
            
        Returns:
            float: 计算结果
        """
        try:
            import time
            start_time = time.time()
            
            # 检查缓存
            cache_key = (property_name, input1_name, input1_value, 
                        input2_name, input2_value, compound_name)
            
            if self._property_cache and cache_key in self._property_cache:
                self.stats['cache_hits'] += 1
                return self._property_cache[cache_key]
            
            # 获取CoolProp化合物名称
            coolprop_name = self.get_coolprop_name(compound_name)
            
            # 调用CoolProp计算
            result = PropsSI(
                property_name,
                input1_name, input1_value,
                input2_name, input2_value,
                coolprop_name
            )
            
            # 更新统计
            self.stats['total_calculations'] += 1
            calc_time = time.time() - start_time
            self.stats['average_calculation_time'] = (
                (self.stats['average_calculation_time'] * 
                 (self.stats['total_calculations'] - 1) + calc_time) /
                self.stats['total_calculations']
            )
            
            # 缓存结果
            if self._property_cache:
                if len(self._property_cache) >= self.settings.cache_size:
                    # 清理最旧的缓存项
                    oldest_key = next(iter(self._property_cache))
                    del self._property_cache[oldest_key]
                
                self._property_cache[cache_key] = result
            
            self.logger.debug(f"CoolProp计算: {property_name}={result:.6g} "
                            f"({input1_name}={input1_value}, {input2_name}={input2_value})")
            
            return result
            
        except Exception as e:
            self.stats['calculation_failures'] += 1
            self.logger.error(f"CoolProp计算失败: {e}")
            raise CalculationError(f"CoolProp性质计算失败: {e}")
    
    def calculate_mixture_property(
        self,
        property_name: str,
        input1_name: str,
        input1_value: float,
        input2_name: str,
        input2_value: float,
        compound_names: List[str],
        mole_fractions: List[float]
    ) -> float:
        """计算混合物性质
        
        Args:
            property_name: 性质名称
            input1_name: 第一个输入参数名称
            input1_value: 第一个输入参数值
            input2_name: 第二个输入参数名称
            input2_value: 第二个输入参数值
            compound_names: 化合物名称列表
            mole_fractions: 摩尔分数列表
            
        Returns:
            float: 计算结果
        """
        try:
            # 获取CoolProp化合物名称
            coolprop_names = [self.get_coolprop_name(name) for name in compound_names]
            
            # 构建混合物字符串
            mixture_string = "&".join([
                f"{name}[{frac}]" 
                for name, frac in zip(coolprop_names, mole_fractions)
            ])
            
            # 调用CoolProp计算
            result = PropsSI(
                property_name,
                input1_name, input1_value,
                input2_name, input2_value,
                mixture_string
            )
            
            self.logger.debug(f"CoolProp混合物计算: {property_name}={result:.6g}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"CoolProp混合物计算失败: {e}")
            raise CalculationError(f"CoolProp混合物性质计算失败: {e}")
    
    def get_critical_properties(self, compound_name: str) -> Dict[str, float]:
        """获取临界性质
        
        Args:
            compound_name: 化合物名称
            
        Returns:
            Dict[str, float]: 临界性质字典
        """
        try:
            coolprop_name = self.get_coolprop_name(compound_name)
            
            properties = {
                'critical_temperature': PropsSI('Tcrit', coolprop_name),      # K
                'critical_pressure': PropsSI('Pcrit', coolprop_name),         # Pa
                'critical_density': PropsSI('Rhocrit', coolprop_name),        # kg/m³
                'acentric_factor': PropsSI('acentric', coolprop_name),        # -
                'triple_point_temperature': PropsSI('Ttriple', coolprop_name), # K
                'triple_point_pressure': PropsSI('Ptriple', coolprop_name),   # Pa
                'molecular_weight': PropsSI('M', coolprop_name),              # kg/mol
                'gas_constant': PropsSI('gas_constant', coolprop_name),       # J/(mol·K)
            }
            
            return properties
            
        except Exception as e:
            self.logger.error(f"获取临界性质失败: {e}")
            raise CalculationError(f"获取临界性质失败: {e}")
    
    def get_saturation_properties(
        self,
        temperature: float,
        compound_name: str
    ) -> Dict[str, float]:
        """获取饱和性质
        
        Args:
            temperature: 温度 (K)
            compound_name: 化合物名称
            
        Returns:
            Dict[str, float]: 饱和性质字典
        """
        try:
            coolprop_name = self.get_coolprop_name(compound_name)
            
            properties = {
                'saturation_pressure': PropsSI('P', 'T', temperature, 'Q', 0, coolprop_name),
                'liquid_density': PropsSI('D', 'T', temperature, 'Q', 0, coolprop_name),
                'vapor_density': PropsSI('D', 'T', temperature, 'Q', 1, coolprop_name),
                'liquid_enthalpy': PropsSI('H', 'T', temperature, 'Q', 0, coolprop_name),
                'vapor_enthalpy': PropsSI('H', 'T', temperature, 'Q', 1, coolprop_name),
                'liquid_entropy': PropsSI('S', 'T', temperature, 'Q', 0, coolprop_name),
                'vapor_entropy': PropsSI('S', 'T', temperature, 'Q', 1, coolprop_name),
                'liquid_cp': PropsSI('C', 'T', temperature, 'Q', 0, coolprop_name),
                'vapor_cp': PropsSI('C', 'T', temperature, 'Q', 1, coolprop_name),
                'surface_tension': PropsSI('I', 'T', temperature, 'Q', 0, coolprop_name),
            }
            
            # 计算汽化潜热
            properties['latent_heat'] = (properties['vapor_enthalpy'] - 
                                       properties['liquid_enthalpy'])
            
            return properties
            
        except Exception as e:
            self.logger.error(f"获取饱和性质失败: {e}")
            raise CalculationError(f"获取饱和性质失败: {e}")
    
    def get_transport_properties(
        self,
        temperature: float,
        pressure: float,
        compound_name: str,
        phase: str = "gas"
    ) -> Dict[str, float]:
        """获取传输性质
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            compound_name: 化合物名称
            phase: 相态 ("gas", "liquid")
            
        Returns:
            Dict[str, float]: 传输性质字典
        """
        try:
            coolprop_name = self.get_coolprop_name(compound_name)
            
            properties = {
                'viscosity': PropsSI('V', 'T', temperature, 'P', pressure, coolprop_name),
                'thermal_conductivity': PropsSI('L', 'T', temperature, 'P', pressure, coolprop_name),
                'prandtl_number': PropsSI('Prandtl', 'T', temperature, 'P', pressure, coolprop_name),
            }
            
            return properties
            
        except Exception as e:
            self.logger.error(f"获取传输性质失败: {e}")
            raise CalculationError(f"获取传输性质失败: {e}")
    
    def flash_calculation(
        self,
        input1_name: str,
        input1_value: float,
        input2_name: str,
        input2_value: float,
        compound_names: List[str],
        mole_fractions: List[float]
    ) -> Dict[str, Any]:
        """闪蒸计算
        
        Args:
            input1_name: 第一个输入参数名称
            input1_value: 第一个输入参数值
            input2_name: 第二个输入参数名称
            input2_value: 第二个输入参数值
            compound_names: 化合物名称列表
            mole_fractions: 摩尔分数列表
            
        Returns:
            Dict[str, Any]: 闪蒸结果
        """
        try:
            # 获取CoolProp化合物名称
            coolprop_names = [self.get_coolprop_name(name) for name in compound_names]
            
            # 构建混合物字符串
            mixture_string = "&".join([
                f"{name}[{frac}]" 
                for name, frac in zip(coolprop_names, mole_fractions)
            ])
            
            # 计算基本性质
            temperature = PropsSI('T', input1_name, input1_value, 
                                input2_name, input2_value, mixture_string)
            pressure = PropsSI('P', input1_name, input1_value, 
                             input2_name, input2_value, mixture_string)
            
            # 判断相态
            phase = PhaseSI(input1_name, input1_value, 
                          input2_name, input2_value, mixture_string)
            
            # 计算其他性质
            density = PropsSI('D', input1_name, input1_value, 
                            input2_name, input2_value, mixture_string)
            enthalpy = PropsSI('H', input1_name, input1_value, 
                             input2_name, input2_value, mixture_string)
            entropy = PropsSI('S', input1_name, input1_value, 
                            input2_name, input2_value, mixture_string)
            
            result = {
                'temperature': temperature,
                'pressure': pressure,
                'phase': phase,
                'density': density,
                'enthalpy': enthalpy,
                'entropy': entropy,
                'converged': True,
                'error_message': None
            }
            
            # 如果是两相，计算相分率和组成
            if 'twophase' in phase.lower() or 'liquid_gas' in phase.lower():
                try:
                    vapor_fraction = PropsSI('Q', input1_name, input1_value, 
                                           input2_name, input2_value, mixture_string)
                    result['vapor_fraction'] = vapor_fraction
                    result['liquid_fraction'] = 1.0 - vapor_fraction
                except:
                    # 如果无法计算相分率，设为单相
                    result['vapor_fraction'] = 1.0 if 'gas' in phase.lower() else 0.0
                    result['liquid_fraction'] = 1.0 - result['vapor_fraction']
            else:
                result['vapor_fraction'] = 1.0 if 'gas' in phase.lower() else 0.0
                result['liquid_fraction'] = 1.0 - result['vapor_fraction']
            
            return result
            
        except Exception as e:
            self.logger.error(f"CoolProp闪蒸计算失败: {e}")
            return {
                'converged': False,
                'error_message': str(e),
                'temperature': None,
                'pressure': None,
                'phase': None
            }
    
    def get_available_compounds(self) -> List[str]:
        """获取可用化合物列表
        
        Returns:
            List[str]: 可用化合物名称列表
        """
        try:
            # 获取CoolProp支持的所有化合物
            fluids = CP.get_global_param_string("FluidsList").split(',')
            
            # 过滤并排序
            available_compounds = []
            for fluid in fluids:
                fluid = fluid.strip()
                if fluid and not fluid.startswith('INCOMP::'):
                    available_compounds.append(fluid)
            
            available_compounds.sort()
            
            return available_compounds
            
        except Exception as e:
            self.logger.error(f"获取可用化合物列表失败: {e}")
            return list(self.compound_mapping.values())
    
    def validate_compound(self, compound_name: str) -> bool:
        """验证化合物是否可用
        
        Args:
            compound_name: 化合物名称
            
        Returns:
            bool: 是否可用
        """
        try:
            self.get_coolprop_name(compound_name)
            return True
        except:
            return False
    
    def get_backend_info(self) -> Dict[str, Any]:
        """获取后端信息
        
        Returns:
            Dict[str, Any]: 后端信息
        """
        try:
            info = {
                'version': CP.get_global_param_string('version'),
                'gitrevision': CP.get_global_param_string('gitrevision'),
                'backend': self.settings.backend,
                'available_backends': CP.get_global_param_string('backend_string').split(','),
                'fluid_count': len(self.get_available_compounds()),
                'settings': {
                    'reference_state': self.settings.reference_state,
                    'units': self.settings.units,
                    'tolerance': self.settings.tolerance,
                    'max_iterations': self.settings.max_iterations
                },
                'stats': self.stats
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取后端信息失败: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """清空缓存"""
        if self._property_cache:
            self._property_cache.clear()
            self.logger.info("CoolProp缓存已清空")


class CoolPropPropertyPackage(PropertyPackageBase):
    """CoolProp物性包
    
    基于CoolProp库的高精度物性包实现。
    """
    
    def __init__(self, compounds: List[CompoundProperties], 
                 settings: Optional[CoolPropSettings] = None):
        super().__init__(compounds)
        
        self.coolprop = CoolPropInterface(settings)
        self.name = "CoolProp"
        self.description = "基于CoolProp库的高精度物性包"
        
        # 验证所有化合物都可用
        self._validate_compounds()
    
    def _validate_compounds(self):
        """验证化合物可用性"""
        
        unavailable_compounds = []
        
        for compound in self.compounds:
            if not self.coolprop.validate_compound(compound.name):
                unavailable_compounds.append(compound.name)
        
        if unavailable_compounds:
            self.logger.warning(f"以下化合物在CoolProp中不可用: {unavailable_compounds}")
    
    def calculate_property(
        self,
        property_name: str,
        temperature: float,
        pressure: float,
        composition: np.ndarray,
        phase_type: PhaseType = PhaseType.VAPOR
    ) -> Union[float, np.ndarray]:
        """计算性质
        
        Args:
            property_name: 性质名称
            temperature: 温度 (K)
            pressure: 压力 (Pa)
            composition: 摩尔组成
            phase_type: 相态类型
            
        Returns:
            Union[float, np.ndarray]: 计算结果
        """
        try:
            compound_names = [compound.name for compound in self.compounds]
            
            if len(self.compounds) == 1:
                # 纯组分
                return self.coolprop.calculate_property(
                    property_name, 'T', temperature, 'P', pressure,
                    compound_names[0]
                )
            else:
                # 混合物
                return self.coolprop.calculate_mixture_property(
                    property_name, 'T', temperature, 'P', pressure,
                    compound_names, composition.tolist()
                )
                
        except Exception as e:
            self.logger.error(f"CoolProp性质计算失败: {e}")
            raise CalculationError(f"性质计算失败: {e}")


# 导出主要类
__all__ = [
    'CoolPropInterface',
    'CoolPropSettings',
    'CoolPropPropertyPackage'
] 