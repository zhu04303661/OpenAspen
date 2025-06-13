"""
DWSIM热力学计算库 - 核心枚举定义
============================================

定义了热力学计算中使用的所有枚举类型，包括相态类型、闪蒸规格、
物性包类型、计算方法等核心枚举。

作者：OpenAspen项目组
版本：1.0.0
"""

from enum import Enum, IntEnum, auto
from typing import Dict, List

class PhaseType(Enum):
    """相态类型枚举"""
    VAPOR = "V"           # 气相
    LIQUID = "L"          # 液相
    LIQUID1 = "L1"        # 液相1（液液分相）
    LIQUID2 = "L2"        # 液相2（液液分相）
    SOLID = "S"           # 固相
    AQUEOUS = "Aq"        # 水相
    UNKNOWN = "Unknown"   # 未知相态

class FlashSpec(Enum):
    """闪蒸规格枚举"""
    P = "P"               # 压力
    T = "T"               # 温度
    H = "H"               # 焓
    S = "S"               # 熵
    V = "V"               # 体积
    U = "U"               # 内能
    PT = "PT"             # 温度-压力闪蒸
    PH = "PH"             # 压力-焓闪蒸
    PS = "PS"             # 压力-熵闪蒸
    TV = "TV"             # 温度-体积闪蒸
    PV = "PV"             # 压力-体积闪蒸
    UV = "UV"             # 内能-体积闪蒸
    TH = "TH"             # 温度-焓闪蒸
    TS = "TS"             # 温度-熵闪蒸
    VF = "VF"             # 汽化率闪蒸
    DEWP = "DEWP"         # 露点闪蒸
    BUBP = "BUBP"         # 泡点闪蒸

class PackageType(Enum):
    """物性包类型枚举"""
    IDEAL = "Ideal"                      # 理想气体
    PENG_ROBINSON = "Peng-Robinson"      # Peng-Robinson状态方程
    SRK = "SRK"                         # Soave-Redlich-Kwong状态方程
    PR_WS = "PR-WS"                     # Peng-Robinson-Wong-Sandler
    PC_SAFT = "PC-SAFT"                 # PC-SAFT状态方程
    SAFT = "SAFT"                       # SAFT状态方程
    NRTL = "NRTL"                       # NRTL活度系数模型
    UNIQUAC = "UNIQUAC"                 # UNIQUAC活度系数模型
    UNIFAC = "UNIFAC"                   # UNIFAC预测模型
    WILSON = "Wilson"                   # Wilson活度系数模型
    HENRY = "Henry"                     # Henry定律
    STEAM_TABLES = "SteamTables"        # 蒸汽表
    SEAWATER = "Seawater"               # 海水模型
    SOUR_WATER = "SourWater"            # 酸性水模型
    BLACK_OIL = "BlackOil"              # 黑油模型
    ELECTROLYTE_NRTL = "ElectrolyteNRTL" # 电解质NRTL模型

class FlashAlgorithmType(Enum):
    """闪蒸算法类型枚举"""
    NESTED_LOOPS = "NestedLoops"         # 嵌套循环算法
    INSIDE_OUT = "InsideOut"             # 内外循环算法
    GIBBS_MINIMIZATION = "GibbsMin"      # Gibbs自由能最小化
    STABILITY_TEST = "StabilityTest"     # 相稳定性测试
    CRITICAL_POINT = "CriticalPoint"     # 临界点计算

class PropertyType(Enum):
    """物性类型枚举"""
    # 热力学性质
    TEMPERATURE = "T"                    # 温度 [K]
    PRESSURE = "P"                       # 压力 [Pa]
    VOLUME = "V"                         # 体积 [m³/mol]
    ENTHALPY = "H"                       # 焓 [J/mol]
    ENTROPY = "S"                        # 熵 [J/mol/K]
    GIBBS_ENERGY = "G"                   # Gibbs自由能 [J/mol]
    HELMHOLTZ_ENERGY = "A"               # Helmholtz自由能 [J/mol]
    INTERNAL_ENERGY = "U"                # 内能 [J/mol]
    HEAT_CAPACITY_CP = "Cp"              # 定压热容 [J/mol/K]
    HEAT_CAPACITY_CV = "Cv"              # 定容热容 [J/mol/K]
    DENSITY = "Rho"                      # 密度 [kg/m³]
    MOLECULAR_WEIGHT = "MW"              # 分子量 [kg/mol]
    COMPRESSIBILITY_FACTOR = "Z"         # 压缩因子 [-]
    FUGACITY = "f"                       # 逸度 [Pa]
    FUGACITY_COEFFICIENT = "phi"         # 逸度系数 [-]
    ACTIVITY_COEFFICIENT = "gamma"       # 活度系数 [-]
    
    # 输运性质
    VISCOSITY = "mu"                     # 粘度 [Pa·s]
    THERMAL_CONDUCTIVITY = "k"           # 导热系数 [W/m/K]
    SURFACE_TENSION = "sigma"            # 表面张力 [N/m]
    DIFFUSIVITY = "D"                    # 扩散系数 [m²/s]

class CalculationMode(Enum):
    """计算模式枚举"""
    RIGOROUS = "Rigorous"                # 严格计算
    APPROXIMATE = "Approximate"          # 近似计算
    CORRELATIONS = "Correlations"        # 关联式计算

class ConvergenceStatus(IntEnum):
    """收敛状态枚举"""
    NOT_STARTED = 0                      # 未开始
    CONVERGED = 1                        # 已收敛
    MAX_ITERATIONS = 2                   # 达到最大迭代次数
    STAGNATION = 3                       # 计算停滞
    DIVERGENCE = 4                       # 发散
    ERROR = 5                            # 计算错误

class MixingRule(Enum):
    """混合规则枚举"""
    VAN_DER_WAALS = "VdW"                # van der Waals混合规则
    WONG_SANDLER = "WS"                  # Wong-Sandler混合规则
    MHV1 = "MHV1"                        # Modified Huron-Vidal 1
    MHV2 = "MHV2"                        # Modified Huron-Vidal 2
    LCVM = "LCVM"                        # Linear Combination of Vidal and Michelsen
    HVID = "HVID"                        # Huron-Vidal-Infinite-Dilution

class ComponentType(Enum):
    """组分类型枚举"""
    NORMAL = "Normal"                    # 普通组分
    PSEUDO = "Pseudo"                    # 拟组分
    ION = "Ion"                          # 离子
    SALT = "Salt"                        # 盐类
    POLYMER = "Polymer"                  # 聚合物

class DatabaseType(Enum):
    """数据库类型枚举"""
    BUILT_IN = "BuiltIn"                 # 内置数据库
    CHEMSEP = "ChemSep"                  # ChemSep数据库
    NIST = "NIST"                        # NIST数据库
    DIPPR = "DIPPR"                      # DIPPR数据库
    COOLPROP = "CoolProp"                # CoolProp数据库
    CHEDL = "ChEDL"                      # ChEDL数据库
    CHEMEO = "Chemeo"                    # Chemeo数据库
    DDB = "DDB"                          # Dortmund数据库
    KDB = "KDB"                          # 韩国数据库

# 枚举工具函数
class EnumUtils:
    """枚举工具类"""
    
    @staticmethod
    def get_phase_types() -> List[PhaseType]:
        """获取所有相态类型"""
        return list(PhaseType)
    
    @staticmethod
    def get_flash_specs() -> List[FlashSpec]:
        """获取所有闪蒸规格"""
        return list(FlashSpec)
    
    @staticmethod
    def get_package_types() -> List[PackageType]:
        """获取所有物性包类型"""
        return list(PackageType)
    
    @staticmethod
    def is_valid_flash_spec(spec1: FlashSpec, spec2: FlashSpec) -> bool:
        """检查闪蒸规格组合是否有效"""
        valid_combinations = {
            (FlashSpec.PT, None),
            (FlashSpec.PH, None),
            (FlashSpec.PS, None),
            (FlashSpec.TV, None),
            (FlashSpec.PV, None),
            (FlashSpec.UV, None),
            (FlashSpec.TH, None),
            (FlashSpec.TS, None),
            (FlashSpec.VF, FlashSpec.PT),
        }
        return (spec1, spec2) in valid_combinations or (spec2, spec1) in valid_combinations
    
    @staticmethod
    def get_property_units(prop_type: PropertyType) -> str:
        """获取物性的单位"""
        units_map = {
            PropertyType.TEMPERATURE: "K",
            PropertyType.PRESSURE: "Pa",
            PropertyType.VOLUME: "m³/mol",
            PropertyType.ENTHALPY: "J/mol",
            PropertyType.ENTROPY: "J/mol/K",
            PropertyType.GIBBS_ENERGY: "J/mol",
            PropertyType.HELMHOLTZ_ENERGY: "J/mol",
            PropertyType.INTERNAL_ENERGY: "J/mol",
            PropertyType.HEAT_CAPACITY_CP: "J/mol/K",
            PropertyType.HEAT_CAPACITY_CV: "J/mol/K",
            PropertyType.DENSITY: "kg/m³",
            PropertyType.MOLECULAR_WEIGHT: "kg/mol",
            PropertyType.COMPRESSIBILITY_FACTOR: "-",
            PropertyType.FUGACITY: "Pa",
            PropertyType.FUGACITY_COEFFICIENT: "-",
            PropertyType.ACTIVITY_COEFFICIENT: "-",
            PropertyType.VISCOSITY: "Pa·s",
            PropertyType.THERMAL_CONDUCTIVITY: "W/m/K",
            PropertyType.SURFACE_TENSION: "N/m",
            PropertyType.DIFFUSIVITY: "m²/s",
        }
        return units_map.get(prop_type, "未知单位")

# 常用枚举组合
VAPOR_LIQUID_PHASES = [PhaseType.VAPOR, PhaseType.LIQUID]
LIQUID_LIQUID_PHASES = [PhaseType.LIQUID1, PhaseType.LIQUID2]
ALL_PHASES = [PhaseType.VAPOR, PhaseType.LIQUID, PhaseType.LIQUID1, 
              PhaseType.LIQUID2, PhaseType.SOLID, PhaseType.AQUEOUS]

BASIC_FLASH_SPECS = [FlashSpec.PT, FlashSpec.PH, FlashSpec.PS]
ADVANCED_FLASH_SPECS = [FlashSpec.TV, FlashSpec.PV, FlashSpec.UV, 
                       FlashSpec.TH, FlashSpec.TS]

EOS_PACKAGES = [PackageType.PENG_ROBINSON, PackageType.SRK, 
                PackageType.PC_SAFT, PackageType.SAFT]
ACTIVITY_PACKAGES = [PackageType.NRTL, PackageType.UNIQUAC, 
                     PackageType.UNIFAC, PackageType.WILSON]

__all__ = [
    "PhaseType", "FlashSpec", "PackageType", "FlashAlgorithmType",
    "PropertyType", "CalculationMode", "ConvergenceStatus", "MixingRule",
    "ComponentType", "DatabaseType", "EnumUtils",
    "VAPOR_LIQUID_PHASES", "LIQUID_LIQUID_PHASES", "ALL_PHASES",
    "BASIC_FLASH_SPECS", "ADVANCED_FLASH_SPECS", 
    "EOS_PACKAGES", "ACTIVITY_PACKAGES"
] 