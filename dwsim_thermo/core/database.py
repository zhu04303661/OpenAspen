"""
热力学数据库接口模块

提供化合物数据库管理和热力学参数查询功能。
对应DWSIM原始代码中的Databases.vb (129KB, 2056行)。

功能包括:
- 化合物基本物性数据管理
- 状态方程参数数据库
- 活度系数模型参数数据库
- 二元交互参数数据库
- 数据验证和完整性检查

参考文献:
- DWSIM热力学库VB.NET原始实现
- NIST Webbook数据库
- DIPPR数据库标准

作者: OpenAspen项目组
版本: 1.0.0
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class CompoundProperties:
    """
    化合物基本物性数据类
    
    包含化合物的基本热力学和物理性质
    """
    # 基本信息
    cas_number: str
    name: str
    molecular_formula: str
    molecular_weight: float  # 分子量 [g/mol]
    
    # 临界性质
    critical_temperature: float  # 临界温度 [K]
    critical_pressure: float     # 临界压力 [Pa]
    critical_volume: float       # 临界体积 [m³/mol]
    critical_compressibility: float  # 临界压缩因子
    acentric_factor: float       # 偏心因子
    
    # 其他重要性质
    normal_boiling_point: float  # 标准沸点 [K]
    normal_melting_point: float  # 标准熔点 [K]
    triple_point_temperature: float = 0.0  # 三相点温度 [K]
    triple_point_pressure: float = 0.0     # 三相点压力 [Pa]
    
    # 理想气体性质
    ideal_gas_cp_coeffs: List[float] = field(default_factory=list)  # 理想气体热容系数
    
    # 蒸汽压参数 (Antoine方程: log10(P) = A - B/(T + C))
    antoine_A: float = 0.0
    antoine_B: float = 0.0
    antoine_C: float = 0.0
    antoine_Tmin: float = 0.0
    antoine_Tmax: float = 0.0
    
    # 液体密度参数 (Rackett方程)
    rackett_constant: float = 0.0
    
    # 其他参数
    parachor: float = 0.0        # 分子体积参数
    dipole_moment: float = 0.0   # 偶极矩 [Debye]
    radius_of_gyration: float = 0.0  # 回转半径

@dataclass  
class EOSParameters:
    """状态方程参数数据类"""
    # SRK/PR参数
    kappa_0: float = 0.0  # SRK/PR的κ参数
    kappa_1: float = 0.0  # PRSV修正参数
    
    # PC-SAFT参数
    pc_saft_m: float = 0.0       # 链段数
    pc_saft_sigma: float = 0.0   # 链段直径 [Å]
    pc_saft_epsilon: float = 0.0 # 链段间作用能 [K]
    pc_saft_kappa: float = 0.0   # 缔合体积参数
    pc_saft_epsilon_assoc: float = 0.0  # 缔合能参数 [K]

@dataclass
class ActivityCoefficientParameters:
    """活度系数模型参数数据类"""
    # UNIFAC参数
    unifac_main_group: int = 0
    unifac_sub_group: int = 0
    unifac_R: float = 0.0  # 体积参数
    unifac_Q: float = 0.0  # 面积参数
    
    # UNIQUAC参数
    uniquac_r: float = 0.0  # 分子体积参数
    uniquac_q: float = 0.0  # 分子面积参数
    
    # Wilson参数
    wilson_volume: float = 0.0  # 分子体积 [cm³/mol]

class DatabaseInterface(ABC):
    """数据库接口抽象基类"""
    
    @abstractmethod
    def get_compound_properties(self, identifier: str) -> Optional[CompoundProperties]:
        """获取化合物基本物性"""
        pass
    
    @abstractmethod
    def get_eos_parameters(self, identifier: str, model: str) -> Optional[EOSParameters]:
        """获取状态方程参数"""
        pass
    
    @abstractmethod
    def get_activity_parameters(self, identifier: str) -> Optional[ActivityCoefficientParameters]:
        """获取活度系数模型参数"""
        pass
    
    @abstractmethod
    def get_binary_parameters(self, comp1: str, comp2: str, model: str) -> Optional[Dict[str, float]]:
        """获取二元交互参数"""
        pass

class JSONDatabase(DatabaseInterface):
    """
    基于JSON文件的数据库实现
    
    适用于小型数据集和快速原型开发
    """
    
    def __init__(self, database_path: str):
        """
        初始化JSON数据库
        
        参数:
            database_path: JSON数据库文件路径
        """
        self.database_path = Path(database_path)
        self.logger = logging.getLogger(__name__)
        
        # 加载数据库
        self._load_database()
    
    def _load_database(self):
        """加载JSON数据库"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                # 创建空数据库结构
                self.data = {
                    'compounds': {},
                    'eos_parameters': {},
                    'activity_parameters': {},
                    'binary_parameters': {}
                }
                self._save_database()
                
        except Exception as e:
            self.logger.error(f"加载JSON数据库失败: {e}")
            self.data = {'compounds': {}, 'eos_parameters': {}, 
                        'activity_parameters': {}, 'binary_parameters': {}}
    
    def _save_database(self):
        """保存JSON数据库"""
        try:
            with open(self.database_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存JSON数据库失败: {e}")
    
    def get_compound_properties(self, identifier: str) -> Optional[CompoundProperties]:
        """
        获取化合物基本物性
        
        参数:
            identifier: 化合物标识符 (CAS号或名称)
            
        返回:
            化合物物性对象，如果未找到则返回None
        """
        compound_data = self.data['compounds'].get(identifier)
        if compound_data:
            try:
                return CompoundProperties(**compound_data)
            except Exception as e:
                self.logger.error(f"解析化合物{identifier}数据失败: {e}")
        
        return None
    
    def get_eos_parameters(self, identifier: str, model: str) -> Optional[EOSParameters]:
        """
        获取状态方程参数
        
        参数:
            identifier: 化合物标识符
            model: 状态方程模型名称
            
        返回:
            状态方程参数对象
        """
        eos_data = self.data['eos_parameters'].get(identifier, {}).get(model)
        if eos_data:
            try:
                return EOSParameters(**eos_data)
            except Exception as e:
                self.logger.error(f"解析{identifier}的{model}参数失败: {e}")
        
        return None
    
    def get_activity_parameters(self, identifier: str) -> Optional[ActivityCoefficientParameters]:
        """
        获取活度系数模型参数
        
        参数:
            identifier: 化合物标识符
            
        返回:
            活度系数参数对象
        """
        activity_data = self.data['activity_parameters'].get(identifier)
        if activity_data:
            try:
                return ActivityCoefficientParameters(**activity_data)
            except Exception as e:
                self.logger.error(f"解析{identifier}的活度系数参数失败: {e}")
        
        return None
    
    def get_binary_parameters(self, comp1: str, comp2: str, model: str) -> Optional[Dict[str, float]]:
        """
        获取二元交互参数
        
        参数:
            comp1: 化合物1标识符
            comp2: 化合物2标识符
            model: 模型名称
            
        返回:
            二元参数字典
        """
        # 尝试两种顺序
        key1 = f"{comp1}-{comp2}"
        key2 = f"{comp2}-{comp1}"
        
        binary_data = (self.data['binary_parameters'].get(key1, {}).get(model) or
                      self.data['binary_parameters'].get(key2, {}).get(model))
        
        return binary_data
    
    def add_compound(self, identifier: str, properties: CompoundProperties):
        """
        添加化合物数据
        
        参数:
            identifier: 化合物标识符
            properties: 化合物物性对象
        """
        self.data['compounds'][identifier] = {
            'cas_number': properties.cas_number,
            'name': properties.name,
            'molecular_formula': properties.molecular_formula,
            'molecular_weight': properties.molecular_weight,
            'critical_temperature': properties.critical_temperature,
            'critical_pressure': properties.critical_pressure,
            'critical_volume': properties.critical_volume,
            'critical_compressibility': properties.critical_compressibility,
            'acentric_factor': properties.acentric_factor,
            'normal_boiling_point': properties.normal_boiling_point,
            'normal_melting_point': properties.normal_melting_point,
            'triple_point_temperature': properties.triple_point_temperature,
            'triple_point_pressure': properties.triple_point_pressure,
            'ideal_gas_cp_coeffs': properties.ideal_gas_cp_coeffs,
            'antoine_A': properties.antoine_A,
            'antoine_B': properties.antoine_B,
            'antoine_C': properties.antoine_C,
            'antoine_Tmin': properties.antoine_Tmin,
            'antoine_Tmax': properties.antoine_Tmax,
            'rackett_constant': properties.rackett_constant,
            'parachor': properties.parachor,
            'dipole_moment': properties.dipole_moment,
            'radius_of_gyration': properties.radius_of_gyration
        }
        self._save_database()
        self.logger.info(f"已添加化合物: {identifier}")
    
    def add_eos_parameters(self, identifier: str, model: str, parameters: EOSParameters):
        """
        添加状态方程参数
        
        参数:
            identifier: 化合物标识符
            model: 状态方程模型
            parameters: 参数对象
        """
        if identifier not in self.data['eos_parameters']:
            self.data['eos_parameters'][identifier] = {}
        
        self.data['eos_parameters'][identifier][model] = {
            'kappa_0': parameters.kappa_0,
            'kappa_1': parameters.kappa_1,
            'pc_saft_m': parameters.pc_saft_m,
            'pc_saft_sigma': parameters.pc_saft_sigma,
            'pc_saft_epsilon': parameters.pc_saft_epsilon,
            'pc_saft_kappa': parameters.pc_saft_kappa,
            'pc_saft_epsilon_assoc': parameters.pc_saft_epsilon_assoc
        }
        self._save_database()
        self.logger.info(f"已添加{identifier}的{model}参数")
    
    def add_binary_parameters(self, comp1: str, comp2: str, model: str, parameters: Dict[str, float]):
        """
        添加二元交互参数
        
        参数:
            comp1: 化合物1标识符
            comp2: 化合物2标识符
            model: 模型名称
            parameters: 参数字典
        """
        key = f"{comp1}-{comp2}"
        
        if key not in self.data['binary_parameters']:
            self.data['binary_parameters'][key] = {}
        
        self.data['binary_parameters'][key][model] = parameters
        self._save_database()
        self.logger.info(f"已添加{comp1}-{comp2}的{model}二元参数")

class SQLiteDatabase(DatabaseInterface):
    """
    基于SQLite的数据库实现
    
    适用于大型数据集和生产环境
    """
    
    def __init__(self, database_path: str):
        """
        初始化SQLite数据库
        
        参数:
            database_path: SQLite数据库文件路径
        """
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据库
        self._initialize_database()
    
    def _initialize_database(self):
        """初始化数据库表结构"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # 创建化合物表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compounds (
                    identifier TEXT PRIMARY KEY,
                    cas_number TEXT,
                    name TEXT,
                    molecular_formula TEXT,
                    molecular_weight REAL,
                    critical_temperature REAL,
                    critical_pressure REAL,
                    critical_volume REAL,
                    critical_compressibility REAL,
                    acentric_factor REAL,
                    normal_boiling_point REAL,
                    normal_melting_point REAL,
                    triple_point_temperature REAL,
                    triple_point_pressure REAL,
                    ideal_gas_cp_coeffs TEXT,
                    antoine_A REAL,
                    antoine_B REAL,
                    antoine_C REAL,
                    antoine_Tmin REAL,
                    antoine_Tmax REAL,
                    rackett_constant REAL,
                    parachor REAL,
                    dipole_moment REAL,
                    radius_of_gyration REAL
                )
            ''')
            
            # 创建状态方程参数表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS eos_parameters (
                    identifier TEXT,
                    model TEXT,
                    kappa_0 REAL,
                    kappa_1 REAL,
                    pc_saft_m REAL,
                    pc_saft_sigma REAL,
                    pc_saft_epsilon REAL,
                    pc_saft_kappa REAL,
                    pc_saft_epsilon_assoc REAL,
                    PRIMARY KEY (identifier, model)
                )
            ''')
            
            # 创建活度系数参数表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_parameters (
                    identifier TEXT PRIMARY KEY,
                    unifac_main_group INTEGER,
                    unifac_sub_group INTEGER,
                    unifac_R REAL,
                    unifac_Q REAL,
                    uniquac_r REAL,
                    uniquac_q REAL,
                    wilson_volume REAL
                )
            ''')
            
            # 创建二元参数表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS binary_parameters (
                    comp1 TEXT,
                    comp2 TEXT,
                    model TEXT,
                    parameters TEXT,
                    PRIMARY KEY (comp1, comp2, model)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"初始化SQLite数据库失败: {e}")
    
    def get_compound_properties(self, identifier: str) -> Optional[CompoundProperties]:
        """获取化合物基本物性"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM compounds WHERE identifier = ?', (identifier,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # 解析理想气体热容系数
                cp_coeffs = json.loads(row[14]) if row[14] else []
                
                return CompoundProperties(
                    cas_number=row[1],
                    name=row[2],
                    molecular_formula=row[3],
                    molecular_weight=row[4],
                    critical_temperature=row[5],
                    critical_pressure=row[6],
                    critical_volume=row[7],
                    critical_compressibility=row[8],
                    acentric_factor=row[9],
                    normal_boiling_point=row[10],
                    normal_melting_point=row[11],
                    triple_point_temperature=row[12],
                    triple_point_pressure=row[13],
                    ideal_gas_cp_coeffs=cp_coeffs,
                    antoine_A=row[15],
                    antoine_B=row[16],
                    antoine_C=row[17],
                    antoine_Tmin=row[18],
                    antoine_Tmax=row[19],
                    rackett_constant=row[20],
                    parachor=row[21],
                    dipole_moment=row[22],
                    radius_of_gyration=row[23]
                )
                
        except Exception as e:
            self.logger.error(f"查询化合物{identifier}失败: {e}")
        
        return None
    
    def get_eos_parameters(self, identifier: str, model: str) -> Optional[EOSParameters]:
        """获取状态方程参数"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM eos_parameters WHERE identifier = ? AND model = ?', 
                          (identifier, model))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return EOSParameters(
                    kappa_0=row[2],
                    kappa_1=row[3],
                    pc_saft_m=row[4],
                    pc_saft_sigma=row[5],
                    pc_saft_epsilon=row[6],
                    pc_saft_kappa=row[7],
                    pc_saft_epsilon_assoc=row[8]
                )
                
        except Exception as e:
            self.logger.error(f"查询{identifier}的{model}参数失败: {e}")
        
        return None
    
    def get_activity_parameters(self, identifier: str) -> Optional[ActivityCoefficientParameters]:
        """获取活度系数模型参数"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM activity_parameters WHERE identifier = ?', (identifier,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return ActivityCoefficientParameters(
                    unifac_main_group=row[1],
                    unifac_sub_group=row[2],
                    unifac_R=row[3],
                    unifac_Q=row[4],
                    uniquac_r=row[5],
                    uniquac_q=row[6],
                    wilson_volume=row[7]
                )
                
        except Exception as e:
            self.logger.error(f"查询{identifier}的活度系数参数失败: {e}")
        
        return None
    
    def get_binary_parameters(self, comp1: str, comp2: str, model: str) -> Optional[Dict[str, float]]:
        """获取二元交互参数"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # 尝试两种顺序
            cursor.execute('SELECT parameters FROM binary_parameters WHERE comp1 = ? AND comp2 = ? AND model = ?', 
                          (comp1, comp2, model))
            row = cursor.fetchone()
            
            if not row:
                cursor.execute('SELECT parameters FROM binary_parameters WHERE comp1 = ? AND comp2 = ? AND model = ?', 
                              (comp2, comp1, model))
                row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return json.loads(row[0])
                
        except Exception as e:
            self.logger.error(f"查询{comp1}-{comp2}的{model}二元参数失败: {e}")
        
        return None

class DatabaseManager:
    """
    数据库管理器
    
    提供统一的数据库接口和多数据源管理
    """
    
    def __init__(self, primary_db: DatabaseInterface, fallback_dbs: List[DatabaseInterface] = None):
        """
        初始化数据库管理器
        
        参数:
            primary_db: 主数据库
            fallback_dbs: 备用数据库列表
        """
        self.primary_db = primary_db
        self.fallback_dbs = fallback_dbs or []
        self.logger = logging.getLogger(__name__)
        
        # 缓存机制
        self._compound_cache = {}
        self._eos_cache = {}
        self._activity_cache = {}
        self._binary_cache = {}
    
    def get_compound_properties(self, identifier: str) -> Optional[CompoundProperties]:
        """
        获取化合物基本物性 (带缓存)
        
        参数:
            identifier: 化合物标识符
            
        返回:
            化合物物性对象
        """
        # 检查缓存
        if identifier in self._compound_cache:
            return self._compound_cache[identifier]
        
        # 从主数据库查询
        result = self.primary_db.get_compound_properties(identifier)
        
        # 如果主数据库没有，尝试备用数据库
        if not result:
            for fallback_db in self.fallback_dbs:
                result = fallback_db.get_compound_properties(identifier)
                if result:
                    break
        
        # 缓存结果
        if result:
            self._compound_cache[identifier] = result
        
        return result
    
    def get_eos_parameters(self, identifier: str, model: str) -> Optional[EOSParameters]:
        """获取状态方程参数 (带缓存)"""
        cache_key = f"{identifier}-{model}"
        
        if cache_key in self._eos_cache:
            return self._eos_cache[cache_key]
        
        result = self.primary_db.get_eos_parameters(identifier, model)
        
        if not result:
            for fallback_db in self.fallback_dbs:
                result = fallback_db.get_eos_parameters(identifier, model)
                if result:
                    break
        
        if result:
            self._eos_cache[cache_key] = result
        
        return result
    
    def get_activity_parameters(self, identifier: str) -> Optional[ActivityCoefficientParameters]:
        """获取活度系数模型参数 (带缓存)"""
        if identifier in self._activity_cache:
            return self._activity_cache[identifier]
        
        result = self.primary_db.get_activity_parameters(identifier)
        
        if not result:
            for fallback_db in self.fallback_dbs:
                result = fallback_db.get_activity_parameters(identifier)
                if result:
                    break
        
        if result:
            self._activity_cache[identifier] = result
        
        return result
    
    def get_binary_parameters(self, comp1: str, comp2: str, model: str) -> Optional[Dict[str, float]]:
        """获取二元交互参数 (带缓存)"""
        cache_key = f"{comp1}-{comp2}-{model}"
        
        if cache_key in self._binary_cache:
            return self._binary_cache[cache_key]
        
        result = self.primary_db.get_binary_parameters(comp1, comp2, model)
        
        if not result:
            for fallback_db in self.fallback_dbs:
                result = fallback_db.get_binary_parameters(comp1, comp2, model)
                if result:
                    break
        
        if result:
            self._binary_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """清除所有缓存"""
        self._compound_cache.clear()
        self._eos_cache.clear()
        self._activity_cache.clear()
        self._binary_cache.clear()
        self.logger.info("已清除数据库缓存")
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息
        
        返回:
            数据库信息字典
        """
        return {
            'primary_database': type(self.primary_db).__name__,
            'fallback_databases': [type(db).__name__ for db in self.fallback_dbs],
            'cache_status': {
                'compounds': len(self._compound_cache),
                'eos_parameters': len(self._eos_cache),
                'activity_parameters': len(self._activity_cache),
                'binary_parameters': len(self._binary_cache)
            }
        }

# 默认数据库实例 (单例模式)
_default_database_manager = None

def get_default_database() -> DatabaseManager:
    """
    获取默认数据库管理器实例
    
    返回:
        默认数据库管理器
    """
    global _default_database_manager
    
    if _default_database_manager is None:
        # 创建默认JSON数据库
        default_db_path = Path(__file__).parent.parent / "data" / "thermodynamics.json"
        default_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_db = JSONDatabase(str(default_db_path))
        _default_database_manager = DatabaseManager(json_db)
    
    return _default_database_manager

def set_default_database(database_manager: DatabaseManager):
    """
    设置默认数据库管理器
    
    参数:
        database_manager: 数据库管理器实例
    """
    global _default_database_manager
    _default_database_manager = database_manager 