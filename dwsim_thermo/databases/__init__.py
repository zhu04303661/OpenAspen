"""
热力学数据库模块

提供化合物数据库管理和热力学参数查询功能。
对应DWSIM原始代码中的Databases.vb (129KB, 2056行)。

作者: OpenAspen项目组
版本: 1.0.0
"""

from .database import (
    CompoundProperties, 
    EOSParameters, 
    ActivityCoefficientParameters,
    DatabaseInterface,
    JSONDatabase,
    SQLiteDatabase,
    DatabaseManager,
    get_default_database,
    set_default_database
)

__all__ = [
    'CompoundProperties',
    'EOSParameters', 
    'ActivityCoefficientParameters',
    'DatabaseInterface',
    'JSONDatabase',
    'SQLiteDatabase',
    'DatabaseManager',
    'get_default_database',
    'set_default_database'
] 