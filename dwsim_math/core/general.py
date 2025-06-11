"""
通用数学函数模块
================

包含各种通用的数学运算函数，如最大值、最小值、求和、加权平均等。
这些函数是从DWSIM.Math/General.vb转换而来的。

主要功能:
- 数组统计函数（最大值、最小值、求和）
- 加权运算
- 数组操作工具

作者: DWSIM团队 (Python转换版本)
许可证: GNU General Public License v3.0
"""

import numpy as np
from typing import List, Union, Optional, Any
import warnings


class MathCommon:
    """
    通用数学函数类
    
    提供各种数组和数值的统计计算功能，包括最大值、最小值、求和、
    加权平均等常用数学运算。
    """
    
    @staticmethod
    def copy_to_vector(array_list: List[List[float]], index: int) -> np.ndarray:
        """
        从二维数组列表中提取指定索引的列，创建一维向量
        
        参数:
            array_list: 二维数组列表，每个元素是一个包含多个值的列表
            index: 要提取的列索引
            
        返回:
            np.ndarray: 提取的一维向量
            
        示例:
            >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> MathCommon.copy_to_vector(data, 1)
            array([2., 5., 8.])
        """
        if not array_list:
            return np.array([])
            
        try:
            values = [row[index] for row in array_list]
            return np.array(values, dtype=float)
        except (IndexError, TypeError) as e:
            raise ValueError(f"无法从数组列表中提取索引 {index} 的数据: {e}")
    
    @staticmethod
    def max_with_mask(values: Union[List[float], np.ndarray], 
                      mask: Union[List[float], np.ndarray]) -> float:
        """
        计算数组中非零掩码位置的最大值
        
        参数:
            values: 数值数组
            mask: 掩码数组，非零位置的值会被考虑
            
        返回:
            float: 掩码非零位置的最大值
            
        示例:
            >>> values = [1, 5, 3, 8, 2]
            >>> mask = [1, 0, 1, 1, 0]  # 只考虑位置0,2,3的值
            >>> MathCommon.max_with_mask(values, mask)
            8.0
        """
        values = np.asarray(values, dtype=float)
        mask = np.asarray(mask, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        if len(values) != len(mask):
            raise ValueError("值数组和掩码数组长度必须相同")
        
        # 创建布尔掩码（非零位置）
        valid_mask = mask != 0
        
        if not np.any(valid_mask):
            return values[0] if len(values) > 0 else 0.0
            
        return float(np.max(values[valid_mask]))
    
    @staticmethod
    def max_value(values: Union[List[float], np.ndarray]) -> float:
        """
        计算数组的最大值
        
        参数:
            values: 数值数组
            
        返回:
            float: 数组的最大值
            
        示例:
            >>> MathCommon.max_value([1, 5, 3, 8, 2])
            8.0
        """
        values = np.asarray(values, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        return float(np.max(values))
    
    @staticmethod
    def min_with_mask(values: Union[List[float], np.ndarray], 
                      mask: Union[List[float], np.ndarray]) -> float:
        """
        计算数组中非零掩码位置的最小值
        
        参数:
            values: 数值数组
            mask: 掩码数组，非零位置的值会被考虑
            
        返回:
            float: 掩码非零位置的最小值
            
        示例:
            >>> values = [1, 5, 3, 8, 2]
            >>> mask = [1, 0, 1, 1, 0]  # 只考虑位置0,2,3的值
            >>> MathCommon.min_with_mask(values, mask)
            1.0
        """
        values = np.asarray(values, dtype=float)
        mask = np.asarray(mask, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        if len(values) != len(mask):
            raise ValueError("值数组和掩码数组长度必须相同")
        
        # 创建布尔掩码（非零位置）
        valid_mask = mask != 0
        
        if not np.any(valid_mask):
            return values[0] if len(values) > 0 else 0.0
            
        return float(np.min(values[valid_mask]))
    
    @staticmethod
    def min_value(values: Union[List[float], np.ndarray]) -> float:
        """
        计算数组的最小值（忽略零值）
        
        参数:
            values: 数值数组
            
        返回:
            float: 数组的最小值（忽略零值）
            
        注意:
            这个函数会忽略数组中的零值，只考虑非零元素的最小值
            
        示例:
            >>> MathCommon.min_value([0, 5, 3, 8, 0, 2])
            2.0
        """
        values = np.asarray(values, dtype=float)
        
        if len(values) == 0:
            return 0.0
        
        # 过滤掉零值
        non_zero_values = values[values != 0]
        
        if len(non_zero_values) == 0:
            return values[0] if len(values) > 0 else 0.0
            
        return float(np.min(non_zero_values))
    
    @staticmethod
    def sum_array(values: Union[List[float], np.ndarray]) -> float:
        """
        计算数组元素的和
        
        参数:
            values: 数值数组
            
        返回:
            float: 数组元素的和
            
        示例:
            >>> MathCommon.sum_array([1, 2, 3, 4, 5])
            15.0
        """
        values = np.asarray(values, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        return float(np.sum(values))
    
    @staticmethod
    def abs_sum(values: Union[List[float], np.ndarray]) -> float:
        """
        计算数组元素绝对值的和
        
        参数:
            values: 数值数组
            
        返回:
            float: 数组元素绝对值的和
            
        示例:
            >>> MathCommon.abs_sum([-1, 2, -3, 4, -5])
            15.0
        """
        values = np.asarray(values, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        return float(np.sum(np.abs(values)))
    
    @staticmethod
    def weighted_average(weights: Union[List[float], np.ndarray], 
                        values: Union[List[float], np.ndarray]) -> float:
        """
        计算加权平均值
        
        参数:
            weights: 权重数组
            values: 数值数组
            
        返回:
            float: 加权平均值
            
        公式:
            weighted_avg = Σ(weight_i * value_i) / Σ(weight_i)
            
        示例:
            >>> weights = [0.3, 0.3, 0.4]
            >>> values = [10, 20, 30]
            >>> MathCommon.weighted_average(weights, values)
            21.0
        """
        weights = np.asarray(weights, dtype=float)
        values = np.asarray(values, dtype=float)
        
        if len(weights) != len(values):
            raise ValueError("权重数组和数值数组长度必须相同")
            
        if len(weights) == 0:
            return 0.0
        
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            raise ZeroDivisionError("权重总和为零，无法计算加权平均值")
            
        weighted_sum = np.sum(weights * values)
        return float(weighted_sum / total_weight)
    
    @staticmethod
    def sum_of_squares(values: Union[List[float], np.ndarray]) -> float:
        """
        计算数组元素的平方和
        
        参数:
            values: 数值数组
            
        返回:
            float: 数组元素的平方和
            
        公式:
            sum_of_squares = Σ(value_i²)
            
        示例:
            >>> MathCommon.sum_of_squares([1, 2, 3, 4])
            30.0  # 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
        """
        values = np.asarray(values, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        return float(np.sum(values ** 2))
    
    @staticmethod
    def standard_deviation(values: Union[List[float], np.ndarray], 
                          sample: bool = True) -> float:
        """
        计算标准差
        
        参数:
            values: 数值数组
            sample: 是否为样本标准差（使用n-1作为分母），默认为True
            
        返回:
            float: 标准差
            
        示例:
            >>> values = [1, 2, 3, 4, 5]
            >>> MathCommon.standard_deviation(values)
            1.5811388300841898
        """
        values = np.asarray(values, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        if len(values) == 1:
            return 0.0
            
        return float(np.std(values, ddof=1 if sample else 0))
    
    @staticmethod
    def variance(values: Union[List[float], np.ndarray], 
                sample: bool = True) -> float:
        """
        计算方差
        
        参数:
            values: 数值数组
            sample: 是否为样本方差（使用n-1作为分母），默认为True
            
        返回:
            float: 方差
            
        示例:
            >>> values = [1, 2, 3, 4, 5]
            >>> MathCommon.variance(values)
            2.5
        """
        values = np.asarray(values, dtype=float)
        
        if len(values) == 0:
            return 0.0
            
        if len(values) == 1:
            return 0.0
            
        return float(np.var(values, ddof=1 if sample else 0))


# 为了保持向后兼容性，提供一些别名函数
def copy_to_vector(array_list: List[List[float]], index: int) -> np.ndarray:
    """copy_to_vector函数的别名"""
    return MathCommon.copy_to_vector(array_list, index)

def max_value(values: Union[List[float], np.ndarray]) -> float:
    """max_value函数的别名"""
    return MathCommon.max_value(values)

def min_value(values: Union[List[float], np.ndarray]) -> float:
    """min_value函数的别名"""
    return MathCommon.min_value(values)

def sum_array(values: Union[List[float], np.ndarray]) -> float:
    """sum_array函数的别名"""
    return MathCommon.sum_array(values)

def weighted_average(weights: Union[List[float], np.ndarray], 
                    values: Union[List[float], np.ndarray]) -> float:
    """weighted_average函数的别名"""
    return MathCommon.weighted_average(weights, values) 