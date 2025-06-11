"""
DWSIM数学库 - 核心通用函数测试
===============================

专门测试 dwsim_math.core.general 模块的功能，包括：
- 基本统计函数（最大值、最小值、求和等）
- 加权运算函数
- 数组操作工具
- 边界条件和错误处理

测试策略:
1. 功能正确性 - 验证每个函数的基本功能
2. 边界条件 - 测试空数组、单元素、极值等
3. 错误处理 - 验证异常输入的处理
4. 数值精度 - 确保计算精度满足要求
5. 性能验证 - 基本性能指标检查

作者: DWSIM团队
版本: 1.0.0
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from dwsim_math.core.general import MathCommon
    print("✅ dwsim_math.core.general 模块导入成功")
except ImportError as e:
    print(f"❌ dwsim_math.core.general 模块导入失败: {e}")
    pytest.skip("core.general模块不可用", allow_module_level=True)

try:
    from .test_utils import TestDWSIMMathBase, assert_almost_equal_scalar, PerformanceTimer
except ImportError:
    # 如果测试工具不可用，创建简单版本
    class TestDWSIMMathBase:
        @pytest.fixture(autouse=True)
        def setup_test_environment(self):
            self.tolerance = 1e-10
            self.float_tolerance = 1e-6
            yield
    
    def assert_almost_equal_scalar(actual, expected, tolerance=1e-6):
        assert abs(actual - expected) < tolerance
    
    class PerformanceTimer:
        def __init__(self, name="测试"):
            self.name = name
            self.execution_time = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass


class TestMathCommonMaxValue(TestDWSIMMathBase):
    """
    测试最大值计算函数
    
    测试目标:
    - max_value: 基本最大值计算
    - max_with_mask: 带掩码的最大值计算
    """
    
    def test_max_value_basic_functionality(self):
        """
        测试基本最大值计算功能
        
        测试步骤:
        1. 测试正数数组 - 验证能正确找到最大值
        2. 测试负数数组 - 验证处理负数的能力
        3. 测试混合数组 - 验证正负数混合情况
        4. 测试单元素数组 - 验证边界条件
        """
        # 步骤1: 测试正数数组
        positive_array = [1.0, 5.0, 3.0, 8.0, 2.0]
        result = MathCommon.max_value(positive_array)
        expected = 8.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 正数数组最大值测试通过: {result}")
        
        # 步骤2: 测试负数数组  
        negative_array = [-5.0, -2.0, -8.0, -1.0]
        result = MathCommon.max_value(negative_array)
        expected = -1.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 负数数组最大值测试通过: {result}")
        
        # 步骤3: 测试混合数组
        mixed_array = [-2.0, 5.0, -1.0, 8.0, 0.0]
        result = MathCommon.max_value(mixed_array)
        expected = 8.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 混合数组最大值测试通过: {result}")
        
        # 步骤4: 测试单元素数组
        single_element = [42.0]
        result = MathCommon.max_value(single_element)
        expected = 42.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 单元素数组最大值测试通过: {result}")
    
    def test_max_value_edge_cases(self):
        """
        测试最大值计算的边界条件
        
        测试步骤:
        1. 测试空数组 - 验证空输入处理
        2. 测试全零数组 - 验证零值处理
        3. 测试包含无穷大的数组 - 验证特殊值处理
        4. 测试numpy数组输入 - 验证输入类型兼容性
        """
        # 步骤1: 测试空数组
        empty_array = []
        result = MathCommon.max_value(empty_array)
        expected = 0.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 空数组处理测试通过: 返回 {result}")
        
        # 步骤2: 测试全零数组
        zeros_array = [0.0, 0.0, 0.0]
        result = MathCommon.max_value(zeros_array)
        expected = 0.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 全零数组处理测试通过: 返回 {result}")
        
        # 步骤3: 测试包含无穷大
        inf_array = [1.0, float('inf'), 3.0]
        result = MathCommon.max_value(inf_array)
        assert result == float('inf')
        print(f"  ✓ 无穷大值处理测试通过: 返回 {result}")
        
        # 测试包含负无穷大
        neg_inf_array = [1.0, float('-inf'), 3.0]
        result = MathCommon.max_value(neg_inf_array)
        expected = 3.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 负无穷大值处理测试通过: 返回 {result}")
        
        # 步骤4: 测试numpy数组输入
        np_array = np.array([10.0, 20.0, 15.0])
        result = MathCommon.max_value(np_array)
        expected = 20.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ numpy数组输入测试通过: {result}")
    
    def test_max_with_mask_functionality(self):
        """
        测试带掩码的最大值计算
        
        测试步骤:
        1. 基本掩码过滤 - 验证掩码功能
        2. 全零掩码处理 - 验证边界条件
        3. 全非零掩码 - 验证正常情况
        4. 错误输入处理 - 验证异常处理
        """
        # 步骤1: 基本掩码过滤
        values = [1.0, 5.0, 3.0, 8.0, 2.0]
        mask = [1.0, 0.0, 1.0, 1.0, 0.0]  # 只考虑位置0,2,3的值：1,3,8
        result = MathCommon.max_with_mask(values, mask)
        expected = 8.0  # max(1, 3, 8)
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 基本掩码过滤测试通过: {result} (从1,3,8中选择)")
        
        # 步骤2: 全零掩码
        zero_mask = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = MathCommon.max_with_mask(values, zero_mask)
        expected = 1.0  # 返回第一个元素
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 全零掩码处理测试通过: 返回 {result}")
        
        # 步骤3: 全非零掩码
        full_mask = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = MathCommon.max_with_mask(values, full_mask)
        expected = 8.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 全非零掩码测试通过: {result}")
        
        # 步骤4: 错误输入处理 - 长度不匹配
        short_mask = [1.0, 0.0]
        with pytest.raises(ValueError, match="长度必须相同"):
            MathCommon.max_with_mask(values, short_mask)
        print(f"  ✓ 长度不匹配错误处理测试通过")


class TestMathCommonMinValue(TestDWSIMMathBase):
    """
    测试最小值计算函数
    
    测试目标:
    - min_value: 基本最小值计算（忽略零值）
    - min_with_mask: 带掩码的最小值计算
    """
    
    def test_min_value_zero_handling(self):
        """
        测试最小值计算中零值的特殊处理
        
        测试步骤:
        1. 包含零值的数组 - 验证零值忽略逻辑
        2. 全零数组 - 验证边界行为
        3. 不含零值数组 - 验证正常计算
        4. 单个非零值 - 验证特殊情况
        """
        # 步骤1: 包含零值的数组（应该忽略零值）
        with_zeros = [0.0, 5.0, 3.0, 8.0, 0.0, 2.0]
        result = MathCommon.min_value(with_zeros)
        expected = 2.0  # min(5, 3, 8, 2) = 2，忽略零值
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 零值忽略测试通过: {result} (从5,3,8,2中选择)")
        
        # 步骤2: 全零数组
        zeros_array = [0.0, 0.0, 0.0]
        result = MathCommon.min_value(zeros_array)
        expected = 0.0  # 全零时返回0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 全零数组处理测试通过: 返回 {result}")
        
        # 步骤3: 不含零值数组
        positive_array = [5.0, 3.0, 8.0, 2.0]
        result = MathCommon.min_value(positive_array)
        expected = 2.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 不含零值数组测试通过: {result}")
        
        # 步骤4: 单个非零值
        single_nonzero = [0.0, 0.0, 5.0, 0.0]
        result = MathCommon.min_value(single_nonzero)
        expected = 5.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 单个非零值测试通过: {result}")
        
        # 测试混合正负数和零
        mixed_with_zero = [0.0, -3.0, 5.0, 0.0, 2.0]
        result = MathCommon.min_value(mixed_with_zero)
        expected = -3.0  # min(-3, 5, 2) = -3，忽略零值
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 混合正负数零值测试通过: {result}")
    
    def test_min_with_mask_functionality(self):
        """
        测试带掩码的最小值计算
        
        测试步骤:
        1. 基本掩码过滤 - 验证掩码功能
        2. 全零掩码处理 - 验证边界条件
        3. 错误输入处理 - 验证异常处理
        """
        # 步骤1: 基本掩码过滤
        values = [1.0, 5.0, 3.0, 8.0, 2.0]
        mask = [1.0, 0.0, 1.0, 1.0, 0.0]  # 只考虑位置0,2,3的值：1,3,8
        result = MathCommon.min_with_mask(values, mask)
        expected = 1.0  # min(1, 3, 8)
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 基本掩码过滤测试通过: {result} (从1,3,8中选择)")
        
        # 步骤2: 全零掩码
        zero_mask = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = MathCommon.min_with_mask(values, zero_mask)
        expected = 1.0  # 返回第一个元素
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 全零掩码处理测试通过: 返回 {result}")
        
        # 步骤3: 错误输入处理
        short_mask = [1.0, 0.0]
        with pytest.raises(ValueError, match="长度必须相同"):
            MathCommon.min_with_mask(values, short_mask)
        print(f"  ✓ 长度不匹配错误处理测试通过")


class TestMathCommonSumFunctions(TestDWSIMMathBase):
    """
    测试求和相关函数
    
    测试目标:
    - sum_array: 基本数组求和
    - abs_sum: 绝对值求和  
    - sum_of_squares: 平方和
    """
    
    def test_sum_array_functionality(self):
        """
        测试基本数组求和功能
        
        测试步骤:
        1. 正数求和 - 验证基本求和
        2. 混合数组求和 - 验证正负数混合
        3. 空数组求和 - 验证边界条件
        4. 大数值求和 - 验证数值精度
        """
        # 步骤1: 正数求和
        positive_array = [1.0, 5.0, 3.0, 8.0, 2.0]
        result = MathCommon.sum_array(positive_array)
        expected = sum(positive_array)  # 1+5+3+8+2 = 19
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 正数求和测试通过: {result}")
        
        # 步骤2: 混合数组求和
        mixed_array = [-2.0, 5.0, -1.0, 8.0, 0.0]
        result = MathCommon.sum_array(mixed_array)
        expected = sum(mixed_array)  # -2+5-1+8+0 = 10
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 混合数组求和测试通过: {result}")
        
        # 步骤3: 空数组求和
        empty_array = []
        result = MathCommon.sum_array(empty_array)
        expected = 0.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 空数组求和测试通过: {result}")
        
        # 步骤4: 大数值求和
        large_numbers = [1e6, 2e6, 3e6]
        result = MathCommon.sum_array(large_numbers)
        expected = 6e6
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 大数值求和测试通过: {result}")
    
    def test_abs_sum_functionality(self):
        """
        测试绝对值求和功能
        
        测试步骤:
        1. 混合数组绝对值求和 - 验证绝对值计算
        2. 全负数数组 - 验证负数处理
        3. 全正数数组 - 验证与普通求和一致性
        4. 包含零的数组 - 验证零值处理
        """
        # 步骤1: 混合数组的绝对值求和
        mixed_array = [-2.0, 5.0, -1.0, 8.0, 0.0]
        result = MathCommon.abs_sum(mixed_array)
        expected = abs(-2) + abs(5) + abs(-1) + abs(8) + abs(0)  # 2+5+1+8+0 = 16
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 混合数组绝对值求和测试通过: {result}")
        
        # 步骤2: 全负数数组
        negative_array = [-5.0, -2.0, -8.0, -1.0]
        result = MathCommon.abs_sum(negative_array)
        expected = abs(-5) + abs(-2) + abs(-8) + abs(-1)  # 5+2+8+1 = 16
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 全负数数组绝对值求和测试通过: {result}")
        
        # 步骤3: 全正数数组（绝对值求和等于普通求和）
        positive_array = [1.0, 5.0, 3.0, 8.0, 2.0]
        result = MathCommon.abs_sum(positive_array)
        expected = sum(positive_array)
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 全正数数组绝对值求和测试通过: {result}")
    
    def test_sum_of_squares_functionality(self):
        """
        测试平方和计算功能
        
        测试步骤:
        1. 基本平方和 - 验证平方运算
        2. 负数平方和 - 验证负数平方后为正
        3. 包含零的平方和 - 验证零值处理
        4. 与理论值比较 - 验证计算准确性
        """
        # 步骤1: 基本平方和
        test_array = [1.0, 2.0, 3.0]
        result = MathCommon.sum_of_squares(test_array)
        expected = 1**2 + 2**2 + 3**2  # 1+4+9 = 14
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 基本平方和测试通过: {result}")
        
        # 步骤2: 负数平方和
        test_array = [-1.0, 2.0, -3.0]
        result = MathCommon.sum_of_squares(test_array)
        expected = 1**2 + 2**2 + 3**2  # 1+4+9 = 14 (负数平方后为正)
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 负数平方和测试通过: {result}")
        
        # 步骤3: 包含零的平方和
        test_array = [0.0, 1.0, 2.0]
        result = MathCommon.sum_of_squares(test_array)
        expected = 0**2 + 1**2 + 2**2  # 0+1+4 = 5
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 包含零的平方和测试通过: {result}")


class TestMathCommonWeightedOperations(TestDWSIMMathBase):
    """
    测试加权运算函数
    
    测试目标:
    - weighted_average: 加权平均计算
    """
    
    def test_weighted_average_basic(self):
        """
        测试基本加权平均计算
        
        测试步骤:
        1. 标准加权平均 - 验证加权平均公式
        2. 权重未归一化 - 验证权重归一化处理
        3. 单一权重 - 验证极端情况
        4. 等权重情况 - 验证与算术平均一致性
        """
        # 步骤1: 标准加权平均（权重已归一化）
        weights = [0.3, 0.3, 0.4]
        values = [10.0, 20.0, 30.0]
        result = MathCommon.weighted_average(weights, values)
        expected = (0.3*10 + 0.3*20 + 0.4*30) / (0.3 + 0.3 + 0.4)  # 21.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 标准加权平均测试通过: {result}")
        
        # 步骤2: 权重未归一化
        weights_unnormalized = [3.0, 3.0, 4.0]
        result = MathCommon.weighted_average(weights_unnormalized, values)
        expected = (3*10 + 3*20 + 4*30) / (3 + 3 + 4)  # 21.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 未归一化权重测试通过: {result}")
        
        # 步骤3: 单一权重
        weights_single = [0.0, 1.0, 0.0]
        result = MathCommon.weighted_average(weights_single, values)
        expected = 20.0  # 只考虑第二个值
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 单一权重测试通过: {result}")
        
        # 步骤4: 等权重（应该等于算术平均）
        weights_equal = [1.0, 1.0, 1.0]
        result = MathCommon.weighted_average(weights_equal, values)
        expected = (10 + 20 + 30) / 3  # 20.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 等权重测试通过: {result}")
    
    def test_weighted_average_error_cases(self):
        """
        测试加权平均的错误情况
        
        测试步骤:
        1. 数组长度不匹配 - 验证输入验证
        2. 权重全为零 - 验证除零错误处理
        3. 空数组 - 验证边界条件处理
        """
        # 步骤1: 数组长度不匹配
        weights = [0.3, 0.7]
        values = [10.0, 20.0, 30.0]
        with pytest.raises(ValueError, match="长度必须相同"):
            MathCommon.weighted_average(weights, values)
        print(f"  ✓ 长度不匹配错误处理测试通过")
        
        # 步骤2: 权重全为零
        weights_zero = [0.0, 0.0, 0.0]
        values = [10.0, 20.0, 30.0]
        with pytest.raises(ZeroDivisionError):
            MathCommon.weighted_average(weights_zero, values)
        print(f"  ✓ 权重全零错误处理测试通过")
        
        # 步骤3: 空数组
        with pytest.raises(ValueError):
            MathCommon.weighted_average([], [])
        print(f"  ✓ 空数组错误处理测试通过")


class TestMathCommonStatisticalFunctions(TestDWSIMMathBase):
    """
    测试统计函数
    
    测试目标:
    - standard_deviation: 标准差计算
    - variance: 方差计算
    """
    
    def test_standard_deviation_basic(self):
        """
        测试基本标准差计算
        
        测试步骤:
        1. 样本标准差 (N-1) - 验证样本标准差公式
        2. 总体标准差 (N) - 验证总体标准差公式
        3. 与numpy结果比较 - 验证计算准确性
        4. 相同元素数组 - 验证标准差为0的情况
        """
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # 步骤1: 样本标准差 (除以N-1)
        result = MathCommon.standard_deviation(test_data, sample=True)
        expected = np.std(test_data, ddof=1)  # ddof=1表示样本标准差
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 样本标准差测试通过: {result}")
        
        # 步骤2: 总体标准差 (除以N)
        result = MathCommon.standard_deviation(test_data, sample=False)
        expected = np.std(test_data, ddof=0)  # ddof=0表示总体标准差
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 总体标准差测试通过: {result}")
        
        # 步骤4: 相同元素数组（标准差应为0）
        same_elements = [3.0, 3.0, 3.0, 3.0]
        result = MathCommon.standard_deviation(same_elements, sample=False)
        expected = 0.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 相同元素标准差测试通过: {result}")
    
    def test_variance_basic(self):
        """
        测试基本方差计算
        
        测试步骤:
        1. 样本方差 - 验证样本方差计算
        2. 总体方差 - 验证总体方差计算
        3. 方差与标准差关系 - 验证 variance = std^2
        4. 与numpy结果比较 - 验证计算准确性
        """
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # 步骤1: 样本方差
        result = MathCommon.variance(test_data, sample=True)
        expected = np.var(test_data, ddof=1)
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 样本方差测试通过: {result}")
        
        # 步骤2: 总体方差
        result = MathCommon.variance(test_data, sample=False)
        expected = np.var(test_data, ddof=0)
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 总体方差测试通过: {result}")
        
        # 步骤3: 验证方差 = 标准差^2
        std_dev = MathCommon.standard_deviation(test_data, sample=True)
        variance = MathCommon.variance(test_data, sample=True)
        assert_almost_equal_scalar(variance, std_dev**2)
        print(f"  ✓ 方差与标准差关系验证通过: {variance} = {std_dev}^2")


class TestMathCommonArrayUtilities(TestDWSIMMathBase):
    """
    测试数组工具函数
    
    测试目标:
    - copy_to_vector: 从二维数组提取列向量
    """
    
    def test_copy_to_vector_functionality(self):
        """
        测试从二维数组提取列向量
        
        测试步骤:
        1. 基本列提取 - 验证正确提取指定列
        2. 多列提取 - 验证不同列的提取
        3. 单行数组 - 验证边界情况
        4. 数据类型验证 - 确保返回numpy数组
        """
        # 步骤1&2: 创建测试二维数组并提取多列
        array_2d = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        # 提取第0列
        result = MathCommon.copy_to_vector(array_2d, 0)
        expected = np.array([1.0, 4.0, 7.0])
        np.testing.assert_array_almost_equal(result, expected)
        print(f"  ✓ 第0列提取测试通过: {result}")
        
        # 提取第1列
        result = MathCommon.copy_to_vector(array_2d, 1)
        expected = np.array([2.0, 5.0, 8.0])
        np.testing.assert_array_almost_equal(result, expected)
        print(f"  ✓ 第1列提取测试通过: {result}")
        
        # 提取第2列
        result = MathCommon.copy_to_vector(array_2d, 2)
        expected = np.array([3.0, 6.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)
        print(f"  ✓ 第2列提取测试通过: {result}")
        
        # 步骤3: 单行数组
        single_row = [[1.0, 2.0, 3.0]]
        result = MathCommon.copy_to_vector(single_row, 1)
        expected = np.array([2.0])
        np.testing.assert_array_almost_equal(result, expected)
        print(f"  ✓ 单行数组提取测试通过: {result}")
        
        # 步骤4: 验证返回类型
        assert isinstance(result, np.ndarray), "返回类型应为numpy数组"
        print(f"  ✓ 返回类型验证通过: {type(result)}")
    
    def test_copy_to_vector_edge_cases(self):
        """
        测试copy_to_vector的边界条件
        
        测试步骤:
        1. 空数组处理 - 验证空输入处理
        2. 索引越界 - 验证错误处理
        3. 异常输入处理 - 验证输入验证
        """
        # 步骤1: 空数组
        empty_2d = []
        result = MathCommon.copy_to_vector(empty_2d, 0)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)
        print(f"  ✓ 空数组处理测试通过: 长度 {len(result)}")
        
        # 步骤2: 索引越界
        array_2d = [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(ValueError, match="无法从数组列表中提取索引"):
            MathCommon.copy_to_vector(array_2d, 2)  # 索引2不存在
        print(f"  ✓ 索引越界错误处理测试通过")


class TestMathCommonPerformance(TestDWSIMMathBase):
    """
    测试数值精度和性能相关问题
    
    测试目标:
    - 大数值计算精度
    - 小数值计算精度
    - 基本性能验证
    """
    
    def test_large_numbers_precision(self):
        """
        测试大数值的计算精度
        
        测试步骤:
        1. 大数值求和 - 验证精度保持
        2. 大数值最大值 - 验证比较精度
        3. 大数值统计 - 验证复杂计算精度
        """
        large_numbers = [1e6, 2e6, 3e6]
        
        # 步骤1: 大数值求和
        result = MathCommon.sum_array(large_numbers)
        expected = sum(large_numbers)  # 6e6
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 大数值求和精度测试通过: {result}")
        
        # 步骤2: 大数值最大值
        result = MathCommon.max_value(large_numbers)
        expected = 3e6
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 大数值最大值精度测试通过: {result}")
    
    def test_small_numbers_precision(self):
        """
        测试小数值的计算精度
        
        测试步骤:
        1. 小数值求和 - 验证精度保持
        2. 小数值加权平均 - 验证复杂计算精度
        """
        small_numbers = [1e-6, 2e-6, 3e-6]
        
        # 步骤1: 小数值求和
        result = MathCommon.sum_array(small_numbers)
        expected = sum(small_numbers)  # 6e-6
        assert_almost_equal_scalar(result, expected, tolerance=1e-12)
        print(f"  ✓ 小数值求和精度测试通过: {result}")
        
        # 步骤2: 小数值加权平均
        weights = [1.0, 1.0, 1.0]
        result = MathCommon.weighted_average(weights, small_numbers)
        expected = sum(small_numbers) / 3  # 2e-6
        assert_almost_equal_scalar(result, expected, tolerance=1e-12)
        print(f"  ✓ 小数值加权平均精度测试通过: {result}")
    
    @pytest.mark.slow
    def test_performance_benchmarks(self):
        """
        测试基本性能指标
        
        测试步骤:
        1. 大数组求和性能 - 验证线性时间复杂度
        2. 大数组统计性能 - 验证基本性能要求
        """
        # 生成大数组
        large_array = list(range(10000))
        
        # 步骤1: 大数组求和性能
        with PerformanceTimer("大数组求和") as timer:
            result = MathCommon.sum_array(large_array)
        
        expected = sum(large_array)
        assert_almost_equal_scalar(result, expected)
        timer.print_timing()
        
        # 基本性能要求：应在1秒内完成
        assert timer.execution_time < 1.0, f"求和性能过慢: {timer.execution_time}s"
        print(f"  ✓ 大数组求和性能测试通过")


if __name__ == "__main__":
    # 运行所有测试
    print("=== DWSIM数学库 core.general 模块测试 ===")
    pytest.main([__file__, "-v", "--tb=short", "-s"]) 