"""
DWSIM数学库测试工具模块
=====================

提供测试基类、断言方法和通用测试数据，用于DWSIM数学库的所有测试模块。

功能包括:
- 基础测试类
- 数值比较断言
- 测试数据生成
- 性能测试工具

作者: DWSIM团队
版本: 1.0.0
"""

import pytest
import numpy as np
import time
from typing import Union, List, Any, Dict
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestDWSIMMathBase:
    """
    DWSIM数学库测试基类
    
    提供所有测试类的公共设置、工具方法和测试数据
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """
        自动设置测试环境
        
        在每个测试方法执行前自动运行，设置必要的测试环境和数据
        """
        # 数值精度设置
        self.tolerance = 1e-10
        self.float_tolerance = 1e-6
        self.integration_tolerance = 1e-4
        
        # 基础测试数据
        self.test_arrays = {
            'positive': [1.0, 5.0, 3.0, 8.0, 2.0],
            'negative': [-5.0, -2.0, -8.0, -1.0],
            'mixed': [-2.0, 5.0, -1.0, 8.0, 0.0],
            'zeros': [0.0, 0.0, 0.0],
            'single': [42.0],
            'empty': [],
            'with_zeros': [0.0, 5.0, 3.0, 8.0, 0.0, 2.0],
            'large': [1e6, 2e6, 3e6],
            'small': [1e-6, 2e-6, 3e-6]
        }
        
        # 测试矩阵
        self.test_matrices = {
            'identity_2x2': np.eye(2),
            'identity_3x3': np.eye(3),
            'singular': np.array([[1, 2], [2, 4]]),  # 奇异矩阵
            'well_conditioned': np.array([[2, 1], [1, 2]]),
            'ill_conditioned': np.array([[1, 1], [1, 1.0001]]),
            'diagonal': np.diag([2, 3, 4]),
            'upper_triangular': np.array([[2, 1, 3], [0, 4, 2], [0, 0, 5]]),
            'lower_triangular': np.array([[2, 0, 0], [1, 4, 0], [3, 2, 5]]),
            'symmetric': np.array([[4, 2, 1], [2, 3, 2], [1, 2, 5]]),
            'random_3x3': None  # 将在使用时生成
        }
        
        # 权重和值的测试数据
        self.test_weights = {
            'normalized': [0.3, 0.3, 0.4],
            'unnormalized': [3.0, 3.0, 4.0],
            'single_weight': [0.0, 1.0, 0.0],
            'equal_weights': [1.0, 1.0, 1.0],
            'zero_weights': [0.0, 0.0, 0.0]
        }
        
        self.test_values = {
            'normal': [10.0, 20.0, 30.0],
            'linear': [1.0, 2.0, 3.0],
            'quadratic': [1.0, 4.0, 9.0]
        }
        
        # 复数测试数据
        self.test_complex = {
            'real': (3.0, 0.0),      # 实数
            'imaginary': (0.0, 4.0),  # 纯虚数
            'standard': (3.0, 4.0),   # 标准复数
            'unit': (1.0, 0.0),      # 单位复数
            'zero': (0.0, 0.0)       # 零复数
        }
        
        # 函数测试数据
        self.test_functions = {
            'polynomial': lambda x: x**2 - 4,           # 多项式
            'transcendental': lambda x: x - np.cos(x),  # 超越函数
            'exponential': lambda x: np.exp(x) - 2,     # 指数函数
            'trigonometric': lambda x: np.sin(x) - 0.5  # 三角函数
        }
        
        # 优化测试问题
        self.optimization_problems = {
            'quadratic': {
                'objective': lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
                'gradient': lambda x: np.array([2*(x[0] - 1), 2*(x[1] - 2)]),
                'solution': np.array([1.0, 2.0]),
                'initial': np.array([10.0, -5.0])
            },
            'rosenbrock': {
                'objective': lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2,
                'gradient': lambda x: np.array([
                    -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                    200 * (x[1] - x[0]**2)
                ]),
                'solution': np.array([1.0, 1.0]),
                'initial': np.array([-1.2, 1.0])
            }
        }
        
        yield
        
        # 测试清理（如果需要）
        self._cleanup_test_data()
    
    def _cleanup_test_data(self):
        """清理测试数据"""
        pass
    
    def generate_random_matrix(self, size: int, seed: int = 42) -> np.ndarray:
        """
        生成随机测试矩阵
        
        参数:
            size: 矩阵大小
            seed: 随机种子
            
        返回:
            np.ndarray: 随机矩阵
        """
        np.random.seed(seed)
        return np.random.rand(size, size)
    
    def generate_test_data(self, size: int, data_type: str = 'random') -> np.ndarray:
        """
        生成测试数据
        
        参数:
            size: 数据大小
            data_type: 数据类型 ('random', 'linear', 'quadratic', 'sinusoidal')
            
        返回:
            np.ndarray: 测试数据
        """
        if data_type == 'random':
            np.random.seed(42)
            return np.random.randn(size)
        elif data_type == 'linear':
            return np.linspace(0, 10, size)
        elif data_type == 'quadratic':
            x = np.linspace(0, 10, size)
            return x**2
        elif data_type == 'sinusoidal':
            x = np.linspace(0, 2*np.pi, size)
            return np.sin(x)
        else:
            raise ValueError(f"未知的数据类型: {data_type}")


def assert_almost_equal_matrix(actual: Union[np.ndarray, List], 
                             expected: Union[np.ndarray, List], 
                             tolerance: float = 1e-6,
                             message: str = None) -> None:
    """
    矩阵近似相等断言
    
    参数:
        actual: 实际矩阵
        expected: 期望矩阵
        tolerance: 容差
        message: 错误消息
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    
    if actual.shape != expected.shape:
        raise AssertionError(f"矩阵形状不匹配: {actual.shape} != {expected.shape}")
    
    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    
    if max_diff > tolerance:
        if message:
            raise AssertionError(f"{message}: 最大差值 {max_diff} > 容差 {tolerance}")
        else:
            raise AssertionError(f"矩阵不匹配: 最大差值 {max_diff} > 容差 {tolerance}")


def assert_almost_equal_scalar(actual: float, 
                             expected: float, 
                             tolerance: float = 1e-6,
                             message: str = None) -> None:
    """
    标量近似相等断言
    
    参数:
        actual: 实际值
        expected: 期望值
        tolerance: 容差
        message: 错误消息
    """
    diff = abs(actual - expected)
    
    if diff > tolerance:
        if message:
            raise AssertionError(f"{message}: |{actual} - {expected}| = {diff} > {tolerance}")
        else:
            raise AssertionError(f"标量不匹配: |{actual} - {expected}| = {diff} > {tolerance}")


def assert_matrix_properties(matrix: np.ndarray, 
                           properties: Dict[str, Any]) -> None:
    """
    验证矩阵属性
    
    参数:
        matrix: 要验证的矩阵
        properties: 属性字典，可包含:
            - 'symmetric': 是否对称
            - 'positive_definite': 是否正定
            - 'singular': 是否奇异
            - 'condition_number': 条件数上限
    """
    if 'symmetric' in properties:
        is_symmetric = np.allclose(matrix, matrix.T)
        assert is_symmetric == properties['symmetric'], \
            f"对称性检查失败: 期望{properties['symmetric']}, 实际{is_symmetric}"
    
    if 'positive_definite' in properties:
        try:
            np.linalg.cholesky(matrix)
            is_positive_definite = True
        except np.linalg.LinAlgError:
            is_positive_definite = False
        
        assert is_positive_definite == properties['positive_definite'], \
            f"正定性检查失败: 期望{properties['positive_definite']}, 实际{is_positive_definite}"
    
    if 'singular' in properties:
        det = np.linalg.det(matrix)
        is_singular = abs(det) < 1e-10
        assert is_singular == properties['singular'], \
            f"奇异性检查失败: 期望{properties['singular']}, 实际{is_singular}, det={det}"
    
    if 'condition_number' in properties:
        cond = np.linalg.cond(matrix)
        assert cond <= properties['condition_number'], \
            f"条件数检查失败: {cond} > {properties['condition_number']}"


class PerformanceTimer:
    """
    性能测试计时器
    
    用于测量函数执行时间和性能基准验证
    """
    
    def __init__(self, name: str = "测试"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.execution_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
    
    def assert_execution_time(self, max_time: float):
        """
        断言执行时间不超过最大时间
        
        参数:
            max_time: 最大允许执行时间（秒）
        """
        if self.execution_time is None:
            raise RuntimeError("计时器未正确使用")
        
        assert self.execution_time <= max_time, \
            f"{self.name}执行时间过长: {self.execution_time:.4f}s > {max_time}s"
    
    def print_timing(self):
        """打印执行时间"""
        if self.execution_time is not None:
            print(f"{self.name}执行时间: {self.execution_time:.4f}s")


def skip_if_module_unavailable(module_name: str):
    """
    如果模块不可用则跳过测试的装饰器
    
    参数:
        module_name: 模块名称
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                __import__(module_name)
                return test_func(*args, **kwargs)
            except ImportError:
                pytest.skip(f"模块 {module_name} 不可用")
        return wrapper
    return decorator


def parametrize_matrix_sizes(sizes: List[int]):
    """
    矩阵大小参数化装饰器
    
    参数:
        sizes: 矩阵大小列表
    """
    return pytest.mark.parametrize("matrix_size", sizes)


def parametrize_tolerances(tolerances: List[float]):
    """
    容差参数化装饰器
    
    参数:
        tolerances: 容差列表
    """
    return pytest.mark.parametrize("tolerance", tolerances)


# 测试数据生成器
class TestDataGenerator:
    """
    测试数据生成器
    
    提供各种类型的测试数据生成方法
    """
    
    @staticmethod
    def generate_ill_conditioned_matrix(size: int, condition_number: float) -> np.ndarray:
        """
        生成指定条件数的病条件矩阵
        
        参数:
            size: 矩阵大小
            condition_number: 目标条件数
            
        返回:
            np.ndarray: 病条件矩阵
        """
        # 生成对角矩阵，对角元素从1到1/condition_number
        diag_values = np.logspace(0, -np.log10(condition_number), size)
        D = np.diag(diag_values)
        
        # 生成随机正交矩阵
        Q, _ = np.linalg.qr(np.random.randn(size, size))
        
        # A = Q * D * Q^T
        return Q @ D @ Q.T
    
    @staticmethod
    def generate_test_functions():
        """
        生成测试函数集合
        
        返回:
            Dict: 测试函数字典
        """
        return {
            'linear': lambda x: 2*x + 1,
            'quadratic': lambda x: x**2 - 4,
            'cubic': lambda x: x**3 - 3*x + 1,
            'exponential': lambda x: np.exp(x) - 2,
            'logarithmic': lambda x: np.log(x) - 1,
            'trigonometric': lambda x: np.sin(x) - 0.5,
            'rational': lambda x: (x**2 - 1) / (x + 1),
            'composite': lambda x: np.sin(x**2) + np.cos(x)
        }


# 导出所有公共函数和类
__all__ = [
    'TestDWSIMMathBase',
    'assert_almost_equal_matrix',
    'assert_almost_equal_scalar',
    'assert_matrix_properties',
    'PerformanceTimer',
    'skip_if_module_unavailable',
    'parametrize_matrix_sizes',
    'parametrize_tolerances',
    'TestDataGenerator'
] 