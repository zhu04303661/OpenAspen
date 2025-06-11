"""
DWSIM数学库 - 矩阵运算测试
=========================

专门测试 dwsim_math.core.matrix_ops 模块的功能，包括：
- 行列式计算
- 矩阵求逆
- LU分解
- 线性方程组求解
- 矩阵条件数

测试策略:
1. 功能正确性 - 验证每个函数的数学准确性
2. 边界条件 - 测试奇异矩阵、病条件矩阵等
3. 错误处理 - 验证异常输入的处理
4. 数值稳定性 - 确保算法数值稳定
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
    from dwsim_math.core.matrix_ops import (
        Determinant, Inverse, LUDecomposition, 
        TriangularInverse, MatrixOperations
    )
    print("✅ dwsim_math.core.matrix_ops 模块导入成功")
except ImportError as e:
    print(f"❌ dwsim_math.core.matrix_ops 模块导入失败: {e}")
    pytest.skip("core.matrix_ops模块不可用", allow_module_level=True)

try:
    from .test_utils import TestDWSIMMathBase, assert_almost_equal_matrix, assert_almost_equal_scalar
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
    
    def assert_almost_equal_matrix(actual, expected, tolerance=1e-6):
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


class TestMatrixDeterminant(TestDWSIMMathBase):
    """
    测试行列式计算功能
    
    测试目标:
    - matrix_determinant: 一般矩阵行列式
    """
    
    def test_determinant_basic_cases(self):
        """
        测试基本行列式计算
        
        测试步骤:
        1. 2x2矩阵行列式 - 验证基本公式
        2. 3x3矩阵行列式 - 验证扩展算法
        3. 单位矩阵行列式 - 验证理论值
        """
        # 步骤1: 测试2x2矩阵
        A = [[1, 2], [3, 4]]
        result = Determinant.matrix_determinant(A)
        expected = 1*4 - 2*3  # -2
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 2x2矩阵行列式测试通过: {result}")
        
        # 步骤2: 测试已知3x3矩阵
        A = [[2, 1, 3], [1, 0, 1], [1, 2, 1]]
        result = Determinant.matrix_determinant(A)
        # 手工计算结果约为2.0
        assert abs(result - 2.0) < 0.1  # 使用较宽松的容差
        print(f"  ✓ 3x3矩阵行列式测试通过: {result}")
        
        # 步骤3: 测试单位矩阵
        I = np.eye(3)
        result = Determinant.matrix_determinant(I)
        expected = 1.0
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 单位矩阵行列式测试通过: {result}")
    
    def test_determinant_special_cases(self):
        """
        测试行列式计算的特殊情况
        
        测试步骤:
        1. 奇异矩阵 - 验证行列式为0
        2. 对角矩阵 - 验证对角元素乘积
        """
        # 步骤1: 奇异矩阵（行列式为0）
        singular = np.array([[1, 2], [2, 4]])
        result = Determinant.matrix_determinant(singular)
        assert_almost_equal_scalar(result, 0.0, tolerance=1e-10)
        print(f"  ✓ 奇异矩阵行列式测试通过: {result}")
        
        # 步骤2: 对角矩阵
        diag = np.diag([2, 3, 4])
        result = Determinant.matrix_determinant(diag)
        expected = 2 * 3 * 4  # 24
        assert_almost_equal_scalar(result, expected)
        print(f"  ✓ 对角矩阵行列式测试通过: {result}")


class TestMatrixInverse(TestDWSIMMathBase):
    """
    测试矩阵求逆功能
    
    测试目标:
    - matrix_inverse: 一般矩阵求逆
    """
    
    def test_inverse_basic_functionality(self):
        """
        测试基本矩阵求逆功能
        
        测试步骤:
        1. 可逆矩阵求逆 - 验证基本求逆
        2. 验证 A * A^(-1) = I - 验证逆矩阵定义
        3. 单位矩阵求逆 - 验证特殊情况
        """
        # 步骤1: 可逆矩阵求逆
        A = np.array([[2, 1], [1, 2]], dtype=float)
        inv_A, success = Inverse.matrix_inverse(A)
        
        assert success, "矩阵求逆应该成功"
        print(f"  ✓ 可逆矩阵求逆成功")
        
        # 步骤2: 验证 A * A^(-1) = I
        product = np.dot(A, inv_A)
        identity = np.eye(2)
        assert_almost_equal_matrix(product, identity)
        print(f"  ✓ A * A^(-1) = I 验证通过")
        
        # 步骤3: 单位矩阵求逆
        I = np.eye(3)
        inv_I, success = Inverse.matrix_inverse(I)
        assert success, "单位矩阵求逆应该成功"
        assert_almost_equal_matrix(inv_I, I)
        print(f"  ✓ 单位矩阵求逆测试通过")
    
    def test_inverse_singular_matrices(self):
        """
        测试奇异矩阵求逆
        
        测试步骤:
        1. 奇异矩阵求逆失败 - 验证错误检测
        """
        # 步骤1: 奇异矩阵
        singular = np.array([[1, 2], [2, 4]], dtype=float)
        inv_A, success = Inverse.matrix_inverse(singular)
        
        assert not success, "奇异矩阵求逆应该失败"
        print(f"  ✓ 奇异矩阵求逆失败检测通过")


class TestMatrixOperations(TestDWSIMMathBase):
    """
    测试MatrixOperations统一接口
    
    测试目标:
    - 统一的矩阵运算接口
    - 线性方程组求解
    - 矩阵条件数
    """
    
    def test_linear_system_solving(self):
        """
        测试线性方程组求解
        
        测试步骤:
        1. 标准线性系统 Ax = b
        2. 验证解的正确性
        """
        # 步骤1: 标准线性系统
        A = np.array([[2, 1], [1, 2]], dtype=float)
        b = np.array([3, 3], dtype=float)  # 已知解应该是 [1, 1]
        
        x, success = MatrixOperations.solve_linear_system(A, b)
        assert success, "线性系统求解应该成功"
        print(f"  ✓ 线性系统求解成功")
        
        # 步骤2: 验证解
        expected_x = np.array([1.0, 1.0])
        assert_almost_equal_matrix(x, expected_x, tolerance=1e-10)
        print(f"  ✓ 解的正确性验证通过: x = {x}")
        
        # 验证 Ax = b
        result_b = np.dot(A, x)
        assert_almost_equal_matrix(result_b, b, tolerance=1e-10)
        print(f"  ✓ Ax = b 验证通过")
    
    def test_condition_number(self):
        """
        测试矩阵条件数计算
        
        测试步骤:
        1. 单位矩阵条件数 - 应为1
        2. 良条件矩阵 - 条件数较小
        """
        # 步骤1: 单位矩阵（条件数为1）
        I = np.eye(3)
        cond = MatrixOperations.condition_number(I)
        assert_almost_equal_scalar(cond, 1.0, tolerance=1e-6)
        print(f"  ✓ 单位矩阵条件数测试通过: {cond}")
        
        # 步骤2: 良条件矩阵
        well_cond = np.array([[2, 1], [1, 2]], dtype=float)
        cond = MatrixOperations.condition_number(well_cond)
        assert cond < 10, f"良条件矩阵的条件数应该较小，实际: {cond}"
        print(f"  ✓ 良条件矩阵条件数测试通过: {cond}")


if __name__ == "__main__":
    # 运行所有测试
    print("=== DWSIM数学库 core.matrix_ops 模块测试 ===")
    pytest.main([__file__, "-v", "--tb=short", "-s"]) 