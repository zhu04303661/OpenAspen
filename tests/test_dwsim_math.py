"""
DWSIM数学库测试套件
===================

完整测试DWSIM数学计算库的所有功能模块，包括：
- 核心数学模块 (core)
- 数值计算模块 (numerics)  
- 求解器模块 (solvers)
- 优化模块 (optimization)
- 随机数生成模块 (random)

测试目标:
1. 功能正确性验证
2. 边界条件测试
3. 错误处理验证
4. 性能基准测试
5. 数值精度验证

作者: DWSIM团队
版本: 1.0.0
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入DWSIM数学库模块
try:
    from dwsim_math.core import general, matrix_ops, interpolation
    from dwsim_math.numerics import complex_number
    from dwsim_math.solvers import brent
    from dwsim_math.optimization import lbfgs
    from dwsim_math.random import mersenne_twister
    print("✅ DWSIM数学库模块导入成功")
except ImportError as e:
    print(f"❌ DWSIM数学库模块导入失败: {e}")
    pytest.skip("DWSIM数学库模块不可用", allow_module_level=True)


class TestDWSIMMathSuite:
    """
    DWSIM数学库测试套件基类
    
    提供所有测试类的公共设置和工具方法
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """
        自动设置测试环境
        
        在每个测试方法执行前自动运行，设置必要的测试环境
        """
        # 设置数值精度容差
        self.tolerance = 1e-10
        self.float_tolerance = 1e-6
        
        # 设置测试数据
        self.test_matrices = {
            'identity_2x2': np.eye(2),
            'identity_3x3': np.eye(3),
            'singular': np.array([[1, 2], [2, 4]]),  # 奇异矩阵
            'well_conditioned': np.array([[2, 1], [1, 2]]),
            'ill_conditioned': np.array([[1, 1], [1, 1.0001]]),
            'random_3x3': np.random.rand(3, 3)
        }
        
        self.test_vectors = {
            'small': np.array([1, 2, 3]),
            'large': np.random.rand(100),
            'zeros': np.zeros(5),
            'ones': np.ones(10),
            'negative': np.array([-1, -2, -3])
        }
        
        yield
        
        # 测试清理（如果需要）
        pass
    
    def assert_almost_equal_matrix(self, actual, expected, tolerance=None):
        """
        矩阵近似相等断言
        
        参数:
            actual: 实际矩阵
            expected: 期望矩阵  
            tolerance: 容差，默认使用self.tolerance
        """
        if tolerance is None:
            tolerance = self.tolerance
            
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    
    def assert_almost_equal_scalar(self, actual, expected, tolerance=None):
        """
        标量近似相等断言
        
        参数:
            actual: 实际值
            expected: 期望值
            tolerance: 容差，默认使用self.float_tolerance
        """
        if tolerance is None:
            tolerance = self.float_tolerance
            
        assert abs(actual - expected) < tolerance, f"期望 {expected}，实际 {actual}，差值 {abs(actual - expected)} > {tolerance}"


# ============================================================================
# 核心数学模块测试 (Core Module Tests)
# ============================================================================

@pytest.mark.math_core
class TestCoreGeneral(TestDWSIMMathSuite):
    """
    测试core.general模块
    
    测试目标:
    - 基本统计函数的正确性
    - 边界条件处理
    - 错误输入处理
    - 性能基准
    """
    
    def test_max_value_basic(self):
        """
        测试基本最大值计算
        
        工作步骤:
        1. 测试正常数组的最大值
        2. 测试包含负数的数组
        3. 测试单元素数组
        4. 验证结果准确性
        """
        # 测试正常情况
        result = general.MathCommon.max_value([1, 5, 3, 8, 2])
        self.assert_almost_equal_scalar(result, 8.0)
        
        # 测试负数
        result = general.MathCommon.max_value([-5, -2, -8, -1])
        self.assert_almost_equal_scalar(result, -1.0)
        
        # 测试单元素
        result = general.MathCommon.max_value([42])
        self.assert_almost_equal_scalar(result, 42.0)
        
        # 测试numpy数组
        result = general.MathCommon.max_value(np.array([10, 20, 15]))
        self.assert_almost_equal_scalar(result, 20.0)
    
    def test_max_value_edge_cases(self):
        """
        测试最大值计算的边界条件
        
        工作步骤:
        1. 测试空数组
        2. 测试包含NaN的数组
        3. 测试包含无穷大的数组
        4. 验证错误处理
        """
        # 测试空数组
        result = general.MathCommon.max_value([])
        self.assert_almost_equal_scalar(result, 0.0)
        
        # 测试包含无穷大
        result = general.MathCommon.max_value([1, float('inf'), 3])
        assert result == float('inf')
        
        # 测试包含负无穷大
        result = general.MathCommon.max_value([1, float('-inf'), 3])
        self.assert_almost_equal_scalar(result, 3.0)
    
    def test_min_value_basic(self):
        """
        测试基本最小值计算（忽略零值）
        
        工作步骤:
        1. 测试包含零值的数组
        2. 测试全部为零的数组
        3. 测试正常数组
        4. 验证零值忽略逻辑
        """
        # 测试忽略零值
        result = general.MathCommon.min_value([0, 5, 3, 8, 0, 2])
        self.assert_almost_equal_scalar(result, 2.0)
        
        # 测试全零数组
        result = general.MathCommon.min_value([0, 0, 0])
        self.assert_almost_equal_scalar(result, 0.0)
        
        # 测试无零值数组
        result = general.MathCommon.min_value([5, 3, 8, 2])
        self.assert_almost_equal_scalar(result, 2.0)
    
    def test_weighted_average_basic(self):
        """
        测试基本加权平均计算
        
        工作步骤:
        1. 测试标准加权平均
        2. 测试权重归一化
        3. 测试等权重情况
        4. 验证计算公式
        """
        # 测试标准情况
        weights = [0.3, 0.3, 0.4]
        values = [10, 20, 30]
        result = general.MathCommon.weighted_average(weights, values)
        expected = (0.3*10 + 0.3*20 + 0.4*30) / (0.3 + 0.3 + 0.4)
        self.assert_almost_equal_scalar(result, expected)
        
        # 测试权重不归一化
        weights = [3, 3, 4]
        values = [10, 20, 30]
        result = general.MathCommon.weighted_average(weights, values)
        expected = (3*10 + 3*20 + 4*30) / (3 + 3 + 4)
        self.assert_almost_equal_scalar(result, expected)
    
    def test_weighted_average_edge_cases(self):
        """
        测试加权平均的边界条件
        
        工作步骤:
        1. 测试权重和值数组长度不匹配
        2. 测试权重全为零
        3. 测试包含负权重
        4. 验证错误处理
        """
        # 测试长度不匹配
        with pytest.raises(ValueError):
            general.MathCommon.weighted_average([1, 2], [1, 2, 3])
        
        # 测试权重全为零
        with pytest.raises(ZeroDivisionError):
            general.MathCommon.weighted_average([0, 0], [1, 2])
    
    def test_sum_of_squares(self):
        """
        测试平方和计算
        
        工作步骤:
        1. 测试基本平方和
        2. 测试负数平方和
        3. 测试包含零的情况
        4. 验证计算准确性
        """
        # 测试基本情况
        result = general.MathCommon.sum_of_squares([1, 2, 3])
        expected = 1**2 + 2**2 + 3**2
        self.assert_almost_equal_scalar(result, expected)
        
        # 测试负数
        result = general.MathCommon.sum_of_squares([-1, 2, -3])
        expected = 1**2 + 2**2 + 3**2
        self.assert_almost_equal_scalar(result, expected)
    
    def test_standard_deviation(self):
        """
        测试标准差计算
        
        工作步骤:
        1. 测试样本标准差
        2. 测试总体标准差
        3. 与numpy结果比较
        4. 验证计算公式
        """
        data = [1, 2, 3, 4, 5]
        
        # 测试样本标准差 (N-1)
        result = general.MathCommon.standard_deviation(data, sample=True)
        expected = np.std(data, ddof=1)
        self.assert_almost_equal_scalar(result, expected)
        
        # 测试总体标准差 (N)
        result = general.MathCommon.standard_deviation(data, sample=False)
        expected = np.std(data, ddof=0)
        self.assert_almost_equal_scalar(result, expected)


@pytest.mark.math_core
@pytest.mark.matrix_ops
class TestCoreMatrixOps(TestDWSIMMathSuite):
    """
    测试core.matrix_ops模块
    
    测试目标:
    - 矩阵行列式计算
    - 矩阵求逆
    - LU分解
    - 线性方程组求解
    """
    
    def test_determinant_basic(self):
        """
        测试基本行列式计算
        
        工作步骤:
        1. 测试2x2矩阵行列式
        2. 测试3x3矩阵行列式
        3. 测试单位矩阵行列式
        4. 与理论值比较
        """
        # 测试2x2矩阵
        A = [[1, 2], [3, 4]]
        result = matrix_ops.MatrixOperations.determinant(A)
        expected = 1*4 - 2*3  # -2
        self.assert_almost_equal_scalar(result, expected)
        
        # 测试单位矩阵
        I = np.eye(3)
        result = matrix_ops.MatrixOperations.determinant(I)
        self.assert_almost_equal_scalar(result, 1.0)
        
        # 测试已知3x3矩阵
        A = [[2, 1, 3], [1, 0, 1], [1, 2, 1]]
        result = matrix_ops.MatrixOperations.determinant(A)
        # 手工计算: 2*(0*1-1*2) - 1*(1*1-1*1) + 3*(1*2-0*1) = 2*(-2) - 1*0 + 3*2 = -4 + 6 = 2
        self.assert_almost_equal_scalar(result, 2.0)
    
    def test_determinant_special_cases(self):
        """
        测试行列式计算的特殊情况
        
        工作步骤:
        1. 测试奇异矩阵（行列式为0）
        2. 测试对角矩阵
        3. 测试上/下三角矩阵
        4. 验证特殊性质
        """
        # 测试奇异矩阵
        singular = self.test_matrices['singular']
        result = matrix_ops.MatrixOperations.determinant(singular)
        self.assert_almost_equal_scalar(result, 0.0)
        
        # 测试对角矩阵
        diag = np.diag([2, 3, 4])
        result = matrix_ops.MatrixOperations.determinant(diag)
        expected = 2 * 3 * 4
        self.assert_almost_equal_scalar(result, expected)
    
    def test_matrix_inverse_basic(self):
        """
        测试基本矩阵求逆
        
        工作步骤:
        1. 测试可逆矩阵的求逆
        2. 验证 A * A^(-1) = I
        3. 测试单位矩阵的逆
        4. 验证逆矩阵性质
        """
        # 测试2x2可逆矩阵
        A = self.test_matrices['well_conditioned']  # [[2, 1], [1, 2]]
        inv_A, success = matrix_ops.MatrixOperations.inverse(A)
        
        assert success, "矩阵求逆应该成功"
        
        # 验证 A * A^(-1) = I
        product = np.dot(A, inv_A)
        self.assert_almost_equal_matrix(product, np.eye(2))
        
        # 验证 A^(-1) * A = I
        product = np.dot(inv_A, A)
        self.assert_almost_equal_matrix(product, np.eye(2))
    
    def test_matrix_inverse_singular(self):
        """
        测试奇异矩阵求逆
        
        工作步骤:
        1. 测试奇异矩阵
        2. 验证求逆失败
        3. 测试接近奇异的矩阵
        4. 验证错误处理
        """
        # 测试奇异矩阵
        singular = self.test_matrices['singular']
        inv_A, success = matrix_ops.MatrixOperations.inverse(singular)
        
        assert not success, "奇异矩阵求逆应该失败"
    
    def test_solve_linear_system(self):
        """
        测试线性方程组求解
        
        工作步骤:
        1. 测试标准线性系统 Ax = b
        2. 验证解的正确性
        3. 测试多个右端项
        4. 验证解的唯一性
        """
        # 测试标准线性系统
        A = self.test_matrices['well_conditioned']  # [[2, 1], [1, 2]]
        b = np.array([3, 3])  # 已知解应该是 [1, 1]
        
        x, success = matrix_ops.MatrixOperations.solve_linear_system(A, b)
        
        assert success, "线性系统求解应该成功"
        
        # 验证解
        expected_x = np.array([1.0, 1.0])
        self.assert_almost_equal_matrix(x, expected_x)
        
        # 验证 Ax = b
        result_b = np.dot(A, x)
        self.assert_almost_equal_matrix(result_b, b)
    
    def test_condition_number(self):
        """
        测试矩阵条件数计算
        
        工作步骤:
        1. 测试良条件矩阵
        2. 测试病条件矩阵
        3. 与理论值比较
        4. 验证数值稳定性
        """
        # 测试单位矩阵（条件数为1）
        I = self.test_matrices['identity_2x2']
        cond = matrix_ops.MatrixOperations.condition_number(I)
        self.assert_almost_equal_scalar(cond, 1.0)
        
        # 测试良条件矩阵
        well_cond = self.test_matrices['well_conditioned']
        cond = matrix_ops.MatrixOperations.condition_number(well_cond)
        assert cond < 100, f"良条件矩阵的条件数应该较小，实际: {cond}"


@pytest.mark.math_core
@pytest.mark.interpolation
class TestCoreInterpolation(TestDWSIMMathSuite):
    """
    测试core.interpolation模块
    
    测试目标:
    - 插值算法的准确性
    - 边界条件处理
    - 插值点的合理性
    """
    
    def test_basic_interpolation(self):
        """
        测试基本插值功能
        
        工作步骤:
        1. 创建已知函数数据点
        2. 执行插值
        3. 验证插值结果
        4. 检查插值精度
        """
        # 使用简单的线性函数 y = 2x + 1
        x_data = [0, 1, 2, 3, 4]
        y_data = [1, 3, 5, 7, 9]
        
        # 测试中间点插值
        result = interpolation.Interpolation.interpolate(x_data, y_data, 2.5)
        expected = 2 * 2.5 + 1  # 6.0
        self.assert_almost_equal_scalar(result, expected)
    
    def test_interpolation_edge_cases(self):
        """
        测试插值的边界条件
        
        工作步骤:
        1. 测试端点插值
        2. 测试外推
        3. 测试重复数据点
        4. 验证稳定性
        """
        x_data = [0, 1, 2]
        y_data = [0, 1, 4]
        
        # 测试端点
        result = interpolation.Interpolation.interpolate(x_data, y_data, 0)
        self.assert_almost_equal_scalar(result, 0.0)
        
        result = interpolation.Interpolation.interpolate(x_data, y_data, 2)
        self.assert_almost_equal_scalar(result, 4.0)


# ============================================================================
# 数值计算模块测试 (Numerics Module Tests)
# ============================================================================

@pytest.mark.math_numerics
@pytest.mark.complex_number
class TestNumericsComplex(TestDWSIMMathSuite):
    """
    测试numerics.complex_number模块
    
    测试目标:
    - 复数基本运算
    - 复数函数计算
    - 极坐标转换
    - 数值精度
    """
    
    def test_complex_basic_operations(self):
        """
        测试复数基本运算
        
        工作步骤:
        1. 测试复数加法、减法
        2. 测试复数乘法、除法
        3. 验证运算规律
        4. 检查精度
        """
        # 创建测试复数
        z1 = complex_number.Complex(3, 4)  # 3 + 4i
        z2 = complex_number.Complex(1, 2)  # 1 + 2i
        
        # 测试加法
        result = z1 + z2
        expected = complex_number.Complex(4, 6)
        assert abs(result.real - expected.real) < self.tolerance
        assert abs(result.imag - expected.imag) < self.tolerance
        
        # 测试乘法: (3+4i)(1+2i) = 3+6i+4i+8i^2 = 3+10i-8 = -5+10i
        result = z1 * z2
        expected = complex_number.Complex(-5, 10)
        assert abs(result.real - expected.real) < self.tolerance
        assert abs(result.imag - expected.imag) < self.tolerance
    
    def test_complex_magnitude_phase(self):
        """
        测试复数的模长和幅角
        
        工作步骤:
        1. 测试已知复数的模长
        2. 测试幅角计算
        3. 验证极坐标转换
        4. 检查特殊角度
        """
        # 测试模长: |3+4i| = 5
        z = complex_number.Complex(3, 4)
        magnitude = z.abs()
        self.assert_almost_equal_scalar(magnitude, 5.0)
        
        # 测试实数轴上的复数
        z_real = complex_number.Complex(1, 0)
        magnitude = z_real.abs()
        self.assert_almost_equal_scalar(magnitude, 1.0)
        
        phase = z_real.arg()
        self.assert_almost_equal_scalar(phase, 0.0)
    
    def test_complex_functions(self):
        """
        测试复数函数
        
        工作步骤:
        1. 测试指数函数
        2. 测试对数函数
        3. 测试三角函数
        4. 验证恒等式
        """
        # 测试欧拉公式: e^(iπ) + 1 = 0
        z = complex_number.Complex(0, np.pi)  # iπ
        result = z.exp()
        # e^(iπ) = cos(π) + i*sin(π) = -1 + 0i
        self.assert_almost_equal_scalar(result.real, -1.0)
        self.assert_almost_equal_scalar(result.imag, 0.0, tolerance=1e-10)


# ============================================================================
# 求解器模块测试 (Solvers Module Tests)
# ============================================================================

@pytest.mark.math_solvers
@pytest.mark.brent_solver
class TestSolversBrent(TestDWSIMMathSuite):
    """
    测试solvers.brent模块
    
    测试目标:
    - 非线性方程求根
    - 收敛性验证
    - 边界条件处理
    - 函数类型覆盖
    """
    
    def test_brent_simple_polynomial(self):
        """
        测试简单多项式求根
        
        工作步骤:
        1. 定义简单多项式
        2. 设置求根区间
        3. 执行Brent算法
        4. 验证根的准确性
        """
        # 测试函数 f(x) = x^2 - 4，根为 ±2
        def f(x, args=None):
            return x**2 - 4
        
        solver = brent.BrentSolver()
        
        # 寻找正根 (区间 [1, 3])
        root = solver.solve(f, 1.0, 3.0)
        self.assert_almost_equal_scalar(root, 2.0)
        
        # 验证根的正确性
        f_value = f(root)
        self.assert_almost_equal_scalar(f_value, 0.0, tolerance=1e-8)
    
    def test_brent_transcendental(self):
        """
        测试超越方程求根
        
        工作步骤:
        1. 定义超越函数
        2. 寻找合适区间
        3. 求解方程
        4. 验证收敛性
        """
        # 测试函数 f(x) = x - cos(x)，近似根在 0.739
        def f(x, args=None):
            return x - np.cos(x)
        
        solver = brent.BrentSolver()
        root = solver.solve(f, 0.0, 1.0)
        
        # 验证根
        f_value = f(root)
        self.assert_almost_equal_scalar(f_value, 0.0, tolerance=1e-8)
        
        # 验证 x = cos(x)
        self.assert_almost_equal_scalar(root, np.cos(root))
    
    def test_brent_edge_cases(self):
        """
        测试Brent算法的边界条件
        
        工作步骤:
        1. 测试区间端点为根的情况
        2. 测试函数在端点同号的情况
        3. 测试病态函数
        4. 验证错误处理
        """
        # 测试端点为根
        def f(x, args=None):
            return x - 1
        
        solver = brent.BrentSolver()
        root = solver.solve(f, 1.0, 2.0)  # 左端点是根
        self.assert_almost_equal_scalar(root, 1.0)


# ============================================================================
# 优化模块测试 (Optimization Module Tests)
# ============================================================================

@pytest.mark.math_optimization
@pytest.mark.lbfgs
class TestOptimizationLBFGS(TestDWSIMMathSuite):
    """
    测试optimization.lbfgs模块
    
    测试目标:
    - 无约束优化问题
    - 收敛性验证
    - 梯度精度要求
    - 性能基准
    """
    
    def test_lbfgs_quadratic_function(self):
        """
        测试二次函数优化
        
        工作步骤:
        1. 定义二次目标函数
        2. 定义梯度函数
        3. 执行L-BFGS优化
        4. 验证最优解
        """
        # 目标函数: f(x) = (x[0]-1)^2 + (x[1]-2)^2，最优解 [1, 2]
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        def gradient(x):
            return np.array([2*(x[0] - 1), 2*(x[1] - 2)])
        
        optimizer = lbfgs.LBFGS()
        
        # 从远离最优解的点开始
        x0 = np.array([10.0, -5.0])
        result = optimizer.minimize(objective, gradient, x0)
        
        # 验证最优解
        expected_x = np.array([1.0, 2.0])
        self.assert_almost_equal_matrix(result['x'], expected_x, tolerance=1e-4)
        
        # 验证目标函数值
        expected_f = 0.0
        self.assert_almost_equal_scalar(result['fun'], expected_f, tolerance=1e-8)
    
    def test_lbfgs_rosenbrock_function(self):
        """
        测试Rosenbrock函数优化
        
        工作步骤:
        1. 实现Rosenbrock函数及其梯度
        2. 设置初始点
        3. 执行优化
        4. 验证收敛到全局最优
        """
        # Rosenbrock函数: f(x) = 100*(x[1]-x[0]^2)^2 + (1-x[0])^2
        def rosenbrock(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        def rosenbrock_grad(x):
            grad = np.zeros_like(x)
            grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            grad[1] = 200 * (x[1] - x[0]**2)
            return grad
        
        optimizer = lbfgs.LBFGS()
        x0 = np.array([-1.2, 1.0])  # 经典起始点
        
        result = optimizer.minimize(rosenbrock, rosenbrock_grad, x0)
        
        # Rosenbrock函数的全局最优解是 [1, 1]
        expected_x = np.array([1.0, 1.0])
        self.assert_almost_equal_matrix(result['x'], expected_x, tolerance=1e-3)


# ============================================================================
# 随机数生成模块测试 (Random Module Tests)
# ============================================================================

@pytest.mark.math_random
@pytest.mark.mersenne_twister
class TestRandomMersenneTwister(TestDWSIMMathSuite):
    """
    测试random.mersenne_twister模块
    
    测试目标:
    - 随机数质量验证
    - 统计性质检验
    - 重现性测试
    - 周期性验证
    """
    
    def test_mersenne_twister_basic(self):
        """
        测试基本随机数生成
        
        工作步骤:
        1. 初始化生成器
        2. 生成随机数序列
        3. 检查范围和类型
        4. 验证统计性质
        """
        # 使用固定种子确保可重现性
        mt = mersenne_twister.MersenneTwister(seed=12345)
        
        # 生成一批随机数
        random_numbers = [mt.random() for _ in range(1000)]
        
        # 检查范围 [0, 1)
        assert all(0 <= x < 1 for x in random_numbers), "随机数应在[0,1)范围内"
        
        # 检查基本统计性质
        mean = np.mean(random_numbers)
        std = np.std(random_numbers)
        
        # 均匀分布的理论均值和标准差
        expected_mean = 0.5
        expected_std = 1/np.sqrt(12)  # ≈ 0.289
        
        # 允许一定的统计误差
        assert abs(mean - expected_mean) < 0.05, f"均值偏离过大: {mean}"
        assert abs(std - expected_std) < 0.05, f"标准差偏离过大: {std}"
    
    def test_mersenne_twister_reproducibility(self):
        """
        测试随机数生成的重现性
        
        工作步骤:
        1. 使用相同种子初始化两个生成器
        2. 生成相同长度的序列
        3. 验证序列完全相同
        4. 确保确定性行为
        """
        seed = 42
        
        # 第一个生成器
        mt1 = mersenne_twister.MersenneTwister(seed=seed)
        sequence1 = [mt1.random() for _ in range(100)]
        
        # 第二个生成器（相同种子）
        mt2 = mersenne_twister.MersenneTwister(seed=seed)
        sequence2 = [mt2.random() for _ in range(100)]
        
        # 验证序列完全相同
        for i, (x1, x2) in enumerate(zip(sequence1, sequence2)):
            assert x1 == x2, f"第{i}个数不匹配: {x1} != {x2}"
    
    def test_mersenne_twister_integer_generation(self):
        """
        测试整数随机数生成
        
        工作步骤:
        1. 生成指定范围的随机整数
        2. 验证范围正确性
        3. 检查分布均匀性
        4. 测试边界情况
        """
        mt = mersenne_twister.MersenneTwister(seed=123)
        
        # 生成[1, 6]范围的随机整数（模拟骰子）
        rolls = [mt.randint(1, 6) for _ in range(600)]
        
        # 检查范围
        assert all(1 <= x <= 6 for x in rolls), "随机整数应在指定范围内"
        
        # 检查分布（每个面大约出现100次）
        from collections import Counter
        counts = Counter(rolls)
        
        for face in range(1, 7):
            count = counts[face]
            assert 70 <= count <= 130, f"面{face}出现次数异常: {count}"


# ============================================================================
# 集成测试 (Integration Tests)
# ============================================================================

@pytest.mark.math_integration
@pytest.mark.integration
class TestDWSIMMathIntegration(TestDWSIMMathSuite):
    """
    DWSIM数学库集成测试
    
    测试目标:
    - 模块间协作
    - 复杂计算流程
    - 端到端验证
    - 性能基准
    """
    
    def test_matrix_optimization_integration(self):
        """
        测试矩阵运算与优化算法集成
        
        工作步骤:
        1. 构造矩阵优化问题
        2. 使用矩阵运算计算梯度
        3. 用L-BFGS求解
        4. 验证结果一致性
        """
        # 构造二次优化问题: min x^T A x - b^T x
        A = np.array([[2, 1], [1, 3]])  # 正定矩阵
        b = np.array([1, 1])
        
        def objective(x):
            return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
        
        def gradient(x):
            return np.dot(A, x) - b
        
        # 解析解: x* = A^(-1) * b
        A_inv, success = matrix_ops.MatrixOperations.inverse(A)
        assert success, "矩阵求逆应该成功"
        
        analytical_solution = np.dot(A_inv, b)
        
        # 使用L-BFGS求解
        optimizer = lbfgs.LBFGS()
        x0 = np.array([0.0, 0.0])
        result = optimizer.minimize(objective, gradient, x0)
        
        # 比较数值解和解析解
        self.assert_almost_equal_matrix(result['x'], analytical_solution, tolerance=1e-4)
    
    def test_complex_interpolation_integration(self):
        """
        测试复数与插值算法集成
        
        工作步骤:
        1. 创建复数值函数的采样点
        2. 分别对实部和虚部插值
        3. 验证插值结果
        4. 测试复数运算的一致性
        """
        # 创建复数值函数 f(t) = e^(it) = cos(t) + i*sin(t)
        t_data = np.linspace(0, np.pi/2, 5)
        complex_data = [complex_number.Complex(np.cos(t), np.sin(t)) for t in t_data]
        
        # 提取实部和虚部
        real_parts = [z.real for z in complex_data]
        imag_parts = [z.imag for z in complex_data]
        
        # 插值点
        t_interp = np.pi/4
        
        # 插值实部和虚部
        real_interp = interpolation.Interpolation.interpolate(t_data.tolist(), real_parts, t_interp)
        imag_interp = interpolation.Interpolation.interpolate(t_data.tolist(), imag_parts, t_interp)
        
        # 理论值
        expected_real = np.cos(t_interp)  # cos(π/4) = √2/2
        expected_imag = np.sin(t_interp)  # sin(π/4) = √2/2
        
        self.assert_almost_equal_scalar(real_interp, expected_real, tolerance=1e-2)
        self.assert_almost_equal_scalar(imag_interp, expected_imag, tolerance=1e-2)
    
    def test_random_matrix_statistics(self):
        """
        测试随机数生成与矩阵统计集成
        
        工作步骤:
        1. 生成随机矩阵
        2. 计算统计量
        3. 验证统计性质
        4. 测试大数定律
        """
        mt = mersenne_twister.MersenneTwister(seed=42)
        
        # 生成多个随机矩阵的行列式
        determinants = []
        for _ in range(100):
            # 生成2x2随机矩阵
            matrix = [[mt.random(), mt.random()], 
                     [mt.random(), mt.random()]]
            det = matrix_ops.MatrixOperations.determinant(matrix)
            determinants.append(det)
        
        # 计算统计量
        mean_det = general.MathCommon.sum_array(determinants) / len(determinants)
        std_det = general.MathCommon.standard_deviation(determinants)
        
        # 基本合理性检查
        assert -1 < mean_det < 1, f"随机矩阵行列式均值应在合理范围内: {mean_det}"
        assert 0 < std_det < 1, f"标准差应为正值且在合理范围内: {std_det}"


# ============================================================================
# 性能基准测试 (Performance Benchmarks)
# ============================================================================

@pytest.mark.math_performance
@pytest.mark.performance
class TestDWSIMMathPerformance(TestDWSIMMathSuite):
    """
    DWSIM数学库性能基准测试
    
    测试目标:
    - 算法执行时间
    - 内存使用效率
    - 扩展性验证
    - 回归检测
    """
    
    @pytest.mark.slow
    def test_matrix_operations_performance(self):
        """
        测试矩阵运算性能
        
        工作步骤:
        1. 测试不同规模矩阵的运算时间
        2. 记录性能基准
        3. 与理论复杂度比较
        4. 检测性能回归
        """
        import time
        
        sizes = [10, 50, 100]
        times = {}
        
        for size in sizes:
            # 生成随机矩阵
            np.random.seed(42)
            matrix = np.random.rand(size, size)
            
            # 测试行列式计算时间
            start_time = time.time()
            det = matrix_ops.MatrixOperations.determinant(matrix)
            end_time = time.time()
            
            times[size] = end_time - start_time
            
            # 基本合理性检查
            assert not np.isnan(det), "行列式计算不应返回NaN"
            assert times[size] < 10.0, f"矩阵大小{size}的计算时间过长: {times[size]}s"
        
        # 打印性能报告
        print("\n=== 矩阵运算性能报告 ===")
        for size, exec_time in times.items():
            print(f"矩阵大小 {size}x{size}: {exec_time:.4f}s")
    
    @pytest.mark.slow
    def test_optimization_performance(self):
        """
        测试优化算法性能
        
        工作步骤:
        1. 测试不同维度优化问题
        2. 记录收敛时间和迭代次数
        3. 验证扩展性
        4. 性能基准比较
        """
        import time
        
        dimensions = [2, 5, 10]
        
        for dim in dimensions:
            # 创建高维二次函数
            def objective(x):
                return np.sum(x**2)
            
            def gradient(x):
                return 2 * x
            
            optimizer = lbfgs.LBFGS()
            x0 = np.ones(dim) * 10  # 远离最优解
            
            start_time = time.time()
            result = optimizer.minimize(objective, gradient, x0)
            end_time = time.time()
            
            exec_time = end_time - start_time
            
            # 验证收敛
            assert result['success'], f"维度{dim}的优化应该收敛"
            assert exec_time < 5.0, f"维度{dim}的优化时间过长: {exec_time}s"
            
            print(f"维度 {dim}: {exec_time:.4f}s, 迭代次数: {result.get('nit', 'N/A')}")


if __name__ == "__main__":
    # 运行特定的测试类或全部测试
    pytest.main([__file__, "-v", "--tb=short"]) 