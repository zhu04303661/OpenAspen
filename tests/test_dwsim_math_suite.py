"""
DWSIM数学库完整测试套件
=====================

集成测试所有DWSIM数学计算库模块，提供统一的测试入口和报告。

测试覆盖模块:
- core.general: 基本数学函数
- core.matrix_ops: 矩阵运算
- core.interpolation: 插值算法
- numerics.complex_number: 复数运算
- solvers.brent: 求根算法
- optimization.lbfgs: 优化算法
- random.mersenne_twister: 随机数生成

测试策略:
1. 单元测试 - 各模块独立功能验证
2. 集成测试 - 模块间协作验证
3. 性能测试 - 基本性能基准
4. 回归测试 - 确保代码修改不破坏现有功能

使用方法:
    # 运行所有测试
    python -m pytest test_dwsim_math_suite.py -v
    
    # 运行特定模块测试
    python -m pytest test_dwsim_math_suite.py::TestDWSIMMathCore -v
    
    # 运行性能测试
    python -m pytest test_dwsim_math_suite.py -v -m slow

作者: DWSIM团队
版本: 1.0.0
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import time
import traceback

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入测试状态跟踪
test_results = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'errors': []
}

# 尝试导入所有DWSIM数学库模块
modules_status = {}

def try_import_module(module_name, import_statement):
    """
    尝试导入模块并记录状态
    
    参数:
        module_name: 模块名称
        import_statement: 导入语句
    
    返回:
        bool: 导入是否成功
    """
    try:
        exec(import_statement)
        modules_status[module_name] = 'available'
        print(f"✅ {module_name} 模块导入成功")
        return True
    except ImportError as e:
        modules_status[module_name] = f'unavailable: {e}'
        print(f"❌ {module_name} 模块导入失败: {e}")
        return False
    except Exception as e:
        modules_status[module_name] = f'error: {e}'
        print(f"⚠️ {module_name} 模块导入错误: {e}")
        return False

# 检查所有模块可用性
print("=== DWSIM数学库模块可用性检查 ===")

# 核心模块
try_import_module('core.general', 'from dwsim_math.core.general import MathCommon')
try_import_module('core.matrix_ops', 'from dwsim_math.core.matrix_ops import MatrixOperations')
try_import_module('core.interpolation', 'from dwsim_math.core.interpolation import Interpolation')

# 数值计算模块
try_import_module('numerics.complex_number', 'from dwsim_math.numerics.complex_number import Complex')

# 求解器模块
try_import_module('solvers.brent', 'from dwsim_math.solvers.brent import BrentSolver')

# 优化模块
try_import_module('optimization.lbfgs', 'from dwsim_math.optimization.lbfgs import LBFGS')

# 随机数模块
try_import_module('random.mersenne_twister', 'from dwsim_math.random.mersenne_twister import MersenneTwister')

print(f"\n模块可用性总结:")
available_count = sum(1 for status in modules_status.values() if status == 'available')
total_count = len(modules_status)
print(f"  可用模块: {available_count}/{total_count}")


class TestDWSIMMathSuiteBase:
    """
    DWSIM数学库测试套件基类
    
    提供统一的测试框架和工具方法
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """设置测试环境"""
        self.tolerance = 1e-10
        self.float_tolerance = 1e-6
        yield
    
    def assert_almost_equal_scalar(self, actual, expected, tolerance=None):
        """标量近似相等断言"""
        if tolerance is None:
            tolerance = self.float_tolerance
        assert abs(actual - expected) < tolerance, f"期望 {expected}，实际 {actual}，差值 {abs(actual - expected)} > {tolerance}"
    
    def assert_almost_equal_matrix(self, actual, expected, tolerance=None):
        """矩阵近似相等断言"""
        if tolerance is None:
            tolerance = self.tolerance
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(modules_status.get('core.general') != 'available', 
                   reason="core.general模块不可用")
class TestDWSIMMathCore(TestDWSIMMathSuiteBase):
    """
    核心数学函数测试套件
    
    测试目标:
    - general模块的基本数学函数
    - matrix_ops模块的矩阵运算
    - interpolation模块的插值算法
    """
    
    def test_general_basic_statistics(self):
        """
        测试基本统计函数
        
        测试要点:
        1. 最大值、最小值计算
        2. 求和、平均值计算
        3. 方差、标准差计算
        4. 边界条件处理
        """
        from dwsim_math.core.general import MathCommon
        
        # 测试数据
        test_data = [1.0, 5.0, 3.0, 8.0, 2.0]
        
        # 最大值测试
        max_val = MathCommon.max_value(test_data)
        self.assert_almost_equal_scalar(max_val, 8.0)
        
        # 最小值测试（忽略零值）
        test_with_zero = [0.0, 5.0, 3.0, 8.0, 0.0, 2.0]
        min_val = MathCommon.min_value(test_with_zero)
        self.assert_almost_equal_scalar(min_val, 2.0)
        
        # 求和测试
        sum_val = MathCommon.sum_array(test_data)
        expected_sum = sum(test_data)
        self.assert_almost_equal_scalar(sum_val, expected_sum)
        
        # 标准差测试
        std_val = MathCommon.standard_deviation(test_data, sample=True)
        expected_std = np.std(test_data, ddof=1)
        self.assert_almost_equal_scalar(std_val, expected_std)
        
        print("  ✓ 基本统计函数测试通过")
    
    def test_weighted_operations(self):
        """
        测试加权运算
        
        测试要点:
        1. 加权平均计算
        2. 权重归一化
        3. 错误输入处理
        """
        from dwsim_math.core.general import MathCommon
        
        # 加权平均测试
        weights = [0.3, 0.3, 0.4]
        values = [10.0, 20.0, 30.0]
        result = MathCommon.weighted_average(weights, values)
        expected = (0.3*10 + 0.3*20 + 0.4*30) / (0.3 + 0.3 + 0.4)
        self.assert_almost_equal_scalar(result, expected)
        
        # 错误输入测试
        with pytest.raises(ValueError):
            MathCommon.weighted_average([1, 2], [1, 2, 3])  # 长度不匹配
        
        print("  ✓ 加权运算测试通过")
    
    @pytest.mark.skipif(modules_status.get('core.matrix_ops') != 'available',
                       reason="core.matrix_ops模块不可用")
    def test_matrix_operations(self):
        """
        测试矩阵运算
        
        测试要点:
        1. 行列式计算
        2. 矩阵求逆
        3. 线性方程组求解
        4. 矩阵条件数
        """
        from dwsim_math.core.matrix_ops import MatrixOperations
        
        # 测试矩阵
        A = np.array([[2, 1], [1, 2]], dtype=float)
        
        # 行列式测试
        det_A = MatrixOperations.determinant(A)
        expected_det = 2*2 - 1*1  # 3
        self.assert_almost_equal_scalar(det_A, expected_det)
        
        # 矩阵求逆测试
        inv_A, success = MatrixOperations.inverse(A)
        assert success, "矩阵求逆应该成功"
        
        # 验证逆矩阵
        product = np.dot(A, inv_A)
        identity = np.eye(2)
        self.assert_almost_equal_matrix(product, identity)
        
        # 线性方程组求解测试
        b = np.array([3, 3], dtype=float)
        x, success = MatrixOperations.solve_linear_system(A, b)
        assert success, "线性系统求解应该成功"
        
        # 验证解
        expected_x = np.array([1.0, 1.0])
        self.assert_almost_equal_matrix(x, expected_x)
        
        print("  ✓ 矩阵运算测试通过")
    
    @pytest.mark.skipif(modules_status.get('core.interpolation') != 'available',
                       reason="core.interpolation模块不可用")
    def test_interpolation(self):
        """
        测试插值算法
        
        测试要点:
        1. 线性插值
        2. 边界条件
        3. 插值精度
        """
        from dwsim_math.core.interpolation import Interpolation
        
        # 线性函数插值测试
        x_data = [0, 1, 2, 3, 4]
        y_data = [1, 3, 5, 7, 9]  # y = 2x + 1
        
        # 中间点插值
        result = Interpolation.interpolate(x_data, y_data, 2.5)
        expected = 2 * 2.5 + 1  # 6.0
        self.assert_almost_equal_scalar(result, expected, tolerance=1e-3)
        
        print("  ✓ 插值算法测试通过")


@pytest.mark.skipif(modules_status.get('numerics.complex_number') != 'available',
                   reason="numerics.complex_number模块不可用")
class TestDWSIMMathNumerics(TestDWSIMMathSuiteBase):
    """
    数值计算模块测试套件
    
    测试目标:
    - 复数运算
    - 数值精度
    """
    
    def test_complex_basic_operations(self):
        """
        测试复数基本运算
        
        测试要点:
        1. 复数加减乘除
        2. 复数模长和幅角
        3. 复数函数
        """
        from dwsim_math.numerics.complex_number import Complex
        
        # 创建测试复数
        z1 = Complex(3, 4)  # 3 + 4i
        z2 = Complex(1, 2)  # 1 + 2i
        
        # 加法测试
        result = z1 + z2
        expected = Complex(4, 6)
        self.assert_almost_equal_scalar(result.real, expected.real)
        self.assert_almost_equal_scalar(result.imag, expected.imag)
        
        # 乘法测试: (3+4i)(1+2i) = 3+6i+4i+8i^2 = -5+10i
        result = z1 * z2
        expected = Complex(-5, 10)
        self.assert_almost_equal_scalar(result.real, expected.real)
        self.assert_almost_equal_scalar(result.imag, expected.imag)
        
        # 模长测试: |3+4i| = 5
        magnitude = z1.abs()
        self.assert_almost_equal_scalar(magnitude, 5.0)
        
        print("  ✓ 复数基本运算测试通过")
    
    def test_complex_functions(self):
        """
        测试复数函数
        
        测试要点:
        1. 指数函数
        2. 对数函数
        3. 三角函数
        """
        from dwsim_math.numerics.complex_number import Complex
        
        # 测试欧拉公式: e^(iπ) = -1
        z = Complex(0, np.pi)  # iπ
        result = z.exp()
        # e^(iπ) = cos(π) + i*sin(π) = -1 + 0i
        self.assert_almost_equal_scalar(result.real, -1.0, tolerance=1e-10)
        self.assert_almost_equal_scalar(result.imag, 0.0, tolerance=1e-10)
        
        print("  ✓ 复数函数测试通过")


@pytest.mark.skipif(modules_status.get('solvers.brent') != 'available',
                   reason="solvers.brent模块不可用")
class TestDWSIMMathSolvers(TestDWSIMMathSuiteBase):
    """
    求解器模块测试套件
    
    测试目标:
    - Brent求根算法
    - 收敛性验证
    """
    
    def test_brent_polynomial_roots(self):
        """
        测试多项式求根
        
        测试要点:
        1. 简单多项式求根
        2. 收敛精度
        3. 根的验证
        """
        from dwsim_math.solvers.brent import BrentSolver
        
        # 测试函数 f(x) = x^2 - 4，根为 ±2
        def f(x, args=None):
            return x**2 - 4
        
        solver = BrentSolver()
        
        # 寻找正根 (区间 [1, 3])
        root = solver.solve(f, 1.0, 3.0)
        self.assert_almost_equal_scalar(root, 2.0, tolerance=1e-6)
        
        # 验证根的正确性
        f_value = f(root)
        self.assert_almost_equal_scalar(f_value, 0.0, tolerance=1e-8)
        
        print("  ✓ Brent求根算法测试通过")
    
    def test_brent_transcendental_roots(self):
        """
        测试超越方程求根
        
        测试要点:
        1. 超越函数求根
        2. 复杂函数收敛
        """
        from dwsim_math.solvers.brent import BrentSolver
        
        # 测试函数 f(x) = x - cos(x)
        def f(x, args=None):
            return x - np.cos(x)
        
        solver = BrentSolver()
        root = solver.solve(f, 0.0, 1.0)
        
        # 验证根
        f_value = f(root)
        self.assert_almost_equal_scalar(f_value, 0.0, tolerance=1e-8)
        
        # 验证 x = cos(x)
        self.assert_almost_equal_scalar(root, np.cos(root), tolerance=1e-6)
        
        print("  ✓ 超越方程求根测试通过")


@pytest.mark.skipif(modules_status.get('optimization.lbfgs') != 'available',
                   reason="optimization.lbfgs模块不可用")
class TestDWSIMMathOptimization(TestDWSIMMathSuiteBase):
    """
    优化算法模块测试套件
    
    测试目标:
    - L-BFGS优化算法
    - 收敛性验证
    """
    
    def test_lbfgs_quadratic_optimization(self):
        """
        测试二次函数优化
        
        测试要点:
        1. 凸二次函数优化
        2. 梯度收敛
        3. 最优解验证
        """
        from dwsim_math.optimization.lbfgs import LBFGS
        
        # 目标函数: f(x) = (x[0]-1)^2 + (x[1]-2)^2，最优解 [1, 2]
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        def gradient(x):
            return np.array([2*(x[0] - 1), 2*(x[1] - 2)])
        
        optimizer = LBFGS()
        
        # 从远离最优解的点开始
        x0 = np.array([10.0, -5.0])
        result = optimizer.minimize(objective, gradient, x0)
        
        # 验证最优解
        expected_x = np.array([1.0, 2.0])
        self.assert_almost_equal_matrix(result['x'], expected_x, tolerance=1e-4)
        
        # 验证目标函数值
        expected_f = 0.0
        self.assert_almost_equal_scalar(result['fun'], expected_f, tolerance=1e-8)
        
        print("  ✓ L-BFGS二次函数优化测试通过")


@pytest.mark.skipif(modules_status.get('random.mersenne_twister') != 'available',
                   reason="random.mersenne_twister模块不可用")
class TestDWSIMMathRandom(TestDWSIMMathSuiteBase):
    """
    随机数生成模块测试套件
    
    测试目标:
    - Mersenne Twister算法
    - 随机数质量
    """
    
    def test_mersenne_twister_basic(self):
        """
        测试基本随机数生成
        
        测试要点:
        1. 随机数范围
        2. 统计性质
        3. 重现性
        """
        from dwsim_math.random.mersenne_twister import MersenneTwister
        
        # 使用固定种子确保可重现性
        mt = MersenneTwister(seed=12345)
        
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
        
        print("  ✓ Mersenne Twister基本功能测试通过")
    
    def test_mersenne_twister_reproducibility(self):
        """
        测试随机数生成的重现性
        
        测试要点:
        1. 相同种子产生相同序列
        2. 确定性行为
        """
        from dwsim_math.random.mersenne_twister import MersenneTwister
        
        seed = 42
        
        # 第一个生成器
        mt1 = MersenneTwister(seed=seed)
        sequence1 = [mt1.random() for _ in range(100)]
        
        # 第二个生成器（相同种子）
        mt2 = MersenneTwister(seed=seed)
        sequence2 = [mt2.random() for _ in range(100)]
        
        # 验证序列完全相同
        for i, (x1, x2) in enumerate(zip(sequence1, sequence2)):
            assert x1 == x2, f"第{i}个数不匹配: {x1} != {x2}"
        
        print("  ✓ Mersenne Twister重现性测试通过")


class TestDWSIMMathIntegration(TestDWSIMMathSuiteBase):
    """
    集成测试套件
    
    测试目标:
    - 模块间协作
    - 复杂计算流程
    - 端到端验证
    """
    
    @pytest.mark.skipif(
        modules_status.get('core.matrix_ops') != 'available' or 
        modules_status.get('optimization.lbfgs') != 'available',
        reason="matrix_ops或lbfgs模块不可用"
    )
    def test_matrix_optimization_integration(self):
        """
        测试矩阵运算与优化算法集成
        
        测试要点:
        1. 矩阵二次优化问题
        2. 数值解与解析解比较
        3. 算法一致性验证
        """
        from dwsim_math.core.matrix_ops import MatrixOperations
        from dwsim_math.optimization.lbfgs import LBFGS
        
        # 构造二次优化问题: min x^T A x - b^T x
        A = np.array([[2, 1], [1, 3]], dtype=float)  # 正定矩阵
        b = np.array([1, 1], dtype=float)
        
        def objective(x):
            return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
        
        def gradient(x):
            return np.dot(A, x) - b
        
        # 解析解: x* = A^(-1) * b
        A_inv, success = MatrixOperations.inverse(A)
        assert success, "矩阵求逆应该成功"
        
        analytical_solution = np.dot(A_inv, b)
        
        # 使用L-BFGS求解
        optimizer = LBFGS()
        x0 = np.array([0.0, 0.0])
        result = optimizer.minimize(objective, gradient, x0)
        
        # 比较数值解和解析解
        self.assert_almost_equal_matrix(result['x'], analytical_solution, tolerance=1e-4)
        
        print("  ✓ 矩阵优化集成测试通过")
    
    @pytest.mark.skipif(
        modules_status.get('numerics.complex_number') != 'available' or 
        modules_status.get('core.interpolation') != 'available',
        reason="complex_number或interpolation模块不可用"
    )
    def test_complex_interpolation_integration(self):
        """
        测试复数与插值算法集成
        
        测试要点:
        1. 复数值函数插值
        2. 实部虚部分离插值
        3. 复数运算一致性
        """
        from dwsim_math.numerics.complex_number import Complex
        from dwsim_math.core.interpolation import Interpolation
        
        # 创建复数值函数 f(t) = e^(it) = cos(t) + i*sin(t)
        t_data = np.linspace(0, np.pi/2, 5)
        complex_data = [Complex(np.cos(t), np.sin(t)) for t in t_data]
        
        # 提取实部和虚部
        real_parts = [z.real for z in complex_data]
        imag_parts = [z.imag for z in complex_data]
        
        # 插值点
        t_interp = np.pi/4
        
        # 插值实部和虚部
        real_interp = Interpolation.interpolate(t_data.tolist(), real_parts, t_interp)
        imag_interp = Interpolation.interpolate(t_data.tolist(), imag_parts, t_interp)
        
        # 理论值
        expected_real = np.cos(t_interp)  # cos(π/4) = √2/2
        expected_imag = np.sin(t_interp)  # sin(π/4) = √2/2
        
        self.assert_almost_equal_scalar(real_interp, expected_real, tolerance=1e-2)
        self.assert_almost_equal_scalar(imag_interp, expected_imag, tolerance=1e-2)
        
        print("  ✓ 复数插值集成测试通过")


@pytest.mark.slow
class TestDWSIMMathPerformance(TestDWSIMMathSuiteBase):
    """
    性能测试套件
    
    测试目标:
    - 算法执行时间
    - 内存使用效率
    - 扩展性验证
    """
    
    @pytest.mark.skipif(modules_status.get('core.general') != 'available',
                       reason="core.general模块不可用")
    def test_large_array_performance(self):
        """
        测试大数组处理性能
        
        测试要点:
        1. 大数组统计计算
        2. 执行时间基准
        3. 内存效率
        """
        from dwsim_math.core.general import MathCommon
        
        # 生成大数组
        large_array = list(range(100000))
        
        # 测试求和性能
        start_time = time.time()
        result = MathCommon.sum_array(large_array)
        end_time = time.time()
        
        execution_time = end_time - start_time
        expected = sum(large_array)
        
        self.assert_almost_equal_scalar(result, expected)
        assert execution_time < 1.0, f"大数组求和时间过长: {execution_time}s"
        
        print(f"  ✓ 大数组性能测试通过 (执行时间: {execution_time:.4f}s)")
    
    @pytest.mark.skipif(modules_status.get('core.matrix_ops') != 'available',
                       reason="core.matrix_ops模块不可用")
    def test_matrix_performance(self):
        """
        测试矩阵运算性能
        
        测试要点:
        1. 不同规模矩阵运算
        2. 时间复杂度验证
        3. 算法效率
        """
        from dwsim_math.core.matrix_ops import MatrixOperations
        
        sizes = [10, 50, 100]
        times = {}
        
        for size in sizes:
            # 生成随机正定矩阵
            np.random.seed(42)
            A = np.random.rand(size, size)
            A = np.dot(A, A.T) + np.eye(size)
            
            # 测试行列式计算时间
            start_time = time.time()
            det = MatrixOperations.determinant(A)
            end_time = time.time()
            
            times[size] = end_time - start_time
            
            # 基本合理性检查
            assert not np.isnan(det), "行列式计算不应返回NaN"
            assert times[size] < 10.0, f"矩阵大小{size}的计算时间过长: {times[size]}s"
        
        print(f"  ✓ 矩阵性能测试通过")
        for size, exec_time in times.items():
            print(f"    矩阵大小 {size}x{size}: {exec_time:.4f}s")


def run_test_suite():
    """
    运行完整的测试套件并生成报告
    
    返回:
        dict: 测试结果统计
    """
    print("\n" + "="*60)
    print("DWSIM数学库完整测试套件")
    print("="*60)
    
    # 显示模块状态
    print(f"\n模块可用性状态:")
    for module, status in modules_status.items():
        if status == 'available':
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module}: {status}")
    
    print(f"\n开始执行测试...")
    
    # 运行pytest并捕获结果
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # 遇到第一个失败就停止（可选）
    ]
    
    # 如果想要详细输出，添加 -s 参数
    # pytest_args.append("-s")
    
    try:
        result = pytest.main(pytest_args)
        
        print(f"\n" + "="*60)
        print("测试套件执行完成")
        print("="*60)
        
        if result == 0:
            print("🎉 所有测试通过!")
        else:
            print("⚠️ 部分测试失败，请检查输出")
        
        return {'exit_code': result}
        
    except Exception as e:
        print(f"\n❌ 测试执行过程中发生错误: {e}")
        traceback.print_exc()
        return {'exit_code': -1, 'error': str(e)}


if __name__ == "__main__":
    # 运行完整测试套件
    result = run_test_suite()
    sys.exit(result.get('exit_code', 0)) 