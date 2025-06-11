"""
DWSIM数学计算库综合测试
======================

测试所有主要模块的功能，验证算法正确性和数值精度。
"""

import sys
import os
import numpy as np
import math
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有模块
from dwsim_math.core.general import MathCommon
from dwsim_math.core.matrix_ops import MatrixOperations, Determinant, Inverse
from dwsim_math.core.interpolation import Interpolation
from dwsim_math.numerics.complex_number import Complex
from dwsim_math.solvers.brent import BrentSolver
from dwsim_math.optimization.lbfgs import LBFGS
from dwsim_math.random.mersenne_twister import MersenneTwister


def test_general_functions():
    """测试通用数学函数"""
    print("=== 测试通用数学函数 ===")
    
    # 测试数据
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    # 测试基本统计函数
    max_val = MathCommon.max_value(data)
    min_val = MathCommon.min_value(data)
    avg = MathCommon.average(data)
    weighted_avg = MathCommon.weighted_average(weights, data)
    sum_squares = MathCommon.sum_of_squares(data)
    
    print(f"数据: {data}")
    print(f"最大值: {max_val} (预期: 10)")
    print(f"最小值: {min_val} (预期: 1)")
    print(f"平均值: {avg:.2f} (预期: 5.5)")
    print(f"加权平均: {weighted_avg:.2f}")
    print(f"平方和: {sum_squares} (预期: 385)")
    
    # 验证结果
    assert max_val == 10, "最大值计算错误"
    assert min_val == 1, "最小值计算错误"
    assert abs(avg - 5.5) < 1e-10, "平均值计算错误"
    assert sum_squares == 385, "平方和计算错误"
    
    print("✓ 通用数学函数测试通过\n")


def test_matrix_operations():
    """测试矩阵操作"""
    print("=== 测试矩阵操作 ===")
    
    # 测试矩阵
    A = [[2, 1, 3], 
         [1, 3, 2], 
         [3, 2, 1]]
    
    # 测试行列式
    det_A = MatrixOperations.determinant(A)
    print(f"矩阵A的行列式: {det_A}")
    
    # 测试逆矩阵
    inv_A, success = MatrixOperations.inverse(A)
    print(f"逆矩阵计算成功: {success}")
    
    if success:
        print("逆矩阵:")
        for row in inv_A:
            print([f"{x:.6f}" for x in row])
        
        # 验证 A * A^(-1) = I
        identity = np.dot(A, inv_A)
        print("A * A^(-1) (应该接近单位矩阵):")
        for row in identity:
            print([f"{x:.6f}" for x in row])
        
        # 检查是否接近单位矩阵
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(identity[i][j] - expected) < 1e-10, f"矩阵乘积不是单位矩阵: ({i},{j}) = {identity[i][j]}"
    
    # 测试线性方程组求解
    b = [1, 2, 3]
    x, success = MatrixOperations.solve_linear_system(A, b)
    if success:
        print(f"线性方程组解: {x}")
        
        # 验证解的正确性
        Ax = [sum(A[i][j] * x[j] for j in range(3)) for i in range(3)]
        print(f"验证 Ax = {Ax}, b = {b}")
        
        for i in range(3):
            assert abs(Ax[i] - b[i]) < 1e-10, f"线性方程组解错误: Ax[{i}] = {Ax[i]}, b[{i}] = {b[i]}"
    
    print("✓ 矩阵操作测试通过\n")


def test_interpolation():
    """测试插值算法"""
    print("=== 测试插值算法 ===")
    
    # 测试数据 (y = x^2)
    x_data = [0, 1, 2, 3, 4]
    y_data = [0, 1, 4, 9, 16]
    
    # 插值点
    x_test = 2.5
    expected = 2.5 ** 2  # 6.25
    
    # 测试不同插值方法
    linear_result = Interpolation.interpolate(x_data, y_data, x_test, method="linear")
    rational_result = Interpolation.interpolate(x_data, y_data, x_test, method="rational")
    polynomial_result = Interpolation.interpolate(x_data, y_data, x_test, method="polynomial")
    
    print(f"插值点: x = {x_test}")
    print(f"理论值: y = {expected}")
    print(f"线性插值: {linear_result:.6f}")
    print(f"有理插值: {rational_result:.6f}")
    print(f"多项式插值: {polynomial_result:.6f}")
    
    # 多项式插值对于二次函数应该是精确的
    assert abs(polynomial_result - expected) < 1e-10, f"多项式插值错误: {polynomial_result} != {expected}"
    
    print("✓ 插值算法测试通过\n")


def test_complex_numbers():
    """测试复数运算"""
    print("=== 测试复数运算 ===")
    
    # 创建复数
    z1 = Complex(3, 4)  # 3 + 4i
    z2 = Complex(1, -2)  # 1 - 2i
    
    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
    
    # 测试基本运算
    z_add = z1 + z2
    z_sub = z1 - z2
    z_mul = z1 * z2
    z_div = z1 / z2
    
    print(f"z1 + z2 = {z_add}")
    print(f"z1 - z2 = {z_sub}")
    print(f"z1 * z2 = {z_mul}")
    print(f"z1 / z2 = {z_div}")
    
    # 验证运算结果
    assert z_add.real == 4 and z_add.imaginary == 2, "复数加法错误"
    assert z_sub.real == 2 and z_sub.imaginary == 6, "复数减法错误"
    
    # 测试复数属性
    print(f"z1的模长: {z1.modulus:.6f}")
    print(f"z1的幅角: {z1.argument:.6f}")
    print(f"z1的共轭: {z1.conjugate}")
    
    # 验证模长
    expected_modulus = math.sqrt(3**2 + 4**2)
    assert abs(z1.modulus - expected_modulus) < 1e-10, "复数模长计算错误"
    
    # 测试复数函数
    z_exp = z1.exp()
    z_log = z1.log()
    z_sqrt = z1.sqrt()
    
    print(f"exp(z1) = {z_exp}")
    print(f"log(z1) = {z_log}")
    print(f"sqrt(z1) = {z_sqrt}")
    
    # 验证 exp(log(z)) = z
    z_exp_log = z_log.exp()
    assert abs(z_exp_log.real - z1.real) < 1e-10 and abs(z_exp_log.imaginary - z1.imaginary) < 1e-10, "exp(log(z)) != z"
    
    print("✓ 复数运算测试通过\n")


def test_brent_solver():
    """测试Brent求解器"""
    print("=== 测试Brent求解器 ===")
    
    solver = BrentSolver()
    
    # 测试方程 x^3 - 2x - 5 = 0
    def func1(x, args):
        return x**3 - 2*x - 5
    
    root1 = solver.solve(func1, 1.0, 3.0)
    print(f"x^3 - 2x - 5 = 0 的根: {root1:.6f}")
    print(f"验证: f({root1}) = {func1(root1, None):.2e}")
    
    # 验证根的精度
    assert abs(func1(root1, None)) < 1e-10, "根求解精度不足"
    
    # 测试优化问题 min (x-2)^2 + 3
    def objective(x, args):
        return (x - 2)**2 + 3
    
    x_min, f_min = solver.find_minimum(objective, 0.0, 5.0)
    print(f"(x-2)^2 + 3 的最小值点: x = {x_min:.6f}")
    print(f"最小值: f = {f_min:.6f}")
    
    # 验证最优解
    assert abs(x_min - 2.0) < 1e-6, "优化求解错误"
    assert abs(f_min - 3.0) < 1e-6, "最优值错误"
    
    print("✓ Brent求解器测试通过\n")


def test_lbfgs_optimizer():
    """测试L-BFGS优化器"""
    print("=== 测试L-BFGS优化器 ===")
    
    # 定义二次函数及其梯度
    def quadratic_func_grad(x):
        # f(x) = (x[0]-1)^2 + (x[1]-2)^2
        f = (x[0] - 1)**2 + (x[1] - 2)**2
        grad = np.array([2*(x[0] - 1), 2*(x[1] - 2)])
        return f, grad
    
    # 创建优化器
    optimizer = LBFGS(eps_g=1e-6, max_iterations=100)
    
    # 优化
    initial_point = np.array([5.0, -3.0])
    result = optimizer.minimize(quadratic_func_grad, initial_point)
    
    print(f"优化成功: {result['success']}")
    print(f"最优解: ({result['x'][0]:.6f}, {result['x'][1]:.6f})")
    print(f"最优值: {result['fun']:.6f}")
    print(f"迭代次数: {result['nit']}")
    print(f"函数评估次数: {result['nfev']}")
    
    # 验证结果（理论最优解是 (1, 2)）
    assert abs(result['x'][0] - 1.0) < 1e-4, "L-BFGS优化x坐标错误"
    assert abs(result['x'][1] - 2.0) < 1e-4, "L-BFGS优化y坐标错误"
    assert result['fun'] < 1e-8, "L-BFGS优化函数值错误"
    
    print("✓ L-BFGS优化器测试通过\n")


def test_mersenne_twister():
    """测试Mersenne Twister随机数生成器"""
    print("=== 测试Mersenne Twister随机数生成器 ===")
    
    # 创建随机数生成器
    rng = MersenneTwister(seed=12345)
    
    # 测试均匀分布
    uniform_samples = [rng.random() for _ in range(1000)]
    mean_uniform = np.mean(uniform_samples)
    print(f"均匀分布[0,1)样本均值: {mean_uniform:.4f} (理论值: 0.5)")
    
    # 验证均值在合理范围内
    assert abs(mean_uniform - 0.5) < 0.1, "均匀分布均值偏差过大"
    
    # 测试正态分布
    normal_samples = [rng.normal(0, 1) for _ in range(10000)]
    mean_normal = np.mean(normal_samples)
    std_normal = np.std(normal_samples)
    
    print(f"标准正态分布样本均值: {mean_normal:.4f} (理论值: 0)")
    print(f"标准正态分布样本标准差: {std_normal:.4f} (理论值: 1)")
    
    # 验证正态分布参数
    assert abs(mean_normal) < 0.1, "正态分布均值偏差过大"
    assert abs(std_normal - 1.0) < 0.1, "正态分布标准差偏差过大"
    
    # 测试随机整数
    random_ints = [rng.randint(1, 11) for _ in range(1000)]
    assert all(1 <= x <= 10 for x in random_ints), "随机整数超出范围"
    
    # 测试随机选择和洗牌
    items = list(range(10))
    choice = rng.choice(items)
    assert choice in items, "随机选择结果不在原列表中"
    
    items_copy = items.copy()
    rng.shuffle(items_copy)
    assert sorted(items_copy) == sorted(items), "洗牌后元素不完整"
    assert items_copy != items, "洗牌没有改变顺序（可能性极小）"
    
    print("✓ Mersenne Twister随机数生成器测试通过\n")


def test_performance():
    """性能测试"""
    print("=== 性能测试 ===")
    
    # 矩阵运算性能
    print("矩阵运算性能:")
    rng = MersenneTwister(seed=42)
    
    sizes = [50, 100]
    for size in sizes:
        # 生成随机矩阵
        matrix = [[rng.normal(0, 1) for _ in range(size)] for _ in range(size)]
        
        # 测试行列式计算时间
        start_time = time.time()
        det_result = MatrixOperations.determinant(matrix)
        elapsed_time = time.time() - start_time
        
        print(f"  {size}×{size} 矩阵行列式计算: {elapsed_time:.4f}s")
    
    # 随机数生成性能
    print("随机数生成性能:")
    rng = MersenneTwister(seed=42)
    
    n_samples = 100000
    start_time = time.time()
    samples = [rng.random() for _ in range(n_samples)]
    elapsed_time = time.time() - start_time
    
    print(f"  生成{n_samples}个随机数: {elapsed_time:.4f}s")
    print(f"  速度: {n_samples/elapsed_time:.0f} 数/秒")
    
    print("✓ 性能测试完成\n")


def run_all_tests():
    """运行所有测试"""
    print("DWSIM数学计算库综合测试")
    print("=" * 50)
    
    try:
        test_general_functions()
        test_matrix_operations()
        test_interpolation()
        test_complex_numbers()
        test_brent_solver()
        test_lbfgs_optimizer()
        test_mersenne_twister()
        test_performance()
        
        print("🎉 所有测试通过！")
        print("DWSIM数学计算库功能正常，可以使用。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1) 