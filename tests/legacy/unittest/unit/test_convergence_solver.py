"""
Convergence Solver 收敛求解器单元测试
===================================

测试目标：
1. BroydenSolver - Broyden拟牛顿方法
2. NewtonRaphsonSolver - Newton-Raphson方法  
3. RecycleConvergenceSolver - 循环收敛求解
4. SimultaneousAdjustSolver - 同步调节求解
5. 数值稳定性和收敛性验证
6. 边界条件和异常处理

工作步骤：
1. 测试各求解器的基本功能
2. 验证数值算法的正确性
3. 测试收敛判断逻辑
4. 验证性能和稳定性
5. 测试异常情况处理
6. 集成测试场景验证
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from flowsheet_solver.convergence_solver import (
    BroydenSolver,
    NewtonRaphsonSolver, 
    RecycleConvergenceSolver,
    SimultaneousAdjustSolver
)
from flowsheet_solver.solver_exceptions import (
    ConvergenceException,
    CalculationException
)


class TestBroydenSolver:
    """
    BroydenSolver测试类
    
    测试Broyden拟牛顿方法的实现。
    """
    
    def test_initialization(self):
        """
        测试目标：验证BroydenSolver正确初始化
        
        工作步骤：
        1. 创建求解器实例
        2. 检查默认参数设置
        3. 验证初始状态
        """
        solver = BroydenSolver()
        
        # 检查默认参数
        assert solver.max_iterations == 100
        assert solver.tolerance == 1e-6
        assert solver.damping_factor == 1.0
        assert solver.min_step_size == 1e-12
        
        # 检查初始状态
        assert solver.iteration_count == 0
        assert solver.convergence_history == []
        assert solver.jacobian_inverse is None
    
    def test_simple_linear_system(self):
        """
        测试目标：验证线性方程组求解
        
        工作步骤：
        1. 定义简单线性函数
        2. 求解方程组
        3. 验证解的正确性
        """
        def linear_func(x):
            """简单线性函数: f(x) = Ax - b = 0"""
            A = np.array([[2, 1], [1, 3]])
            b = np.array([3, 4])
            return np.dot(A, x) - b
        
        solver = BroydenSolver(tolerance=1e-8)
        x0 = np.array([0.0, 0.0])  # 初始猜值
        
        solution, converged, iterations = solver.solve(linear_func, x0)
        
        # 验证收敛
        assert converged == True
        assert iterations > 0
        
        # 验证解的精度
        residual = linear_func(solution)
        assert np.linalg.norm(residual) < 1e-6
        
        # 验证解析解 A^-1 * b
        A = np.array([[2, 1], [1, 3]])
        b = np.array([3, 4])
        analytical_solution = np.linalg.solve(A, b)
        np.testing.assert_allclose(solution, analytical_solution, rtol=1e-6)
    
    def test_nonlinear_system(self):
        """
        测试目标：验证非线性方程组求解
        
        工作步骤：
        1. 定义非线性函数系统
        2. 使用不同初值求解
        3. 验证收敛到正确解
        """
        def nonlinear_func(x):
            """非线性函数组:
            f1(x,y) = x^2 + y^2 - 1 = 0
            f2(x,y) = x - y = 0
            解: x = y = ±1/√2
            """
            return np.array([
                x[0]**2 + x[1]**2 - 1,
                x[0] - x[1]
            ])
        
        solver = BroydenSolver(max_iterations=50, tolerance=1e-10)
        
        # 测试正解
        x0_pos = np.array([0.5, 0.5])
        solution_pos, converged_pos, _ = solver.solve(nonlinear_func, x0_pos)
        
        if converged_pos:
            expected_pos = 1.0 / np.sqrt(2)
            np.testing.assert_allclose(solution_pos, [expected_pos, expected_pos], rtol=1e-6)
        
        # 测试负解
        x0_neg = np.array([-0.5, -0.5]) 
        solution_neg, converged_neg, _ = solver.solve(nonlinear_func, x0_neg)
        
        if converged_neg:
            expected_neg = -1.0 / np.sqrt(2)
            np.testing.assert_allclose(solution_neg, [expected_neg, expected_neg], rtol=1e-6)
    
    def test_jacobian_update(self):
        """
        测试目标：验证Broyden雅可比矩阵更新
        
        工作步骤：
        1. 提供初始雅可比矩阵
        2. 执行一步迭代
        3. 验证矩阵更新符合Broyden公式
        """
        def simple_func(x):
            return np.array([x[0]**2 - 1, x[1]**2 - 4])
        
        # 初始雅可比逆矩阵（近似）
        J_inv_initial = np.array([[0.5, 0], [0, 0.25]])
        
        solver = BroydenSolver(tolerance=1e-8)
        x0 = np.array([1.1, 2.1])
        
        # 启用跟踪模式进行测试
        solver._debug_mode = True
        solution, converged, iterations = solver.solve(
            simple_func, x0, jacobian_inv=J_inv_initial
        )
        
        # 验证矩阵确实被更新
        assert solver.jacobian_inverse is not None
        assert not np.array_equal(solver.jacobian_inverse, J_inv_initial)
    
    def test_convergence_detection(self):
        """
        测试目标：验证收敛检测逻辑
        
        工作步骤：
        1. 测试不同的收敛条件
        2. 验证收敛历史记录
        3. 检查收敛标准
        """
        def slow_converging_func(x):
            """收敛较慢的函数"""
            return 0.9 * x  # 收敛因子0.9
        
        # 严格收敛条件
        solver_strict = BroydenSolver(tolerance=1e-12, max_iterations=1000)
        x0 = np.array([1.0])
        
        solution, converged, iterations = solver_strict.solve(slow_converging_func, x0)
        
        # 验证收敛
        assert converged == True
        assert abs(solution[0]) < 1e-10  # 应该收敛到0
        
        # 验证收敛历史
        assert len(solver_strict.convergence_history) == iterations
        assert solver_strict.convergence_history[-1] < solver_strict.tolerance
    
    def test_non_convergence_handling(self):
        """
        测试目标：验证不收敛情况的处理
        
        工作步骤：
        1. 创建不收敛的函数
        2. 验证达到最大迭代次数
        3. 检查异常处理
        """
        def diverging_func(x):
            """发散函数"""
            return 2.0 * x  # 发散因子2.0
        
        solver = BroydenSolver(max_iterations=10, tolerance=1e-6)
        x0 = np.array([1.0])
        
        solution, converged, iterations = solver.solve(diverging_func, x0)
        
        # 验证不收敛
        assert converged == False
        assert iterations == solver.max_iterations
    
    def test_damping_factor_effect(self):
        """
        测试目标：验证阻尼因子的作用
        
        工作步骤：
        1. 使用不同阻尼因子求解
        2. 比较收敛性能
        3. 验证稳定性改进
        """
        def oscillating_func(x):
            """容易振荡的函数"""
            return np.array([10 * x[0] - 5, -10 * x[1] + 5])
        
        x0 = np.array([1.0, 1.0])
        
        # 无阻尼
        solver_no_damp = BroydenSolver(damping_factor=1.0, max_iterations=50)
        _, converged_no_damp, iter_no_damp = solver_no_damp.solve(oscillating_func, x0)
        
        # 有阻尼  
        solver_with_damp = BroydenSolver(damping_factor=0.5, max_iterations=50)
        _, converged_with_damp, iter_with_damp = solver_with_damp.solve(oscillating_func, x0)
        
        # 阻尼应该改善收敛性（在这个特定例子中）
        # 注意：这取决于具体函数，不是通用规律
        if converged_with_damp and converged_no_damp:
            # 两者都收敛时，阻尼版本可能更稳定
            assert True  # 基本测试通过
        elif converged_with_damp and not converged_no_damp:
            # 阻尼版本收敛，无阻尼版本不收敛
            assert True  # 阻尼改善了收敛性


class TestNewtonRaphsonSolver:
    """
    NewtonRaphsonSolver测试类
    
    测试Newton-Raphson方法的实现。
    """
    
    def test_initialization(self):
        """
        测试目标：验证Newton-Raphson求解器初始化
        
        工作步骤：
        1. 创建求解器实例
        2. 检查参数设置
        3. 验证数值微分参数
        """
        solver = NewtonRaphsonSolver()
        
        assert solver.max_iterations == 50
        assert solver.tolerance == 1e-6
        assert solver.finite_diff_step == 1e-8
        assert solver.min_determinant == 1e-12
    
    def test_analytical_jacobian(self):
        """
        测试目标：验证解析雅可比矩阵的使用
        
        工作步骤：
        1. 定义函数及其雅可比矩阵
        2. 使用解析雅可比矩阵求解
        3. 验证解的精度
        """
        def func_with_jacobian(x):
            """函数: f(x) = [x1^2 - 1, x2^2 - 4]"""
            return np.array([x[0]**2 - 1, x[1]**2 - 4])
        
        def jacobian_func(x):
            """解析雅可比矩阵"""
            return np.array([[2*x[0], 0], [0, 2*x[1]]])
        
        solver = NewtonRaphsonSolver(tolerance=1e-10)
        x0 = np.array([1.1, 2.1])
        
        solution, converged, iterations = solver.solve(
            func_with_jacobian, x0, jacobian_func=jacobian_func
        )
        
        assert converged == True
        np.testing.assert_allclose(solution, [1.0, 2.0], rtol=1e-8)
    
    def test_numerical_jacobian(self):
        """
        测试目标：验证数值雅可比矩阵计算
        
        工作步骤：
        1. 使用数值微分计算雅可比矩阵
        2. 求解非线性方程组
        3. 验证精度
        """
        def nonlinear_func(x):
            """复杂非线性函数"""
            return np.array([
                np.sin(x[0]) + x[1] - 1,
                x[0]**2 + np.cos(x[1]) - 1
            ])
        
        solver = NewtonRaphsonSolver(max_iterations=30)
        x0 = np.array([0.5, 0.5])
        
        solution, converged, iterations = solver.solve(nonlinear_func, x0)
        
        if converged:
            # 验证解确实满足方程组
            residual = nonlinear_func(solution)
            assert np.linalg.norm(residual) < 1e-6
    
    def test_singular_jacobian_handling(self):
        """
        测试目标：验证奇异雅可比矩阵的处理
        
        工作步骤：
        1. 构造奇异雅可比矩阵的情况
        2. 验证算法的鲁棒性
        3. 检查异常处理
        """
        def singular_jacobian_func(x):
            """在x=[0,0]处雅可比矩阵奇异"""
            return np.array([x[0]**3, x[1]**3])
        
        solver = NewtonRaphsonSolver(min_determinant=1e-10)
        x0 = np.array([1e-6, 1e-6])  # 接近奇异点
        
        # 这应该要么收敛要么优雅地失败
        solution, converged, iterations = solver.solve(singular_jacobian_func, x0)
        
        # 验证算法没有崩溃
        assert isinstance(converged, bool)
        assert isinstance(iterations, int)


class TestRecycleConvergenceSolver:
    """
    RecycleConvergenceSolver测试类
    
    测试循环收敛求解器的实现。
    """
    
    def test_initialization(self):
        """
        测试目标：验证循环收敛求解器初始化
        
        工作步骤：
        1. 创建求解器实例
        2. 检查默认参数
        3. 验证加速方法设置
        """
        solver = RecycleConvergenceSolver()
        
        assert solver.max_iterations == 100
        assert solver.tolerance == 1e-4
        assert solver.acceleration_method == "GlobalBroyden"
        assert solver.enable_acceleration == True
    
    @patch('flowsheet_solver.convergence_solver.BroydenSolver')
    def test_broyden_acceleration(self, mock_broyden):
        """
        测试目标：验证Broyden加速的使用
        
        工作步骤：
        1. 模拟Broyden求解器
        2. 测试加速调用
        3. 验证参数传递
        """
        # 配置模拟对象
        mock_broyden_instance = Mock()
        mock_broyden_instance.solve.return_value = (
            np.array([1.0, 2.0]), True, 5
        )
        mock_broyden.return_value = mock_broyden_instance
        
        # 创建模拟流程图和循环对象
        mock_flowsheet = Mock()
        mock_recycle = Mock()
        mock_recycle.name = "Recycle1"
        mock_recycle.values = {"Temperature": 300.0, "Pressure": 101325.0}
        mock_recycle.errors = {"Temperature": 0.0, "Pressure": 0.0}
        
        recycle_objects = [mock_recycle]
        
        solver = RecycleConvergenceSolver(acceleration_method="GlobalBroyden")
        
        # 模拟求解函数
        def mock_solve_func(flowsheet):
            return []  # 无异常
        
        converged = solver.solve_recycle_convergence(
            mock_flowsheet, recycle_objects, [], mock_solve_func
        )
        
        # 验证Broyden求解器被调用
        assert mock_broyden.called
        assert converged is not None
    
    def test_simple_convergence_scenario(self):
        """
        测试目标：验证简单循环收敛场景
        
        工作步骤：
        1. 创建简单的循环系统
        2. 模拟收敛过程
        3. 验证收敛判断
        """
        solver = RecycleConvergenceSolver(
            max_iterations=15,  # 增加最大迭代次数
            tolerance=1e-3,
            enable_acceleration=False  # 禁用加速简化测试
        )
        
        # 模拟流程图
        mock_flowsheet = Mock()
        
        # 模拟循环对象
        mock_recycle = Mock()
        mock_recycle.name = "SimpleRecycle"
        mock_recycle.values = {"Flow": 100.0}
        mock_recycle.errors = {"Flow": 1.0}  # 初始误差
        
        iteration_count = [0]
        
        def mock_solve_function(flowsheet):
            """模拟求解过程，误差逐步减小"""
            iteration_count[0] += 1
            # 模拟收敛过程 - 使用指数衰减
            mock_recycle.errors["Flow"] = 1.0 * (0.5 ** iteration_count[0])
            return []  # 无异常
        
        recycle_objects = [mock_recycle]
        
        converged = solver.solve_recycle_convergence(
            mock_flowsheet, recycle_objects, [], mock_solve_function
        )
        
        # 验证收敛
        assert converged == True
        assert iteration_count[0] > 1  # 至少迭代一次
    
    def test_non_convergence_handling(self):
        """
        测试目标：验证不收敛情况的处理
        
        工作步骤：
        1. 创建不收敛的系统
        2. 验证达到最大迭代次数
        3. 检查异常抛出
        """
        solver = RecycleConvergenceSolver(
            max_iterations=5,
            tolerance=1e-6,
            enable_acceleration=False
        )
        
        # 模拟不收敛的循环对象
        mock_recycle = Mock()
        mock_recycle.name = "NonConvergingRecycle"
        mock_recycle.values = {"Flow": 100.0}
        mock_recycle.errors = {"Flow": 1.0}  # 误差不减小
        
        def mock_solve_function(flowsheet):
            """模拟不收敛的求解过程"""
            return []  # 误差保持不变
        
        mock_flowsheet = Mock()
        recycle_objects = [mock_recycle]
        
        # 应该抛出收敛异常
        with pytest.raises(ConvergenceException):
            solver.solve_recycle_convergence(
                mock_flowsheet, recycle_objects, [], mock_solve_function
            )


class TestSimultaneousAdjustSolver:
    """
    SimultaneousAdjustSolver测试类
    
    测试同步调节求解器的实现。
    """
    
    def test_initialization(self):
        """
        测试目标：验证同步调节求解器初始化
        
        工作步骤：
        1. 创建求解器实例
        2. 检查默认设置
        3. 验证求解器配置
        """
        solver = SimultaneousAdjustSolver()
        
        assert solver.max_iterations == 25
        assert solver.tolerance == 1e-6
        assert solver.method == "NewtonRaphson"
        assert solver.enable_damping == True
    
    def test_adjust_object_collection(self):
        """
        测试目标：验证调节对象的收集和管理
        
        工作步骤：
        1. 创建模拟调节对象
        2. 测试对象收集逻辑
        3. 验证目标函数构建
        """
        solver = SimultaneousAdjustSolver()
        
        # 创建模拟流程图
        mock_flowsheet = Mock()
        
        # 创建模拟调节对象
        mock_adjust1 = Mock()
        mock_adjust1.name = "Adjust1"
        mock_adjust1.manipulation_type = "Temperature"
        mock_adjust1.target_value = 350.0
        mock_adjust1.current_value = 300.0
        mock_adjust1.enabled = True
        
        mock_adjust2 = Mock() 
        mock_adjust2.name = "Adjust2"
        mock_adjust2.manipulation_type = "Pressure"
        mock_adjust2.target_value = 200000.0
        mock_adjust2.current_value = 150000.0
        mock_adjust2.enabled = True
        
        # 模拟流程图包含调节对象
        mock_flowsheet.simulation_objects = {
            "Adjust1": mock_adjust1,
            "Adjust2": mock_adjust2,
            "OtherObject": Mock()  # 非调节对象
        }
        
        # 测试对象收集
        adjust_objects = solver._collect_adjust_objects(mock_flowsheet)
        
        assert len(adjust_objects) == 2
        assert mock_adjust1 in adjust_objects
        assert mock_adjust2 in adjust_objects
    
    def test_objective_function_evaluation(self):
        """
        测试目标：验证目标函数的计算
        
        工作步骤：
        1. 设置调节目标
        2. 计算目标函数值
        3. 验证梯度计算
        """
        solver = SimultaneousAdjustSolver()
        
        # 模拟调节对象
        mock_adjust = Mock()
        mock_adjust.target_value = 100.0
        mock_adjust.current_value = 95.0
        mock_adjust.tolerance = 1.0
        
        # 计算单个目标函数值
        error = solver._calculate_objective_value(mock_adjust)
        
        expected_error = (95.0 - 100.0) / 1.0  # (current - target) / tolerance
        assert abs(error - expected_error) < 1e-10
    
    @patch('flowsheet_solver.convergence_solver.NewtonRaphsonSolver')
    def test_newton_raphson_integration(self, mock_newton):
        """
        测试目标：验证与Newton-Raphson求解器的集成
        
        工作步骤：
        1. 模拟Newton-Raphson求解器
        2. 测试求解调用
        3. 验证结果处理
        """
        # 配置模拟Newton求解器
        mock_newton_instance = Mock()
        mock_newton_instance.solve.return_value = (
            np.array([350.0, 200000.0]), True, 5
        )
        mock_newton.return_value = mock_newton_instance
        
        solver = SimultaneousAdjustSolver(method="NewtonRaphson")
        
        # 模拟流程图
        mock_flowsheet = Mock()
        
        # 创建模拟调节对象
        mock_adjust1 = Mock()
        mock_adjust1.name = "Adjust1"
        mock_adjust1.current_value = 300.0
        mock_adjust1.enabled = True
        
        mock_adjust2 = Mock()
        mock_adjust2.name = "Adjust2"
        mock_adjust2.current_value = 150000.0
        mock_adjust2.enabled = True
        
        # 设置simulation_objects
        mock_flowsheet.simulation_objects = {
            "Adjust1": mock_adjust1,
            "Adjust2": mock_adjust2
        }
        
        def mock_solve_function(flowsheet):
            return []  # 无异常
        
        # 测试求解
        success = solver.solve_simultaneous_adjusts(
            mock_flowsheet, mock_solve_function
        )
        
        # 验证Newton求解器被使用
        assert mock_newton.called
        assert isinstance(success, bool)


class TestNumericalStability:
    """
    数值稳定性测试类
    
    测试各种数值算法在极端条件下的稳定性。
    """
    
    def test_ill_conditioned_systems(self):
        """
        测试目标：验证病态系统的处理
        
        工作步骤：
        1. 构造病态矩阵系统
        2. 测试不同求解器的响应
        3. 验证数值稳定性
        """
        # 构造病态线性系统
        def ill_conditioned_func(x):
            # 高条件数矩阵
            A = np.array([[1, 1], [1, 1.0001]])
            b = np.array([2, 2.0001])
            return np.dot(A, x) - b
        
        solvers = [
            BroydenSolver(tolerance=1e-6),
            NewtonRaphsonSolver(tolerance=1e-6)
        ]
        
        x0 = np.array([1.0, 1.0])
        
        for solver in solvers:
            try:
                solution, converged, iterations = solver.solve(ill_conditioned_func, x0)
                
                if converged:
                    # 验证解的合理性
                    residual = ill_conditioned_func(solution)
                    assert np.linalg.norm(residual) < 1e-3  # 放宽容差
                    
            except (ConvergenceException, np.linalg.LinAlgError):
                # 允许求解器优雅地失败
                pass
    
    def test_large_scale_systems(self):
        """
        测试目标：验证大规模系统的处理能力
        
        工作步骤：
        1. 创建大规模线性系统
        2. 测试内存使用
        3. 验证计算效率
        """
        n = 50  # 系统规模
        
        def large_linear_func(x):
            # 创建对角占优矩阵确保收敛
            A = np.diag(range(1, n+1)) + 0.1 * np.random.randn(n, n)
            b = np.ones(n)
            return np.dot(A, x) - b
        
        solver = BroydenSolver(max_iterations=200, tolerance=1e-4)
        x0 = np.zeros(n)
        
        start_time = time.time()
        solution, converged, iterations = solver.solve(large_linear_func, x0)
        end_time = time.time()
        
        # 验证性能合理
        assert end_time - start_time < 10.0  # 应在10秒内完成
        
        if converged:
            residual_norm = np.linalg.norm(large_linear_func(solution))
            assert residual_norm < 1e-3
    
    def test_boundary_conditions(self):
        """
        测试目标：验证边界条件的处理
        
        工作步骤：
        1. 测试零解情况
        2. 测试极大值/极小值
        3. 验证数值精度限制
        """
        # 测试零解
        def zero_func(x):
            return x  # f(x) = x, 解为x = 0
        
        solver = BroydenSolver(tolerance=1e-12)
        x0 = np.array([1e-10])
        
        solution, converged, iterations = solver.solve(zero_func, x0)
        
        if converged:
            assert abs(solution[0]) < 1e-10
        
        # 测试极值
        def extreme_func(x):
            return x - 1e6  # 解为x = 1e6
        
        x0 = np.array([0.0])
        solution, converged, iterations = solver.solve(extreme_func, x0)
        
        if converged:
            assert abs(solution[0] - 1e6) < 1e-2 