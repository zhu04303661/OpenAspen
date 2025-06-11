"""
Performance Benchmarks 性能基准测试
=================================

测试目标：
1. FlowsheetSolver性能基准测试
2. 收敛算法性能评估
3. 远程求解器性能测试
4. 内存使用情况监控
5. 可扩展性测试
6. 负载压力测试

工作步骤：
1. 设置性能测试环境
2. 定义基准测试用例
3. 执行性能测量
4. 分析性能指标
5. 生成性能报告
6. 建立性能回归检测
"""

import pytest
import time
import psutil
import os
import gc
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from flowsheet_solver.solver import (
    FlowsheetSolver,
    SolverSettings,
    SolverMode
)
from flowsheet_solver.convergence_solver import (
    BroydenSolver,
    NewtonRaphsonSolver
)
from flowsheet_solver.calculation_args import CalculationArgs, ObjectType


class PerformanceProfiler:
    """
    性能分析器
    
    用于测量和分析求解器性能指标。
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.start_cpu_percent = None
        
    def start_profiling(self):
        """开始性能分析"""
        gc.collect()  # 强制垃圾回收
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.start_cpu_percent = self.process.cpu_percent()
        
    def end_profiling(self):
        """结束性能分析并返回结果"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        end_cpu_percent = self.process.cpu_percent()
        
        return {
            'execution_time': end_time - self.start_time,
            'memory_usage': end_memory - self.start_memory,
            'peak_memory': end_memory,
            'cpu_usage': end_cpu_percent,
            'memory_mb': (end_memory - self.start_memory) / 1024 / 1024
        }


@pytest.mark.performance
class TestFlowsheetSolverPerformance:
    """
    FlowsheetSolver性能测试类
    
    测试主求解器在不同规模和配置下的性能表现。
    """
    
    @pytest.fixture
    def performance_profiler(self):
        """性能分析器fixture"""
        return PerformanceProfiler()
    
    @pytest.fixture(params=[10, 50, 100, 200])
    def scalable_flowsheet(self, request):
        """
        可扩展流程图fixture
        
        根据参数创建不同规模的流程图用于可扩展性测试。
        """
        num_objects = request.param
        
        flowsheet = Mock()
        flowsheet.name = f"ScalableFlowsheet_{num_objects}"
        
        objects = {}
        
        # 创建指定数量的设备对象
        for i in range(num_objects):
            obj = Mock()
            obj.name = f"Object{i:03d}"
            obj.graphic_object = Mock()
            obj.graphic_object.object_type = "Heater" if i % 2 == 0 else "Cooler"
            obj.graphic_object.calculated = False
            obj.graphic_object.active = True
            objects[obj.name] = obj
        
        flowsheet.simulation_objects = objects
        return flowsheet, num_objects
    
    def test_synchronous_solving_scalability(self, scalable_flowsheet, performance_profiler):
        """
        测试目标：验证同步求解的可扩展性
        
        工作步骤：
        1. 创建不同规模的流程图
        2. 测量求解时间和内存使用
        3. 分析可扩展性特征
        """
        flowsheet, num_objects = scalable_flowsheet
        
        settings = SolverSettings(
            enable_parallel_processing=False,
            max_iterations=100
        )
        solver = FlowsheetSolver(settings)
        
        # 模拟计算
        def mock_calculate(calc_args):
            time.sleep(0.001)  # 1ms每个对象
            calc_args.set_success(0.001, 1)
            return []
        
        # 性能测试
        performance_profiler.start_profiling()
        
        with patch.object(solver, '_calculate_object', side_effect=mock_calculate):
            exceptions = solver.solve_flowsheet(flowsheet, mode=SolverMode.SYNCHRONOUS)
        
        perf_results = performance_profiler.end_profiling()
        
        # 验证结果
        assert len(exceptions) == 0
        
        # 性能指标
        time_per_object = perf_results['execution_time'] / num_objects
        memory_per_object = perf_results['memory_mb'] / num_objects
        
        # 性能断言（可以根据实际需求调整）
        assert time_per_object < 0.01  # 每对象少于10ms
        assert memory_per_object < 1.0  # 每对象少于1MB
        
        # 记录性能指标
        print(f"\n同步求解性能 - {num_objects}个对象:")
        print(f"  总时间: {perf_results['execution_time']:.3f}s")
        print(f"  每对象时间: {time_per_object:.6f}s")
        print(f"  内存使用: {perf_results['memory_mb']:.2f}MB")
        print(f"  每对象内存: {memory_per_object:.4f}MB")
    
    def test_parallel_solving_performance(self, performance_profiler):
        """
        测试目标：验证并行求解的性能提升
        
        工作步骤：
        1. 比较串行vs并行求解性能
        2. 测量并行化效果
        3. 验证性能提升
        """
        # 创建适合并行处理的流程图
        num_objects = 50
        flowsheet = Mock()
        flowsheet.name = "ParallelTestFlowsheet"
        
        objects = {}
        for i in range(num_objects):
            obj = Mock()
            obj.name = f"ParallelObject{i:03d}"
            obj.graphic_object = Mock()
            obj.graphic_object.object_type = "Heater"
            obj.graphic_object.calculated = False
            obj.graphic_object.active = True
            objects[obj.name] = obj
        
        flowsheet.simulation_objects = objects
        
        def mock_cpu_intensive_calculate(calc_args):
            """模拟CPU密集型计算"""
            time.sleep(0.01)  # 10ms计算时间
            calc_args.set_success(0.01, 1)
            return []
        
        # 测试串行执行
        serial_settings = SolverSettings(enable_parallel_processing=False)
        serial_solver = FlowsheetSolver(serial_settings)
        
        performance_profiler.start_profiling()
        with patch.object(serial_solver, '_calculate_object', side_effect=mock_cpu_intensive_calculate):
            serial_solver.solve_flowsheet(flowsheet, mode=SolverMode.SYNCHRONOUS)
        serial_results = performance_profiler.end_profiling()
        
        # 测试并行执行
        parallel_settings = SolverSettings(
            enable_parallel_processing=True,
            max_thread_multiplier=4
        )
        parallel_solver = FlowsheetSolver(parallel_settings)
        
        performance_profiler.start_profiling()
        with patch.object(parallel_solver, '_calculate_object', side_effect=mock_cpu_intensive_calculate):
            parallel_solver.solve_flowsheet(flowsheet, mode=SolverMode.PARALLEL)
        parallel_results = performance_profiler.end_profiling()
        
        # 分析性能提升
        speedup = serial_results['execution_time'] / parallel_results['execution_time']
        efficiency = speedup / 4  # 4核并行效率
        
        # 验证并行性能提升
        assert speedup > 1.5  # 至少1.5倍加速
        assert efficiency > 0.3  # 至少30%效率
        
        print(f"\n并行求解性能比较:")
        print(f"  串行时间: {serial_results['execution_time']:.3f}s")
        print(f"  并行时间: {parallel_results['execution_time']:.3f}s")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  并行效率: {efficiency:.2%}")
    
    def test_memory_efficiency(self, performance_profiler):
        """
        测试目标：验证内存使用效率
        
        工作步骤：
        1. 测试大规模求解的内存使用
        2. 检查内存泄漏
        3. 验证内存回收
        """
        # 创建大规模流程图
        large_flowsheet = Mock()
        large_flowsheet.name = "LargeMemoryTestFlowsheet"
        
        num_objects = 500
        objects = {}
        
        for i in range(num_objects):
            obj = Mock()
            obj.name = f"LargeObject{i:04d}"
            obj.graphic_object = Mock()
            obj.graphic_object.object_type = "ComplexUnit"
            obj.graphic_object.calculated = False
            obj.graphic_object.active = True
            # 模拟对象包含大量数据
            obj.large_data = ["data"] * 100  # 100个元素
            objects[obj.name] = obj
        
        large_flowsheet.simulation_objects = objects
        
        # 记录初始内存
        initial_memory = performance_profiler.process.memory_info().rss
        
        solver = FlowsheetSolver()
        
        def mock_memory_intensive_calculate(calc_args):
            # 模拟创建临时大数据
            temp_data = ["temp"] * 50
            time.sleep(0.002)
            calc_args.set_success(0.002, 1)
            del temp_data  # 显式删除临时数据
            return []
        
        # 执行内存测试
        performance_profiler.start_profiling()
        
        with patch.object(solver, '_calculate_object', side_effect=mock_memory_intensive_calculate):
            exceptions = solver.solve_flowsheet(large_flowsheet)
        
        peak_memory = performance_profiler.process.memory_info().rss
        
        # 强制垃圾回收
        del solver
        del large_flowsheet
        gc.collect()
        
        final_memory = performance_profiler.process.memory_info().rss
        
        # 内存分析
        peak_usage_mb = (peak_memory - initial_memory) / 1024 / 1024
        leaked_mb = (final_memory - initial_memory) / 1024 / 1024
        memory_per_object = peak_usage_mb / num_objects
        
        # 内存效率断言
        assert peak_usage_mb < 200  # 峰值内存少于200MB
        assert leaked_mb < 10  # 内存泄漏少于10MB
        assert memory_per_object < 0.5  # 每对象内存少于0.5MB
        
        print(f"\n内存效率测试 - {num_objects}个对象:")
        print(f"  峰值内存使用: {peak_usage_mb:.2f}MB")
        print(f"  每对象内存: {memory_per_object:.4f}MB")
        print(f"  内存泄漏: {leaked_mb:.2f}MB")


@pytest.mark.performance
class TestConvergenceSolverPerformance:
    """
    收敛求解器性能测试类
    
    测试各种收敛算法的性能表现。
    """
    
    def test_broyden_solver_performance(self, performance_profiler):
        """
        测试目标：验证Broyden求解器性能
        
        工作步骤：
        1. 测试不同规模的方程组求解
        2. 测量收敛速度和计算时间
        3. 分析性能特征
        """
        import numpy as np
        
        # 测试不同规模的问题
        problem_sizes = [5, 10, 20, 50]
        performance_results = []
        
        for n in problem_sizes:
            # 创建n维线性方程组
            def create_linear_problem(size):
                A = np.random.randn(size, size) + size * np.eye(size)  # 对角占优确保收敛
                b = np.random.randn(size)
                
                def func(x):
                    return np.dot(A, x) - b
                
                return func, np.linalg.solve(A, b)  # 解析解
            
            problem_func, analytical_solution = create_linear_problem(n)
            
            # 性能测试
            solver = BroydenSolver(tolerance=1e-8, max_iterations=100)
            x0 = np.zeros(n)
            
            performance_profiler.start_profiling()
            solution, converged, iterations = solver.solve(problem_func, x0)
            perf_results = performance_profiler.end_profiling()
            
            # 验证精度
            if converged:
                error = np.linalg.norm(solution - analytical_solution)
                assert error < 1e-6
            
            performance_results.append({
                'size': n,
                'converged': converged,
                'iterations': iterations,
                'time': perf_results['execution_time'],
                'time_per_iteration': perf_results['execution_time'] / iterations if iterations > 0 else 0
            })
        
        # 分析性能趋势
        for result in performance_results:
            print(f"\nBroyden求解器性能 - {result['size']}维问题:")
            print(f"  收敛: {result['converged']}")
            print(f"  迭代次数: {result['iterations']}")
            print(f"  总时间: {result['time']:.6f}s")
            print(f"  每迭代时间: {result['time_per_iteration']:.6f}s")
            
            # 性能断言
            if result['converged']:
                assert result['time'] < 1.0  # 1秒内完成
                assert result['iterations'] < 50  # 少于50次迭代
    
    def test_newton_raphson_performance(self, performance_profiler):
        """
        测试目标：验证Newton-Raphson求解器性能
        
        工作步骤：
        1. 测试非线性方程组求解性能
        2. 比较数值微分vs解析雅可比的性能
        3. 分析收敛特性
        """
        import numpy as np
        
        # 定义非线性问题
        def nonlinear_system(x):
            """非线性方程组: Rosenbrock函数的梯度 = 0"""
            return np.array([
                -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                200 * (x[1] - x[0]**2)
            ])
        
        def jacobian_analytical(x):
            """解析雅可比矩阵"""
            return np.array([
                [-400 * x[1] + 1200 * x[0]**2 + 2, -400 * x[0]],
                [-400 * x[0], 200]
            ])
        
        x0 = np.array([-1.0, 1.0])
        
        # 测试数值微分版本
        solver_numerical = NewtonRaphsonSolver(tolerance=1e-8, max_iterations=100)
        
        performance_profiler.start_profiling()
        solution_num, converged_num, iter_num = solver_numerical.solve(nonlinear_system, x0)
        perf_numerical = performance_profiler.end_profiling()
        
        # 测试解析雅可比版本
        solver_analytical = NewtonRaphsonSolver(tolerance=1e-8, max_iterations=100)
        
        performance_profiler.start_profiling()
        solution_ana, converged_ana, iter_ana = solver_analytical.solve(
            nonlinear_system, x0, jacobian_func=jacobian_analytical
        )
        perf_analytical = performance_profiler.end_profiling()
        
        # 性能比较
        if converged_num and converged_ana:
            speedup = perf_numerical['execution_time'] / perf_analytical['execution_time']
            
            print(f"\nNewton-Raphson性能比较:")
            print(f"  数值微分 - 时间: {perf_numerical['execution_time']:.6f}s, 迭代: {iter_num}")
            print(f"  解析雅可比 - 时间: {perf_analytical['execution_time']:.6f}s, 迭代: {iter_ana}")
            print(f"  解析雅可比加速比: {speedup:.2f}x")
            
            # 验证解析雅可比更快
            assert speedup > 1.5  # 至少1.5倍加速
            
            # 验证解的一致性
            solution_error = np.linalg.norm(solution_num - solution_ana)
            assert solution_error < 1e-6


@pytest.mark.performance
@pytest.mark.slow
class TestStressAndLoadTesting:
    """
    压力和负载测试类
    
    测试系统在极端条件下的性能和稳定性。
    """
    
    def test_high_concurrency_stress(self):
        """
        测试目标：验证高并发情况下的性能
        
        工作步骤：
        1. 创建大量并发求解任务
        2. 测试系统稳定性
        3. 验证性能不降级
        """
        def create_concurrent_task(task_id):
            """创建并发任务"""
            # 创建小规模流程图
            flowsheet = Mock()
            flowsheet.name = f"ConcurrentTask_{task_id}"
            
            objects = {}
            for i in range(20):  # 20个对象
                obj = Mock()
                obj.name = f"T{task_id}_Obj{i:02d}"
                obj.graphic_object = Mock()
                obj.graphic_object.object_type = "SimpleUnit"
                obj.graphic_object.calculated = False
                obj.graphic_object.active = True
                objects[obj.name] = obj
            
            flowsheet.simulation_objects = objects
            
            # 创建求解器
            settings = SolverSettings(
                enable_parallel_processing=True,
                max_thread_multiplier=2
            )
            solver = FlowsheetSolver(settings)
            
            def mock_calculate(calc_args):
                time.sleep(0.005)  # 5ms计算
                calc_args.set_success(0.005, 1)
                return []
            
            # 执行求解
            start_time = time.time()
            with patch.object(solver, '_calculate_object', side_effect=mock_calculate):
                exceptions = solver.solve_flowsheet(flowsheet)
            end_time = time.time()
            
            return {
                'task_id': task_id,
                'time': end_time - start_time,
                'exceptions': len(exceptions),
                'success': len(exceptions) == 0
            }
        
        # 高并发测试
        num_concurrent_tasks = 20
        max_workers = 10
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(create_concurrent_task, i)
                for i in range(num_concurrent_tasks)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 分析结果
        successful_tasks = [r for r in results if r['success']]
        task_times = [r['time'] for r in results]
        
        # 验证稳定性
        assert len(successful_tasks) == num_concurrent_tasks  # 所有任务成功
        assert total_time < 30  # 30秒内完成
        
        # 分析性能统计
        avg_task_time = statistics.mean(task_times)
        max_task_time = max(task_times)
        min_task_time = min(task_times)
        std_dev = statistics.stdev(task_times) if len(task_times) > 1 else 0
        
        print(f"\n高并发压力测试结果:")
        print(f"  并发任务数: {num_concurrent_tasks}")
        print(f"  工作线程数: {max_workers}")
        print(f"  总执行时间: {total_time:.2f}s")
        print(f"  成功任务数: {len(successful_tasks)}")
        print(f"  平均任务时间: {avg_task_time:.3f}s")
        print(f"  最大任务时间: {max_task_time:.3f}s")
        print(f"  最小任务时间: {min_task_time:.3f}s")
        print(f"  时间标准差: {std_dev:.3f}s")
        
        # 性能一致性验证
        assert std_dev < 0.5  # 时间变异不超过0.5秒
    
    def test_memory_pressure_endurance(self):
        """
        测试目标：验证内存压力下的持续性能
        
        工作步骤：
        1. 长时间运行大内存消耗任务
        2. 监控内存使用趋势
        3. 验证无内存泄漏
        """
        process = psutil.Process(os.getpid())
        memory_samples = []
        
        # 创建内存密集型流程图
        def create_memory_intensive_flowsheet():
            flowsheet = Mock()
            flowsheet.name = "MemoryIntensiveFlowsheet"
            
            objects = {}
            for i in range(100):  # 100个对象
                obj = Mock()
                obj.name = f"MemObj{i:03d}"
                obj.graphic_object = Mock()
                obj.graphic_object.object_type = "MemoryIntensiveUnit"
                obj.graphic_object.calculated = False
                obj.graphic_object.active = True
                # 每个对象包含大量数据
                obj.large_dataset = [f"data_{j}" for j in range(1000)]
                objects[obj.name] = obj
            
            flowsheet.simulation_objects = objects
            return flowsheet
        
        def mock_memory_heavy_calculate(calc_args):
            # 模拟内存密集型计算
            temp_large_data = [[i] * 100 for i in range(50)]
            time.sleep(0.01)
            calc_args.set_success(0.01, 1)
            del temp_large_data
            return []
        
        # 持续运行测试
        num_iterations = 10
        solver = FlowsheetSolver()
        
        for iteration in range(num_iterations):
            print(f"内存压力测试 - 迭代 {iteration + 1}/{num_iterations}")
            
            # 记录内存使用
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            flowsheet = create_memory_intensive_flowsheet()
            
            with patch.object(solver, '_calculate_object', side_effect=mock_memory_heavy_calculate):
                exceptions = solver.solve_flowsheet(flowsheet)
            
            # 清理
            del flowsheet
            gc.collect()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append({
                'iteration': iteration,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_diff': memory_after - memory_before
            })
            
            # 验证无严重内存泄漏
            assert len(exceptions) == 0
            
            time.sleep(0.1)  # 短暂休息
        
        # 分析内存趋势
        memory_diffs = [sample['memory_diff'] for sample in memory_samples]
        total_memory_increase = memory_samples[-1]['memory_after'] - memory_samples[0]['memory_before']
        avg_memory_diff = statistics.mean(memory_diffs)
        
        print(f"\n内存压力持续测试结果:")
        print(f"  测试迭代数: {num_iterations}")
        print(f"  总内存增长: {total_memory_increase:.2f}MB")
        print(f"  平均每次内存变化: {avg_memory_diff:.2f}MB")
        print(f"  最大单次内存增长: {max(memory_diffs):.2f}MB")
        
        # 内存稳定性验证
        assert total_memory_increase < 50  # 总增长少于50MB
        assert avg_memory_diff < 5  # 平均增长少于5MB
    
    def test_computational_complexity_scaling(self):
        """
        测试目标：验证计算复杂度的扩展性
        
        工作步骤：
        1. 测试不同规模问题的求解时间
        2. 分析时间复杂度
        3. 验证算法效率
        """
        # 测试不同规模
        problem_sizes = [10, 20, 50, 100, 200]
        performance_data = []
        
        for size in problem_sizes:
            print(f"复杂度测试 - {size}个对象")
            
            # 创建指定规模的流程图
            flowsheet = Mock()
            flowsheet.name = f"ComplexityTest_{size}"
            
            objects = {}
            for i in range(size):
                obj = Mock()
                obj.name = f"ComplexObj{i:03d}"
                obj.graphic_object = Mock()
                obj.graphic_object.object_type = "StandardUnit"
                obj.graphic_object.calculated = False
                obj.graphic_object.active = True
                objects[obj.name] = obj
            
            flowsheet.simulation_objects = objects
            
            # 性能测试
            solver = FlowsheetSolver()
            
            def mock_standard_calculate(calc_args):
                # 模拟标准计算时间（与规模无关）
                time.sleep(0.002)  # 2ms
                calc_args.set_success(0.002, 1)
                return []
            
            start_time = time.time()
            with patch.object(solver, '_calculate_object', side_effect=mock_standard_calculate):
                exceptions = solver.solve_flowsheet(flowsheet)
            end_time = time.time()
            
            execution_time = end_time - start_time
            time_per_object = execution_time / size
            
            performance_data.append({
                'size': size,
                'time': execution_time,
                'time_per_object': time_per_object,
                'success': len(exceptions) == 0
            })
            
            # 验证成功
            assert len(exceptions) == 0
        
        # 分析复杂度
        sizes = [d['size'] for d in performance_data]
        times = [d['time'] for d in performance_data]
        times_per_object = [d['time_per_object'] for d in performance_data]
        
        print(f"\n计算复杂度分析:")
        for data in performance_data:
            print(f"  规模 {data['size']:3d}: 总时间 {data['time']:.3f}s, "
                  f"每对象时间 {data['time_per_object']:.6f}s")
        
        # 验证线性复杂度（时间与对象数量成正比）
        # 允许一定的变异，但应该保持大致线性关系
        time_per_object_variation = max(times_per_object) / min(times_per_object)
        assert time_per_object_variation < 3.0  # 变异不超过3倍
        
        # 验证可接受的性能
        assert max(times_per_object) < 0.01  # 每对象少于10ms 