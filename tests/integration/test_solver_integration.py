"""
Solver Integration 求解器集成测试
===============================

测试目标：
1. FlowsheetSolver与收敛求解器的集成
2. 远程求解器的完整工作流程
3. 复杂流程图的端到端求解
4. 异常处理的集成测试
5. 性能和可靠性验证
6. 多组件协同工作验证

工作步骤：
1. 设置完整的测试环境
2. 创建复杂的测试场景
3. 验证组件间交互
4. 测试端到端工作流程
5. 验证错误处理和恢复
6. 性能和稳定性测试
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from flowsheet_solver.solver import (
    FlowsheetSolver,
    SolverSettings,
    SolverMode
)
from flowsheet_solver.convergence_solver import (
    BroydenSolver,
    RecycleConvergenceSolver,
    SimultaneousAdjustSolver
)
from flowsheet_solver.remote_solvers import (
    AzureSolverClient,
    TCPSolverClient
)
from flowsheet_solver.calculation_args import (
    CalculationArgs,
    ObjectType
)
from flowsheet_solver.solver_exceptions import (
    SolverException,
    ConvergenceException,
    NetworkException,
    CalculationException
)


class TestFlowsheetSolverConvergenceIntegration:
    """
    FlowsheetSolver与收敛求解器集成测试
    
    测试主求解器与各种收敛算法的集成。
    """
    
    @pytest.fixture
    def complex_flowsheet_with_recycles(self):
        """
        创建包含循环的复杂流程图fixture
        
        返回一个模拟的复杂流程图，包含：
        - 多个设备对象
        - 循环流股
        - 调节对象
        - 复杂的依赖关系
        """
        flowsheet = Mock()
        flowsheet.name = "ComplexFlowsheetWithRecycles"
        
        # 创建设备对象
        objects = {}
        
        # Feed stream
        feed = Mock()
        feed.name = "Feed"
        feed.graphic_object = Mock()
        feed.graphic_object.object_type = "MaterialStream"
        feed.graphic_object.calculated = True
        feed.graphic_object.active = True
        objects["Feed"] = feed
        
        # Reactor
        reactor = Mock()
        reactor.name = "Reactor"
        reactor.graphic_object = Mock()
        reactor.graphic_object.object_type = "Reactor"
        reactor.graphic_object.calculated = False
        reactor.graphic_object.active = True
        objects["Reactor"] = reactor
        
        # Separator
        separator = Mock()
        separator.name = "Separator"
        separator.graphic_object = Mock()
        separator.graphic_object.object_type = "Separator"
        separator.graphic_object.calculated = False
        separator.graphic_object.active = True
        objects["Separator"] = separator
        
        # Recycle stream
        recycle = Mock()
        recycle.name = "Recycle"
        recycle.graphic_object = Mock()
        recycle.graphic_object.object_type = "Recycle"
        recycle.graphic_object.calculated = False
        recycle.graphic_object.active = True
        recycle.values = {"Flow": 100.0, "Temperature": 300.0}
        recycle.errors = {"Flow": 1.0, "Temperature": 5.0}
        objects["Recycle"] = recycle
        
        # Adjust object for temperature control
        temp_adjust = Mock()
        temp_adjust.name = "TempAdjust"
        temp_adjust.graphic_object = Mock()
        temp_adjust.graphic_object.object_type = "Adjust"
        temp_adjust.graphic_object.calculated = False
        temp_adjust.graphic_object.active = True
        temp_adjust.target_value = 350.0
        temp_adjust.current_value = 300.0
        temp_adjust.tolerance = 1.0
        temp_adjust.enabled = True
        objects["TempAdjust"] = temp_adjust
        
        flowsheet.simulation_objects = objects
        return flowsheet
    
    def test_recycle_convergence_integration(self, complex_flowsheet_with_recycles):
        """
        测试目标：验证循环收敛求解的完整集成
        
        工作步骤：
        1. 设置FlowsheetSolver和RecycleConvergenceSolver
        2. 执行包含循环的流程图求解
        3. 验证收敛算法被正确调用
        4. 检查收敛结果
        """
        # 创建求解器
        settings = SolverSettings(
            max_iterations=50,
            tolerance=1e-4,
            enable_parallel_processing=False
        )
        main_solver = FlowsheetSolver(settings)
        
        # 模拟循环收敛过程
        convergence_iterations = [0]
        
        def mock_calculate_object(calc_args):
            """模拟对象计算，包括收敛逻辑"""
            time.sleep(0.01)  # 模拟计算时间
            
            if calc_args.object_type == ObjectType.Recycle:
                # 模拟循环收敛
                convergence_iterations[0] += 1
                
                # 模拟收敛过程（误差逐渐减小）
                if convergence_iterations[0] < 5:
                    # 未收敛
                    calc_args.flowsheet.simulation_objects["Recycle"].errors = {
                        "Flow": 1.0 / convergence_iterations[0],
                        "Temperature": 5.0 / convergence_iterations[0]
                    }
                    calc_args.set_success(0.01, convergence_iterations[0])
                else:
                    # 收敛
                    calc_args.flowsheet.simulation_objects["Recycle"].errors = {
                        "Flow": 1e-5,
                        "Temperature": 1e-5
                    }
                    calc_args.set_success(0.01, convergence_iterations[0])
            else:
                # 其他对象正常计算
                calc_args.set_success(0.01, 1)
            
            return []
        
        # 集成测试
        with patch.object(main_solver, '_calculate_object', side_effect=mock_calculate_object):
            exceptions = main_solver.solve_flowsheet(complex_flowsheet_with_recycles)
        
        # 验证结果
        assert len(exceptions) == 0  # 无异常
        assert convergence_iterations[0] >= 5  # 进行了收敛迭代
        
        # 验证性能统计
        stats = main_solver.performance_stats
        assert stats['successful_objects'] > 0
        assert stats['total_time'] > 0
    
    def test_simultaneous_adjust_integration(self, complex_flowsheet_with_recycles):
        """
        测试目标：验证同步调节求解的集成
        
        工作步骤：
        1. 设置包含调节对象的求解
        2. 验证同步调节算法集成
        3. 检查调节收敛过程
        """
        settings = SolverSettings(tolerance=1e-6)
        main_solver = FlowsheetSolver(settings)
        
        adjust_iterations = [0]
        
        def mock_calculate_with_adjust(calc_args):
            """模拟包含调节的计算"""
            if calc_args.object_type == ObjectType.Adjust:
                adjust_iterations[0] += 1
                
                # 模拟调节收敛
                current_error = abs(calc_args.flowsheet.simulation_objects["TempAdjust"].current_value - 
                                  calc_args.flowsheet.simulation_objects["TempAdjust"].target_value)
                
                if current_error > 1e-3:
                    # 调整current_value向target_value收敛
                    current = calc_args.flowsheet.simulation_objects["TempAdjust"].current_value
                    target = calc_args.flowsheet.simulation_objects["TempAdjust"].target_value
                    calc_args.flowsheet.simulation_objects["TempAdjust"].current_value = current + 0.1 * (target - current)
                
                calc_args.set_success(0.01, adjust_iterations[0])
            else:
                calc_args.set_success(0.01, 1)
            
            return []
        
        with patch.object(main_solver, '_calculate_object', side_effect=mock_calculate_with_adjust):
            exceptions = main_solver.solve_flowsheet(complex_flowsheet_with_recycles)
        
        # 验证调节过程
        assert len(exceptions) == 0
        assert adjust_iterations[0] > 0
        
        # 验证最终调节值接近目标值
        final_value = complex_flowsheet_with_recycles.simulation_objects["TempAdjust"].current_value
        target_value = complex_flowsheet_with_recycles.simulation_objects["TempAdjust"].target_value
        assert abs(final_value - target_value) < 10  # 允许一定误差


class TestRemoteSolverIntegration:
    """
    远程求解器集成测试
    
    测试远程求解功能的完整集成。
    """
    
    @pytest.fixture
    def mock_azure_environment(self):
        """创建模拟Azure环境"""
        with patch('flowsheet_solver.remote_solvers.azure_solver_client.ServiceBusClient') as mock_servicebus:
            mock_client = Mock()
            mock_sender = Mock()
            mock_receiver = Mock()
            
            mock_client.get_queue_sender.return_value = mock_sender
            mock_client.get_queue_receiver.return_value = mock_receiver
            mock_servicebus.from_connection_string.return_value = mock_client
            
            yield {
                'servicebus': mock_servicebus,
                'client': mock_client,
                'sender': mock_sender,
                'receiver': mock_receiver
            }
    
    @pytest.fixture
    def mock_tcp_environment(self):
        """创建模拟TCP环境"""
        with patch('socket.socket') as mock_socket_class:
            mock_socket = Mock()
            mock_socket_class.return_value = mock_socket
            
            yield {
                'socket_class': mock_socket_class,
                'socket': mock_socket
            }
    
    def test_azure_solver_end_to_end(self, mock_azure_environment, mock_flowsheet):
        """
        测试目标：验证Azure求解器端到端工作流程
        
        工作步骤：
        1. 建立Azure连接
        2. 发送求解请求
        3. 接收和处理响应
        4. 验证完整流程
        """
        azure_env = mock_azure_environment
        
        # 设置响应模拟
        import json
        response_data = {
            "request_id": "test-azure-123",
            "status": "success",
            "result": {
                "exceptions": [],
                "calculation_time": 2.5,
                "solved_objects": ["Heater1", "Stream2"]
            }
        }
        
        mock_message = Mock()
        mock_message.body = json.dumps(response_data)
        azure_env['receiver'].receive_messages.return_value = [mock_message]
        
        # 创建Azure客户端
        azure_client = AzureSolverClient(
            connection_string="test_connection_string",
            queue_name="test-queue",
            timeout_seconds=30
        )
        
        # 执行端到端测试
        azure_client.connect()
        
        # 验证连接建立
        assert azure_env['servicebus'].from_connection_string.called
        assert azure_client.is_connected == True
        
        # 执行求解
        with patch.object(azure_client, 'send_solve_request', return_value="test-azure-123"):
            exceptions = azure_client.solve_flowsheet(mock_flowsheet)
        
        # 验证结果
        assert exceptions == []
        assert azure_env['sender'].send_message.called
        assert azure_env['receiver'].receive_messages.called
    
    def test_tcp_solver_end_to_end(self, mock_tcp_environment, mock_flowsheet):
        """
        测试目标：验证TCP求解器端到端工作流程
        
        工作步骤：
        1. 建立TCP连接
        2. 发送求解数据
        3. 接收响应数据
        4. 验证完整通信流程
        """
        tcp_env = mock_tcp_environment
        
        # 设置TCP响应模拟
        import json
        response_data = {
            "status": "success",
            "result": {
                "exceptions": [],
                "calculation_time": 1.8
            }
        }
        response_json = json.dumps(response_data)
        response_bytes = response_json.encode('utf-8')
        
        # 模拟TCP接收过程
        tcp_env['socket'].recv.side_effect = [
            len(response_bytes).to_bytes(4, 'big'),  # 长度头
            response_bytes  # 数据
        ]
        
        # 创建TCP客户端
        tcp_client = TCPSolverClient("localhost", 8080, timeout_seconds=30)
        
        # 执行端到端测试
        tcp_client.connect()
        
        # 验证连接
        assert tcp_env['socket_class'].called
        assert tcp_client.is_connected == True
        
        # 执行求解
        exceptions = tcp_client.solve_flowsheet(mock_flowsheet)
        
        # 验证结果
        assert exceptions == []
        assert tcp_env['socket'].sendall.called
        assert tcp_env['socket'].recv.called
    
    def test_remote_solver_failover(self, mock_azure_environment, mock_tcp_environment, mock_flowsheet):
        """
        测试目标：验证远程求解器故障转移
        
        工作步骤：
        1. 主求解器（Azure）失败
        2. 自动切换到备用求解器（TCP）
        3. 验证无缝切换
        """
        # 设置Azure失败
        azure_env = mock_azure_environment
        azure_env['servicebus'].from_connection_string.side_effect = Exception("Azure connection failed")
        
        # 设置TCP成功
        tcp_env = mock_tcp_environment
        import json
        success_response = json.dumps({"status": "success", "result": {"exceptions": []}})
        tcp_env['socket'].recv.side_effect = [
            len(success_response.encode()).to_bytes(4, 'big'),
            success_response.encode()
        ]
        
        # 创建求解器实例
        azure_client = AzureSolverClient("test_connection_string")
        tcp_client = TCPSolverClient("localhost", 8080)
        
        # 实现故障转移逻辑
        def solve_with_failover(flowsheet):
            try:
                azure_client.connect()
                return azure_client.solve_flowsheet(flowsheet)
            except Exception:
                # Azure失败，切换到TCP
                tcp_client.connect()
                return tcp_client.solve_flowsheet(flowsheet)
        
        # 执行故障转移测试
        result = solve_with_failover(mock_flowsheet)
        
        # 验证结果
        assert result == []
        assert tcp_client.is_connected == True


class TestComplexScenarioIntegration:
    """
    复杂场景集成测试
    
    测试系统在复杂、真实场景下的完整表现。
    """
    
    @pytest.fixture
    def industrial_scale_flowsheet(self):
        """
        创建工业规模的流程图fixture
        
        模拟一个大型化工流程，包含：
        - 50+个设备对象
        - 多个循环
        - 复杂的依赖关系
        - 各种类型的单元操作
        """
        flowsheet = Mock()
        flowsheet.name = "IndustrialScaleFlowsheet"
        
        objects = {}
        
        # 创建多个流股
        for i in range(20):
            stream = Mock()
            stream.name = f"Stream{i:02d}"
            stream.graphic_object = Mock()
            stream.graphic_object.object_type = "MaterialStream"
            stream.graphic_object.calculated = (i < 5)  # 前5个已计算
            stream.graphic_object.active = True
            objects[stream.name] = stream
        
        # 创建多个设备
        equipment_types = ["Heater", "Cooler", "Mixer", "Splitter", "Reactor", "Column"]
        for i in range(30):
            equipment = Mock()
            equipment.name = f"Equipment{i:02d}"
            equipment.graphic_object = Mock()
            equipment.graphic_object.object_type = equipment_types[i % len(equipment_types)]
            equipment.graphic_object.calculated = False
            equipment.graphic_object.active = True
            objects[equipment.name] = equipment
        
        # 创建多个循环对象
        for i in range(5):
            recycle = Mock()
            recycle.name = f"Recycle{i:02d}"
            recycle.graphic_object = Mock()
            recycle.graphic_object.object_type = "Recycle"
            recycle.graphic_object.calculated = False
            recycle.graphic_object.active = True
            recycle.values = {"Flow": 100.0 + i * 10}
            recycle.errors = {"Flow": 1.0}
            objects[recycle.name] = recycle
        
        flowsheet.simulation_objects = objects
        return flowsheet
    
    def test_large_scale_solving_performance(self, industrial_scale_flowsheet):
        """
        测试目标：验证大规模流程图的求解性能
        
        工作步骤：
        1. 设置高性能求解配置
        2. 执行大规模流程图求解
        3. 验证性能指标
        4. 检查内存使用和时间消耗
        """
        # 配置高性能设置
        settings = SolverSettings(
            max_iterations=200,
            tolerance=1e-4,
            timeout_seconds=600,
            enable_parallel_processing=True,
            max_thread_multiplier=4
        )
        
        solver = FlowsheetSolver(settings)
        
        # 模拟快速计算
        calculation_times = []
        
        def mock_fast_calculate(calc_args):
            start_time = time.time()
            time.sleep(0.001)  # 1ms计算时间
            end_time = time.time()
            
            calculation_times.append(end_time - start_time)
            calc_args.set_success(end_time - start_time, 1)
            return []
        
        # 执行大规模求解
        start_time = time.time()
        
        with patch.object(solver, '_calculate_object', side_effect=mock_fast_calculate):
            exceptions = solver.solve_flowsheet(
                industrial_scale_flowsheet,
                mode=SolverMode.PARALLEL
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能
        assert len(exceptions) == 0  # 无异常
        assert total_time < 60  # 应在1分钟内完成
        
        # 验证统计数据
        stats = solver.performance_stats
        assert stats['total_objects'] > 50  # 处理了大量对象
        assert stats['successful_objects'] > 50
        assert stats['average_time_per_object'] < 0.1  # 平均每对象少于100ms
        
        print(f"Solved {stats['total_objects']} objects in {total_time:.2f}s")
        print(f"Average time per object: {stats['average_time_per_object']:.4f}s")
    
    def test_concurrent_solving_scenarios(self, industrial_scale_flowsheet):
        """
        测试目标：验证并发求解场景
        
        工作步骤：
        1. 创建多个求解器实例
        2. 并发执行多个求解任务
        3. 验证并发安全性和性能
        """
        def create_solver_task(flowsheet, task_id):
            """创建求解任务"""
            settings = SolverSettings(
                enable_parallel_processing=True,
                max_thread_multiplier=2
            )
            solver = FlowsheetSolver(settings)
            
            def mock_calculate(calc_args):
                time.sleep(0.01)  # 10ms
                calc_args.set_success(0.01, 1)
                return []
            
            with patch.object(solver, '_calculate_object', side_effect=mock_calculate):
                start_time = time.time()
                exceptions = solver.solve_flowsheet(flowsheet)
                end_time = time.time()
                
                return {
                    'task_id': task_id,
                    'exceptions': exceptions,
                    'time': end_time - start_time,
                    'stats': solver.performance_stats
                }
        
        # 创建多个并发任务
        num_tasks = 3
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = [
                executor.submit(create_solver_task, industrial_scale_flowsheet, i)
                for i in range(num_tasks)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证并发结果
        assert len(results) == num_tasks
        assert all(len(result['exceptions']) == 0 for result in results)
        assert all(result['stats']['total_objects'] > 0 for result in results)
        
        # 验证并发性能（应该比串行快）
        serial_time = sum(result['time'] for result in results)
        assert total_time < serial_time * 0.8  # 至少20%的性能提升
        
        print(f"Concurrent solving: {total_time:.2f}s vs Serial: {serial_time:.2f}s")
    
    def test_error_recovery_integration(self, industrial_scale_flowsheet):
        """
        测试目标：验证复杂场景下的错误恢复
        
        工作步骤：
        1. 模拟多种类型的计算错误
        2. 验证错误处理和恢复机制
        3. 检查系统稳定性
        """
        settings = SolverSettings(
            solver_break_on_exception=False,  # 不中断继续计算
            max_iterations=100
        )
        solver = FlowsheetSolver(settings)
        
        failed_objects = []
        calculation_count = [0]
        
        def mock_calculate_with_errors(calc_args):
            """模拟包含错误的计算"""
            calculation_count[0] += 1
            
            # 某些对象计算失败
            if "Equipment05" in calc_args.name or "Equipment15" in calc_args.name:
                failed_objects.append(calc_args.name)
                raise CalculationException(f"计算失败: {calc_args.name}")
            
            # 其他对象正常计算
            time.sleep(0.005)  # 5ms
            calc_args.set_success(0.005, 1)
            return []
        
        # 执行包含错误的求解
        exceptions = solver.solve_flowsheet(
            industrial_scale_flowsheet,
            mode=SolverMode.SYNCHRONOUS
        )
        
        # 验证错误处理
        assert len(exceptions) > 0  # 有异常被捕获
        assert len(failed_objects) > 0  # 有对象失败
        
        # 验证继续计算
        stats = solver.performance_stats
        assert stats['successful_objects'] > 0  # 有对象成功
        assert stats['failed_objects'] > 0  # 有对象失败
        assert stats['total_objects'] == stats['successful_objects'] + stats['failed_objects']
        
        # 验证异常类型
        calculation_exceptions = [exc for exc in exceptions if isinstance(exc, CalculationException)]
        assert len(calculation_exceptions) > 0
        
        print(f"Handled {len(exceptions)} exceptions, {stats['successful_objects']} successful objects")


class TestEventSystemIntegration:
    """
    事件系统集成测试
    
    测试事件系统在复杂场景下的集成表现。
    """
    
    def test_comprehensive_event_monitoring(self, mock_flowsheet):
        """
        测试目标：验证完整的事件监控系统
        
        工作步骤：
        1. 注册所有类型的事件处理器
        2. 执行完整求解流程
        3. 验证所有事件被正确触发
        4. 检查事件数据完整性
        """
        solver = FlowsheetSolver()
        
        # 事件收集器
        event_log = []
        
        def create_event_handler(event_type):
            def handler(*args, **kwargs):
                event_log.append({
                    'type': event_type,
                    'timestamp': time.time(),
                    'args': args,
                    'kwargs': kwargs
                })
            return handler
        
        # 注册各种事件处理器
        events_to_monitor = [
            "flowsheet_calculation_started",
            "flowsheet_calculation_finished",
            "calculating_object",
            "object_calculation_completed",
            "iteration_completed",
            "convergence_achieved",
            "solver_error_occurred"
        ]
        
        for event_type in events_to_monitor:
            solver.add_event_handler(event_type, create_event_handler(event_type))
        
        # 模拟计算过程
        def mock_calculate_with_events(calc_args):
            # 触发对象计算开始事件
            solver._fire_event("calculating_object", calc_args)
            
            time.sleep(0.01)
            calc_args.set_success(0.01, 1)
            
            # 触发对象计算完成事件
            solver._fire_event("object_calculation_completed", calc_args)
            return []
        
        # 执行求解
        with patch.object(solver, '_calculate_object', side_effect=mock_calculate_with_events):
            exceptions = solver.solve_flowsheet(mock_flowsheet)
        
        # 验证事件
        assert len(event_log) > 0
        
        # 检查特定事件
        started_events = [e for e in event_log if e['type'] == 'flowsheet_calculation_started']
        finished_events = [e for e in event_log if e['type'] == 'flowsheet_calculation_finished']
        
        assert len(started_events) >= 1
        assert len(finished_events) >= 1
        
        # 验证事件顺序
        assert started_events[0]['timestamp'] < finished_events[0]['timestamp']
        
        print(f"Captured {len(event_log)} events during solving")
    
    def test_real_time_progress_monitoring(self, industrial_scale_flowsheet):
        """
        测试目标：验证实时进度监控
        
        工作步骤：
        1. 设置进度监控
        2. 执行大规模求解
        3. 验证进度更新
        4. 检查监控数据准确性
        """
        solver = FlowsheetSolver()
        
        # 进度监控数据
        progress_data = {
            'total_objects': 0,
            'completed_objects': 0,
            'current_object': None,
            'start_time': None,
            'estimated_completion': None
        }
        
        def on_calculation_started(flowsheet):
            progress_data['start_time'] = time.time()
            progress_data['total_objects'] = len([
                obj for obj in flowsheet.simulation_objects.values()
                if not obj.graphic_object.calculated and obj.graphic_object.active
            ])
        
        def on_object_completed(calc_args):
            progress_data['completed_objects'] += 1
            progress_data['current_object'] = calc_args.name
            
            # 估算完成时间
            if progress_data['completed_objects'] > 0 and progress_data['start_time']:
                elapsed = time.time() - progress_data['start_time']
                avg_time = elapsed / progress_data['completed_objects']
                remaining = progress_data['total_objects'] - progress_data['completed_objects']
                progress_data['estimated_completion'] = time.time() + (avg_time * remaining)
        
        # 注册进度监控
        solver.add_event_handler("flowsheet_calculation_started", on_calculation_started)
        solver.add_event_handler("object_calculation_completed", on_object_completed)
        
        # 模拟计算
        def mock_timed_calculate(calc_args):
            time.sleep(0.02)  # 20ms计算时间
            calc_args.set_success(0.02, 1)
            solver._fire_event("object_calculation_completed", calc_args)
            return []
        
        # 执行监控求解
        with patch.object(solver, '_calculate_object', side_effect=mock_timed_calculate):
            exceptions = solver.solve_flowsheet(industrial_scale_flowsheet)
        
        # 验证进度监控
        assert progress_data['total_objects'] > 0
        assert progress_data['completed_objects'] > 0
        assert progress_data['start_time'] is not None
        
        # 检查完成率
        completion_rate = progress_data['completed_objects'] / progress_data['total_objects']
        print(f"Completion rate: {completion_rate:.2%}")
        print(f"Monitored {progress_data['completed_objects']}/{progress_data['total_objects']} objects") 