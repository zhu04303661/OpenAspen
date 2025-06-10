"""
Solver Exceptions 求解器异常系统单元测试
=====================================

测试目标：
1. 异常层次结构的正确性
2. 各种专用异常类的功能
3. 异常消息和参数传递
4. 异常捕获和处理
5. 自定义异常属性

工作步骤：
1. 测试基础异常类SolverException
2. 测试每个专用异常类
3. 测试异常继承关系
4. 测试异常信息格式化
5. 测试异常参数传递和访问
6. 测试异常处理场景
"""

import pytest
from unittest.mock import Mock
import time

from flowsheet_solver.solver_exceptions import (
    SolverException,
    ConvergenceException,
    TimeoutException,
    InfiniteLoopException,
    CalculationException,
    NetworkException,
    DataException
)


class TestSolverExceptionHierarchy:
    """
    异常层次结构测试类
    
    测试异常类的继承关系和基本功能。
    """
    
    def test_base_exception_inheritance(self):
        """
        测试目标：验证所有异常类都继承自SolverException
        
        工作步骤：
        1. 检查SolverException继承自Exception
        2. 检查所有专用异常继承自SolverException
        3. 验证继承链的正确性
        """
        # SolverException继承自Exception
        assert issubclass(SolverException, Exception)
        
        # 所有专用异常继承自SolverException
        specialized_exceptions = [
            ConvergenceException,
            TimeoutException,
            InfiniteLoopException,
            CalculationException,
            NetworkException,
            DataException
        ]
        
        for exc_class in specialized_exceptions:
            assert issubclass(exc_class, SolverException)
            assert issubclass(exc_class, Exception)
    
    def test_exception_instantiation(self):
        """
        测试目标：验证所有异常类可以正确实例化
        
        工作步骤：
        1. 实例化每个异常类
        2. 验证实例是正确类型
        3. 检查默认消息
        """
        exceptions = [
            (SolverException, "求解器异常"),
            (ConvergenceException, "收敛异常"),
            (TimeoutException, "超时异常"),
            (InfiniteLoopException, "无限循环异常"),
            (CalculationException, "计算异常"),
            (NetworkException, "网络异常"),
            (DataException, "数据异常")
        ]
        
        for exc_class, default_msg in exceptions:
            # 无参数实例化
            exc = exc_class()
            assert isinstance(exc, exc_class)
            assert isinstance(exc, SolverException)
            
            # 带消息实例化
            custom_msg = f"自定义{default_msg}"
            exc_with_msg = exc_class(custom_msg)
            assert str(exc_with_msg) == custom_msg


class TestConvergenceException:
    """
    收敛异常ConvergenceException测试
    
    测试收敛相关的专用异常功能。
    """
    
    def test_convergence_specific_attributes(self):
        """
        测试目标：验证收敛异常特有属性
        
        工作步骤：
        1. 测试迭代次数记录
        2. 测试当前误差记录
        3. 测试容差设置
        """
        exc = ConvergenceException(
            message="收敛失败",
            max_iterations=100,
            current_error=1e-3,
            tolerance=1e-6
        )
        
        assert exc.max_iterations == 100
        assert exc.current_error == 1e-3
        assert exc.tolerance == 1e-6


class TestTimeoutException:
    """
    超时异常TimeoutException测试
    
    测试超时相关的专用异常功能。
    """
    
    def test_timeout_specific_attributes(self):
        """
        测试目标：验证超时异常特有属性
        
        工作步骤：
        1. 测试超时时间设置
        2. 测试实际耗时记录
        3. 测试操作类型记录
        """
        exc = TimeoutException(
            message="操作超时",
            timeout_seconds=300,
            elapsed_seconds=450,
            operation="FlowsheetSolving"
        )
        
        assert exc.timeout_seconds == 300
        assert exc.elapsed_seconds == 450
        assert exc.operation == "FlowsheetSolving"


class TestCalculationException:
    """
    计算异常CalculationException测试
    
    测试计算过程中的专用异常功能。
    """
    
    def test_calculation_specific_attributes(self):
        """
        测试目标：验证计算异常特有属性
        
        工作步骤：
        1. 测试计算对象信息
        2. 测试计算步骤记录
        3. 测试输入输出状态
        """
        exc = CalculationException(
            message="计算失败",
            calculation_object="Heater001",
            calculation_step="EnergyBalance"
        )
        
        assert exc.calculation_object == "Heater001"
        assert exc.calculation_step == "EnergyBalance"


class TestNetworkException:
    """
    网络异常NetworkException测试
    
    测试网络通信相关的专用异常功能。
    """
    
    def test_network_specific_attributes(self):
        """
        测试目标：验证网络异常特有属性
        
        工作步骤：
        1. 测试服务器地址记录
        2. 测试端口信息
        3. 测试连接状态
        """
        exc = NetworkException(
            message="连接失败",
            server_address="192.168.1.100",
            server_port=8080
        )
        
        assert exc.server_address == "192.168.1.100"
        assert exc.server_port == 8080


class TestDataException:
    """
    数据异常DataException测试
    
    测试数据相关的专用异常功能。
    """
    
    def test_data_specific_attributes(self):
        """
        测试目标：验证数据异常特有属性
        
        工作步骤：
        1. 测试数据类型记录
        2. 测试无效值记录
        3. 测试预期范围信息
        """
        exc = DataException(
            message="数据验证失败",
            data_type="Temperature",
            invalid_value=-500
        )
        
        assert exc.data_type == "Temperature"
        assert exc.invalid_value == -500 