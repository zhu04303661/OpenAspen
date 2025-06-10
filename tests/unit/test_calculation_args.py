"""
CalculationArgs 计算参数类单元测试
=================================

测试目标：
1. CalculationArgs类的初始化和属性
2. ObjectType和CalculationStatus枚举  
3. 参数验证和转换
4. 状态管理方法
5. 数据复制和字符串表示

工作步骤：
1. 测试基本属性设置和获取
2. 测试枚举类型的转换和验证
3. 测试状态管理方法
4. 测试数据有效性检查
5. 测试对象复制和比较
6. 测试异常情况处理
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from flowsheet_solver.calculation_args import (
    CalculationArgs, 
    ObjectType, 
    CalculationStatus
)


class TestObjectType:
    """
    ObjectType枚举测试类
    
    测试对象类型枚举的定义和使用。
    """
    
    def test_object_type_values(self):
        """
        测试目标：验证ObjectType枚举包含所有预期的值
        
        工作步骤：
        1. 检查所有枚举值是否存在
        2. 验证枚举值的字符串表示
        3. 确保没有重复值
        """
        # 检查所有预期的枚举值
        expected_types = [
            "MaterialStream", "EnergyStream", "UnitOperation", 
            "Recycle", "EnergyRecycle", "Specification", 
            "Adjust", "Unknown"
        ]
        
        for type_name in expected_types:
            assert hasattr(ObjectType, type_name.upper().replace("STREAM", "_STREAM"))
        
        # 验证字符串值
        assert ObjectType.MaterialStream.value == "MaterialStream"
        assert ObjectType.EnergyStream.value == "EnergyStream"  
        assert ObjectType.UnitOperation.value == "UnitOperation"
        assert ObjectType.Recycle.value == "Recycle"
        assert ObjectType.Unknown.value == "Unknown"
    
    def test_object_type_uniqueness(self):
        """
        测试目标：确保所有枚举值都是唯一的
        
        工作步骤：
        1. 获取所有枚举值
        2. 检查是否有重复值
        """
        values = [obj_type.value for obj_type in ObjectType]
        assert len(values) == len(set(values)), "ObjectType枚举值存在重复"


class TestCalculationStatus:
    """
    CalculationStatus枚举测试类
    
    测试计算状态枚举的定义和使用。
    """
    
    def test_calculation_status_values(self):
        """
        测试目标：验证CalculationStatus枚举包含所有预期状态
        
        工作步骤：
        1. 检查所有状态值是否存在
        2. 验证状态值的字符串表示
        """
        expected_statuses = [
            "NotStarted", "Running", "Completed", 
            "Error", "Cancelled", "Timeout"
        ]
        
        for status_name in expected_statuses:
            status_attr = status_name.upper()
            if status_name == "NotStarted":
                status_attr = "NOTSTARTED" 
            assert hasattr(CalculationStatus, status_attr)
        
        # 验证字符串值
        assert CalculationStatus.NotStarted.value == "NotStarted"
        assert CalculationStatus.Running.value == "Running"
        assert CalculationStatus.Completed.value == "Completed"
        assert CalculationStatus.Error.value == "Error"


class TestCalculationArgs:
    """
    CalculationArgs类测试类
    
    测试计算参数类的所有功能。
    """
    
    def test_default_initialization(self):
        """
        测试目标：验证CalculationArgs的默认初始化
        
        工作步骤：
        1. 创建默认实例
        2. 检查所有属性的默认值
        3. 验证数据类型
        """
        # 创建默认实例
        calc_args = CalculationArgs()
        
        # 检查默认值
        assert calc_args.name == ""
        assert calc_args.tag == ""
        assert calc_args.object_type == ObjectType.Unknown
        assert calc_args.sender == ""
        assert calc_args.calculated == False
        assert calc_args.error_message == ""
        assert calc_args.calculation_time == 0.0
        assert calc_args.iteration_count == 0
        assert calc_args.priority == 0
    
    def test_custom_initialization(self):
        """
        测试目标：验证CalculationArgs的自定义初始化
        
        工作步骤：
        1. 使用自定义参数创建实例
        2. 验证所有参数被正确设置
        3. 测试不同的数据类型组合
        """
        # 创建自定义实例
        calc_args = CalculationArgs(
            name="TestObject",
            tag="测试对象",
            object_type=ObjectType.MaterialStream,
            sender="FlowsheetSolver",
            calculated=True,
            error_message="Test error",
            calculation_time=1.5,
            iteration_count=3,
            priority=1
        )
        
        # 验证所有属性
        assert calc_args.name == "TestObject"
        assert calc_args.tag == "测试对象"
        assert calc_args.object_type == ObjectType.MaterialStream
        assert calc_args.sender == "FlowsheetSolver"
        assert calc_args.calculated == True
        assert calc_args.error_message == "Test error"
        assert calc_args.calculation_time == 1.5
        assert calc_args.iteration_count == 3
        assert calc_args.priority == 1
    
    def test_object_type_string_conversion(self):
        """
        测试目标：验证字符串到ObjectType的自动转换
        
        工作步骤：
        1. 使用字符串创建实例
        2. 验证自动转换为正确的枚举值
        3. 测试无效字符串的处理
        """
        # 有效字符串转换
        calc_args = CalculationArgs(object_type="MaterialStream")
        assert calc_args.object_type == ObjectType.MaterialStream
        
        calc_args = CalculationArgs(object_type="UnitOperation")
        assert calc_args.object_type == ObjectType.UnitOperation
        
        # 无效字符串转换
        calc_args = CalculationArgs(object_type="InvalidType")
        assert calc_args.object_type == ObjectType.Unknown
    
    def test_reset_calculation_state(self):
        """
        测试目标：验证计算状态重置功能
        
        工作步骤：
        1. 创建具有计算状态的实例
        2. 调用重置方法
        3. 验证状态被正确重置
        """
        # 创建具有计算状态的实例
        calc_args = CalculationArgs(
            calculated=True,
            error_message="Some error",
            calculation_time=2.5,
            iteration_count=5
        )
        
        # 重置计算状态
        calc_args.reset_calculation_state()
        
        # 验证状态被重置
        assert calc_args.calculated == False
        assert calc_args.error_message == ""
        assert calc_args.calculation_time == 0.0
        assert calc_args.iteration_count == 0
    
    def test_set_error(self):
        """
        测试目标：验证错误设置功能
        
        工作步骤：
        1. 创建实例并设置错误
        2. 验证错误信息和状态
        3. 测试空错误信息
        """
        calc_args = CalculationArgs(calculated=True)
        
        # 设置错误
        error_msg = "计算失败"
        calc_args.set_error(error_msg)
        
        # 验证错误状态
        assert calc_args.error_message == error_msg
        assert calc_args.calculated == False
        
        # 测试空错误信息
        calc_args.set_error("")
        assert calc_args.error_message == ""
        assert calc_args.calculated == False
    
    def test_set_success(self):
        """
        测试目标：验证成功状态设置功能
        
        工作步骤：
        1. 创建具有错误状态的实例
        2. 设置成功状态
        3. 验证状态和计算信息
        """
        calc_args = CalculationArgs(
            calculated=False,
            error_message="Previous error"
        )
        
        # 设置成功状态
        calc_time = 1.2
        iterations = 3
        calc_args.set_success(calc_time, iterations)
        
        # 验证成功状态
        assert calc_args.calculated == True
        assert calc_args.error_message == ""
        assert calc_args.calculation_time == calc_time
        assert calc_args.iteration_count == iterations
        
        # 测试默认参数
        calc_args.set_success()
        assert calc_args.calculated == True
        assert calc_args.calculation_time == 0.0
        assert calc_args.iteration_count == 0
    
    def test_copy(self):
        """
        测试目标：验证对象复制功能
        
        工作步骤：
        1. 创建原始实例
        2. 创建副本
        3. 验证副本的独立性
        4. 验证所有属性被正确复制
        """
        # 创建原始实例
        original = CalculationArgs(
            name="Original",
            tag="原始对象",
            object_type=ObjectType.UnitOperation,
            sender="Test",
            calculated=True,
            error_message="Test error",
            calculation_time=2.0,
            iteration_count=4,
            priority=2
        )
        
        # 创建副本
        copy = original.copy()
        
        # 验证副本不是同一个对象
        assert copy is not original
        
        # 验证所有属性被正确复制
        assert copy.name == original.name
        assert copy.tag == original.tag
        assert copy.object_type == original.object_type
        assert copy.sender == original.sender
        assert copy.calculated == original.calculated
        assert copy.error_message == original.error_message
        assert copy.calculation_time == original.calculation_time
        assert copy.iteration_count == original.iteration_count
        assert copy.priority == original.priority
        
        # 验证独立性
        copy.name = "Modified"
        assert original.name == "Original"
    
    def test_is_valid(self):
        """
        测试目标：验证参数有效性检查功能
        
        工作步骤：
        1. 测试有效参数组合
        2. 测试无效参数组合
        3. 验证边界条件
        """
        # 有效参数
        valid_args = CalculationArgs(
            name="ValidObject",
            object_type=ObjectType.MaterialStream
        )
        assert valid_args.is_valid() == True
        
        # 空名称
        invalid_name = CalculationArgs(
            name="",
            object_type=ObjectType.MaterialStream
        )
        assert invalid_name.is_valid() == False
        
        # Unknown类型
        invalid_type = CalculationArgs(
            name="ValidObject",
            object_type=ObjectType.Unknown
        )
        assert invalid_type.is_valid() == False
        
        # 同时无效
        invalid_both = CalculationArgs(name="", object_type=ObjectType.Unknown)
        assert invalid_both.is_valid() == False
    
    def test_string_representation(self):
        """
        测试目标：验证字符串表示功能
        
        工作步骤：
        1. 测试__str__方法
        2. 测试__repr__方法
        3. 验证不同状态下的字符串表示
        """
        # 测试基本字符串表示
        calc_args = CalculationArgs(
            name="TestObject",
            tag="测试对象", 
            object_type=ObjectType.MaterialStream,
            sender="FlowsheetSolver"
        )
        
        str_repr = str(calc_args)
        assert "TestObject" in str_repr
        assert "测试对象" in str_repr
        assert "MaterialStream" in str_repr
        assert "FlowsheetSolver" in str_repr
        assert "未计算" in str_repr
        
        # 测试计算完成状态
        calc_args.calculated = True
        str_repr = str(calc_args)
        assert "已计算" in str_repr
        
        # 测试错误状态
        calc_args.set_error("计算错误")
        str_repr = str(calc_args)
        assert "错误" in str_repr
        assert "计算错误" in str_repr
        
        # 测试__repr__方法
        repr_str = repr(calc_args)
        assert repr_str == str(calc_args)
    
    def test_edge_cases(self):
        """
        测试目标：验证边界条件和特殊情况
        
        工作步骤：
        1. 测试极大值和极小值
        2. 测试特殊字符
        3. 测试None值处理
        """
        # 测试极大计算时间
        calc_args = CalculationArgs()
        calc_args.set_success(calculation_time=1e6, iteration_count=999999)
        assert calc_args.calculation_time == 1e6
        assert calc_args.iteration_count == 999999
        
        # 测试负值
        calc_args.calculation_time = -1.0
        calc_args.iteration_count = -1
        calc_args.priority = -1
        assert calc_args.calculation_time == -1.0
        assert calc_args.iteration_count == -1
        assert calc_args.priority == -1
        
        # 测试特殊字符
        calc_args = CalculationArgs(
            name="对象™",
            tag="测试@#$%",
            error_message="错误：计算失败！"
        )
        assert calc_args.name == "对象™"
        assert calc_args.tag == "测试@#$%"
        assert calc_args.error_message == "错误：计算失败！"


class TestCalculationArgsIntegration:
    """
    CalculationArgs集成测试类
    
    测试与其他组件的集成场景。
    """
    
    def test_calculation_workflow(self):
        """
        测试目标：模拟完整的计算工作流程
        
        工作步骤：
        1. 创建计算参数
        2. 模拟计算开始
        3. 模拟计算过程
        4. 模拟计算完成
        5. 验证状态变化
        """
        # 创建初始计算参数
        calc_args = CalculationArgs(
            name="Heater001",
            tag="主加热器",
            object_type=ObjectType.UnitOperation,
            sender="FlowsheetSolver"
        )
        
        # 验证初始状态
        assert calc_args.is_valid() == True
        assert calc_args.calculated == False
        
        # 模拟计算开始（重置状态）
        calc_args.reset_calculation_state()
        start_time = datetime.now()
        
        # 模拟计算过程（这里可以添加更多中间状态）
        calc_args.iteration_count = 1
        
        # 模拟计算成功完成
        end_time = datetime.now()
        calc_time = (end_time - start_time).total_seconds()
        calc_args.set_success(calc_time, 5)
        
        # 验证最终状态
        assert calc_args.calculated == True
        assert calc_args.error_message == ""
        assert calc_args.calculation_time >= 0
        assert calc_args.iteration_count == 5
    
    def test_error_recovery_workflow(self):
        """
        测试目标：模拟错误恢复工作流程
        
        工作步骤：
        1. 模拟计算失败
        2. 处理错误
        3. 重试计算
        4. 成功完成
        """
        calc_args = CalculationArgs(
            name="ProblemObject",
            object_type=ObjectType.UnitOperation
        )
        
        # 模拟第一次计算失败
        calc_args.set_error("初始计算失败")
        assert calc_args.calculated == False
        assert "初始计算失败" in calc_args.error_message
        
        # 模拟错误处理和重试
        calc_args.reset_calculation_state()
        assert calc_args.error_message == ""
        
        # 模拟第二次计算成功
        calc_args.set_success(1.5, 3)
        assert calc_args.calculated == True
        assert calc_args.error_message == ""
    
    def test_multiple_objects_scenario(self):
        """
        测试目标：测试多对象计算场景
        
        工作步骤：
        1. 创建多个计算参数对象
        2. 验证对象独立性
        3. 模拟批量操作
        """
        # 创建多个对象
        objects = [
            CalculationArgs(name=f"Stream{i}", object_type=ObjectType.MaterialStream)
            for i in range(1, 6)
        ]
        
        # 验证对象独立性
        objects[0].set_success(1.0, 1)
        objects[1].set_error("错误")
        
        assert objects[0].calculated == True
        assert objects[1].calculated == False
        assert objects[2].calculated == False  # 未受影响
        
        # 模拟批量状态重置
        for obj in objects:
            obj.reset_calculation_state()
        
        # 验证所有对象状态被重置
        for obj in objects:
            assert obj.calculated == False
            assert obj.error_message == "" 