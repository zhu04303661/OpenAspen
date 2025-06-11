"""
DWSIM 单元操作模块测试
===================

完整测试DWSIM单元操作的Python实现，包括：
- 基础类测试
- 具体单元操作测试  
- 集成求解器测试
- 错误处理测试
- 性能测试

确保从原VB.NET版本1:1转换的功能完全正确。
"""

import unittest
import sys
import os
import logging
from unittest.mock import Mock, patch

# 添加路径以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from dwsim_operations.base_classes import (
        UnitOpBaseClass, SpecialOpBaseClass, SimulationObjectClass,
        ConnectionPoint, GraphicObject
    )
    from dwsim_operations.unit_operations import (
        Mixer, Splitter, Heater, Cooler, HeatExchanger,
        Pump, Compressor, Valve, ComponentSeparator,
        Filter, Vessel, Tank, PressureBehavior
    )
    from dwsim_operations.integration import (
        UnitOperationRegistry, IntegratedFlowsheetSolver,
        create_integrated_solver, register_custom_operation
    )
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


class TestBaseClasses(unittest.TestCase):
    """测试基础类"""
    
    def setUp(self):
        """测试前准备"""
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """测试后清理"""
        logging.disable(logging.NOTSET)
    
    def test_connection_point(self):
        """测试连接点"""
        cp = ConnectionPoint()
        
        # 测试初始状态
        self.assertFalse(cp.is_attached)
        self.assertEqual(cp.attached_connector_name, "")
        self.assertEqual(cp.attached_to_name, "")
        
        # 测试连接
        cp.attach("connector1", "object1")
        self.assertTrue(cp.is_attached)
        self.assertEqual(cp.attached_connector_name, "connector1")
        self.assertEqual(cp.attached_to_name, "object1")
        
        # 测试断开连接
        cp.detach()
        self.assertFalse(cp.is_attached)
        self.assertEqual(cp.attached_connector_name, "")
        self.assertEqual(cp.attached_to_name, "")
    
    def test_graphic_object(self):
        """测试图形对象"""
        go = GraphicObject(tag="TEST", name="test_object")
        
        self.assertEqual(go.tag, "TEST")
        self.assertEqual(go.name, "test_object")
        self.assertFalse(go.calculated)
        self.assertTrue(go.active)
        self.assertEqual(len(go.input_connectors), 0)
        self.assertEqual(len(go.output_connectors), 0)
        self.assertEqual(len(go.energy_connectors), 0)
    
    def test_simulation_object_class(self):
        """测试仿真对象分类枚举"""
        # 测试所有枚举值
        self.assertEqual(SimulationObjectClass.Streams.value, "Streams")
        self.assertEqual(SimulationObjectClass.MixersSplitters.value, "MixersSplitters")
        self.assertEqual(SimulationObjectClass.HeatExchangers.value, "HeatExchangers")
    
    def test_special_op_base_class(self):
        """测试特殊操作基础类"""
        special_op = SpecialOpBaseClass()
        
        # 测试初始状态
        self.assertEqual(special_op.object_class, SimulationObjectClass.Logical)
        self.assertFalse(special_op.calculated)
        
        # 测试计算
        special_op.calculate()
        self.assertTrue(special_op.calculated)


class MockUnitOperation(UnitOpBaseClass):
    """用于测试的模拟单元操作"""
    
    def __init__(self, name="TEST", description="Test Operation"):
        super().__init__()
        self.name = name
        self.description = description
    
    def calculate(self, args=None):
        """模拟计算"""
        self.calculated = True


class TestUnitOperations(unittest.TestCase):
    """测试具体单元操作"""
    
    def setUp(self):
        """测试前准备"""
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """测试后清理"""
        logging.disable(logging.NOTSET)
    
    def test_mixer_creation(self):
        """测试混合器创建"""
        mixer = Mixer("MIX-001", "测试混合器")
        
        # 测试基本属性
        self.assertEqual(mixer.name, "MIX-001")
        self.assertEqual(mixer.component_description, "测试混合器")
        self.assertEqual(mixer.object_class, SimulationObjectClass.MixersSplitters)
        self.assertEqual(mixer.pressure_calculation, PressureBehavior.MINIMUM)
        
        # 测试图形对象
        self.assertIsNotNone(mixer.graphic_object)
        self.assertEqual(len(mixer.graphic_object.input_connectors), 6)
        self.assertEqual(len(mixer.graphic_object.output_connectors), 1)
    
    def test_mixer_pressure_behavior(self):
        """测试混合器压力行为"""
        mixer = Mixer()
        
        # 测试默认值
        self.assertEqual(mixer.pressure_calculation, PressureBehavior.MINIMUM)
        
        # 测试设置不同值
        mixer.pressure_calculation = PressureBehavior.AVERAGE
        self.assertEqual(mixer.pressure_calculation, PressureBehavior.AVERAGE)
        
        mixer.pressure_calculation = PressureBehavior.MAXIMUM
        self.assertEqual(mixer.pressure_calculation, PressureBehavior.MAXIMUM)
    
    def test_mixer_validation(self):
        """测试混合器验证"""
        mixer = Mixer("MIX-001")
        
        # 测试无连接的验证（应该失败）
        with self.assertRaises(ValueError):
            mixer._validate_connections()
    
    def test_mixer_clone(self):
        """测试混合器克隆"""
        original = Mixer("MIX-001", "原始混合器")
        original.pressure_calculation = PressureBehavior.AVERAGE
        
        # 测试JSON克隆
        cloned = original.clone_json()
        self.assertEqual(cloned.name, original.name)
        self.assertEqual(cloned.pressure_calculation, original.pressure_calculation)
        self.assertIsNot(cloned, original)
    
    def test_all_unit_operations_creation(self):
        """测试所有单元操作的创建"""
        operations = [
            (Mixer, "混合器"),
            (Splitter, "分离器"),
            (Heater, "加热器"),
            (Cooler, "冷却器"),
            (HeatExchanger, "热交换器"),
            (Pump, "泵"),
            (Compressor, "压缩机"),
            (Valve, "阀门"),
            (ComponentSeparator, "组分分离器"),
            (Filter, "过滤器"),
            (Vessel, "容器"),
            (Tank, "储罐")
        ]
        
        for op_class, description in operations:
            with self.subTest(operation=op_class.__name__):
                op = op_class(f"TEST-{op_class.__name__}", f"测试{description}")
                
                # 基本属性测试
                self.assertIsInstance(op, UnitOpBaseClass)
                self.assertTrue(op.name.startswith("TEST-"))
                self.assertIsNotNone(op.graphic_object)
                self.assertFalse(op.calculated)
                
                # 测试计算方法存在
                self.assertTrue(hasattr(op, 'calculate'))
                self.assertTrue(callable(op.calculate))


class TestUnitOperationRegistry(unittest.TestCase):
    """测试单元操作注册表"""
    
    def setUp(self):
        """测试前准备"""
        logging.disable(logging.CRITICAL)
        self.registry = UnitOperationRegistry()
    
    def tearDown(self):
        """测试后清理"""
        logging.disable(logging.NOTSET)
    
    def test_registry_initialization(self):
        """测试注册表初始化"""
        available_ops = self.registry.get_available_operations()
        
        # 检查默认操作是否已注册
        expected_ops = [
            'Mixer', 'Splitter', 'Heater', 'Cooler', 'HeatExchanger',
            'Pump', 'Compressor', 'Valve', 'ComponentSeparator',
            'Filter', 'Vessel', 'Tank'
        ]
        
        for op in expected_ops:
            self.assertIn(op, available_ops)
    
    def test_custom_operation_registration(self):
        """测试自定义操作注册"""
        # 注册自定义操作
        self.registry.register_operation("MockUnitOperation", MockUnitOperation)
        
        # 检查是否已注册
        self.assertTrue(self.registry.is_registered("MockUnitOperation"))
        
        # 创建实例
        operation = self.registry.create_operation("MockUnitOperation", "MOCK-001")
        self.assertIsInstance(operation, MockUnitOperation)
        self.assertEqual(operation.name, "MOCK-001")
    
    def test_invalid_operation_registration(self):
        """测试无效操作注册"""
        class InvalidOperation:
            pass
        
        with self.assertRaises(ValueError):
            self.registry.register_operation("Invalid", InvalidOperation)
    
    def test_unknown_operation_creation(self):
        """测试创建未知操作"""
        with self.assertRaises(ValueError):
            self.registry.create_operation("UnknownOperation", "TEST-001")


class TestIntegratedFlowsheetSolver(unittest.TestCase):
    """测试集成FlowsheetSolver"""
    
    def setUp(self):
        """测试前准备"""
        logging.disable(logging.CRITICAL)
        self.solver = create_integrated_solver()
    
    def tearDown(self):
        """测试后清理"""
        logging.disable(logging.NOTSET)
    
    def test_solver_creation(self):
        """测试求解器创建"""
        self.assertIsInstance(self.solver, IntegratedFlowsheetSolver)
        self.assertIsNotNone(self.solver.unit_operation_registry)
        self.assertEqual(len(self.solver.unit_operations), 0)
    
    def test_add_unit_operation(self):
        """测试添加单元操作"""
        mixer = Mixer("MIX-001", "测试混合器")
        self.solver.add_unit_operation(mixer)
        
        self.assertEqual(len(self.solver.unit_operations), 1)
        self.assertIn("MIX-001", self.solver.unit_operations)
        self.assertEqual(self.solver.unit_operations["MIX-001"], mixer)
    
    def test_create_and_add_operation(self):
        """测试创建并添加操作"""
        heater = self.solver.create_and_add_operation(
            "Heater", "HX-001", "测试加热器"
        )
        
        self.assertIsInstance(heater, Heater)
        self.assertEqual(heater.name, "HX-001")
        self.assertIn("HX-001", self.solver.unit_operations)
    
    def test_remove_unit_operation(self):
        """测试移除单元操作"""
        # 添加操作
        self.solver.create_and_add_operation("Mixer", "MIX-001")
        self.assertEqual(len(self.solver.unit_operations), 1)
        
        # 移除操作
        self.solver.remove_unit_operation("MIX-001")
        self.assertEqual(len(self.solver.unit_operations), 0)
        self.assertNotIn("MIX-001", self.solver.unit_operations)
    
    def test_get_operation_by_name(self):
        """测试按名称获取操作"""
        mixer = self.solver.create_and_add_operation("Mixer", "MIX-001")
        
        retrieved = self.solver.get_operation_by_name("MIX-001")
        self.assertEqual(retrieved, mixer)
        
        # 测试不存在的操作
        none_retrieved = self.solver.get_operation_by_name("NON-EXISTENT")
        self.assertIsNone(none_retrieved)
    
    def test_get_operations_by_type(self):
        """测试按类型获取操作"""
        # 添加不同类型的操作
        self.solver.create_and_add_operation("Mixer", "MIX-001")
        self.solver.create_and_add_operation("Mixer", "MIX-002")
        self.solver.create_and_add_operation("Heater", "HX-001")
        
        mixers = self.solver.get_operations_by_type(Mixer)
        self.assertEqual(len(mixers), 2)
        for mixer in mixers:
            self.assertIsInstance(mixer, Mixer)
    
    def test_calculation_summary(self):
        """测试计算摘要"""
        # 添加一些操作
        self.solver.create_and_add_operation("Mixer", "MIX-001")
        self.solver.create_and_add_operation("Heater", "HX-001")
        
        summary = self.solver.get_calculation_summary()
        
        self.assertEqual(summary['total_operations'], 2)
        self.assertEqual(summary['calculated_operations'], 0)
        self.assertEqual(summary['error_operations'], 0)
        self.assertIn('Mixer', summary['operations_by_type'])
        self.assertIn('Heater', summary['operations_by_type'])
    
    def test_reset_all_calculations(self):
        """测试重置所有计算"""
        # 添加操作并设置为已计算
        mixer = self.solver.create_and_add_operation("Mixer", "MIX-001")
        mixer.calculated = True
        
        # 重置
        self.solver.reset_all_calculations()
        self.assertFalse(mixer.calculated)
    
    def test_export_import_config(self):
        """测试配置导出和导入"""
        # 添加操作
        self.solver.create_and_add_operation("Mixer", "MIX-001", "混合器1")
        self.solver.create_and_add_operation("Heater", "HX-001", "加热器1")
        
        # 导出配置
        config = self.solver.export_operations_config()
        
        self.assertIn('operations', config)
        self.assertIn('metadata', config)
        self.assertEqual(config['metadata']['total_count'], 2)
        self.assertIn('MIX-001', config['operations'])
        self.assertIn('HX-001', config['operations'])
        
        # 创建新求解器并导入
        new_solver = create_integrated_solver()
        new_solver.import_operations_config(config)
        
        self.assertEqual(len(new_solver.unit_operations), 2)
        self.assertIn('MIX-001', new_solver.unit_operations)
        self.assertIn('HX-001', new_solver.unit_operations)
    
    def test_validation(self):
        """测试验证功能"""
        # 添加操作
        self.solver.create_and_add_operation("Mixer", "MIX-001")
        
        # 验证（应该有错误，因为没有属性包）
        errors = self.solver.validate_all_operations()
        
        # 应该有验证错误
        self.assertIn('MIX-001', errors)
        self.assertTrue(len(errors['MIX-001']) > 0)


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def setUp(self):
        """测试前准备"""
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """测试后清理"""
        logging.disable(logging.NOTSET)
    
    def test_invalid_operation_type(self):
        """测试无效操作类型"""
        solver = create_integrated_solver()
        
        with self.assertRaises(ValueError):
            solver.create_and_add_operation("InvalidType", "TEST-001")
    
    def test_calculate_nonexistent_operation(self):
        """测试计算不存在的操作"""
        solver = create_integrated_solver()
        
        with self.assertRaises(ValueError):
            solver.calculate_unit_operation("NON-EXISTENT")
    
    def test_add_invalid_operation(self):
        """测试添加无效操作"""
        solver = create_integrated_solver()
        
        with self.assertRaises(TypeError):
            solver.add_unit_operation("not_an_operation")


class TestPerformance(unittest.TestCase):
    """测试性能"""
    
    def setUp(self):
        """测试前准备"""
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """测试后清理"""
        logging.disable(logging.NOTSET)
    
    def test_large_number_of_operations(self):
        """测试大量操作的性能"""
        solver = create_integrated_solver()
        
        # 创建大量操作
        import time
        start_time = time.time()
        
        for i in range(100):
            solver.create_and_add_operation("Mixer", f"MIX-{i:03d}")
        
        creation_time = time.time() - start_time
        
        # 检查创建时间（应该在合理范围内）
        self.assertLess(creation_time, 5.0)  # 5秒内完成
        self.assertEqual(len(solver.unit_operations), 100)
        
        # 测试摘要生成时间
        start_time = time.time()
        summary = solver.get_calculation_summary()
        summary_time = time.time() - start_time
        
        self.assertLess(summary_time, 1.0)  # 1秒内完成
        self.assertEqual(summary['total_operations'], 100)


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestBaseClasses,
        TestUnitOperations,
        TestUnitOperationRegistry,
        TestIntegratedFlowsheetSolver,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("DWSIM 单元操作模块测试")
    print("="*50)
    print("测试从原VB.NET版本1:1转换的Python实现")
    print("="*50)
    
    success = run_all_tests()
    
    if success:
        print("\n" + "="*50)
        print("所有测试通过！✓")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("部分测试失败！✗")
        print("="*50)
        sys.exit(1) 