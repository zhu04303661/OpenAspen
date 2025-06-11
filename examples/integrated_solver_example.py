"""
集成FlowsheetSolver示例
=====================

演示如何使用集成的FlowsheetSolver来创建和计算DWSIM单元操作。

包含以下示例：
1. 创建和配置单元操作
2. 构建简单的工艺流程
3. 执行计算和获取结果
4. 错误处理和调试

这个示例展示了从原VB.NET版本1:1转换的Python实现的完整功能。
"""

import sys
import os
import logging

# 添加路径以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dwsim_operations.integration import (
    create_integrated_solver, 
    IntegratedFlowsheetSolver,
    UnitOperationRegistry
)
from dwsim_operations.unit_operations import Mixer, Splitter, Heater
from flowsheet_solver.solver import SolverSettings


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_operations():
    """
    示例1: 基本单元操作的创建和使用
    """
    print("\n" + "="*60)
    print("示例1: 基本单元操作的创建和使用")
    print("="*60)
    
    # 创建集成求解器
    solver = create_integrated_solver()
    
    # 创建混合器
    mixer = solver.create_and_add_operation(
        operation_type="Mixer",
        name="MIX-001",
        description="原料混合器"
    )
    
    # 创建加热器
    heater = solver.create_and_add_operation(
        operation_type="Heater", 
        name="HX-001",
        description="预热器"
    )
    
    # 创建分离器
    splitter = solver.create_and_add_operation(
        operation_type="Splitter",
        name="SPL-001", 
        description="产品分离器"
    )
    
    print(f"已创建 {len(solver.unit_operations)} 个单元操作:")
    for name, operation in solver.unit_operations.items():
        print(f"  - {name}: {operation.__class__.__name__} ({operation.tag})")
    
    # 获取计算摘要
    summary = solver.get_calculation_summary()
    print(f"\n计算摘要:")
    print(f"  总操作数: {summary['total_operations']}")
    print(f"  已计算: {summary['calculated_operations']}")
    print(f"  错误数: {summary['error_operations']}")
    print(f"  按类型分组: {summary['operations_by_type']}")


def example_2_mixer_calculation():
    """
    示例2: 混合器计算示例
    """
    print("\n" + "="*60)
    print("示例2: 混合器计算示例")
    print("="*60)
    
    # 创建集成求解器
    solver = create_integrated_solver()
    
    # 创建混合器
    mixer = solver.create_and_add_operation(
        operation_type="Mixer",
        name="MIXER-001",
        description="三路混合器"
    )
    
    print(f"混合器详细信息:")
    print(f"  名称: {mixer.name}")
    print(f"  标签: {mixer.tag}")
    print(f"  描述: {mixer.description}")
    print(f"  类型: {mixer.__class__.__name__}")
    print(f"  压力计算模式: {mixer.pressure_calculation}")
    print(f"  输入连接点数: {len(mixer.graphic_object.input_connectors)}")
    print(f"  输出连接点数: {len(mixer.graphic_object.output_connectors)}")
    
    # 获取调试报告
    try:
        debug_report = mixer.get_debug_report()
        print(f"\n调试报告:")
        print(debug_report)
    except Exception as e:
        print(f"调试报告生成失败: {e}")


def example_3_operations_registry():
    """
    示例3: 单元操作注册表使用
    """
    print("\n" + "="*60)
    print("示例3: 单元操作注册表使用")
    print("="*60)
    
    # 获取注册表实例
    registry = UnitOperationRegistry()
    
    # 显示所有可用的操作类型
    available_ops = registry.get_available_operations()
    print(f"可用的单元操作类型 ({len(available_ops)} 种):")
    for i, op_type in enumerate(available_ops, 1):
        print(f"  {i:2d}. {op_type}")
    
    # 测试创建不同类型的操作
    print(f"\n创建不同类型的操作:")
    for op_type in ['Mixer', 'Heater', 'Pump', 'Valve']:
        try:
            operation = registry.create_operation(op_type, f"TEST-{op_type}-001")
            print(f"  ✓ {op_type}: {operation.name} - {operation.tag}")
        except Exception as e:
            print(f"  ✗ {op_type}: 创建失败 - {e}")


def example_4_configuration_export_import():
    """
    示例4: 配置导出和导入
    """
    print("\n" + "="*60)
    print("示例4: 配置导出和导入")
    print("="*60)
    
    # 创建求解器并添加操作
    solver1 = create_integrated_solver()
    
    # 添加多个操作
    operations_config = [
        ("Mixer", "MIX-001", "主混合器"),
        ("Heater", "HX-001", "预热器"),
        ("Pump", "P-001", "输送泵"),
        ("Cooler", "HX-002", "冷却器"),
        ("Splitter", "SPL-001", "产品分离器")
    ]
    
    for op_type, name, desc in operations_config:
        solver1.create_and_add_operation(op_type, name, desc)
    
    print(f"原始求解器中的操作数: {len(solver1.unit_operations)}")
    
    # 导出配置
    config = solver1.export_operations_config()
    print(f"导出的配置:")
    print(f"  操作总数: {config['metadata']['total_count']}")
    print(f"  操作列表:")
    for name, op_config in config['operations'].items():
        print(f"    - {name}: {op_config['type']} ({op_config['tag']})")
    
    # 创建新的求解器并导入配置
    solver2 = create_integrated_solver()
    solver2.import_operations_config(config)
    
    print(f"\n导入后求解器中的操作数: {len(solver2.unit_operations)}")
    print(f"导入的操作:")
    for name, operation in solver2.unit_operations.items():
        print(f"  - {name}: {operation.__class__.__name__} ({operation.tag})")


def example_5_error_handling():
    """
    示例5: 错误处理和验证
    """
    print("\n" + "="*60)
    print("示例5: 错误处理和验证")
    print("="*60)
    
    solver = create_integrated_solver()
    
    # 创建一些操作
    mixer = solver.create_and_add_operation("Mixer", "MIX-001", "测试混合器")
    heater = solver.create_and_add_operation("Heater", "HX-001", "测试加热器")
    
    # 测试验证功能
    print("验证所有操作:")
    validation_errors = solver.validate_all_operations()
    
    if validation_errors:
        print("发现验证错误:")
        for op_name, errors in validation_errors.items():
            print(f"  {op_name}:")
            for error in errors:
                print(f"    - {error}")
    else:
        print("所有操作验证通过")
    
    # 测试不存在的操作类型
    print(f"\n测试创建不存在的操作类型:")
    try:
        solver.create_and_add_operation("NonExistentType", "TEST-001")
    except Exception as e:
        print(f"  ✓ 正确捕获错误: {e}")
    
    # 测试计算不存在的操作
    print(f"\n测试计算不存在的操作:")
    try:
        solver.calculate_unit_operation("NON-EXISTENT")
    except Exception as e:
        print(f"  ✓ 正确捕获错误: {e}")


def example_6_custom_operation():
    """
    示例6: 自定义单元操作
    """
    print("\n" + "="*60)
    print("示例6: 自定义单元操作")
    print("="*60)
    
    # 定义自定义单元操作
    from dwsim_operations.base_classes import UnitOpBaseClass, SimulationObjectClass
    
    class CustomReactor(UnitOpBaseClass):
        """自定义反应器"""
        
        def __init__(self, name: str = "", description: str = ""):
            super().__init__()
            self.object_class = SimulationObjectClass.Reactors
            self.component_name = name or "CustomReactor"
            self.component_description = description or "自定义反应器"
            self.name = name or "REACTOR-001"
            self.tag = name or "自定义反应器"
            
            # 反应器特定属性
            self.reaction_temperature = 373.15  # K
            self.conversion = 0.8  # 转化率
        
        def calculate(self, args=None):
            """计算方法"""
            if self.debug_mode:
                self.append_debug_line("自定义反应器计算开始")
                self.append_debug_line(f"反应温度: {self.reaction_temperature} K")
                self.append_debug_line(f"转化率: {self.conversion * 100}%")
            
            # 这里添加具体的反应器计算逻辑
            self.calculated = True
    
    # 注册自定义操作
    from dwsim_operations.integration import register_custom_operation
    register_custom_operation("CustomReactor", CustomReactor)
    
    # 创建求解器并使用自定义操作
    solver = create_integrated_solver()
    
    # 创建自定义反应器
    reactor = solver.create_and_add_operation(
        "CustomReactor", 
        "REACTOR-001", 
        "高温催化反应器"
    )
    
    print(f"创建的自定义反应器:")
    print(f"  名称: {reactor.name}")
    print(f"  类型: {reactor.__class__.__name__}")
    print(f"  反应温度: {reactor.reaction_temperature} K")
    print(f"  转化率: {reactor.conversion * 100}%")
    
    # 执行计算
    success = solver.calculate_unit_operation("REACTOR-001")
    print(f"  计算结果: {'成功' if success else '失败'}")


def main():
    """
    主函数 - 运行所有示例
    """
    print("DWSIM 集成FlowsheetSolver使用示例")
    print("="*60)
    print("这个示例展示了从原VB.NET版本1:1转换的Python实现")
    print("包含完整的单元操作功能和flowsheet_solver集成")
    
    try:
        # 运行所有示例
        example_1_basic_operations()
        example_2_mixer_calculation()
        example_3_operations_registry()
        example_4_configuration_export_import()
        example_5_error_handling()
        example_6_custom_operation()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 