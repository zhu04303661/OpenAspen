"""
pytest配置文件
============

定义全局测试配置、共享fixtures和测试工具函数。
"""

import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
import sys
import os

# 确保项目根目录在Python路径中
# 从tests目录向上一级找到项目根目录
tests_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(tests_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"tests/conftest.py: 添加项目根目录到Python路径: {project_root}")
print(f"tests/conftest.py: tests目录: {tests_dir}")

# 验证flowsheet_solver包可以被找到
try:
    import flowsheet_solver
    print(f"tests/conftest.py: ✅ flowsheet_solver导入成功: {flowsheet_solver.__file__}")
except ImportError as e:
    print(f"tests/conftest.py: ❌ flowsheet_solver导入失败: {e}")

# 配置测试日志
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="session")
def test_data_dir():
    """
    测试数据目录fixture
    
    Returns:
        Path: 测试数据目录路径
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir():
    """
    临时目录fixture，用于测试期间的文件操作
    
    Returns:
        Path: 临时目录路径
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_flowsheet():
    """
    模拟流程图对象fixture
    
    创建一个包含基本属性和方法的模拟流程图对象，用于测试。
    
    Returns:
        Mock: 模拟的流程图对象
    """
    flowsheet = Mock()
    
    # 基本属性
    flowsheet.solved = False
    flowsheet.error_message = ""
    flowsheet.calculation_queue = Mock()
    
    # 模拟仿真对象集合
    flowsheet.simulation_objects = {
        "Stream1": create_mock_material_stream("Stream1"),
        "Stream2": create_mock_material_stream("Stream2"),
        "Heater1": create_mock_unit_operation("Heater1", "Heater"),
        "Mixer1": create_mock_unit_operation("Mixer1", "Mixer"),
        "Recycle1": create_mock_recycle_object("Recycle1")
    }
    
    # 流程图选项
    flowsheet.flowsheet_options = Mock()
    flowsheet.flowsheet_options.simultaneous_adjust_solver_enabled = True
    
    return flowsheet


def create_mock_material_stream(name: str):
    """
    创建模拟物质流对象
    
    Args:
        name: 流股名称
        
    Returns:
        Mock: 模拟的物质流对象
    """
    stream = Mock()
    stream.name = name
    stream.calculated = False
    stream.error_message = ""
    stream.last_updated = None
    
    # 图形对象
    stream.graphic_object = Mock()
    stream.graphic_object.object_type = "MaterialStream"
    stream.graphic_object.active = True
    stream.graphic_object.calculated = False
    stream.graphic_object.name = name
    
    # 连接器
    stream.graphic_object.input_connectors = []
    stream.graphic_object.output_connectors = [Mock()]
    stream.graphic_object.output_connectors[0].is_attached = False
    
    return stream


def create_mock_unit_operation(name: str, operation_type: str):
    """
    创建模拟单元操作对象
    
    Args:
        name: 单元操作名称
        operation_type: 操作类型
        
    Returns:
        Mock: 模拟的单元操作对象
    """
    unit_op = Mock()
    unit_op.name = name
    unit_op.calculated = False
    unit_op.error_message = ""
    unit_op.last_updated = None
    
    # 图形对象
    unit_op.graphic_object = Mock()
    unit_op.graphic_object.object_type = operation_type
    unit_op.graphic_object.active = True
    unit_op.graphic_object.calculated = False
    unit_op.graphic_object.name = name
    
    # 连接器
    unit_op.graphic_object.input_connectors = [Mock(), Mock()]
    unit_op.graphic_object.output_connectors = [Mock()]
    
    # 模拟连接
    for connector in unit_op.graphic_object.input_connectors:
        connector.is_attached = True
        connector.attached_connector = Mock()
        connector.attached_connector.attached_from = Mock()
        connector.attached_connector.attached_from.object_type = "MaterialStream"
        connector.attached_connector.attached_from.name = f"Input_{name}"
    
    for connector in unit_op.graphic_object.output_connectors:
        connector.is_attached = False
    
    # 求解方法
    unit_op.solve = Mock()
    
    return unit_op


def create_mock_recycle_object(name: str):
    """
    创建模拟Recycle对象
    
    Args:
        name: Recycle对象名称
        
    Returns:
        Mock: 模拟的Recycle对象
    """
    recycle = Mock()
    recycle.name = name
    recycle.calculated = False
    recycle.converged = False
    recycle.acceleration_method = "GlobalBroyden"
    
    # 图形对象
    recycle.graphic_object = Mock()
    recycle.graphic_object.object_type = "Recycle"
    recycle.graphic_object.active = True
    recycle.graphic_object.calculated = False
    recycle.graphic_object.name = name
    
    # Recycle特有属性
    recycle.values = {"Temperature": 298.15, "Pressure": 101325.0, "Flow": 100.0}
    recycle.errors = {"Temperature": 0.0, "Pressure": 0.0, "Flow": 0.0}
    recycle.convergence_history = Mock()
    recycle.convergence_history.get_average_error = Mock(return_value=1e-3)
    
    # 方法
    recycle.set_outlet_stream_properties = Mock()
    
    return recycle


@pytest.fixture
def solver_settings():
    """
    求解器设置fixture
    
    Returns:
        SolverSettings: 测试用的求解器设置
    """
    from flowsheet_solver.solver import SolverSettings
    
    return SolverSettings(
        max_iterations=50,
        tolerance=1e-6,
        timeout_seconds=30,
        max_thread_multiplier=1,
        enable_parallel_processing=True,
        solver_break_on_exception=False
    )


@pytest.fixture
def sample_calculation_args():
    """
    示例计算参数fixture
    
    Returns:
        List[CalculationArgs]: 计算参数列表
    """
    from flowsheet_solver.calculation_args import CalculationArgs, ObjectType
    
    return [
        CalculationArgs(
            name="Stream1",
            tag="进料流",
            object_type=ObjectType.MaterialStream,
            sender="FlowsheetSolver"
        ),
        CalculationArgs(
            name="Heater1", 
            tag="加热器",
            object_type=ObjectType.UnitOperation,
            sender="FlowsheetSolver"
        ),
        CalculationArgs(
            name="Recycle1",
            tag="循环流",
            object_type=ObjectType.Recycle,
            sender="FlowsheetSolver"
        )
    ]


@pytest.fixture
def sample_matrices():
    """
    示例矩阵数据fixture，用于数值算法测试
    
    Returns:
        dict: 包含各种测试矩阵的字典
    """
    return {
        "small_identity": np.eye(3),
        "small_random": np.random.rand(3, 3),
        "singular": np.array([[1, 2], [2, 4]]),
        "well_conditioned": np.array([[4, 1], [1, 3]]),
        "ill_conditioned": np.array([[1, 1], [1, 1.0001]])
    }


@pytest.fixture
def sample_vectors():
    """
    示例向量数据fixture
    
    Returns:
        dict: 包含各种测试向量的字典  
    """
    return {
        "small_ones": np.ones(3),
        "small_zeros": np.zeros(3),
        "small_random": np.random.rand(3),
        "large_random": np.random.rand(100)
    }


def pytest_configure(config):
    """
    pytest配置钩子函数
    
    添加自定义标记和配置。
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    修改测试项目收集结果
    
    为没有标记的测试自动添加适当的标记。
    """
    for item in items:
        # 为没有标记的测试添加unit标记
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # 为performance目录下的测试添加slow标记
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance) 