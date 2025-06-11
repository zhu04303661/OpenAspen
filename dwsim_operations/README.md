# DWSIM 单元操作 Python 实现

## 概述

这是 DWSIM5 单元操作模块的完整 Python 实现，从原 VB.NET 版本 1:1 转换而来。该模块提供了完整的单元操作功能，包括基础类、具体操作实现和与现有 FlowsheetSolver 的无缝集成。

## 功能特性

### 🎯 核心功能
- **完整的单元操作实现**：混合器、分离器、热交换器、泵、压缩机等
- **基础类架构**：提供可扩展的单元操作基础框架
- **集成求解器**：与现有 FlowsheetSolver 完美衔接
- **图形对象支持**：包含连接点和图形表示功能
- **属性包管理**：支持物性计算和热力学模型

### 🛠️ 技术特点
- **1:1 功能转换**：保持与原 VB.NET 版本完全一致的功能
- **类型安全**：使用 Python 类型提示确保代码质量
- **完整注释**：所有类和方法都有详细的中文注释
- **模块化设计**：清晰的模块结构便于维护和扩展
- **异常处理**：完善的错误处理和验证机制

## 模块结构

```
dwsim_operations/
├── __init__.py                 # 模块初始化
├── base_classes.py            # 基础类定义
├── unit_operations.py         # 具体单元操作实现
├── integration.py             # FlowsheetSolver 集成
└── README.md                  # 文档说明
```

## 快速开始

### 1. 基本使用

```python
from dwsim_operations.integration import create_integrated_solver
from dwsim_operations.unit_operations import Mixer

# 创建集成求解器
solver = create_integrated_solver()

# 创建混合器
mixer = solver.create_and_add_operation(
    operation_type="Mixer",
    name="MIX-001", 
    description="原料混合器"
)

# 获取计算摘要
summary = solver.get_calculation_summary()
print(f"总操作数: {summary['total_operations']}")
```

### 2. 创建自定义单元操作

```python
from dwsim_operations.base_classes import UnitOpBaseClass, SimulationObjectClass
from dwsim_operations.integration import register_custom_operation

class CustomReactor(UnitOpBaseClass):
    """自定义反应器"""
    
    def __init__(self, name="", description=""):
        super().__init__()
        self.object_class = SimulationObjectClass.Reactors
        self.name = name or "REACTOR-001"
        self.reaction_temperature = 373.15  # K
    
    def calculate(self, args=None):
        """计算方法"""
        # 添加具体计算逻辑
        self.calculated = True

# 注册自定义操作
register_custom_operation("CustomReactor", CustomReactor)

# 使用自定义操作
solver = create_integrated_solver()
reactor = solver.create_and_add_operation("CustomReactor", "R-001")
```

### 3. 流程图计算

```python
# 创建完整流程
solver = create_integrated_solver()

# 添加单元操作
mixer = solver.create_and_add_operation("Mixer", "MIX-001", "混合器")
heater = solver.create_and_add_operation("Heater", "HX-001", "加热器") 
pump = solver.create_and_add_operation("Pump", "P-001", "输送泵")

# 计算所有操作
results = solver.calculate_all_operations()

# 查看结果
for op_name, success in results.items():
    status = "成功" if success else "失败"
    print(f"{op_name}: {status}")
```

## 支持的单元操作

### 🔄 混合与分离
- **Mixer** - 混合器：多股物料流混合
- **Splitter** - 分离器：单股物料流分离

### 🌡️ 传热设备  
- **Heater** - 加热器：物料流加热
- **Cooler** - 冷却器：物料流冷却
- **HeatExchanger** - 热交换器：流体间热交换

### 💨 流体机械
- **Pump** - 泵：液体增压
- **Compressor** - 压缩机：气体压缩
- **Valve** - 阀门：压力降低

### 🧪 分离设备
- **ComponentSeparator** - 组分分离器：按组分分离
- **Filter** - 过滤器：固液分离
- **Vessel** - 容器：相分离和储存
- **Tank** - 储罐：物料储存

## API 参考

### 基础类

#### UnitOpBaseClass
所有单元操作的基础类，提供：
- 基本属性管理（名称、标签、描述）
- 计算状态跟踪
- 图形对象管理
- 属性包引用
- 调试功能

主要方法：
- `calculate(args)` - 执行计算（抽象方法）
- `solve()` - 求解包装器，包含状态管理
- `validate()` - 验证操作有效性
- `get_debug_report()` - 获取调试报告

#### SimulationObjectClass
单元操作分类枚举：
- `Streams` - 物料流和能量流
- `MixersSplitters` - 混合器和分离器
- `HeatExchangers` - 传热设备
- `PressureChangers` - 流体机械
- `SeparationEquipment` - 分离设备
- `Reactors` - 反应器
- `Logical` - 逻辑操作

### 集成求解器

#### IntegratedFlowsheetSolver
扩展原有 FlowsheetSolver 的集成求解器：

```python
# 创建求解器
solver = IntegratedFlowsheetSolver(settings)

# 操作管理
solver.add_unit_operation(operation, flowsheet)
solver.remove_unit_operation(operation_name)
solver.create_and_add_operation(type, name, description)

# 计算功能
solver.calculate_unit_operation(operation_name)
solver.calculate_all_operations(in_dependency_order=True)

# 查询功能
solver.get_operation_by_name(name)
solver.get_operations_by_type(operation_type)
solver.get_calculation_summary()

# 配置管理
config = solver.export_operations_config()
solver.import_operations_config(config)

# 验证和重置
errors = solver.validate_all_operations()
solver.reset_all_calculations()
```

### 具体单元操作

#### Mixer（混合器）
```python
mixer = Mixer(name="MIX-001", description="原料混合器")

# 设置压力计算模式
mixer.pressure_calculation = PressureBehavior.MINIMUM  # 最小值
mixer.pressure_calculation = PressureBehavior.MAXIMUM  # 最大值  
mixer.pressure_calculation = PressureBehavior.AVERAGE  # 平均值

# 连接点信息
print(f"输入连接点: {len(mixer.graphic_object.input_connectors)}")   # 6个
print(f"输出连接点: {len(mixer.graphic_object.output_connectors)}")  # 1个
```

#### Heater（加热器）
```python
heater = Heater(name="HX-001", description="预热器")

# 设置操作参数
heater.outlet_temperature = 373.15  # 出口温度 [K]
heater.heat_duty = 1000.0           # 热负荷 [kW]
heater.calculation_mode = "OutletTemperature"  # 计算模式
```

## 测试

运行完整测试套件：

```bash
cd OpenAspen
python tests/test_dwsim_operations.py
```

测试内容包括：
- 基础类功能测试
- 所有单元操作创建和计算测试
- 集成求解器功能测试
- 错误处理测试
- 性能测试

## 示例

查看完整使用示例：

```bash
cd OpenAspen
python examples/integrated_solver_example.py
```

示例包括：
1. 基本单元操作创建和使用
2. 混合器详细计算示例
3. 操作注册表使用
4. 配置导出和导入
5. 错误处理演示
6. 自定义操作创建

## 与 FlowsheetSolver 的集成

### 完美衔接
- 继承原有 FlowsheetSolver 的所有功能
- 添加单元操作管理和计算能力
- 保持原有事件系统和异常处理
- 兼容现有计算参数和状态管理

### 事件处理
```python
def on_calculation_started(obj_name):
    print(f"开始计算: {obj_name}")

def on_calculation_finished(obj_name, success):
    print(f"计算完成: {obj_name}, 结果: {'成功' if success else '失败'}")

# 注册事件处理器
solver.add_event_handler('unit_op_calculation_started', on_calculation_started)
solver.add_event_handler('unit_op_calculation_finished', on_calculation_finished)
```

### 计算参数传递
```python
from flowsheet_solver.calculation_args import CalculationArgs, ObjectType

# 创建计算参数
calc_args = CalculationArgs(
    name="MIX-001",
    object_type=ObjectType.UNITOPERATION,
    sender="FlowsheetSolver"
)

# 执行计算
success = solver.calculate_unit_operation("MIX-001", calc_args)
```

## 开发指南

### 添加新的单元操作

1. **继承基础类**：
```python
from dwsim_operations.base_classes import UnitOpBaseClass

class NewOperation(UnitOpBaseClass):
    def __init__(self, name="", description=""):
        super().__init__()
        # 设置属性
        
    def calculate(self, args=None):
        # 实现计算逻辑
        pass
```

2. **注册操作**：
```python
from dwsim_operations.integration import register_custom_operation
register_custom_operation("NewOperation", NewOperation)
```

3. **添加测试**：
在 `tests/test_dwsim_operations.py` 中添加相应测试。

### 代码风格
- 使用类型提示
- 添加详细的中文文档字符串
- 遵循 PEP 8 代码规范
- 包含完整的异常处理

### 调试功能
```python
# 启用调试模式
operation.debug_mode = True

# 获取调试报告
debug_report = operation.get_debug_report()
print(debug_report)
```

## 性能考虑

- 大量操作时使用批量操作方法
- 适当使用缓存避免重复计算
- 异步计算支持（继承自 FlowsheetSolver）
- 内存使用优化

## 故障排除

### 常见问题

1. **导入错误**：
   - 确保路径正确设置
   - 检查依赖模块是否存在

2. **计算失败**：
   - 检查连接是否正确设置
   - 验证属性包是否配置
   - 查看错误消息和调试报告

3. **性能问题**：
   - 减少不必要的计算
   - 使用批量操作
   - 检查内存使用情况

### 日志配置
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 许可证

本项目遵循与 DWSIM 相同的开源许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南
1. Fork 项目
2. 创建功能分支
3. 添加测试
4. 确保所有测试通过
5. 提交 Pull Request

## 更新日志

### v1.0.0
- 完整实现所有基础单元操作
- 与 FlowsheetSolver 完美集成
- 包含完整的测试套件和示例
- 从原 VB.NET 版本 1:1 转换完成 