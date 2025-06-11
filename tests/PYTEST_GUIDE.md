# DWSIM 单元操作测试 pytest 使用指南

## 概述

本指南介绍如何使用 pytest 框架运行 DWSIM 单元操作测试。pytest 提供了更强大的测试管理、标记系统和报告功能。

## 文件结构

```
OpenAspen/tests/
├── pytest.ini                           # pytest配置文件
├── conftest_dwsim.py                    # DWSIM测试专用fixtures
├── test_dwsim_operations_pytest.py     # 完整功能测试（pytest版本）
├── test_specific_operations_pytest.py   # 具体操作测试（pytest版本）
├── run_pytest_dwsim.py                 # pytest测试运行器
├── PYTEST_GUIDE.md                     # 本文档
└── reports/                             # 测试报告目录
```

## 快速开始

### 1. 安装依赖

```bash
# 基础pytest
pip install pytest

# 可选的pytest插件
pip install pytest-html pytest-cov pytest-xdist pytest-mock
```

### 2. 运行所有测试

```bash
cd OpenAspen/tests

# 方式1: 直接使用pytest
pytest -v

# 方式2: 使用专用运行器
python run_pytest_dwsim.py

# 方式3: 使用可执行脚本
./run_pytest_dwsim.py
```

## 测试标记系统

### 可用标记

| 标记 | 描述 | 示例用法 |
|------|------|----------|
| `foundation` | 基础框架测试 | `pytest -m foundation` |
| `basic_ops` | 基本单元操作测试 | `pytest -m basic_ops` |
| `reactors` | 反应器系统测试 | `pytest -m reactors` |
| `logical` | 逻辑模块测试 | `pytest -m logical` |
| `advanced` | 高级单元操作测试 | `pytest -m advanced` |
| `cape_open` | CAPE-OPEN集成测试 | `pytest -m cape_open` |
| `solver` | 求解器测试 | `pytest -m solver` |
| `validation` | 验证调试测试 | `pytest -m validation` |
| `mixer` | 混合器测试 | `pytest -m mixer` |
| `heater` | 加热器测试 | `pytest -m heater` |
| `pump` | 泵测试 | `pytest -m pump` |
| `heat_exchanger` | 热交换器测试 | `pytest -m heat_exchanger` |
| `valve` | 阀门测试 | `pytest -m valve` |
| `splitter` | 分离器测试 | `pytest -m splitter` |
| `integration` | 集成测试 | `pytest -m integration` |
| `performance` | 性能测试 | `pytest -m performance` |
| `unit` | 单元测试 | `pytest -m unit` |
| `smoke` | 冒烟测试 | `pytest -m smoke` |
| `slow` | 慢速测试 | `pytest -m slow` |
| `fast` | 快速测试 | `pytest -m fast` |

### 标记使用示例

```bash
# 运行混合器和加热器测试
pytest -m "mixer or heater"

# 运行除了慢速测试之外的所有测试
pytest -m "not slow"

# 运行基本操作但排除性能测试
pytest -m "basic_ops and not performance"

# 运行特定组件的快速测试
pytest -m "mixer and fast"
```

## 专用运行器使用

### 基本用法

```bash
# 查看帮助
./run_pytest_dwsim.py --help

# 列出所有可用标记
./run_pytest_dwsim.py --list-marks

# 运行所有测试
./run_pytest_dwsim.py

# 运行快速测试（排除slow标记）
./run_pytest_dwsim.py --quick

# 运行性能测试
./run_pytest_dwsim.py --performance

# 运行冒烟测试
./run_pytest_dwsim.py --smoke
```

### 选择测试套件

```bash
# 运行完整功能测试
./run_pytest_dwsim.py --suite comprehensive

# 运行具体操作测试
./run_pytest_dwsim.py --suite specific

# 运行原始测试
./run_pytest_dwsim.py --suite original
```

### 按标记过滤

```bash
# 包含特定标记
./run_pytest_dwsim.py --markers mixer heater

# 排除特定标记
./run_pytest_dwsim.py --exclude slow performance

# 组合使用
./run_pytest_dwsim.py --markers basic_ops --exclude slow
```

### 按组件运行

```bash
# 运行混合器组件测试
./run_pytest_dwsim.py --component mixer

# 运行反应器组件测试
./run_pytest_dwsim.py --component reactors

# 运行求解器组件测试
./run_pytest_dwsim.py --component solver
```

## 高级功能

### 并行执行

```bash
# 自动并行（需要pytest-xdist）
./run_pytest_dwsim.py --parallel

# 指定进程数
pytest -n 4

# 按CPU核心数并行
pytest -n auto
```

### 覆盖率分析

```bash
# 生成覆盖率报告（需要pytest-cov）
./run_pytest_dwsim.py --coverage

# 或直接使用pytest
pytest --cov=dwsim_operations --cov-report=html
```

### HTML报告

```bash
# 生成HTML报告（需要pytest-html）
./run_pytest_dwsim.py --html-report

# 或直接使用pytest
pytest --html=report.html --self-contained-html
```

### 失败控制

```bash
# 最多失败5个测试后停止
./run_pytest_dwsim.py --maxfail 5

# 第一个失败后停止
pytest -x

# 最后失败的测试优先运行
pytest --lf

# 失败和通过的测试都运行
pytest --ff
```

## 测试输出控制

### 详细程度

```bash
# 安静模式
pytest -q

# 详细模式
pytest -v

# 非常详细模式
pytest -vv

# 显示所有输出
pytest -s
```

### 进度显示

```bash
# 显示测试进度百分比
pytest --tb=short --quiet --disable-warnings

# 显示最慢的10个测试
pytest --durations=10

# 显示所有测试时间
pytest --durations=0
```

## 调试和开发

### 调试失败的测试

```bash
# 详细的错误信息
pytest --tb=long

# 进入pdb调试器
pytest --pdb

# 只运行失败的测试
pytest --lf

# 运行特定测试
pytest -k "test_mixer_pressure"
```

### 测试收集

```bash
# 只收集测试，不运行
pytest --collect-only

# 显示测试收集的详细信息
pytest --collect-only -q
```

## 配置文件

### pytest.ini 配置

```ini
[tool:pytest]
# 测试目录
testpaths = .

# 测试文件模式
python_files = test_*.py

# 测试类模式
python_classes = Test*

# 测试函数模式
python_functions = test_*

# 默认选项
addopts = -v --tb=short --strict-markers

# 标记定义
markers =
    mixer: 混合器测试
    heater: 加热器测试
    # ... 其他标记
```

### conftest.py fixtures

主要的测试设备（fixtures）：

- `disable_logging`: 禁用日志噪音
- `integrated_solver`: 集成求解器实例
- `sample_mixer`: 测试用混合器
- `sample_heater`: 测试用加热器
- `sample_pump`: 测试用泵
- `sample_mixer_data`: 混合器测试数据
- `sample_heater_data`: 加热器测试数据
- `calculation_error_threshold`: 计算误差阈值
- `performance_timer`: 性能计时器

## 最佳实践

### 1. 测试组织

```python
@pytest.mark.mixer
class TestMixerDetailedFunctionality:
    """混合器详细功能测试"""
    
    def test_pressure_calculation_minimum_mode(self, sample_mixer, sample_mixer_data):
        """测试最小压力计算模式"""
        # 测试实现
        pass
```

### 2. 参数化测试

```python
@pytest.mark.parametrize("pressure_mode", [
    PressureBehavior.MINIMUM,
    PressureBehavior.MAXIMUM,
    PressureBehavior.AVERAGE
])
def test_mixer_pressure_modes(sample_mixer, pressure_mode):
    """参数化测试混合器压力模式"""
    sample_mixer.pressure_calculation = pressure_mode
    assert sample_mixer.pressure_calculation == pressure_mode
```

### 3. 错误处理测试

```python
def test_connection_validation(self, sample_mixer):
    """测试连接验证"""
    with pytest.raises(ValueError, match="混合器必须连接输出物料流"):
        sample_mixer._validate_connections()
```

### 4. 性能测试

```python
@pytest.mark.performance
@pytest.mark.slow
def test_large_mixer_performance(self, integrated_solver, performance_timer):
    """测试大量混合器的性能"""
    performance_timer.start()
    # 性能测试实现
    assert performance_timer.stop() < 1.0  # 1秒内完成
```

## 持续集成

### GitHub Actions 示例

```yaml
name: DWSIM Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install pytest pytest-cov pytest-html
        pip install -r requirements.txt
    
    - name: Run quick tests
      run: |
        cd OpenAspen/tests
        ./run_pytest_dwsim.py --quick --coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 故障排除

### 常见问题

1. **ImportError**: 确保项目路径正确添加到 `sys.path`
2. **标记未定义**: 检查 `pytest.ini` 中的标记定义
3. **Fixture未找到**: 确保 `conftest.py` 在正确位置
4. **测试收集失败**: 检查测试文件命名和类/函数命名规范

### 调试命令

```bash
# 检查pytest版本
pytest --version

# 检查配置
pytest --markers

# 详细的错误跟踪
pytest --tb=long -vv

# 显示所有输出
pytest -s --capture=no
```

## 总结

使用 pytest 框架为 DWSIM 单元操作测试提供了以下优势：

1. **强大的标记系统**: 灵活的测试过滤和组织
2. **丰富的插件生态**: 覆盖率、HTML报告、并行执行等
3. **参数化测试**: 简化重复测试用例
4. **优秀的错误报告**: 清晰的失败信息
5. **易于集成**: 与CI/CD系统完美集成
6. **fixtures系统**: 共享测试设备和数据

通过 pytest 框架，确保了 DWSIM 单元操作 Python 实现与 VB.NET 版本的完全一致性和可靠性。 