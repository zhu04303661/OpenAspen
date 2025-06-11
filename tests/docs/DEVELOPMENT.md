# DWSIM 测试系统开发指南

## 概述

本指南为DWSIM测试系统的开发者和维护者提供详细的技术文档和最佳实践。

## 🏗️ 系统架构

### 目录结构
```
tests/
├── pytest.ini                    # pytest全局配置
├── conftest.py                   # 全局fixtures
├── run_unified_pytest.py         # 统一运行器
├── 
├── unified/                      # 统一测试实现
│   ├── test_dwsim_unified.py     # 主测试文件
│   └── conftest_dwsim.py        # DWSIM专用fixtures
├── 
├── legacy/                       # 历史版本保留
│   ├── unittest/                # 原unittest实现
│   ├── pytest_old/             # 旧pytest实现  
│   └── runners/                 # 旧运行器
├── 
├── data/                         # 测试数据
├── reports/                      # 测试报告
└── docs/                         # 文档
```

### 核心组件

#### 1. 统一测试文件 (`unified/test_dwsim_unified.py`)
- **887行代码**，**36个测试用例**
- 11个测试类，按功能模块组织
- 完整的pytest标记系统
- 自适应环境检测

#### 2. 统一运行器 (`run_unified_pytest.py`) 
- **533行代码**，功能丰富的CLI
- 支持标记过滤、组件测试、并行执行
- 集成覆盖率分析和报告生成
- 智能错误处理和恢复

#### 3. 配置系统 (`pytest.ini`, `conftest.py`)
- 全局pytest配置
- 测试环境设置
- 日志和报告配置

## 🔧 开发环境设置

### 依赖安装

```bash
# 基础开发环境
pip install pytest pytest-xdist pytest-cov pytest-html

# 开发工具
pip install black flake8 mypy pre-commit

# DWSIM依赖
pip install numpy scipy pandas matplotlib
```

### 环境配置

```bash
# 设置Python路径
export PYTHONPATH="/path/to/dwsim5/OpenAspen:$PYTHONPATH"

# 开发模式
export DWSIM_DEBUG=1
export PYTEST_VERBOSE=1
```

## 📝 添加新测试

### 1. 基本测试结构

```python
@pytest.mark.your_mark
class TestYourFeature:
    """
    您的功能测试
    
    验证：
    1. 基本功能
    2. 边界条件
    3. 错误处理
    """
    
    @pytest.mark.skipif(not DWSIM_AVAILABLE, reason="DWSIM模块不可用")
    def test_your_function(self):
        """测试您的功能"""
        try:
            # 创建测试环境
            solver = create_integrated_solver()
            
            # 执行测试
            result = your_function()
            
            # 验证结果
            assert result is not None
            assert result.some_property == expected_value
            
        except Exception as e:
            pytest.skip(f"无法创建测试环境: {e}")
```

### 2. 标记使用

```python
# 功能标记
@pytest.mark.foundation      # 基础框架
@pytest.mark.basic_ops       # 基本操作
@pytest.mark.advanced        # 高级功能

# 设备标记
@pytest.mark.mixer          # 混合器
@pytest.mark.heater         # 加热器
@pytest.mark.pump           # 泵

# 类型标记  
@pytest.mark.unit           # 单元测试
@pytest.mark.integration    # 集成测试
@pytest.mark.performance    # 性能测试
@pytest.mark.smoke          # 冒烟测试

# 速度标记
@pytest.mark.fast           # 快速测试
@pytest.mark.slow           # 慢速测试
```

### 3. 参数化测试

```python
@pytest.mark.parametrize("input_value,expected", [
    (100, 200),
    (150, 300),
    (200, 400),
])
@pytest.mark.skipif(not DWSIM_AVAILABLE, reason="DWSIM模块不可用")
def test_parametrized_function(input_value, expected):
    """参数化测试示例"""
    result = your_function(input_value)
    assert result == expected
```

### 4. 数值验证测试

```python
def test_numerical_calculation(self):
    """数值计算验证"""
    # 设置测试数据
    inlet_conditions = {
        "temperature": 298.15,  # K
        "pressure": 101325.0,   # Pa
        "mass_flow": 1000.0     # kg/h
    }
    
    # 执行计算
    result = calculate_something(inlet_conditions)
    
    # 验证数值精度
    assert abs(result.outlet_temperature - 373.15) < 1e-6
    assert abs(result.heat_duty - 50000.0) < 1.0
```

## 🔍 测试设计原则

### 1. 独立性
每个测试应该独立运行，不依赖其他测试的结果：

```python
def test_independent_function(self):
    """独立测试示例"""
    # 创建自己的测试环境
    solver = create_integrated_solver()
    operation = solver.create_and_add_operation("Mixer", "MIX-001")
    
    # 执行测试
    operation.calculate()
    
    # 验证结果 - 不依赖其他测试
    assert operation.calculated
```

### 2. 可重复性
测试结果应该是确定的和可重复的：

```python
def test_repeatable_calculation(self):
    """可重复测试示例"""
    # 使用固定的输入数据
    input_data = {"value": 100.0}
    
    # 多次执行应该得到相同结果
    result1 = calculate_function(input_data)
    result2 = calculate_function(input_data)
    
    assert result1 == result2
```

### 3. 自我验证
测试应该自动验证结果，不需要手动检查：

```python
def test_self_validating(self):
    """自我验证测试示例"""
    mixer = create_mixer()
    
    # 设置输入流
    mixer.add_inlet_stream(flow=100, temperature=300)
    mixer.add_inlet_stream(flow=150, temperature=350)
    
    # 执行计算
    mixer.calculate()
    
    # 自动验证质量守恒
    total_inlet = 100 + 150
    assert abs(mixer.outlet_flow - total_inlet) < 1e-6
    
    # 自动验证能量平衡
    expected_temp = (100*300 + 150*350) / (100 + 150)
    assert abs(mixer.outlet_temperature - expected_temp) < 0.1
```

## 🏷️ 标记系统扩展

### 添加新标记

1. **在运行器中注册**:
```python
# run_unified_pytest.py
self.available_marks = [
    # ... 现有标记
    "your_new_mark",  # 新标记
]
```

2. **在配置中描述**:
```python
# run_unified_pytest.py
mark_descriptions = {
    # ... 现有描述
    "your_new_mark": "您的新标记描述",
}
```

3. **在组件映射中添加**:
```python
# run_unified_pytest.py  
component_markers = {
    # ... 现有组件
    "your_component": ["your_new_mark"],
}
```

### 标记最佳实践

- 使用描述性名称
- 保持一致的命名约定
- 避免标记过度细分
- 文档化标记用途

## 🧪 Fixtures 开发

### 创建新的Fixture

```python
# unified/conftest_dwsim.py

@pytest.fixture(scope="session")
def your_session_fixture():
    """会话级别的fixture"""
    setup_data = expensive_setup()
    yield setup_data
    cleanup(setup_data)

@pytest.fixture(scope="function")  
def your_function_fixture():
    """函数级别的fixture"""
    return create_test_data()

@pytest.fixture(params=[1, 2, 3])
def parametrized_fixture(request):
    """参数化fixture"""
    return create_object(request.param)
```

### Fixture 最佳实践

- 使用适当的作用域 (`session`, `module`, `class`, `function`)
- 清理资源 (使用 `yield` 而不是 `return`)
- 避免fixture之间的依赖
- 提供有意义的默认值

## 📊 性能测试

### 性能基准测试

```python
@pytest.mark.performance
@pytest.mark.slow
def test_performance_benchmark(self):
    """性能基准测试"""
    import time
    
    # 性能测试设置
    large_data = generate_large_dataset(size=10000)
    
    # 测量执行时间
    start_time = time.time()
    result = process_large_data(large_data)
    execution_time = time.time() - start_time
    
    # 验证性能要求
    assert execution_time < 10.0  # 10秒以内
    assert len(result) == 10000
    
    # 记录性能指标
    print(f"执行时间: {execution_time:.2f}秒")
    print(f"处理速度: {len(result)/execution_time:.0f} 项/秒")
```

### 内存使用测试

```python
@pytest.mark.memory
def test_memory_usage(self):
    """内存使用测试"""
    import psutil
    import os
    
    # 获取初始内存使用
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # 执行操作
    large_objects = [create_large_object() for _ in range(1000)]
    
    # 检查内存增长
    current_memory = process.memory_info().rss
    memory_growth = current_memory - initial_memory
    
    # 验证内存使用合理
    assert memory_growth < 100 * 1024 * 1024  # 100MB限制
    
    # 清理
    del large_objects
```

## 🔧 调试技巧

### 1. 测试调试

```python
def test_with_debugging(self):
    """带调试的测试"""
    # 设置调试断点
    import pdb; pdb.set_trace()
    
    # 详细日志
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # 执行测试
    result = your_function()
    
    # 调试输出
    print(f"调试信息: {result}")
```

### 2. 条件调试

```python
import os

DEBUG = os.getenv('DWSIM_DEBUG', '0') == '1'

def test_conditional_debug(self):
    """条件调试测试"""
    if DEBUG:
        print(f"调试模式启用")
        
    result = your_function()
    
    if DEBUG:
        print(f"结果: {result}")
```

### 3. 失败时的状态保存

```python
def test_with_state_saving(self):
    """保存失败状态的测试"""
    try:
        operation = create_operation()
        operation.calculate()
        assert operation.calculated
        
    except Exception as e:
        # 保存失败状态
        import json
        state = {
            "error": str(e),
            "operation_state": operation.__dict__,
            "timestamp": str(datetime.now())
        }
        
        with open("debug_state.json", "w") as f:
            json.dump(state, f, indent=2)
            
        raise  # 重新抛出异常
```

## 📈 持续集成

### GitHub Actions 配置

```yaml
# .github/workflows/tests.yml
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
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        pip install pytest pytest-xdist pytest-cov pytest-html
        pip install -r requirements.txt
        
    - name: Run smoke tests
      run: |
        cd OpenAspen/tests
        python run_unified_pytest.py --smoke
        
    - name: Run full test suite
      run: |
        cd OpenAspen/tests  
        python run_unified_pytest.py --quick --parallel --coverage
        
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### 测试策略

1. **Pull Request**: 冒烟测试 + 快速测试
2. **主分支推送**: 完整测试套件
3. **发布前**: 完整测试 + 性能测试
4. **定期**: 性能基准测试

## 🔍 代码质量

### 代码风格

```bash
# 使用 black 格式化
black unified/test_dwsim_unified.py

# 使用 flake8 检查
flake8 unified/ --max-line-length=100

# 使用 mypy 类型检查
mypy unified/test_dwsim_unified.py
```

### 测试覆盖率

```bash
# 生成覆盖率报告
python run_unified_pytest.py --coverage

# 查看覆盖率详情
coverage report -m

# 查看HTML报告
open reports/coverage/index.html
```

### 代码审查清单

- [ ] 测试独立性
- [ ] 适当的异常处理
- [ ] 数值精度验证
- [ ] 性能要求满足
- [ ] 文档字符串完整
- [ ] 标记使用正确
- [ ] 清理资源

## 📚 扩展开发

### 添加新的测试运行器功能

```python
# run_unified_pytest.py

def run_custom_feature(self) -> int:
    """添加自定义功能"""
    print("🔧 运行自定义功能")
    
    # 实现您的功能
    # ...
    
    return 0

# 在main()函数中添加命令行选项
parser.add_argument(
    "--custom-feature",
    action="store_true",
    help="运行自定义功能"
)

# 在main()函数中处理选项
if args.custom_feature:
    return runner.run_custom_feature()
```

### 集成新的测试工具

```python
# 添加新的报告格式
def generate_custom_report(self, test_results):
    """生成自定义报告"""
    report_data = {
        "summary": test_results.summary,
        "details": test_results.details,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("custom_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
```

## 🚨 故障排除

### 常见开发问题

1. **导入错误**
   - 检查 PYTHONPATH
   - 验证模块结构

2. **测试超时** 
   - 增加超时限制
   - 优化测试逻辑

3. **内存泄漏**
   - 使用适当的作用域
   - 清理大型对象

4. **并发问题**
   - 避免共享状态
   - 使用适当的锁机制

### 调试命令

```bash
# 详细调试
python run_unified_pytest.py --markers foundation -- -vv --tb=long

# 性能分析
python run_unified_pytest.py -- --durations=0

# 内存分析  
python -m memory_profiler run_unified_pytest.py --smoke
```

## 📋 发布检查清单

发布新版本前的检查：

- [ ] 所有测试通过
- [ ] 覆盖率 > 85%
- [ ] 性能基准达标
- [ ] 文档更新
- [ ] 变更日志更新
- [ ] 版本号更新
- [ ] 兼容性测试

---

这个开发指南提供了全面的技术文档。遵循这些最佳实践可以确保测试系统的质量和可维护性！ 