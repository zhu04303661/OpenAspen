# DWSIM 测试系统使用指南

## 概述

本指南详细介绍了整理后的DWSIM测试系统的使用方法。系统已完全重构为更清晰、更高效的结构。

## 🎯 核心概念

### 统一测试系统
所有DWSIM单元操作测试现在整合在一个文件中：`unified/test_dwsim_unified.py`

### 标记系统
使用pytest标记进行灵活的测试分类和过滤：

```bash
# 架构层级标记
foundation      # 基础框架测试
basic_ops       # 基本单元操作
advanced        # 高级功能
integration     # 集成测试

# 系统模块标记  
reactors        # 反应器系统
logical         # 逻辑模块
solver          # 求解器测试
cape_open       # CAPE-OPEN集成
validation      # 验证调试

# 设备标记
mixer           # 混合器
heater          # 加热器
pump            # 泵
heat_exchanger  # 热交换器
valve           # 阀门
splitter        # 分离器

# 类型标记
unit            # 单元测试
performance     # 性能测试
smoke           # 冒烟测试
fast            # 快速测试
slow            # 慢速测试
```

## 🚀 基本使用

### 安装依赖

```bash
# 基础依赖
pip install pytest

# 增强功能（可选）
pip install pytest-xdist pytest-cov pytest-html
```

### 进入测试目录

```bash
cd OpenAspen/tests
```

### 基本命令

```bash
# 运行所有测试
python run_unified_pytest.py

# 快速验证（冒烟测试）
python run_unified_pytest.py --smoke

# 快速测试（排除慢速）
python run_unified_pytest.py --quick

# 性能测试
python run_unified_pytest.py --performance
```

## 🔍 测试选择

### 按标记选择

```bash
# 运行特定标记的测试
python run_unified_pytest.py --markers foundation
python run_unified_pytest.py --markers mixer heater

# 排除特定标记
python run_unified_pytest.py --exclude slow
python run_unified_pytest.py --exclude slow performance
```

### 按组件选择

```bash
# 基础框架
python run_unified_pytest.py --component foundation

# 混合器
python run_unified_pytest.py --component mixer

# 加热器  
python run_unified_pytest.py --component heater

# 求解器
python run_unified_pytest.py --component solver
```

## 📊 高级功能

### 并行执行

```bash
# 自动并行（需要pytest-xdist）
python run_unified_pytest.py --parallel

# 组合使用
python run_unified_pytest.py --quick --parallel
```

### 覆盖率分析

```bash
# 生成覆盖率报告（需要pytest-cov）
python run_unified_pytest.py --coverage

# 覆盖率 + HTML报告
python run_unified_pytest.py --coverage --html-report
```

### HTML报告

```bash
# 生成HTML测试报告（需要pytest-html）
python run_unified_pytest.py --html-report

# 报告保存在 reports/html/ 目录
```

### 失败控制

```bash
# 最多失败5个测试后停止
python run_unified_pytest.py --maxfail 5

# 第一个失败后停止
python run_unified_pytest.py --maxfail 1
```

## 📋 信息查询

### 查看可用标记

```bash
python run_unified_pytest.py --list-marks
```

输出示例：
```
📋 可用的测试标记:
----------------------------------------

架构层级:
  foundation      - 基础框架测试
  basic_ops       - 基本单元操作测试
  advanced        - 高级单元操作测试
  integration     - 集成测试
  
具体设备:
  mixer           - 混合器测试
  heater          - 加热器测试
  pump            - 泵测试
  ...
```

### 收集测试统计

```bash
python run_unified_pytest.py --collect
```

输出示例：
```
📝 收集测试用例...
✅ 测试收集成功
📊 发现 36 个测试用例

测试类别分布:
  TestDWSIMFoundations: 3个测试
  TestBasicUnitOperations: 7个测试
  TestSolverIntegration: 3个测试
  ...
```

### 查看帮助

```bash
python run_unified_pytest.py --help
```

## 🔧 高级用法示例

### 开发时测试

```bash
# 快速验证代码修改
python run_unified_pytest.py --smoke

# 测试特定修改的组件
python run_unified_pytest.py --component mixer

# 详细输出调试
python run_unified_pytest.py --component mixer -v
```

### CI/CD集成

```bash
# 完整回归测试
python run_unified_pytest.py --quick --parallel

# 生成报告
python run_unified_pytest.py --coverage --html-report --maxfail 10

# 快速验证
python run_unified_pytest.py --smoke --maxfail 1
```

### 性能监控

```bash
# 性能基准测试
python run_unified_pytest.py --performance

# 大型流程图测试
python run_unified_pytest.py --markers slow

# 快速性能检查
python run_unified_pytest.py --markers fast
```

## 📁 输出文件位置

### 测试报告
```
reports/
├── coverage/           # 覆盖率报告
│   ├── index.html     # 覆盖率主页
│   └── ...
├── html/              # HTML测试报告
│   ├── dwsim_unified_test_report_*.html
│   └── ...
└── logs/              # 测试日志
    └── pytest.log
```

### 访问报告

```bash
# 查看覆盖率报告
open reports/coverage/index.html

# 查看HTML测试报告
open reports/html/dwsim_unified_test_report_*.html
```

## 🛠️ 自定义配置

### 临时修改设置

```bash
# 自定义pytest参数
python run_unified_pytest.py --quick -- --tb=long

# 增加详细程度
python run_unified_pytest.py --smoke -- -vv

# 显示本地变量
python run_unified_pytest.py --markers foundation -- -l
```

### 环境变量

```bash
# 设置最大并行数
export PYTEST_XDIST_WORKERS=4
python run_unified_pytest.py --parallel

# 禁用警告
export PYTHONWARNINGS=ignore
python run_unified_pytest.py
```

## 🔍 调试技巧

### 测试失败调试

```bash
# 显示失败的详细信息
python run_unified_pytest.py --markers foundation -- --tb=long

# 进入调试器
python run_unified_pytest.py --markers foundation -- --pdb

# 显示最慢的测试
python run_unified_pytest.py -- --durations=0
```

### 日志调试

```bash
# 启用日志输出
python run_unified_pytest.py --markers foundation -- --log-cli-level=DEBUG

# 查看日志文件
tail -f reports/logs/pytest.log
```

### 跳过测试分析

```bash
# 显示跳过的测试
python run_unified_pytest.py -- -rs

# 显示跳过原因
python run_unified_pytest.py -- -rsx
```

## 📈 性能优化

### 并行测试

```bash
# 自动检测CPU核心数
python run_unified_pytest.py --parallel

# 手动指定进程数
python run_unified_pytest.py -- -n 4
```

### 测试选择优化

```bash
# 只运行快速测试
python run_unified_pytest.py --markers fast

# 排除耗时测试
python run_unified_pytest.py --exclude slow performance

# 渐进测试策略
python run_unified_pytest.py --smoke     # 先运行冒烟测试
python run_unified_pytest.py --quick     # 再运行快速测试
python run_unified_pytest.py             # 最后完整测试
```

## 🚨 常见问题

### 1. 测试文件找不到

**问题**: `❌ 统一测试文件不存在`

**解决**:
```bash
# 确保在正确目录
cd OpenAspen/tests

# 检查文件存在
ls -la unified/test_dwsim_unified.py
```

### 2. 模块导入失败

**问题**: `警告：无法导入dwsim_operations模块`

**解决**:
```bash
# 检查Python路径
python -c "import sys; print(sys.path)"

# 检查模块
python -c "from dwsim_operations import *; print('导入成功')"
```

### 3. pytest插件缺失

**问题**: `⚠️ pytest-xdist未安装，无法并行执行`

**解决**:
```bash
# 安装所需插件
pip install pytest-xdist pytest-cov pytest-html
```

### 4. 权限问题

**问题**: 无法创建报告目录

**解决**:
```bash
# 创建必要目录
mkdir -p reports/{coverage,html,logs}

# 检查权限
ls -la reports/
```

## 📚 扩展阅读

- `../README.md` - 主要文档
- `DEVELOPMENT.md` - 开发指南
- `../legacy/` - 历史版本参考
- `../unified/` - 当前测试实现

## 💡 最佳实践

1. **开发时**: 使用 `--smoke` 快速验证
2. **调试时**: 使用特定组件和详细输出
3. **CI/CD**: 使用 `--quick --parallel` 平衡速度和覆盖率
4. **发布前**: 运行完整测试套件
5. **性能监控**: 定期运行 `--performance` 测试

---

这个指南涵盖了统一测试系统的所有主要功能。有任何问题请参考其他文档或使用 `--help` 查看最新选项！ 