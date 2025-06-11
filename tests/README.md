# DWSIM 测试系统

## 📁 目录结构

```
tests/
├── pytest.ini                    # pytest配置文件
├── conftest.py                   # 全局fixtures和配置
├── run_unified_pytest.py         # 统一测试运行器 🚀
├── README.md                     # 本文档
├── 
├── unified/                      # 🎯 统一测试（主要使用）
│   ├── test_dwsim_unified.py     # 所有DWSIM单元操作测试
│   └── conftest_dwsim.py        # DWSIM专用fixtures
│
├── legacy/                       # 📦 遗留测试（保留备份）
│   ├── unittest/                # 原unittest格式测试
│   ├── pytest_old/             # 旧的pytest测试
│   └── runners/                 # 旧的测试运行器
│
├── data/                         # 📊 测试数据
│   ├── fixtures/                # 测试数据文件
│   └── samples/                 # 示例数据
│
├── reports/                      # 📋 测试报告输出
│   ├── coverage/                # 覆盖率报告
│   ├── html/                    # HTML测试报告
│   └── logs/                    # 测试日志
│
└── docs/                         # 📚 详细文档
    ├── USAGE.md                 # 使用指南
    └── DEVELOPMENT.md           # 开发指南
```

## 🚀 快速开始

### 主要测试系统（推荐）

```bash
# 进入测试目录
cd OpenAspen/tests

# 运行所有测试
python run_unified_pytest.py

# 运行冒烟测试（快速验证）
python run_unified_pytest.py --smoke

# 运行特定组件测试
python run_unified_pytest.py --component mixer

# 查看所有可用选项
python run_unified_pytest.py --help
```

### 测试类型

| 测试标记 | 描述 | 示例命令 |
|---------|------|----------|
| `smoke` | 冒烟测试 - 快速验证基本功能 | `--smoke` |
| `fast` | 快速测试 - 执行迅速的测试 | `--markers fast` |
| `foundation` | 基础框架测试 | `--component foundation` |
| `mixer` | 混合器测试 | `--component mixer` |
| `heater` | 加热器测试 | `--component heater` |
| `performance` | 性能测试 | `--performance` |

## 📊 测试覆盖

### 统一测试系统覆盖范围

- **36个测试用例** - 覆盖所有DWSIM单元操作功能
- **11个测试类** - 按功能模块组织
- **20个测试标记** - 灵活的过滤和组织系统

### 主要测试模块

1. **基础框架** (`foundation`)
   - SimulationObjectClass枚举
   - UnitOpBaseClass基础结构
   - ConnectionPoint连接管理

2. **基本单元操作** (`basic_ops`)
   - 混合器、分离器、加热器、冷却器
   - 泵、压缩机、阀门、热交换器

3. **高级功能** (`advanced`)
   - 反应器系统
   - 逻辑模块
   - CAPE-OPEN集成

4. **系统集成** (`integration`)
   - 求解器集成
   - 完整流程图测试
   - 性能基准测试

## 🛠️ 配置文件

### pytest.ini
主要pytest配置，包含：
- 测试发现路径
- 默认执行选项
- 日志配置
- 警告过滤

### conftest.py
全局fixtures，提供：
- 基础测试环境
- 通用工具函数
- 测试数据管理

## 📈 性能基准

| 测试类型 | 测试数量 | 执行时间 | 内存使用 |
|---------|---------|---------|---------|
| 冒烟测试 | 3个 | ~2秒 | <100MB |
| 快速测试 | 35个 | ~2.5秒 | <200MB |
| 完整测试 | 36个 | ~15秒 | <500MB |
| 性能测试 | 2个 | ~10秒 | <1GB |

## 🔧 开发指南

### 添加新测试
1. 在 `unified/test_dwsim_unified.py` 中添加测试方法
2. 使用合适的pytest标记
3. 遵循现有的命名规范
4. 确保测试独立性

### 测试最佳实践
- 使用描述性的测试名称
- 添加适当的pytest标记
- 包含必要的文档字符串
- 确保测试可重复运行

### 调试测试
```bash
# 详细输出
python run_unified_pytest.py --markers foundation -v

# 收集测试信息
python run_unified_pytest.py --collect

# 生成覆盖率报告
python run_unified_pytest.py --coverage
```

## 📚 相关文档

- `docs/USAGE.md` - 详细使用指南
- `docs/DEVELOPMENT.md` - 开发和扩展指南
- `legacy/` - 旧版本测试系统（保留参考）

## 🎯 迁移说明

从旧测试系统迁移到统一系统：

### 旧系统 → 新系统
- `test_dwsim_operations_pytest.py` → `unified/test_dwsim_unified.py`
- `run_pytest_dwsim.py` → `run_unified_pytest.py`
- 分散的测试文件 → 统一的测试文件

### 优势
✅ 更简洁的结构  
✅ 统一的运行器  
✅ 更好的组织方式  
✅ 完整的文档  
✅ 灵活的测试过滤  

## 🆘 故障排除

### 常见问题

1. **找不到测试文件**
   ```
   确保运行器从 tests/ 目录执行
   检查 unified/test_dwsim_unified.py 是否存在
   ```

2. **模块导入失败**
   ```
   检查 PYTHONPATH 设置
   确认 dwsim_operations 模块可用
   ```

3. **fixtures 不可用**
   ```
   检查 conftest.py 和 unified/conftest_dwsim.py
   确认fixture名称正确
   ```

## 📞 支持

如有问题或需要帮助：
1. 查看 `docs/` 目录中的详细文档
2. 检查 `legacy/` 目录中的历史实现
3. 使用 `--help` 查看命令行选项

---

**推荐使用统一测试系统进行所有DWSIM单元操作测试！** 🎉 

# 总测试统计
📊 发现 61 个测试用例

# 核心功能验证
✅ FlowsheetSolver核心测试: 5 passed in 0.10s
✅ 计算参数系统测试: 3 passed in 0.22s  
✅ 冒烟测试: 3 passed in 0.10s
✅ 远程求解器测试: 1 passed, 2 skipped (正常) 