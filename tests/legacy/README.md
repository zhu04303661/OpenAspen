# Legacy 测试系统

## 概述

这个目录保存了DWSIM测试系统的历史版本，用于参考和向后兼容。

## 📁 目录结构

```
legacy/
├── unittest/           # 原始的unittest格式测试
│   └── unit/          # 单元测试文件
├── pytest_old/        # 旧的pytest测试实现
│   ├── integration/   # 集成测试
│   └── performance/   # 性能测试
└── runners/           # 历史测试运行器
    ├── run_all_dwsim_tests.py     # unittest运行器
    ├── run_pytest_dwsim.py       # 旧pytest运行器
    └── run_tests.py              # 通用测试运行器
```

## 🎯 用途

### 1. 参考实现
- 查看原始测试逻辑
- 理解历史设计决策
- 验证迁移正确性

### 2. 向后兼容
- 支持旧版本代码
- 渐进式迁移
- 应急备份

### 3. 测试对比
- 验证新旧实现一致性
- 性能对比基准
- 功能覆盖验证

## ⚠️ 重要说明

**这些是历史版本，不推荐用于新开发！**

请使用统一测试系统：
- 主测试文件：`../unified/test_dwsim_unified.py`
- 统一运行器：`../run_unified_pytest.py`

## 🔄 迁移状态

| 文件 | 状态 | 迁移到 |
|------|------|--------|
| `test_dwsim_operations_pytest.py` | ✅ 已迁移 | `unified/test_dwsim_unified.py` |
| `test_specific_operations_pytest.py` | ✅ 已整合 | `unified/test_dwsim_unified.py` |
| `run_pytest_dwsim.py` | ✅ 已替换 | `run_unified_pytest.py` |
| 各种unittest文件 | ✅ 已重写 | `unified/test_dwsim_unified.py` |

## 📚 使用方法

### 查看历史实现
```bash
# 查看unittest版本
cd legacy/unittest
python -m pytest unit/

# 查看旧pytest版本  
cd legacy/pytest_old
python -m pytest integration/ performance/

# 运行历史运行器
cd legacy/runners
python run_all_dwsim_tests.py
```

### 对比测试
```bash
# 新旧版本性能对比
cd legacy/runners
time python run_pytest_dwsim.py --performance

cd ../..
time python run_unified_pytest.py --performance
```

## 🗂️ 文件说明

### unittest/ 目录
保存原始的unittest格式测试，包括：
- 完整的单元操作测试
- 测试覆盖报告
- 性能基准数据

### pytest_old/ 目录  
保存早期的pytest实现，包括：
- 分散的测试文件
- 旧的fixture设计
- 原始的标记系统

### runners/ 目录
保存各种历史测试运行器：
- `run_all_dwsim_tests.py` - 最早的unittest运行器
- `run_pytest_dwsim.py` - 第一代pytest运行器
- `run_tests.py` - 通用测试脚本

## 🔧 维护说明

### 只读保存
- 不再更新这些文件
- 仅用于参考和对比
- 保持历史完整性

### 清理计划
- 验证迁移完成后可考虑删除
- 保留重要的参考文档
- 归档到版本控制系统

---

**推荐使用新的统一测试系统，获得更好的开发体验！** 🚀 