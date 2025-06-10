#!/usr/bin/env python3
"""
调试测试 - 了解pytest导入问题
"""
import sys
import os
print("=== DEBUG TEST ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"脚本路径: {__file__}")
print(f"Python路径:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("\n=== 测试导入 ===")

# 测试1: 直接导入
try:
    import flowsheet_solver
    print("✅ 直接导入 flowsheet_solver 成功")
    print(f"   位置: {flowsheet_solver.__file__}")
    print(f"   版本: {flowsheet_solver.__version__}")
except Exception as e:
    print(f"❌ 直接导入 flowsheet_solver 失败: {e}")

# 测试2: 子模块导入
try:
    from flowsheet_solver import calculation_args
    print("✅ 导入 flowsheet_solver.calculation_args 成功")
    print(f"   位置: {calculation_args.__file__}")
except Exception as e:
    print(f"❌ 导入 flowsheet_solver.calculation_args 失败: {e}")

# 测试3: 检查包内容
try:
    import flowsheet_solver
    print(f"✅ flowsheet_solver 包内容: {dir(flowsheet_solver)}")
    
    # 检查是否有 __path__
    if hasattr(flowsheet_solver, '__path__'):
        print(f"   __path__: {flowsheet_solver.__path__}")
    else:
        print("   ❌ 没有 __path__ 属性")
        
    # 检查flowsheet_solver目录
    fs_dir = os.path.dirname(flowsheet_solver.__file__)
    print(f"   目录内容: {os.listdir(fs_dir)}")
    
except Exception as e:
    print(f"❌ 检查包内容失败: {e}")

print("\n=== 测试完成 ===")

def test_imports():
    """pytest测试函数"""
    import flowsheet_solver
    from flowsheet_solver.calculation_args import CalculationArgs, ObjectType
    assert CalculationArgs is not None
    assert ObjectType is not None 