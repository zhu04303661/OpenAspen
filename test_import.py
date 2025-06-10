#!/usr/bin/env python3
"""
测试导入脚本
"""

import sys
import os

print("当前工作目录:", os.getcwd())
print("Python路径:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("\n测试导入...")

try:
    import flowsheet_solver
    print("✅ flowsheet_solver包导入成功")
    print(f"   包位置: {flowsheet_solver.__file__ if hasattr(flowsheet_solver, '__file__') else '未知'}")
except Exception as e:
    print(f"❌ flowsheet_solver包导入失败: {e}")
    sys.exit(1)

try:
    from flowsheet_solver import calculation_args
    print("✅ calculation_args模块导入成功")
    print(f"   模块位置: {calculation_args.__file__}")
except Exception as e:
    print(f"❌ calculation_args模块导入失败: {e}")
    sys.exit(1)

try:
    from flowsheet_solver.calculation_args import CalculationArgs, ObjectType, CalculationStatus
    print("✅ CalculationArgs等类导入成功")
except Exception as e:
    print(f"❌ CalculationArgs等类导入失败: {e}")
    sys.exit(1)

try:
    from flowsheet_solver.solver import FlowsheetSolver
    print("✅ FlowsheetSolver类导入成功")
except Exception as e:
    print(f"❌ FlowsheetSolver类导入失败: {e}")
    sys.exit(1)

print("\n所有导入测试通过！") 