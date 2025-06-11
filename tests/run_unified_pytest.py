#!/usr/bin/env python3
"""
DWSIM 单元操作与数学库统一pytest运行器
===================================

整合所有DWSIM单元操作和数学库测试的统一运行器。

功能特性：
1. 统一的测试文件管理 (DWSIM单元操作 + 数学库)
2. 完整的标记过滤系统
3. 按组件和模块分类测试
4. 性能测试和基准测试
5. 覆盖率分析
6. 并行测试执行
7. 详细的测试报告

支持的测试模块：
- DWSIM单元操作测试 (test_dwsim_unified.py)
- DWSIM数学库测试 (test_dwsim_math.py)
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


class UnifiedDWSIMPytestRunner:
    """
    DWSIM 统一pytest运行器
    
    管理整合后的测试文件和执行策略
    """
    
    def __init__(self):
        """初始化统一运行器"""
        self.test_dir = current_dir
        self.project_root = project_root
        
        # 测试文件映射
        self.test_files = {
            "dwsim_unified": "unified/test_dwsim_unified.py",
            "dwsim_math": "test_dwsim_math.py"
        }
        
        # 检查测试文件是否存在
        self.available_test_files = {}
        for name, file_path in self.test_files.items():
            full_path = self.test_dir / file_path
            if full_path.exists():
                self.available_test_files[name] = full_path
            else:
                print(f"⚠️  测试文件不存在: {full_path}")
        
        if not self.available_test_files:
            print("❌ 没有可用的测试文件")
            sys.exit(1)
        
        # 测试标记
        self.available_marks = [
            # 架构层级标记
            "foundation", "basic_ops", "advanced", "integration",
            # 系统模块标记
            "reactors", "logical", "solver", "cape_open", "validation",
            # 具体设备标记
            "mixer", "splitter", "heater", "cooler", "pump", "compressor", 
            "valve", "heat_exchanger",
            # 测试类型标记
            "unit", "performance", "smoke", "slow", "fast",
            # 特殊功能标记
            "parametrize", "error_handling", "memory", "concurrent",
            # 扩展核心系统标记
            "calculation_args", "solver_exceptions", "flowsheet_solver",
            "convergence_solver", "remote_solvers", "extended_operations",
            "benchmarks",
            # 数学模块标记
            "math_core", "math_numerics", "math_solvers", "math_optimization",
            "math_random", "math_integration", "math_performance",
            "matrix_ops", "interpolation", "complex_number", "brent_solver",
            "lbfgs", "mersenne_twister"
        ]
        
        # 默认pytest参数
        self.default_pytest_args = [
            "-v",                    # verbose
            "--tb=short",           # short traceback
            "--color=yes",          # colored output
            "--durations=10",       # show 10 slowest tests
            "--disable-warnings",   # disable warnings for cleaner output
        ]
    
    def run_tests(self, 
                  markers: Optional[List[str]] = None,
                  exclude_markers: Optional[List[str]] = None,
                  parallel: bool = False,
                  coverage: bool = False,
                  html_report: bool = False,
                  performance_only: bool = False,
                  smoke_only: bool = False,
                  verbose: bool = True,
                  maxfail: Optional[int] = None,
                  extra_args: Optional[List[str]] = None,
                  test_files: Optional[List[str]] = None) -> int:
        """
        运行统一测试
        
        Args:
            markers: 包含的标记列表
            exclude_markers: 排除的标记列表
            parallel: 是否并行执行
            coverage: 是否生成覆盖率报告
            html_report: 是否生成HTML报告
            performance_only: 只运行性能测试
            smoke_only: 只运行冒烟测试
            verbose: 详细输出
            maxfail: 最大失败数
            extra_args: 额外的pytest参数
            test_files: 指定的测试文件列表
            
        Returns:
            int: 退出代码
        """
        print("🚀 启动 DWSIM 单元操作统一测试")
        print("=" * 60)
        
        # 构建pytest命令
        cmd = ["python", "-m", "pytest"]
        
        # 添加默认参数
        cmd.extend(self.default_pytest_args)
        
        # 指定测试文件
        if test_files:
            # 使用指定的测试文件
            for test_file in test_files:
                if test_file in self.available_test_files:
                    cmd.append(str(self.available_test_files[test_file]))
                else:
                    print(f"⚠️  未知测试文件: {test_file}")
        else:
            # 运行所有可用的测试文件
            for file_path in self.available_test_files.values():
                cmd.append(str(file_path))
        
        # 处理标记过滤
        if performance_only:
            cmd.extend(["-m", "performance"])
        elif smoke_only:
            cmd.extend(["-m", "smoke"])
        elif markers or exclude_markers:
            marker_expr = self._build_marker_expression(markers, exclude_markers)
            if marker_expr:
                cmd.extend(["-m", marker_expr])
        
        # 并行执行
        if parallel:
            try:
                import pytest_xdist  # noqa
                cmd.extend(["-n", "auto"])
                print("📦 启用并行测试执行")
            except ImportError:
                print("⚠️  pytest-xdist未安装，无法并行执行")
        
        # 覆盖率报告
        if coverage:
            try:
                import pytest_cov  # noqa
                cmd.extend([
                    "--cov=dwsim_operations",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov"
                ])
                print("📊 启用覆盖率分析")
            except ImportError:
                print("⚠️  pytest-cov未安装，无法生成覆盖率报告")
        
        # HTML报告
        if html_report:
            try:
                import pytest_html  # noqa
                report_file = self.test_dir / "reports" / f"dwsim_unified_test_report_{int(time.time())}.html"
                report_file.parent.mkdir(exist_ok=True)
                cmd.extend(["--html", str(report_file), "--self-contained-html"])
                print(f"📋 HTML报告将保存至: {report_file}")
            except ImportError:
                print("⚠️  pytest-html未安装，无法生成HTML报告")
        
        # 最大失败数
        if maxfail:
            cmd.extend(["--maxfail", str(maxfail)])
        
        # 额外参数
        if extra_args:
            cmd.extend(extra_args)
        
        # 显示执行的命令
        if verbose:
            print(f"🔍 执行命令: {' '.join(cmd)}")
            print("=" * 60)
        
        # 设置工作目录
        os.chdir(self.test_dir)
        
        # 执行测试
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=False)
            exit_code = result.returncode
        except KeyboardInterrupt:
            print("\n⚠️  测试被用户中断")
            exit_code = 130
        except Exception as e:
            print(f"❌ 测试执行失败: {e}")
            exit_code = 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 显示结果
        print("\n" + "=" * 60)
        print("🏁 测试执行完成")
        print(f"⏱️  总执行时间: {duration:.2f}秒")
        
        if exit_code == 0:
            print("✅ 所有测试通过!")
        else:
            print(f"❌ 测试失败 (退出代码: {exit_code})")
        
        print("=" * 60)
        
        return exit_code
    
    def _build_marker_expression(self, 
                                 include_markers: Optional[List[str]] = None,
                                 exclude_markers: Optional[List[str]] = None) -> str:
        """
        构建pytest标记表达式
        
        Args:
            include_markers: 包含的标记
            exclude_markers: 排除的标记
            
        Returns:
            str: 标记表达式
        """
        expressions = []
        
        if include_markers:
            # 验证标记
            for marker in include_markers:
                if marker not in self.available_marks:
                    print(f"⚠️  警告: 未知标记 '{marker}'")
            
            if len(include_markers) == 1:
                expressions.append(include_markers[0])
            else:
                expressions.append(f"({' or '.join(include_markers)})")
        
        if exclude_markers:
            # 验证标记
            for marker in exclude_markers:
                if marker not in self.available_marks:
                    print(f"⚠️  警告: 未知标记 '{marker}'")
            
            for marker in exclude_markers:
                expressions.append(f"not {marker}")
        
        return " and ".join(expressions)
    
    def list_available_marks(self):
        """列出所有可用的测试标记"""
        print("📋 可用的测试标记:")
        print("-" * 40)
        
        mark_categories = {
            "架构层级": ["foundation", "basic_ops", "advanced", "integration"],
            "系统模块": ["reactors", "logical", "solver", "cape_open", "validation"],
            "具体设备": ["mixer", "splitter", "heater", "cooler", "pump", "compressor", "valve", "heat_exchanger"],
            "测试类型": ["unit", "performance", "smoke", "slow", "fast"],
            "特殊功能": ["parametrize", "error_handling", "memory", "concurrent"],
            "扩展核心系统": ["calculation_args", "solver_exceptions", "flowsheet_solver", "convergence_solver", "remote_solvers", "extended_operations", "benchmarks"],
            "数学计算模块": ["math_core", "math_numerics", "math_solvers", "math_optimization", "math_random", "math_integration", "math_performance"],
            "数学具体组件": ["matrix_ops", "interpolation", "complex_number", "brent_solver", "lbfgs", "mersenne_twister"]
        }
        
        mark_descriptions = {
            # 架构层级
            "foundation": "基础框架测试 - SimulationObjectClass、UnitOpBaseClass等",
            "basic_ops": "基本单元操作测试 - 混合器、加热器、泵等",
            "advanced": "高级单元操作测试 - 精馏塔、管道等复杂操作",
            "integration": "集成测试 - 验证组件协同工作",
            
            # 系统模块
            "reactors": "反应器系统测试 - BaseReactor、Gibbs、PFR等",
            "logical": "逻辑模块测试 - Adjust、Spec、Recycle等",
            "solver": "求解器测试 - 集成求解器和计算顺序",
            "cape_open": "CAPE-OPEN集成测试 - 第三方组件互操作",
            "validation": "验证调试测试 - 输入验证、错误处理",
            
            # 具体设备
            "mixer": "混合器测试 - 压力计算、质量能量平衡",
            "splitter": "分离器测试 - 分流比计算",
            "heater": "加热器测试 - 热量计算模式",
            "cooler": "冷却器测试 - 冷却计算",
            "pump": "泵测试 - 扬程计算、效率、NPSH",
            "compressor": "压缩机测试 - 压缩比、功耗",
            "valve": "阀门测试 - 压降计算、Cv值",
            "heat_exchanger": "热交换器测试 - LMTD计算、传热方程",
            
            # 测试类型
            "unit": "单元测试 - 单个功能点测试",
            "performance": "性能测试 - 计算效率和响应时间",
            "smoke": "冒烟测试 - 快速验证基本功能",
            "slow": "慢速测试 - 耗时较长的测试用例",
            "fast": "快速测试 - 执行迅速的测试用例",
            
            # 特殊功能
            "parametrize": "参数化测试 - 多参数测试用例",
            "error_handling": "错误处理测试 - 异常情况验证",
            "memory": "内存测试 - 内存使用和泄漏检测",
            "concurrent": "并发测试 - 多线程和并行处理",
            
            # 扩展核心系统标记
            "calculation_args": "计算参数系统测试 - CalculationArgs类和枚举",
            "solver_exceptions": "求解器异常系统测试 - 异常层次和处理",
            "flowsheet_solver": "FlowsheetSolver核心测试 - 主求解器功能",
            "convergence_solver": "收敛求解器测试 - Broyden、Newton-Raphson等",
            "remote_solvers": "远程求解器测试 - TCP、Azure客户端",
            "extended_operations": "扩展单元操作测试 - 压缩机、阀门、管道等",
            "benchmarks": "基准性能测试 - 大型流程图、内存、并行计算",
            
            # 数学计算模块标记
            "math_core": "核心数学模块测试 - 基础统计、通用数学函数",
            "math_numerics": "数值计算模块测试 - 复数运算、数值方法",
            "math_solvers": "数学求解器测试 - Brent算法、非线性方程求解",
            "math_optimization": "优化算法测试 - L-BFGS、无约束优化",
            "math_random": "随机数生成测试 - Mersenne Twister、统计验证",
            "math_integration": "数学集成测试 - 模块间协作验证",
            "math_performance": "数学性能测试 - 算法效率、扩展性基准",
            
            # 数学具体组件标记
            "matrix_ops": "矩阵操作测试 - 行列式、求逆、线性方程组",
            "interpolation": "插值算法测试 - 数据插值、边界条件处理",
            "complex_number": "复数运算测试 - 基本运算、极坐标转换",
            "brent_solver": "Brent求解器测试 - 根查找、收敛验证",
            "lbfgs": "L-BFGS优化测试 - 二次函数、Rosenbrock函数优化",
            "mersenne_twister": "Mersenne Twister测试 - 随机数质量、重现性验证"
        }
        
        for category, marks in mark_categories.items():
            print(f"\n{category}:")
            for mark in marks:
                desc = mark_descriptions.get(mark, "无描述")
                print(f"  {mark:<15} - {desc}")
    
    def run_quick_tests(self) -> int:
        """运行快速测试（排除slow标记）"""
        print("🏃 运行快速测试")
        return self.run_tests(exclude_markers=["slow"])
    
    def run_performance_tests(self) -> int:
        """运行性能测试"""
        print("📈 运行性能测试")
        return self.run_tests(performance_only=True)
    
    def run_smoke_tests(self) -> int:
        """运行冒烟测试"""
        print("💨 运行冒烟测试")
        return self.run_tests(smoke_only=True)
    
    def run_math_tests(self) -> int:
        """运行数学模块测试"""
        print("🧮 运行数学模块测试")
        return self.run_tests(test_files=["dwsim_math"])
    
    def run_dwsim_tests(self) -> int:
        """运行DWSIM单元操作测试"""
        print("⚗️  运行DWSIM单元操作测试")
        return self.run_tests(test_files=["dwsim_unified"])
    
    def run_by_component(self, component: str) -> int:
        """按组件运行测试"""
        component_markers = {
            # DWSIM 单元操作组件
            "foundation": ["foundation"],
            "mixer": ["mixer"],
            "heater": ["heater"],
            "pump": ["pump"],
            "heat_exchanger": ["heat_exchanger"],
            "valve": ["valve"],
            "splitter": ["splitter"],
            "reactors": ["reactors"],
            "logical": ["logical"],
            "solver": ["solver"],
            "cape_open": ["cape_open"],
            "validation": ["validation"],
            "basic_ops": ["basic_ops"],
            "advanced": ["advanced"],
            # 扩展核心系统组件
            "calculation_args": ["calculation_args"],
            "solver_exceptions": ["solver_exceptions"],
            "flowsheet_solver": ["flowsheet_solver"],
            "convergence_solver": ["convergence_solver"],
            "remote_solvers": ["remote_solvers"],
            "extended_operations": ["extended_operations"],
            "benchmarks": ["benchmarks"],
            "compressor": ["compressor"],
            "performance_tests": ["performance", "benchmarks"],
            "core_solver": ["flowsheet_solver", "convergence_solver", "solver"],
            "exceptions": ["solver_exceptions", "error_handling"],
            # 数学模块组件
            "math": ["math_core", "math_numerics", "math_solvers", "math_optimization", "math_random"],
            "math_core": ["math_core", "matrix_ops", "interpolation"],
            "math_numerics": ["math_numerics", "complex_number"],
            "math_solvers": ["math_solvers", "brent_solver"],
            "math_optimization": ["math_optimization", "lbfgs"],
            "math_random": ["math_random", "mersenne_twister"],
            "math_integration": ["math_integration"],
            "math_performance": ["math_performance", "slow"],
            "matrix_ops": ["matrix_ops"],
            "interpolation": ["interpolation"],
            "complex_number": ["complex_number"],
            "brent_solver": ["brent_solver"],
            "lbfgs": ["lbfgs"],
            "mersenne_twister": ["mersenne_twister"]
        }
        
        if component not in component_markers:
            print(f"❌ 未知组件: {component}")
            print(f"可用组件: {', '.join(component_markers.keys())}")
            return 1
        
        markers = component_markers[component]
        print(f"🔧 运行 {component} 组件测试")
        return self.run_tests(markers=markers)
    
    def run_collection_test(self) -> int:
        """运行测试收集验证"""
        print("📝 收集测试用例...")
        
        cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
        # 添加所有可用的测试文件
        for file_path in self.available_test_files.values():
            cmd.append(str(file_path))
        
        os.chdir(self.test_dir)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("✅ 测试收集成功")
                lines = result.stdout.strip().split('\n')
                test_count = 0
                for line in lines:
                    if '::' in line and 'test_' in line:
                        test_count += 1
                print(f"📊 发现 {test_count} 个测试用例")
                
                # 显示测试类别统计
                print("\n测试类别分布:")
                categories = {}
                for line in lines:
                    if '::Test' in line:
                        class_name = line.split('::')[1].split('::')[0]
                        categories[class_name] = categories.get(class_name, 0) + 1
                
                for category, count in sorted(categories.items()):
                    print(f"  {category}: {count}个测试")
                
                return 0
            else:
                print("❌ 测试收集失败")
                print(result.stderr)
                return result.returncode
                
        except Exception as e:
            print(f"❌ 测试收集出错: {e}")
            return 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DWSIM 单元操作统一pytest运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                          # 运行所有测试
  %(prog)s --quick                  # 运行快速测试
  %(prog)s --performance            # 运行性能测试
  %(prog)s --smoke                  # 运行冒烟测试
  %(prog)s --math                   # 只运行数学模块测试
  %(prog)s --dwsim                  # 只运行DWSIM单元操作测试
  %(prog)s --markers mixer heater   # 运行混合器和加热器测试
  %(prog)s --exclude slow           # 排除慢速测试
  %(prog)s --component mixer        # 运行混合器组件测试
  %(prog)s --component math_core    # 运行核心数学模块测试
  %(prog)s --parallel --coverage    # 并行执行并生成覆盖率
  %(prog)s --list-marks             # 列出所有可用标记
  %(prog)s --collect                # 收集并统计测试用例
        """
    )
    
    # 测试选择参数
    parser.add_argument(
        "--markers", "-m",
        nargs="*",
        help="包含的测试标记"
    )
    
    parser.add_argument(
        "--exclude", "-e", 
        nargs="*",
        help="排除的测试标记"
    )
    
    parser.add_argument(
        "--component", "-c",
        help="运行特定组件的测试"
    )
    
    # 快捷选项
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="运行快速测试（排除slow）"
    )
    
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="只运行性能测试"
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true", 
        help="只运行冒烟测试"
    )
    
    parser.add_argument(
        "--math",
        action="store_true",
        help="只运行数学模块测试"
    )
    
    parser.add_argument(
        "--dwsim",
        action="store_true",
        help="只运行DWSIM单元操作测试"
    )
    
    # 执行选项
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="并行执行测试"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="生成覆盖率报告"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="生成HTML测试报告"
    )
    
    parser.add_argument(
        "--maxfail",
        type=int,
        help="最大失败测试数"
    )
    
    # 信息选项
    parser.add_argument(
        "--list-marks",
        action="store_true",
        help="列出所有可用的测试标记"
    )
    
    parser.add_argument(
        "--collect",
        action="store_true",
        help="收集并统计测试用例"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        default=True,
        help="详细输出"
    )
    
    # 额外pytest参数
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="额外的pytest参数"
    )
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = UnifiedDWSIMPytestRunner()
    
    # 处理信息选项
    if args.list_marks:
        runner.list_available_marks()
        return 0
    
    if args.collect:
        return runner.run_collection_test()
    
    # 处理快捷选项
    if args.quick:
        return runner.run_quick_tests()
    elif args.performance:
        return runner.run_performance_tests()
    elif args.smoke:
        return runner.run_smoke_tests()
    elif args.math:
        return runner.run_math_tests()
    elif args.dwsim:
        return runner.run_dwsim_tests()
    elif args.component:
        return runner.run_by_component(args.component)
    
    # 常规测试执行
    return runner.run_tests(
        markers=args.markers,
        exclude_markers=args.exclude,
        parallel=args.parallel,
        coverage=args.coverage,
        html_report=args.html_report,
        verbose=args.verbose,
        maxfail=args.maxfail,
        extra_args=args.pytest_args
    )


if __name__ == "__main__":
    sys.exit(main()) 