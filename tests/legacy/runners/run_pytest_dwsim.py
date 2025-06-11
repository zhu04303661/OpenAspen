#!/usr/bin/env python3
"""
DWSIM 单元操作 pytest 测试运行器
==============================

使用pytest框架运行DWSIM单元操作的所有测试。

功能特性：
1. 支持测试标记和过滤
2. 生成详细的测试报告
3. 性能测试和基准测试
4. 覆盖率分析
5. 并行测试执行
6. 自定义测试配置

基于DWSIM.UnitOperations VB.NET代码的全面验证。
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


class DWSIMPytestRunner:
    """
    DWSIM pytest测试运行器
    
    提供灵活的测试执行和报告功能
    """
    
    def __init__(self):
        """初始化测试运行器"""
        self.test_dir = current_dir
        self.project_root = project_root
        
        # 测试文件
        self.test_files = {
            "comprehensive": "test_dwsim_operations_pytest.py",
            "specific": "test_specific_operations_pytest.py",
            "original": "test_dwsim_operations.py"
        }
        
        # 测试标记
        self.available_marks = [
            "foundation", "basic_ops", "reactors", "logical", "advanced",
            "cape_open", "solver", "validation", "mixer", "heater", "pump", 
            "heat_exchanger", "valve", "splitter", "integration", "performance",
            "unit", "smoke", "slow", "fast"
        ]
        
        # 默认pytest参数
        self.default_pytest_args = [
            "-v",                    # verbose
            "--tb=short",           # short traceback
            "--strict-markers",     # strict marker checking
            "--color=yes",          # colored output
            "--durations=10",       # show 10 slowest tests
        ]
    
    def run_tests(self, 
                  test_suite: str = "all",
                  markers: Optional[List[str]] = None,
                  exclude_markers: Optional[List[str]] = None,
                  parallel: bool = False,
                  coverage: bool = False,
                  html_report: bool = False,
                  performance_only: bool = False,
                  smoke_only: bool = False,
                  verbose: bool = True,
                  maxfail: Optional[int] = None,
                  extra_args: Optional[List[str]] = None) -> int:
        """
        运行测试
        
        Args:
            test_suite: 测试套件 ("all", "comprehensive", "specific", "original")
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
            
        Returns:
            int: 退出代码
        """
        print("🚀 启动 DWSIM 单元操作 pytest 测试")
        print("=" * 60)
        
        # 构建pytest命令
        cmd = ["python", "-m", "pytest"]
        
        # 添加默认参数
        cmd.extend(self.default_pytest_args)
        
        # 确定测试文件
        test_files = self._get_test_files(test_suite)
        cmd.extend(test_files)
        
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
                report_file = self.test_dir / "reports" / f"dwsim_test_report_{int(time.time())}.html"
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
    
    def _get_test_files(self, test_suite: str) -> List[str]:
        """
        获取要运行的测试文件
        
        Args:
            test_suite: 测试套件名称
            
        Returns:
            List[str]: 测试文件路径列表
        """
        if test_suite == "all":
            return list(self.test_files.values())
        elif test_suite in self.test_files:
            return [self.test_files[test_suite]]
        else:
            # 当作文件名处理
            if os.path.exists(test_suite):
                return [test_suite]
            else:
                raise ValueError(f"未知的测试套件或文件: {test_suite}")
    
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
        
        mark_descriptions = {
            "foundation": "基础框架测试",
            "basic_ops": "基本单元操作测试",
            "reactors": "反应器系统测试",
            "logical": "逻辑模块测试",
            "advanced": "高级单元操作测试",
            "cape_open": "CAPE-OPEN集成测试",
            "solver": "求解器测试",
            "validation": "验证调试测试",
            "mixer": "混合器测试",
            "heater": "加热器测试",
            "pump": "泵测试",
            "heat_exchanger": "热交换器测试",
            "valve": "阀门测试",
            "splitter": "分离器测试",
            "integration": "集成测试",
            "performance": "性能测试",
            "unit": "单元测试",
            "smoke": "冒烟测试",
            "slow": "慢速测试",
            "fast": "快速测试"
        }
        
        for mark in self.available_marks:
            desc = mark_descriptions.get(mark, "无描述")
            print(f"  {mark:<15} - {desc}")
    
    def run_quick_tests(self) -> int:
        """运行快速测试（排除slow标记）"""
        print("🏃 运行快速测试")
        return self.run_tests(exclude_markers=["slow"], performance_only=False)
    
    def run_performance_tests(self) -> int:
        """运行性能测试"""
        print("📈 运行性能测试")
        return self.run_tests(performance_only=True)
    
    def run_smoke_tests(self) -> int:
        """运行冒烟测试"""
        print("💨 运行冒烟测试")
        return self.run_tests(smoke_only=True)
    
    def run_by_component(self, component: str) -> int:
        """按组件运行测试"""
        component_markers = {
            "mixer": ["mixer"],
            "heater": ["heater"],
            "pump": ["pump"],
            "heat_exchanger": ["heat_exchanger"],
            "valve": ["valve"],
            "splitter": ["splitter"],
            "reactors": ["reactors"],
            "logical": ["logical"],
            "solver": ["solver"]
        }
        
        if component not in component_markers:
            print(f"❌ 未知组件: {component}")
            print(f"可用组件: {', '.join(component_markers.keys())}")
            return 1
        
        markers = component_markers[component]
        print(f"🔧 运行 {component} 组件测试")
        return self.run_tests(markers=markers)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DWSIM 单元操作 pytest 测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                          # 运行所有测试
  %(prog)s --quick                  # 运行快速测试
  %(prog)s --performance            # 运行性能测试
  %(prog)s --smoke                  # 运行冒烟测试
  %(prog)s --markers mixer heater   # 运行混合器和加热器测试
  %(prog)s --exclude slow           # 排除慢速测试
  %(prog)s --component mixer        # 运行混合器组件测试
  %(prog)s --parallel --coverage    # 并行执行并生成覆盖率
  %(prog)s --list-marks             # 列出所有可用标记
        """
    )
    
    # 测试选择参数
    parser.add_argument(
        "--suite", "-s",
        choices=["all", "comprehensive", "specific", "original"],
        default="all",
        help="选择测试套件"
    )
    
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
    runner = DWSIMPytestRunner()
    
    # 处理信息选项
    if args.list_marks:
        runner.list_available_marks()
        return 0
    
    # 处理快捷选项
    if args.quick:
        return runner.run_quick_tests()
    elif args.performance:
        return runner.run_performance_tests()
    elif args.smoke:
        return runner.run_smoke_tests()
    elif args.component:
        return runner.run_by_component(args.component)
    
    # 常规测试执行
    return runner.run_tests(
        test_suite=args.suite,
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