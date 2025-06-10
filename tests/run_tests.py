#!/usr/bin/env python3
"""
DWSIM FlowsheetSolver 测试运行脚本
==============================

提供多种测试运行选项和便捷的测试执行方式。

使用方法：
    python run_tests.py --all                 # 运行所有测试
    python run_tests.py --unit                # 仅运行单元测试
    python run_tests.py --integration         # 仅运行集成测试
    python run_tests.py --performance         # 仅运行性能测试
    python run_tests.py --quick               # 快速测试（排除慢速测试）
    python run_tests.py --coverage            # 生成覆盖率报告
    python run_tests.py --stress              # 压力测试
    python run_tests.py --memory              # 内存测试
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path


class TestRunner:
    """测试运行器类"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        
        # 确保日志目录存在
        log_dir = self.tests_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
    def run_command(self, cmd, description=None):
        """运行命令并输出结果"""
        if description:
            print(f"\n{'='*60}")
            print(f"执行: {description}")
            print(f"{'='*60}")
        
        print(f"命令: {' '.join(cmd)}")
        print()
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\n执行时间: {duration:.2f}秒")
        
        if result.returncode == 0:
            print("✅ 执行成功")
        else:
            print("❌ 执行失败")
            
        return result.returncode
    
    def run_unit_tests(self):
        """运行单元测试"""
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/unit/",
            "-m", "unit",
            "--tb=short"
        ]
        return self.run_command(cmd, "单元测试")
    
    def run_integration_tests(self):
        """运行集成测试"""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "-m", "integration",
            "--tb=short"
        ]
        return self.run_command(cmd, "集成测试")
    
    def run_performance_tests(self):
        """运行性能测试"""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/performance/",
            "-m", "performance",
            "--tb=short",
            "-s"  # 显示性能输出
        ]
        return self.run_command(cmd, "性能基准测试")
    
    def run_quick_tests(self):
        """运行快速测试（排除慢速测试）"""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "tests/integration/",
            "-m", "not slow and not performance",
            "--tb=short"
        ]
        return self.run_command(cmd, "快速测试")
    
    def run_stress_tests(self):
        """运行压力测试"""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/performance/",
            "-m", "stress",
            "--tb=short",
            "-s"
        ]
        return self.run_command(cmd, "压力测试")
    
    def run_memory_tests(self):
        """运行内存测试"""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/performance/",
            "-m", "memory",
            "--tb=short",
            "-s"
        ]
        return self.run_command(cmd, "内存测试")
    
    def run_all_tests(self):
        """运行所有测试"""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "--tb=short"
        ]
        return self.run_command(cmd, "完整测试套件")
    
    def run_coverage_report(self):
        """生成覆盖率报告"""
        # 首先运行测试收集覆盖率
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "tests/integration/",
            "--cov=flowsheet_solver",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ]
        
        result = self.run_command(cmd, "覆盖率测试")
        
        if result == 0:
            print(f"\n📊 覆盖率报告已生成:")
            print(f"   HTML报告: {self.project_root}/htmlcov/index.html")
            print(f"   XML报告:  {self.project_root}/coverage.xml")
        
        return result
    
    def run_specific_test(self, test_path):
        """运行特定测试文件或函数"""
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v",
            "--tb=short"
        ]
        return self.run_command(cmd, f"特定测试: {test_path}")
    
    def check_dependencies(self):
        """检查测试依赖"""
        print("检查测试依赖...")
        
        required_packages = [
            "pytest",
            "pytest-cov", 
            "psutil",
            "numpy"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (缺失)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n安装缺失的依赖:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("\n✅ 所有依赖都已安装")
        return True
    
    def lint_code(self):
        """代码检查"""
        print("执行代码质量检查...")
        
        # 检查是否安装了flake8
        try:
            import flake8
        except ImportError:
            print("❌ flake8未安装，跳过代码检查")
            print("安装: pip install flake8")
            return 0
        
        cmd = [
            sys.executable, "-m", "flake8",
            "flowsheet_solver/",
            "--max-line-length=100",
            "--ignore=E203,W503"
        ]
        
        return self.run_command(cmd, "代码质量检查")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DWSIM FlowsheetSolver 测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--all", action="store_true", 
                       help="运行所有测试")
    parser.add_argument("--unit", action="store_true", 
                       help="运行单元测试")
    parser.add_argument("--integration", action="store_true", 
                       help="运行集成测试") 
    parser.add_argument("--performance", action="store_true", 
                       help="运行性能测试")
    parser.add_argument("--quick", action="store_true", 
                       help="快速测试(排除慢速测试)")
    parser.add_argument("--coverage", action="store_true", 
                       help="生成覆盖率报告")
    parser.add_argument("--stress", action="store_true", 
                       help="运行压力测试")
    parser.add_argument("--memory", action="store_true", 
                       help="运行内存测试")
    parser.add_argument("--check-deps", action="store_true", 
                       help="检查测试依赖")
    parser.add_argument("--lint", action="store_true", 
                       help="代码质量检查")
    parser.add_argument("--test", type=str, metavar="PATH", 
                       help="运行特定测试文件或函数")
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助
    if not any(vars(args).values()):
        parser.print_help()
        return 0
    
    runner = TestRunner()
    
    # 检查依赖
    if args.check_deps:
        success = runner.check_dependencies()
        return 0 if success else 1
    
    # 代码检查
    if args.lint:
        return runner.lint_code()
    
    # 检查基本依赖是否满足
    if not runner.check_dependencies():
        print("\n❌ 依赖检查失败，请先安装缺失的依赖")
        return 1
    
    results = []
    
    # 运行指定的测试
    if args.test:
        results.append(runner.run_specific_test(args.test))
    elif args.unit:
        results.append(runner.run_unit_tests())
    elif args.integration:
        results.append(runner.run_integration_tests())
    elif args.performance:
        results.append(runner.run_performance_tests())
    elif args.quick:
        results.append(runner.run_quick_tests())
    elif args.stress:
        results.append(runner.run_stress_tests())
    elif args.memory:
        results.append(runner.run_memory_tests())
    elif args.coverage:
        results.append(runner.run_coverage_report())
    elif args.all:
        results.append(runner.run_all_tests())
    
    # 总结结果
    print(f"\n{'='*60}")
    print("测试运行总结")
    print(f"{'='*60}")
    
    if all(result == 0 for result in results):
        print("✅ 所有测试都通过了！")
        return 0
    else:
        print("❌ 部分测试失败")
        failed_count = sum(1 for result in results if result != 0)
        print(f"失败的测试组数: {failed_count}/{len(results)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 