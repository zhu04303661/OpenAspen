"""
DWSIM数学库测试模块

简化版测试模块，专注于基本功能验证。
"""

__version__ = "1.0.0"

# 测试配置
TEST_TOLERANCE = 1e-10
FLOAT_TOLERANCE = 1e-6
PERFORMANCE_TIMEOUT = 10.0

# 基本测试运行器
def run_all_tests():
    """运行所有可用的测试"""
    import pytest
    import os
    test_dir = os.path.dirname(__file__)
    return pytest.main([test_dir, '-v', '--tb=short', '--no-cov']) 