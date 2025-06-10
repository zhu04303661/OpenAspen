"""
pytest配置文件
确保模块导入路径正确配置
"""
import sys
import os

# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"conftest.py: 添加项目根目录到Python路径: {project_root}") 