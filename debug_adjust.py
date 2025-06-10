#!/usr/bin/env python3

from unittest.mock import Mock
from flowsheet_solver.convergence_solver import SimultaneousAdjustSolver

def test_adjust_collection():
    solver = SimultaneousAdjustSolver()
    
    # 创建模拟流程图
    mock_flowsheet = Mock()
    
    # 创建模拟调节对象
    mock_adjust1 = Mock()
    mock_adjust1.name = "Adjust1"
    mock_adjust1.manipulation_type = "Temperature"
    mock_adjust1.target_value = 350.0
    mock_adjust1.current_value = 300.0
    mock_adjust1.enabled = True
    
    mock_adjust2 = Mock()
    mock_adjust2.name = "Adjust2"
    mock_adjust2.manipulation_type = "Pressure"
    mock_adjust2.target_value = 200000.0
    mock_adjust2.current_value = 150000.0
    mock_adjust2.enabled = True
    
    # 创建非调节对象
    mock_other = Mock()
    mock_other.name = "OtherObject"
    
    # 模拟流程图包含调节对象
    mock_flowsheet.simulation_objects = {
        "Adjust1": mock_adjust1,
        "Adjust2": mock_adjust2,
        "OtherObject": mock_other
    }
    
    print("=== 调试信息 ===")
    print(f"simulation_objects: {mock_flowsheet.simulation_objects}")
    
    for obj_name, obj in mock_flowsheet.simulation_objects.items():
        print(f"\n对象: {obj_name}")
        print(f"  hasattr(obj, 'name'): {hasattr(obj, 'name')}")
        if hasattr(obj, 'name'):
            print(f"  obj.name: {obj.name}")
            print(f"  obj.name.startswith('Adjust'): {obj.name.startswith('Adjust')}")
        print(f"  hasattr(obj, 'enabled'): {hasattr(obj, 'enabled')}")
        if hasattr(obj, 'enabled'):
            print(f"  obj.enabled: {obj.enabled}")
            print(f"  obj.enabled is True: {obj.enabled is True}")
            print(f"  bool(obj.enabled): {bool(obj.enabled)}")
    
    # 手动测试逻辑
    print("\n=== 手动测试逻辑 ===")
    adjust_objects_manual = []
    
    for obj_name, obj in mock_flowsheet.simulation_objects.items():
        print(f"\n处理对象: {obj_name}")
        is_adjust = False
        
        # 方法1：检查graphic_object.object_type
        if hasattr(obj, 'graphic_object') and hasattr(obj.graphic_object, 'object_type'):
            is_adjust = obj.graphic_object.object_type == "Adjust"
            print(f"  方法1 - graphic_object检查: {is_adjust}")
        
        # 方法2：检查对象名称（用于测试）
        if not is_adjust and hasattr(obj, 'name') and obj.name and obj.name.startswith('Adjust'):
            is_adjust = True
            print(f"  方法2 - 名称检查: {is_adjust}")
        
        # 方法3：检查对象类型属性
        if not is_adjust and hasattr(obj, 'object_type'):
            is_adjust = obj.object_type == "Adjust"
            print(f"  方法3 - object_type检查: {is_adjust}")
        
        if not is_adjust:
            print(f"  没有匹配的检查方法")
        
        # 只有确认是调节对象时才检查enabled状态
        if is_adjust:
            enabled = getattr(obj, 'enabled', True)
            print(f"  enabled值: {enabled}")
            print(f"  enabled is True: {enabled is True}")
            print(f"  hasattr(enabled, '__bool__'): {hasattr(enabled, '__bool__')}")
            if hasattr(enabled, '__bool__'):
                print(f"  bool(enabled): {bool(enabled)}")
            
            # 确保enabled是布尔值或可以转换为布尔值
            if enabled is True or (hasattr(enabled, '__bool__') and bool(enabled)):
                adjust_objects_manual.append(obj)
                print(f"  ✅ 添加到调节对象列表")
            else:
                print(f"  ❌ 未添加到调节对象列表")
        else:
            print(f"  不是调节对象，跳过")
    
    print(f"\n手动收集到的调节对象数量: {len(adjust_objects_manual)}")
    
    # 测试对象收集
    adjust_objects = solver._collect_adjust_objects(mock_flowsheet)
    
    print(f"\n方法收集到的调节对象数量: {len(adjust_objects)}")
    for i, obj in enumerate(adjust_objects):
        print(f"  {i+1}: {obj.name}")

if __name__ == "__main__":
    test_adjust_collection() 