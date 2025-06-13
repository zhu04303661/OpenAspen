#!/usr/bin/env python3
"""
DWSIM热力学计算库 - 完整演示程序
===================================

展示热力学计算库的主要功能：
1. 化合物和相的创建与管理
2. 理想气体和Peng-Robinson物性包
3. PT闪蒸计算
4. 热力学性质计算
5. 工业案例演示

作者：OpenAspen项目组
版本：1.0.0
运行：python demo_thermo.py
"""

import numpy as np
import sys
import os

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dwsim_thermo.core.compound import Compound, COMMON_COMPOUNDS
    from dwsim_thermo.core.phase import Phase
    from dwsim_thermo.core.enums import PhaseType, PackageType, ConvergenceStatus
    from dwsim_thermo.property_packages.ideal import IdealPropertyPackage
    from dwsim_thermo.property_packages.peng_robinson import PengRobinsonPackage
    print("✅ 成功导入所有DWSIM热力学模块")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def print_separator(title: str, width: int = 80):
    """打印分隔线"""
    print("\n" + "="*width)
    print(f" {title} ".center(width, "="))
    print("="*width)

def print_subsection(title: str, width: int = 60):
    """打印子标题"""
    print(f"\n{'-'*width}")
    print(f" {title} ".center(width, "-"))
    print(f"{'-'*width}")

def demo_basic_compounds():
    """演示基础化合物功能"""
    print_separator("基础化合物功能演示")
    
    # 创建化合物
    print("\n1. 创建和配置化合物")
    water = Compound("水", cas="7732-18-5", formula="H2O")
    methane = Compound("甲烷", cas="74-82-8", formula="CH4")
    ethane = Compound("乙烷", cas="74-84-0", formula="C2H6")
    
    # 加载物性数据
    print("   正在加载物性数据...")
    water.load_properties_from_database()
    methane.load_properties_from_database()
    ethane.load_properties_from_database()
    
    compounds = [water, methane, ethane]
    
    # 显示化合物信息
    print("\n2. 化合物基础信息")
    for comp in compounds:
        print(f"\n   {comp.name} ({comp.formula})")
        print(f"   - CAS号: {comp.cas_number}")
        print(f"   - 分子量: {comp.properties.molecular_weight:.6f} kg/mol")
        print(f"   - 临界温度: {comp.properties.critical_temperature:.2f} K")
        print(f"   - 临界压力: {comp.properties.critical_pressure:.0f} Pa")
        print(f"   - 偏心因子: {comp.properties.acentric_factor:.4f}")
    
    # 计算物性
    print("\n3. 物性计算示例 (25°C, 1 atm)")
    T = 298.15  # K
    P = 101325.0  # Pa
    
    for comp in compounds:
        comp.set_state(T, P)
        try:
            vapor_pressure = comp.calculate_vapor_pressure(T)
            cp_ig = comp.calculate_ideal_gas_cp(T)
            print(f"\n   {comp.name}:")
            print(f"   - 饱和蒸汽压: {vapor_pressure:.0f} Pa")
            print(f"   - 理想气体热容: {cp_ig:.2f} J/mol/K")
        except Exception as e:
            print(f"   {comp.name}: 计算失败 - {e}")
    
    return compounds

def demo_phase_operations():
    """演示相操作功能"""
    print_separator("相操作功能演示")
    
    # 使用预定义化合物
    water = COMMON_COMPOUNDS["水"]
    methane = COMMON_COMPOUNDS["甲烷"]
    ethane = COMMON_COMPOUNDS["乙烷"]
    compounds = [water, methane, ethane]
    
    # 创建混合物相
    print("\n1. 创建三元混合物相")
    composition = [0.1, 0.5, 0.4]  # 水10%, 甲烷50%, 乙烷40%
    
    vapor_phase = Phase(PhaseType.VAPOR, compounds, composition, "气相混合物")
    liquid_phase = Phase(PhaseType.LIQUID, compounds, composition, "液相混合物")
    
    T = 298.15  # K
    P = 101325.0  # Pa
    
    vapor_phase.set_temperature_pressure(T, P)
    liquid_phase.set_temperature_pressure(T, P)
    
    print(f"   组成: 水{composition[0]*100:.1f}%, 甲烷{composition[1]*100:.1f}%, 乙烷{composition[2]*100:.1f}%")
    print(f"   条件: {T:.1f} K, {P:.0f} Pa")
    
    # 显示相信息
    print("\n2. 相基础信息")
    for phase in [vapor_phase, liquid_phase]:
        print(f"\n   {phase.name}:")
        print(f"   - 平均分子量: {phase.molecular_weight:.6f} kg/mol")
        print(f"   - 组分数量: {phase.n_components}")
        
        mass_fractions = phase.mass_fractions
        for i, comp in enumerate(compounds):
            print(f"   - {comp.name}: 摩尔分数={phase.mole_fractions[i]:.3f}, 质量分数={mass_fractions[i]:.3f}")
    
    # 计算混合性质
    print("\n3. 混合性质计算")
    try:
        cp_ig_vapor = vapor_phase.calculate_ideal_gas_cp()
        cp_ig_liquid = liquid_phase.calculate_ideal_gas_cp()
        
        print(f"   气相理想气体热容: {cp_ig_vapor:.2f} J/mol/K")
        print(f"   液相理想气体热容: {cp_ig_liquid:.2f} J/mol/K")
        
        # 临界性质
        critical_props = vapor_phase.get_critical_properties()
        print(f"   混合物临界温度: {critical_props['critical_temperature']:.2f} K")
        print(f"   混合物临界压力: {critical_props['critical_pressure']:.0f} Pa")
        
    except Exception as e:
        print(f"   混合性质计算失败: {e}")
    
    return vapor_phase, liquid_phase

def demo_ideal_property_package():
    """演示理想气体物性包"""
    print_separator("理想气体物性包演示")
    
    # 创建化合物和物性包
    compounds = [COMMON_COMPOUNDS["甲烷"], COMMON_COMPOUNDS["乙烷"]]
    ideal_pp = IdealPropertyPackage(compounds)
    
    print(f"\n物性包信息:")
    model_info = ideal_pp.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 设置计算条件
    T = 200.0  # K
    P = 500000.0  # Pa (5 bar)
    z = np.array([0.7, 0.3])  # 甲烷70%, 乙烷30%
    
    print(f"\n计算条件:")
    print(f"  温度: {T} K")
    print(f"  压力: {P/1000:.0f} kPa")
    print(f"  组成: 甲烷{z[0]*100:.1f}%, 乙烷{z[1]*100:.1f}%")
    
    # 进行PT闪蒸
    print(f"\n进行PT闪蒸计算...")
    result = ideal_pp.flash_pt(z, P, T)
    
    print(f"\n闪蒸结果:")
    print(f"  收敛状态: {result.converged}")
    print(f"  收敛状态码: {result.convergence_status.name}")
    print(f"  汽化率: {result.vapor_fraction:.4f}")
    
    if result.converged:
        print(f"  焓: {result.enthalpy:.2f} J/mol")
        print(f"  熵: {result.entropy:.2f} J/mol/K")
        print(f"  体积: {result.volume:.6f} m³/mol")
        
        if result.vapor_phase:
            print(f"\n  气相组成:")
            for i, comp in enumerate(compounds):
                print(f"    {comp.name}: {result.vapor_phase.mole_fractions[i]:.4f}")
        
        if result.liquid_phase:
            print(f"\n  液相组成:")
            for i, comp in enumerate(compounds):
                print(f"    {comp.name}: {result.liquid_phase.mole_fractions[i]:.4f}")
    else:
        print(f"  错误信息: {result.error_message}")
    
    # 测试其他闪蒸类型
    print(f"\n进行PH闪蒸计算 (目标焓: -5000 J/mol)...")
    ph_result = ideal_pp.flash_ph(z, P, -5000.0)
    print(f"  PH闪蒸结果: {ph_result.converged}, {ph_result.error_message}")
    
    return ideal_pp, result

def demo_peng_robinson_package():
    """演示Peng-Robinson物性包"""
    print_separator("Peng-Robinson状态方程演示")
    
    try:
        # 创建化合物和物性包
        compounds = [COMMON_COMPOUNDS["甲烷"], COMMON_COMPOUNDS["乙烷"]]
        pr_pp = PengRobinsonPackage(compounds)
        
        print(f"\nPR物性包信息:")
        model_info = pr_pp.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # 设置计算条件
        T = 250.0  # K
        P = 2000000.0  # Pa (20 bar)
        z = np.array([0.6, 0.4])  # 甲烷60%, 乙烷40%
        
        print(f"\n计算条件:")
        print(f"  温度: {T} K")
        print(f"  压力: {P/1000000:.1f} MPa")
        print(f"  组成: 甲烷{z[0]*100:.1f}%, 乙烷{z[1]*100:.1f}%")
        
        # 创建相来测试单相性质
        print(f"\n单相性质计算:")
        vapor_phase = Phase(PhaseType.VAPOR, compounds, z)
        liquid_phase = Phase(PhaseType.LIQUID, compounds, z)
        
        vapor_phase.set_temperature_pressure(T, P)
        liquid_phase.set_temperature_pressure(T, P)
        
        try:
            # 压缩因子
            z_v = pr_pp.calculate_compressibility_factor(vapor_phase, T, P)
            z_l = pr_pp.calculate_compressibility_factor(liquid_phase, T, P)
            print(f"  气相压缩因子: {z_v:.4f}")
            print(f"  液相压缩因子: {z_l:.4f}")
            
            # 逸度系数
            phi_v = pr_pp.calculate_fugacity_coefficient(vapor_phase, T, P)
            phi_l = pr_pp.calculate_fugacity_coefficient(liquid_phase, T, P)
            print(f"  气相逸度系数: {phi_v}")
            print(f"  液相逸度系数: {phi_l}")
            
            # 焓和熵偏差
            h_dep_v = pr_pp.calculate_enthalpy_departure(vapor_phase, T, P)
            s_dep_v = pr_pp.calculate_entropy_departure(vapor_phase, T, P)
            print(f"  气相焓偏差: {h_dep_v:.2f} J/mol")
            print(f"  气相熵偏差: {s_dep_v:.2f} J/mol/K")
            
        except Exception as e:
            print(f"  单相性质计算失败: {e}")
        
        # 进行PR闪蒸
        print(f"\n进行PR-PT闪蒸计算...")
        result = pr_pp.flash_pt(z, P, T)
        
        print(f"\nPR闪蒸结果:")
        print(f"  收敛状态: {result.converged}")
        print(f"  迭代次数: {result.iterations}")
        print(f"  最终残差: {result.residual:.2e}")
        print(f"  汽化率: {result.vapor_fraction:.4f}")
        
        if result.converged:
            print(f"  焓: {result.enthalpy:.2f} J/mol")
            print(f"  熵: {result.entropy:.2f} J/mol/K")
            print(f"  密度: {result.pressure/(result.volume * 8.314 * T) * compounds[0].properties.molecular_weight:.2f} kg/m³")
            
            if result.vapor_phase:
                print(f"\n  气相组成:")
                for i, comp in enumerate(compounds):
                    print(f"    {comp.name}: {result.vapor_phase.mole_fractions[i]:.4f}")
            
            if result.liquid_phase:
                print(f"\n  液相组成:")
                for i, comp in enumerate(compounds):
                    print(f"    {result.liquid_phase.mole_fractions[i]:.4f}")
        else:
            print(f"  错误信息: {result.error_message}")
        
        return pr_pp, result
        
    except Exception as e:
        print(f"❌ Peng-Robinson演示失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def demo_industrial_case():
    """工业案例演示"""
    print_separator("工业案例演示：天然气脱水")
    
    try:
        # 天然气脱水案例
        print("\n案例：天然气中的水含量计算")
        print("条件：高压天然气管道, 50°C, 50 bar")
        
        # 创建组分
        compounds = [
            COMMON_COMPOUNDS["水"],
            COMMON_COMPOUNDS["甲烷"],
            COMMON_COMPOUNDS["乙烷"]
        ]
        
        # 工业条件
        T = 323.15  # K (50°C)
        P = 5000000.0  # Pa (50 bar)
        
        # 典型天然气组成（含微量水）
        z = np.array([0.001, 0.899, 0.100])  # 水0.1%, 甲烷89.9%, 乙烷10%
        
        print(f"\n进料条件:")
        print(f"  温度: {T-273.15:.1f} °C")
        print(f"  压力: {P/100000:.1f} bar")
        print(f"  组成: 水{z[0]*100:.1f}%, 甲烷{z[1]*100:.1f}%, 乙烷{z[2]*100:.1f}%")
        
        # 使用PR物性包
        pr_pp = PengRobinsonPackage(compounds)
        
        # 闪蒸计算
        print(f"\n进行相平衡计算...")
        result = pr_pp.flash_pt(z, P, T)
        
        if result.converged:
            print(f"\n计算结果:")
            print(f"  汽化率: {result.vapor_fraction:.6f}")
            print(f"  液相分率: {1-result.vapor_fraction:.6f}")
            
            if result.vapor_phase and result.liquid_phase:
                vapor_water = result.vapor_phase.mole_fractions[0]
                liquid_water = result.liquid_phase.mole_fractions[0]
                
                print(f"\n水含量分布:")
                print(f"  气相中水含量: {vapor_water*1000000:.2f} ppm")
                print(f"  液相中水含量: {liquid_water*100:.2f} %")
                
                # 实际工程应用数据
                print(f"\n工程意义:")
                if vapor_water > 0.0001:  # 100 ppm
                    print(f"  ⚠️  气相水含量过高，需要脱水处理")
                else:
                    print(f"  ✅ 气相水含量在可接受范围内")
                
                if result.liquid_phase and (1-result.vapor_fraction) > 0.001:
                    print(f"  💧 有自由水析出，需要气液分离")
            
        else:
            print(f"❌ 计算失败: {result.error_message}")
        
        # 对比不同温度的影响
        print(f"\n温度影响分析:")
        temperatures = [283.15, 298.15, 323.15, 348.15]  # 10, 25, 50, 75°C
        
        for temp in temperatures:
            temp_result = pr_pp.flash_pt(z, P, temp)
            if temp_result.converged and temp_result.vapor_phase:
                water_content = temp_result.vapor_phase.mole_fractions[0] * 1000000
                print(f"  {temp-273.15:4.0f}°C: {water_content:8.1f} ppm 水")
        
    except Exception as e:
        print(f"❌ 工业案例演示失败: {e}")
        import traceback
        traceback.print_exc()

def demo_performance_stats():
    """性能统计演示"""
    print_separator("性能统计信息")
    
    try:
        # 创建物性包
        compounds = [COMMON_COMPOUNDS["甲烷"], COMMON_COMPOUNDS["乙烷"]]
        ideal_pp = IdealPropertyPackage(compounds)
        pr_pp = PengRobinsonPackage(compounds)
        
        # 进行多次计算
        print("\n进行性能测试...")
        n_calculations = 10
        
        T = 250.0
        P = 1000000.0
        z = np.array([0.5, 0.5])
        
        # 理想气体计算
        for i in range(n_calculations):
            ideal_pp.flash_pt(z, P, T + i*10)
        
        # PR计算
        for i in range(n_calculations):
            pr_pp.flash_pt(z, P, T + i*10)
        
        # 显示统计信息
        print(f"\n理想气体物性包统计:")
        ideal_stats = ideal_pp.get_calculation_stats()
        for key, value in ideal_stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nPeng-Robinson物性包统计:")
        pr_stats = pr_pp.get_calculation_stats()
        for key, value in pr_stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ 性能统计失败: {e}")

def main():
    """主函数"""
    print("🚀 DWSIM热力学计算库演示程序")
    print("作者：OpenAspen项目组")
    print("版本：1.0.0")
    
    try:
        # 基础功能演示
        compounds = demo_basic_compounds()
        
        # 相操作演示
        vapor_phase, liquid_phase = demo_phase_operations()
        
        # 理想气体物性包演示
        ideal_pp, ideal_result = demo_ideal_property_package()
        
        # Peng-Robinson物性包演示
        pr_pp, pr_result = demo_peng_robinson_package()
        
        # 工业案例演示
        demo_industrial_case()
        
        # 性能统计
        demo_performance_stats()
        
        # 总结
        print_separator("演示完成总结")
        print("\n✅ 成功演示了以下功能：")
        print("  📋 化合物创建和物性数据管理")
        print("  🔬 相的创建和组成管理")
        print("  💨 理想气体物性包计算")
        print("  🔧 Peng-Robinson状态方程")
        print("  ⚡ PT闪蒸计算")
        print("  🏭 工业案例：天然气脱水")
        print("  📊 性能统计和监控")
        
        print(f"\n🎯 DWSIM热力学计算库已就绪，可用于:")
        print("  • 相平衡计算")
        print("  • 热力学性质预测")
        print("  • 工艺模拟")
        print("  • 设备设计计算")
        
    except Exception as e:
        print(f"\n❌ 演示程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n🏁 演示程序执行完成！")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
