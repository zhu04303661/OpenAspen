# DWSIM.Thermodynamics 调用关系图和时序图
## Call Relationship and Sequence Diagrams

**文档版本**: 1.0  
**创建日期**: 2024年12月  
**描述**: DWSIM热力学计算库详细调用关系和时序分析

---

## 1. 总体调用关系图

```mermaid
graph TD
    subgraph "用户层 (User Layer)"
        USER[用户应用程序]
        EXCEL[Excel插件]
        CAPEOPEN_CLIENT[CAPE-OPEN客户端]
    end
    
    subgraph "接口层 (Interface Layer)"
        THERMO_API[Thermodynamics.vb<br/>主API接口]
        EXCEL_API[Excel.vb<br/>Excel接口]
        CAPEOPEN_API[CAPE-OPEN.vb<br/>标准接口]
        SHORTCUTS[ShortcutUtilities.vb<br/>快捷工具]
    end
    
    subgraph "核心计算层 (Core Layer)"
        PP_FACTORY[PropertyPackage<br/>工厂方法]
        FLASH_FACTORY[FlashAlgorithm<br/>工厂方法]
        
        subgraph "物性包调用链"
            PP_BASE[PropertyPackage.vb]
            PP_SRK[SoaveRedlichKwong.vb]
            PP_PR[PengRobinson.vb]
            PP_ACTIVITY[ActivityCoeff Models]
            PP_SPECIAL[Special Models]
        end
        
        subgraph "闪蒸算法调用链"
            FLASH_BASE[FlashAlgorithmBase.vb]
            FLASH_NL[NestedLoops.vb]
            FLASH_IO[InsideOut.vb]
            FLASH_GIBBS[GibbsMinimization.vb]
        end
    end
    
    subgraph "基础服务层 (Base Services)"
        THERMO_BASE[ThermodynamicsBase.vb]
        MICHELSEN[MichelsenBase.vb]
        PROPERTY_METHODS[PropertyMethods.vb]
        ELECTROLYTE_PROPS[ElectrolyteProperties.vb]
    end
    
    subgraph "数据服务层 (Data Services)"
        DATABASES[Databases]
        RESOURCES[Resources]
        HELPERS[Helper Classes]
    end
    
    %% 用户层调用
    USER --> THERMO_API
    EXCEL --> EXCEL_API
    CAPEOPEN_CLIENT --> CAPEOPEN_API
    
    %% 接口层调用
    THERMO_API --> PP_FACTORY
    THERMO_API --> FLASH_FACTORY
    EXCEL_API --> THERMO_API
    CAPEOPEN_API --> THERMO_API
    SHORTCUTS --> PP_BASE
    
    %% 工厂调用
    PP_FACTORY --> PP_BASE
    FLASH_FACTORY --> FLASH_BASE
    
    %% 物性包调用链
    PP_BASE --> THERMO_BASE
    PP_BASE --> PROPERTY_METHODS
    PP_SRK --> PP_BASE
    PP_PR --> PP_BASE
    PP_ACTIVITY --> PP_BASE
    PP_SPECIAL --> PP_BASE
    
    %% 闪蒸算法调用链
    FLASH_BASE --> MICHELSEN
    FLASH_BASE --> PP_BASE
    FLASH_NL --> FLASH_BASE
    FLASH_IO --> FLASH_BASE
    FLASH_GIBBS --> FLASH_BASE
    
    %% 基础服务调用
    THERMO_BASE --> DATABASES
    PROPERTY_METHODS --> RESOURCES
    ELECTROLYTE_PROPS --> HELPERS
    
    classDef userLayer fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef interfaceLayer fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef coreLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef baseLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef dataLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class USER,EXCEL,CAPEOPEN_CLIENT userLayer
    class THERMO_API,EXCEL_API,CAPEOPEN_API,SHORTCUTS interfaceLayer
    class PP_FACTORY,FLASH_FACTORY,PP_BASE,PP_SRK,PP_PR,FLASH_BASE,FLASH_NL,FLASH_IO coreLayer
    class THERMO_BASE,MICHELSEN,PROPERTY_METHODS,ELECTROLYTE_PROPS baseLayer
    class DATABASES,RESOURCES,HELPERS dataLayer
```

## 2. 物性计算调用关系图

```mermaid
graph TD
    subgraph "物性计算调用链"
        CLIENT[客户端调用]
        
        subgraph "API层"
            CALC_PROP[CalculateProperties()]
            DW_CALC[DW_Calc系列方法]
        end
        
        subgraph "物性包层"
            PP_MAIN[PropertyPackage主类]
            PP_SPECIFIC[具体物性包实现]
        end
        
        subgraph "计算方法层"
            ENTHALPY[DW_CalcEnthalpy()]
            ENTROPY[DW_CalcEntropy()]
            FUGACITY[DW_CalcFugCoeff()]
            ACTIVITY[DW_CalcActivityCoeff()]
            KVALUE[DW_CalcKvalue()]
            DENSITY[DW_CalcDensity()]
            CP[DW_CalcCp()]
        end
        
        subgraph "基础计算层"
            EOS_CALC[状态方程计算]
            ACTIVITY_CALC[活度系数计算]
            MIXING_RULES[混合规则]
            PURE_PROPS[纯组分性质]
        end
        
        subgraph "数据层"
            COMPOUND_DATA[化合物数据]
            BIP_DATA[二元交互参数]
            CORRELATION[关联式数据]
        end
    end
    
    %% 调用关系
    CLIENT --> CALC_PROP
    CALC_PROP --> DW_CALC
    DW_CALC --> PP_MAIN
    PP_MAIN --> PP_SPECIFIC
    
    PP_SPECIFIC --> ENTHALPY
    PP_SPECIFIC --> ENTROPY
    PP_SPECIFIC --> FUGACITY
    PP_SPECIFIC --> ACTIVITY
    PP_SPECIFIC --> KVALUE
    PP_SPECIFIC --> DENSITY
    PP_SPECIFIC --> CP
    
    ENTHALPY --> EOS_CALC
    ENTROPY --> EOS_CALC
    FUGACITY --> EOS_CALC
    ACTIVITY --> ACTIVITY_CALC
    KVALUE --> EOS_CALC
    KVALUE --> ACTIVITY_CALC
    DENSITY --> EOS_CALC
    CP --> EOS_CALC
    
    EOS_CALC --> MIXING_RULES
    EOS_CALC --> PURE_PROPS
    ACTIVITY_CALC --> BIP_DATA
    MIXING_RULES --> BIP_DATA
    PURE_PROPS --> COMPOUND_DATA
    PURE_PROPS --> CORRELATION
    
    classDef apiLayer fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef ppLayer fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef methodLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef calcLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef dataLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class CLIENT,CALC_PROP,DW_CALC apiLayer
    class PP_MAIN,PP_SPECIFIC ppLayer
    class ENTHALPY,ENTROPY,FUGACITY,ACTIVITY,KVALUE,DENSITY,CP methodLayer
    class EOS_CALC,ACTIVITY_CALC,MIXING_RULES,PURE_PROPS calcLayer
    class COMPOUND_DATA,BIP_DATA,CORRELATION dataLayer
```

## 3. 闪蒸计算调用关系图

```mermaid
graph TD
    subgraph "闪蒸计算调用链"
        FLASH_CLIENT[闪蒸计算请求]
        
        subgraph "闪蒸API层"
            FLASH_PT[Flash_PT()]
            FLASH_PH[Flash_PH()]
            FLASH_PS[Flash_PS()]
            FLASH_TV[Flash_TV()]
        end
        
        subgraph "算法选择层"
            ALGORITHM_FACTORY[算法工厂]
            NESTED_LOOPS[嵌套循环算法]
            INSIDE_OUT[Inside-Out算法]
            GIBBS_MIN[Gibbs最小化算法]
        end
        
        subgraph "算法执行层"
            INIT_FLASH[初始化闪蒸]
            OUTER_LOOP[外循环]
            INNER_LOOP[内循环]
            CONVERGENCE_CHECK[收敛检查]
            STABILITY_TEST[稳定性测试]
        end
        
        subgraph "物性调用层"
            K_VALUE_CALC[K值计算]
            FUGACITY_CALC[逸度系数计算]
            ACTIVITY_CALC[活度系数计算]
            PHASE_PROPS[相性质计算]
        end
        
        subgraph "数值求解层"
            NEWTON_RAPHSON[Newton-Raphson求解]
            BRENT_METHOD[Brent方法]
            WEGSTEIN_ACCEL[Wegstein加速]
            LINE_SEARCH[线搜索]
        end
    end
    
    %% 调用关系
    FLASH_CLIENT --> FLASH_PT
    FLASH_CLIENT --> FLASH_PH
    FLASH_CLIENT --> FLASH_PS
    FLASH_CLIENT --> FLASH_TV
    
    FLASH_PT --> ALGORITHM_FACTORY
    FLASH_PH --> ALGORITHM_FACTORY
    FLASH_PS --> ALGORITHM_FACTORY
    FLASH_TV --> ALGORITHM_FACTORY
    
    ALGORITHM_FACTORY --> NESTED_LOOPS
    ALGORITHM_FACTORY --> INSIDE_OUT
    ALGORITHM_FACTORY --> GIBBS_MIN
    
    NESTED_LOOPS --> INIT_FLASH
    INSIDE_OUT --> INIT_FLASH
    GIBBS_MIN --> INIT_FLASH
    
    INIT_FLASH --> OUTER_LOOP
    OUTER_LOOP --> INNER_LOOP
    INNER_LOOP --> CONVERGENCE_CHECK
    CONVERGENCE_CHECK --> STABILITY_TEST
    
    OUTER_LOOP --> K_VALUE_CALC
    INNER_LOOP --> FUGACITY_CALC
    INNER_LOOP --> ACTIVITY_CALC
    CONVERGENCE_CHECK --> PHASE_PROPS
    
    K_VALUE_CALC --> NEWTON_RAPHSON
    FUGACITY_CALC --> BRENT_METHOD
    OUTER_LOOP --> WEGSTEIN_ACCEL
    STABILITY_TEST --> LINE_SEARCH
    
    classDef clientLayer fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef apiLayer fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef algorithmLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef executionLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef propertyLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef numericalLayer fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    
    class FLASH_CLIENT clientLayer
    class FLASH_PT,FLASH_PH,FLASH_PS,FLASH_TV apiLayer
    class ALGORITHM_FACTORY,NESTED_LOOPS,INSIDE_OUT,GIBBS_MIN algorithmLayer
    class INIT_FLASH,OUTER_LOOP,INNER_LOOP,CONVERGENCE_CHECK,STABILITY_TEST executionLayer
    class K_VALUE_CALC,FUGACITY_CALC,ACTIVITY_CALC,PHASE_PROPS propertyLayer
    class NEWTON_RAPHSON,BRENT_METHOD,WEGSTEIN_ACCEL,LINE_SEARCH numericalLayer
```

## 4. PT闪蒸计算时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant API as Thermodynamics API
    participant Factory as Algorithm Factory
    participant Algorithm as Flash Algorithm
    participant PP as Property Package
    participant Base as Base Classes
    participant Solver as Numerical Solver
    
    Client->>API: Flash_PT(T, P, z)
    API->>Factory: CreateFlashAlgorithm("NestedLoops")
    Factory->>Algorithm: new NestedLoops()
    API->>Algorithm: Flash_PT(T, P, z, PP)
    
    Algorithm->>Algorithm: InitializeFlash()
    Algorithm->>PP: DW_CalcKvalue(T, P, z)
    PP->>Base: CalculateFugacityCoeff()
    Base->>PP: fugacity_coefficients
    PP->>Algorithm: K_values
    
    Algorithm->>Algorithm: EstimateVaporFraction()
    
    loop 外循环 (Outer Loop)
        Algorithm->>Algorithm: UpdateCompositions()
        
        loop 内循环 (Inner Loop)
            Algorithm->>PP: DW_CalcFugCoeff(T, P, x_liquid)
            PP->>Algorithm: phi_liquid
            Algorithm->>PP: DW_CalcFugCoeff(T, P, y_vapor)
            PP->>Algorithm: phi_vapor
            Algorithm->>Algorithm: UpdateKValues()
            Algorithm->>Algorithm: CheckInnerConvergence()
        end
        
        Algorithm->>Solver: SolveVaporFraction()
        Solver->>Algorithm: new_vapor_fraction
        Algorithm->>Algorithm: CheckOuterConvergence()
        
        alt 未收敛
            Algorithm->>Algorithm: WegsteinAcceleration()
        end
    end
    
    Algorithm->>Algorithm: CheckStability()
    Algorithm->>PP: CalculatePhaseProperties()
    PP->>Algorithm: phase_properties
    Algorithm->>API: FlashResult
    API->>Client: 闪蒸结果
```

## 5. 物性计算时序图

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant API as Thermodynamics API
    participant PP as Property Package
    participant EOS as 状态方程
    participant Activity as 活度系数模型
    participant Data as 数据库
    
    Client->>API: CalculateProperties(compounds, T, P, composition)
    API->>PP: CreatePropertyPackage("SRK")
    
    PP->>Data: LoadCompoundData(compounds)
    Data->>PP: compound_properties
    PP->>Data: LoadBinaryParameters(compounds)
    Data->>PP: binary_parameters
    
    Client->>PP: DW_CalcEnthalpy(T, P, composition)
    PP->>EOS: CalculateEnthalpyDeparture()
    EOS->>EOS: SolveCubicEOS()
    EOS->>EOS: CalculateCompressibilityFactor()
    EOS->>PP: enthalpy_departure
    PP->>PP: CalculateIdealGasEnthalpy()
    PP->>Client: total_enthalpy
    
    Client->>PP: DW_CalcFugCoeff(T, P, composition)
    PP->>EOS: CalculateFugacityCoeff()
    EOS->>EOS: CalculateMixingRules()
    EOS->>EOS: SolveCubicEOS()
    EOS->>PP: fugacity_coefficients
    PP->>Client: fugacity_coefficients
    
    Client->>PP: DW_CalcActivityCoeff(T, composition)
    PP->>Activity: CalculateActivityCoeff()
    Activity->>Activity: CalculateLocalComposition()
    Activity->>Activity: CalculateExcessGibbs()
    Activity->>PP: activity_coefficients
    PP->>Client: activity_coefficients
```

## 6. 错误处理和异常流程图

```mermaid
graph TD
    subgraph "错误处理流程"
        INPUT[输入数据]
        VALIDATE[数据验证]
        
        subgraph "验证检查"
            CHECK_COMP[化合物检查]
            CHECK_COND[条件检查]
            CHECK_PARAMS[参数检查]
        end
        
        subgraph "计算执行"
            CALC_START[开始计算]
            CALC_PROCESS[计算过程]
            CONVERGENCE[收敛检查]
        end
        
        subgraph "异常处理"
            CONV_ERROR[收敛失败]
            DATA_ERROR[数据错误]
            CALC_ERROR[计算错误]
            SYSTEM_ERROR[系统错误]
        end
        
        subgraph "错误恢复"
            RETRY[重试机制]
            FALLBACK[备用算法]
            DEFAULT[默认值]
            REPORT[错误报告]
        end
        
        OUTPUT[输出结果]
    end
    
    %% 正常流程
    INPUT --> VALIDATE
    VALIDATE --> CHECK_COMP
    VALIDATE --> CHECK_COND
    VALIDATE --> CHECK_PARAMS
    
    CHECK_COMP --> CALC_START
    CHECK_COND --> CALC_START
    CHECK_PARAMS --> CALC_START
    
    CALC_START --> CALC_PROCESS
    CALC_PROCESS --> CONVERGENCE
    CONVERGENCE --> OUTPUT
    
    %% 异常流程
    CHECK_COMP -->|验证失败| DATA_ERROR
    CHECK_COND -->|验证失败| DATA_ERROR
    CHECK_PARAMS -->|验证失败| DATA_ERROR
    
    CALC_PROCESS -->|计算异常| CALC_ERROR
    CONVERGENCE -->|不收敛| CONV_ERROR
    CALC_PROCESS -->|系统异常| SYSTEM_ERROR
    
    %% 错误恢复
    CONV_ERROR --> RETRY
    CONV_ERROR --> FALLBACK
    DATA_ERROR --> DEFAULT
    CALC_ERROR --> RETRY
    SYSTEM_ERROR --> REPORT
    
    RETRY --> CALC_PROCESS
    FALLBACK --> CALC_PROCESS
    DEFAULT --> OUTPUT
    REPORT --> OUTPUT
    
    classDef normalFlow fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef errorFlow fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef recoveryFlow fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class INPUT,VALIDATE,CHECK_COMP,CHECK_COND,CHECK_PARAMS,CALC_START,CALC_PROCESS,CONVERGENCE,OUTPUT normalFlow
    class CONV_ERROR,DATA_ERROR,CALC_ERROR,SYSTEM_ERROR errorFlow
    class RETRY,FALLBACK,DEFAULT,REPORT recoveryFlow
```

## 7. 性能监控调用图

```mermaid
graph TD
    subgraph "性能监控系统"
        MONITOR_START[监控开始]
        
        subgraph "性能指标收集"
            TIME_MEASURE[执行时间测量]
            MEMORY_MEASURE[内存使用测量]
            ITERATION_COUNT[迭代次数统计]
            CONVERGENCE_RATE[收敛率统计]
        end
        
        subgraph "性能分析"
            BOTTLENECK[瓶颈分析]
            EFFICIENCY[效率分析]
            COMPARISON[算法比较]
            TREND[趋势分析]
        end
        
        subgraph "优化建议"
            ALGORITHM_SUGGEST[算法建议]
            PARAMETER_TUNE[参数调优]
            CACHE_OPTIMIZE[缓存优化]
            PARALLEL_SUGGEST[并行化建议]
        end
        
        MONITOR_END[监控结束]
        REPORT_GEN[生成报告]
    end
    
    %% 监控流程
    MONITOR_START --> TIME_MEASURE
    MONITOR_START --> MEMORY_MEASURE
    MONITOR_START --> ITERATION_COUNT
    MONITOR_START --> CONVERGENCE_RATE
    
    TIME_MEASURE --> BOTTLENECK
    MEMORY_MEASURE --> EFFICIENCY
    ITERATION_COUNT --> COMPARISON
    CONVERGENCE_RATE --> TREND
    
    BOTTLENECK --> ALGORITHM_SUGGEST
    EFFICIENCY --> PARAMETER_TUNE
    COMPARISON --> CACHE_OPTIMIZE
    TREND --> PARALLEL_SUGGEST
    
    ALGORITHM_SUGGEST --> MONITOR_END
    PARAMETER_TUNE --> MONITOR_END
    CACHE_OPTIMIZE --> MONITOR_END
    PARALLEL_SUGGEST --> MONITOR_END
    
    MONITOR_END --> REPORT_GEN
    
    classDef monitorClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef measureClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef analysisClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef optimizeClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class MONITOR_START,MONITOR_END,REPORT_GEN monitorClass
    class TIME_MEASURE,MEMORY_MEASURE,ITERATION_COUNT,CONVERGENCE_RATE measureClass
    class BOTTLENECK,EFFICIENCY,COMPARISON,TREND analysisClass
    class ALGORITHM_SUGGEST,PARAMETER_TUNE,CACHE_OPTIMIZE,PARALLEL_SUGGEST optimizeClass
```

---

## 调用关系总结

### 1. 分层调用架构
- **用户层**: 提供多种接口入口
- **API层**: 统一的接口抽象
- **核心层**: 具体算法实现
- **基础层**: 通用服务和工具
- **数据层**: 数据存储和管理

### 2. 关键调用路径
- **物性计算路径**: Client → API → PropertyPackage → EOS/Activity → Base → Data
- **闪蒸计算路径**: Client → API → FlashAlgorithm → PropertyPackage → Solver
- **错误处理路径**: Exception → Handler → Recovery → Report

### 3. 性能优化点
- **缓存机制**: 避免重复计算
- **算法选择**: 根据问题特征选择最优算法
- **并行计算**: 支持多线程计算
- **内存管理**: 高效的内存使用

### 4. 扩展性设计
- **工厂模式**: 支持新算法和模型的添加
- **策略模式**: 支持算法的动态切换
- **观察者模式**: 支持计算过程的监控
- **适配器模式**: 支持外部库的集成

---

**文档状态**: ✅ 完成  
**最后更新**: 2024年12月  
**维护者**: OpenAspen项目组 