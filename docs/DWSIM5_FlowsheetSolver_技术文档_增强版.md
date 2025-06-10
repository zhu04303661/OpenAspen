# DWSIM5 FlowsheetSolver 技术文档 (增强版)

## 1. 概述

FlowsheetSolver是DWSIM5流程模拟软件的核心求解引擎，负责整个流程图的计算调度、依赖解析、收敛检查和异步计算。它是业务逻辑层的关键组件，协调各个单元操作的计算顺序，处理循环流程和同步调节，确保整个流程图的成功求解。

### 1.1 核心职责

- **计算调度**: 确定流程图中对象的计算顺序和依赖关系
- **依赖解析**: 分析对象间的连接关系，构建计算拓扑图
- **收敛检查**: 处理循环流程的收敛性判断和迭代求解
- **异步计算**: 支持多线程和并行计算模式
- **远程求解**: 支持Azure云计算和TCP网络分布式计算
- **错误处理**: 捕获和处理计算过程中的异常情况

### 1.2 技术特点

- 基于VB.NET开发，面向.NET Framework 4.6.1
- 支持同步和异步计算模式
- 内置多种任务调度器（STA、限制并发级别等）
- 集成Broyden方法加速收敛
- 支持分布式计算和云计算

## 2. 核心模块结构图

### 2.1 系统架构总览 (Mermaid)

```mermaid
graph TB
    subgraph "DWSIM5 FlowsheetSolver 核心架构"
        subgraph "主控制器层"
            A1[SolveFlowsheet<br/>主求解器]
            A2[ProcessCalculationQueue<br/>队列处理器] 
            A3[CalculateObject<br/>对象计算器]
            A1 --> A2
            A2 --> A3
        end
        
        subgraph "计算调度层"
            B1[GetSolvingList<br/>求解列表生成]
            B2[DependencyAnalysis<br/>依赖关系分析]
            B3[CalculationOrder<br/>计算顺序优化]
            B1 --> B2 --> B3
        end
        
        subgraph "收敛求解层"
            C1[RecycleSolver<br/>循环求解器]
            C2[SimultaneousAdjust<br/>同步调节求解器]
            C3[ConvergenceCheck<br/>收敛检查器]
            C1 <--> C2
            C2 <--> C3
        end
        
        subgraph "并行计算层"
            D1[TaskScheduler<br/>任务调度器]
            D2[ThreadManagement<br/>线程管理器]
            D3[ResourceManager<br/>资源管理器]
            D1 --> D2 --> D3
        end
        
        subgraph "远程计算层"
            E1[AzureSolverClient<br/>Azure云计算客户端]
            E2[TCPSolverClient<br/>TCP网络客户端]
            E3[DistributedManager<br/>分布式管理器]
            E1 --> E3
            E2 --> E3
        end
        
        subgraph "事件与监控层"
            F1[EventSystem<br/>事件系统]
            F2[ErrorHandler<br/>错误处理器]
            F3[Diagnostics<br/>诊断系统]
            F1 --> F2 --> F3
        end
        
        A1 --> B1
        B3 --> C1
        C3 --> D1
        D3 --> E3
        A1 --> F1
    end
```

### 2.2 模块交互关系图 (Mermaid)

```mermaid
sequenceDiagram
    participant User as 用户请求
    participant SF as SolveFlowsheet
    participant GSL as GetSolvingList
    participant DA as 依赖分析
    participant PQ as ProcessQueue
    participant CO as CalculateObject
    participant RS as RecycleSolver
    participant CC as ConvergenceCheck
    participant TS as TaskScheduler
    participant TM as ThreadManagement
    participant ES as EventSystem
    participant EH as ErrorHandler

    User->>SF: 发起求解请求
    SF->>GSL: 获取求解列表
    GSL->>DA: 分析依赖关系
    DA-->>GSL: 返回依赖结果
    GSL-->>SF: 返回计算顺序
    SF->>PQ: 处理计算队列
    PQ->>CO: 计算对象
    CO->>RS: 循环求解
    RS->>CC: 收敛检查
    CC-->>RS: 收敛结果
    RS-->>CO: 完成循环计算
    CO->>TS: 任务调度
    TS->>TM: 线程管理
    TM-->>TS: 执行完成
    TS-->>CO: 并行计算完成
    CO-->>PQ: 对象计算完成
    PQ-->>SF: 队列处理完成
    SF->>ES: 触发事件
    ES->>EH: 处理结果
    EH-->>User: 返回结果
```

### 2.3 详细组件架构图 (ASCII - 原版保留)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DWSIM5 FlowsheetSolver 核心架构                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              主控制器层                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐ │
│  │    SolveFlowsheet   │   │  ProcessCalculation │   │    CalculateObject     │ │
│  │     (主求解器)      │◄──┤      Queue          │◄──┤     (对象计算器)       │ │
│  │                     │   │   (队列处理器)      │   │                         │ │
│  │ • 模式选择          │   │ • 队列管理          │   │ • 对象分派              │ │
│  │ • 全局控制          │   │ • 异常处理          │   │ • 类型识别              │ │
│  │ • 资源管理          │   │ • 进度监控          │   │ • 状态更新              │ │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              计算调度层                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐ │
│  │   GetSolvingList    │   │     Dependency      │   │   CalculationOrder     │ │
│  │   (求解列表生成)    │──►│     Analysis        │──►│    Optimization         │ │
│  │                     │   │   (依赖关系分析)    │   │   (计算顺序优化)        │ │
│  │ • 拓扑排序          │   │ • 连接分析          │   │ • 层次分组              │ │
│  │ • 环路检测          │   │ • 数据流追踪        │   │ • 并行识别              │ │
│  │ • 终点识别          │   │ • 约束检查          │   │ • 优先级排序            │ │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              收敛求解层                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐ │
│  │  RecycleSolver      │   │ SimultaneousAdjust  │   │   ConvergenceCheck     │ │
│  │   (循环求解器)      │   │    Solver           │   │    (收敛检查器)        │ │
│  │                     │   │  (同步调节求解器)   │   │                         │ │
│  │ • Broyden加速       │   │ • Newton迭代        │   │ • 误差计算              │ │
│  │ • 误差函数          │   │ • 雅可比矩阵        │   │ • 容差检查              │ │
│  │ • 变量更新          │   │ • 梯度计算          │   │ • 振荡检测              │ │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              并行计算层                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐ │
│  │   TaskScheduler     │   │   ThreadManagement  │   │   ResourceManager      │ │
│  │   (任务调度器)      │   │    (线程管理器)     │   │    (资源管理器)        │ │
│  │                     │   │                     │   │                         │ │
│  │ • 并发控制          │   │ • 线程池管理        │   │ • 内存分配              │ │
│  │ • 任务分配          │   │ • STA支持           │   │ • GPU资源               │ │
│  │ • 负载均衡          │   │ • 同步机制          │   │ • 缓存策略              │ │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              远程计算层                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐ │
│  │  AzureSolverClient  │   │  TCPSolverClient    │   │   DistributedManager   │ │
│  │  (Azure云计算客户端)│   │  (TCP网络客户端)    │   │   (分布式管理器)       │ │
│  │                     │   │                     │   │                         │ │
│  │ • 服务总线通信      │   │ • TCP连接管理       │   │ • 负载分发              │ │
│  │ • 数据压缩传输      │   │ • 协议处理          │   │ • 结果聚合              │ │
│  │ • 消息队列管理      │   │ • 心跳检测          │   │ • 故障恢复              │ │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              事件与监控层                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐ │
│  │   EventSystem       │   │   ErrorHandler      │   │    Diagnostics         │ │
│  │   (事件系统)        │   │   (错误处理器)      │   │    (诊断系统)          │ │
│  │                     │   │                     │   │                         │ │
│  │ • 事件发布          │   │ • 异常捕获          │   │ • 性能监控              │ │
│  │ • 生命周期跟踪      │   │ • 错误恢复          │   │ • 日志记录              │ │
│  │ • 脚本触发          │   │ • 用户通知          │   │ • 调试信息              │ │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 3. 核心算法与计算公式

### 3.1 拓扑排序算法

**算法目标**: 确定流程图对象的计算顺序，确保依赖关系正确

**数学表示**:
设流程图为有向图 G(V, E)，其中：
- V = {v₁, v₂, ..., vₙ} 为对象集合  
- E ⊆ V × V 为连接关系集合

**Kahn算法实现**:
```
1. 计算入度: indegree[v] = |{u ∈ V : (u,v) ∈ E}|
2. 初始化队列: Q = {v ∈ V : indegree[v] = 0}
3. While Q ≠ ∅:
   a. v = Q.dequeue()
   b. result.add(v)
   c. For each (v,u) ∈ E:
      - indegree[u]--
      - if indegree[u] = 0: Q.enqueue(u)
```

**复杂度**: O(V + E)

#### 3.1.1 拓扑排序算法流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - 拓扑排序]) --> CalcIndegree[计算所有节点入度<br/>indegree[v] = |{u ∈ V : (u,v) ∈ E}|]
    CalcIndegree --> InitQueue[初始化队列<br/>Q = {v ∈ V : indegree[v] = 0}]
    InitQueue --> CheckQueue{队列Q是否为空?}
    
    CheckQueue -->|空| ReturnResult[返回拓扑排序结果<br/>result]
    CheckQueue -->|非空| DequeueNode[从队列取出节点<br/>v = Q.dequeue()]
    
    DequeueNode --> AddToResult[添加到结果序列<br/>result.add(v)]
    AddToResult --> ProcessEdges[处理所有出边<br/>For each (v,u) ∈ E]
    
    ProcessEdges --> DecIndegree[减少目标节点入度<br/>indegree[u]--]
    DecIndegree --> CheckIndegree{indegree[u] = 0?}
    CheckIndegree -->|是| EnqueueNode[加入队列<br/>Q.enqueue(u)]
    CheckIndegree -->|否| NextEdge{还有其他出边?}
    
    EnqueueNode --> NextEdge
    NextEdge -->|是| ProcessEdges
    NextEdge -->|否| CheckQueue
    
    ReturnResult --> CheckCycle{检查环路<br/>result.length = |V|?}
    CheckCycle -->|是| Success[排序成功<br/>无环路]
    CheckCycle -->|否| CycleDetected[检测到环路<br/>图不是DAG]
    
    Success --> End([结束])
    CycleDetected --> End
```

### 3.2 Broyden收敛加速算法

**目标**: 求解非线性方程组 F(x) = 0

**Broyden更新公式**:
```
J_{k+1} = J_k + (Δy_k - J_k·Δx_k)(Δx_k^T) / (Δx_k^T·Δx_k)

其中:
- Δx_k = x_{k+1} - x_k (变量增量)
- Δy_k = F(x_{k+1}) - F(x_k) (函数值增量)
- J_k: 雅可比矩阵近似
```

**Sherman-Morrison逆矩阵更新**:
```
J_{k+1}^{-1} = J_k^{-1} + (Δx_k - J_k^{-1}·Δy_k)(Δx_k^T·J_k^{-1}) / (Δx_k^T·J_k^{-1}·Δy_k)
```

**收敛条件**:
```
||F(x_k)|| < ε₁ ∧ ||Δx_k|| < ε₂ ∧ ||Δx_k||/||x_k|| < ε₃
```

#### 3.2.1 Broyden算法流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - Broyden方法]) --> InitVars[初始化变量<br/>x₀, J₀, k=0]
    InitVars --> SetTolerance[设置收敛容差<br/>ε₁, ε₂, ε₃]
    
    SetTolerance --> CalcF[计算函数值<br/>F(x_k)]
    CalcF --> CheckConv1{检查收敛条件<br/>||F(x_k)|| < ε₁?}
    CheckConv1 -->|是| Converged[收敛成功]
    CheckConv1 -->|否| SolveLinear[求解线性系统<br/>J_k · Δx_k = -F(x_k)]
    
    SolveLinear --> UpdateX[更新变量<br/>x_{k+1} = x_k + Δx_k]
    UpdateX --> CheckConv2{检查变量收敛<br/>||Δx_k|| < ε₂?}
    CheckConv2 -->|是| Converged
    CheckConv2 -->|否| CheckConv3{检查相对变化<br/>||Δx_k||/||x_k|| < ε₃?}
    CheckConv3 -->|是| Converged
    CheckConv3 -->|否| CalcNewF[计算新函数值<br/>F(x_{k+1})]
    
    CalcNewF --> CalcDeltas[计算增量<br/>Δy_k = F(x_{k+1}) - F(x_k)]
    CalcDeltas --> BroydenUpdate[Broyden矩阵更新<br/>J_{k+1} = J_k + (Δy_k - J_k·Δx_k)(Δx_k^T)/(Δx_k^T·Δx_k)]
    
    BroydenUpdate --> IncrementK[k = k + 1]
    IncrementK --> CheckMaxIter{k < max_iter?}
    CheckMaxIter -->|是| CalcF
    CheckMaxIter -->|否| MaxIterError[达到最大迭代次数<br/>收敛失败]
    
    Converged --> End([结束 - 成功])
    MaxIterError --> End
```

### 3.3 Newton-Raphson同步调节算法

**多变量Newton方法**:
对于m个调节对象的系统：

```
F(x) = [f₁(x₁,...,xₘ)]   [0]
       [f₂(x₁,...,xₘ)] = [0]
       [     ⋮      ]   [⋮]
       [fₘ(x₁,...,xₘ)]   [0]
```

**雅可比矩阵**:
```
J(x) = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₘ]
       [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₘ]
       [   ⋮        ⋮      ⋱     ⋮   ]
       [∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₘ]
```

**Newton迭代**:
```
J(x_k)·Δx_k = -F(x_k)
x_{k+1} = x_k + Δx_k
```

**数值微分**:
```
∂f_i/∂x_j ≈ [f_i(x + h·e_j) - f_i(x - h·e_j)] / (2h)

其中 h = ε·max(|x_j|, 1)·sign(x_j), ε ≈ 1.49×10⁻⁸
```

#### 3.3.1 Newton-Raphson算法流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - Newton-Raphson]) --> CollectAdjusts[收集调节对象<br/>同步调节列表]
    CollectAdjusts --> InitGuess[设置初始猜值<br/>x₀ = [x₁₀, x₂₀, ..., xₘ₀]]
    InitGuess --> SetParams[设置算法参数<br/>max_iter, tolerance, step_size]
    
    SetParams --> IterLoop{迭代循环<br/>k < max_iter}
    IterLoop -->|继续| CalcObjective[计算目标函数<br/>F(x_k) = [f₁, f₂, ..., fₘ]]
    IterLoop -->|退出| MaxIterExit[达到最大迭代]
    
    CalcObjective --> CheckNormF{检查函数范数<br/>||F(x_k)|| < tolerance?}
    CheckNormF -->|是| SuccessExit[收敛成功]
    CheckNormF -->|否| CalcJacobian[计算雅可比矩阵<br/>J(x_k)]
    
    CalcJacobian --> NumericalDiff[数值微分<br/>∂f_i/∂x_j ≈ [f_i(x+h·e_j) - f_i(x-h·e_j)]/(2h)]
    NumericalDiff --> SolveLinearSys[求解线性系统<br/>J(x_k)·Δx_k = -F(x_k)]
    
    SolveLinearSys --> CheckSingular{矩阵是否奇异?}
    CheckSingular -->|是| SingularError[矩阵奇异错误]
    CheckSingular -->|否| UpdateVars[更新变量<br/>x_{k+1} = x_k + Δx_k]
    
    UpdateVars --> CheckStepSize{检查步长<br/>||Δx_k|| < step_tolerance?}
    CheckStepSize -->|是| StepTooSmall[步长过小<br/>可能收敛]
    CheckStepSize -->|否| UpdateAdjustObjects[更新调节对象<br/>设置新的操作变量]
    
    UpdateAdjustObjects --> IncrementIter[k = k + 1]
    IncrementIter --> IterLoop
    
    SuccessExit --> End([结束 - 成功])
    MaxIterExit --> End
    SingularError --> End
    StepTooSmall --> End
    
    subgraph "雅可比矩阵计算"
        direction TB
        StartJac[开始雅可比计算] --> LoopI{i = 1 to m}
        LoopI -->|继续| LoopJ{j = 1 to m}
        LoopJ -->|继续| CalcPartial[计算偏导数<br/>∂f_i/∂x_j]
        CalcPartial --> LoopJ
        LoopJ -->|完成| LoopI
        LoopI -->|完成| EndJac[雅可比矩阵完成]
    end
    
    CalcJacobian -.-> StartJac
    EndJac -.-> SolveLinearSys
```

### 3.4 计算顺序确定算法 (GetSolvingList)

#### 3.4.1 GetSolvingList算法流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - GetSolvingList]) --> InitLists[初始化列表字典<br/>lists = Dictionary()]
    InitLists --> FindEndpoints[寻找终点对象<br/>无出口连接的流股和设备]
    FindEndpoints --> SetLevel0[设置第0层<br/>lists[0] = endpoints]
    
    SetLevel0 --> StartLoop[开始逆向追踪<br/>listidx = 0]
    StartLoop --> IncrementIdx[listidx += 1]
    IncrementIdx --> InitCurrentLevel[初始化当前层<br/>lists[listidx] = []]
    
    InitCurrentLevel --> LoopPrevLevel{遍历上一层对象<br/>For each obj in lists[listidx-1]}
    LoopPrevLevel -->|继续| CheckInputs[检查输入连接器<br/>For each connector in obj.InputConnectors]
    LoopPrevLevel -->|完成| CheckLevelEmpty{当前层为空?<br/>lists[listidx].Count = 0}
    
    CheckInputs --> IsAttached{连接器已连接?<br/>connector.IsAttached}
    IsAttached -->|是| AddSource[添加源对象<br/>lists[listidx].Add(connector.Source)]
    IsAttached -->|否| NextConnector{下一个连接器}
    
    AddSource --> NextConnector
    NextConnector -->|有| CheckInputs
    NextConnector -->|无| LoopPrevLevel
    
    CheckLevelEmpty -->|是| BuildResult[构建结果列表<br/>倒序合并所有层]
    CheckLevelEmpty -->|否| StartLoop
    
    BuildResult --> RemoveDuplicates[移除重复对象<br/>保持依赖顺序]
    RemoveDuplicates --> ValidateOrder[验证计算顺序<br/>依赖关系检查]
    
    ValidateOrder --> InsertRecycles{需要插入Recycle?<br/>检测到循环}
    InsertRecycles -->|是| AddRecycleObjects[插入Recycle对象<br/>切断循环依赖]
    InsertRecycles -->|否| FinalResult[最终计算顺序]
    
    AddRecycleObjects --> FinalResult
    FinalResult --> End([结束 - 返回objstack])
    
    subgraph "循环检测"
        direction TB
        StartCycle[开始循环检测] --> DFSSearch[深度优先搜索<br/>标记访问状态]
        DFSSearch --> FoundCycle{发现后向边?<br/>指向正在访问的节点}
        FoundCycle -->|是| MarkCycle[标记循环路径]
        FoundCycle -->|否| ContinueSearch[继续搜索]
        MarkCycle --> EndCycle[循环检测完成]
        ContinueSearch --> DFSSearch
    end
    
    ValidateOrder -.-> StartCycle
    EndCycle -.-> InsertRecycles
```

## 4. 系统工作流程图

### 4.1 主求解流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始]) --> CheckStatus{检查求解器状态<br/>CalculatorBusy}
    CheckStatus -->|忙碌状态| WaitError[等待或返回错误]
    CheckStatus -->|空闲状态| SetGlobalState[设置全局状态<br/>设置忙碌标志]
    
    SetGlobalState --> CreateToken[创建取消令牌<br/>CancellationToken]
    CreateToken --> SelectMode[选择求解模式<br/>0-4: 五种模式]
    
    SelectMode --> BuildList[构建对象列表<br/>GetSolvingList]
    BuildList --> AnalyzeDep[依赖关系分析<br/>连接关系检查]
    AnalyzeDep --> CalcOrder[计算顺序确定<br/>拓扑排序算法]
    
    CalcOrder --> DetectCycle[检测循环依赖<br/>环路识别算法]
    DetectCycle --> InsertRecycle[插入Recycle对象<br/>循环切断处理]
    InsertRecycle --> CreateTasks[创建计算任务<br/>任务队列构建]
    
    CreateTasks --> ChoosePath{选择执行路径}
    ChoosePath -->|mode=0| SyncPath[同步执行路径]
    ChoosePath -->|mode=1| AsyncPath[异步执行路径]
    ChoosePath -->|mode=2| ParallelPath[并行执行路径]
    ChoosePath -->|mode=3| AzurePath[Azure云计算路径]
    ChoosePath -->|mode=4| TCPPath[TCP分布式路径]
    
    SyncPath --> ExecuteTasks[执行计算任务<br/>ProcessQueue]
    AsyncPath --> ExecuteTasks
    ParallelPath --> ExecuteTasks
    AzurePath --> ExecuteTasks
    TCPPath --> ExecuteTasks
    
    ExecuteTasks --> MonitorProgress[监控计算进度<br/>事件触发机制]
    MonitorProgress --> ProcessResults[处理计算结果<br/>状态更新]
    
    ProcessResults --> ConvergenceCheck{收敛性检查<br/>Recycle收敛判断}
    ConvergenceCheck -->|未收敛| SolveSync[同步调节求解<br/>SolveSimultaneous]
    ConvergenceCheck -->|已收敛| ValidateResults[结果验证<br/>计算完整性]
    
    SolveSync --> ValidateResults
    ValidateResults --> CleanupRes[清理资源<br/>内存释放]
    CleanupRes --> UpdateState[更新流程图状态<br/>Solved标志]
    UpdateState --> TriggerEvent[触发完成事件<br/>FlowsheetFinished]
    
    TriggerEvent --> End([结束])
    WaitError --> End
```

### 4.2 对象计算工作流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - 收到计算请求]) --> ParseArgs[解析计算参数<br/>CalculationArgs]
    ParseArgs --> GetObjRef[获取对象引用<br/>从仿真对象集合]
    
    GetObjRef --> TriggerStart[触发开始事件<br/>UnitOpCalculationStarted]
    TriggerStart --> CheckStatus{检查对象状态<br/>Active/Enabled}
    
    CheckStatus -->|启用| JudgeType{判断对象类型}
    CheckStatus -->|禁用| End([结束])
    
    JudgeType -->|MaterialStream| MatStream[物质流计算<br/>• 物性计算<br/>• 闪蒸计算<br/>• 热力学平衡<br/>• 相平衡]
    JudgeType -->|EnergyStream| EngStream[能量流计算<br/>• 能量平衡]
    JudgeType -->|UnitOperation| UnitOp[单元操作计算<br/>• 调用Solve()<br/>• 附属工具计算<br/>• 规格计算]
    
    MatStream --> PreScript[执行前置脚本<br/>ProcessScripts<br/>ObjectCalcStart]
    EngStream --> PreScript
    UnitOp --> PreScript
    
    PreScript --> UpdateStatus[更新对象状态<br/>• Calculated=True<br/>• LastUpdated<br/>• ErrorMessage]
    
    UpdateStatus --> CheckResults{检查计算结果}
    CheckResults -->|有错误| HandleError[错误处理]
    CheckResults -->|成功| CheckOnlyMe{OnlyMe标志?}
    
    CheckOnlyMe -->|True| PostScript[执行后置脚本<br/>ProcessScripts<br/>ObjectCalcEnd]
    CheckOnlyMe -->|False| HandleDownstream[处理下游对象<br/>• 更新连接流股<br/>• 触发下游计算<br/>• 传播计算状态]
    
    HandleDownstream --> PostScript
    PostScript --> TriggerFinish[触发完成事件<br/>UnitOpCalculationFinished]
    TriggerFinish --> UpdateUI[更新界面显示]
    UpdateUI --> End
    
    HandleError --> End
```

### 4.3 循环收敛工作流程图 (原ASCII版本)

```
                          循环收敛求解工作流程

开始 (检测到循环流程)
  │
  ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  识别循环对象   │ ──►│  收集Recycle变量│ ──►│  设置初始猜值   │
│ (Recycle搜索)   │    │ (切断变量列表)  │    │ (初始化向量x₀)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  设置收敛容差   │ ──►│  初始化矩阵     │ ──►│  开始迭代循环   │
│ (tolerance设置) │    │ (Broyden矩阵)   │    │ (k=0, converged=F)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                                                     │
  ▼                                                     ▼
┌─────────────────┐                        ┌─────────────────┐
│  迭代循环开始   │◄───────────────────────┤  设置Recycle值  │
│ DO WHILE        │                        │ (更新切断变量)  │
│ k < max_iter    │                        └─────────────────┘
└─────────────────┘                                  │
  │                                                  ▼
  ▼                                        ┌─────────────────┐
┌─────────────────┐                        │  执行流程图计算 │
│  计算误差函数   │◄───────────────────────┤ProcessQueue     │
│ error = f(x_k)  │                        │ (完整计算一轮)  │
└─────────────────┘                        └─────────────────┘
  │                                                  │
  ▼                                                  ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  检查收敛条件   │ ──►│  计算新的误差   │ ──►│  更新误差向量   │
│||error|| < tol? │    │error_new=g(x_k) │    │ errors[k]=error │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │ 是                           │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  收敛成功       │    │  Broyden更新    │ ──►│  计算新变量值   │
│  退出循环       │    │J_{k+1}=J_k+ΔyΔx^T│   │x_{k+1}=x_k+Δx_k│
└─────────────────┘    │ /(Δx^T·Δx)      │    └─────────────────┘
  │                    └─────────────────┘              │
  ▼                              │                      ▼
结束                             ▼                    k += 1
                       ┌─────────────────┐              │
                       │  检查迭代次数   │◄─────────────┘
                       │ k < max_iter?   │
                       └─────────────────┘
                                 │ 是
                                 └──────► 继续迭代
                                 │ 否
                                 ▼
                       ┌─────────────────┐
                       │  达到最大迭代   │
                       │  抛出收敛异常   │
                       └─────────────────┘
```

### 4.4 循环收敛工作流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - 检测到循环流程]) --> IdentifyRecycle[识别循环对象<br/>Recycle搜索]
    IdentifyRecycle --> CollectVars[收集Recycle变量<br/>切断变量列表]
    CollectVars --> SetInitial[设置初始猜值<br/>初始化向量x₀]
    
    SetInitial --> SetTolerance[设置收敛容差<br/>tolerance设置]
    SetTolerance --> InitMatrix[初始化矩阵<br/>Broyden矩阵]
    InitMatrix --> StartIteration[开始迭代循环<br/>k=0, converged=False]
    
    StartIteration --> IterationLoop{迭代循环开始<br/>DO WHILE<br/>k < max_iter}
    IterationLoop -->|继续迭代| SetRecycleVals[设置Recycle值<br/>更新切断变量]
    
    SetRecycleVals --> ExecuteCalc[执行流程图计算<br/>ProcessQueue<br/>完整计算一轮]
    ExecuteCalc --> CalcError[计算误差函数<br/>error = f(x_k)]
    
    CalcError --> CheckConvergence{检查收敛条件<br/>||error|| < tol?}
    CheckConvergence -->|已收敛| ConvergedSuccess[收敛成功<br/>退出循环]
    CheckConvergence -->|未收敛| CalcNewError[计算新的误差<br/>error_new = g(x_k)]
    
    CalcNewError --> UpdateErrorVec[更新误差向量<br/>errors[k] = error]
    UpdateErrorVec --> BroydenUpdate[Broyden更新<br/>J_{k+1} = J_k + ΔyΔx^T/(Δx^T·Δx)]
    BroydenUpdate --> CalcNewVars[计算新变量值<br/>x_{k+1} = x_k + Δx_k]
    
    CalcNewVars --> IncrementK[k += 1]
    IncrementK --> CheckMaxIter{检查迭代次数<br/>k < max_iter?}
    CheckMaxIter -->|是| IterationLoop
    CheckMaxIter -->|否| MaxIterReached[达到最大迭代<br/>抛出收敛异常]
    
    ConvergedSuccess --> End([结束])
    MaxIterReached --> ExceptionHandling[异常处理]
    ExceptionHandling --> End
```

### 4.5 并行计算调度流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - mode=2 并行模式]) --> AnalyzeParallel[分析并行可能性<br/>依赖关系检查]
    AnalyzeParallel --> BuildLevels[构建计算层次<br/>filteredlist]
    BuildLevels --> CreateTasks[创建任务组<br/>Task创建]
    
    CreateTasks --> ConfigScheduler{配置TaskScheduler}
    ConfigScheduler -->|Default| DefaultSched[TaskScheduler.Default]
    ConfigScheduler -->|STA| STASched[StaTaskScheduler]
    ConfigScheduler -->|Limited| LimitedSched[LimitedConcurrencyLevelTaskScheduler]
    
    DefaultSched --> SetConcurrency[设置并发级别<br/>MaxDegreeOfParallelism]
    STASched --> SetConcurrency
    LimitedSched --> SetConcurrency
    
    SetConcurrency --> AllocateResources[分配计算资源<br/>CPU/GPU/Memory]
    AllocateResources --> CreateMainTask[创建主任务<br/>Task(Sub()...)]
    CreateMainTask --> StartMainTask[启动主任务<br/>maintask.Start<br/>TaskScheduler]
    
    StartMainTask --> LevelLoop{层次化计算<br/>For Each Level<br/>in filteredlist}
    LevelLoop -->|下一层| ExecuteParallel[并行执行对象<br/>Parallel.ForEach<br/>currentlevel]
    
    ExecuteParallel --> CalcObj[• CalculateObject<br/>• 状态同步<br/>• 异常捕获]
    CalcObj --> CheckLevelComplete{层次完成检查<br/>所有任务完成?}
    CheckLevelComplete -->|否| LevelLoop
    CheckLevelComplete -->|是| CollectResults[收集计算结果<br/>• 成功对象列表<br/>• 异常对象列表<br/>• 性能统计]
    
    CollectResults --> PostProcess[后续处理<br/>• Recycle收敛<br/>• 同步调节<br/>• 状态更新]
    
    subgraph "并行监控"
        MonitorTasks[监控任务状态<br/>• 超时检查<br/>• 取消检查<br/>• 进度更新]
        ProcessTaskResults[处理任务结果<br/>• 成功计算<br/>• 异常处理<br/>• 状态更新]
        SyncWait[同步等待<br/>Wait/WaitAll]
    end
    
    ExecuteParallel -.-> MonitorTasks
    MonitorTasks -.-> ProcessTaskResults
    ProcessTaskResults -.-> SyncWait
    SyncWait -.-> CheckLevelComplete
    
    PostProcess --> CleanupParallel[清理资源<br/>• 释放任务<br/>• 清理内存<br/>• 重置状态]
    CleanupParallel --> End([结束])
```

### 4.6 远程计算工作流程图 (Mermaid)

```mermaid
flowchart TD
    Start([开始 - mode=3/4 远程模式]) --> SelectRemoteType{选择远程类型}
    SelectRemoteType -->|mode=3| AzureClient[Azure云计算]
    SelectRemoteType -->|mode=4| TCPClient[TCP网络计算]
    
    AzureClient --> AzureConnect[建立Azure连接<br/>Service Bus]
    TCPClient --> TCPConnect[建立TCP连接<br/>Socket]
    
    AzureConnect --> VerifyAzure[验证Azure服务器<br/>• 连接检查<br/>• 握手协议]
    TCPConnect --> VerifyTCP[验证TCP服务器<br/>• 连接检查<br/>• 握手协议]
    
    VerifyAzure --> SerializeFlowsheet[序列化流程图<br/>SaveToXML()]
    VerifyTCP --> SerializeFlowsheet
    
    SerializeFlowsheet --> CompressData[数据压缩<br/>GZip压缩]
    CompressData --> CheckSize{检查数据大小<br/>>256KB?}
    CheckSize -->|是| FragmentData[分片传输]
    CheckSize -->|否| DirectSend[直接发送]
    
    FragmentData --> SendData[发送数据<br/>• 消息队列<br/>• 请求ID<br/>• 重试机制]
    DirectSend --> SendData
    
    SendData --> WaitCalculation[等待计算<br/>• 超时监控<br/>• 心跳检查<br/>• 取消支持]
    WaitCalculation --> ReceiveResults[接收结果<br/>• 数据重组<br/>• 解压缩<br/>• 反序列化]
    
    ReceiveResults --> UpdateLocalState[更新本地状态<br/>• 对象状态<br/>• 计算时间<br/>• 错误信息]
    UpdateLocalState --> ValidateResults[验证计算结果<br/>• 完整性检查<br/>• 数据有效性<br/>• 异常处理]
    
    ValidateResults --> CleanupConnection[清理连接<br/>• 关闭连接<br/>• 释放资源<br/>• 状态重置]
    CleanupConnection --> End([结束])
    
    subgraph "错误处理"
        TimeoutError[超时异常] --> RetryConnection[重试连接]
        NetworkError[网络异常] --> RetryConnection
        DataError[数据异常] --> RetryConnection
        RetryConnection -->|重试次数<3| SendData
        RetryConnection -->|重试失败| End
    end
    
    WaitCalculation -.-> TimeoutError
    SendData -.-> NetworkError
    ReceiveResults -.-> DataError
```

## 5. 实际应用案例

### 5.1 复杂循环流程求解

**案例**: 带有多个回流的精馏塔系统

```
进料 ──► 预热器 ──► 精馏塔 ──► 产品
                      │     │
                      ▼     ▼
                   冷凝器  再沸器
                      │     │
                      ▼     ▼
                   回流 ──► 底流 ──► Recycle
                      │            │
                      └────────────┘
```

**求解步骤**:
1. 识别Recycle切断点 (回流组成、流量)
2. 设置初始猜值 (x₀ = [T₀, P₀, F₀])
3. Broyden迭代收敛
4. 验证物料平衡和能量平衡

### 5.2 大型流程图并行计算

**案例**: 石化联合装置流程图

**对象数量**: 200+ 设备, 500+ 物流
**并行策略**: 
- Level 0: 原料进料流 (并行度=8)
- Level 1: 反应器组 (并行度=4)  
- Level 2: 分离单元 (并行度=6)
- Level 3: 产品精制 (并行度=3)

**性能提升**: 相比串行计算加速比约3.2倍

## 6. 性能优化策略

### 6.1 内存管理优化

```vb
' 及时释放大型对象
If Settings.EnableGPUProcessing Then
    Settings.gpu.DisableMultithreading()
    Settings.gpu.FreeAll()
End If

' 清理计算队列
fqueue.CalculationQueue.Clear()
objstack.Clear()
lists.Clear()
```

### 6.2 并发控制优化

```vb
' 动态调整并发级别
Dim nthreads As Integer = Settings.MaxThreadMultiplier * Environment.ProcessorCount

' 选择最优任务调度器
Select Case Settings.TaskScheduler
    Case 0: Settings.AppTaskScheduler = TaskScheduler.Default
    Case 1: Settings.AppTaskScheduler = New StaTaskScheduler(nthreads)
    Case 2: Settings.AppTaskScheduler = New LimitedConcurrencyLevelTaskScheduler(nthreads)
End Select
```

## 6.5 Mermaid图表总览

本文档包含了多个Mermaid格式的可视化图表，提供了DWSIM5 FlowsheetSolver的全面图形化展示：

### 6.5.1 系统架构图表
- **2.1 系统架构总览**: 7层分层架构的整体视图
- **2.2 模块交互关系图**: 时序图展示模块间的交互流程

### 6.5.2 核心算法流程图
- **3.1.1 拓扑排序算法流程图**: Kahn算法的详细执行步骤
- **3.2.1 Broyden算法流程图**: 非线性方程组求解的迭代过程
- **3.3.1 Newton-Raphson算法流程图**: 同步调节器的多变量Newton方法
- **3.4.1 GetSolvingList算法流程图**: 计算顺序确定的逆向追踪算法

### 6.5.3 系统工作流程图
- **4.1 主求解流程图**: FlowsheetSolver的完整求解流程
- **4.2 对象计算工作流程图**: 单个对象的计算处理流程
- **4.4 循环收敛工作流程图**: Recycle循环的Broyden迭代求解
- **4.5 并行计算调度流程图**: 多线程并行计算的任务调度
- **4.6 远程计算工作流程图**: Azure/TCP分布式计算流程

### 6.5.4 图表使用建议

**开发者视角**:
- 使用系统架构图理解模块组织结构
- 参考算法流程图实现具体计算逻辑
- 通过工作流程图掌握系统运行机制

**维护者视角**:
- 利用交互关系图定位问题模块
- 基于算法流程图优化性能瓶颈
- 借助工作流程图进行功能扩展

**用户视角**:
- 通过主求解流程图了解系统能力
- 参考远程计算流程图配置分布式环境
- 使用并行计算流程图优化计算性能

### 6.5.5 图表技术特点

**Mermaid语法优势**:
- 支持版本控制和差异比较
- 可以嵌入到Markdown文档中
- 支持多种图表类型（流程图、时序图、架构图）
- 渲染效果清晰，适合技术文档

**图表设计原则**:
- 层次化展示复杂系统结构
- 突出关键决策点和数据流向
- 平衡细节程度和整体可读性
- 保持风格一致性和视觉连贯性

## 7. 总结

FlowsheetSolver通过其分层的模块化架构、先进的数值算法和灵活的并行处理能力，为DWSIM5提供了强大的流程图求解能力。其核心优势包括：

- **智能调度**: 基于图论的拓扑排序算法确保计算顺序正确性
- **快速收敛**: Broyden和Newton方法的结合提供了优异的收敛性能  
- **并行支持**: 多层次的并行策略充分利用现代多核处理器
- **分布式计算**: Azure云计算和TCP网络支持扩展到集群环境
- **错误处理**: 完善的异常捕获和恢复机制确保系统稳定性

该求解器为复杂化工流程的仿真计算提供了可靠的技术基础，是DWSIM5软件的核心技术优势所在。 