# DWSIM5 FlowsheetSolver 技术文档

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

## 2. 架构设计

### 2.1 核心模块结构图

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

### 2.2 模块交互关系图

```
用户请求 ──► SolveFlowsheet ──► GetSolvingList ──► 依赖分析
    │               │                │              │
    │               ▼                ▼              ▼
    │        ProcessQueue ──► CalculateObject ──► 对象求解
    │               │                │              │
    │               ▼                ▼              ▼
    │        RecycleSolver ──► ConvergenceCheck ──► 收敛判断
    │               │                │              │
    │               ▼                ▼              ▼
    │      TaskScheduler ──► ThreadManagement ──► 并行执行
    │               │                │              │
    │               ▼                ▼              ▼
    └──► EventSystem ──── ErrorHandler ──── 结果返回
```

### 2.3 依赖关系

FlowsheetSolver依赖以下核心组件：

```
DWSIM.FlowsheetSolver
├── DWSIM.Interfaces (接口定义)
├── DWSIM.SharedClasses (共享类库)
├── DWSIM.GlobalSettings (全局设置)
├── DWSIM.ExtensionMethods (扩展方法)
├── DWSIM.MathOps (数学运算)
├── DWSIM.Inspector (调试检查器)
├── DWSIM.XMLSerializer (XML序列化)
├── Microsoft.ServiceBus (Azure服务总线)
└── TcpComm (TCP通信库)
```

## 3. 核心算法与计算公式

### 3.1 计算顺序确定算法

#### 3.1.1 拓扑排序算法

**算法目标**: 确定流程图对象的计算顺序，确保依赖关系正确

**数学表示**:
设流程图为有向图 G(V, E)，其中：
- V = {v₁, v₂, ..., vₙ} 为对象集合
- E ⊆ V × V 为连接关系集合

拓扑排序算法：
```
Kahn's Algorithm:
1. 计算所有节点的入度: indegree[v] = |{u ∈ V : (u,v) ∈ E}|
2. 将所有入度为0的节点加入队列Q
3. While Q非空:
   a. 取出节点u
   b. 将u加入结果序列
   c. For each (u,v) ∈ E:
      - indegree[v] -= 1
      - if indegree[v] = 0: 将v加入队列Q
```

**复杂度分析**:
- 时间复杂度: O(V + E)
- 空间复杂度: O(V)

#### 3.1.2 环路检测算法

**深度优先搜索检测环路**:
```
DFS_Cycle_Detection:
状态定义: WHITE(0), GRAY(1), BLACK(2)

function hasCycle(G):
    for each v ∈ V:
        color[v] = WHITE
    
    for each v ∈ V:
        if color[v] = WHITE:
            if DFS_Visit(v) = TRUE:
                return TRUE
    return FALSE

function DFS_Visit(u):
    color[u] = GRAY
    for each v ∈ Adj[u]:
        if color[v] = GRAY:
            return TRUE  // 发现后向边，存在环路
        if color[v] = WHITE and DFS_Visit(v) = TRUE:
            return TRUE
    color[u] = BLACK
    return FALSE
```

### 3.2 Broyden收敛加速算法

#### 3.2.1 Broyden方法数学原理

**目标**: 求解非线性方程组 F(x) = 0

**Broyden公式**:
```
经典Broyden更新公式:
J_{k+1} = J_k + (Δy_k - J_k·Δx_k)(Δx_k^T) / (Δx_k^T·Δx_k)

其中:
- J_k: 第k次迭代的雅可比矩阵近似
- Δx_k = x_{k+1} - x_k: 变量增量
- Δy_k = F(x_{k+1}) - F(x_k): 函数值增量
- x_{k+1} = x_k - J_k^{-1}·F(x_k): Newton迭代公式
```

**Sherman-Morrison公式应用**:
```
逆矩阵更新公式:
J_{k+1}^{-1} = J_k^{-1} + (Δx_k - J_k^{-1}·Δy_k)(Δx_k^T·J_k^{-1}) / (Δx_k^T·J_k^{-1}·Δy_k)

这样避免了矩阵求逆的昂贵计算
```

#### 3.2.2 收敛性判断

**收敛条件**:
```
1. 残差收敛: ||F(x_k)|| < ε₁
2. 变量收敛: ||x_{k+1} - x_k|| < ε₂
3. 相对变化: ||x_{k+1} - x_k|| / ||x_k|| < ε₃

其中 ε₁, ε₂, ε₃ 为用户设定的容差
```

**收敛速度分析**:
- Broyden方法: 超线性收敛 (1 < q < 2)
- Newton方法: 二次收敛 (q = 2)
- 收敛率: lim(k→∞) ||x_{k+1} - x*|| / ||x_k - x*||^q = L < ∞

### 3.3 Newton-Raphson同步调节算法

#### 3.3.1 多变量Newton方法

**系统方程**:
对于同步调节问题，设有m个调节对象：
```
F(x) = [f₁(x₁, x₂, ..., xₘ)]   [0]
       [f₂(x₁, x₂, ..., xₘ)] = [0]
       [        ⋮        ]   [⋮]
       [fₘ(x₁, x₂, ..., xₘ)]   [0]
```

**雅可比矩阵**:
```
J(x) = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₘ]
       [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₘ]
       [   ⋮        ⋮      ⋱     ⋮   ]
       [∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₘ]
```

**Newton迭代公式**:
```
x_{k+1} = x_k - J(x_k)^{-1} · F(x_k)

或者求解线性系统:
J(x_k) · Δx_k = -F(x_k)
x_{k+1} = x_k + Δx_k
```

#### 3.3.2 数值微分计算梯度

**前向差分**:
```
∂f_i/∂x_j ≈ [f_i(x + h·e_j) - f_i(x)] / h

其中 e_j 是第j个单位向量，h为步长
```

**中心差分**（更高精度）:
```
∂f_i/∂x_j ≈ [f_i(x + h·e_j) - f_i(x - h·e_j)] / (2h)
```

**自适应步长选择**:
```
h_j = ε · max(|x_j|, 1) · sign(x_j)

其中 ε ≈ √(机器精度) ≈ 1.49×10⁻⁸
```

### 3.4 误差函数与容差计算

#### 3.4.1 误差函数定义

**绝对误差平方和**:
```
NSSE = Σᵢ₌₁ᵐ [f_i(x)]²

其中 f_i(x) 为第i个调节对象的误差函数
```

**相对误差**:
```
Relative_Error_i = |calculated_value_i - target_value_i| / |target_value_i|
```

**加权误差**:
```
Weighted_Error = Σᵢ₌₁ᵐ w_i · |error_i|

其中 w_i 为第i个变量的权重因子
```

#### 3.4.2 收敛判断准则

**多重收敛条件**:
```
收敛 ⟺ (∀i: |f_i(x)| < tol_i) ∧ (||Δx|| < ε_x) ∧ (iter < max_iter)

其中:
- tol_i: 第i个变量的容差
- ε_x: 变量变化量容差  
- max_iter: 最大迭代次数
```

## 4. 系统工作流程图

### 4.1 主求解流程图

```
                          FlowsheetSolver 主工作流程

开始
  │
  ▼
┌─────────────────┐
│  检查求解器状态  │ ──► 忙碌状态 ──► 等待或返回错误
│ (CalculatorBusy)│
└─────────────────┘
  │ 空闲状态
  ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  设置全局状态   │ ──►│  创建取消令牌   │ ──►│  选择求解模式   │
│ (设置忙碌标志)  │    │(CancellationToken)│   │ (0-4: 五种模式) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  构建对象列表   │ ──►│  依赖关系分析   │ ──►│  计算顺序确定   │
│ GetSolvingList  │    │(连接关系检查)    │    │ (拓扑排序算法)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  检测循环依赖   │ ──►│  插入Recycle对象│ ──►│  创建计算任务   │
│ (环路识别算法)  │    │ (循环切断处理)  │    │ (任务队列构建)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │
  ▼
┌─────────────────┐
│  选择执行路径   │
│                 │
└─────────────────┘
  │
  ├─── mode=0 ──► 同步执行路径
  ├─── mode=1 ──► 异步执行路径  
  ├─── mode=2 ──► 并行执行路径
  ├─── mode=3 ──► Azure云计算路径
  └─── mode=4 ──► TCP分布式路径
  │
  ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  执行计算任务   │ ──►│  监控计算进度   │ ──►│  处理计算结果   │
│ProcessQueue     │    │ (事件触发机制)  │    │ (状态更新)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  收敛性检查     │ ──►│  同步调节求解   │ ──►│  结果验证       │
│(Recycle收敛判断)│    │SolveSimultaneous│    │ (计算完整性)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  清理资源       │ ──►│  更新流程图状态 │ ──►│  触发完成事件   │
│ (内存释放)      │    │ (Solved标志)    │    │FlowsheetFinished│
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │
  ▼
结束
```

### 4.2 对象计算工作流程

```
                        单对象计算工作流程
                           CalculateObject

开始 (收到计算请求)
  │
  ▼
┌─────────────────┐    ┌─────────────────┐
│  解析计算参数   │ ──►│  获取对象引用   │
│ CalculationArgs │    │ (从仿真对象集合)│
└─────────────────┘    └─────────────────┘
  │                              │
  ▼                              ▼
┌─────────────────┐    ┌─────────────────┐
│  触发开始事件   │ ──►│  检查对象状态   │
│UnitOpCalculation│    │ (Active/Enabled)│
│Started          │    └─────────────────┘
└─────────────────┘              │
  │                              ▼
  ▼                    ┌─────────────────┐
┌─────────────────┐    │  判断对象类型   │
│  执行前置脚本   │◄───┤                 │
│ProcessScripts   │    │  ┌─ MaterialStream
│(ObjectCalcStart)│    │  ├─ EnergyStream  
└─────────────────┘    │  └─ UnitOperation
  │                    └─────────────────┘
  ▼                              │
┌─────────────────┐              ▼
│    分派计算     │    ┌─────────────────┐
│                 │◄───┤  根据类型调用   │
│ MaterialStream  │    │  相应计算方法   │
│ ├─ 物性计算     │    └─────────────────┘
│ ├─ 闪蒸计算     │              │
│ ├─ 热力学平衡   │              ▼
│ └─ 相平衡       │    ┌─────────────────┐
│                 │    │  更新对象状态   │
│ EnergyStream    │    │ ├─ Calculated=T │
│ └─ 能量平衡     │    │ ├─ LastUpdated  │
│                 │    │ └─ ErrorMessage │
│ UnitOperation   │    └─────────────────┘
│ ├─ 调用Solve()  │              │
│ ├─ 附属工具计算 │              ▼
│ └─ 规格计算     │    ┌─────────────────┐
└─────────────────┘    │  处理下游对象   │
  │                    │ (如果OnlyMe=F)  │
  ▼                    │ ├─ 更新连接流股  │
┌─────────────────┐    │ ├─ 触发下游计算  │
│  检查计算结果   │    │ └─ 传播计算状态  │
│ ├─ 错误检查     │    └─────────────────┘
│ ├─ 收敛验证     │              │
│ └─ 数据有效性   │              ▼
└─────────────────┘    ┌─────────────────┐
  │                    │  执行后置脚本   │
  ▼                    │ProcessScripts   │
┌─────────────────┐    │(ObjectCalcEnd)  │
│  触发完成事件   │◄───┤                 │
│UnitOpCalculation│    └─────────────────┘
│Finished         │              │
└─────────────────┘              ▼
  │                            结束
  ▼
更新界面显示
```

### 4.3 循环收敛工作流程

```
                          循环收敛求解工作流程
                             Recycle Solving

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
┌─────────────────┐              ▼                    k += 1
│  标记收敛完成   │    ┌─────────────────┐              │
│ converged=TRUE  │    │  检查迭代次数   │◄─────────────┘
└─────────────────┘    │ k < max_iter?   │
  │                    └─────────────────┘
  ▼                              │ 是
结束                             └──────► 继续迭代
                                 │ 否
                                 ▼
                       ┌─────────────────┐
                       │  达到最大迭代   │
                       │  抛出收敛异常   │
                       └─────────────────┘
                                 │
                                 ▼
                               异常处理
```

### 4.4 并行计算工作流程

```
                          并行计算调度工作流程
                           Parallel Execution

开始 (mode=2 并行模式)
  │
  ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  分析并行可能性 │ ──►│  构建计算层次   │ ──►│  创建任务组     │
│ (依赖关系检查)  │    │ (filteredlist)  │    │ (Task创建)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  配置TaskScheduler │ │  设置并发级别   │ ──►│  分配计算资源   │
│ ├─ Default      │ ──►│MaxDegreeParallel│    │ (CPU/GPU/Memory)│
│ ├─ STA          │    │ism              │    └─────────────────┘
│ └─ Limited      │    └─────────────────┘              │
└─────────────────┘                                     ▼
  │                                            ┌─────────────────┐
  ▼                                            │  启动主任务     │
┌─────────────────┐                            │ maintask.Start  │
│  创建主任务     │◄───────────────────────────┤ (TaskScheduler) │
│ Task(Sub()...)  │                            └─────────────────┘
└─────────────────┘                                      │
  │                                                      ▼
  ▼                                            ┌─────────────────┐
┌─────────────────┐                            │  层次化计算     │
│  监控任务状态   │◄───────────────────────────┤ For Each Level  │
│ ├─ 超时检查     │                            │ in filteredlist │
│ ├─ 取消检查     │                            └─────────────────┘
│ └─ 进度更新     │                                      │
└─────────────────┘                                      ▼
  │                                            ┌─────────────────┐
  ▼                                            │  并行执行对象   │
┌─────────────────┐                            │ Parallel.ForEach│
│  处理任务结果   │◄───────────────────────────┤ (currentlevel)  │
│ ├─ 成功计算     │                            │ ├─ CalculateObj │
│ ├─ 异常处理     │                            │ ├─ 状态同步     │
│ └─ 状态更新     │                            │ └─ 异常捕获     │
└─────────────────┘                            └─────────────────┘
  │                                                      │
  ▼                                                      ▼
┌─────────────────┐                            ┌─────────────────┐
│  同步等待       │                            │  层次完成检查   │
│ Wait/WaitAll    │◄───────────────────────────┤ 所有任务完成?   │
└─────────────────┘                            └─────────────────┘
  │                                                      │ 否
  ▼                                                      └─► 继续下一层
┌─────────────────┐                                      │ 是
│  清理资源       │                                      ▼
│ ├─ 释放任务     │                            ┌─────────────────┐
│ ├─ 清理内存     │                            │  收集计算结果   │
│ └─ 重置状态     │◄───────────────────────────┤ ├─ 成功对象列表  │
└─────────────────┘                            │ ├─ 异常对象列表  │
  │                                            │ └─ 性能统计     │
  ▼                                            └─────────────────┘
结束                                                     │
                                                         ▼
                                               ┌─────────────────┐
                                               │  后续处理       │
                                               │ ├─ Recycle收敛  │
                                               │ ├─ 同步调节     │
                                               │ └─ 状态更新     │
                                               └─────────────────┘
```

### 4.5 远程计算工作流程

```
                          远程计算工作流程 (Azure/TCP)

开始 (mode=3/4 远程模式)
  │
  ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  选择远程类型   │ ──►│  建立连接       │ ──►│  验证服务器     │
│ ├─ Azure (3)    │    │ ├─ Azure: SvcBus│    │ ├─ 连接检查     │
│ └─ TCP (4)      │    │ └─ TCP: Socket  │    │ └─ 握手协议     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  序列化流程图   │ ──►│  数据压缩       │ ──►│  分片传输       │
│ SaveToXML()     │    │ GZip压缩        │    │ (如果>256KB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  发送数据       │ ──►│  等待计算       │ ──►│  接收结果       │
│ ├─ 消息队列     │    │ ├─ 超时监控     │    │ ├─ 数据重组     │
│ ├─ 请求ID       │    │ ├─ 心跳检查     │    │ ├─ 解压缩       │
│ └─ 重试机制     │    │ └─ 取消支持     │    │ └─ 反序列化     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │                              │                      │
  ▼                              ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  更新本地状态   │ ──►│  验证计算结果   │ ──►│  清理连接       │
│ ├─ 对象状态     │    │ ├─ 完整性检查   │    │ ├─ 关闭连接     │
│ ├─ 计算时间     │    │ ├─ 数据有效性   │    │ ├─ 释放资源     │
│ └─ 错误信息     │    │ └─ 异常处理     │    │ └─ 状态重置     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
  │
  ▼
结束
```

## 5. 核心功能模块

### 5.1 对象计算模块

#### 5.1.1 CalculateObject方法

```vb
Public Shared Sub CalculateObject(
    ByVal fobj As Object, 
    ByVal objArgs As CalculationArgs, 
    ByVal sender As Object, 
    Optional ByVal OnlyMe As Boolean = False
)
```

**功能说明**: 计算单个流程图对象的核心方法

**参数**:
- `fobj`: 流程图对象
- `objArgs`: 计算参数，包含对象名称、类型和状态
- `sender`: 调用者标识
- `OnlyMe`: 是否只计算当前对象

**处理流程**:
1. 根据对象类型分派计算任务
2. 处理MaterialStream、EnergyStream和UnitOperation
3. 更新连接的下游对象
4. 触发相关事件和脚本

#### 5.1.2 异步计算支持

```vb
Public Shared Sub CalculateObjectAsync(
    ByVal fobj As Object, 
    ByVal objArgs As CalculationArgs, 
    ct As Threading.CancellationToken
)
```

**特点**:
- 支持取消令牌机制
- 适用于后台线程计算
- 减少UI阻塞

### 5.2 物流计算模块

#### 5.2.1 CalculateMaterialStream方法

专门处理物质流计算的方法，包含：
- 物性计算
- 闪蒸计算
- 热力学平衡
- 组分平衡

**流程**:
```
物流对象初始化 → 调用求解器 → 更新物性 → 触发下游计算 → 标记完成
```

### 5.3 计算队列处理

#### 5.3.1 ProcessCalculationQueue方法

```vb
Public Shared Function ProcessCalculationQueue(
    ByVal fobj As Object, 
    Optional ByVal Isolated As Boolean = False,
    Optional ByVal FlowsheetSolverMode As Boolean = False,
    Optional ByVal mode As Integer = 0,
    Optional orderedlist As Object = Nothing,
    Optional ByVal ct As Threading.CancellationToken = Nothing,
    Optional ByVal Adjusting As Boolean = False
) As List(Of Exception)
```

**计算模式**:
- `mode = 0`: 主线程同步计算
- `mode = 1`: 后台线程异步计算  
- `mode = 2`: 后台并行线程计算

#### 5.3.2 错误处理机制

```vb
Private Shared Sub CheckExceptionForAdditionalInfo(ex As Exception)
    If Not ex.Data.Contains("DetailedDescription") Then
        ex.Data.Add("DetailedDescription", "This error was raised during the calculation of a Unit Operation or Material Stream.")
    End If
    If Not ex.Data.Contains("UserAction") Then
        ex.Data.Add("UserAction", "Check input parameters. If this error keeps occurring, try another Property Package and/or Flash Algorithm.")
    End If
End Sub
```

### 5.4 流程图求解引擎

#### 5.4.1 SolveFlowsheet核心方法

```vb
Public Shared Function SolveFlowsheet(
    ByVal fobj As Object, 
    mode As Integer, 
    Optional ByVal ts As CancellationTokenSource = Nothing,
    Optional frompgrid As Boolean = False, 
    Optional Adjusting As Boolean = False,
    Optional ByVal FinishSuccess As Action = Nothing,
    Optional ByVal FinishWithErrors As Action = Nothing,
    Optional ByVal FinishAny As Action = Nothing,
    Optional ByVal ChangeCalcOrder As Boolean = False
) As List(Of Exception)
```

**求解模式**:
- `mode = 0`: 同步计算（主线程）
- `mode = 1`: 异步计算（后台线程）
- `mode = 2`: 异步并行计算
- `mode = 3`: Azure服务总线计算
- `mode = 4`: 网络分布式计算

#### 5.4.2 计算顺序确定算法

**GetSolvingList方法**实现计算顺序的自动确定：

1. **终点识别**: 找到没有出口连接的物流和设备
2. **逆向追踪**: 从终点开始逆向构建依赖链
3. **层次分组**: 将对象按计算层次分组
4. **环路检测**: 识别循环并插入Recycle对象
5. **顺序优化**: 生成最优的计算顺序

**算法伪代码**:
```
lists = 空字典
lists[0] = 终点对象列表

DO
    listidx += 1
    lists[listidx] = 空列表
    FOR EACH obj IN lists[listidx-1]
        FOR EACH connector IN obj.InputConnectors
            IF connector.IsAttached THEN
                lists[listidx].Add(connector.Source)
            END IF
        END FOR
    END FOR
LOOP UNTIL lists[listidx].Count = 0

返回逆序排列的对象列表
```

### 5.5 循环求解和收敛

#### 5.5.1 Recycle处理机制

FlowsheetSolver使用Broyden方法处理循环流程的收敛：

```vb
' 全局Broyden方法收敛加速
MathEx.Broyden.broydn(totalv - 1, recvars, recerrs, recdvars, recvarsb, recerrsb, rechess, If(icount < 2, 0, 1))
```

**收敛检查流程**:
1. 设置初始值和容差
2. 迭代计算误差函数
3. 应用Broyden加速方法
4. 检查收敛条件
5. 更新变量并重新计算

#### 5.5.2 同步调节求解器

**SolveSimultaneousAdjusts方法**使用Newton方法求解多个调节对象：

```vb
' Newton方法求解非线性方程组
fx = FunctionValueSync(fobj, x)
dfdx = FunctionGradientSync(fobj, x)
success = MathEx.SysLin.rsolve.rmatrixsolve(dfdx, fx, x.Length, dx)
```

**求解过程**:
1. 收集所有标记为同步调节的对象
2. 构建雅可比矩阵
3. 使用Newton-Raphson方法迭代
4. 检查收敛性和最大迭代次数

## 6. 任务调度系统

### 6.1 LimitedConcurrencyLevelTaskScheduler

**设计目标**: 控制并发任务数量，防止系统资源耗尽

**核心特性**:
- 限制最大并发级别
- 基于ThreadPool实现
- 支持任务内联执行
- 线程安全的任务队列管理

**实现机制**:
```vb
Protected Overrides Sub QueueTask(ByVal t As Task)
    SyncLock (_tasks)
        _tasks.AddLast(t)
        If (_delegatesQueuedOrRunning < _maxDegreeOfParallelism) Then
            _delegatesQueuedOrRunning = _delegatesQueuedOrRunning + 1
            NotifyThreadPoolOfPendingWork()
        End If
    End SyncLock
End Sub
```

### 6.2 STATaskScheduler

**用途**: 为需要STA(Single Threaded Apartment)模式的组件提供任务调度

**特点**:
- 单线程单元模式
- 支持COM组件调用
- 避免线程模型冲突

## 7. 远程求解系统

### 7.1 Azure云计算求解器

**AzureSolverClient类**实现基于Microsoft Azure Service Bus的分布式计算：

**连接建立**:
```vb
nm = NamespaceManager.CreateFromConnectionString(connectionString)
qcs = QueueClient.CreateFromConnectionString(connectionString, queueNameS)
qcc = QueueClient.CreateFromConnectionString(connectionString, queueNameC)
```

**数据传输流程**:
1. 序列化流程图为XML
2. 压缩数据减少传输量
3. 分片传输（如果超过256KB限制）
4. 等待服务器端计算结果
5. 接收并反序列化结果

**消息格式**:
```vb
msg.Properties.Add("requestID", requestID)
msg.Properties.Add("type", "data")
msg.Properties.Add("origin", "client")
msg.Properties.Add("multipart", False)
```

### 7.2 TCP网络求解器

**TCPSolverClient类**提供基于TCP/IP的网络分布式求解：

**特点**:
- 直接TCP连接，低延迟
- 适用于局域网环境
- 简化的协议设计

## 8. 事件系统

FlowsheetSolver定义了完整的事件系统用于监控计算过程：

### 8.1 事件定义

```vb
Public Shared Event UnitOpCalculationStarted As CustomEvent
Public Shared Event UnitOpCalculationFinished As CustomEvent
Public Shared Event FlowsheetCalculationStarted As CustomEvent
Public Shared Event FlowsheetCalculationFinished As CustomEvent
Public Shared Event MaterialStreamCalculationStarted As CustomEvent
Public Shared Event MaterialStreamCalculationFinished As CustomEvent
Public Shared Event CalculatingObject As CustomEvent2
```

### 8.2 事件触发时机

- **UnitOpCalculationStarted/Finished**: 单元操作计算开始/结束
- **FlowsheetCalculationStarted/Finished**: 流程图计算开始/结束  
- **MaterialStreamCalculationStarted/Finished**: 物流计算开始/结束
- **CalculatingObject**: 正在计算的对象信息

## 9. 错误处理和异常管理

### 9.1 异常分类

FlowsheetSolver处理多种类型的异常：

- **AggregateException**: 聚合多个子异常
- **OperationCanceledException**: 用户取消操作
- **TimeoutException**: 计算超时
- **数值计算异常**: 收敛失败、矩阵奇异等

### 9.2 异常处理策略

```vb
Try
    ' 计算逻辑
Catch agex As AggregateException
    ' 处理聚合异常
    For Each ex In agex.Flatten().InnerExceptions
        ' 记录每个子异常
        fgui.ShowMessage(ex.Message, IFlowsheet.MessageType.GeneralError)
    Next
Catch ex As OperationCanceledException
    ' 处理取消异常
    age = New AggregateException("计算被取消", ex)
Finally
    ' 清理资源
    GlobalSettings.Settings.CalculatorBusy = False
End Try
```

## 10. 性能优化

### 10.1 并行计算优化

- **并行任务调度**: 使用自定义TaskScheduler控制并发
- **GPU加速**: 集成CUDA支持进行并行数值计算
- **内存管理**: 及时释放计算资源

### 10.2 收敛加速算法

- **Broyden方法**: 用于加速非线性方程组收敛
- **自适应步长**: 根据收敛历史调整迭代步长
- **多层次求解**: 分层处理复杂流程图

## 11. 配置和设置

### 11.1 全局设置项

通过GlobalSettings模块控制求解器行为：

- **SolverTimeoutSeconds**: 求解超时时间
- **MaxThreadMultiplier**: 最大线程倍数
- **EnableParallelProcessing**: 启用并行处理
- **EnableGPUProcessing**: 启用GPU加速
- **TaskScheduler**: 任务调度器选择

### 11.2 流程图特定选项

每个流程图可以设置独立的求解选项：

- **SimultaneousAdjustSolverEnabled**: 启用同步调节求解器
- **CalculationQueue**: 计算队列配置

## 12. 使用示例

### 12.1 基本求解流程

```vb
' 创建流程图对象
Dim flowsheet As IFlowsheet = GetFlowsheet()

' 设置求解模式
Dim mode As Integer = 1 ' 后台异步模式

' 开始求解
Dim exceptions As List(Of Exception) = FlowsheetSolver.SolveFlowsheet(
    flowsheet, 
    mode,
    Nothing, ' CancellationTokenSource
    False,   ' frompgrid
    False,   ' Adjusting
    Sub() Console.WriteLine("求解成功"),      ' FinishSuccess
    Sub() Console.WriteLine("求解出错"),      ' FinishWithErrors  
    Sub() Console.WriteLine("求解完成")       ' FinishAny
)
```

### 12.2 单对象计算

```vb
' 创建计算参数
Dim args As New CalculationArgs With {
    .Name = "MIXER-001",
    .ObjectType = ObjectType.Mixer,
    .Sender = "FlowsheetSolver",
    .Calculated = False
}

' 计算单个对象
FlowsheetSolver.CalculateObject(flowsheet, args, Nothing)
```

### 12.3 异步计算模式

```vb
' 创建取消令牌
Dim cts As New CancellationTokenSource()

' 异步计算对象
Task.Run(Sub()
    Try
        FlowsheetSolver.CalculateObjectAsync(flowsheet, args, cts.Token)
    Catch ex As OperationCanceledException
        Console.WriteLine("计算已取消")
    End Try
End Sub)

' 可以随时取消
cts.Cancel()
```

## 13. 调试和诊断

### 13.1 Inspector集成

FlowsheetSolver集成了Inspector模块用于调试：

```vb
Dim IObj As Inspector.InspectorItem = Inspector.Host.GetNewInspectorItem()
IObj?.Paragraphs.Add("开始求解流程图...")
IObj?.Paragraphs.Add("计算对象顺序: " & String.Join(", ", objstack))
```

### 13.2 日志记录

通过IFlowsheet.ShowMessage方法记录计算过程：

```vb
fgui.ShowMessage("开始求解流程图", IFlowsheet.MessageType.Information)
fgui.ShowMessage("计算耗时: " & (Date.Now - d1).ToString("g"), IFlowsheet.MessageType.Information)
```

## 14. 扩展和定制

### 14.1 自定义任务调度器

可以继承TaskScheduler类实现自定义调度策略：

```vb
Public Class CustomTaskScheduler
    Inherits TaskScheduler
    
    Protected Overrides Sub QueueTask(task As Task)
        ' 自定义任务队列逻辑
    End Sub
    
    Protected Overrides Function TryExecuteTaskInline(task As Task, taskWasPreviouslyQueued As Boolean) As Boolean
        ' 自定义内联执行逻辑
    End Function
End Class
```

### 14.2 远程求解器扩展

可以实现新的远程求解器支持其他分布式计算平台：

```vb
Public Class CustomRemoteSolver
    Public Sub SolveFlowsheet(fobj As Object)
        ' 实现自定义远程求解逻辑
    End Sub
End Class
```

## 15. 最佳实践

### 15.1 性能优化建议

1. **选择合适的求解模式**: 根据流程图复杂度選择同步或异步模式
2. **控制并发级别**: 避免创建过多线程导致资源竞争
3. **合理设置超时**: 根据流程图规模设置适当的超时时间
4. **监控内存使用**: 大型流程图可能消耗大量内存

### 15.2 错误处理建议

1. **详细的异常信息**: 记录足够的上下文信息便于调试
2. **优雅的降级策略**: 部分计算失败时的处理策略
3. **用户友好的错误提示**: 将技术错误转换为用户可理解的信息

### 15.3 可维护性建议

1. **模块化设计**: 将不同功能分离到独立的类中
2. **接口抽象**: 使用接口定义核心契约
3. **单元测试**: 为关键算法编写单元测试
4. **文档完善**: 保持代码注释和文档的同步

## 16. 总结

FlowsheetSolver是DWSIM5的核心计算引擎，它通过智能的计算调度、强大的收敛算法和灵活的并行处理能力，确保了复杂流程图的可靠求解。其模块化的设计和丰富的扩展接口，为用户和开发者提供了强大的定制能力。

该组件的主要技术优势包括：

- **智能调度**: 自动分析对象依赖关系并确定最优计算顺序
- **收敛加速**: 集成Broyden等高效数值方法
- **并行支持**: 多线程和GPU并行计算能力  
- **分布式计算**: 支持云计算和网络分布式求解
- **错误处理**: 完善的异常捕获和错误恢复机制
- **可扩展性**: 开放的架构支持功能扩展和定制

FlowsheetSolver为DWSIM5提供了强大而灵活的计算基础，是整个软件系统的技术核心。 