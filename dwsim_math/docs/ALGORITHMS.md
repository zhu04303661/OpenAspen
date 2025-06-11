# DWSIM数学计算库算法文档

## 1. 概述

DWSIM数学计算库是一个综合性的数值计算和优化库，包含以下主要功能模块：

- **核心数学模块** (`core`): 基础数学运算和矩阵操作
- **数值计算模块** (`numerics`): 复数运算、线性代数、常微分方程
- **求解器模块** (`solvers`): 非线性方程求解、线性系统求解
- **优化模块** (`optimization`): 各种优化算法
- **随机数生成模块** (`random`): 多种伪随机数生成器
- **群体智能优化模块** (`swarm`): 粒子群优化、差分进化等
- **特殊函数模块** (`special`): Gamma函数、Bessel函数等

## 2. 算法总览

### 2.1 算法分类表

| 算法类别 | 算法名称 | 英文全称 | 主要应用领域 |
|---------|----------|----------|--------------|
| **核心数学算法** |
| 矩阵LU分解 | LU Decomposition | Lower-Upper Decomposition | 线性代数计算、线性方程组求解、矩阵求逆的基础算法，广泛应用于科学计算和工程分析 |
| 矩阵求逆 | Matrix Inversion | Matrix Inversion | 控制系统分析、统计回归分析、图像处理中的滤波算法 |
| 行列式计算 | Determinant Calculation | Determinant Calculation | 线性系统可解性判断、矩阵特征值计算、几何变换分析 |
| 重心插值 | Barycentric Interpolation | Barycentric Interpolation | 数据拟合、信号处理、数值分析中的函数逼近 |
| **常微分方程求解器** |
| Euler方法 | Euler Method | Euler's Method | 简单ODE数值求解、控制系统仿真、物理现象建模的入门方法 |
| Heun方法 | Heun Method | Heun's Method | 提高精度的ODE求解、工程动力学分析、生物数学建模 |
| 四阶Runge-Kutta | RK4 | Fourth-Order Runge-Kutta | 高精度ODE求解的标准方法，广泛应用于航空航天、机械工程、生物医学仿真 |
| Dormand-Prince | RK45 | Dormand-Prince Method | 自适应步长ODE求解，适用于刚性和非刚性问题的高效数值积分 |
| Adams-Bashforth | AB | Adams-Bashforth Methods | 多步显式ODE求解、长时间积分、天体力学计算 |
| Adams-Moulton | AM | Adams-Moulton Methods | 多步隐式ODE求解、刚性问题处理、化学反应动力学 |
| BDF方法 | BDF | Backward Differentiation Formula | 刚性微分方程求解、化学工程过程建模、电路仿真 |
| **非线性方程求解** |
| Brent算法 | Brent Method | Brent's Root-Finding Method | 单变量非线性方程求根、工程设计优化、物理参数估计的鲁棒方法 |
| Broyden方法 | Broyden Method | Broyden's Method | 多变量非线性方程组求解、化工过程设计、结构力学分析 |
| **数值优化算法** |
| BFGS | BFGS | Broyden-Fletcher-Goldfarb-Shanno | 无约束优化的经典拟牛顿法，广泛应用于机器学习参数估计、工程设计优化 |
| L-BFGS | L-BFGS | Limited-memory BFGS | 大规模无约束优化问题的高效算法，特别适用于机器学习、深度学习、大数据分析 |
| Levenberg-Marquardt | LM | Levenberg-Marquardt Algorithm | 非线性最小二乘问题的专用算法，广泛应用于曲线拟合、参数估计、神经网络训练 |
| 共轭梯度法 | CG | Conjugate Gradient Method | 大规模线性和非线性优化、稀疏矩阵求解、图像重建 |
| 信赖域算法 | Trust Region | Trust Region Methods | 鲁棒的非线性优化、工程结构优化、金融风险建模 |
| **群体智能优化** |
| 粒子群优化 | PSO | Particle Swarm Optimization | 多目标优化、神经网络训练、工程设计优化、调度问题求解 |
| 差分进化 | DE | Differential Evolution | 全局优化、函数优化、工程参数调优、机器学习超参数优化 |
| 遗传算法 | GA | Genetic Algorithm | 组合优化、进化计算、机器学习特征选择、工程设计优化 |
| 蚁群优化 | ACO | Ant Colony Optimization | 路径规划、旅行商问题、网络路由优化、供应链管理 |
| 人工蜂群算法 | ABC | Artificial Bee Colony | 函数优化、特征选择、图像处理、无线传感器网络优化 |
| 萤火虫算法 | FA | Firefly Algorithm | 多模态优化、工程设计、图像处理、无线传感器网络部署 |
| 灰狼优化 | GWO | Grey Wolf Optimizer | 工程优化、机器学习、电力系统优化、图像分割 |
| 鲸鱼优化算法 | WOA | Whale Optimization Algorithm | 结构工程优化、机器学习、电力系统调度、图像处理 |
| **随机数生成器** |
| Mersenne Twister | MT | Mersenne Twister | 高质量伪随机数生成、蒙特卡罗仿真、统计采样、密码学应用 |
| XorShift | XorShift | XorShift Generator | 快速随机数生成、游戏开发、简单仿真、并行计算 |
| KISS | KISS | Keep It Simple Stupid | 组合式随机数生成、统计测试、科学计算中的随机采样 |
| LFSR | LFSR | Linear Feedback Shift Register | 数字信号处理、密码学、测试序列生成、通信系统 |
| Well512 | Well | Well Equidistributed Long-period Linear | 高维均匀分布、蒙特卡罗方法、数值积分、统计仿真 |
| PCG | PCG | Permuted Congruential Generator | 现代高效随机数生成、游戏开发、科学计算、机器学习 |
| **特殊函数** |
| Gamma函数 | Gamma | Gamma Function | 概率统计、特殊函数计算、贝叶斯分析、物理学应用 |
| Bessel函数 | Bessel | Bessel Functions | 波动方程求解、信号处理、物理学、工程振动分析 |
| 椭圆积分 | Elliptic | Elliptic Integrals | 椭圆轨道计算、弹性理论、电磁学、几何学应用 |
| 误差函数 | Erf | Error Function | 概率统计、正态分布计算、扩散过程、质量控制 |
| Fresnel积分 | Fresnel | Fresnel Integrals | 光学衍射、波动理论、信号处理、工程光学设计 |
| **数值积分算法** |
| 梯形公式 | Trapezoidal | Trapezoidal Rule | 基础数值积分、工程计算、物理量计算的简单有效方法 |
| Simpson公式 | Simpson | Simpson's Rule | 高精度数值积分、科学计算、工程分析的标准方法 |
| Gauss-Legendre | Gauss-Legendre | Gauss-Legendre Quadrature | 高精度积分、有限元分析、科学计算中的精确数值积分 |
| Gauss-Hermite | Gauss-Hermite | Gauss-Hermite Quadrature | 无穷区间积分、概率积分、物理学中的量子力学计算 |
| 蒙特卡罗积分 | Monte Carlo | Monte Carlo Integration | 高维积分、复杂几何区域积分、金融风险评估、物理仿真 |
| 自适应积分 | Adaptive | Adaptive Integration | 复杂函数积分、工程计算、精度要求高的科学计算 |

### 2.2 算法应用领域分布

| 应用领域 | 主要算法 | 典型应用场景 |
|----------|----------|--------------|
| **机器学习与数据科学** | L-BFGS, BFGS, PSO, GA, LM | 模型训练、参数优化、特征选择、超参数调优 |
| **工程仿真与CAD** | RK4, BDF, Brent, 矩阵操作 | 结构分析、流体仿真、热传导、动力学分析 |
| **金融计算** | 蒙特卡罗, 随机数生成器, 数值积分 | 风险评估、期权定价、投资组合优化 |
| **图像与信号处理** | Bessel函数, Fresnel积分, 插值算法 | 滤波、降噪、特征提取、图像重建 |
| **物理学与化学** | ODE求解器, 特殊函数, 数值积分 | 量子力学、分子动力学、反应动力学 |
| **运筹学与调度** | 群体智能算法, 遗传算法, 蚁群优化 | 路径规划、资源分配、生产调度 |
| **统计与概率** | 随机数生成器, 误差函数, Gamma函数 | 统计推断、假设检验、概率建模 |
| **控制系统** | ODE求解器, 矩阵算法, 优化算法 | 控制器设计、系统辨识、鲁棒控制 |

## 3. 核心数学模块 (Core)

### 3.1 通用数学函数 (`general.py`)

#### 基本统计函数

**最大值计算**:
$$\max(x_1, x_2, \ldots, x_n) = \max_{i} x_i$$

**最小值计算**:
$$\min(x_1, x_2, \ldots, x_n) = \min_{i} x_i$$

**加权平均**:
$$\bar{x}_w = \frac{\sum_{i=1}^n w_i x_i}{\sum_{i=1}^n w_i}$$

其中 $w_i$ 是权重，$x_i$ 是对应的数值。

**平方和**:
$$\sum_{i=1}^n x_i^2$$

#### 使用示例
```python
from dwsim_math.core.general import MathCommon

# 计算加权平均
weights = [0.3, 0.3, 0.4]
values = [10, 20, 30]
weighted_avg = MathCommon.weighted_average(weights, values)
```

### 3.2 矩阵操作 (`matrix_ops.py`)

#### 3.2.1 LU分解

**算法原理**:
将矩阵 $A$ 分解为下三角矩阵 $L$ 和上三角矩阵 $U$ 的乘积：
$$A = LU$$

或带置换的形式：
$$PA = LU$$

其中 $P$ 是置换矩阵。

**算法步骤**:
1. 初始化置换数组
2. 对于每一列 $k$，选择主元
3. 执行行交换（如果需要）
4. 计算下三角部分的乘数：$L_{ik} = \frac{A_{ik}}{A_{kk}}$
5. 更新剩余子矩阵：$A_{ij} = A_{ij} - L_{ik} \cdot A_{kj}$

#### 3.2.2 行列式计算

对于LU分解后的矩阵，行列式计算为：
$$\det(A) = \det(P) \cdot \det(L) \cdot \det(U) = (-1)^{\text{置换次数}} \cdot 1 \cdot \prod_{i=1}^n U_{ii}$$

#### 3.2.3 矩阵求逆

基于LU分解的矩阵求逆算法：
1. 计算 $A = LU$
2. 求解 $UX = L^{-1}$
3. 应用列置换得到 $A^{-1}$

#### 使用示例
```python
from dwsim_math.core.matrix_ops import MatrixOperations

# 计算行列式
A = [[1, 2], [3, 4]]
det_A = MatrixOperations.determinant(A)

# 计算逆矩阵
inv_A, success = MatrixOperations.inverse(A)
```

### 3.3 插值算法 (`interpolation.py`)

#### 3.3.1 重心插值公式

**数学公式**:
$$f(t) = \frac{\sum_{i=0}^{n-1} \frac{w_i f_i}{t - x_i}}{\sum_{i=0}^{n-1} \frac{w_i}{t - x_i}}$$

其中：
- $x_i$ 是插值节点
- $f_i = f(x_i)$ 是函数值
- $w_i$ 是重心权重

#### 3.3.2 Floater-Hormann有理插值

权重计算公式：
$$w_i = (-1)^i \binom{d}{i \bmod (d+1)}$$

其中 $d$ 是插值阶数。

#### 使用示例
```python
from dwsim_math.core.interpolation import Interpolation

x = [0, 1, 2, 3, 4]
y = [1, 2, 5, 10, 17]
result = Interpolation.interpolate(x, y, 2.5)
```

## 4. 数值计算模块 (Numerics)

### 4.1 复数运算 (`complex_number.py`)

#### 4.1.1 复数基本运算

**复数表示**:
$$z = a + bi$$

其中 $a$ 是实部，$b$ 是虚部，$i$ 是虚数单位。

**模长**:
$$|z| = \sqrt{a^2 + b^2}$$

**幅角**:
$$\arg(z) = \arctan2(b, a)$$

**复数乘法**:
$$(a + bi)(c + di) = (ac - bd) + (ad + bc)i$$

**复数除法**:
$$\frac{a + bi}{c + di} = \frac{(ac + bd) + (bc - ad)i}{c^2 + d^2}$$

#### 4.1.2 复数函数

**指数函数**:
$$e^{a + bi} = e^a(\cos b + i\sin b)$$

**对数函数**:
$$\ln(z) = \ln|z| + i\arg(z)$$

**三角函数**:
$$\sin(z) = \sin(a)\cosh(b) + i\cos(a)\sinh(b)$$
$$\cos(z) = \cos(a)\cosh(b) - i\sin(a)\sinh(b)$$

#### 使用示例
```python
from dwsim_math.numerics.complex_number import Complex

z1 = Complex(3, 4)  # 3 + 4i
z2 = Complex(1, 2)  # 1 + 2i
result = z1 * z2    # 复数乘法
print(f"结果: {result}")  # -5 + 10i
```

### 4.2 线性代数模块 (`linear_algebra/`)

#### 4.2.1 矩阵类 (`matrix.py`)

**矩阵加法**:
$$C = A + B \Rightarrow C_{ij} = A_{ij} + B_{ij}$$

**矩阵乘法**:
$$C = AB \Rightarrow C_{ij} = \sum_{k=1}^n A_{ik}B_{kj}$$

**矩阵转置**:
$$A^T_{ij} = A_{ji}$$

#### 4.2.2 特征值分解

对于矩阵 $A$，特征值 $\lambda$ 和特征向量 $v$ 满足：
$$Av = \lambda v$$

特征多项式：
$$\det(A - \lambda I) = 0$$

#### 4.2.3 奇异值分解 (SVD)

将矩阵 $A$ 分解为：
$$A = U\Sigma V^T$$

其中：
- $U$ 是 $m \times m$ 正交矩阵
- $\Sigma$ 是 $m \times n$ 对角矩阵
- $V$ 是 $n \times n$ 正交矩阵

### 4.3 常微分方程求解器 (`ode/`)

#### 4.3.1 龙格-库塔方法族

**一阶Euler方法**:
$$y_{n+1} = y_n + h f(t_n, y_n)$$

**二阶Heun方法**:
$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + h, y_n + h k_1)$$
$$y_{n+1} = y_n + \frac{h}{2}(k_1 + k_2)$$

**二阶中点方法**:
$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1)$$
$$y_{n+1} = y_n + h k_2$$

**四阶经典龙格-库塔公式**:
$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

其中：
$$k_1 = f(t_n, y_n)$$
$$k_2 = f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1\right)$$
$$k_3 = f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2\right)$$
$$k_4 = f(t_n + h, y_n + h k_3)$$

**Dormand-Prince方法 (RK45)**:
使用6阶段的自适应步长龙格-库塔方法：
$$y_{n+1} = y_n + h \sum_{i=1}^{6} b_i k_i$$

其中系数满足：
$$\sum_{i=1}^{6} b_i = 1, \quad \sum_{i=1}^{6} \hat{b}_i = 1$$

**误差估计**:
$$\text{err} = h \left| \sum_{i=1}^{6} (b_i - \hat{b}_i) k_i \right|$$

#### 4.3.2 Adams族方法

**Adams-Bashforth方法（显式）**:

**2步AB方法**:
$$y_{n+1} = y_n + \frac{h}{2}(3f_n - f_{n-1})$$

**3步AB方法**:
$$y_{n+1} = y_n + \frac{h}{12}(23f_n - 16f_{n-1} + 5f_{n-2})$$

**4步AB方法**:
$$y_{n+1} = y_n + \frac{h}{24}(55f_n - 59f_{n-1} + 37f_{n-2} - 9f_{n-3})$$

**Adams-Moulton方法（隐式）**:

**1步AM方法**:
$$y_{n+1} = y_n + \frac{h}{2}(f_{n+1} + f_n)$$

**2步AM方法**:
$$y_{n+1} = y_n + \frac{h}{12}(5f_{n+1} + 8f_n - f_{n-1})$$

**3步AM方法**:
$$y_{n+1} = y_n + \frac{h}{24}(9f_{n+1} + 19f_n - 5f_{n-1} + f_{n-2})$$

#### 4.3.3 预估-校正方法

**PECE格式**:
1. **预估**: 使用Adams-Bashforth公式计算 $y_{n+1}^{(0)}$
2. **求值**: 计算 $f_{n+1}^{(0)} = f(t_{n+1}, y_{n+1}^{(0)})$
3. **校正**: 使用Adams-Moulton公式计算 $y_{n+1}^{(1)}$
4. **再求值**: 计算 $f_{n+1}^{(1)} = f(t_{n+1}, y_{n+1}^{(1)})$

#### 4.3.4 刚性方程求解器

**隐式Euler方法**:
$$y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})$$

**梯形方法**:
$$y_{n+1} = y_n + \frac{h}{2}[f(t_n, y_n) + f(t_{n+1}, y_{n+1})]$$

**BDF方法（向后差分公式）**:

**1阶BDF**:
$$y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})$$

**2阶BDF**:
$$\frac{3}{2} y_{n+1} - 2 y_n + \frac{1}{2} y_{n-1} = h f(t_{n+1}, y_{n+1})$$

**k阶BDF通式**:
$$\sum_{j=0}^{k} \alpha_j y_{n+1-j} = h \beta f(t_{n+1}, y_{n+1})$$

## 5. 求解器模块 (Solvers)

### 5.1 Brent求根方法 (`brent.py`)

#### 算法原理

Brent方法结合了二分法、割线法和反二次插值法的优点。

**二分法更新**:
$$x_{k+1} = \frac{a_k + b_k}{2}$$

**割线法更新**:
$$x_{k+1} = x_k - f(x_k)\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}$$

**反二次插值**:
当有三个点 $(a,f_a)$, $(b,f_b)$, $(c,f_c)$ 时，通过这三点的抛物线求与x轴的交点。

#### 收敛性

Brent方法保证：
- 如果初始区间包含根，则一定收敛
- 收敛速度介于线性和超线性之间
- 最坏情况下退化为二分法

#### 使用示例
```python
from dwsim_math.solvers.brent import BrentSolver

def f(x, args):
    return x**3 - 2*x - 5

solver = BrentSolver()
root = solver.solve(f, 1.0, 3.0)
```

### 5.2 Broyden方法 (`broyden.py`)

#### 算法原理

用于求解非线性方程组 $F(x) = 0$。

**Broyden更新公式**:
$$B_{k+1} = B_k + \frac{(y_k - B_k s_k)s_k^T}{s_k^T s_k}$$

其中：
- $s_k = x_{k+1} - x_k$
- $y_k = F(x_{k+1}) - F(x_k)$
- $B_k$ 是Jacobian矩阵的近似

#### 迭代公式
$$x_{k+1} = x_k - B_k^{-1} F(x_k)$$

## 6. 优化模块 (Optimization)

### 6.1 BFGS算法 (`bfgs.py`)

#### 算法原理

BFGS (Broyden-Fletcher-Goldfarb-Shanno) 是一种拟牛顿优化算法，用于求解无约束优化问题。

**目标问题**:
$$\min_{x \in \mathbb{R}^n} f(x)$$

**牛顿法的基本思想**:
$$x_{k+1} = x_k - H_k^{-1} \nabla f(x_k)$$

其中 $H_k$ 是Hessian矩阵 $\nabla^2 f(x_k)$。

#### BFGS更新公式

**Hessian逆矩阵的更新**:
$$H_{k+1} = H_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{H_k s_k s_k^T H_k}{s_k^T H_k s_k}$$

其中：
- $s_k = x_{k+1} - x_k$ (位置差)
- $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ (梯度差)

**拟牛顿条件**:
$$H_{k+1} y_k = s_k$$

#### Sherman-Morrison-Woodbury公式

BFGS更新可以写成：
$$H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}$$

#### 收敛性质

- **超线性收敛**: 当 $H_0$ 是正定的且 $y_k^T s_k > 0$ 时
- **全局收敛**: 结合线搜索策略
- **内存需求**: $O(n^2)$

#### 使用示例
```python
from dwsim_math.optimization.bfgs import BFGS

def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

def gradient(x):
    return [2*(x[0] - 1), 2*(x[1] - 2.5)]

optimizer = BFGS()
result = optimizer.minimize(objective, gradient, [0, 0])
```

### 6.2 L-BFGS算法 (`lbfgs.py`)

#### 算法原理

Limited-memory BFGS是BFGS方法的内存优化版本，避免显式存储和操作 $n \times n$ 的Hessian逆矩阵。

#### 两环递归算法

**第一环（向后递归）**:
```
for i = k-1, k-2, ..., k-m:
    α_i = ρ_i s_i^T q
    q = q - α_i y_i
```

**初始化**:
$$r = H_k^{(0)} q$$

其中 $H_k^{(0)} = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}} I$

**第二环（向前递归）**:
```
for i = k-m, k-m+1, ..., k-1:
    β = ρ_i y_i^T r
    r = r + s_i (α_i - β)
```

其中 $\rho_i = \frac{1}{y_i^T s_i}$

#### 存储需求
- BFGS: $O(n^2)$
- L-BFGS: $O(mn)$，其中 $m$ 通常取 3-20

#### 使用示例
```python
from dwsim_math.optimization.lbfgs import LBFGS

def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

def gradient(x):
    return [2*(x[0] - 1), 2*(x[1] - 2.5)]

optimizer = LBFGS(memory_size=10)
result = optimizer.minimize(objective, gradient, [0, 0])
```

### 6.3 Levenberg-Marquardt算法 (`lm.py`)

#### 算法原理

用于求解非线性最小二乘问题：
$$\min_{x \in \mathbb{R}^n} \frac{1}{2}\|f(x)\|^2 = \frac{1}{2}\sum_{i=1}^m f_i(x)^2$$

其中 $f: \mathbb{R}^n \to \mathbb{R}^m$ 是残差向量函数。

#### LM更新公式

**线性方程组**:
$$(J_k^T J_k + \lambda_k I)\delta_k = -J_k^T f(x_k)$$

**位置更新**:
$$x_{k+1} = x_k + \delta_k$$

其中：
- $J_k = \nabla f(x_k)$ 是Jacobian矩阵，$J_{ij} = \frac{\partial f_i}{\partial x_j}(x_k)$
- $\lambda_k > 0$ 是阻尼参数（Levenberg参数）
- $\delta_k$ 是搜索方向

#### 算法解释

**当 $\lambda_k \to 0$**: 算法接近高斯-牛顿法
$$\delta_k \approx -(J_k^T J_k)^{-1} J_k^T f(x_k)$$

**当 $\lambda_k \to \infty$**: 算法接近梯度下降法
$$\delta_k \approx -\frac{1}{\lambda_k} J_k^T f(x_k)$$

#### 阻尼参数自适应调整

**增益比**:
$$\rho_k = \frac{\|f(x_k)\|^2 - \|f(x_k + \delta_k)\|^2}{L_k(0) - L_k(\delta_k)}$$

其中 $L_k(\delta) = \frac{1}{2}\|f(x_k) + J_k\delta\|^2 + \frac{\lambda_k}{2}\|\delta\|^2$

**参数更新策略**:
- 如果 $\rho_k > 0.75$：$\lambda_{k+1} = \lambda_k / 3$
- 如果 $\rho_k < 0.25$：$\lambda_{k+1} = 3\lambda_k$
- 否则：$\lambda_{k+1} = \lambda_k$

#### 使用示例
```python
from dwsim_math.optimization.lm import LevenbergMarquardt
import numpy as np

def residual_function(x):
    # 拟合指数函数 y = a * exp(b * t)
    t = np.array([0, 1, 2, 3, 4])
    y_obs = np.array([1, 2.7, 7.4, 20.1, 54.6])
    y_pred = x[0] * np.exp(x[1] * t)
    return y_pred - y_obs

def jacobian_function(x):
    t = np.array([0, 1, 2, 3, 4])
    exp_bt = np.exp(x[1] * t)
    J = np.zeros((5, 2))
    J[:, 0] = exp_bt  # ∂f/∂a
    J[:, 1] = x[0] * t * exp_bt  # ∂f/∂b
    return J

optimizer = LevenbergMarquardt()
result = optimizer.minimize(residual_function, jacobian_function, [1.0, 1.0])
```

### 6.4 非线性共轭梯度法 (`conjugate_gradient.py`)

#### 算法原理

共轭梯度法是一种用于求解大规模优化问题的迭代方法。

**基本迭代公式**:
$$x_{k+1} = x_k + \alpha_k d_k$$

其中 $d_k$ 是搜索方向，$\alpha_k$ 是步长。

#### 搜索方向更新

**初始方向**:
$$d_0 = -\nabla f(x_0)$$

**共轭方向**:
$$d_{k+1} = -\nabla f(x_{k+1}) + \beta_k d_k$$

#### $\beta_k$ 的不同公式

**Fletcher-Reeves公式**:
$$\beta_k^{FR} = \frac{\|\nabla f(x_{k+1})\|^2}{\|\nabla f(x_k)\|^2}$$

**Polak-Ribière公式**:
$$\beta_k^{PR} = \frac{\nabla f(x_{k+1})^T (\nabla f(x_{k+1}) - \nabla f(x_k))}{\|\nabla f(x_k)\|^2}$$

**Hestenes-Stiefel公式**:
$$\beta_k^{HS} = \frac{\nabla f(x_{k+1})^T (\nabla f(x_{k+1}) - \nabla f(x_k))}{d_k^T (\nabla f(x_{k+1}) - \nabla f(x_k))}$$

**Dai-Yuan公式**:
$$\beta_k^{DY} = \frac{\|\nabla f(x_{k+1})\|^2}{d_k^T (\nabla f(x_{k+1}) - \nabla f(x_k))}$$

#### 重启策略

当 $\nabla f(x_{k+1})^T \nabla f(x_k) \geq 0.1 \|\nabla f(x_k)\|^2$ 时执行重启：
$$d_{k+1} = -\nabla f(x_{k+1})$$

### 6.5 信赖域算法 (`trust_region.py`)

#### 算法原理

信赖域方法在每次迭代中求解子问题：
$$\min_{d} m_k(d) = f(x_k) + \nabla f(x_k)^T d + \frac{1}{2} d^T B_k d$$
$$\text{s.t. } \|d\| \leq \Delta_k$$

其中：
- $B_k$ 是Hessian矩阵的近似
- $\Delta_k > 0$ 是信赖域半径

#### 信赖域半径更新

**增益比**:
$$\rho_k = \frac{f(x_k) - f(x_k + d_k)}{m_k(0) - m_k(d_k)}$$

**半径更新规则**:
- 如果 $\rho_k < 0.25$：$\Delta_{k+1} = 0.25\Delta_k$
- 如果 $\rho_k > 0.75$ 且 $\|d_k\| = \Delta_k$：$\Delta_{k+1} = 2\Delta_k$
- 否则：$\Delta_{k+1} = \Delta_k$

#### 子问题求解

**Cauchy点**:
$$x_k^C = x_k - \tau_k \frac{\Delta_k}{\|\nabla f(x_k)\|} \nabla f(x_k)$$

其中：
$$\tau_k = \begin{cases}
1 & \text{if } \nabla f(x_k)^T B_k \nabla f(x_k) \leq 0 \\
\min\left(1, \frac{\|\nabla f(x_k)\|^3}{\Delta_k \nabla f(x_k)^T B_k \nabla f(x_k)}\right) & \text{otherwise}
\end{cases}$$

## 7. 随机数生成模块 (Random)

### 7.1 Mersenne Twister (`mersenne_twister.py`)

#### 算法原理

基于线性反馈移位寄存器的伪随机数生成器。

**周期长度**: $2^{19937} - 1$

**状态更新**:
$$x_{k+n} = x_{k+m} \oplus ((x_k \& \text{UPPER\_MASK}) | (x_{k+1} \& \text{LOWER\_MASK})) A$$

其中 $A$ 是系数矩阵。

**输出变换**:
```
y = x[i]
y ^= (y >> 11)
y ^= (y << 7) & 0x9d2c5680
y ^= (y << 15) & 0xefc60000
y ^= (y >> 18)
```

#### 特点
- 周期极长：$2^{19937} - 1$
- 分布均匀性好
- 通过多种统计检验
- 计算效率高

### 7.2 XorShift算法 (`xorshift.py`)

#### 算法原理

基于异或操作的简单快速PRNG。

**基本XorShift**:
```
x ^= x << a
x ^= x >> b  
x ^= x << c
```

其中 $a$, $b$, $c$ 是预设的位移量。

### 7.3 KISS算法 (`kiss.py`)

#### 算法原理

Keep It Simple Stupid，结合多个简单生成器。

**KISS组合**:
$$\text{KISS} = \text{Congruential} + \text{3-shift} + \text{MWC}$$

**线性同余生成器**:
$$x_{n+1} = (ax_n + c) \bmod m$$

**MWC（Multiply-With-Carry）**:
$$x_n = (a \cdot x_{n-1} + c_n) \bmod b$$
$$c_{n+1} = \lfloor (a \cdot x_{n-1} + c_n) / b \rfloor$$

### 6.4 线性反馈移位寄存器 (`lfsr.py`)

#### Galois LFSR

**递推关系**:
$$s_{n+k} = c_1 s_{n+k-1} + c_2 s_{n+k-2} + \cdots + c_k s_n$$

其中运算在有限域 $GF(2)$ 中进行（即异或运算）。

**特征多项式**:
$$P(x) = x^k + c_1 x^{k-1} + c_2 x^{k-2} + \cdots + c_k$$

**最大周期**: $2^k - 1$（当特征多项式是本原多项式时）

### 6.5 Well512算法 (`well.py`)

#### 算法原理

Well (Well Equidistributed Long-period Linear) 是Mersenne Twister的改进版本。

**状态转移函数**:
$$x_i = x_{i-32} \oplus (x_{i-3} | x_{i-25}) \oplus x_{i-1}$$

**输出函数**:
包含多个异或和位移操作，确保输出的统计性质。

#### 特点
- 周期长度: $2^{512} - 1$
- 更好的等分布性质
- 更快的收敛到均匀分布

### 6.6 PCG算法 (`pcg.py`)

#### 算法原理

PCG (Permuted Congruential Generator) 结合线性同余生成器和置换函数。

**内部状态更新**:
$$\text{state} = a \cdot \text{state} + c$$

**输出函数**:
$$\text{output} = \text{permute}(\text{state} \gg r)$$

其中置换函数通常是异或移位操作。

#### 优势
- 空间效率高
- 统计质量好
- 可预测性低

**线性同余生成器**:
$$x_{n+1} = (ax_n + c) \bmod m$$

**MWC（Multiply-With-Carry）**:
$$x_n = (a \cdot x_{n-1} + c_n) \bmod b$$
$$c_{n+1} = \lfloor (a \cdot x_{n-1} + c_n) / b \rfloor$$

### 7.4 线性反馈移位寄存器 (`lfsr.py`)

#### Galois LFSR

**递推关系**:
$$s_{n+k} = c_1 s_{n+k-1} + c_2 s_{n+k-2} + \cdots + c_k s_n$$

其中运算在有限域 $GF(2)$ 中进行（即异或运算）。

**特征多项式**:
$$P(x) = x^k + c_1 x^{k-1} + c_2 x^{k-2} + \cdots + c_k$$

**最大周期**: $2^k - 1$（当特征多项式是本原多项式时）

### 7.5 Well512算法 (`well.py`)

#### 算法原理

Well (Well Equidistributed Long-period Linear) 是Mersenne Twister的改进版本。

**状态转移函数**:
$$x_i = x_{i-32} \oplus (x_{i-3} | x_{i-25}) \oplus x_{i-1}$$

**输出函数**:
包含多个异或和位移操作，确保输出的统计性质。

#### 特点
- 周期长度: $2^{512} - 1$
- 更好的等分布性质
- 更快的收敛到均匀分布

### 7.6 PCG算法 (`pcg.py`)

#### 算法原理

PCG (Permuted Congruential Generator) 结合线性同余生成器和置换函数。

**内部状态更新**:
$$\text{state} = a \cdot \text{state} + c$$

**输出函数**:
$$\text{output} = \text{permute}(\text{state} \gg r)$$

其中置换函数通常是异或移位操作。

#### 优势
- 空间效率高
- 统计质量好
- 可预测性低

## 8. 群体智能优化模块 (Swarm)

### 8.1 粒子群优化 (PSO) (`pso.py`)

#### 算法原理

**速度更新公式**:
$$v_{i}^{t+1} = \omega v_{i}^{t} + c_1 r_1 (p_{i}^{t} - x_{i}^{t}) + c_2 r_2 (g^{t} - x_{i}^{t})$$

**位置更新公式**:
$$x_{i}^{t+1} = x_{i}^{t} + v_{i}^{t+1}$$

其中：
- $\omega$ 是惯性权重
- $c_1, c_2$ 是加速度系数
- $r_1, r_2$ 是随机数
- $p_i$ 是个体最优位置
- $g$ 是全局最优位置

#### 参数选择
- $\omega \in [0.4, 0.9]$
- $c_1, c_2 \in [0, 4]$
- 群体大小：20-50

#### 使用示例
```python
from dwsim_math.swarm.optimizers.pso import PSO
from dwsim_math.swarm.problems.sphere import SphereProblem

problem = SphereProblem(dimensions=10)
optimizer = PSO(swarm_size=30)
result = optimizer.optimize(problem)
```

### 8.2 差分进化 (DE) (`de.py`)

#### 算法原理

**变异操作**:
$$v_i = x_{r1} + F(x_{r2} - x_{r3})$$

**交叉操作**:
$$u_{i,j} = \begin{cases}
v_{i,j} & \text{if } \text{rand}(0,1) \leq CR \text{ or } j = j_{\text{rand}} \\
x_{i,j} & \text{otherwise}
\end{cases}$$

**选择操作**:
$$x_i^{t+1} = \begin{cases}
u_i & \text{if } f(u_i) \leq f(x_i^t) \\
x_i^t & \text{otherwise}
\end{cases}$$

其中：
- $F$ 是缩放因子
- $CR$ 是交叉概率
- $r1, r2, r3$ 是随机选择的不同个体索引

### 7.3 遗传算法 (GA) (`genetic_algorithm.py`)

#### 算法原理

模拟生物进化过程的优化算法。

**编码**:
- 二进制编码: $x = [1,0,1,1,0,1,0,1]$
- 实数编码: $x = [x_1, x_2, \ldots, x_n]$

**选择操作**:

**轮盘赌选择**:
选择概率: $P_i = \frac{f_i}{\sum_{j=1}^N f_j}$

**锦标赛选择**:
从种群中随机选择 $k$ 个个体，选择其中适应度最好的。

**交叉操作**:

**单点交叉**:
$$\text{parent1}: [a_1, a_2, | a_3, a_4, a_5]$$
$$\text{parent2}: [b_1, b_2, | b_3, b_4, b_5]$$
$$\text{child1}: [a_1, a_2, | b_3, b_4, b_5]$$

**算术交叉**（实数编码）:
$$\text{child1} = \alpha \cdot \text{parent1} + (1-\alpha) \cdot \text{parent2}$$
$$\text{child2} = (1-\alpha) \cdot \text{parent1} + \alpha \cdot \text{parent2}$$

**变异操作**:

**位变异**（二进制编码）:
以概率 $p_m$ 翻转每个位。

**高斯变异**（实数编码）:
$$x'_i = x_i + \mathcal{N}(0, \sigma^2)$$

### 7.4 蚁群优化 (ACO) (`ant_colony.py`)

#### 算法原理

模拟蚂蚁觅食行为的优化算法。

**信息素更新**:
$$\tau_{ij}(t+1) = (1-\rho) \tau_{ij}(t) + \sum_{k=1}^m \Delta \tau_{ij}^k$$

其中：
- $\rho$ 是信息素挥发率
- $\Delta \tau_{ij}^k$ 是第 $k$ 只蚂蚁在边 $(i,j)$ 上留下的信息素

**转移概率**:
$$p_{ij}^k(t) = \frac{[\tau_{ij}(t)]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}(t)]^\alpha [\eta_{il}]^\beta}$$

其中：
- $\alpha$ 是信息素重要程度参数
- $\beta$ 是启发式信息重要程度参数
- $\eta_{ij}$ 是启发式信息（通常是 $1/d_{ij}$）

**信息素增量**:
$$\Delta \tau_{ij}^k = \begin{cases}
Q/L_k & \text{if ant } k \text{ uses edge } (i,j) \\
0 & \text{otherwise}
\end{cases}$$

其中 $Q$ 是常数，$L_k$ 是第 $k$ 只蚂蚁的路径长度。

### 7.5 人工蜂群算法 (ABC) (`artificial_bee_colony.py`)

#### 算法原理

模拟蜜蜂采蜜行为的优化算法。

**雇佣蜂阶段**:
$$v_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj})$$

其中：
- $k \in \{1,2,\ldots,SN\}, k \neq i$
- $j \in \{1,2,\ldots,D\}$ 随机选择
- $\phi_{ij}$ 是 $[-1,1]$ 的随机数

**观察蜂选择概率**:
$$P_i = \frac{\text{fit}_i}{\sum_{n=1}^{SN} \text{fit}_n}$$

其中 $\text{fit}_i$ 是适应度值。

**侦察蜂阶段**:
如果食物源连续 $\text{limit}$ 次未改进，则随机产生新解：
$$x_{ij} = x_{j}^{\min} + \text{rand}(0,1) \times (x_{j}^{\max} - x_{j}^{\min})$$

### 7.6 萤火虫算法 (FA) (`firefly.py`)

#### 算法原理

模拟萤火虫发光吸引的优化算法。

**吸引度函数**:
$$\beta(r) = \beta_0 e^{-\gamma r^m}$$

其中：
- $\beta_0$ 是最大吸引度
- $\gamma$ 是光强吸收系数
- $r$ 是萤火虫间距离
- $m \geq 1$

**位置更新**:
$$x_i = x_i + \beta(r_{ij})(x_j - x_i) + \alpha (\text{rand} - 0.5)$$

其中：
- $\alpha$ 是随机化参数
- $\text{rand}$ 是 $[0,1]$ 的随机数

**距离计算**:
$$r_{ij} = \sqrt{\sum_{k=1}^d (x_{i,k} - x_{j,k})^2}$$

### 7.7 灰狼优化算法 (GWO) (`grey_wolf.py`)

#### 算法原理

模拟灰狼社会层级和狩猎行为。

**社会层级**:
- $\alpha$: 领导狼（最优解）
- $\beta$: 副领导狼（次优解）
- $\delta$: 第三等级狼（第三优解）
- $\omega$: 其他狼

**位置更新**:
$$\vec{D}_\alpha = |\vec{C}_1 \cdot \vec{X}_\alpha - \vec{X}|$$
$$\vec{D}_\beta = |\vec{C}_2 \cdot \vec{X}_\beta - \vec{X}|$$
$$\vec{D}_\delta = |\vec{C}_3 \cdot \vec{X}_\delta - \vec{X}|$$

$$\vec{X}_1 = \vec{X}_\alpha - \vec{A}_1 \cdot \vec{D}_\alpha$$
$$\vec{X}_2 = \vec{X}_\beta - \vec{A}_2 \cdot \vec{D}_\beta$$
$$\vec{X}_3 = \vec{X}_\delta - \vec{A}_3 \cdot \vec{D}_\delta$$

$$\vec{X}(t+1) = \frac{\vec{X}_1 + \vec{X}_2 + \vec{X}_3}{3}$$

**参数更新**:
$$\vec{A} = 2\vec{a} \cdot \vec{r}_1 - \vec{a}$$
$$\vec{C} = 2 \cdot \vec{r}_2$$

其中 $\vec{a}$ 从 2 线性递减到 0。

### 7.8 鲸鱼优化算法 (WOA) (`whale_optimization.py`)

#### 算法原理

模拟座头鲸的捕食行为。

**包围猎物**:
$$\vec{D} = |\vec{C} \cdot \vec{X}^*(t) - \vec{X}(t)|$$
$$\vec{X}(t+1) = \vec{X}^*(t) - \vec{A} \cdot \vec{D}$$

**螺旋更新位置**:
$$\vec{X}(t+1) = \vec{D}' \cdot e^{bl} \cdot \cos(2\pi l) + \vec{X}^*(t)$$

其中：
- $\vec{D}' = |\vec{X}^*(t) - \vec{X}(t)|$
- $b$ 是螺旋形状常数
- $l$ 是 $[-1,1]$ 的随机数

**搜索猎物**（随机选择）:
$$\vec{D} = |\vec{C} \cdot \vec{X}_{\text{rand}} - \vec{X}|$$
$$\vec{X}(t+1) = \vec{X}_{\text{rand}} - \vec{A} \cdot \vec{D}$$

**参数**:
$$\vec{A} = 2\vec{a} \cdot \vec{r} - \vec{a}$$
$$\vec{C} = 2 \cdot \vec{r}$$

其中 $\vec{a}$ 从 2 线性递减到 0。

### 7.3 遗传算法 (GA) (`genetic_algorithm.py`)

#### 算法原理

模拟生物进化过程的优化算法。

**编码**:
- 二进制编码: $x = [1,0,1,1,0,1,0,1]$
- 实数编码: $x = [x_1, x_2, \ldots, x_n]$

**选择操作**:

**轮盘赌选择**:
选择概率: $P_i = \frac{f_i}{\sum_{j=1}^N f_j}$

**锦标赛选择**:
从种群中随机选择 $k$ 个个体，选择其中适应度最好的。

**交叉操作**:

**单点交叉**:
$$\text{parent1}: [a_1, a_2, | a_3, a_4, a_5]$$
$$\text{parent2}: [b_1, b_2, | b_3, b_4, b_5]$$
$$\text{child1}: [a_1, a_2, | b_3, b_4, b_5]$$

**算术交叉**（实数编码）:
$$\text{child1} = \alpha \cdot \text{parent1} + (1-\alpha) \cdot \text{parent2}$$
$$\text{child2} = (1-\alpha) \cdot \text{parent1} + \alpha \cdot \text{parent2}$$

**变异操作**:

**位变异**（二进制编码）:
以概率 $p_m$ 翻转每个位。

**高斯变异**（实数编码）:
$$x'_i = x_i + \mathcal{N}(0, \sigma^2)$$

### 7.4 蚁群优化 (ACO) (`ant_colony.py`)

#### 算法原理

模拟蚂蚁觅食行为的优化算法。

**信息素更新**:
$$\tau_{ij}(t+1) = (1-\rho) \tau_{ij}(t) + \sum_{k=1}^m \Delta \tau_{ij}^k$$

其中：
- $\rho$ 是信息素挥发率
- $\Delta \tau_{ij}^k$ 是第 $k$ 只蚂蚁在边 $(i,j)$ 上留下的信息素

**转移概率**:
$$p_{ij}^k(t) = \frac{[\tau_{ij}(t)]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}(t)]^\alpha [\eta_{il}]^\beta}$$

其中：
- $\alpha$ 是信息素重要程度参数
- $\beta$ 是启发式信息重要程度参数
- $\eta_{ij}$ 是启发式信息（通常是 $1/d_{ij}$）

**信息素增量**:
$$\Delta \tau_{ij}^k = \begin{cases}
Q/L_k & \text{if ant } k \text{ uses edge } (i,j) \\
0 & \text{otherwise}
\end{cases}$$

其中 $Q$ 是常数，$L_k$ 是第 $k$ 只蚂蚁的路径长度。

### 7.5 人工蜂群算法 (ABC) (`artificial_bee_colony.py`)

#### 算法原理

模拟蜜蜂采蜜行为的优化算法。

**雇佣蜂阶段**:
$$v_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj})$$

其中：
- $k \in \{1,2,\ldots,SN\}, k \neq i$
- $j \in \{1,2,\ldots,D\}$ 随机选择
- $\phi_{ij}$ 是 $[-1,1]$ 的随机数

**观察蜂选择概率**:
$$P_i = \frac{\text{fit}_i}{\sum_{n=1}^{SN} \text{fit}_n}$$

其中 $\text{fit}_i$ 是适应度值。

**侦察蜂阶段**:
如果食物源连续 $\text{limit}$ 次未改进，则随机产生新解：
$$x_{ij} = x_{j}^{\min} + \text{rand}(0,1) \times (x_{j}^{\max} - x_{j}^{\min})$$

### 7.6 萤火虫算法 (FA) (`firefly.py`)

#### 算法原理

模拟萤火虫发光吸引的优化算法。

**吸引度函数**:
$$\beta(r) = \beta_0 e^{-\gamma r^m}$$

其中：
- $\beta_0$ 是最大吸引度
- $\gamma$ 是光强吸收系数
- $r$ 是萤火虫间距离
- $m \geq 1$

**位置更新**:
$$x_i = x_i + \beta(r_{ij})(x_j - x_i) + \alpha (\text{rand} - 0.5)$$

其中：
- $\alpha$ 是随机化参数
- $\text{rand}$ 是 $[0,1]$ 的随机数

**距离计算**:
$$r_{ij} = \sqrt{\sum_{k=1}^d (x_{i,k} - x_{j,k})^2}$$

### 7.7 灰狼优化算法 (GWO) (`grey_wolf.py`)

#### 算法原理

模拟灰狼社会层级和狩猎行为。

**社会层级**:
- $\alpha$: 领导狼（最优解）
- $\beta$: 副领导狼（次优解）
- $\delta$: 第三等级狼（第三优解）
- $\omega$: 其他狼

**位置更新**:
$$\vec{D}_\alpha = |\vec{C}_1 \cdot \vec{X}_\alpha - \vec{X}|$$
$$\vec{D}_\beta = |\vec{C}_2 \cdot \vec{X}_\beta - \vec{X}|$$
$$\vec{D}_\delta = |\vec{C}_3 \cdot \vec{X}_\delta - \vec{X}|$$

$$\vec{X}_1 = \vec{X}_\alpha - \vec{A}_1 \cdot \vec{D}_\alpha$$
$$\vec{X}_2 = \vec{X}_\beta - \vec{A}_2 \cdot \vec{D}_\beta$$
$$\vec{X}_3 = \vec{X}_\delta - \vec{A}_3 \cdot \vec{D}_\delta$$

$$\vec{X}(t+1) = \frac{\vec{X}_1 + \vec{X}_2 + \vec{X}_3}{3}$$

**参数更新**:
$$\vec{A} = 2\vec{a} \cdot \vec{r}_1 - \vec{a}$$
$$\vec{C} = 2 \cdot \vec{r}_2$$

其中 $\vec{a}$ 从 2 线性递减到 0。

### 7.8 鲸鱼优化算法 (WOA) (`whale_optimization.py`)

#### 算法原理

模拟座头鲸的捕食行为。

**包围猎物**:
$$\vec{D} = |\vec{C} \cdot \vec{X}^*(t) - \vec{X}(t)|$$
$$\vec{X}(t+1) = \vec{X}^*(t) - \vec{A} \cdot \vec{D}$$

**螺旋更新位置**:
$$\vec{X}(t+1) = \vec{D}' \cdot e^{bl} \cdot \cos(2\pi l) + \vec{X}^*(t)$$

其中：
- $\vec{D}' = |\vec{X}^*(t) - \vec{X}(t)|$
- $b$ 是螺旋形状常数
- $l$ 是 $[-1,1]$ 的随机数

**搜索猎物**（随机选择）:
$$\vec{D} = |\vec{C} \cdot \vec{X}_{\text{rand}} - \vec{X}|$$
$$\vec{X}(t+1) = \vec{X}_{\text{rand}} - \vec{A} \cdot \vec{D}$$

**参数**:
$$\vec{A} = 2\vec{a} \cdot \vec{r} - \vec{a}$$
$$\vec{C} = 2 \cdot \vec{r}$$

其中 $\vec{a}$ 从 2 线性递减到 0。

## 9. 特殊函数模块 (Special)

### 9.1 Gamma函数 (`gamma.py`)

#### 数学定义

**Gamma函数**:
$$\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} dt$$

**递推关系**:
$$\Gamma(z+1) = z\Gamma(z)$$

**特殊值**:
- $\Gamma(1) = 1$
- $\Gamma(n) = (n-1)!$ 对于正整数 $n$
- $\Gamma(1/2) = \sqrt{\pi}$

#### Stirling近似

对于大的 $z$：
$$\Gamma(z) \approx \sqrt{\frac{2\pi}{z}} \left(\frac{z}{e}\right)^z$$

### 9.2 Bessel函数 (`bessel.py`)

#### 第一类Bessel函数 $J_\nu(z)$

**Bessel方程**:
$$z^2 \frac{d^2 y}{dz^2} + z \frac{dy}{dz} + (z^2 - \nu^2)y = 0$$

**级数展开**:
$$J_\nu(z) = \left(\frac{z}{2}\right)^\nu \sum_{m=0}^{\infty} \frac{(-1)^m}{m!\Gamma(\nu + m + 1)} \left(\frac{z}{2}\right)^{2m}$$

**整数阶递推关系**:
$$J_{n-1}(z) + J_{n+1}(z) = \frac{2n}{z} J_n(z)$$

**微分关系**:
$$\frac{d}{dz}[z^\nu J_\nu(z)] = z^\nu J_{\nu-1}(z)$$
$$\frac{d}{dz}[z^{-\nu} J_\nu(z)] = -z^{-\nu} J_{\nu+1}(z)$$

#### 第二类Bessel函数 $Y_\nu(z)$（Weber函数）

**定义**:
$$Y_\nu(z) = \frac{J_\nu(z) \cos(\nu\pi) - J_{-\nu}(z)}{\sin(\nu\pi)}$$

**整数阶情况**:
$$Y_n(z) = \lim_{\nu \to n} Y_\nu(z)$$

#### 修正Bessel函数

**第一类修正Bessel函数**:
$$I_\nu(z) = i^{-\nu} J_\nu(iz) = \left(\frac{z}{2}\right)^\nu \sum_{m=0}^{\infty} \frac{1}{m!\Gamma(\nu + m + 1)} \left(\frac{z}{2}\right)^{2m}$$

**第二类修正Bessel函数**:
$$K_\nu(z) = \frac{\pi}{2} \frac{I_{-\nu}(z) - I_\nu(z)}{\sin(\nu\pi)}$$

#### 汉克尔函数（第三类Bessel函数）

**第一类汉克尔函数**:
$$H_\nu^{(1)}(z) = J_\nu(z) + i Y_\nu(z)$$

**第二类汉克尔函数**:
$$H_\nu^{(2)}(z) = J_\nu(z) - i Y_\nu(z)$$

#### 渐近展开

**大 $z$ 时的渐近展开**:
$$J_\nu(z) \sim \sqrt{\frac{2}{\pi z}} \cos\left(z - \frac{\nu\pi}{2} - \frac{\pi}{4}\right)$$
$$Y_\nu(z) \sim \sqrt{\frac{2}{\pi z}} \sin\left(z - \frac{\nu\pi}{2} - \frac{\pi}{4}\right)$$

### 9.3 椭圆积分 (`elliptic.py`)

#### 第一类完全椭圆积分

**定义**:
$$K(k) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - k^2 \sin^2 \theta}}$$

**级数展开**:
$$K(k) = \frac{\pi}{2} \sum_{n=0}^{\infty} \left[\frac{(2n)!}{2^{2n}(n!)^2}\right]^2 k^{2n}$$

#### 第二类完全椭圆积分

**定义**:
$$E(k) = \int_0^{\pi/2} \sqrt{1 - k^2 \sin^2 \theta} \, d\theta$$

**级数展开**:
$$E(k) = \frac{\pi}{2} \sum_{n=0}^{\infty} \left[\frac{(2n)!}{2^{2n}(n!)^2}\right]^2 \frac{k^{2n}}{1 - 2n}$$

#### Legendre-Jacobi椭圆积分

**第一类不完全椭圆积分**:
$$F(\phi, k) = \int_0^\phi \frac{d\theta}{\sqrt{1 - k^2 \sin^2 \theta}}$$

**第二类不完全椭圆积分**:
$$E(\phi, k) = \int_0^\phi \sqrt{1 - k^2 \sin^2 \theta} \, d\theta$$

**第三类不完全椭圆积分**:
$$\Pi(n, \phi, k) = \int_0^\phi \frac{d\theta}{(1 - n \sin^2 \theta)\sqrt{1 - k^2 \sin^2 \theta}}$$

### 9.4 误差函数 (`error_function.py`)

#### 误差函数

**定义**:
$$\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt$$

**级数展开**:
$$\text{erf}(z) = \frac{2}{\sqrt{\pi}} \sum_{n=0}^{\infty} \frac{(-1)^n z^{2n+1}}{n!(2n+1)}$$

**渐近展开**（大 $z$ 时）:
$$\text{erf}(z) \sim 1 - \frac{e^{-z^2}}{\sqrt{\pi}z} \left(1 - \frac{1}{2z^2} + \frac{3}{4z^4} - \cdots\right)$$

#### 互补误差函数

**定义**:
$$\text{erfc}(z) = 1 - \text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_z^\infty e^{-t^2} dt$$

#### 复数误差函数（Faddeeva函数）

**定义**:
$$w(z) = e^{-z^2} \text{erfc}(-iz)$$

#### Fresnel积分

**Fresnel正弦积分**:
$$S(x) = \int_0^x \sin\left(\frac{\pi t^2}{2}\right) dt$$

**Fresnel余弦积分**:
$$C(x) = \int_0^x \cos\left(\frac{\pi t^2}{2}\right) dt$$

## 10. 积分器模块 (Integration)

### 10.1 Newton-Cotes积分公式

#### 梯形公式

**简单梯形公式**:
$$\int_a^b f(x)dx \approx \frac{h}{2}[f(a) + f(b)]$$

其中 $h = b - a$。

**复合梯形公式**:
$$\int_a^b f(x)dx \approx \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]$$

其中 $h = \frac{b-a}{n}$，$x_i = a + ih$。

**误差估计**:
$$E = -\frac{h^3}{12}(b-a)f''(\xi), \quad \xi \in [a,b]$$

#### Simpson公式

**简单Simpson公式**:
$$\int_a^b f(x)dx \approx \frac{h}{3}[f(a) + 4f(\frac{a+b}{2}) + f(b)]$$

其中 $h = \frac{b-a}{2}$。

**复合Simpson公式**:
$$\int_a^b f(x)dx \approx \frac{h}{3}\left[f(x_0) + 4\sum_{i=1,3,5}^{n-1} f(x_i) + 2\sum_{i=2,4,6}^{n-2} f(x_i) + f(x_n)\right]$$

其中 $h = \frac{b-a}{n}$，$n$ 必须是偶数。

**误差估计**:
$$E = -\frac{h^5}{90}(b-a)f^{(4)}(\xi), \quad \xi \in [a,b]$$

#### Simpson 3/8公式

**基本公式**:
$$\int_a^b f(x)dx \approx \frac{3h}{8}[f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3)]$$

其中 $h = \frac{b-a}{3}$。

#### 高阶Newton-Cotes公式

**Boole公式**（5点公式）:
$$\int_a^b f(x)dx \approx \frac{2h}{45}[7f(x_0) + 32f(x_1) + 12f(x_2) + 32f(x_3) + 7f(x_4)]$$

### 10.2 自适应积分 (`adaptive.py`)

#### 自适应Simpson积分

**算法原理**:
1. 计算整个区间的Simpson积分 $S_1$
2. 分割区间为两半，分别计算Simpson积分 $S_2$
3. 如果 $|S_1 - S_2| < \epsilon$，则接受 $S_2$
4. 否则递归细分

**误差控制**:
$$\left|\int_a^b f(x)dx - S_2\right| \approx \frac{|S_1 - S_2|}{15}$$

#### 自适应Gauss-Kronrod积分

**Gauss-Legendre积分**:
$$\int_{-1}^1 f(x)dx \approx \sum_{i=1}^n w_i f(x_i)$$

**Kronrod扩展**:
在Gauss节点基础上增加 $n+1$ 个点，形成 $2n+1$ 点公式。

**误差估计**:
$$E \approx |I_{2n+1} - I_n|$$

其中 $I_n$ 是 $n$ 点Gauss积分，$I_{2n+1}$ 是 $2n+1$ 点Kronrod积分。

### 10.3 Gauss求积公式 (`gauss.py`)

#### Gauss-Legendre求积

**权重和节点**:
节点 $x_i$ 是Legendre多项式 $P_n(x)$ 的零点：
$$P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n}(x^2-1)^n$$

**权重**:
$$w_i = \frac{2}{(1-x_i^2)[P_n'(x_i)]^2}$$

**积分公式**:
$$\int_{-1}^1 f(x)dx \approx \sum_{i=1}^n w_i f(x_i)$$

#### Gauss-Laguerre求积

**用于积分区间 $[0, \infty)$**:
$$\int_0^\infty e^{-x} f(x)dx \approx \sum_{i=1}^n w_i f(x_i)$$

其中 $x_i$ 是Laguerre多项式 $L_n(x)$ 的零点。

#### Gauss-Hermite求积

**用于积分区间 $(-\infty, \infty)$**:
$$\int_{-\infty}^\infty e^{-x^2} f(x)dx \approx \sum_{i=1}^n w_i f(x_i)$$

其中 $x_i$ 是Hermite多项式 $H_n(x)$ 的零点。

#### Gauss-Chebyshev求积

**第一类Chebyshev多项式**:
$$\int_{-1}^1 \frac{f(x)}{\sqrt{1-x^2}}dx \approx \sum_{i=1}^n w_i f(x_i)$$

其中 $x_i = \cos\left(\frac{(2i-1)\pi}{2n}\right)$，$w_i = \frac{\pi}{n}$。

### 10.4 多维积分 (`multidimensional.py`)

#### 蒙特卡罗积分

**基本思想**:
$$\int_D f(x)dx \approx \frac{|D|}{N} \sum_{i=1}^N f(x_i)$$

其中 $x_i$ 是在域 $D$ 中的随机点，$|D|$ 是域的体积。

**方差**:
$$\text{Var}(I) = \frac{|D|^2}{N} \left[\frac{1}{N}\sum_{i=1}^N f(x_i)^2 - \left(\frac{1}{N}\sum_{i=1}^N f(x_i)\right)^2\right]$$

**重要性采样**:
使用概率密度函数 $p(x)$ 采样：
$$\int_D f(x)dx \approx \frac{1}{N} \sum_{i=1}^N \frac{f(x_i)}{p(x_i)}$$

#### 多维Gauss求积

**积分区域为超立方体**:
$$\int_{[-1,1]^d} f(x_1, \ldots, x_d) dx_1 \cdots dx_d \approx \sum_{i_1=1}^{n_1} \cdots \sum_{i_d=1}^{n_d} w_{i_1} \cdots w_{i_d} f(x_{i_1}, \ldots, x_{i_d})$$

#### 稀疏网格积分（Smolyak积分）

**基本思想**:
减少多维积分中的节点数量，从 $O(n^d)$ 降低到 $O(n \log^{d-1} n)$。

**Smolyak公式**:
$$A(q,d) = \sum_{q-d+1 \leq |i| \leq q} (-1)^{q-|i|} \binom{d-1}{q-|i|} (U^{i_1} \otimes \cdots \otimes U^{i_d})$$

其中 $U^i$ 是一维求积公式。

### 10.5 奇异积分 (`singular.py`)

#### 分部积分

**处理形如 $\int_a^b \frac{f(x)}{(x-c)^\alpha} dx$ 的积分**，其中 $c \in [a,b]$，$0 < \alpha < 1$。

**变量替换**:
设 $x - c = t^\beta$，其中 $\beta = \frac{1}{1-\alpha}$。

#### 渐近展开

**处理振荡积分**:
$$\int_a^\infty f(x) e^{i\omega x} dx$$

**驻相法**:
寻找 $\phi'(x) = 0$ 的驻相点，其中 $\phi(x)$ 是相位函数。

#### Cauchy主值积分

**定义**:
$$\text{P.V.} \int_a^b \frac{f(x)}{x-c} dx = \lim_{\epsilon \to 0^+} \left[\int_a^{c-\epsilon} \frac{f(x)}{x-c} dx + \int_{c+\epsilon}^b \frac{f(x)}{x-c} dx\right]$$

**计算方法**:
$$\text{P.V.} \int_a^b \frac{f(x)}{x-c} dx = \int_a^b \frac{f(x) - f(c)}{x-c} dx + f(c) \ln\left|\frac{b-c}{a-c}\right|$$

## 11. 模块依赖关系

### 11.1 依赖层次结构

```
Level 1 (基础层):
├── core.general          # 基础数学函数
├── numerics.complex      # 复数运算
└── random.base          # 随机数基类

Level 2 (核心数值层):
├── core.matrix_ops      # 依赖 core.general
├── core.interpolation   # 依赖 core.general
├── numerics.linear_algebra # 依赖 numerics.complex
├── random.mersenne_twister # 依赖 random.base
├── random.xorshift      # 依赖 random.base
├── random.kiss          # 依赖 random.base
├── random.lfsr          # 依赖 random.base
├── random.well          # 依赖 random.base
└── random.pcg           # 依赖 random.base

Level 3 (求解器层):
├── solvers.brent        # 依赖 core.general
├── solvers.broyden      # 依赖 core.matrix_ops
├── solvers.linear_system # 依赖 core.matrix_ops
├── numerics.ode         # 依赖 numerics.linear_algebra
├── special.gamma        # 依赖 core.general
├── special.bessel       # 依赖 core.general
├── special.elliptic     # 依赖 core.general
└── special.error_function # 依赖 core.general

Level 4 (优化算法层):
├── optimization.bfgs    # 依赖 solvers.*, core.matrix_ops
├── optimization.lbfgs   # 依赖 optimization.bfgs
├── optimization.lm      # 依赖 numerics.linear_algebra, solvers.*
├── optimization.conjugate_gradient # 依赖 core.general
├── optimization.trust_region # 依赖 core.matrix_ops, solvers.*
├── swarm.pso           # 依赖 random.*, core.*
├── swarm.de            # 依赖 random.*, core.*
├── swarm.ga            # 依赖 random.*, core.*
├── swarm.aco           # 依赖 random.*, core.*
├── swarm.abc           # 依赖 random.*, core.*
├── swarm.firefly       # 依赖 random.*, core.*
├── swarm.grey_wolf     # 依赖 random.*, core.*
└── swarm.whale         # 依赖 random.*, core.*

Level 5 (积分器层):
├── integration.newton_cotes # 依赖 core.general
├── integration.adaptive # 依赖 integration.newton_cotes
├── integration.gauss    # 依赖 special.*, core.*
├── integration.multidimensional # 依赖 random.*, integration.gauss
└── integration.singular # 依赖 special.*, integration.*

Level 6 (应用层):
├── swarm.problems       # 依赖 optimization.*, special.*
└── benchmarks          # 依赖 所有模块
```

### 11.2 模块间接口

#### 核心接口
- `MathCommon`: 提供基础数学函数
- `MatrixOperations`: 提供矩阵运算接口
- `Complex`: 提供复数运算接口

#### 求解器接口
- `BrentSolver`: 单变量方程求根
- `LinearSystemSolver`: 线性方程组求解

#### 优化器接口
- `Optimizer`: 优化器基类
- `Problem`: 优化问题基类

## 12. 性能优化建议

### 12.1 数值计算优化

1. **使用NumPy向量化操作**
   ```python
   # 优化前
   result = []
   for i in range(len(x)):
       result.append(x[i] ** 2)
   
   # 优化后
   result = np.square(x)
   ```

2. **避免不必要的内存分配**
   ```python
   # 预分配数组
   result = np.empty(n)
   for i in range(n):
       result[i] = compute(x[i])
   ```

3. **使用Numba JIT编译**
   ```python
   from numba import jit
   
   @jit(nopython=True)
   def fast_function(x):
       return x ** 2 + 2 * x + 1
   ```

### 12.2 内存优化

1. **就地操作**
   ```python
   # 避免创建新数组
   A += B  # 而不是 A = A + B
   ```

2. **使用适当的数据类型**
   ```python
   # 根据精度需求选择数据类型
   x = np.array(data, dtype=np.float32)  # 而不是默认的float64
   ```

## 13. 测试和验证

### 13.1 单元测试示例

```python
import unittest
import numpy as np
from dwsim_math.core.matrix_ops import MatrixOperations

class TestMatrixOperations(unittest.TestCase):
    
    def test_determinant_2x2(self):
        A = [[1, 2], [3, 4]]
        det = MatrixOperations.determinant(A)
        self.assertAlmostEqual(det, -2.0, places=10)
    
    def test_inverse_identity(self):
        I = np.eye(3)
        inv_I, success = MatrixOperations.inverse(I)
        self.assertTrue(success)
        np.testing.assert_array_almost_equal(inv_I, I)
```

### 13.2 数值精度验证

```python
def verify_numerical_accuracy():
    """验证数值算法的精度"""
    # 测试已知解析解的问题
    # 比较数值解与解析解的误差
    pass
```

## 14. 使用指南

### 14.1 快速开始

```python
# 导入库
import dwsim_math as dm
import numpy as np

# 矩阵运算
A = [[1, 2], [3, 4]]
det_A = dm.core.matrix_ops.determinant(A)
inv_A, success = dm.core.matrix_ops.inverse(A)

# 复数运算
z = dm.Complex(3, 4)
result = z.exp()
magnitude = z.abs()
phase = z.arg()

# 方程求根
def f(x, args):
    return x**3 - 2*x - 5
root = dm.solvers.brent_solve(f, 1.0, 3.0)

# 优化问题
problem = dm.swarm.problems.SphereProblem(10)

# 粒子群优化
pso = dm.swarm.optimizers.PSO(swarm_size=30)
pso_result = pso.optimize(problem)

# 差分进化
de = dm.swarm.optimizers.DE(population_size=50)
de_result = de.optimize(problem)

# 遗传算法
ga = dm.swarm.optimizers.GA(population_size=100)
ga_result = ga.optimize(problem)

# BFGS优化
def objective(x):
    return np.sum(x**2)  # 简单的球函数

def gradient(x):
    return 2*x

bfgs = dm.optimization.BFGS()
bfgs_result = bfgs.minimize(objective, gradient, np.random.rand(5))

# 数值积分
def integrand(x):
    return np.exp(-x**2)

# Simpson积分
simpson_result = dm.integration.simpson(integrand, -2, 2, n=1000)

# Gauss-Legendre积分
gauss_result = dm.integration.gauss_legendre(integrand, -2, 2, n=10)

# 随机数生成
mt = dm.random.MersenneTwister(seed=12345)
random_numbers = [mt.random() for _ in range(1000)]

# 特殊函数
gamma_val = dm.special.gamma(2.5)
bessel_val = dm.special.bessel_j(0, 1.0)
erf_val = dm.special.erf(1.0)

# 常微分方程求解
def ode_func(t, y):
    return -2*y + np.sin(t)

solver = dm.numerics.ode.RungeKutta4()
t_span = (0, 5)
y0 = [1.0]
solution = solver.solve(ode_func, t_span, y0, step_size=0.01)
```

### 14.2 最佳实践

1. **错误处理**
   ```python
   try:
       result = solver.solve(function, a, b)
   except ValueError as e:
       print(f"求解失败: {e}")
   ```

2. **参数验证**
   ```python
   if not isinstance(matrix, np.ndarray):
       matrix = np.asarray(matrix)
   ```

3. **性能监控**
   ```python
   import time
   start_time = time.time()
   result = expensive_computation()
   elapsed_time = time.time() - start_time
   ```

## 15. 扩展开发

### 15.1 添加新算法

1. 继承适当的基类
2. 实现必要的抽象方法
3. 添加完整的文档字符串
4. 编写单元测试

### 15.2 性能基准测试

```python
import timeit

def benchmark_algorithm():
    """算法性能基准测试"""
    setup_code = "import dwsim_math as dm"
    test_code = "dm.core.matrix_ops.determinant([[1,2],[3,4]])"
    
    time_result = timeit.timeit(test_code, setup=setup_code, number=10000)
    print(f"执行时间: {time_result:.6f} 秒")
```

---

**文档版本**: 1.0.0  
**最后更新**: 2024年  
**维护者**: DWSIM团队 