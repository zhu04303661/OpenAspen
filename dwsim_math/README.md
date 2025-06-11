# DWSIM 数学计算库 (Python版本)

## 项目简介

这是DWSIM（开源过程仿真器）数学计算库的Python版本，从原来的VB.NET和C#代码完整转换而来。该库提供了过程仿真中所需的各种数学算法和数值计算功能。

## 模块结构

### 1. 核心数学模块 (`dwsim_math.core`)
- **general.py**: 通用数学函数（最大值、最小值、求和、加权平均等）
- **matrix_ops.py**: 矩阵操作（行列式、逆矩阵、LU分解等）
- **interpolation.py**: 插值算法（线性、样条、多项式插值）
- **extrapolation.py**: 外推算法
- **intersection.py**: 交点计算算法

### 2. 求解器模块 (`dwsim_math.solvers`)
- **linear_system.py**: 线性方程组求解器
- **brent.py**: Brent方法求根
- **broyden.py**: Broyden方法求解非线性方程组
- **lm.py**: Levenberg-Marquardt算法
- **lm_fit.py**: LM拟合算法
- **lp_solve.py**: 线性规划求解器

### 3. 优化模块 (`dwsim_math.optimization`)
- **lbfgs.py**: L-BFGS优化算法
- **lbfgsb.py**: L-BFGS-B有界优化算法
- **brent_minimize.py**: Brent最小化算法
- **gdem.py**: 全局差分进化算法

### 4. 数值计算模块 (`dwsim_math.numerics`)
- **complex.py**: 复数运算
- **linear_algebra/**: 线性代数模块
  - **matrix.py**: 矩阵类
  - **vector.py**: 向量类
  - **eigenvalue.py**: 特征值计算
  - **svd.py**: 奇异值分解
- **ode/**: 常微分方程求解器
  - **runge_kutta.py**: 龙格-库塔方法
  - **adams_moulton.py**: Adams-Moulton方法
  - **gear_bdf.py**: Gear BDF方法

### 5. 随机数生成模块 (`dwsim_math.random`)
- **random_base.py**: 随机数生成器基类
- **mersenne_twister.py**: 梅森旋转算法
- **xorshift.py**: XorShift算法
- **kiss.py**: KISS算法
- **cmwc4096.py**: CMWC4096算法
- **distributions/**: 各种分布
  - **gaussian.py**: 正态分布
  - **uniform.py**: 均匀分布
  - **sphere.py**: 球面分布

### 6. 群体智能优化模块 (`dwsim_math.swarm`)
- **problem.py**: 优化问题基类
- **optimizers/**: 各种优化器
  - **pso.py**: 粒子群优化
  - **de.py**: 差分进化
  - **jde.py**: 自适应差分进化
  - **mol.py**: 蛾子优化算法
  - **mesh.py**: 网格搜索
- **problems/**: 标准测试问题
  - **sphere.py**: 球函数
  - **rosenbrock.py**: Rosenbrock函数
  - **ackley.py**: Ackley函数
  - **rastrigin.py**: Rastrigin函数

### 7. 特殊函数模块 (`dwsim_math.special`)
- **gamma.py**: Gamma函数及相关函数
- **bessel.py**: 贝塞尔函数
- **elliptic.py**: 椭圆积分

## 主要功能

### 1. 线性代数运算
```python
from dwsim_math.core.matrix_ops import MatrixOperations
from dwsim_math.numerics.linear_algebra.matrix import Matrix

# 创建矩阵
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

# 矩阵运算
C = A * B  # 矩阵乘法
det_A = A.determinant()  # 行列式
inv_A = A.inverse()  # 逆矩阵
```

### 2. 非线性方程求解
```python
from dwsim_math.solvers.brent import BrentSolver

def f(x):
    return x**3 - 2*x - 5

solver = BrentSolver()
root = solver.solve(f, 1.0, 3.0)  # 在区间[1,3]中求根
```

### 3. 优化问题求解
```python
from dwsim_math.optimization.lbfgs import LBFGS

def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

optimizer = LBFGS()
result = optimizer.minimize(objective, [0, 0])
```

### 4. 常微分方程求解
```python
from dwsim_math.numerics.ode.runge_kutta import RungeKutta45

def ode_system(t, y):
    return [-y[0], y[1]]

solver = RungeKutta45()
solution = solver.solve(ode_system, [0, 10], [1, 0])
```

### 5. 群体智能优化
```python
from dwsim_math.swarm.optimizers.pso import PSO
from dwsim_math.swarm.problems.sphere import SphereProblem

problem = SphereProblem(dimensions=10)
optimizer = PSO(swarm_size=30)
result = optimizer.optimize(problem)
```

### 6. 随机数生成
```python
from dwsim_math.random.mersenne_twister import MersenneTwister
from dwsim_math.random.distributions.gaussian import GaussianDistribution

rng = MersenneTwister(seed=12345)
gauss = GaussianDistribution(rng)
random_numbers = gauss.sample(1000, mean=0, std=1)
```

## 安装和依赖

### 必需依赖
```
numpy >= 1.20.0
scipy >= 1.7.0
matplotlib >= 3.3.0
```

### 可选依赖
```
numba >= 0.50.0  # 用于加速计算
joblib >= 1.0.0  # 用于并行计算
```

## 使用示例

### 完整的优化问题求解示例
```python
import numpy as np
from dwsim_math.swarm.optimizers.pso import PSO
from dwsim_math.swarm.problem import Problem

class CustomProblem(Problem):
    def __init__(self):
        super().__init__()
        self._name = "Custom Optimization Problem"
        self._dimensionality = 2
        self._lower_bound = np.array([-5.0, -5.0])
        self._upper_bound = np.array([5.0, 5.0])
        self._min_fitness = 0.0
    
    def fitness(self, parameters):
        x, y = parameters
        return (x - 1)**2 + (y - 2)**2

# 创建问题实例
problem = CustomProblem()

# 创建优化器
optimizer = PSO(
    swarm_size=50,
    max_iterations=1000,
    inertia_weight=0.9,
    cognitive_coefficient=2.0,
    social_coefficient=2.0
)

# 执行优化
result = optimizer.optimize(problem)
print(f"最优解: {result.parameters}")
print(f"最优值: {result.fitness}")
```

## 性能特点

1. **高效数值计算**: 使用NumPy和SciPy作为底层计算引擎
2. **内存优化**: 智能内存管理，避免不必要的数组复制
3. **并行计算**: 支持多线程和多进程并行计算
4. **数值稳定性**: 采用数值稳定的算法实现
5. **可扩展性**: 模块化设计，易于扩展新功能

## 测试和验证

每个模块都包含完整的单元测试，验证与原始DWSIM库的计算结果一致性。

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定模块测试
python -m pytest tests/test_matrix_ops.py
```

## 许可证

本项目遵循GNU General Public License v3.0许可证，与原始DWSIM项目保持一致。 