# DWSIM数学计算库使用示例

## 1. 快速开始

### 1.1 安装依赖

```bash
pip install numpy scipy matplotlib
```

### 1.2 导入库

```python
import dwsim_math as dm
import numpy as np
```

## 2. 核心数学模块示例

### 2.1 通用数学函数

```python
from dwsim_math.core.general import MathCommon

# 计算加权平均
weights = [0.3, 0.4, 0.3]
values = [10, 20, 15]
weighted_avg = MathCommon.weighted_average(weights, values)
print(f"加权平均: {weighted_avg}")  # 16.5

# 计算标准差
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
std_dev = MathCommon.standard_deviation(data)
print(f"标准差: {std_dev}")

# 计算平方和
sum_of_squares = MathCommon.sum_of_squares(data)
print(f"平方和: {sum_of_squares}")
```

### 2.2 矩阵操作

```python
from dwsim_math.core.matrix_ops import MatrixOperations
import numpy as np

# 创建矩阵
A = [[2, 1, 3], 
     [1, 3, 2], 
     [3, 2, 1]]

# 计算行列式
det_A = MatrixOperations.determinant(A)
print(f"行列式: {det_A}")

# 计算逆矩阵
inv_A, success = MatrixOperations.inverse(A)
if success:
    print("逆矩阵:")
    print(inv_A)
    
    # 验证 A * A^(-1) = I
    identity = np.dot(A, inv_A)
    print("验证 A * A^(-1):")
    print(identity)

# 求解线性方程组 Ax = b
b = [1, 2, 3]
x, success = MatrixOperations.solve_linear_system(A, b)
if success:
    print(f"方程组解: {x}")
```

### 2.3 插值算法

```python
from dwsim_math.core.interpolation import Interpolation
import matplotlib.pyplot as plt

# 准备数据
x_data = [0, 1, 2, 3, 4, 5]
y_data = [1, 4, 9, 16, 25, 36]  # y = x^2 + 1

# 插值点
x_interp = 2.5

# 不同插值方法
linear_result = Interpolation.interpolate(x_data, y_data, x_interp, method="linear")
rational_result = Interpolation.interpolate(x_data, y_data, x_interp, method="rational")
polynomial_result = Interpolation.interpolate(x_data, y_data, x_interp, method="polynomial")

print(f"线性插值在x={x_interp}: {linear_result}")
print(f"有理插值在x={x_interp}: {rational_result}")
print(f"多项式插值在x={x_interp}: {polynomial_result}")

# 绘制插值结果
x_plot = np.linspace(0, 5, 100)
y_linear = [Interpolation.interpolate(x_data, y_data, x, "linear") for x in x_plot]
y_rational = [Interpolation.interpolate(x_data, y_data, x, "rational") for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ro', label='原始数据')
plt.plot(x_plot, y_linear, 'b-', label='线性插值')
plt.plot(x_plot, y_rational, 'g-', label='有理插值')
plt.legend()
plt.grid(True)
plt.title('插值方法比较')
plt.show()
```

## 3. 数值计算模块示例

### 3.1 复数运算

```python
from dwsim_math.numerics.complex_number import Complex
import math

# 创建复数
z1 = Complex(3, 4)  # 3 + 4i
z2 = Complex(1, -2)  # 1 - 2i

print(f"z1 = {z1}")
print(f"z2 = {z2}")

# 基本运算
print(f"z1 + z2 = {z1 + z2}")
print(f"z1 - z2 = {z1 - z2}")
print(f"z1 * z2 = {z1 * z2}")
print(f"z1 / z2 = {z1 / z2}")

# 复数属性
print(f"z1的模长: {z1.modulus}")
print(f"z1的幅角: {z1.argument}")
print(f"z1的共轭: {z1.conjugate}")

# 复数函数
print(f"exp(z1) = {z1.exp()}")
print(f"ln(z1) = {z1.log()}")
print(f"sqrt(z1) = {z1.sqrt()}")
print(f"z1^2 = {z1.power(2)}")

# 三角函数
print(f"sin(z1) = {z1.sin()}")
print(f"cos(z1) = {z1.cos()}")

# 从字符串创建复数
z3 = Complex.from_string("2+3i")
print(f"从字符串创建: {z3}")

# 从极坐标创建复数
z4 = Complex.from_polar(5, math.pi/4)
print(f"从极坐标创建: {z4}")
```

## 4. 求解器模块示例

### 4.1 Brent方法求根

```python
from dwsim_math.solvers.brent import BrentSolver
import numpy as np
import matplotlib.pyplot as plt

# 定义要求根的函数
def func1(x, args):
    return x**3 - 2*x - 5

def func2(x, args):
    return np.sin(x) - 0.5

def func3(x, args):
    return np.exp(x) - 2*x - 1

# 创建求解器
solver = BrentSolver()

# 求解第一个方程
print("求解 x^3 - 2x - 5 = 0")
root1 = solver.solve(func1, 1.0, 3.0)
print(f"根: {root1}")
print(f"验证: f({root1}) = {func1(root1, None)}")

# 求解第二个方程
print("\n求解 sin(x) - 0.5 = 0")
root2 = solver.solve(func2, 0.0, 1.0)
print(f"根: {root2}")
print(f"验证: f({root2}) = {func2(root2, None)}")

# 求解第三个方程
print("\n求解 exp(x) - 2x - 1 = 0")
root3 = solver.solve(func3, 0.0, 2.0)
print(f"根: {root3}")
print(f"验证: f({root3}) = {func3(root3, None)}")

# 寻找函数最小值
def objective(x, args):
    return (x - 2)**2 + 3

print("\n寻找 (x-2)^2 + 3 的最小值")
x_min, f_min = solver.find_minimum(objective, 0.0, 5.0)
print(f"最小值点: {x_min}")
print(f"最小值: {f_min}")

# 绘制函数图像
x = np.linspace(0, 5, 1000)
y = [(xi - 2)**2 + 3 for xi in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = (x-2)² + 3')
plt.plot(x_min, f_min, 'ro', markersize=8, label=f'最小值 ({x_min:.3f}, {f_min:.3f})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('函数优化示例')
plt.legend()
plt.grid(True)
plt.show()
```

## 5. 优化模块示例

### 5.1 L-BFGS优化

```python
from dwsim_math.optimization.lbfgs import LBFGS
import numpy as np

# 定义Rosenbrock函数及其梯度
def rosenbrock_func_grad(x):
    """Rosenbrock函数: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    grad = np.zeros(2)
    grad[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    grad[1] = 200*(x[1] - x[0]**2)
    
    return f, grad

# 创建优化器
optimizer = LBFGS(m=7, eps_g=1e-6, max_iterations=1000)

# 设置迭代回调函数
def iteration_callback(x, f, g):
    print(f"迭代: x=({x[0]:.6f}, {x[1]:.6f}), f={f:.6f}, ||g||={np.linalg.norm(g):.6f}")
    return True  # 继续迭代

optimizer.set_iteration_callback(iteration_callback)

# 优化
initial_point = np.array([-1.2, 1.0])
result = optimizer.minimize(rosenbrock_func_grad, initial_point)

print("\n优化结果:")
print(f"成功: {result['success']}")
print(f"最优解: ({result['x'][0]:.6f}, {result['x'][1]:.6f})")
print(f"最优值: {result['fun']:.6f}")
print(f"迭代次数: {result['nit']}")
print(f"函数评估次数: {result['nfev']}")
print(f"状态信息: {result['message']}")

# 验证结果（理论最优解是 (1, 1)）
print(f"\n理论最优解: (1, 1)")
print(f"误差: ({abs(result['x'][0] - 1):.6f}, {abs(result['x'][1] - 1):.6f})")
```

### 5.2 多目标优化示例

```python
# 定义二次函数优化问题
def quadratic_func_grad(x):
    """二次函数: f(x) = x^T Q x + b^T x + c"""
    Q = np.array([[2, 1], [1, 3]])
    b = np.array([1, -2])
    c = 5
    
    f = 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(b, x) + c
    grad = np.dot(Q, x) + b
    
    return f, grad

# 优化
result = optimizer.minimize(quadratic_func_grad, np.array([0, 0]))

print("\n二次函数优化结果:")
print(f"最优解: ({result['x'][0]:.6f}, {result['x'][1]:.6f})")
print(f"最优值: {result['fun']:.6f}")

# 解析解验证
Q = np.array([[2, 1], [1, 3]])
b = np.array([1, -2])
analytical_solution = -np.linalg.solve(Q, b)
print(f"解析解: ({analytical_solution[0]:.6f}, {analytical_solution[1]:.6f})")
```

## 6. 随机数生成模块示例

### 6.1 Mersenne Twister生成器

```python
from dwsim_math.random.mersenne_twister import MersenneTwister
import matplotlib.pyplot as plt

# 创建随机数生成器
rng = MersenneTwister(seed=12345)

# 生成不同类型的随机数
print("随机数生成示例:")

# 均匀分布
uniform_samples = [rng.random() for _ in range(10)]
print(f"均匀分布[0,1): {uniform_samples}")

# 指定范围的均匀分布
uniform_range = [rng.uniform(10, 20) for _ in range(5)]
print(f"均匀分布[10,20): {uniform_range}")

# 随机整数
random_ints = [rng.randint(1, 100) for _ in range(10)]
print(f"随机整数[1,100): {random_ints}")

# 正态分布
normal_samples = [rng.normal(0, 1) for _ in range(10)]
print(f"标准正态分布: {normal_samples}")

# 指数分布
exp_samples = [rng.exponential(2.0) for _ in range(5)]
print(f"指数分布(λ=2): {exp_samples}")

# 随机选择
fruits = ['apple', 'banana', 'orange', 'grape', 'kiwi']
chosen = rng.choice(fruits, size=3)
print(f"随机选择: {chosen}")

# 随机抽样
sample = rng.sample(fruits, 3)
print(f"无重复抽样: {sample}")

# 洗牌
deck = list(range(1, 11))
print(f"原始顺序: {deck}")
rng.shuffle(deck)
print(f"洗牌后: {deck}")

# 生成大量样本进行统计分析
n_samples = 10000

# 正态分布样本
normal_data = [rng.normal(0, 1) for _ in range(n_samples)]

# 均匀分布样本  
uniform_data = [rng.random() for _ in range(n_samples)]

# 绘制直方图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(normal_data, bins=50, density=True, alpha=0.7)
ax1.set_title('正态分布 N(0,1)')
ax1.set_xlabel('值')
ax1.set_ylabel('密度')
ax1.grid(True)

ax2.hist(uniform_data, bins=50, density=True, alpha=0.7)
ax2.set_title('均匀分布 U(0,1)')
ax2.set_xlabel('值')
ax2.set_ylabel('密度')
ax2.grid(True)

plt.tight_layout()
plt.show()

# 统计特性验证
print(f"\n统计验证:")
print(f"正态分布样本均值: {np.mean(normal_data):.4f} (理论值: 0)")
print(f"正态分布样本标准差: {np.std(normal_data):.4f} (理论值: 1)")
print(f"均匀分布样本均值: {np.mean(uniform_data):.4f} (理论值: 0.5)")
print(f"均匀分布样本标准差: {np.std(uniform_data):.4f} (理论值: {1/np.sqrt(12):.4f})")
```

## 7. 蒙特卡罗模拟示例

### 7.1 估算π值

```python
def estimate_pi_monte_carlo(n_samples=1000000):
    """使用蒙特卡罗方法估算π值"""
    rng = MersenneTwister(seed=42)
    
    inside_circle = 0
    for _ in range(n_samples):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside_circle += 1
    
    pi_estimate = 4 * inside_circle / n_samples
    return pi_estimate

# 不同样本量的估算
sample_sizes = [1000, 10000, 100000, 1000000]
for n in sample_sizes:
    pi_est = estimate_pi_monte_carlo(n)
    error = abs(pi_est - np.pi)
    print(f"样本量: {n:7d}, π估计值: {pi_est:.6f}, 误差: {error:.6f}")
```

### 7.2 期权定价蒙特卡罗模拟

```python
def black_scholes_monte_carlo(S0, K, T, r, sigma, n_simulations=100000):
    """
    使用蒙特卡罗方法计算欧式看涨期权价格
    
    参数:
    S0: 当前股价
    K: 执行价格
    T: 到期时间
    r: 无风险利率
    sigma: 波动率
    n_simulations: 模拟次数
    """
    rng = MersenneTwister(seed=123)
    
    option_values = []
    
    for _ in range(n_simulations):
        # 生成标准正态随机数
        z = rng.normal(0, 1)
        
        # 计算到期时的股价
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        
        # 计算期权回报
        payoff = max(ST - K, 0)
        option_values.append(payoff)
    
    # 计算期权价格（折现）
    option_price = np.exp(-r * T) * np.mean(option_values)
    
    return option_price, np.std(option_values)

# 期权参数
S0 = 100    # 当前股价
K = 100     # 执行价格  
T = 1       # 1年到期
r = 0.05    # 5%无风险利率
sigma = 0.2 # 20%波动率

price, std_dev = black_scholes_monte_carlo(S0, K, T, r, sigma)

print(f"\n期权定价蒙特卡罗模拟:")
print(f"当前股价: ${S0}")
print(f"执行价格: ${K}")
print(f"到期时间: {T}年")
print(f"无风险利率: {r*100}%")
print(f"波动率: {sigma*100}%")
print(f"期权价格: ${price:.4f}")
print(f"价格标准差: ${std_dev:.4f}")

# Black-Scholes解析解对比
from scipy.stats import norm

d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
analytical_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

print(f"Black-Scholes解析解: ${analytical_price:.4f}")
print(f"蒙特卡罗误差: ${abs(price - analytical_price):.4f}")
```

## 8. 完整应用示例：数值积分

```python
def numerical_integration_comparison():
    """比较不同数值积分方法"""
    
    # 定义被积函数
    def f(x):
        return np.exp(-x*x) * np.cos(2*x)
    
    # 积分区间
    a, b = 0, 2
    
    # 方法1：蒙特卡罗积分
    def monte_carlo_integration(func, a, b, n_samples=100000):
        rng = MersenneTwister(seed=999)
        samples = [rng.uniform(a, b) for _ in range(n_samples)]
        func_values = [func(x) for x in samples]
        integral = (b - a) * np.mean(func_values)
        return integral
    
    # 方法2：梯形法则
    def trapezoidal_rule(func, a, b, n_intervals=1000):
        h = (b - a) / n_intervals
        x = np.linspace(a, b, n_intervals + 1)
        y = [func(xi) for xi in x]
        integral = h * (0.5*y[0] + sum(y[1:-1]) + 0.5*y[-1])
        return integral
    
    # 方法3：Simpson法则
    def simpson_rule(func, a, b, n_intervals=1000):
        if n_intervals % 2 == 1:
            n_intervals += 1  # 确保是偶数
        
        h = (b - a) / n_intervals
        x = np.linspace(a, b, n_intervals + 1)
        y = [func(xi) for xi in x]
        
        integral = h/3 * (y[0] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-2:2]) + y[-1])
        return integral
    
    # 计算积分
    mc_result = monte_carlo_integration(f, a, b)
    trap_result = trapezoidal_rule(f, a, b)
    simp_result = simpson_rule(f, a, b)
    
    print("数值积分方法比较:")
    print(f"被积函数: exp(-x²)cos(2x)")
    print(f"积分区间: [{a}, {b}]")
    print(f"蒙特卡罗方法: {mc_result:.6f}")
    print(f"梯形法则:     {trap_result:.6f}")
    print(f"Simpson法则:  {simp_result:.6f}")
    
    # 使用scipy作为参考
    from scipy import integrate
    reference, _ = integrate.quad(f, a, b)
    print(f"SciPy参考值:  {reference:.6f}")
    
    print(f"\n误差分析:")
    print(f"蒙特卡罗误差: {abs(mc_result - reference):.6f}")
    print(f"梯形法误差:   {abs(trap_result - reference):.6f}")
    print(f"Simpson误差:  {abs(simp_result - reference):.6f}")

# 运行积分比较
numerical_integration_comparison()
```

## 9. 性能测试示例

```python
import time

def performance_comparison():
    """性能比较测试"""
    
    print("性能比较测试:")
    print("="*50)
    
    # 1. 矩阵运算性能
    print("1. 矩阵行列式计算性能:")
    
    sizes = [50, 100, 200]
    for size in sizes:
        # 生成随机矩阵
        rng = MersenneTwister(seed=42)
        matrix = [[rng.normal(0, 1) for _ in range(size)] for _ in range(size)]
        
        # 测试DWSIM实现
        start_time = time.time()
        det_dwsim = MatrixOperations.determinant(matrix)
        dwsim_time = time.time() - start_time
        
        # 测试NumPy实现
        matrix_np = np.array(matrix)
        start_time = time.time()
        det_numpy = np.linalg.det(matrix_np)
        numpy_time = time.time() - start_time
        
        print(f"  {size}×{size} 矩阵:")
        print(f"    DWSIM:  {dwsim_time:.4f}s (结果: {det_dwsim:.6f})")
        print(f"    NumPy:  {numpy_time:.4f}s (结果: {det_numpy:.6f})")
        print(f"    误差:   {abs(det_dwsim - det_numpy):.2e}")
        print(f"    速度比: {dwsim_time/numpy_time:.2f}x")
    
    # 2. 随机数生成性能
    print("\n2. 随机数生成性能:")
    
    n_samples = 1000000
    
    # 测试DWSIM Mersenne Twister
    rng = MersenneTwister(seed=42)
    start_time = time.time()
    dwsim_samples = [rng.random() for _ in range(n_samples)]
    dwsim_time = time.time() - start_time
    
    # 测试NumPy
    np.random.seed(42)
    start_time = time.time()
    numpy_samples = np.random.random(n_samples)
    numpy_time = time.time() - start_time
    
    print(f"  生成{n_samples}个随机数:")
    print(f"    DWSIM:  {dwsim_time:.4f}s")
    print(f"    NumPy:  {numpy_time:.4f}s")
    print(f"    速度比: {dwsim_time/numpy_time:.2f}x")
    
    # 统计验证
    dwsim_mean = np.mean(dwsim_samples)
    numpy_mean = np.mean(numpy_samples)
    print(f"    DWSIM均值: {dwsim_mean:.6f}")
    print(f"    NumPy均值: {numpy_mean:.6f}")

# 运行性能测试
performance_comparison()
```

这些示例展示了DWSIM数学计算库的主要功能和使用方法。每个示例都包含了详细的注释和说明，帮助用户快速上手和理解各个模块的用途。 