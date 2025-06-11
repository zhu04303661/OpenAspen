"""
Mersenne Twister随机数生成器
============================

实现MT19937 Mersenne Twister伪随机数生成器。
这些功能是从DWSIM.Math.RandomOps/MersenneTwister.cs转换而来的。

Mersenne Twister特点:
- 周期长度: 2^19937 - 1
- 分布均匀性好
- 通过多种统计检验
- 计算效率高

作者: DWSIM团队 (Python转换版本)
许可证: GNU General Public License v3.0

原始算法版权:
Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura
"""

import numpy as np
from typing import Union, List, Optional
import time


class MersenneTwister:
    """
    Mersenne Twister MT19937随机数生成器
    
    这是一个高质量的伪随机数生成器，具有非常长的周期和良好的
    分布特性。适用于蒙特卡罗模拟、统计分析等应用。
    """
    
    # MT19937常数
    N = 624
    M = 397
    MATRIX_A = 0x9908b0df
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7fffffff
    
    def __init__(self, seed: Optional[Union[int, List[int]]] = None):
        """
        初始化Mersenne Twister生成器
        
        参数:
            seed: 种子值，可以是单个整数或整数列表
                 如果为None，使用当前时间作为种子
        """
        # 状态数组
        self.mt = np.zeros(self.N, dtype=np.uint32)
        self.mti = self.N + 1  # 状态索引
        
        # 初始化种子
        if seed is None:
            self.seed_with_time()
        elif isinstance(seed, (list, np.ndarray)):
            self.seed_with_array(seed)
        else:
            self.seed_with_value(int(seed))
    
    def seed_with_time(self):
        """使用当前时间作为种子"""
        current_time = int(time.time() * 1000000) % (2**32)
        self.seed_with_value(current_time)
    
    def seed_with_value(self, seed: int):
        """
        使用单个整数值初始化种子
        
        参数:
            seed: 32位无符号整数种子
        """
        seed = int(seed) % (2**32)  # 确保是32位
        self.mt[0] = np.uint32(seed)
        
        # 使用线性同余生成器初始化状态数组
        for i in range(1, self.N):
            self.mt[i] = np.uint32(
                1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i
            )
        
        self.mti = self.N
    
    def seed_with_array(self, seeds: Union[List[int], np.ndarray]):
        """
        使用整数数组初始化种子
        
        参数:
            seeds: 整数数组，用于初始化状态
        """
        seeds = [int(s) % (2**32) for s in seeds]
        
        # 首先用默认种子初始化
        self.seed_with_value(19650218)
        
        i = 1
        j = 0
        k = max(self.N, len(seeds))
        
        # 第一阶段：混合种子数组
        for _ in range(k):
            self.mt[i] = np.uint32(
                (self.mt[i] ^ ((self.mt[i-1] ^ (self.mt[i-1] >> 30)) * 1664525)) + 
                seeds[j] + j
            )
            
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            
            j += 1
            if j >= len(seeds):
                j = 0
        
        # 第二阶段：确保非零状态
        for _ in range(self.N - 1):
            self.mt[i] = np.uint32(
                (self.mt[i] ^ ((self.mt[i-1] ^ (self.mt[i-1] >> 30)) * 1566083941)) - i
            )
            
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
        
        self.mt[0] = np.uint32(0x80000000)  # 确保最高位为1
        self.mti = self.N
    
    def rand_uint32(self) -> int:
        """
        生成32位无符号随机整数
        
        返回:
            int: 范围在[0, 2^32-1]的随机整数
        """
        # 如果需要，生成新的N个值
        if self.mti >= self.N:
            self._generate_numbers()
        
        # 从状态数组中取值
        y = self.mt[self.mti]
        self.mti += 1
        
        # 调质变换（Tempering）
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)
        
        return int(y)
    
    def _generate_numbers(self):
        """生成下一批N个随机数"""
        mag01 = [0, self.MATRIX_A]
        
        # 第一部分：0 <= i < N-M
        for i in range(self.N - self.M):
            y = (self.mt[i] & self.UPPER_MASK) | (self.mt[i + 1] & self.LOWER_MASK)
            self.mt[i] = self.mt[i + self.M] ^ (y >> 1) ^ mag01[y & 1]
        
        # 第二部分：N-M <= i < N-1
        for i in range(self.N - self.M, self.N - 1):
            y = (self.mt[i] & self.UPPER_MASK) | (self.mt[i + 1] & self.LOWER_MASK)
            self.mt[i] = self.mt[i + self.M - self.N] ^ (y >> 1) ^ mag01[y & 1]
        
        # 最后一个元素
        y = (self.mt[self.N - 1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
        self.mt[self.N - 1] = self.mt[self.M - 1] ^ (y >> 1) ^ mag01[y & 1]
        
        self.mti = 0
    
    def random(self) -> float:
        """
        生成[0, 1)范围内的随机浮点数
        
        返回:
            float: [0, 1)范围内的随机数
        """
        # 使用与C#版本相同的转换公式
        # 原始C#: (rand + 1) * (1.0 / (RandMax + 2))
        # 对于32位: (rand + 1) / (2^32 + 1)
        rand = self.rand_uint32()
        return (rand + 1.0) / (2**32 + 1.0)
    
    def uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        """
        生成[a, b)范围内的均匀分布随机数
        
        参数:
            a: 下界
            b: 上界
            
        返回:
            float: [a, b)范围内的随机数
        """
        return a + (b - a) * self.random()
    
    def randint(self, low: int, high: int) -> int:
        """
        生成[low, high]范围内的随机整数（包含两端）
        
        参数:
            low: 下界（包含）
            high: 上界（包含）
            
        返回:
            int: [low, high]范围内的随机整数
        """
        if low > high:
            raise ValueError("low不能大于high")
        
        range_size = high - low + 1  # +1因为包含两端
        # 使用rejection sampling避免偏差
        max_valid = (2**32 // range_size) * range_size
        
        while True:
            value = self.rand_uint32()
            if value < max_valid:
                return low + (value % range_size)
    
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        生成正态分布随机数
        
        使用Box-Muller变换生成正态分布随机数
        
        参数:
            mu: 均值
            sigma: 标准差
            
        返回:
            float: 正态分布随机数
        """
        if not hasattr(self, '_has_spare'):
            self._has_spare = False
        
        if self._has_spare:
            self._has_spare = False
            return self._spare * sigma + mu
        
        self._has_spare = True
        
        # Box-Muller变换
        u1 = self.random()
        u2 = self.random()
        
        # 避免log(0)
        while u1 == 0:
            u1 = self.random()
        
        mag = sigma * np.sqrt(-2.0 * np.log(u1))
        
        self._spare = mag * np.cos(2.0 * np.pi * u2)
        return mag * np.sin(2.0 * np.pi * u2) + mu
    
    def exponential(self, lambd: float = 1.0) -> float:
        """
        生成指数分布随机数
        
        参数:
            lambd: 率参数（lambda）
            
        返回:
            float: 指数分布随机数
        """
        u = self.random()
        while u == 0:  # 避免log(0)
            u = self.random()
        
        return -np.log(u) / lambd
    
    def choice(self, a: Union[List, np.ndarray], size: Optional[int] = None) -> Union[any, List]:
        """
        从数组中随机选择元素
        
        参数:
            a: 输入数组
            size: 选择的数量，如果为None返回单个元素
            
        返回:
            选择的元素或元素列表
        """
        a = list(a) if not isinstance(a, list) else a
        
        if size is None:
            return a[self.randint(0, len(a) - 1)]
        else:
            return [a[self.randint(0, len(a) - 1)] for _ in range(size)]
    
    def shuffle(self, a: List) -> None:
        """
        就地随机打乱数组
        
        使用Fisher-Yates洗牌算法
        
        参数:
            a: 要打乱的列表
        """
        for i in range(len(a) - 1, 0, -1):
            j = self.randint(0, i)
            a[i], a[j] = a[j], a[i]
    
    def sample(self, population: List, k: int) -> List:
        """
        从总体中无重复地随机抽样
        
        参数:
            population: 总体列表
            k: 抽样数量
            
        返回:
            List: 抽样结果
        """
        if k > len(population):
            raise ValueError("抽样数量不能超过总体大小")
        
        population_copy = population.copy()
        result = []
        
        for _ in range(k):
            index = self.randint(0, len(population_copy) - 1)
            result.append(population_copy.pop(index))
        
        return result
    
    def get_state(self) -> tuple:
        """
        获取生成器的当前状态
        
        返回:
            tuple: (mt数组, mti索引)
        """
        return (self.mt.copy(), self.mti)
    
    def set_state(self, state: tuple):
        """
        设置生成器状态
        
        参数:
            state: 由get_state()返回的状态元组
        """
        self.mt, self.mti = state
        self.mt = np.array(self.mt, dtype=np.uint32)
    
    @property
    def name(self) -> str:
        """生成器名称"""
        return "Mersenne Twister MT19937"
    
    @property
    def period(self) -> int:
        """生成器周期"""
        return 2**19937 - 1


# 便捷函数和全局实例
_default_rng = MersenneTwister()

def seed(seed_value: Optional[Union[int, List[int]]] = None):
    """设置默认随机数生成器的种子"""
    global _default_rng
    _default_rng = MersenneTwister(seed_value)

def random() -> float:
    """生成[0, 1)范围内的随机浮点数"""
    return _default_rng.random()

def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """生成[a, b)范围内的均匀分布随机数"""
    return _default_rng.uniform(a, b)

def randint(low: int, high: int) -> int:
    """生成[low, high]范围内的随机整数（包含两端）"""
    return _default_rng.randint(low, high)

def normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    """生成正态分布随机数"""
    return _default_rng.normal(mu, sigma)

def choice(a: Union[List, np.ndarray], size: Optional[int] = None):
    """从数组中随机选择元素"""
    return _default_rng.choice(a, size)

def shuffle(a: List) -> None:
    """就地随机打乱数组"""
    _default_rng.shuffle(a) 