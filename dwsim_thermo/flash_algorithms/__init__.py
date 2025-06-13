"""
DWSIM热力学计算库 - 闪蒸算法模块
===============================

提供各种闪蒸算法的实现，包括：
- 嵌套循环算法 (Nested Loops)
- Inside-Out算法
- Gibbs自由能最小化算法
- 算法工厂和管理器

作者：OpenAspen项目组
版本：2.0.0
"""

from .base_flash import FlashAlgorithmBase, FlashCalculationResult
from .nested_loops import NestedLoopsFlash
from .inside_out import InsideOutFlash
from .gibbs_minimization import GibbsMinimizationFlash, GibbsMinimizationSettings

from typing import Dict, Type, Optional, List
import logging

# 算法注册表
FLASH_ALGORITHMS: Dict[str, Type[FlashAlgorithmBase]] = {
    "nested_loops": NestedLoopsFlash,
    "inside_out": InsideOutFlash,
    "gibbs_minimization": GibbsMinimizationFlash,
}

# 算法别名
ALGORITHM_ALIASES = {
    "nl": "nested_loops",
    "io": "inside_out",
    "gibbs": "gibbs_minimization",
    "gm": "gibbs_minimization",
    "nested": "nested_loops",
    "inside": "inside_out",
    "minimization": "gibbs_minimization"
}

class FlashAlgorithmFactory:
    """闪蒸算法工厂类
    
    提供统一的算法创建和管理接口。
    支持算法注册、创建、配置和性能监控。
    """
    
    def __init__(self):
        self.logger = logging.getLogger("FlashAlgorithmFactory")
        self._algorithm_cache: Dict[str, FlashAlgorithmBase] = {}
        self._performance_stats: Dict[str, Dict] = {}
    
    @classmethod
    def create_algorithm(
        cls,
        algorithm_name: str,
        **kwargs
    ) -> FlashAlgorithmBase:
        """创建闪蒸算法实例
        
        Args:
            algorithm_name: 算法名称或别名
            **kwargs: 算法特定的参数
            
        Returns:
            FlashAlgorithmBase: 算法实例
            
        Raises:
            ValueError: 不支持的算法名称
        """
        # 解析算法名称
        name = cls._resolve_algorithm_name(algorithm_name)
        
        if name not in FLASH_ALGORITHMS:
            available = list(FLASH_ALGORITHMS.keys()) + list(ALGORITHM_ALIASES.keys())
            raise ValueError(f"不支持的算法: {algorithm_name}. 可用算法: {available}")
        
        algorithm_class = FLASH_ALGORITHMS[name]
        
        # 创建算法实例
        try:
            if name == "gibbs_minimization":
                # Gibbs最小化算法需要特殊设置
                settings = kwargs.pop('settings', None)
                if settings is None and kwargs:
                    # 从kwargs创建设置
                    settings = GibbsMinimizationSettings(**kwargs)
                algorithm = algorithm_class(settings)
            else:
                # 其他算法
                algorithm = algorithm_class(**kwargs)
            
            logging.getLogger("FlashAlgorithmFactory").info(
                f"成功创建算法: {algorithm.name}"
            )
            
            return algorithm
            
        except Exception as e:
            raise RuntimeError(f"创建算法{algorithm_name}失败: {e}")
    
    @classmethod
    def _resolve_algorithm_name(cls, name: str) -> str:
        """解析算法名称（处理别名）"""
        name_lower = name.lower().strip()
        return ALGORITHM_ALIASES.get(name_lower, name_lower)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """获取可用算法列表"""
        return list(FLASH_ALGORITHMS.keys())
    
    @classmethod
    def get_algorithm_aliases(cls) -> Dict[str, str]:
        """获取算法别名映射"""
        return ALGORITHM_ALIASES.copy()
    
    @classmethod
    def register_algorithm(
        cls,
        name: str,
        algorithm_class: Type[FlashAlgorithmBase],
        aliases: Optional[List[str]] = None
    ):
        """注册新的闪蒸算法
        
        Args:
            name: 算法名称
            algorithm_class: 算法类
            aliases: 算法别名列表
        """
        if not issubclass(algorithm_class, FlashAlgorithmBase):
            raise ValueError("算法类必须继承自FlashAlgorithmBase")
        
        FLASH_ALGORITHMS[name] = algorithm_class
        
        if aliases:
            for alias in aliases:
                ALGORITHM_ALIASES[alias.lower()] = name
        
        logging.getLogger("FlashAlgorithmFactory").info(
            f"注册算法: {name}, 别名: {aliases or []}"
        )

class FlashAlgorithmManager:
    """闪蒸算法管理器
    
    提供算法选择、性能监控和自动优化功能。
    """
    
    def __init__(self):
        self.logger = logging.getLogger("FlashAlgorithmManager")
        self._algorithms: Dict[str, FlashAlgorithmBase] = {}
        self._performance_history: Dict[str, List[Dict]] = {}
        self._default_algorithm = "nested_loops"
        self._auto_selection_enabled = False
    
    def add_algorithm(
        self,
        name: str,
        algorithm: FlashAlgorithmBase
    ):
        """添加算法到管理器"""
        self._algorithms[name] = algorithm
        if name not in self._performance_history:
            self._performance_history[name] = []
        
        self.logger.info(f"添加算法到管理器: {name}")
    
    def get_algorithm(self, name: str) -> FlashAlgorithmBase:
        """获取算法实例"""
        if name not in self._algorithms:
            # 自动创建算法
            algorithm = FlashAlgorithmFactory.create_algorithm(name)
            self.add_algorithm(name, algorithm)
        
        return self._algorithms[name]
    
    def set_default_algorithm(self, name: str):
        """设置默认算法"""
        if name not in FLASH_ALGORITHMS and name not in ALGORITHM_ALIASES:
            raise ValueError(f"未知算法: {name}")
        
        self._default_algorithm = FlashAlgorithmFactory._resolve_algorithm_name(name)
        self.logger.info(f"设置默认算法: {self._default_algorithm}")
    
    def get_default_algorithm(self) -> FlashAlgorithmBase:
        """获取默认算法"""
        return self.get_algorithm(self._default_algorithm)
    
    def enable_auto_selection(self, enabled: bool = True):
        """启用/禁用自动算法选择"""
        self._auto_selection_enabled = enabled
        self.logger.info(f"自动算法选择: {'启用' if enabled else '禁用'}")
    
    def select_best_algorithm(
        self,
        problem_characteristics: Dict
    ) -> str:
        """根据问题特征选择最佳算法
        
        Args:
            problem_characteristics: 问题特征字典
                - n_components: 组分数
                - temperature: 温度
                - pressure: 压力
                - phase_behavior: 相行为类型
                - accuracy_requirement: 精度要求
                
        Returns:
            str: 推荐的算法名称
        """
        if not self._auto_selection_enabled:
            return self._default_algorithm
        
        n_comp = problem_characteristics.get('n_components', 2)
        temp = problem_characteristics.get('temperature', 298.15)
        pressure = problem_characteristics.get('pressure', 101325.0)
        phase_behavior = problem_characteristics.get('phase_behavior', 'simple')
        accuracy = problem_characteristics.get('accuracy_requirement', 'normal')
        
        # 算法选择逻辑
        if phase_behavior == 'complex' or accuracy == 'high':
            # 复杂相行为或高精度要求：使用Gibbs最小化
            return "gibbs_minimization"
        elif n_comp <= 3 and phase_behavior == 'simple':
            # 简单系统：使用嵌套循环
            return "nested_loops"
        elif n_comp > 3:
            # 多组分系统：使用Inside-Out
            return "inside_out"
        else:
            # 默认情况
            return self._default_algorithm
    
    def record_performance(
        self,
        algorithm_name: str,
        performance_data: Dict
    ):
        """记录算法性能数据"""
        if algorithm_name not in self._performance_history:
            self._performance_history[algorithm_name] = []
        
        self._performance_history[algorithm_name].append(performance_data)
        
        # 保持历史记录在合理范围内
        if len(self._performance_history[algorithm_name]) > 1000:
            self._performance_history[algorithm_name] = \
                self._performance_history[algorithm_name][-500:]
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """获取性能摘要"""
        summary = {}
        
        for algo_name, history in self._performance_history.items():
            if not history:
                continue
            
            # 计算统计信息
            cpu_times = [h.get('cpu_time', 0) for h in history]
            iterations = [h.get('iterations', 0) for h in history]
            success_rate = sum(1 for h in history if h.get('converged', False)) / len(history)
            
            summary[algo_name] = {
                'total_calls': len(history),
                'success_rate': success_rate,
                'avg_cpu_time': sum(cpu_times) / len(cpu_times) if cpu_times else 0,
                'avg_iterations': sum(iterations) / len(iterations) if iterations else 0,
                'min_cpu_time': min(cpu_times) if cpu_times else 0,
                'max_cpu_time': max(cpu_times) if cpu_times else 0
            }
        
        return summary
    
    def get_algorithm_recommendation(
        self,
        problem_characteristics: Dict
    ) -> Dict[str, str]:
        """获取算法推荐和理由
        
        Returns:
            Dict包含推荐算法和推荐理由
        """
        recommended = self.select_best_algorithm(problem_characteristics)
        
        # 生成推荐理由
        n_comp = problem_characteristics.get('n_components', 2)
        phase_behavior = problem_characteristics.get('phase_behavior', 'simple')
        accuracy = problem_characteristics.get('accuracy_requirement', 'normal')
        
        reasons = []
        
        if recommended == "gibbs_minimization":
            if phase_behavior == 'complex':
                reasons.append("复杂相行为需要严格的热力学方法")
            if accuracy == 'high':
                reasons.append("高精度要求适合Gibbs最小化方法")
            reasons.append("全局收敛保证和相稳定性分析")
        
        elif recommended == "inside_out":
            if n_comp > 3:
                reasons.append("多组分系统适合Inside-Out算法")
            reasons.append("良好的收敛性和计算效率")
        
        elif recommended == "nested_loops":
            if n_comp <= 3:
                reasons.append("简单系统使用嵌套循环算法效率高")
            if phase_behavior == 'simple':
                reasons.append("简单相行为适合传统方法")
        
        return {
            'algorithm': recommended,
            'reasons': reasons,
            'confidence': 'high' if len(reasons) >= 2 else 'medium'
        }

# 创建全局管理器实例
flash_manager = FlashAlgorithmManager()

# 便捷函数
def create_flash_algorithm(name: str, **kwargs) -> FlashAlgorithmBase:
    """创建闪蒸算法的便捷函数"""
    return FlashAlgorithmFactory.create_algorithm(name, **kwargs)

def get_available_algorithms() -> List[str]:
    """获取可用算法列表的便捷函数"""
    return FlashAlgorithmFactory.get_available_algorithms()

def register_algorithm(
    name: str,
    algorithm_class: Type[FlashAlgorithmBase],
    aliases: Optional[List[str]] = None
):
    """注册算法的便捷函数"""
    FlashAlgorithmFactory.register_algorithm(name, algorithm_class, aliases)

# 导出的公共接口
__all__ = [
    # 基类和结果类
    "FlashAlgorithmBase",
    "FlashCalculationResult",
    
    # 具体算法
    "NestedLoopsFlash",
    "InsideOutFlash", 
    "GibbsMinimizationFlash",
    "GibbsMinimizationSettings",
    
    # 工厂和管理器
    "FlashAlgorithmFactory",
    "FlashAlgorithmManager",
    "flash_manager",
    
    # 便捷函数
    "create_flash_algorithm",
    "get_available_algorithms",
    "register_algorithm",
    
    # 算法注册表
    "FLASH_ALGORITHMS",
    "ALGORITHM_ALIASES"
] 