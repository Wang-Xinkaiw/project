# strategies/baseline_strategy.py
"""
基线ADMM策略 - 与112(1).py一致的简单策略
这是保证能收敛的基准策略，DeepSeek生成的策略需要比它更好
"""

from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any


class BaselineADMMStrategy(BaseTuningStrategy):
    """
    基线ADMM策略 - 简单的固定倍增策略
    
    这与112(1).py中的标准ADMM策略完全一致：
    - 初始 rho = 0.1
    - 每次迭代 rho = min(rho * 1.1, max_rho)
    
    这是保证收敛的最简单策略，作为其他策略的比较基准。
    """
    
    def __init__(self, 
                 initial_beta: float = 0.1,  # 与112(1).py一致
                 growth_factor: float = 1.1,  # 每次增长10%
                 max_beta: float = 1e4):
        """
        初始化基线策略
        
        参数:
            initial_beta: 初始惩罚参数（默认0.1，与112(1).py一致）
            growth_factor: 每次迭代的增长因子（默认1.1）
            max_beta: 最大惩罚参数
        """
        self.beta = initial_beta
        self.initial_beta = initial_beta
        self.growth_factor = growth_factor
        self.max_beta = max_beta
    
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新惩罚参数 - 简单的固定倍增
        
        与112(1).py完全一致：每次迭代无条件增大beta
        """
        # 无条件增大beta（与112(1).py的标准ADMM一致）
        self.beta = min(self.beta * self.growth_factor, self.max_beta)
        
        return {'beta': self.beta}
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取当前策略参数"""
        return {
            'beta': self.beta,
            'initial_beta': self.initial_beta,
            'growth_factor': self.growth_factor,
            'max_beta': self.max_beta
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置策略参数"""
        if 'beta' in params:
            self.beta = params['beta']
        if 'initial_beta' in params:
            self.initial_beta = params['initial_beta']
        if 'growth_factor' in params:
            self.growth_factor = params['growth_factor']
        if 'max_beta' in params:
            self.max_beta = params['max_beta']
    
    def reset(self) -> None:
        """重置策略状态"""
        self.beta = self.initial_beta
