# strategies/initial_strategy.py
"""初始测试策略"""

from strategies.base_strategy import BaseTuningStrategy

class InitialADMMStrategy(BaseTuningStrategy):
    """初始ADMM策略 - 用于测试"""
    
    def __init__(self, initial_beta=1.0):
        self.beta = initial_beta
        self.history = []
    
    def update_parameters(self, iteration_state):
        """简单的自适应调整"""
        return {'beta': self.beta, 'adjustment': 'keep'}
    
    def get_parameters(self):
        return {'beta': self.beta}
    
    def set_parameters(self, params):
        if 'beta' in params:
            self.beta = params['beta']
