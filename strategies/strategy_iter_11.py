from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedADMMStrategy(BaseTuningStrategy):
    def __init__(self):
        self.min_beta = 1e-4
        self.max_beta = 1e6
        self.mu = 11.0  # 略降低阈值，增加调整频率
        self.tau_inc = 2.0  # 增大增量因子，加速beta增长
        self.tau_dec = 2.0  # 保持减量因子不变
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        current_beta = iteration_state.get('beta', 1.0)
        iteration_count = iteration_state.get('iteration', 0)
        
        # 在早期迭代中避免过于频繁的调整
        if iteration_count < 20 and dual_res < 1e-10:
            return {'beta': current_beta}
            
        if primal_res is None or dual_res is None or dual_res < 1e-15:
            return {'beta': current_beta}
            
        if dual_res > 1e-10:
            ratio = primal_res / dual_res
            # 增加调整敏感度，使beta变化更明显
            if ratio > self.mu:
                new_beta = current_beta * self.tau_inc
            elif ratio < 1.0 / self.mu:
                new_beta = current_beta / self.tau_dec
            else:
                new_beta = current_beta
        else:
            new_beta = current_beta
            
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        return {'beta': float(new_beta)}
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_beta': self.min_beta, 
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        if 'min_beta' in params:
            self.min_beta = params['min_beta']
        if 'max_beta' in params:
            self.max_beta = params['max_beta']
        if 'mu' in params:
            self.mu = params['mu']
        if 'tau_inc' in params:
            self.tau_inc = params['tau_inc']
        if 'tau_dec' in params:
            self.tau_dec = params['tau_dec']