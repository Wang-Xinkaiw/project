from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedADMMStrategy(BaseTuningStrategy):
    def __init__(self):
        self.min_beta = 1e-3
        self.max_beta = 1e6
        self.mu = 9.5  # 保持不变
        self.tau_inc = 1.7
        self.tau_dec = 1.9
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        current_beta = iteration_state.get('beta', 1.0)
        
        if primal_res is None or dual_res is None or dual_res < 1e-15:
            return {'beta': current_beta}
            
        if dual_res > 1e-10:
            ratio = primal_res / dual_res
            
            # 降低触发调整的阈值，使策略更保守
            if ratio > self.mu * 0.8:  # 从0.9降低到0.8，减少调整频率
                new_beta = current_beta * self.tau_inc
            elif ratio < 1.0 / (self.mu * 0.8):  # 对应调整阈值也降低
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