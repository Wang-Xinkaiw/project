from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedADMMStrategy(BaseTuningStrategy):
    def __init__(self):
        self.min_beta = 1e-6
        self.max_beta = 1e6
        self.mu = 15.0  # 增大mu阈值，使调整更保守
        self.tau_inc = 1.5  # 减小增量因子，更保守地增加beta
        self.tau_dec = 2.5  # 增大减量因子，更保守地减少beta
        self.min_iter = 10  # 最小迭代次数，用于稳定调整
        self.smoothing_factor = 0.3  # 增大平滑因子，减少调整波动
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取所有需要的信息
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        current_beta = iteration_state.get('beta', 1.0)
        iteration = iteration_state.get('iteration', 0)
        
        # 计算新beta
        if primal_res is None or dual_res is None:
            return {'beta': current_beta}
            
        # 在早期迭代中使用平滑调整
        if iteration < self.min_iter:
            # 使用平滑因子避免大幅调整
            adjusted_ratio = max(primal_res / dual_res, 0.5)  # 限制最小比值
        else:
            # 正常调整
            adjusted_ratio = primal_res / dual_res
            
        # 使用平滑因子避免大幅调整
        if dual_res > 1e-10:
            if adjusted_ratio > self.mu:
                new_beta = current_beta * self.tau_inc
            elif adjusted_ratio < 1.0 / self.mu:
                new_beta = current_beta / self.tau_dec
            else:
                new_beta = current_beta * (1.0 + self.smoothing_factor/5)
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
            'tau_dec': self.tau_dec,
            'min_iter': self.min_iter,
            'smoothing_factor': self.smoothing_factor
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
        if 'min_iter' in params:
            self.min_iter = params['min_iter']
        if 'smoothing_factor' in params:
            self.smoothing_factor = params['smoothing_factor']