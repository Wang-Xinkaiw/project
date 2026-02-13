from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class AdaptiveBetaStrategy(BaseTuningStrategy):
    """
    ADMM惩罚参数beta的自适应调整策略
    
    针对l1_regression和elastic_net_regression问题的改进策略：
    1. beta单调递增，初始beta=1.0，每次迭代按1.1-1.5倍增大，上限1e4
    2. 根据残差情况动态调整增长幅度
    3. 当接近收敛时减缓beta增长
    """
    
    def __init__(self):
        # 基础参数
        self.initial_beta = 1.0
        self.max_beta = 1e4
        self.min_beta = 1e-6
        
        # 增长参数
        self.base_growth_factor = 1.2  # 基础增长因子
        self.max_growth_factor = 1.5   # 最大增长因子
        self.min_growth_factor = 1.1   # 最小增长因子
        
        # 收敛检测参数
        self.convergence_threshold = 1e-3
        self.residual_ratio_threshold = 10.0  # 残差比阈值
        
        # 历史记录
        self.primal_history = []
        self.dual_history = []
        self.history_window = 5
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新ADMM参数，主要调整惩罚参数beta
        
        Args:
            iteration_state: 包含迭代信息的字典，包括：
                - iteration: 当前迭代次数
                - primal_residual: 原始残差
                - dual_residual: 对偶残差
                - beta: 当前beta值
                - objective: 目标函数值
                - converged: 是否收敛
                
        Returns:
            Dict[str, Any]: 包含更新后的参数，只返回{'beta': new_beta_value}
        """
        # 从iteration_state获取所有需要的信息
        current_beta = iteration_state.get('beta', self.initial_beta)
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        iteration = iteration_state.get('iteration', 0)
        converged = iteration_state.get('converged', False)
        
        # 保存历史记录用于趋势分析
        if primal_res is not None and dual_res is not None:
            self.primal_history.append(primal_res)
            self.dual_history.append(dual_res)
            if len(self.primal_history) > self.history_window:
                self.primal_history.pop(0)
                self.dual_history.pop(0)
        
        # 如果已经收敛或达到最大beta，直接返回当前beta
        if converged or current_beta >= self.max_beta:
            return {'beta': min(current_beta, self.max_beta)}
        
        # 计算自适应增长因子
        growth_factor = self._calculate_growth_factor(primal_res, dual_res, iteration_state)
        
        # 计算新beta值（只增不减）
        new_beta = current_beta * growth_factor
        
        # 限制beta在有效范围内
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        return {'beta': float(new_beta)}
    
    def _calculate_growth_factor(self, primal_res: float, dual_res: float, 
                                iteration_state: Dict[str, Any]) -> float:
        """
        计算自适应增长因子
        
        Args:
            primal_res: 原始残差
            dual_res: 对偶残差
            iteration_state: 迭代状态
            
        Returns:
            float: 增长因子
        """
        # 基础增长因子
        growth_factor = self.base_growth_factor
        
        # 如果残差信息可用，根据残差情况调整增长因子
        if primal_res is not None and dual_res is not None:
            # 避免除零错误
            if dual_res > 1e-10:
                residual_ratio = primal_res / dual_res
                
                # 如果原始残差远大于对偶残差，需要更快地增加beta
                if residual_ratio > self.residual_ratio_threshold:
                    growth_factor = min(self.max_growth_factor, growth_factor * 1.2)
                
                # 如果对偶残差远大于原始残差，稍微减慢beta增长
                elif residual_ratio < 1.0 / self.residual_ratio_threshold:
                    growth_factor = max(self.min_growth_factor, growth_factor * 0.8)
            
            # 检查残差下降趋势
            if len(self.primal_history) >= 2:
                recent_primal = self.primal_history[-1]
                previous_primal = self.primal_history[0]
                
                # 如果残差快速下降，减缓beta增长
                if previous_primal > 1e-10 and recent_primal / previous_primal < 0.5:
                    growth_factor = max(self.min_growth_factor, growth_factor * 0.9)
        
        # 确保增长因子在合理范围内
        growth_factor = np.clip(growth_factor, self.min_growth_factor, self.max_growth_factor)
        
        return growth_factor
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取策略参数
        
        Returns:
            Dict[str, Any]: 参数字典
        """
        return {
            'initial_beta': self.initial_beta,
            'max_beta': self.max_beta,
            'min_beta': self.min_beta,
            'base_growth_factor': self.base_growth_factor,
            'max_growth_factor': self.max_growth_factor,
            'min_growth_factor': self.min_growth_factor,
            'convergence_threshold': self.convergence_threshold,
            'residual_ratio_threshold': self.residual_ratio_threshold,
            'history_window': self.history_window
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置策略参数
        
        Args:
            params: 参数字典
        """
        valid_params = {
            'initial_beta', 'max_beta', 'min_beta', 
            'base_growth_factor', 'max_growth_factor', 'min_growth_factor',
            'convergence_threshold', 'residual_ratio_threshold', 'history_window'
        }
        
        for key, value in params.items():
            if key in valid_params and hasattr(self, key):
                setattr(self, key, value)