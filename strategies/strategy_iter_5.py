from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class OptimizedAdaptiveBetaStrategy(BaseTuningStrategy):
    """
    优化ADMM惩罚参数beta自适应调整策略
    
    针对测试结果的优化：
    1. 降低基础增长因子，避免beta增长过快
    2. 增加残差判断阈值，减少不必要的beta调整
    3. 移除停滞检测机制，简化策略逻辑
    4. 回归问题采用更保守的单调增长
    """
    
    def __init__(self):
        # 基础参数
        self.initial_beta = 1.0
        self.max_beta = 1e4
        self.min_beta = 1e-6
        
        # 自适应调整参数 - 降低增长因子
        self.mu = 8.0  # 降低残差比阈值
        self.tau_inc = 1.25  # 降低增长幅度
        self.tau_dec = 1.25  # 降低减少幅度
        
        # 收敛检测参数
        self.convergence_threshold = 1e-3
        
        # 状态跟踪
        self.last_primal_res = None
        self.last_dual_res = None
        
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
        primal_res = iteration_state.get('primal_residual')
        dual_res = iteration_state.get('dual_residual')
        converged = iteration_state.get('converged', False)
        
        # 如果已经收敛或缺少残差信息，返回当前beta
        if converged or primal_res is None or dual_res is None:
            return {'beta': float(np.clip(current_beta, self.min_beta, self.max_beta))}
        
        # 保存当前残差用于下次比较
        self.last_primal_res = primal_res
        self.last_dual_res = dual_res
        
        # 计算新beta值 - 使用更保守的自适应策略
        new_beta = self._calculate_new_beta(current_beta, primal_res, dual_res, iteration_state)
        
        # 限制beta在有效范围内
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        return {'beta': float(new_beta)}
    
    def _calculate_new_beta(self, current_beta: float, primal_res: float, 
                           dual_res: float, iteration_state: Dict[str, Any]) -> float:
        """
        计算新的beta值
        
        Args:
            current_beta: 当前beta值
            primal_res: 原始残差
            dual_res: 对偶残差
            iteration_state: 迭代状态
            
        Returns:
            float: 新的beta值
        """
        iteration = iteration_state.get('iteration', 0)
        
        # 早期迭代：使用较快的增长策略
        if iteration < 20:
            # 前20次迭代采用较快但可控的增长
            growth_factor = 1.2 if iteration < 5 else 1.1
            return current_beta * growth_factor
        
        # 判断是否为回归问题（根据残差模式）
        is_regression = self._is_regression_problem(iteration_state)
        
        if is_regression:
            # 回归问题：beta只增不减，采用单调上升策略
            # 基于原始残差大小调整增长幅度
            if primal_res > 1.0:
                growth_factor = 1.3  # 残差大，增长快
            elif primal_res > 0.1:
                growth_factor = 1.15  # 残差中等，中等增长
            elif primal_res > 0.01:
                growth_factor = 1.05  # 残差小，缓慢增长
            else:
                growth_factor = 1.02  # 残差很小，非常缓慢增长
            
            return min(current_beta * growth_factor, self.max_beta)
        else:
            # 非回归问题：使用经典自适应策略
            # 避免除零错误
            if dual_res < 1e-10:
                return current_beta
            
            # 计算残差比例
            residual_ratio = primal_res / dual_res
            
            if residual_ratio > self.mu:
                # 原始残差远大于对偶残差，增加beta
                new_beta = current_beta * self.tau_inc
            elif residual_ratio < 1.0 / self.mu:
                # 对偶残差远大于原始残差，减少beta
                new_beta = current_beta / self.tau_dec
            else:
                # 残差平衡，保持beta不变
                new_beta = current_beta
            
            return new_beta
    
    def _is_regression_problem(self, iteration_state: Dict[str, Any]) -> bool:
        """
        判断是否为回归问题（l1_regression或elastic_net_regression）
        
        Args:
            iteration_state: 迭代状态
            
        Returns:
            bool: 是否为回归问题
        """
        primal_res = iteration_state.get('primal_residual')
        dual_res = iteration_state.get('dual_residual')
        iteration = iteration_state.get('iteration', 0)
        
        if primal_res is None or dual_res is None:
            return False
        
        # 回归问题特征：在早期迭代中，对偶残差增长迅速
        # 并且原始残差相对较小但对偶残差较大
        if iteration > 50 and dual_res > 10.0 and primal_res < 1.0:
            # 检查残差比例：对偶残差远大于原始残差
            if dual_res > 10 * primal_res:
                return True
        
        # 如果残差都很大但比例接近，也可能是回归问题
        if primal_res > 0.5 and dual_res > 0.5:
            ratio = primal_res / dual_res
            if 0.5 < ratio < 2.0:
                return True
                
        return False
    
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
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'convergence_threshold': self.convergence_threshold
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置策略参数
        
        Args:
            params: 参数字典
        """
        if 'initial_beta' in params:
            self.initial_beta = params['initial_beta']
        if 'max_beta' in params:
            self.max_beta = params['max_beta']
        if 'min_beta' in params:
            self.min_beta = params['min_beta']
        if 'mu' in params:
            self.mu = params['mu']
        if 'tau_inc' in params:
            self.tau_inc = params['tau_inc']
        if 'tau_dec' in params:
            self.tau_dec = params['tau_dec']
        if 'convergence_threshold' in params:
            self.convergence_threshold = params['convergence_threshold']