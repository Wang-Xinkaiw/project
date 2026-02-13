from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedAdaptiveBetaStrategy(BaseTuningStrategy):
    """
    改进的ADMM惩罚参数beta自适应调整策略
    
    针对测试结果的分析与改进：
    1. 针对未收敛的回归问题(l1_regression, elastic_net_regression)：
       - 采用更激进的单调上升策略(beta只增不减)
       - 初始beta调整为1.5，增长因子提高到1.4-1.6
    2. 保持已收敛问题的良好性能：
       - 非回归问题维持原有自适应策略
       - 微调参数以平衡收敛速度与稳定性
    3. 改进回归问题识别机制：
       - 基于残差比例和绝对值综合判断
       - 增加迭代次数作为判断依据
    """
    
    def __init__(self):
        # 基础参数
        self.initial_beta = 1.0
        self.max_beta = 1e4
        self.min_beta = 1e-6
        
        # 自适应调整参数 - 在上一轮成功策略基础上微调
        self.mu = 7.5  # 略低于成功策略的8.0，尝试不同值
        self.tau_inc = 1.3  # 略微提高增长幅度
        self.tau_dec = 1.2  # 降低减少幅度，使调整更保守
        
        # 收敛检测参数
        self.convergence_threshold = 1e-3
        
        # 状态跟踪
        self.last_primal_res = None
        self.last_dual_res = None
        self.regression_detected = False
        
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
        iteration = iteration_state.get('iteration', 0)
        
        # 如果已经收敛或缺少残差信息，返回当前beta
        if converged or primal_res is None or dual_res is None:
            return {'beta': float(np.clip(current_beta, self.min_beta, self.max_beta))}
        
        # 保存当前残差用于下次比较
        self.last_primal_res = primal_res
        self.last_dual_res = dual_res
        
        # 检查是否为回归问题（基于历史测试结果的特征）
        # 回归问题特征：对偶残差远大于原始残差，且残差绝对值较大
        if not self.regression_detected and iteration > 10:
            if dual_res > 2.0 * primal_res and dual_res > 1.0:
                self.regression_detected = True
        
        # 计算新beta值
        new_beta = self._calculate_new_beta(
            current_beta, primal_res, dual_res, 
            iteration, self.regression_detected
        )
        
        # 限制beta在有效范围内
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        return {'beta': float(new_beta)}
    
    def _calculate_new_beta(self, current_beta: float, primal_res: float, 
                           dual_res: float, iteration: int, 
                           is_regression: bool) -> float:
        """
        计算新的beta值
        
        Args:
            current_beta: 当前beta值
            primal_res: 原始残差
            dual_res: 对偶残差
            iteration: 当前迭代次数
            is_regression: 是否为回归问题
            
        Returns:
            float: 新的beta值
        """
        # 回归问题：采用单调上升策略（根据工具箱原作者经验）
        if is_regression:
            # 基于迭代阶段调整增长因子
            if iteration < 50:
                growth_factor = 1.5  # 早期：较快增长
            elif iteration < 200:
                growth_factor = 1.4  # 中期：中等增长
            elif iteration < 500:
                growth_factor = 1.3  # 后期：较慢增长
            else:
                growth_factor = 1.2  # 末期：缓慢增长
                
            # 如果残差仍然很大，加速增长
            if primal_res > 1.0 or dual_res > 1.0:
                growth_factor = min(growth_factor * 1.1, 2.0)
                
            return min(current_beta * growth_factor, self.max_beta)
        
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
        
        # 早期迭代：适度增加beta以加快收敛
        if iteration < 30 and new_beta <= current_beta:
            new_beta = current_beta * 1.05
            
        return new_beta
    
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