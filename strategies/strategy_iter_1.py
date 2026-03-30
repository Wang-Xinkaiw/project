import numpy as np
from typing import Dict, Any

# 假设 BaseTuningStrategy 基类已定义
class BaseTuningStrategy:
    """ADMM参数调整策略基类"""
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def adjust_beta(iteration_state: Dict[str, Any]) -> float:
        """获取当前参数"""
        return self.params
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置参数"""
        self.params.update(params)


class EnhancedBetaTuningStrategy(BaseTuningStrategy):
    """
    增强的ADMM惩罚参数beta自适应调整策略
    
    针对l1_regression和elastic_net_regression问题的改进策略：
    - 针对含噪声问题，采用beta单调递增策略（只增不减）
    - 结合残差平衡机制，避免过度惩罚
    - 引入历史信息，提高策略鲁棒性
    """
    
    def __init__(self, **kwargs):
        """
        初始化策略参数
        
        Args:
            initial_beta: 初始beta值，默认1.0
            min_beta: beta最小值，默认1e-6
            max_beta: beta最大值，默认1e4
            growth_factor: 增长因子，默认1.5-2.5
            mu: 残差平衡阈值，默认10.0
            patience: 未收敛容忍次数，默认5
            fast_growth_threshold: 快速增长阈值，默认5.0
            slow_growth_threshold: 慢速增长阈值，默认0.1
        """
        super().__init__(**kwargs)
        
        # 默认参数
        default_params = {
            'initial_beta': 1.0,
            'min_beta': 1e-6,
            'max_beta': 1e4,
            'growth_factor': 1.8,  # 默认1.8，可根据问题调整
            'mu': 10.0,
            'patience': 5,
            'fast_growth_threshold': 5.0,
            'slow_growth_threshold': 0.1,
            'history_window': 3,  # 历史窗口大小
            'convergence_tolerance': 1e-6
        }
        
        # 更新默认参数
        default_params.update(kwargs)
        self.params = default_params
        
        # 初始化历史记录
        self.primal_history = []
        self.dual_history = []
        self.beta_history = []
        self.stagnation_count = 0
        
    def adjust_beta(iteration_state: Dict[str, Any]) -> float:
            iteration_state: 包含当前迭代状态的字典
        
        Returns:
            Dict[str, Any]: 更新后的参数字典，包含新的beta值
        """
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual')
        dual_res = iteration_state.get('dual_residual')
        current_beta = iteration_state.get('beta', self.params['initial_beta'])
        converged = iteration_state.get('converged', False)
        
        # 第一次迭代，使用初始beta
        if iteration == 0:
            new_beta = self.params['initial_beta']
            self.beta_history.append(new_beta)
            return {'beta': new_beta}
        
        # 如果已收敛，保持当前beta
        if converged:
            return {'beta': current_beta}
        
        # 处理缺失值
        if primal_res is None or dual_res is None:
            return {'beta': current_beta}
        
        # 更新历史记录
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        self.beta_history.append(current_beta)
        
        # 限制历史记录长度
        window = self.params['history_window']
        if len(self.primal_history) > window:
            self.primal_history = self.primal_history[-window:]
            self.dual_history = self.dual_history[-window:]
            self.beta_history = self.beta_history[-window:]
        
        # 针对含噪声回归问题的特殊策略：beta单调递增
        # 但结合残差平衡进行适度调整
        
        # 计算历史平均残差
        if len(self.primal_history) > 1:
            avg_primal = np.mean(self.primal_history)
            avg_dual = np.mean(self.dual_history)
            
            # 检查收敛停滞
            recent_primal_change = abs(self.primal_history[-1] - self.primal_history[-2])
            if recent_primal_change < self.params['convergence_tolerance']:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        else:
            avg_primal = primal_res
            avg_dual = dual_res
        
        # 策略1: 残差平衡调整
        if dual_res > 1e-10:
            # 计算残差比，避免除零
            ratio = primal_res / dual_res if dual_res > 0 else float('inf')
            
            # 基础增长因子
            growth = self.params['growth_factor']
            
            # 根据残差比调整增长因子
            if ratio > self.params['fast_growth_threshold']:
                # 原始残差过大，快速增加beta
                growth_factor = min(growth * 1.3, 2.5)
            elif ratio < self.params['slow_growth_threshold']:
                # 对偶残差过大，慢速增加beta
                growth_factor = max(growth * 0.7, 1.1)
            else:
                # 残差相对平衡，使用标准增长因子
                growth_factor = growth
            
            # 检查收敛停滞，如果停滞多次，增加beta以推动收敛
            if self.stagnation_count > self.params['patience']:
                growth_factor = min(growth_factor * 1.5, 3.0)
            
            # 计算新beta值（只增不减）
            new_beta = current_beta * growth_factor
        
        else:
            # 如果对偶残差非常小，保持当前beta
            new_beta = current_beta * self.params['growth_factor']
        
        # 策略2: 安全界限限制
        # 确保beta在合理范围内
        min_beta = self.params['min_beta']
        max_beta = self.params['max_beta']
        
        # 对beta进行平滑限制，避免突变
        beta_change = new_beta / current_beta
        if beta_change > 10.0:
            # 如果变化太大，限制增长
            new_beta = current_beta * 5.0
        
        new_beta = np.clip(new_beta, min_beta, max_beta)
        
        # 策略3: 针对迭代次数的调整
        # 早期迭代：较快增长以快速逼近
        # 后期迭代：较慢增长以精细调整
        if iteration < 10:
            # 早期迭代，允许较快增长
            pass
        elif iteration > 50:
            # 后期迭代，减缓增长
            max_growth = 1.5
            if new_beta > current_beta * max_growth:
                new_beta = current_beta * max_growth
        
        # 更新历史记录
        self.beta_history.append(new_beta)
        
        return {'beta': float(new_beta)}
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取当前参数"""
        return self.params.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置参数"""
        self.params.update(params)