# strategies/standard_admm_strategy.py
"""
标准ADMM策略类 - 基于论文公式17和26
论文参考: De-Ren Han. 'A Survey on Some Recent Developments of ADMM' (2022)
公式17: 标准ADMM迭代 (第8-9页)
公式26: 自适应惩罚参数调整 (第12页)
"""

import numpy as np
from typing import Dict, Any
from strategies.base_strategy import BaseTuningStrategy


class StandardADMMStrategy(BaseTuningStrategy):
    """
    标准ADMM参数调整策略（论文公式26）
    
    自适应惩罚参数调整规则:
    β^{k+1} = (1+μ)β^k,   if ||r^k|| > v₁||s^k||
    β^{k+1} = β^k/(1+μ),  if ||s^k|| > v₂||r^k||
    β^{k+1} = β^k,        otherwise
    """
    
    def __init__(self, 
                 initial_beta: float = 1.0,
                 mu: float = 0.1,
                 v1: float = 10.0,
                 v2: float = 10.0):
        """
        初始化标准ADMM策略
        
        参数:
            initial_beta: 初始惩罚参数β₀
            mu: 调整幅度参数μ (论文中的μ)
            v1: 原始残差阈值倍数v₁ (论文中的v₁)
            v2: 对偶残差阈值倍数v₂ (论文中的v₂)
        """
        self.beta = initial_beta
        self.mu = mu
        self.v1 = v1
        self.v2 = v2
        
        # 参数验证
        self._validate_parameters()
        
        # 历史记录（用于监控）
        self.adjustment_history = []
    
    def _validate_parameters(self):
        """验证参数范围"""
        if self.beta <= 0:
            raise ValueError(f"惩罚参数β必须为正，当前值: {self.beta}")
        if self.mu <= 0:
            raise ValueError(f"调整幅度μ必须为正，当前值: {self.mu}")
        if self.v1 <= 0:
            raise ValueError(f"阈值v₁必须为正，当前值: {self.v1}")
        if self.v2 <= 0:
            raise ValueError(f"阈值v₂必须为正，当前值: {self.v2}")
    
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据迭代状态更新ADMM参数（论文公式26）
        
        参数:
            iteration_state: 必须包含:
                - 'primal_residual': 原始残差 ||r^k|| = ||A₁x₁^k + A₂x₂^k - b||
                - 'dual_residual': 对偶残差 ||s^k|| = ||β^k A₁^T A₂ (x₂^{k+1} - x₂^k)||
                - 'iteration': 当前迭代次数k
                
        返回:
            更新后的参数字典:
                - 'beta': 新的惩罚参数β^{k+1}
                - 'adjustment_type': 调整类型 ('increase', 'decrease', 'keep')
        """
        # 提取必要信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', 0.0)
        dual_res = iteration_state.get('dual_residual', 0.0)
        
        # 防止除以零
        primal_safe = max(primal_res, 1e-16)
        dual_safe = max(dual_res, 1e-16)
        
        # 记录当前状态
        old_beta = self.beta
        adjustment_type = "keep"
        
        # 论文公式26：从第二次迭代开始调整 (k > 0)
        if iteration > 0:
            if primal_safe > self.v1 * dual_safe:
                # 原始残差相对较大，增大惩罚参数
                self.beta *= (1 + self.mu)
                adjustment_type = "increase"
            elif dual_safe > self.v2 * primal_safe:
                # 对偶残差相对较大，减小惩罚参数
                self.beta /= (1 + self.mu)
                adjustment_type = "decrease"
        
        # 记录调整历史
        self.adjustment_history.append({
            'iteration': iteration,
            'old_beta': old_beta,
            'new_beta': self.beta,
            'adjustment_type': adjustment_type,
            'primal_residual': primal_res,
            'dual_residual': dual_res
        })
        
        return {
            'beta': self.beta
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取当前策略参数"""
        return {
            'beta': self.beta,
            'mu': self.mu,
            'v1': self.v1,
            'v2': self.v2
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """设置策略参数"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self._validate_parameters()
    
    def reset(self):
        """重置策略状态（保留配置参数）"""
        # 清除历史记录
        self.adjustment_history = []
        # 注意：这里不重置beta，因为beta是算法状态的一部分
        # 如果需要重置beta，可以在外部调用set_parameters
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取策略性能指标"""
        if not self.adjustment_history:
            return {}
        
        # 计算调整频率
        total_iterations = len(self.adjustment_history)
        if total_iterations == 0:
            return {}
        
        adjustments = [h for h in self.adjustment_history if h['adjustment_type'] != 'keep']
        adjustment_count = len(adjustments)
        
        return {
            'adjustment_frequency': adjustment_count / total_iterations if total_iterations > 0 else 0,
            'total_adjustments': adjustment_count,
            'increase_count': len([h for h in self.adjustment_history if h['adjustment_type'] == 'increase']),
            'decrease_count': len([h for h in self.adjustment_history if h['adjustment_type'] == 'decrease']),
            'final_beta': self.beta,
            'beta_range': max(h['new_beta'] for h in self.adjustment_history) / 
                         min(h['new_beta'] for h in self.adjustment_history) if self.adjustment_history else 1.0
        }