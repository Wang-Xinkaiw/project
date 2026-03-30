from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础超参数 - 基于历史成功策略微调
        self.min_beta = 0.3         # 降低最小值以提供更多调整空间
        self.max_beta = 1e4         # 保持最大值
        self.mu = 2.0               # 残差平衡阈值 - 保持成功策略
        self.tau_inc = 1.6          # 降低增加因子以更平稳调整
        self.tau_dec = 1.4          # 增加减少因子以加快调整响应
        
        # 初始化参数
        self.initial_beta = 0.8     # 适中的初始值
        self.adaptive_mode = "balanced"  # 调整模式: balanced/aggressive/conservative
        
        # 历史记录和收敛检测
        self.primal_history = []
        self.dual_history = []
        self.history_size = 5
        self.stagnation_count = 0
        self.max_stagnation = 10
        
        # 低秩问题专用参数
        self.low_rank_detected = False
        self.smooth_adjustment_factor = 0.8  # 低秩问题的平滑调整因子
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        objective = iteration_state.get('objective', None)
        converged = iteration_state.get('converged', False)
        
        # 初始化beta（第一次迭代）
        if iteration == 0:
            new_beta = self.initial_beta
            return {'beta': float(new_beta)}
        
        # 如果已收敛，保持当前beta
        if converged:
            return {'beta': float(current_beta)}
        
        # 处理缺失的残差值
        if primal_res is None or dual_res is None:
            return {'beta': float(current_beta)}
        
        # 检测低秩矩阵问题（基于残差特性）
        if iteration > 20 and not self.low_rank_detected:
            if dual_res is not None and primal_res is not None:
                # 低秩问题通常有特定的残差模式
                if dual_res > 1e-6 and primal_res > 1e-6:
                    residual_ratio = primal_res / (dual_res + 1e-12)
                    if 0.5 < residual_ratio < 2.0:
                        self.low_rank_detected = True
        
        # 更新历史记录
        self._update_history(primal_res, dual_res)
        
        # 根据迭代阶段选择调整策略
        if iteration < 20:
            # 早期阶段：保守调整
            new_beta = self._early_stage_strategy(current_beta, primal_res, dual_res)
        elif iteration > 100 and self._detect_stagnation():
            # 后期停滞阶段：积极调整打破僵局
            new_beta = self._aggressive_strategy(current_beta, primal_res, dual_res, iteration)
        else:
            # 中期阶段：标准自适应策略
            new_beta = self._standard_strategy(current_beta, primal_res, dual_res, iteration)
        
        # 应用低秩问题特殊平滑处理
        if self.low_rank_detected:
            new_beta = self._apply_low_rank_smoothing(current_beta, new_beta)
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 回归问题特殊处理：beta只增不减（针对l1_regression等）
        if iteration > 50 and primal_res > 1e-3 and dual_res < primal_res * 0.1:
            new_beta = max(new_beta, current_beta)
        
        return {'beta': float(new_beta)}
    
    def _update_history(self, primal_res: float, dual_res: float) -> None:
        """更新历史记录"""
        if primal_res is not None:
            self.primal_history.append(primal_res)
        if dual_res is not None:
            self.dual_history.append(dual_res)
        
        # 保持历史记录大小
        if len(self.primal_history) > self.history_size:
            self.primal_history.pop(0)
        if len(self.dual_history) > self.history_size:
            self.dual_history.pop(0)
    
    def _detect_stagnation(self) -> bool:
        """检测收敛停滞情况"""
        if len(self.primal_history) < 3:
            return False
        
        # 计算最近几次迭代的残差变化
        primal_changes = []
        for i in range(1, len(self.primal_history)):
            if self.primal_history[i-1] > 0:
                change = abs(self.primal_history[i] - self.primal_history[i-1]) / self.primal_history[i-1]
                primal_changes.append(change)
        
        # 如果变化率都很小，认为停滞
        if len(primal_changes) > 2:
            avg_change = np.mean(primal_changes[-3:])
            if avg_change < 0.01:  # 变化率小于1%
                self.stagnation_count += 1
            else:
                self.stagnation_count = max(0, self.stagnation_count - 1)
        
        return self.stagnation_count >= self.max_stagnation
    
    def _early_stage_strategy(self, current_beta: float, 
                             primal_res: float, dual_res: float) -> float:
        """早期迭代阶段策略"""
        # 早期阶段保持相对稳定，避免过大调整
        if dual_res < 1e-12:
            return current_beta
        
        ratio = primal_res / (dual_res + 1e-12)
        
        # 使用更保守的参数
        early_mu = self.mu * 1.2  # 更宽松的阈值
        early_tau_inc = 1.3
        early_tau_dec = 1.2
        
        if ratio > early_mu:
            new_beta = current_beta * early_tau_inc
        elif ratio < 1.0 / early_mu:
            new_beta = current_beta / early_tau_dec
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _standard_strategy(self, current_beta: float,
                          primal_res: float, dual_res: float,
                          iteration: int) -> float:
        """标准自适应调整策略"""
        if dual_res < 1e-12:
            return current_beta
        
        ratio = primal_res / (dual_res + 1e-12)
        
        # 根据迭代次数微调参数
        if iteration < 50:
            local_mu = self.mu
            local_tau_inc = self.tau_inc
            local_tau_dec = self.tau_dec
        else:
            # 后期阶段稍微偏向增加beta以促进收敛
            local_mu = self.mu * 0.9
            local_tau_inc = min(self.tau_inc * 1.1, 2.0)
            local_tau_dec = self.tau_dec
        
        if ratio > local_mu:
            new_beta = current_beta * local_tau_inc
        elif ratio < 1.0 / local_mu:
            new_beta = current_beta / local_tau_dec
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _aggressive_strategy(self, current_beta: float,
                            primal_res: float, dual_res: float,
                            iteration: int) -> float:
        """停滞阶段的积极调整策略"""
        if dual_res < 1e-12:
            return current_beta * 1.5  # 积极增加beta
        
        ratio = primal_res / (dual_res + 1e-12)
        
        # 更激进的调整参数
        agg_mu = self.mu * 0.7  # 更严格的阈值
        agg_tau_inc = 2.0
        agg_tau_dec = 1.2
        
        if ratio > agg_mu:
            new_beta = current_beta * agg_tau_inc
        elif ratio < 1.0 / agg_mu:
            new_beta = current_beta / agg_tau_dec
        else:
            # 即使平衡也稍微增加以打破停滞
            new_beta = current_beta * 1.1
        
        return new_beta
    
    def _apply_low_rank_smoothing(self, current_beta: float, new_beta: float) -> float:
        """低秩问题的平滑调整"""
        # 避免beta剧烈变化，使用平滑过渡
        smoothed_beta = current_beta * self.smooth_adjustment_factor + new_beta * (1 - self.smooth_adjustment_factor)
        return smoothed_beta
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'initial_beta': self.initial_beta,
            'history_size': self.history_size,
            'max_stagnation': self.max_stagnation,
            'smooth_adjustment_factor': self.smooth_adjustment_factor
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
        if 'initial_beta' in params:
            self.initial_beta = params['initial_beta']
        if 'history_size' in params:
            self.history_size = params['history_size']
        if 'max_stagnation' in params:
            self.max_stagnation = params['max_stagnation']
        if 'smooth_adjustment_factor' in params:
            self.smooth_adjustment_factor = params['smooth_adjustment_factor']