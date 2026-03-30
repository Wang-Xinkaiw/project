from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class AdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础超参数配置 - 针对l1_regression问题调整
        self.min_beta = 0.5         # 进一步提高最小值
        self.max_beta = 1e4         # beta最大值保持不变
        self.mu = 2.0               # 残差平衡阈值 - 保持成功策略
        self.tau_inc = 1.8          # 适度增大beta增加因子
        self.tau_dec = 1.3          # 减小beta减少因子，避免过度减小
        
        # 回归问题特殊处理
        self.initial_beta = 1.0     # 初始beta值
        self.regression_threshold = 150  # 回归问题特殊处理阈值
        self.regression_mode = False     # 是否进入回归模式
        
        # 收敛监控
        self.primal_history = []
        self.dual_history = []
        self.objective_history = []
        self.max_history_size = 3
        
        # 收敛状态跟踪
        self.last_primal = None
        self.last_dual = None
        self.last_objective = None
        self.slow_progress_count = 0
        
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
        
        # 更新历史记录
        self._update_history(primal_res, dual_res, objective)
        
        # 处理缺失的残差值
        if primal_res is None or dual_res is None:
            return {'beta': float(current_beta)}
        
        # 检测是否进入回归问题模式（针对l1_regression等含噪声问题）
        # 当迭代次数较多且残差下降缓慢时，采用单调递增策略
        if iteration > self.regression_threshold and not converged:
            # 检测收敛缓慢情况
            is_slow_convergence = self._detect_slow_convergence(primal_res, dual_res, objective)
            
            if is_slow_convergence:
                self.regression_mode = True
                self.slow_progress_count += 1
                
                # 回归问题特殊处理：beta单调递增策略
                if self.slow_progress_count >= 3:
                    # 连续3次迭代进展缓慢，采用更激进的增长
                    new_beta = current_beta * 2.2
                    self.slow_progress_count = 0
                else:
                    # 适度增长
                    new_beta = current_beta * 1.5
            else:
                # 正常进展，回归模式下仍然适度增长
                if self.regression_mode:
                    new_beta = current_beta * 1.2
                else:
                    # 标准策略
                    new_beta = self._standard_admm_strategy(current_beta, primal_res, dual_res, iteration)
                self.slow_progress_count = max(0, self.slow_progress_count - 1)
        else:
            # 正常情况使用标准策略
            new_beta = self._standard_admm_strategy(current_beta, primal_res, dual_res, iteration)
            self.regression_mode = False
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        return {'beta': float(new_beta)}
    
    def _update_history(self, primal_res: float, dual_res: float, objective: float) -> None:
        """更新历史记录"""
        if primal_res is not None:
            self.primal_history.append(primal_res)
        if dual_res is not None:
            self.dual_history.append(dual_res)
        if objective is not None:
            self.objective_history.append(objective)
        
        # 保持历史记录大小
        if len(self.primal_history) > self.max_history_size:
            self.primal_history.pop(0)
        if len(self.dual_history) > self.max_history_size:
            self.dual_history.pop(0)
        if len(self.objective_history) > self.max_history_size:
            self.objective_history.pop(0)
        
        # 保存最新值
        self.last_primal = primal_res
        self.last_dual = dual_res
        self.last_objective = objective
    
    def _detect_slow_convergence(self, primal_res: float, dual_res: float, objective: float) -> bool:
        """检测收敛缓慢的情况"""
        if len(self.primal_history) < 2 or len(self.objective_history) < 2:
            return False
        
        # 检查残差变化率
        if self.primal_history[0] is not None and primal_res is not None:
            primal_change = abs(primal_res - self.primal_history[0]) / (self.primal_history[0] + 1e-12)
        else:
            primal_change = 0
        
        # 检查目标函数变化率
        if (len(self.objective_history) >= 2 and 
            self.objective_history[-1] is not None and 
            self.objective_history[0] is not None and
            abs(self.objective_history[0]) > 1e-12):
            objective_change = abs(self.objective_history[-1] - self.objective_history[0]) / (abs(self.objective_history[0]) + 1e-12)
        else:
            objective_change = 0
        
        # 判断条件：残差变化很小且目标函数变化很小
        return primal_change < 0.02 and objective_change < 0.02
    
    def _standard_admm_strategy(self, current_beta: float, 
                               primal_res: float, dual_res: float,
                               iteration: int) -> float:
        """标准ADMM自适应调整策略"""
        # 处理极小的对偶残差
        if dual_res < 1e-12:
            return current_beta
        
        # 计算残差比
        ratio = primal_res / (dual_res + 1e-12)
        
        # 根据迭代阶段调整策略参数
        if iteration < 30:
            # 早期阶段：保守调整，避免过早增大beta
            local_mu = self.mu * 0.8
            local_tau_inc = 1.3
            local_tau_dec = 1.2
        elif iteration < 100:
            # 中期阶段：适度调整
            local_mu = self.mu
            local_tau_inc = self.tau_inc
            local_tau_dec = self.tau_dec
        else:
            # 后期阶段：更倾向于增加beta以促进收敛
            local_mu = self.mu * 0.9  # 稍微降低阈值，使更容易增加beta
            local_tau_inc = min(self.tau_inc * 1.1, 2.0)
            local_tau_dec = max(self.tau_dec * 0.8, 1.1)
        
        # 基于残差比调整beta
        if ratio > local_mu:
            # 原始残差过大，增加惩罚强度
            new_beta = current_beta * local_tau_inc
        elif ratio < 1.0 / local_mu:
            # 对偶残差过大，减少惩罚强度
            new_beta = current_beta / local_tau_dec
        else:
            # 残差平衡，保持当前beta
            new_beta = current_beta
        
        return new_beta
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'initial_beta': self.initial_beta,
            'regression_threshold': self.regression_threshold,
            'max_history_size': self.max_history_size
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
        if 'regression_threshold' in params:
            self.regression_threshold = params['regression_threshold']
        if 'max_history_size' in params:
            self.max_history_size = params['max_history_size']