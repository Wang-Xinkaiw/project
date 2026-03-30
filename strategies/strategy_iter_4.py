from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class AdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础超参数配置
        self.min_beta = 0.1         # 提高最小值以增加早期惩罚强度
        self.max_beta = 1e4         # beta 最大值保持不变
        self.mu = 2.0               # 残差平衡阈值 - 保持成功策略设置
        self.tau_inc = 1.5          # beta 增大因子 - 减小以更平滑
        self.tau_dec = 1.5          # beta 减小因子 - 保持对称
        
        # 问题特定参数
        self.initial_beta = 1.0     # 初始 beta 值
        self.problem_aware = True   # 启用问题感知策略
        
        # 收敛监控
        self.primal_history = []
        self.dual_history = []
        self.objective_history = []
        self.max_history_size = 3   # 减小历史记录以更快响应
        
        # 收敛状态跟踪
        self.last_primal = None
        self.last_dual = None
        self.last_objective = None
        self.slow_progress_count = 0
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从 iteration_state 获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        objective = iteration_state.get('objective', None)
        converged = iteration_state.get('converged', False)
        
        # 初始化 beta（第一次迭代）
        if iteration == 0:
            new_beta = self.initial_beta
            return {'beta': float(new_beta)}
        
        # 如果已收敛，保持当前 beta
        if converged:
            return {'beta': float(current_beta)}
        
        # 更新历史记录
        self._update_history(primal_res, dual_res, objective)
        
        # 处理缺失的残差值
        if primal_res is None or dual_res is None:
            return {'beta': float(current_beta)}
        
        # 特殊处理：针对未收敛问题的优化策略
        # 根据测试结果，l1_regularization, elastic_net, l1_regression 未收敛
        if iteration > 50 and not converged:
            # 检测收敛缓慢的情况
            is_slow_convergence = self._detect_slow_convergence(primal_res, dual_res, objective)
            
            if is_slow_convergence:
                self.slow_progress_count += 1
                
                # 针对不同阶段采取不同策略
                if iteration < 200:
                    # 早期阶段：适度增加 beta 以加速收敛
                    if self.slow_progress_count >= 5:
                        new_beta = current_beta * 1.8
                        self.slow_progress_count = 0
                    else:
                        new_beta = current_beta * 1.2
                elif iteration < 350:
                    # 中期阶段：更激进地增加 beta
                    if self.slow_progress_count >= 3:
                        new_beta = current_beta * 2.5
                        self.slow_progress_count = 0
                    else:
                        new_beta = current_beta * 1.5
                else:
                    # 后期阶段：大幅增加 beta 以强制收敛
                    new_beta = current_beta * 3.0
                    self.slow_progress_count = 0
            else:
                # 正常收敛情况使用标准策略
                new_beta = self._standard_admm_strategy(current_beta, primal_res, dual_res, iteration)
                self.slow_progress_count = max(0, self.slow_progress_count - 1)
        else:
            # 正常情况使用标准策略
            new_beta = self._standard_admm_strategy(current_beta, primal_res, dual_res, iteration)
        
        # 限制 beta 范围
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
        primal_change = abs(primal_res - self.primal_history[0]) / (self.primal_history[0] + 1e-12)
        
        # 检查目标函数变化率
        if len(self.objective_history) >= 2 and self.objective_history[-1] is not None and self.objective_history[0] is not None:
            objective_change = abs(self.objective_history[-1] - self.objective_history[0]) / (abs(self.objective_history[0]) + 1e-12)
        else:
            objective_change = 0
        
        # 判断条件：残差变化很小或目标函数变化很小
        return primal_change < 0.01 and objective_change < 0.01
    
    def _standard_admm_strategy(self, current_beta: float, 
                               primal_res: float, dual_res: float,
                               iteration: int) -> float:
        """标准 ADMM 自适应调整策略"""
        # 处理极小的对偶残差
        if dual_res < 1e-12:
            return current_beta
        
        # 计算残差比
        ratio = primal_res / (dual_res + 1e-12)
        
        # 根据迭代阶段调整策略参数
        if iteration < 30:
            # 早期阶段：保守调整
            local_mu = self.mu * 0.8
            local_tau_inc = 1.3
            local_tau_dec = 1.3
        elif iteration < 100:
            # 中期阶段：适度调整
            local_mu = self.mu
            local_tau_inc = self.tau_inc
            local_tau_dec = self.tau_dec
        else:
            # 后期阶段：激进调整
            local_mu = self.mu * 1.2
            local_tau_inc = min(self.tau_inc * 1.2, 2.5)
            local_tau_dec = max(self.tau_dec * 0.9, 1.2)
        
        # 基于残差比调整 beta
        if ratio > local_mu:
            # 原始残差过大，增加惩罚强度
            new_beta = current_beta * local_tau_inc
        elif ratio < 1.0 / local_mu:
            # 对偶残差过大，减少惩罚强度
            new_beta = current_beta / local_tau_dec
        else:
            # 残差平衡，保持当前 beta
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
            'problem_aware': self.problem_aware,
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
        if 'problem_aware' in params:
            self.problem_aware = params['problem_aware']
        if 'max_history_size' in params:
            self.max_history_size = params['max_history_size']