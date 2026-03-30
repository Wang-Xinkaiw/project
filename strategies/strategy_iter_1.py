from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class AdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础参数
        self.initial_beta = 1.0
        self.min_beta = 1e-6
        self.max_beta = 1e4
        
        # 自适应调整参数
        self.mu = 10.0               # 残差平衡阈值
        self.tau_inc = 2.0           # 增大因子
        self.tau_dec = 2.0           # 减小因子
        self.growth_factor = 1.5     # 单调增长因子（针对回归问题）
        
        # 历史信息记录
        self.primal_history = []
        self.dual_history = []
        self.beta_history = []
        self.max_history_len = 10
        
        # 问题类型识别
        self.is_regression_problem = False  # 针对l1_regression/elastic_net_regression
        self.monotonic_mode = False         # 是否启用单调增长模式
        self.iteration_count = 0
        
        # 收敛监控
        self.convergence_stalled = False
        self.stall_counter = 0
        self.stall_threshold = 5
        self.last_objective = float('inf')
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        objective = iteration_state.get('objective', None)
        converged = iteration_state.get('converged', False)
        
        # 更新迭代计数
        self.iteration_count = iteration
        
        # 检测问题类型（基于收敛行为模式）
        self._detect_problem_type(iteration_state)
        
        # 如果是回归问题且未收敛，启用单调增长模式
        if self.is_regression_problem and not converged and iteration > 10:
            self.monotonic_mode = True
        
        # 监控收敛停滞
        self._monitor_convergence_stall(objective)
        
        # 处理None值
        if primal_res is None or dual_res is None:
            new_beta = current_beta
        else:
            # 记录历史信息
            self._update_history(primal_res, dual_res, current_beta)
            
            if self.monotonic_mode:
                # 单调增长策略（针对回归问题）
                new_beta = self._monotonic_update(current_beta, iteration)
            else:
                # 标准自适应策略
                new_beta = self._adaptive_update(current_beta, primal_res, dual_res, iteration)
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 对于低秩问题，使用更平滑的调整
        if self._is_low_rank_problem():
            # 限制调整幅度，避免剧烈变化
            beta_change = abs(new_beta - current_beta) / current_beta
            if beta_change > 0.5:  # 限制变化幅度在50%以内
                new_beta = current_beta * 1.5 if new_beta > current_beta else current_beta / 1.5
        
        # 如果检测到收敛停滞，尝试调整beta打破僵局
        if self.convergence_stalled and iteration > 20:
            # 中等幅度调整beta以跳出局部最优
            adjustment = 1.3 if np.random.rand() > 0.5 else 0.7
            new_beta = current_beta * adjustment
            self.stall_counter = 0
            self.convergence_stalled = False
        
        return {'beta': float(new_beta)}
    
    def _detect_problem_type(self, iteration_state: Dict[str, Any]) -> None:
        """根据收敛行为检测问题类型"""
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        
        if iteration < 5:
            return
        
        # 检测回归问题：原始残差下降缓慢，对偶残差波动较大
        if len(self.primal_history) > 4 and len(self.dual_history) > 4:
            primal_trend = np.mean(np.diff(self.primal_history[-4:]))
            dual_var = np.std(self.dual_history[-4:]) / (np.mean(self.dual_history[-4:]) + 1e-10)
            
            # 如果原始残差变化缓慢且对偶残差波动大，可能是回归问题
            if abs(primal_trend) < 1e-3 and dual_var > 0.5:
                self.is_regression_problem = True
    
    def _monotonic_update(self, current_beta: float, iteration: int) -> float:
        """单调增长更新策略（针对回归问题）"""
        # 初始迭代阶段缓慢增长
        if iteration < 10:
            growth = 1.1
        elif iteration < 20:
            growth = 1.3
        else:
            growth = self.growth_factor
        
        new_beta = current_beta * growth
        
        # 后期减缓增长
        if current_beta > 1000:
            growth = min(growth, 1.1)
            new_beta = current_beta * growth
        
        return new_beta
    
    def _adaptive_update(self, current_beta: float, primal_res: float, 
                        dual_res: float, iteration: int) -> float:
        """标准自适应更新策略"""
        # 处理零除或极小值
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        # 计算残差比
        ratio = primal_safe / dual_safe
        
        # 早期迭代阶段更保守
        if iteration < 10:
            mu_adjusted = self.mu * 2.0  # 更宽容的阈值
            tau_inc_adjusted = min(self.tau_inc, 1.5)
            tau_dec_adjusted = min(self.tau_dec, 1.5)
        else:
            mu_adjusted = self.mu
            tau_inc_adjusted = self.tau_inc
            tau_dec_adjusted = self.tau_dec
        
        # 基于残差比调整
        if dual_safe > epsilon:
            if ratio > mu_adjusted:
                # 原始残差大，增大beta
                new_beta = current_beta * tau_inc_adjusted
            elif ratio < 1.0 / mu_adjusted:
                # 对偶残差大，减小beta
                new_beta = current_beta / tau_dec_adjusted
            else:
                # 残差平衡，保持beta
                new_beta = current_beta
        else:
            new_beta = current_beta
        
        # 考虑历史趋势
        if len(self.primal_history) >= 3 and len(self.dual_history) >= 3:
            primal_trend = np.mean(np.diff(self.primal_history[-3:]))
            dual_trend = np.mean(np.diff(self.dual_history[-3:]))
            
            # 如果原始残差在增加而对偶残差在减少，需要增大beta
            if primal_trend > 0 and dual_trend < 0:
                new_beta = current_beta * 1.2
            # 如果原始残差在减少而对偶残差在增加，需要减小beta
            elif primal_trend < 0 and dual_trend > 0:
                new_beta = current_beta / 1.2
        
        return new_beta
    
    def _update_history(self, primal_res: float, dual_res: float, beta: float) -> None:
        """更新历史记录"""
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        self.beta_history.append(beta)
        
        # 保持历史记录长度
        if len(self.primal_history) > self.max_history_len:
            self.primal_history.pop(0)
            self.dual_history.pop(0)
            self.beta_history.pop(0)
    
    def _monitor_convergence_stall(self, objective: float) -> None:
        """监控收敛停滞"""
        if objective is None:
            return
        
        # 计算目标函数变化
        if self.last_objective < float('inf'):
            rel_change = abs(objective - self.last_objective) / (abs(self.last_objective) + 1e-10)
            
            # 如果相对变化很小，增加停滞计数器
            if rel_change < 1e-5:
                self.stall_counter += 1
            else:
                self.stall_counter = max(0, self.stall_counter - 1)
        
        # 更新最后的目标函数值
        self.last_objective = objective
        
        # 如果停滞计数器超过阈值，标记为停滞
        if self.stall_counter >= self.stall_threshold:
            self.convergence_stalled = True
    
    def _is_low_rank_problem(self) -> bool:
        """判断是否为低秩矩阵问题（基于历史行为模式）"""
        if len(self.beta_history) < 5:
            return False
        
        # 低秩问题通常需要更稳定的beta
        beta_changes = np.abs(np.diff(self.beta_history[-5:])) / self.beta_history[-5:-1]
        avg_beta_change = np.mean(beta_changes)
        
        # 如果beta变化幅度很小，可能是低秩问题
        return avg_beta_change < 0.1
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'initial_beta': self.initial_beta,
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'growth_factor': self.growth_factor,
            'max_history_len': self.max_history_len,
            'stall_threshold': self.stall_threshold
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        param_mapping = {
            'initial_beta': 'initial_beta',
            'min_beta': 'min_beta',
            'max_beta': 'max_beta',
            'mu': 'mu',
            'tau_inc': 'tau_inc',
            'tau_dec': 'tau_dec',
            'growth_factor': 'growth_factor',
            'max_history_len': 'max_history_len',
            'stall_threshold': 'stall_threshold'
        }
        
        for key, attr_name in param_mapping.items():
            if key in params:
                setattr(self, attr_name, params[key])