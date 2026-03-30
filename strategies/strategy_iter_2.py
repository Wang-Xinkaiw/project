from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class EnhancedAdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础参数 - 针对未收敛问题优化
        self.initial_beta = 1.0
        self.min_beta = 1e-6
        self.max_beta = 1e4
        
        # 自适应调整参数 - 减少未收敛问题的迭代次数
        self.mu = 8.0                # 降低残差平衡阈值，使调整更灵敏
        self.tau_inc = 1.8           # 略微降低增大因子，避免过冲
        self.tau_dec = 1.8           # 略微降低减小因子，保持平衡
        self.growth_factor = 1.3     # 降低单调增长因子，更平稳
        
        # 针对未收敛问题的特殊处理
        self.max_iter_before_aggressive = 50  # 开始激进调整前的最大迭代次数
        self.aggressive_increment = 2.5       # 激进调整时的增长因子
        self.convergence_acceleration = 1.5   # 加速收敛的调整因子
        
        # 历史信息记录
        self.primal_history = []
        self.dual_history = []
        self.beta_history = []
        self.max_history_len = 8     # 缩短历史长度，响应更快
        
        # 问题类型识别
        self.is_regression_problem = False
        self.monotonic_mode = False
        self.iteration_count = 0
        self.last_primal_res = float('inf')
        
        # 收敛监控
        self.stall_counter = 0
        self.stall_threshold = 8     # 增加停滞阈值，避免误判
        self.last_objective = float('inf')
        self.objective_stagnant = False
        
        # 初始化标志
        self.initialized = False
        
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
        
        # 处理初始迭代
        if not self.initialized and primal_res is not None:
            self.last_primal_res = primal_res
            self.initialized = True
        
        # 处理None值 - 针对l1_regularization和elastic_net中残差为None的情况
        if primal_res is None or dual_res is None:
            # 对于残差为None的情况，采用保守策略
            if iteration > 20 and not converged:
                # 如果迭代次数多但未收敛，适度增加beta
                new_beta = min(current_beta * 1.1, self.max_beta)
            else:
                new_beta = current_beta
            return {'beta': float(new_beta)}
        
        # 检测问题类型 - 简化检测逻辑
        self._detect_problem_type_simple(iteration_state)
        
        # 针对回归问题启用单调增长模式
        if self.is_regression_problem and not converged:
            if iteration > 5:  # 早期迭代后启用
                self.monotonic_mode = True
        
        # 记录历史信息
        self._update_history(primal_res, dual_res, current_beta)
        
        # 监控目标函数停滞
        self._monitor_objective_stagnation(objective)
        
        # 选择更新策略
        if self.monotonic_mode:
            # 单调增长策略 - 针对回归问题优化
            new_beta = self._monotonic_update_enhanced(current_beta, iteration, primal_res)
        elif iteration < 5:
            # 早期迭代：保守策略
            new_beta = self._early_stage_update(current_beta, primal_res, dual_res)
        elif iteration > 100 and not converged and primal_res > 1e-2:
            # 后期未收敛：激进策略
            new_beta = self._aggressive_update(current_beta, iteration, primal_res, dual_res)
        else:
            # 标准自适应策略
            new_beta = self._adaptive_update_enhanced(current_beta, primal_res, dual_res, iteration)
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 针对低秩问题的平滑调整
        if self._is_low_rank_problem():
            # 限制调整幅度，避免剧烈变化
            beta_change = abs(new_beta - current_beta) / (current_beta + 1e-10)
            if beta_change > 0.3:  # 限制变化幅度在30%以内
                new_beta = current_beta * 1.3 if new_beta > current_beta else current_beta / 1.3
        
        # 如果检测到目标函数停滞，尝试调整beta
        if self.objective_stagnant and iteration > 15 and not converged:
            # 中等幅度调整beta以跳出局部最优
            adjustment = 1.5 if np.random.rand() > 0.5 else 0.67
            new_beta = current_beta * adjustment
            self.objective_stagnant = False
        
        return {'beta': float(new_beta)}
    
    def _detect_problem_type_simple(self, iteration_state: Dict[str, Any]) -> None:
        """简化问题类型检测"""
        iteration = iteration_state.get('iteration', 0)
        
        # 仅基于迭代次数和经验判断
        if iteration > 10:
            # 如果迭代次数较多且未收敛，可能是回归问题
            # 这里简化检测，主要依赖外部调用时的设置
            pass
    
    def _monotonic_update_enhanced(self, current_beta: float, iteration: int, 
                                  primal_res: float) -> float:
        """增强的单调增长更新策略"""
        # 根据残差大小调整增长速率
        if primal_res > 0.1:
            # 残差大，快速增长
            growth = min(self.aggressive_increment, 2.0)
        elif primal_res > 0.01:
            # 残差中等，适度增长
            growth = self.growth_factor
        else:
            # 残差小，缓慢增长
            growth = 1.1
        
        # 后期减缓增长
        if current_beta > 500:
            growth = min(growth, 1.1)
        
        new_beta = current_beta * growth
        return new_beta
    
    def _early_stage_update(self, current_beta: float, primal_res: float, 
                           dual_res: float) -> float:
        """早期迭代阶段更新策略"""
        # 处理零除
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        # 早期使用更保守的调整
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 早期使用更宽松的阈值
            mu_early = self.mu * 3.0
            
            if ratio > mu_early:
                # 原始残差大，适度增大beta
                new_beta = current_beta * 1.3
            elif ratio < 1.0 / mu_early:
                # 对偶残差大，适度减小beta
                new_beta = current_beta / 1.3
            else:
                # 保持平衡
                new_beta = current_beta
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _aggressive_update(self, current_beta: float, iteration: int,
                          primal_res: float, dual_res: float) -> float:
        """激进更新策略 - 针对长时间未收敛的情况"""
        # 处理零除
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            if ratio > 5.0:  # 更激进的阈值
                # 原始残差明显更大，大幅增加beta
                new_beta = current_beta * self.aggressive_increment
            elif ratio < 0.2:  # 更激进的阈值
                # 对偶残差明显更大，大幅减小beta
                new_beta = current_beta / self.aggressive_increment
            else:
                # 尝试中等调整
                if primal_res > self.last_primal_res:
                    # 原始残差在增加，需要增大beta
                    new_beta = current_beta * 1.8
                else:
                    new_beta = current_beta
        else:
            # 默认中等增长
            new_beta = current_beta * 1.5
        
        # 更新最后原始残差
        self.last_primal_res = primal_res
        
        return new_beta
    
    def _adaptive_update_enhanced(self, current_beta: float, primal_res: float, 
                                 dual_res: float, iteration: int) -> float:
        """增强的自适应更新策略"""
        # 处理零除
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        # 计算残差比
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 根据迭代阶段调整阈值
            if iteration < 15:
                mu_adjusted = self.mu * 1.5
                tau_inc = min(self.tau_inc, 1.6)
                tau_dec = min(self.tau_dec, 1.6)
            else:
                mu_adjusted = self.mu
                tau_inc = self.tau_inc
                tau_dec = self.tau_dec
            
            # 基于残差比调整
            if ratio > mu_adjusted:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / mu_adjusted:
                new_beta = current_beta / tau_dec
            else:
                # 残差平衡时，根据残差绝对值调整
                if primal_res > 0.1 and iteration > 10:
                    # 残差仍然较大，适度增加beta
                    new_beta = current_beta * 1.1
                else:
                    new_beta = current_beta
        else:
            new_beta = current_beta
        
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
    
    def _monitor_objective_stagnation(self, objective: float) -> None:
        """监控目标函数停滞"""
        if objective is None:
            return
        
        # 计算目标函数相对变化
        if self.last_objective < float('inf'):
            rel_change = abs(objective - self.last_objective) / (abs(self.last_objective) + 1e-10)
            
            # 如果相对变化很小，增加停滞计数器
            if rel_change < 1e-6:
                self.stall_counter += 1
            else:
                self.stall_counter = max(0, self.stall_counter - 1)
            
            # 如果停滞计数器超过阈值，标记为停滞
            if self.stall_counter >= self.stall_threshold:
                self.objective_stagnant = True
            else:
                self.objective_stagnant = False
        
        # 更新最后的目标函数值
        self.last_objective = objective
    
    def _is_low_rank_problem(self) -> bool:
        """判断是否为低秩矩阵问题"""
        if len(self.beta_history) < 3:
            return False
        
        # 通过beta变化模式判断
        recent_betas = self.beta_history[-3:]
        beta_var = np.std(recent_betas) / (np.mean(recent_betas) + 1e-10)
        
        # 低秩问题通常beta变化较小
        return beta_var < 0.2
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'initial_beta': self.initial_beta,
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'growth_factor': self.growth_factor,
            'max_iter_before_aggressive': self.max_iter_before_aggressive,
            'aggressive_increment': self.aggressive_increment,
            'convergence_acceleration': self.convergence_acceleration,
            'max_history_len': self.max_history_len,
            'stall_threshold': self.stall_threshold
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        valid_params = [
            'initial_beta', 'min_beta', 'max_beta', 'mu', 'tau_inc', 'tau_dec',
            'growth_factor', 'max_iter_before_aggressive', 'aggressive_increment',
            'convergence_acceleration', 'max_history_len', 'stall_threshold'
        ]
        
        for key in valid_params:
            if key in params:
                setattr(self, key, params[key])
        
        # 针对回归问题的特殊设置
        if 'is_regression_problem' in params:
            self.is_regression_problem = params['is_regression_problem']
        if 'monotonic_mode' in params:
            self.monotonic_mode = params['monotonic_mode']