from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedAdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础参数配置
        self.initial_beta = 1.0
        self.min_beta = 1e-6
        self.max_beta = 1e4
        
        # 基于历史成功策略的参数 - 轻微调整
        self.mu = 8.5          # 残差平衡阈值，稍低于历史平均值
        self.tau_inc = 1.9     # 增大因子 - 略微增加
        self.tau_dec = 1.7     # 减小因子 - 略微减小
        
        # 针对未收敛问题的特殊策略
        self.aggressive_factor = 2.5      # 激进调整因子
        self.early_phase_iterations = 20  # 早期阶段迭代次数
        
        # 回归问题参数 - 根据建议调整
        self.regression_initial_beta = 0.5
        self.regression_growth_factor = 1.8
        self.regression_max_beta = 1e4
        
        # 收敛监控
        self.primal_history = []
        self.dual_history = []
        self.max_history_len = 3
        
        # 状态记录
        self.iteration_count = 0
        self.last_primal_res = float('inf')
        self.last_dual_res = float('inf')
        self.stagnation_counter = 0
        self.max_stagnation = 8
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        converged = iteration_state.get('converged', False)
        
        self.iteration_count = iteration
        
        # 处理None值 - 针对某些问题中残差为None的情况
        if primal_res is None or dual_res is None:
            # 对于未收敛问题采用更积极的策略
            if iteration > 100 and not converged:
                # 如果迭代次数多但未收敛，显著增加beta
                new_beta = min(current_beta * self.aggressive_factor, self.max_beta)
            elif iteration > 50 and not converged:
                # 中等迭代次数，适度增加beta
                new_beta = min(current_beta * 1.5, self.max_beta)
            else:
                new_beta = current_beta
            return {'beta': float(new_beta)}
        
        # 记录历史信息
        self._update_history(primal_res, dual_res)
        
        # 针对未收敛问题的特殊处理
        if iteration >= 100 and primal_res is not None and dual_res is not None:
            if primal_res > 0.1 or dual_res > 0.1:
                # 残差仍然较大，采用激进策略
                new_beta = self._aggressive_update(current_beta, primal_res, dual_res)
            else:
                # 使用标准策略
                new_beta = self._standard_adaptive_update(current_beta, primal_res, dual_res)
        elif iteration < self.early_phase_iterations:
            # 早期迭代：使用温和但有效的调整策略
            new_beta = self._early_stage_update(current_beta, primal_res, dual_res)
        else:
            # 中期迭代：标准自适应策略
            new_beta = self._standard_adaptive_update(current_beta, primal_res, dual_res)
        
        # 检测停滞并处理
        if self._is_stagnating(primal_res, dual_res):
            # 重置停滞计数器并尝试跳出停滞
            self.stagnation_counter = 0
            stagnation_adjustment = self._stagnation_update(current_beta, primal_res, dual_res)
            # 结合两种策略的结果
            new_beta = (new_beta + stagnation_adjustment) / 2.0
        
        # 针对回归问题的特殊处理（根据问题特性说明）
        if self._is_likely_regression_problem(primal_res, dual_res, iteration):
            regression_beta = self._regression_monotonic_update(current_beta, iteration)
            # 取两者中较大的值（偏向于回归策略）
            new_beta = max(new_beta, regression_beta)
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 更新最后残差值
        self.last_primal_res = primal_res
        self.last_dual_res = dual_res
        
        return {'beta': float(new_beta)}
    
    def _is_likely_regression_problem(self, primal_res: float, dual_res: float, 
                                    iteration: int) -> bool:
        """判断是否可能是回归问题"""
        # 回归问题通常有噪声项，残差行为可能不同
        if iteration > 30:
            # 如果原始残差相对较小但对偶残差较大，可能是回归问题
            if primal_res < 0.01 and dual_res > primal_res * 10:
                return True
            # 如果残差下降缓慢但仍在较高水平，也可能是回归问题
            if primal_res > 0.05 and dual_res > 0.05 and iteration > 50:
                return True
        return False
    
    def _regression_monotonic_update(self, current_beta: float, iteration: int) -> float:
        """回归问题的单调增长更新策略"""
        # 根据建议：初始beta=0.1~1.0，增长率1.1~1.5，上限1e4
        if iteration == 0:
            new_beta = self.regression_initial_beta
        else:
            # 根据迭代阶段调整增长率
            if iteration < 50:
                growth = self.regression_growth_factor
            elif iteration < 150:
                growth = 1.4
            elif iteration < 300:
                growth = 1.2
            else:
                growth = 1.05
            
            new_beta = current_beta * growth
        
        # 确保不超过上限
        return min(new_beta, self.regression_max_beta)
    
    def _aggressive_update(self, current_beta: float, primal_res: float, 
                          dual_res: float) -> float:
        """针对未收敛问题的激进更新策略"""
        # 处理零除保护
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 使用更激进的阈值
            if ratio > 15.0:
                # 原始残差非常大，大幅增加beta
                new_beta = current_beta * 3.0
            elif ratio > 5.0:
                new_beta = current_beta * 2.0
            elif ratio < 0.067:  # 1/15
                # 对偶残差非常大，大幅减小beta
                new_beta = current_beta / 3.0
            elif ratio < 0.2:    # 1/5
                new_beta = current_beta / 2.0
            else:
                # 中等调整
                if primal_res > 0.5:
                    new_beta = current_beta * 1.8
                elif primal_res < 0.01:
                    new_beta = current_beta * 0.8
                else:
                    new_beta = current_beta
        else:
            # 默认大幅增加beta
            new_beta = current_beta * self.aggressive_factor
        
        return new_beta
    
    def _early_stage_update(self, current_beta: float, primal_res: float,
                          dual_res: float) -> float:
        """早期迭代阶段更新策略"""
        # 处理零除保护
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 早期使用更宽松的阈值
            early_mu = self.mu * 1.5
            
            if ratio > early_mu:
                # 原始残差相对较大，适度增大beta
                new_beta = current_beta * 1.6
            elif ratio < 1.0 / early_mu:
                # 对偶残差相对较大，适度减小beta
                new_beta = current_beta / 1.6
            else:
                # 残差平衡，根据残差绝对值微调
                if primal_res > 0.5:
                    new_beta = current_beta * 1.3
                elif primal_res < 0.01:
                    new_beta = current_beta * 0.9
                else:
                    new_beta = current_beta
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _is_stagnating(self, primal_res: float, dual_res: float) -> bool:
        """检测是否陷入停滞状态"""
        if len(self.primal_history) < 2:
            return False
        
        # 检查最近两次迭代的残差变化
        if len(self.primal_history) >= 2:
            primal_change = abs(self.primal_history[-1] - self.primal_history[-2]) / (self.primal_history[-2] + 1e-10)
            dual_change = abs(self.dual_history[-1] - self.dual_history[-2]) / (self.dual_history[-2] + 1e-10)
            
            # 如果残差变化很小，增加停滞计数器
            if primal_change < 0.005 and dual_change < 0.005:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        return self.stagnation_counter >= self.max_stagnation
    
    def _stagnation_update(self, current_beta: float, primal_res: float,
                          dual_res: float) -> float:
        """停滞状态更新策略"""
        # 处理零除保护
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 根据残差比进行激进调整
            if ratio > 3.0:
                new_beta = current_beta * 2.2
            elif ratio < 0.33:
                new_beta = current_beta / 2.2
            else:
                # 随机扰动以跳出停滞
                import random
                perturbation = 0.8 + random.random() * 0.4  # 0.8到1.2之间的随机数
                new_beta = current_beta * perturbation
        else:
            # 默认增加beta
            new_beta = current_beta * 1.7
        
        return new_beta
    
    def _standard_adaptive_update(self, current_beta: float, primal_res: float,
                                dual_res: float) -> float:
        """标准自适应更新策略"""
        # 处理零除保护
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 使用配置的阈值和调整因子
            if ratio > self.mu:
                new_beta = current_beta * self.tau_inc
            elif ratio < 1.0 / self.mu:
                new_beta = current_beta / self.tau_dec
            else:
                # 残差平衡时，根据残差绝对值微调
                if primal_res > 0.1:
                    new_beta = current_beta * 1.15
                elif primal_res < 1e-3:
                    new_beta = current_beta * 0.9
                else:
                    new_beta = current_beta
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _update_history(self, primal_res: float, dual_res: float) -> None:
        """更新历史记录"""
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        
        # 保持历史记录长度
        if len(self.primal_history) > self.max_history_len:
            self.primal_history.pop(0)
            self.dual_history.pop(0)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'initial_beta': self.initial_beta,
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'aggressive_factor': self.aggressive_factor,
            'early_phase_iterations': self.early_phase_iterations,
            'regression_initial_beta': self.regression_initial_beta,
            'regression_growth_factor': self.regression_growth_factor,
            'regression_max_beta': self.regression_max_beta,
            'max_history_len': self.max_history_len,
            'max_stagnation': self.max_stagnation
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        valid_params = [
            'initial_beta', 'min_beta', 'max_beta', 'mu', 'tau_inc', 'tau_dec',
            'aggressive_factor', 'early_phase_iterations', 'regression_initial_beta',
            'regression_growth_factor', 'regression_max_beta', 'max_history_len',
            'max_stagnation'
        ]
        
        for key in valid_params:
            if key in params:
                setattr(self, key, params[key])