from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedAdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础参数配置
        self.initial_beta = 1.0
        self.min_beta = 1e-6
        self.max_beta = 1e4
        
        # 自适应调整参数 - 根据历史建议微调
        self.mu = 9.0          # 残差平衡阈值，基于历史成功策略平均值
        self.tau_inc = 1.8     # 增大因子
        self.tau_dec = 1.8     # 减小因子
        
        # 回归问题专用参数
        self.monotonic_growth_factor = 1.5  # 单调增长因子
        self.regression_threshold = 0.01    # 判断是否采用单调策略的残差阈值
        
        # 收敛监控
        self.primal_history = []
        self.dual_history = []
        self.max_history_len = 5
        
        # 状态记录
        self.iteration_count = 0
        self.last_primal_res = float('inf')
        self.last_dual_res = float('inf')
        self.stagnation_counter = 0
        self.max_stagnation = 10
        
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
            # 对于残差为None的情况，采用保守策略
            if iteration > 50 and not converged:
                # 如果迭代次数多但未收敛，适度增加beta
                new_beta = min(current_beta * 1.1, self.max_beta)
            else:
                new_beta = current_beta
            return {'beta': float(new_beta)}
        
        # 记录历史信息
        self._update_history(primal_res, dual_res)
        
        # 判断是否应该采用单调增长策略（针对含噪声项的回归问题）
        should_use_monotonic = self._should_use_monotonic_strategy(primal_res, dual_res, iteration)
        
        # 选择更新策略
        if should_use_monotonic:
            # 单调增长策略 - 针对回归问题
            new_beta = self._monotonic_update(current_beta, iteration, primal_res)
        elif iteration < 10:
            # 早期迭代：使用温和调整策略
            new_beta = self._early_stage_update(current_beta, primal_res, dual_res)
        elif self._is_stagnating(primal_res, dual_res):
            # 检测到停滞：采用激进的调整策略
            new_beta = self._stagnation_update(current_beta, primal_res, dual_res)
        else:
            # 标准自适应调整策略
            new_beta = self._standard_adaptive_update(current_beta, primal_res, dual_res)
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 更新最后残差值
        self.last_primal_res = primal_res
        self.last_dual_res = dual_res
        
        return {'beta': float(new_beta)}
    
    def _should_use_monotonic_strategy(self, primal_res: float, dual_res: float, 
                                     iteration: int) -> bool:
        """判断是否应该使用单调增长策略"""
        # 基于经验规则：如果残差较小但迭代次数较多，可能是回归问题
        if iteration > 20 and primal_res < self.regression_threshold:
            return True
        
        # 如果残差已经很小但算法仍在运行，采用单调策略可能有助于收敛
        if primal_res < 1e-3 and dual_res < 1e-3 and iteration > 30:
            return True
            
        return False
    
    def _monotonic_update(self, current_beta: float, iteration: int, 
                         primal_res: float) -> float:
        """单调增长更新策略 - 针对回归问题"""
        # 根据迭代阶段调整增长速率
        if iteration < 30:
            # 早期：中等增长
            growth_factor = self.monotonic_growth_factor
        elif iteration < 100:
            # 中期：较慢增长
            growth_factor = 1.2
        else:
            # 后期：非常慢的增长
            growth_factor = 1.05
        
        # 根据残差大小微调增长因子
        if primal_res > 0.1:
            growth_factor = min(growth_factor * 1.2, 2.5)
        elif primal_res < 1e-4:
            growth_factor = max(growth_factor * 0.9, 1.01)
        
        new_beta = current_beta * growth_factor
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
            early_mu = self.mu * 2.0
            
            if ratio > early_mu:
                # 原始残差相对较大，适度增大beta
                new_beta = current_beta * 1.5
            elif ratio < 1.0 / early_mu:
                # 对偶残差相对较大，适度减小beta
                new_beta = current_beta / 1.5
            else:
                # 残差平衡，保持当前beta
                new_beta = current_beta
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _is_stagnating(self, primal_res: float, dual_res: float) -> bool:
        """检测是否陷入停滞状态"""
        if len(self.primal_history) < 3:
            return False
        
        # 计算最近几次迭代的残差变化
        recent_primal = self.primal_history[-3:]
        recent_dual = self.dual_history[-3:]
        
        # 检查残差是否基本不变
        primal_change = max(recent_primal) / (min(recent_primal) + 1e-10) - 1.0
        dual_change = max(recent_dual) / (min(recent_dual) + 1e-10) - 1.0
        
        # 如果残差变化很小，增加停滞计数器
        if primal_change < 0.01 and dual_change < 0.01:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        return self.stagnation_counter >= self.max_stagnation
    
    def _stagnation_update(self, current_beta: float, primal_res: float,
                          dual_res: float) -> float:
        """停滞状态更新策略"""
        # 重置停滞计数器
        self.stagnation_counter = 0
        
        # 处理零除保护
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            if ratio > 5.0:
                # 原始残差明显更大，大幅增加beta
                new_beta = current_beta * 2.5
            elif ratio < 0.2:
                # 对偶残差明显更大，大幅减小beta
                new_beta = current_beta / 2.5
            else:
                # 尝试中等调整以跳出停滞
                adjustment = 1.8 if primal_res > 0.01 else 0.6
                new_beta = current_beta * adjustment
        else:
            # 默认调整
            new_beta = current_beta * 1.5
        
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
                    new_beta = current_beta * 1.1
                elif primal_res < 1e-4:
                    new_beta = current_beta * 0.95
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
            'monotonic_growth_factor': self.monotonic_growth_factor,
            'regression_threshold': self.regression_threshold,
            'max_history_len': self.max_history_len,
            'max_stagnation': self.max_stagnation
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        valid_params = [
            'initial_beta', 'min_beta', 'max_beta', 'mu', 'tau_inc', 'tau_dec',
            'monotonic_growth_factor', 'regression_threshold', 'max_history_len',
            'max_stagnation'
        ]
        
        for key in valid_params:
            if key in params:
                setattr(self, key, params[key])