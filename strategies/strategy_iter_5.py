from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class EnhancedBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础参数配置
        self.initial_beta = 1.0
        self.min_beta = 1e-6
        self.max_beta = 1e4
        
        # 基于历史成功策略的参数调整
        self.mu = 9.0           # 残差平衡阈值，略高于上一轮
        self.tau_inc = 1.8      # 增大因子
        self.tau_dec = 1.8      # 减小因子
        
        # 针对回归问题的优化参数
        self.regression_initial_beta = 0.8      # 初始beta略降低
        self.regression_growth_factor = 2.2     # 增长率提高
        self.regression_max_beta = 1e4
        self.regression_detection_threshold = 0.3  # 回归问题检测阈值
        
        # 低秩问题特殊处理
        self.low_rank_smoothing_factor = 1.2    # 更平滑的调整
        
        # 收敛监控
        self.primal_history = []
        self.dual_history = []
        self.max_history_len = 5
        
        # 状态跟踪
        self.iteration_count = 0
        self.last_primal_res = float('inf')
        self.last_dual_res = float('inf')
        self.problem_type = None  # 自动检测问题类型
        
        # 回归问题标记
        self.is_regression_problem = False
        self.regression_phase_started = False
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        converged = iteration_state.get('converged', False)
        objective = iteration_state.get('objective', None)
        
        self.iteration_count = iteration
        
        # 首次迭代初始化
        if iteration == 0:
            self.problem_type = self._detect_problem_type(iteration_state)
            if self.problem_type == "regression":
                self.is_regression_problem = True
                current_beta = self.regression_initial_beta
                return {'beta': float(current_beta)}
        
        # 处理None值
        if primal_res is None or dual_res is None:
            # 针对回归问题采用单调增长策略
            if self.is_regression_problem and iteration > 10:
                new_beta = self._regression_monotonic_update(current_beta, iteration)
                new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
                return {'beta': float(new_beta)}
            return {'beta': float(current_beta)}
        
        # 记录历史信息
        self._update_history(primal_res, dual_res)
        
        # 检测是否为回归问题（基于残差行为）
        if iteration > 20 and not self.is_regression_problem:
            if self._detect_regression_pattern(primal_res, dual_res, iteration):
                self.is_regression_problem = True
                self.regression_phase_started = True
        
        # 根据不同问题类型采用不同策略
        if self.is_regression_problem:
            # 回归问题：采用单调递增策略
            if iteration > 50 and primal_res > 0.1:
                # 如果残差仍然较大，采用更激进的增长
                growth_factor = min(self.regression_growth_factor * 1.3, 3.0)
                new_beta = current_beta * growth_factor
            else:
                new_beta = self._regression_monotonic_update(current_beta, iteration)
        elif self.problem_type == "low_rank":
            # 低秩矩阵问题：平滑调整
            new_beta = self._low_rank_update(current_beta, primal_res, dual_res)
        elif self.problem_type == "multi_view":
            # 多视图聚类问题：平衡调整
            new_beta = self._multi_view_update(current_beta, primal_res, dual_res)
        elif self.problem_type == "tracelasso":
            # Trace Lasso问题：考虑相关性
            new_beta = self._tracelasso_update(current_beta, primal_res, dual_res)
        else:
            # 标准稀疏优化问题：自适应调整
            new_beta = self._standard_adaptive_update(current_beta, primal_res, dual_res)
        
        # 额外处理：针对长时间未收敛的情况
        if iteration > 100 and not converged:
            if primal_res > 0.05 or dual_res > 0.05:
                # 如果残差仍然较大，尝试更激进的调整
                if primal_res > dual_res * 5:
                    new_beta = min(new_beta * 2.5, self.max_beta)
                elif dual_res > primal_res * 5:
                    new_beta = max(new_beta / 2.5, self.min_beta)
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 更新最后残差值
        self.last_primal_res = primal_res
        self.last_dual_res = dual_res
        
        return {'beta': float(new_beta)}
    
    def _detect_problem_type(self, iteration_state: Dict[str, Any]) -> str:
        """尝试检测问题类型"""
        # 基于可用信息和启发式方法
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        
        # 这里只是简单的检测，实际可能需要更多信息
        # 返回默认类型，让后续基于残差行为调整
        return "standard"
    
    def _detect_regression_pattern(self, primal_res: float, dual_res: float, 
                                 iteration: int) -> bool:
        """检测回归问题的残差模式"""
        # 回归问题通常有噪声项，原始残差下降缓慢
        if iteration > 30:
            # 检查残差比例
            if dual_res > 1e-10:
                ratio = primal_res / dual_res
                # 如果原始残差相对较小但对偶残差较大且下降缓慢
                if primal_res < 0.1 and ratio < 0.2 and iteration > 50:
                    return True
            
            # 检查历史下降趋势
            if len(self.primal_history) >= 3:
                recent_primal = self.primal_history[-3:]
                primal_decline = (recent_primal[0] - recent_primal[-1]) / max(recent_primal[0], 1e-10)
                
                if primal_decline < 0.01 and primal_res > 0.05:
                    return True
        
        return False
    
    def _regression_monotonic_update(self, current_beta: float, iteration: int) -> float:
        """回归问题的单调增长更新策略"""
        # 根据问题特性说明：初始beta=0.1~1.0，增长率1.1~1.5，上限1e4
        if iteration < 20:
            # 早期：温和增长
            growth = 1.2
        elif iteration < 100:
            # 中期：中等增长
            growth = self.regression_growth_factor
        elif iteration < 300:
            # 后期：缓慢增长
            growth = 1.1
        else:
            # 接近上限：微调
            growth = 1.05
        
        new_beta = current_beta * growth
        
        # 确保不超过上限
        return min(new_beta, self.regression_max_beta)
    
    def _low_rank_update(self, current_beta: float, primal_res: float, 
                        dual_res: float) -> float:
        """低秩矩阵问题的平滑调整策略"""
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 使用更平滑的调整策略
            if ratio > self.mu * 1.5:
                # 原始残差较大，适度增加
                new_beta = current_beta * self.low_rank_smoothing_factor
            elif ratio < 1.0 / (self.mu * 1.5):
                # 对偶残差较大，适度减少
                new_beta = current_beta / self.low_rank_smoothing_factor
            else:
                # 残差平衡，微调
                if primal_res > 0.1:
                    new_beta = current_beta * 1.05
                elif primal_res < 1e-4:
                    new_beta = current_beta * 0.95
                else:
                    new_beta = current_beta
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _multi_view_update(self, current_beta: float, primal_res: float,
                          dual_res: float) -> float:
        """多视图聚类问题的平衡调整策略"""
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 需要协调多个视图的平衡，采用保守调整
            if ratio > self.mu * 2:
                new_beta = current_beta * 1.5
            elif ratio < 1.0 / (self.mu * 2):
                new_beta = current_beta / 1.5
            else:
                # 在平衡范围内，根据残差绝对值调整
                avg_res = (primal_res + dual_res) / 2
                if avg_res > 0.2:
                    new_beta = current_beta * 1.2
                elif avg_res < 0.01:
                    new_beta = current_beta * 0.9
                else:
                    new_beta = current_beta
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _tracelasso_update(self, current_beta: float, primal_res: float,
                          dual_res: float) -> float:
        """Trace Lasso问题的相关性感知调整"""
        epsilon = 1e-10
        primal_safe = max(primal_res, epsilon)
        dual_safe = max(dual_res, epsilon)
        
        if dual_safe > epsilon:
            ratio = primal_safe / dual_safe
            
            # 考虑设计矩阵相关性，采用适度调整
            if ratio > self.mu * 1.8:
                new_beta = current_beta * 1.6
            elif ratio < 1.0 / (self.mu * 1.8):
                new_beta = current_beta / 1.6
            else:
                # 根据残差绝对值和历史趋势调整
                if primal_res > 0.15:
                    new_beta = current_beta * 1.15
                elif primal_res < 0.005:
                    new_beta = current_beta * 0.85
                else:
                    new_beta = current_beta
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _standard_adaptive_update(self, current_beta: float, primal_res: float,
                                dual_res: float) -> float:
        """标准自适应更新策略"""
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
                elif primal_res < 1e-3:
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
            'regression_initial_beta': self.regression_initial_beta,
            'regression_growth_factor': self.regression_growth_factor,
            'regression_max_beta': self.regression_max_beta,
            'regression_detection_threshold': self.regression_detection_threshold,
            'low_rank_smoothing_factor': self.low_rank_smoothing_factor,
            'max_history_len': self.max_history_len
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        valid_params = [
            'initial_beta', 'min_beta', 'max_beta', 'mu', 'tau_inc', 'tau_dec',
            'regression_initial_beta', 'regression_growth_factor', 'regression_max_beta',
            'regression_detection_threshold', 'low_rank_smoothing_factor', 'max_history_len'
        ]
        
        for key in valid_params:
            if key in params:
                setattr(self, key, params[key])