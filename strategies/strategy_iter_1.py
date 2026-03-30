from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class AdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础超参数配置
        self.min_beta = 1e-6        # beta 最小值
        self.max_beta = 1e4         # beta 最大值
        self.mu = 2.0               # 残差平衡阈值
        self.tau_inc = 1.5          # beta 增大因子
        self.tau_dec = 1.5          # beta 减小因子
        
        # 自适应调整参数
        self.initial_beta = 1.0     # 初始 beta 值
        self.beta_increase_factor = 1.2  # 对于回归问题的固定增长因子
        self.adaptive_mode = 'residual'  # 自适应模式：'residual', 'regression', 'low_rank'
        
        # 历史信息记录
        self.primal_history = []
        self.dual_history = []
        self.max_history_size = 10
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从 iteration_state 获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        current_beta = iteration_state.get('beta', self.initial_beta)
        converged = iteration_state.get('converged', False)
        
        # 初始化 beta（第一次迭代）
        if iteration == 0:
            new_beta = self.initial_beta
        elif converged:
            new_beta = current_beta
        else:
            # 更新历史记录
            self._update_history(primal_res, dual_res)
            
            # 根据自适应模式选择调整策略
            if self.adaptive_mode == 'regression':
                new_beta = self._regression_strategy(current_beta)
            elif self.adaptive_mode == 'low_rank':
                new_beta = self._low_rank_strategy(current_beta, iteration)
            else:  # 默认基于残差的自适应策略
                new_beta = self._residual_based_strategy(current_beta, primal_res, dual_res)
        
        # 限制 beta 范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        return {'beta': float(new_beta)}
    
    def _update_history(self, primal_res: float, dual_res: float) -> None:
        """更新残差历史记录"""
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        
        # 保持历史记录大小
        if len(self.primal_history) > self.max_history_size:
            self.primal_history.pop(0)
            self.dual_history.pop(0)
    
    def _residual_based_strategy(self, current_beta: float, 
                                 primal_res: float, dual_res: float) -> float:
        """基于残差比的自适应调整策略"""
        # 处理 None 值或极小的对偶残差
        if primal_res is None or dual_res is None or dual_res < 1e-12:
            return current_beta
        
        # 计算残差比
        ratio = primal_res / (dual_res + 1e-12)
        
        # 基于残差比调整 beta
        if ratio > self.mu:
            # 原始残差过大，增加惩罚强度
            new_beta = current_beta * self.tau_inc
        elif ratio < 1.0 / self.mu:
            # 对偶残差过大，减少惩罚强度
            new_beta = current_beta / self.tau_dec
        else:
            # 残差平衡，保持当前 beta
            new_beta = current_beta
        
        # 使用历史信息平滑调整
        if len(self.primal_history) >= 3:
            primal_trend = self._calculate_trend(self.primal_history)
            dual_trend = self._calculate_trend(self.dual_history)
            
            # 如果原始残差持续下降而对偶残差上升，略微增加 beta
            if primal_trend < 0 and dual_trend > 0:
                new_beta = current_beta * 1.1
            # 如果对偶残差持续下降而原始残差上升，略微减少 beta
            elif dual_trend < 0 and primal_trend > 0:
                new_beta = current_beta * 0.9
        
        return new_beta
    
    def _regression_strategy(self, current_beta: float) -> float:
        """回归问题的单调递增策略"""
        # beta 只增不减，适用于含噪声项的回归问题
        new_beta = current_beta * self.beta_increase_factor
        return new_beta
    
    def _low_rank_strategy(self, current_beta: float, iteration: int) -> float:
        """低秩问题的平滑调整策略"""
        # 早期迭代：缓慢增加以建立良好的初始解
        if iteration < 20:
            new_beta = current_beta * 1.05
        # 中期迭代：根据收敛情况调整
        elif iteration < 100:
            if len(self.primal_history) >= 5:
                # 检查原始残差的变化趋势
                primal_std = np.std(self.primal_history[-5:])
                if primal_std < 0.01:  # 原始残差稳定
                    new_beta = current_beta * 1.02
                else:  # 原始残差波动较大
                    new_beta = current_beta * 1.1
            else:
                new_beta = current_beta * 1.1
        # 后期迭代：稳定 beta 以精细调整
        else:
            new_beta = current_beta
        
        return new_beta
    
    def _calculate_trend(self, history: list) -> float:
        """计算历史数据的趋势（斜率）"""
        if len(history) < 2:
            return 0.0
        
        # 使用简单线性回归计算趋势
        x = np.arange(len(history))
        y = np.array(history)
        
        # 归一化 x 以避免数值问题
        x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-12)
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-12)
        
        # 计算斜率
        if len(history) > 1:
            slope = np.sum(x_normalized * y_normalized) / np.sum(x_normalized**2)
        else:
            slope = 0.0
        
        return slope
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'initial_beta': self.initial_beta,
            'beta_increase_factor': self.beta_increase_factor,
            'adaptive_mode': self.adaptive_mode,
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
        if 'beta_increase_factor' in params:
            self.beta_increase_factor = params['beta_increase_factor']
        if 'adaptive_mode' in params:
            self.adaptive_mode = params['adaptive_mode']
        if 'max_history_size' in params:
            self.max_history_size = params['max_history_size']