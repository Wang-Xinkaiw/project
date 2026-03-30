from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class EnhancedBetaStrategyV2(BaseTuningStrategy):
    def __init__(self):
        # 基础参数配置 - 基于历史成功策略微调
        self.initial_beta = 0.8
        self.min_beta = 1e-6
        self.max_beta = 1e4
        
        # 自适应调整参数 - 在成功策略基础上微调
        self.mu = 9.2           # 残差平衡阈值，历史成功策略平均8.9，略微上调
        self.tau_inc = 1.75     # 增大因子，略微下调以获得更平滑调整
        self.tau_dec = 1.85     # 减小因子，略微上调以更快响应对偶残差过大
        
        # 回归问题专用参数
        self.regression_initial_beta = 0.5      # 降低初始值以更好适应噪声
        self.regression_growth_factor = 1.4     # 使用建议范围中值
        self.regression_max_beta = 1e4
        
        # 低秩问题特殊处理 - 修复运行时错误
        self.low_rank_smoothing_factor = 1.15   # 更平滑的调整
        self.low_rank_min_beta = 0.1           # 设置下限避免过小
        self.low_rank_max_beta = 5e3           # 设置上限避免过大
        
        # 收敛监控和历史跟踪
        self.primal_history = []
        self.dual_history = []
        self.max_history_len = 8              # 增加历史长度以更好检测趋势
        
        # 状态跟踪
        self.iteration_count = 0
        self.last_primal_res = float('inf')
        self.last_dual_res = float('inf')
        self.stuck_counter = 0                # 记录停滞次数
        self.last_beta_change = 0             # 记录上次beta变化时的迭代
        self.beta_increase_count = 0          # beta增加次数
        
        # 问题类型检测相关（避免运行时错误）
        self.problem_type_hint = None
        self.regression_detected = False
        self.low_rank_detected = False
        
        # 收敛加速参数
        self.acceleration_factor = 1.0        # 加速因子
        self.acceleration_threshold = 100     # 启动加速的迭代阈值
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        converged = iteration_state.get('converged', False)
        
        self.iteration_count = iteration
        
        # 首次迭代初始化
        if iteration == 0:
            # 重置所有状态
            self.reset_state()
            current_beta = self.initial_beta
            return {'beta': float(current_beta)}
        
        # 如果已经收敛，保持当前beta
        if converged:
            return {'beta': float(current_beta)}
        
        # 处理None值或无效值
        if primal_res is None or dual_res is None or primal_res < 0 or dual_res < 0:
            # 使用保守策略
            if iteration > 50:
                new_beta = current_beta * 1.1
            else:
                new_beta = current_beta
            new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
            return {'beta': float(new_beta)}
        
        # 记录历史信息
        self._update_history(primal_res, dual_res)
        
        # 基于残差行为检测问题类型（避免运行时错误）
        self._detect_problem_type_from_residuals(primal_res, dual_res, iteration)
        
        # 计算残差比（避免除零）
        epsilon = 1e-12
        dual_res_safe = max(dual_res, epsilon)
        primal_dual_ratio = primal_res / dual_res_safe
        
        # 核心自适应调整策略
        if self.regression_detected:
            # 回归问题：单调递增策略
            new_beta = self._regression_update(current_beta, primal_res, dual_res, iteration)
        elif self.low_rank_detected:
            # 低秩问题：平滑调整策略
            new_beta = self._low_rank_safe_update(current_beta, primal_res, dual_res, primal_dual_ratio)
        else:
            # 标准自适应策略
            new_beta = self._standard_adaptive_update(current_beta, primal_res, dual_res, primal_dual_ratio)
        
        # 加速策略：如果迭代次数较多但尚未收敛，尝试加速
        if iteration > self.acceleration_threshold and not converged:
            if primal_res > 0.05 or dual_res > 0.05:
                # 分析最近迭代的收敛速度
                convergence_speed = self._analyze_convergence_speed()
                if convergence_speed < 0.01:  # 收敛缓慢
                    # 尝试更激进的调整
                    if primal_res > dual_res * 3:
                        new_beta = min(new_beta * 1.8, self.max_beta)
                    elif dual_res > primal_res * 3:
                        new_beta = max(new_beta / 1.8, self.min_beta)
        
        # 避免beta频繁震荡
        new_beta = self._stabilize_beta(current_beta, new_beta, iteration)
        
        # 限制beta范围
        if self.low_rank_detected:
            new_beta = np.clip(new_beta, self.low_rank_min_beta, self.low_rank_max_beta)
        else:
            new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 更新状态跟踪
        self._update_tracking_state(current_beta, new_beta, iteration)
        
        return {'beta': float(new_beta)}
    
    def reset_state(self):
        """重置所有状态变量"""
        self.primal_history = []
        self.dual_history = []
        self.iteration_count = 0
        self.last_primal_res = float('inf')
        self.last_dual_res = float('inf')
        self.stuck_counter = 0
        self.last_beta_change = 0
        self.beta_increase_count = 0
        self.problem_type_hint = None
        self.regression_detected = False
        self.low_rank_detected = False
        self.acceleration_factor = 1.0
    
    def _detect_problem_type_from_residuals(self, primal_res: float, dual_res: float, iteration: int):
        """基于残差行为检测问题类型（安全版本）"""
        if iteration < 20:
            return  # 早期迭代，信息不足
        
        # 检测回归问题模式：原始残差下降缓慢，对偶残差相对较大
        if len(self.primal_history) >= 10:
            recent_primal = self.primal_history[-10:]
            recent_dual = self.dual_history[-10:]
            
            # 计算最近残差的平均下降率
            primal_decline_rate = (recent_primal[0] - recent_primal[-1]) / max(recent_primal[0], 1e-10)
            dual_decline_rate = (recent_dual[0] - recent_dual[-1]) / max(recent_dual[0], 1e-10)
            
            # 回归问题检测条件
            if (primal_decline_rate < 0.05 and primal_res > 0.1 and 
                dual_res > primal_res * 2 and iteration > 50):
                self.regression_detected = True
                self.problem_type_hint = "regression"
        
        # 检测低秩问题模式：残差波动较小，收敛较平稳
        if len(self.primal_history) >= 15:
            primal_std = np.std(self.primal_history[-15:])
            dual_std = np.std(self.dual_history[-15:])
            
            if primal_std < 0.01 and dual_std < 0.01 and primal_res < 0.5:
                self.low_rank_detected = True
                self.problem_type_hint = "low_rank"
    
    def _regression_update(self, current_beta: float, primal_res: float, 
                          dual_res: float, iteration: int) -> float:
        """回归问题的单调增长更新策略（安全版本）"""
        # 根据迭代阶段调整增长率
        if iteration < 30:
            growth = 1.2  # 早期温和增长
        elif iteration < 150:
            growth = self.regression_growth_factor  # 中期标准增长
        elif iteration < 300:
            growth = 1.15  # 后期缓慢增长
        else:
            growth = 1.05  # 末尾微调
        
        # 如果原始残差仍然较大，稍微加快增长
        if primal_res > 0.2 and iteration > 100:
            growth = min(growth * 1.1, 1.8)
        
        new_beta = current_beta * growth
        
        # 确保不超过上限
        return min(new_beta, self.regression_max_beta)
    
    def _low_rank_safe_update(self, current_beta: float, primal_res: float, 
                             dual_res: float, primal_dual_ratio: float) -> float:
        """低秩矩阵问题的平滑调整策略（安全版本）"""
        # 确保参数在合理范围内
        mu_adjusted = self.mu * 1.2  # 对低秩问题使用更宽松的阈值
        
        if primal_dual_ratio > mu_adjusted * 1.3:
            # 原始残差较大，适度增加
            new_beta = current_beta * self.low_rank_smoothing_factor
        elif primal_dual_ratio < 1.0 / (mu_adjusted * 1.3):
            # 对偶残差较大，适度减少
            new_beta = current_beta / self.low_rank_smoothing_factor
        else:
            # 残差平衡，微调
            if primal_res > 0.08:
                new_beta = current_beta * 1.08
            elif primal_res < 1e-4:
                new_beta = current_beta * 0.92
            else:
                new_beta = current_beta
        
        return new_beta
    
    def _standard_adaptive_update(self, current_beta: float, primal_res: float,
                                 dual_res: float, primal_dual_ratio: float) -> float:
        """标准自适应更新策略"""
        # 使用配置的阈值和调整因子
        if primal_dual_ratio > self.mu:
            # 原始残差相对较大，增大beta
            new_beta = current_beta * self.tau_inc
            self.beta_increase_count += 1
        elif primal_dual_ratio < 1.0 / self.mu:
            # 对偶残差相对较大，减小beta
            new_beta = current_beta / self.tau_dec
        else:
            # 残差平衡时，根据残差绝对值微调
            if primal_res > 0.1:
                new_beta = current_beta * 1.08
            elif primal_res < 1e-3:
                new_beta = current_beta * 0.92
            else:
                new_beta = current_beta
        
        return new_beta
    
    def _analyze_convergence_speed(self) -> float:
        """分析最近迭代的收敛速度"""
        if len(self.primal_history) < 5:
            return 1.0  # 默认值
        
        # 计算最近5次迭代的原始残差平均下降率
        recent_primal = self.primal_history[-5:]
        total_decline = 0
        count = 0
        
        for i in range(1, len(recent_primal)):
            if recent_primal[i-1] > 1e-10:
                decline = (recent_primal[i-1] - recent_primal[i]) / recent_primal[i-1]
                total_decline += max(decline, 0)  # 只考虑下降
                count += 1
        
        return total_decline / max(count, 1)
    
    def _stabilize_beta(self, current_beta: float, new_beta: float, iteration: int) -> float:
        """稳定beta调整，避免频繁震荡"""
        # 如果beta变化很小，保持原值
        if abs(new_beta - current_beta) / max(current_beta, 1e-10) < 0.05:
            return current_beta
        
        # 限制beta变化频率
        if iteration - self.last_beta_change < 3:
            # 最近刚调整过，保持原值或微调
            return current_beta * 1.05 if new_beta > current_beta else current_beta * 0.95
        
        # 如果beta频繁上下震荡，采用更保守的策略
        if self.stuck_counter > 5:
            # 使用几何平均平滑
            smoothed_beta = np.sqrt(current_beta * new_beta)
            return smoothed_beta
        
        return new_beta
    
    def _update_history(self, primal_res: float, dual_res: float) -> None:
        """更新历史记录"""
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        
        # 保持历史记录长度
        if len(self.primal_history) > self.max_history_len:
            self.primal_history.pop(0)
            self.dual_history.pop(0)
    
    def _update_tracking_state(self, current_beta: float, new_beta: float, iteration: int):
        """更新状态跟踪变量"""
        # 更新beta变化记录
        if abs(new_beta - current_beta) / max(current_beta, 1e-10) > 0.1:
            self.last_beta_change = iteration
        
        # 更新停滞计数器
        if len(self.primal_history) >= 3:
            recent_primal = self.primal_history[-3:]
            if max(recent_primal) - min(recent_primal) < 0.01:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
    
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
            'low_rank_smoothing_factor': self.low_rank_smoothing_factor,
            'low_rank_min_beta': self.low_rank_min_beta,
            'low_rank_max_beta': self.low_rank_max_beta,
            'max_history_len': self.max_history_len,
            'acceleration_threshold': self.acceleration_threshold
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        valid_params = [
            'initial_beta', 'min_beta', 'max_beta', 'mu', 'tau_inc', 'tau_dec',
            'regression_initial_beta', 'regression_growth_factor', 'regression_max_beta',
            'low_rank_smoothing_factor', 'low_rank_min_beta', 'low_rank_max_beta',
            'max_history_len', 'acceleration_threshold'
        ]
        
        for key in valid_params:
            if key in params:
                setattr(self, key, params[key])