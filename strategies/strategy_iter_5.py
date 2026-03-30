from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class AdaptiveBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基于历史成功策略的参数配置
        self.initial_beta = 0.8                # 初始beta值
        self.min_beta = 1e-8                   # beta下界
        self.max_beta = 1e4                    # beta上界
        self.mu = 6.8                          # 残差平衡阈值，基于历史成功策略微调
        self.tau_inc = 1.32                    # 增大因子，比历史成功策略略低
        self.tau_dec = 1.22                    # 减小因子，比历史成功策略略高
        
        # 收敛检测参数
        self.primal_tol = 1e-4                 # 原始残差容差
        self.dual_tol = 1e-4                   # 对偶残差容差
        self.no_improvement_window = 5         # 无改进检测窗口
        
        # 自适应调整参数
        self.last_primal_res = None            # 上一次原始残差
        self.last_dual_res = None              # 上一次对偶残差
        self.beta_history = []                 # beta历史记录
        self.primal_history = []               # 原始残差历史
        self.dual_history = []                 # 对偶残差历史
        self.convergence_counter = 0           # 收敛计数器
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        converged = iteration_state.get('converged', False)
        
        # 处理None值或已收敛情况
        if primal_res is None or dual_res is None or converged:
            return {'beta': current_beta}
        
        # 初始化检查
        if iteration == 0:
            self.last_primal_res = primal_res
            self.last_dual_res = dual_res
            self.beta_history = [current_beta]
            self.primal_history = [primal_res]
            self.dual_history = [dual_res]
            return {'beta': current_beta}
        
        # 更新历史记录
        self.beta_history.append(current_beta)
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        
        # 保持历史记录长度
        max_history = 30
        if len(self.beta_history) > max_history:
            self.beta_history.pop(0)
            self.primal_history.pop(0)
            self.dual_history.pop(0)
        
        # 基本自适应策略：基于残差比调整
        new_beta = current_beta
        
        if primal_res > 1e-12 and dual_res > 1e-12:
            ratio = primal_res / dual_res
            
            # 计算残差变化的相对量
            if self.last_primal_res is not None and self.last_dual_res is not None:
                primal_change = abs(primal_res - self.last_primal_res) / (self.last_primal_res + 1e-12)
                dual_change = abs(dual_res - self.last_dual_res) / (self.last_dual_res + 1e-12)
                
                # 如果残差变化剧烈，采用更温和的调整
                if primal_change > 0.5 or dual_change > 0.5:
                    # 变化剧烈，采用更温和的调整因子
                    tau_inc = 1.0 + (self.tau_inc - 1.0) * 0.7
                    tau_dec = 1.0 + (self.tau_dec - 1.0) * 0.7
                else:
                    tau_inc = self.tau_inc
                    tau_dec = self.tau_dec
            else:
                tau_inc = self.tau_inc
                tau_dec = self.tau_dec
            
            # 标准残差比调整逻辑
            if ratio > self.mu:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / self.mu:
                new_beta = current_beta / tau_dec
            else:
                # 残差平衡，保持beta不变
                new_beta = current_beta
        
        # 收敛加速策略
        if iteration > 10:
            # 检测是否在收敛
            if len(self.primal_history) >= 3 and len(self.dual_history) >= 3:
                recent_primal = self.primal_history[-3:]
                recent_dual = self.dual_history[-3:]
                
                # 检查是否在下降
                primal_decreasing = all(recent_primal[i] <= recent_primal[i-1] * 1.05 for i in range(1, 3))
                dual_decreasing = all(recent_dual[i] <= recent_dual[i-1] * 1.05 for i in range(1, 3))
                
                if primal_decreasing and dual_decreasing:
                    self.convergence_counter += 1
                else:
                    self.convergence_counter = max(0, self.convergence_counter - 1)
                
                # 如果连续多次迭代在收敛，调整策略
                if self.convergence_counter >= 3:
                    # 收敛趋势良好，轻微调整以加速
                    if primal_res > dual_res * 1.5:
                        new_beta = min(new_beta * 1.05, self.max_beta)
                    elif dual_res > primal_res * 1.5:
                        new_beta = max(new_beta / 1.05, self.min_beta)
        
        # 无改进检测
        if iteration > 20:
            if len(self.primal_history) >= self.no_improvement_window + 1:
                # 检查最近几次迭代是否没有改进
                window_start = -self.no_improvement_window - 1
                window_end = -1
                
                recent_primal = self.primal_history[window_start:window_end]
                recent_dual = self.dual_history[window_start:window_end]
                
                # 计算平均改进
                primal_improvement = sum(recent_primal[i] - recent_primal[i+1] for i in range(len(recent_primal)-1))
                dual_improvement = sum(recent_dual[i] - recent_dual[i+1] for i in range(len(recent_dual)-1))
                
                # 如果没有改进，尝试调整beta
                if primal_improvement < 0 and dual_improvement < 0:
                    if primal_res > dual_res * 3:
                        new_beta = min(new_beta * 1.5, self.max_beta)
                    elif dual_res > primal_res * 3:
                        new_beta = max(new_beta / 1.5, self.min_beta)
        
        # 早期迭代稳定性控制
        if iteration < 5:
            # 限制早期变化幅度
            max_early_change = 1.25
            if new_beta > current_beta * max_early_change:
                new_beta = current_beta * max_early_change
            elif new_beta < current_beta / max_early_change:
                new_beta = current_beta / max_early_change
        
        # 限制beta范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 更新上一次残差
        self.last_primal_res = primal_res
        self.last_dual_res = dual_res
        
        return {'beta': float(new_beta)}
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'initial_beta': self.initial_beta,
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'primal_tol': self.primal_tol,
            'dual_tol': self.dual_tol,
            'no_improvement_window': self.no_improvement_window
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        if 'initial_beta' in params:
            self.initial_beta = params['initial_beta']
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
        if 'primal_tol' in params:
            self.primal_tol = params['primal_tol']
        if 'dual_tol' in params:
            self.dual_tol = params['dual_tol']
        if 'no_improvement_window' in params:
            self.no_improvement_window = params['no_improvement_window']