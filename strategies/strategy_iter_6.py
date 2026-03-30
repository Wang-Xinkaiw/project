from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedBetaStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基于历史成功策略的优化参数配置
        self.initial_beta = 0.7                # 初始beta值，略低于历史成功值
        self.min_beta = 1e-8                   # beta下界
        self.max_beta = 1e4                    # beta上界
        self.mu = 6.6                          # 残差平衡阈值，比历史成功值略低
        self.tau_inc = 1.28                    # 增大因子，比历史成功策略略低
        self.tau_dec = 1.24                    # 减小因子，比历史成功策略略高
        
        # 收敛检测参数
        self.primal_tol = 1e-4                 # 原始残差容差
        self.dual_tol = 1e-4                   # 对偶残差容差
        self.stagnation_window = 8             # 停滞检测窗口，比历史成功策略略长
        
        # 自适应调整参数
        self.last_primal_res = None            # 上一次原始残差
        self.last_dual_res = None              # 上一次对偶残差
        self.beta_history = []                 # beta历史记录
        self.primal_history = []               # 原始残差历史
        self.dual_history = []                 # 对偶残差历史
        self.stagnation_counter = 0            # 停滞计数器
        self.iteration_since_last_adjust = 0   # 上次调整后的迭代次数
        
        # 问题类型自适应参数
        self.last_objective = None             # 上一次目标函数值
        self.objective_history = []            # 目标函数历史
        self.objective_improvement_threshold = 1e-6  # 目标函数改进阈值
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        current_beta = iteration_state.get('beta', self.initial_beta)
        converged = iteration_state.get('converged', False)
        objective = iteration_state.get('objective', None)
        
        # 处理None值或已收敛情况
        if primal_res is None or dual_res is None or converged:
            return {'beta': current_beta}
        
        # 初始化检查
        if iteration == 0:
            self.last_primal_res = primal_res
            self.last_dual_res = dual_res
            self.last_objective = objective
            self.beta_history = [current_beta]
            self.primal_history = [primal_res]
            self.dual_history = [dual_res]
            if objective is not None:
                self.objective_history = [objective]
            return {'beta': current_beta}
        
        # 更新历史记录
        self.beta_history.append(current_beta)
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        if objective is not None:
            self.objective_history.append(objective)
        
        # 保持历史记录长度
        max_history = 25
        if len(self.beta_history) > max_history:
            self.beta_history.pop(0)
            self.primal_history.pop(0)
            self.dual_history.pop(0)
        if len(self.objective_history) > max_history:
            self.objective_history.pop(0)
        
        # 基本自适应策略：基于残差比调整
        new_beta = current_beta
        self.iteration_since_last_adjust += 1
        
        # 检测残差是否为有效数值
        if primal_res > 1e-12 and dual_res > 1e-12:
            ratio = primal_res / dual_res
            
            # 计算残差变化的相对量
            if self.last_primal_res is not None and self.last_dual_res is not None:
                primal_change = abs(primal_res - self.last_primal_res) / (self.last_primal_res + 1e-12)
                dual_change = abs(dual_res - self.last_dual_res) / (self.last_dual_res + 1e-12)
                
                # 根据变化幅度调整调整因子
                if primal_change > 0.6 or dual_change > 0.6:
                    # 变化剧烈，采用更温和的调整因子
                    tau_inc = 1.0 + (self.tau_inc - 1.0) * 0.65
                    tau_dec = 1.0 + (self.tau_dec - 1.0) * 0.65
                elif primal_change < 0.05 and dual_change < 0.05:
                    # 变化很小，采用稍强的调整因子
                    tau_inc = 1.0 + (self.tau_inc - 1.0) * 1.1
                    tau_dec = 1.0 + (self.tau_dec - 1.0) * 1.1
                else:
                    tau_inc = self.tau_inc
                    tau_dec = self.tau_dec
            else:
                tau_inc = self.tau_inc
                tau_dec = self.tau_dec
            
            # 标准残差比调整逻辑
            if ratio > self.mu:
                new_beta = current_beta * tau_inc
                self.iteration_since_last_adjust = 0
            elif ratio < 1.0 / self.mu:
                new_beta = current_beta / tau_dec
                self.iteration_since_last_adjust = 0
            else:
                # 残差平衡，保持beta不变
                new_beta = current_beta
        
        # 目标函数改进检测（如果可用）
        if objective is not None and self.last_objective is not None:
            objective_improvement = abs(objective - self.last_objective) / (abs(self.last_objective) + 1e-12)
            
            # 如果目标函数改进很小，考虑调整beta
            if objective_improvement < self.objective_improvement_threshold and iteration > 10:
                if primal_res > dual_res * 1.8:
                    new_beta = min(current_beta * 1.1, self.max_beta)
                    self.iteration_since_last_adjust = 0
                elif dual_res > primal_res * 1.8:
                    new_beta = max(current_beta / 1.1, self.min_beta)
                    self.iteration_since_last_adjust = 0
            
            self.last_objective = objective
        
        # 停滞检测和恢复策略
        if iteration > 15:
            if len(self.primal_history) >= self.stagnation_window + 1:
                # 检查最近几次迭代是否停滞
                window_start = -self.stagnation_window - 1
                window_end = -1
                
                recent_primal = self.primal_history[window_start:window_end]
                recent_dual = self.dual_history[window_start:window_end]
                
                # 计算平均变化
                primal_change_avg = sum(abs(recent_primal[i] - recent_primal[i+1]) for i in range(len(recent_primal)-1)) / (len(recent_primal)-1)
                dual_change_avg = sum(abs(recent_dual[i] - recent_dual[i+1]) for i in range(len(recent_dual)-1)) / (len(recent_dual)-1)
                
                primal_initial = recent_primal[0]
                primal_final = recent_primal[-1]
                dual_initial = recent_dual[0]
                dual_final = recent_dual[-1]
                
                # 检查是否停滞（变化很小）
                primal_stagnant = abs(primal_final - primal_initial) < primal_initial * 0.05 and primal_change_avg < primal_initial * 0.02
                dual_stagnant = abs(dual_final - dual_initial) < dual_initial * 0.05 and dual_change_avg < dual_initial * 0.02
                
                if primal_stagnant and dual_stagnant:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = max(0, self.stagnation_counter - 1)
                
                # 如果停滞超过阈值，进行较大调整
                if self.stagnation_counter >= 2 and self.iteration_since_last_adjust > 3:
                    if primal_res > dual_res:
                        new_beta = min(current_beta * 1.4, self.max_beta)
                    else:
                        new_beta = max(current_beta / 1.4, self.min_beta)
                    self.stagnation_counter = 0
                    self.iteration_since_last_adjust = 0
        
        # 早期迭代稳定性控制
        if iteration < 6:
            # 限制早期变化幅度
            max_early_change = 1.3
            if new_beta > current_beta * max_early_change:
                new_beta = current_beta * max_early_change
            elif new_beta < current_beta / max_early_change:
                new_beta = current_beta / max_early_change
        
        # 中期迭代的温和调整
        elif 6 <= iteration < 20:
            # 限制中期变化幅度
            max_mid_change = 1.45
            if new_beta > current_beta * max_mid_change:
                new_beta = current_beta * max_mid_change
            elif new_beta < current_beta / max_mid_change:
                new_beta = current_beta / max_mid_change
        
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
            'stagnation_window': self.stagnation_window,
            'objective_improvement_threshold': self.objective_improvement_threshold
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
        if 'stagnation_window' in params:
            self.stagnation_window = params['stagnation_window']
        if 'objective_improvement_threshold' in params:
            self.objective_improvement_threshold = params['objective_improvement_threshold']