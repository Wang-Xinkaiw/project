from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class EnhancedADMMStrategy(BaseTuningStrategy):
    def __init__(self):
        # 基础超参数配置
        self.min_beta = 1e-3        # beta 最小值 - 提高以增加惩罚强度
        self.max_beta = 1e4         # beta 最大值
        self.mu = 2.0               # 残差平衡阈值 - 保持成功策略设置
        self.tau_inc = 2.5          # beta 增大因子 - 增大以加速收敛
        self.tau_dec = 1.3          # beta 减小因子 - 微调
        
        # 问题特定策略参数
        self.initial_beta = 1.0     # 初始 beta 值
        self.regression_growth_factor = 2.0  # 回归问题的beta增长因子
        self.low_rank_growth_factor = 1.2    # 低秩问题的增长因子
        
        # 收敛监控和自适应调整
        self.primal_history = []
        self.dual_history = []
        self.max_history_size = 5   # 减小历史记录大小
        
        # 停滞检测
        self.stagnation_count = 0
        self.stagnation_threshold = 10  # 增加停滞阈值
        
        # 问题类型推断
        self.problem_type = None
        self.infer_problem_type = True  # 启用问题类型推断
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从 iteration_state 获取状态信息
        iteration = iteration_state.get('iteration', 0)
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        current_beta = iteration_state.get('beta', self.initial_beta)
        objective = iteration_state.get('objective', 0.0)
        converged = iteration_state.get('converged', False)
        
        # 初始化 beta（第一次迭代）
        if iteration == 0:
            new_beta = self.initial_beta
            return {'beta': float(new_beta)}
        
        # 如果已收敛，保持当前 beta
        if converged:
            return {'beta': float(current_beta)}
        
        # 更新历史记录用于收敛分析
        self._update_history(primal_res, dual_res)
        
        # 推断问题类型（基于迭代特征）
        if self.infer_problem_type and iteration < 10:
            self._infer_problem_type(primal_res, dual_res, iteration)
        
        # 针对未收敛问题采用更激进的策略
        if iteration > 100 and not converged:
            # 检查是否属于未收敛问题类型
            if self._is_unconverged_problem_type():
                new_beta = self._aggressive_strategy(current_beta, iteration, primal_res, dual_res)
            else:
                new_beta = self._standard_strategy(current_beta, primal_res, dual_res, iteration)
        else:
            new_beta = self._standard_strategy(current_beta, primal_res, dual_res, iteration)
        
        # 限制 beta 范围
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        return {'beta': float(new_beta)}
    
    def _update_history(self, primal_res: float, dual_res: float) -> None:
        """更新残差历史记录"""
        if primal_res is not None:
            self.primal_history.append(primal_res)
        if dual_res is not None:
            self.dual_history.append(dual_res)
        
        # 保持历史记录大小
        if len(self.primal_history) > self.max_history_size:
            self.primal_history.pop(0)
        if len(self.dual_history) > self.max_history_size:
            self.dual_history.pop(0)
    
    def _infer_problem_type(self, primal_res: float, dual_res: float, iteration: int) -> None:
        """基于早期迭代特征推断问题类型"""
        if iteration == 5 and primal_res is not None and dual_res is not None:
            # 根据残差模式推断
            if primal_res > 10 * dual_res:
                # 原始残差显著大于对偶残差，可能是回归问题
                self.problem_type = 'regression'
            elif dual_res > 10 * primal_res:
                # 对偶残差显著大于原始残差，可能是稀疏优化问题
                self.problem_type = 'sparse'
            else:
                # 残差相对平衡，可能是低秩或其他问题
                self.problem_type = 'balanced'
    
    def _is_unconverged_problem_type(self) -> bool:
        """判断是否属于未收敛问题类型"""
        # 基于历史经验，回归和稀疏问题更容易不收敛
        return self.problem_type in ['regression', 'sparse']
    
    def _standard_strategy(self, current_beta: float, 
                          primal_res: float, dual_res: float,
                          iteration: int) -> float:
        """标准自适应调整策略"""
        # 处理 None 值或极小的残差
        if primal_res is None or dual_res is None or dual_res < 1e-12:
            return current_beta
        
        # 计算残差比
        ratio = primal_res / (dual_res + 1e-12)
        
        # 早期迭代更积极地调整
        if iteration < 50:
            temp_tau_inc = min(self.tau_inc * 1.3, 3.0)
            temp_tau_dec = max(self.tau_dec * 0.9, 1.1)
            temp_mu = self.mu * 0.8
        else:
            temp_tau_inc = self.tau_inc
            temp_tau_dec = self.tau_dec
            temp_mu = self.mu
        
        # 基于残差比调整 beta
        if ratio > temp_mu:
            # 原始残差过大，增加惩罚强度
            new_beta = current_beta * temp_tau_inc
        elif ratio < 1.0 / temp_mu:
            # 对偶残差过大，减少惩罚强度
            new_beta = current_beta / temp_tau_dec
        else:
            # 残差平衡，保持当前 beta
            new_beta = current_beta
        
        # 检查停滞情况
        if len(self.primal_history) >= 3:
            # 计算最近3次迭代的残差变化
            primal_change = abs(self.primal_history[-1] - self.primal_history[-3]) / (self.primal_history[-3] + 1e-12)
            
            if primal_change < 0.01:  # 变化很小
                self.stagnation_count += 1
            else:
                self.stagnation_count = max(0, self.stagnation_count - 1)
            
            # 如果停滞次数超过阈值，尝试调整 beta
            if self.stagnation_count >= self.stagnation_threshold:
                # 停滞时增大 beta 以突破局部最优
                new_beta = current_beta * 1.8
                self.stagnation_count = 0
        
        return new_beta
    
    def _aggressive_strategy(self, current_beta: float, 
                           iteration: int, 
                           primal_res: float, 
                           dual_res: float) -> float:
        """针对未收敛问题的激进策略"""
        # 对于回归问题（含噪声项），采用单调递增策略
        if self.problem_type == 'regression' or (primal_res is not None and dual_res is not None and primal_res < 0.1):
            # beta 只增不减，增长因子逐渐减小
            if iteration < 200:
                growth = self.regression_growth_factor
            elif iteration < 300:
                growth = 1.5
            else:
                growth = 1.2
            
            new_beta = current_beta * growth
        
        # 对于稀疏优化问题，采用混合策略
        elif self.problem_type == 'sparse':
            if primal_res is not None and dual_res is not None:
                ratio = primal_res / (dual_res + 1e-12)
                
                if ratio > 5.0:
                    # 原始残差非常大，大幅增加 beta
                    new_beta = current_beta * 3.0
                elif ratio < 0.2:
                    # 对偶残差非常大，适当减小 beta
                    new_beta = current_beta / 1.5
                else:
                    # 中等情况，适度增加 beta
                    new_beta = current_beta * 1.5
            else:
                new_beta = current_beta * 1.5
        
        # 其他情况使用标准策略
        else:
            new_beta = self._standard_strategy(current_beta, primal_res, dual_res, iteration)
        
        return new_beta
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_beta': self.min_beta,
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'initial_beta': self.initial_beta,
            'regression_growth_factor': self.regression_growth_factor,
            'low_rank_growth_factor': self.low_rank_growth_factor,
            'max_history_size': self.max_history_size,
            'stagnation_threshold': self.stagnation_threshold,
            'infer_problem_type': self.infer_problem_type
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
        if 'regression_growth_factor' in params:
            self.regression_growth_factor = params['regression_growth_factor']
        if 'low_rank_growth_factor' in params:
            self.low_rank_growth_factor = params['low_rank_growth_factor']
        if 'max_history_size' in params:
            self.max_history_size = params['max_history_size']
        if 'stagnation_threshold' in params:
            self.stagnation_threshold = params['stagnation_threshold']
        if 'infer_problem_type' in params:
            self.infer_problem_type = params['infer_problem_type']