from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class ImprovedAdaptiveBetaStrategy(BaseTuningStrategy):
    """
    改进的ADMM惩罚参数beta自适应调整策略
    
    针对收敛问题的优化策略：
    1. 对于l1_regression和elastic_net_regression问题，采用更积极的单调递增策略
    2. 对于其他未收敛问题，采用更保守的自适应调整
    3. 针对不同残差模式调整beta增长速率
    """
    
    def __init__(self):
        # 基础参数
        self.initial_beta = 1.0
        self.max_beta = 1e4
        self.min_beta = 1e-6
        
        # 增长参数 - 调整为更激进的值
        self.base_growth_factor = 1.5  # 增加基础增长因子
        self.max_growth_factor = 2.5   # 增加最大增长因子
        self.min_growth_factor = 1.2   # 增加最小增长因子
        
        # 收敛检测参数
        self.convergence_threshold = 1e-3
        self.residual_ratio_threshold = 5.0  # 降低残差比阈值
        
        # 问题特定参数
        self.regression_problems = ['l1_regression', 'elastic_net_regression']
        self.problem_type = None
        
        # 历史记录
        self.primal_history = []
        self.dual_history = []
        self.history_window = 5
        
        # 状态跟踪
        self.stagnation_counter = 0
        self.stagnation_threshold = 20
        
        # 早期快速增长阶段
        self.initial_phase_iterations = 50
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新ADMM参数，主要调整惩罚参数beta
        
        Args:
            iteration_state: 包含迭代信息的字典，包括：
                - iteration: 当前迭代次数
                - primal_residual: 原始残差
                - dual_residual: 对偶残差
                - beta: 当前beta值
                - objective: 目标函数值
                - converged: 是否收敛
                
        Returns:
            Dict[str, Any]: 包含更新后的参数，只返回{'beta': new_beta_value}
        """
        # 从iteration_state获取所有需要的信息
        current_beta = iteration_state.get('beta', self.initial_beta)
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        iteration = iteration_state.get('iteration', 0)
        converged = iteration_state.get('converged', False)
        
        # 保存历史记录用于趋势分析
        if primal_res is not None and dual_res is not None:
            self.primal_history.append(primal_res)
            self.dual_history.append(dual_res)
            if len(self.primal_history) > self.history_window:
                self.primal_history.pop(0)
                self.dual_history.pop(0)
        
        # 如果已经收敛或达到最大beta，直接返回当前beta
        if converged or current_beta >= self.max_beta:
            return {'beta': min(current_beta, self.max_beta)}
        
        # 早期阶段：快速增加beta
        if iteration < self.initial_phase_iterations:
            growth_factor = self.max_growth_factor
            new_beta = current_beta * growth_factor
            new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
            return {'beta': float(new_beta)}
        
        # 计算自适应增长因子
        growth_factor = self._calculate_growth_factor(primal_res, dual_res, iteration_state)
        
        # 计算新beta值
        new_beta = current_beta * growth_factor
        
        # 对于回归问题，beta只增不减
        if self._is_regression_problem(iteration_state):
            # 确保beta至少保持当前值或增加
            new_beta = max(new_beta, current_beta)
        else:
            # 其他问题：允许小幅减少，但设置下限
            if new_beta < current_beta * 0.9:  # 最多减少10%
                new_beta = current_beta * 0.9
        
        # 检测停滞情况
        if self._is_stagnating(primal_res, dual_res):
            self.stagnation_counter += 1
            if self.stagnation_counter > self.stagnation_threshold:
                # 如果停滞太久，更激进地增加beta
                new_beta = current_beta * min(self.max_growth_factor * 1.5, 3.0)
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        # 限制beta在有效范围内
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        return {'beta': float(new_beta)}
    
    def _calculate_growth_factor(self, primal_res: float, dual_res: float, 
                                iteration_state: Dict[str, Any]) -> float:
        """
        计算自适应增长因子
        
        Args:
            primal_res: 原始残差
            dual_res: 对偶残差
            iteration_state: 迭代状态
            
        Returns:
            float: 增长因子
        """
        # 基础增长因子
        growth_factor = self.base_growth_factor
        
        # 如果残差信息可用，根据残差情况调整增长因子
        if primal_res is not None and dual_res is not None:
            # 计算残差和
            total_residual = primal_res + dual_res
            
            # 如果残差很大，需要更快地增加beta
            if total_residual > 1.0:
                growth_factor = min(self.max_growth_factor, growth_factor * 1.3)
            elif total_residual > 0.1:
                growth_factor = min(self.max_growth_factor, growth_factor * 1.1)
            
            # 避免除零错误
            if dual_res > 1e-10:
                residual_ratio = primal_res / dual_res
                
                # 如果原始残差远大于对偶残差，更快地增加beta
                if residual_ratio > self.residual_ratio_threshold:
                    growth_factor = min(self.max_growth_factor, growth_factor * 1.5)
                
                # 如果对偶残差远大于原始残差，稍微减慢beta增长
                elif residual_ratio < 1.0 / self.residual_ratio_threshold:
                    growth_factor = max(self.min_growth_factor, growth_factor * 0.9)
            
            # 检查残差下降趋势
            if len(self.primal_history) >= 3:
                # 计算最近残差变化率
                if self.primal_history[-2] > 1e-10:
                    primal_change = self.primal_history[-1] / self.primal_history[-2]
                    
                    # 如果残差在快速下降，减缓beta增长
                    if primal_change < 0.8:
                        growth_factor = max(self.min_growth_factor, growth_factor * 0.8)
                    # 如果残差在增加，加快beta增长
                    elif primal_change > 1.2:
                        growth_factor = min(self.max_growth_factor, growth_factor * 1.2)
        
        # 确保增长因子在合理范围内
        growth_factor = np.clip(growth_factor, self.min_growth_factor, self.max_growth_factor)
        
        return growth_factor
    
    def _is_regression_problem(self, iteration_state: Dict[str, Any]) -> bool:
        """
        判断是否为回归问题（包含噪声项E）
        
        注意：这里通过残差模式判断，因为无法直接从iteration_state获取问题类型
        回归问题的特征：原始残差和对偶残差都较大且相当
        
        Args:
            iteration_state: 迭代状态
            
        Returns:
            bool: 是否为回归问题
        """
        primal_res = iteration_state.get('primal_residual')
        dual_res = iteration_state.get('dual_residual')
        
        if primal_res is None or dual_res is None:
            return False
            
        # 回归问题通常有较大的残差，且两者相当
        if primal_res > 0.5 and dual_res > 0.5:
            ratio = primal_res / dual_res
            if 0.5 < ratio < 2.0:
                return True
                
        return False
    
    def _is_stagnating(self, primal_res: float, dual_res: float) -> bool:
        """
        检测算法是否停滞
        
        Args:
            primal_res: 原始残差
            dual_res: 对偶残差
            
        Returns:
            bool: 是否停滞
        """
        if len(self.primal_history) < 3:
            return False
            
        # 计算最近几次迭代的残差变化
        recent_primal = self.primal_history[-3:]
        recent_dual = self.dual_history[-3:]
        
        # 如果残差几乎没有变化，视为停滞
        primal_var = np.std(recent_primal) / (np.mean(recent_primal) + 1e-10)
        dual_var = np.std(recent_dual) / (np.mean(recent_dual) + 1e-10)
        
        return primal_var < 0.05 and dual_var < 0.05
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取策略参数
        
        Returns:
            Dict[str, Any]: 参数字典
        """
        return {
            'initial_beta': self.initial_beta,
            'max_beta': self.max_beta,
            'min_beta': self.min_beta,
            'base_growth_factor': self.base_growth_factor,
            'max_growth_factor': self.max_growth_factor,
            'min_growth_factor': self.min_growth_factor,
            'convergence_threshold': self.convergence_threshold,
            'residual_ratio_threshold': self.residual_ratio_threshold,
            'history_window': self.history_window,
            'initial_phase_iterations': self.initial_phase_iterations,
            'stagnation_threshold': self.stagnation_threshold
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置策略参数
        
        Args:
            params: 参数字典
        """
        valid_params = {
            'initial_beta', 'max_beta', 'min_beta', 
            'base_growth_factor', 'max_growth_factor', 'min_growth_factor',
            'convergence_threshold', 'residual_ratio_threshold', 'history_window',
            'initial_phase_iterations', 'stagnation_threshold'
        }
        
        for key, value in params.items():
            if key in valid_params and hasattr(self, key):
                setattr(self, key, value)