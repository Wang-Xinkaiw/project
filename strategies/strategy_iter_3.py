from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class EnhancedBetaStrategy(BaseTuningStrategy):
    """
    增强版ADMM惩罚参数beta自适应调整策略
    
    基于测试结果分析优化：
    1. 针对l1_regression和elastic_net_regression采用单调递增策略
    2. 针对其他收敛问题优化残差平衡逻辑
    3. 针对对偶残差过大的问题调整衰减策略
    """
    
    def __init__(self):
        # 基础参数
        self.initial_beta = 1.0
        self.max_beta = 1e4
        self.min_beta = 1e-6
        
        # 调整参数 - 基于测试结果优化
        self.base_growth_factor = 1.4  # 略微降低基础增长因子
        self.max_growth_factor = 2.2   # 降低最大增长因子，避免过快增长
        self.min_growth_factor = 0.8   # 允许beta减小，应对对偶残差大的问题
        
        # 残差平衡参数
        self.mu = 15.0  # 增大平衡阈值，更容忍残差不平衡
        self.tau_inc = 1.3  # 降低增长因子
        self.tau_dec = 1.2  # 降低衰减因子
        
        # 阶段控制
        self.early_phase_iterations = 30  # 缩短早期阶段
        self.mid_phase_iterations = 100   # 增加中期阶段
        
        # 历史记录
        self.primal_history = []
        self.dual_history = []
        self.history_window = 3  # 缩短历史窗口
        
        # 状态跟踪
        self.iteration_count = 0
        self.last_beta_change = 0
        
        # 问题特定参数
        self.regression_growth_factor = 1.8  # 回归问题专用增长因子
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新ADMM参数，主要调整惩罚参数beta
        
        Args:
            iteration_state: 包含迭代信息的字典
            
        Returns:
            Dict[str, Any]: 包含更新后的参数，只返回{'beta': new_beta_value}
        """
        # 从iteration_state获取所有需要的信息
        current_beta = iteration_state.get('beta', self.initial_beta)
        primal_res = iteration_state.get('primal_residual', None)
        dual_res = iteration_state.get('dual_residual', None)
        iteration = iteration_state.get('iteration', 0)
        converged = iteration_state.get('converged', False)
        
        self.iteration_count = iteration
        
        # 如果已经收敛或达到最大beta，直接返回
        if converged or current_beta >= self.max_beta:
            return {'beta': min(current_beta, self.max_beta)}
        
        # 保存历史记录
        if primal_res is not None and dual_res is not None:
            self.primal_history.append(primal_res)
            self.dual_history.append(dual_res)
            if len(self.primal_history) > self.history_window:
                self.primal_history.pop(0)
                self.dual_history.pop(0)
        
        # 判断是否为回归问题（根据残差模式）
        is_regression = self._is_regression_problem(iteration_state)
        
        # 对于回归问题，采用单调递增策略
        if is_regression:
            new_beta = current_beta * self.regression_growth_factor
        else:
            # 非回归问题采用自适应策略
            new_beta = self._adaptive_beta_strategy(current_beta, primal_res, dual_res, iteration)
        
        # 限制beta在有效范围内
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        # 跟踪beta变化
        if abs(new_beta - current_beta) > 1e-6:
            self.last_beta_change = iteration
        
        return {'beta': float(new_beta)}
    
    def _adaptive_beta_strategy(self, current_beta: float, primal_res: float, 
                                dual_res: float, iteration: int) -> float:
        """
        自适应beta调整策略
        
        Args:
            current_beta: 当前beta值
            primal_res: 原始残差
            dual_res: 对偶残差
            iteration: 当前迭代次数
            
        Returns:
            float: 新beta值
        """
        # 早期阶段：适度增长
        if iteration < self.early_phase_iterations:
            return current_beta * self.base_growth_factor
        
        # 如果残差信息不可用，保持当前beta
        if primal_res is None or dual_res is None:
            return current_beta
        
        # 计算残差比
        if dual_res > 1e-10:
            ratio = primal_res / dual_res
        else:
            ratio = 1.0
        
        # 中期阶段：基于残差比调整
        if iteration < self.mid_phase_iterations:
            # 原始残差远大于对偶残差，需要增加beta
            if ratio > self.mu:
                return current_beta * self.tau_inc
            # 对偶残差远大于原始残差，需要减少beta
            elif ratio < 1.0 / self.mu:
                return current_beta / self.tau_dec
            else:
                return current_beta
        
        # 后期阶段：更精细的调整
        if iteration >= self.mid_phase_iterations:
            # 对偶残差过大时，主动减小beta
            if dual_res > 10.0 and primal_res < 0.1:
                return current_beta * 0.95
            
            # 检测残差变化趋势
            if len(self.primal_history) >= 2:
                primal_trend = self.primal_history[-1] / self.primal_history[0] if self.primal_history[0] > 0 else 1.0
                dual_trend = self.dual_history[-1] / self.dual_history[0] if self.dual_history[0] > 0 else 1.0
                
                # 如果残差都在下降，保持当前beta
                if primal_trend < 1.0 and dual_trend < 1.0:
                    return current_beta
                
                # 如果对偶残差在上升而原始残差在下降，减小beta
                if primal_trend < 0.9 and dual_trend > 1.1:
                    return current_beta * 0.9
        
        return current_beta
    
    def _is_regression_problem(self, iteration_state: Dict[str, Any]) -> bool:
        """
        判断是否为回归问题（包含噪声项E）
        
        Args:
            iteration_state: 迭代状态
            
        Returns:
            bool: 是否为回归问题
        """
        primal_res = iteration_state.get('primal_residual')
        dual_res = iteration_state.get('dual_residual')
        
        if primal_res is None or dual_res is None:
            return False
        
        # 回归问题的特征：对偶残差远大于原始残差
        if dual_res > 50.0 and primal_res < 0.1:
            # 计算残差比
            if dual_res > 1e-10:
                ratio = primal_res / dual_res
                if ratio < 0.01:  # 原始残差远小于对偶残差
                    return True
        
        return False
    
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
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec,
            'early_phase_iterations': self.early_phase_iterations,
            'mid_phase_iterations': self.mid_phase_iterations,
            'regression_growth_factor': self.regression_growth_factor
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
            'mu', 'tau_inc', 'tau_dec',
            'early_phase_iterations', 'mid_phase_iterations',
            'regression_growth_factor'
        }
        
        for key, value in params.items():
            if key in valid_params and hasattr(self, key):
                setattr(self, key, value)