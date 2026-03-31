from typing import Dict, Any
import numpy as np


class BaseTuningStrategy:
    """基础参数调整策略基类"""
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def adjust_beta(iteration_state: Dict[str, Any]) -> float:
        """获取当前参数"""
        return self.params.copy()
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        self.params.update(kwargs)


class AdaptiveBetaStrategy(BaseTuningStrategy):
    """
    ADMM惩罚参数beta自适应调整策略
    
    核心特点：
    1. 针对回归问题(l1_regression, elastic_net_regression)采用单调递增策略
    2. 针对其他问题采用残差比自适应策略
    3. 考虑迭代历史信息，避免震荡
    4. 具有鲁棒的数值处理机制
    """
    
    def __init__(self, 
                 initial_beta: float = 1.0,
                 min_beta: float = 1e-6,
                 max_beta: float = 1e4,
                 mu: float = 10.0,
                 tau_inc: float = 2.0,
                 tau_dec: float = 2.0,
                 growth_rate: float = 1.1,
                 problem_type: str = "l1_regularization",
                 history_window: int = 5):
        """
        初始化自适应beta调整策略
        
        参数:
        ----------
        initial_beta : float
            初始beta值，默认1.0
        min_beta : float
            beta最小值，默认1e-6
        max_beta : float
            beta最大值，默认1e4
        mu : float
            残差比阈值，默认10.0
        tau_inc : float
            beta增大因子，默认2.0
        tau_dec : float
            beta减小因子，默认2.0
        growth_rate : float
            单调递增时的增长率，默认1.1
        problem_type : str
            问题类型，决定调整策略
        history_window : int
            历史信息窗口大小，用于检测震荡
        """
        super().__init__(
            initial_beta=initial_beta,
            min_beta=min_beta,
            max_beta=max_beta,
            mu=mu,
            tau_inc=tau_inc,
            tau_dec=tau_dec,
            growth_rate=growth_rate,
            problem_type=problem_type,
            history_window=history_window
        )
        
        # 内部状态变量
        self.beta_history = []
        self.primal_history = []
        self.dual_history = []
        self.is_regression = problem_type in ["l1_regression", "elastic_net_regression"]
    
    def _detect_oscillation(self, current_beta: float, window: int = 3) -> bool:
        """检测beta值是否在震荡"""
        if len(self.beta_history) < window * 2:
            return False
        
        recent_betas = self.beta_history[-window:]
        variations = np.abs(np.diff(recent_betas) / recent_betas[:-1])
        
        # 如果最近几次调整方向频繁变化，说明在震荡
        signs = np.sign(np.diff(recent_betas))
        sign_changes = np.sum(np.abs(np.diff(signs)))
        
        return sign_changes >= window - 2
    
    def _regression_strategy(self, iteration_state: Dict[str, Any]) -> float:
        """回归问题专用策略：单调递增"""
        current_beta = iteration_state.get('beta', self.params['initial_beta'])
        iteration = iteration_state.get('iteration', 0)
        
        # 根据迭代次数调整增长率
        if iteration < 50:
            # 前期：快速增长
            effective_growth = self.params['growth_rate'] * 1.2
        elif iteration < 200:
            # 中期：正常增长
            effective_growth = self.params['growth_rate']
        else:
            # 后期：缓慢增长
            effective_growth = 1.01
        
        new_beta = current_beta * effective_growth
        
        # 限制范围
        return float(np.clip(new_beta, self.params['min_beta'], self.params['max_beta']))
    
    def _residual_ratio_strategy(self, iteration_state: Dict[str, Any]) -> float:
        """基于残差比的自适应策略"""
        primal_res = iteration_state.get('primal_residual')
        dual_res = iteration_state.get('dual_residual')
        current_beta = iteration_state.get('beta', self.params['initial_beta'])
        iteration = iteration_state.get('iteration', 0)
        
        # 处理缺失值或无效值
        if primal_res is None or dual_res is None or dual_res < 1e-10:
            return current_beta
        
        # 计算残差比
        ratio = primal_res / max(dual_res, 1e-10)
        
        # 根据迭代阶段调整阈值
        if iteration < 50:
            # 前期：更激进的调整
            effective_mu = self.params['mu'] / 2
            effective_tau_inc = min(self.params['tau_inc'] * 1.5, 3.0)
            effective_tau_dec = min(self.params['tau_dec'] * 1.5, 3.0)
        elif iteration < 200:
            # 中期：正常调整
            effective_mu = self.params['mu']
            effective_tau_inc = self.params['tau_inc']
            effective_tau_dec = self.params['tau_dec']
        else:
            # 后期：保守调整
            effective_mu = self.params['mu'] * 2
            effective_tau_inc = max(self.params['tau_inc'] * 0.5, 1.2)
            effective_tau_dec = max(self.params['tau_dec'] * 0.5, 1.2)
        
        # 检查是否震荡
        if self._detect_oscillation(current_beta):
            # 震荡时采用更保守的策略
            effective_tau_inc = max(effective_tau_inc * 0.7, 1.1)
            effective_tau_dec = max(effective_tau_dec * 0.7, 1.1)
        
        # 根据残差比调整beta
        if ratio > effective_mu:
            new_beta = current_beta * effective_tau_inc
        elif ratio < 1.0 / effective_mu:
            new_beta = current_beta / effective_tau_dec
        else:
            # 平衡状态：小幅调整或保持
            if primal_res > 1e-3:  # 残差仍较大，小幅增加
                new_beta = current_beta * 1.05
            else:
                new_beta = current_beta
        
        # 限制范围
        new_beta = float(np.clip(new_beta, self.params['min_beta'], self.params['max_beta']))
        
        # 更新历史记录
        self.beta_history.append(new_beta)
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        
        # 保持历史记录大小
        if len(self.beta_history) > self.params['history_window'] * 2:
            self.beta_history = self.beta_history[-self.params['history_window'] * 2:]
            self.primal_history = self.primal_history[-self.params['history_window'] * 2:]
            self.dual_history = self.dual_history[-self.params['history_window'] * 2:]
        
        return new_beta
    
    def adjust_beta(iteration_state: Dict[str, Any]) -> float:
        ----------
        iteration_state : Dict[str, Any]
            迭代状态字典，包含当前迭代信息
        
        返回:
        ----------
        Dict[str, Any]
            包含更新后beta值的字典
        """
        converged = iteration_state.get('converged', False)
        
        # 如果已收敛，不再调整beta
        if converged:
            current_beta = iteration_state.get('beta', self.params['initial_beta'])
            return {'beta': current_beta}
        
        # 根据问题类型选择策略
        if self.is_regression:
            new_beta = self._regression_strategy(iteration_state)
        else:
            new_beta = self._residual_ratio_strategy(iteration_state)
        
        return {'beta': new_beta}