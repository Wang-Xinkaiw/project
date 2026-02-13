from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class OptimizedAdaptiveBetaStrategy(BaseTuningStrategy):
    """
    针对未收敛问题优化的ADMM惩罚参数beta自适应调整策略
    
    主要改进方向：
    1. 对回归问题采用更明确的识别和特殊处理策略
    2. 调整增长因子范围，避免beta过快达到上限
    3. 增加收敛检测机制，对接近收敛的问题减小beta增长
    4. 针对对偶残差大的问题添加特殊处理逻辑
    """
    
    def __init__(self):
        # 基础参数
        self.initial_beta = 1.0
        self.max_beta = 1e4
        self.min_beta = 1e-6
        
        # 增长参数 - 调整为更温和的增长
        self.base_growth_factor = 1.3  # 减小基础增长因子
        self.max_growth_factor = 2.0   # 减小最大增长因子
        self.min_growth_factor = 1.05  # 增加最小增长因子，避免beta下降过快
        
        # 收敛检测参数
        self.convergence_threshold = 1e-3
        self.residual_ratio_threshold = 3.0  # 进一步降低残差比阈值
        
        # 回归问题参数（针对l1_regression和elastic_net_regression）
        self.regression_initial_beta = 1.0
        self.regression_growth_factor = 1.5  # 回归问题使用固定增长因子
        
        # 历史记录
        self.primal_history = []
        self.dual_history = []
        self.history_window = 10  # 增加窗口大小
        
        # 状态跟踪
        self.stagnation_counter = 0
        self.stagnation_threshold = 15  # 减少停滞检测阈值
        
        # 早期阶段
        self.initial_phase_iterations = 30  # 减少初始阶段迭代次数
        
        # 特殊问题处理
        self.dual_residual_large_counter = 0
        self.dual_residual_threshold = 50.0  # 对偶残差大的阈值
        
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
        if converged:
            return {'beta': current_beta}
        
        if current_beta >= self.max_beta:
            # 达到最大beta后，如果残差仍然很大，尝试小幅调整
            if primal_res > self.convergence_threshold or dual_res > self.convergence_threshold:
                # 如果原始残差远小于对偶残差，小幅减少beta
                if dual_res > 1e-10 and primal_res / dual_res < 0.01:
                    return {'beta': current_beta * 0.95}
                # 否则保持最大beta
                else:
                    return {'beta': self.max_beta}
            else:
                return {'beta': current_beta}
        
        # 判断是否为回归问题（基于残差模式）
        is_regression = self._is_regression_problem(primal_res, dual_res)
        
        # 早期阶段：温和增加beta
        if iteration < self.initial_phase_iterations:
            if is_regression:
                # 回归问题早期更快增长
                growth_factor = min(1.8, self.regression_growth_factor * 1.2)
            else:
                growth_factor = min(1.5, self.base_growth_factor * 1.2)
            
            new_beta = current_beta * growth_factor
            new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
            return {'beta': float(new_beta)}
        
        # 处理对偶残差特别大的情况
        if dual_res > self.dual_residual_threshold:
            self.dual_residual_large_counter += 1
            # 如果连续多次对偶残差都很大，减小beta
            if self.dual_residual_large_counter >= 3:
                # 减小beta，但设置下限
                new_beta = current_beta * 0.8
                self.dual_residual_large_counter = 0  # 重置计数器
                new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
                return {'beta': float(new_beta)}
        else:
            self.dual_residual_large_counter = max(0, self.dual_residual_large_counter - 1)
        
        # 根据不同问题类型采用不同策略
        if is_regression:
            # 回归问题：beta只增不减，使用固定增长因子
            growth_factor = self.regression_growth_factor
            new_beta = current_beta * growth_factor
        else:
            # 其他问题：自适应调整策略
            growth_factor = self._calculate_adaptive_growth_factor(primal_res, dual_res)
            
            # 计算新beta值，允许小幅减少
            if growth_factor < 1.0:
                # 如果beta要减少，最多减少到当前值的90%
                new_beta = current_beta * max(growth_factor, 0.9)
            else:
                new_beta = current_beta * growth_factor
        
        # 检测停滞情况
        if self._is_stagnating(primal_res, dual_res):
            self.stagnation_counter += 1
            if self.stagnation_counter > self.stagnation_threshold:
                # 如果停滞太久，更温和地调整beta
                if is_regression:
                    # 回归问题稍微加快增长
                    new_beta = current_beta * min(self.regression_growth_factor * 1.3, 2.0)
                else:
                    # 其他问题尝试小幅振荡
                    if np.random.random() < 0.5:
                        new_beta = current_beta * 1.2
                    else:
                        new_beta = current_beta * 0.9
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        # 检测接近收敛的情况
        if self._is_near_convergence(primal_res, dual_res):
            # 接近收敛时，减小beta变化幅度
            if new_beta > current_beta:
                new_beta = current_beta * min(1.1, growth_factor)
            elif new_beta < current_beta:
                new_beta = current_beta * max(0.95, growth_factor)
        
        # 限制beta在有效范围内
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        
        return {'beta': float(new_beta)}
    
    def _calculate_adaptive_growth_factor(self, primal_res: float, dual_res: float) -> float:
        """
        计算自适应增长因子
        
        Args:
            primal_res: 原始残差
            dual_res: 对偶残差
            
        Returns:
            float: 增长因子
        """
        # 基础增长因子
        growth_factor = self.base_growth_factor
        
        if primal_res is None or dual_res is None:
            return growth_factor
        
        # 计算残差和
        total_residual = primal_res + dual_res
        
        # 根据残差绝对大小调整
        if total_residual > 10.0:
            growth_factor = min(self.max_growth_factor, growth_factor * 1.5)
        elif total_residual > 1.0:
            growth_factor = min(self.max_growth_factor, growth_factor * 1.2)
        elif total_residual < 0.01:
            # 残差很小时，保持beta稳定
            growth_factor = 1.0
        
        # 根据残差相对大小调整
        if dual_res > 1e-10:
            residual_ratio = primal_res / dual_res
            
            # 如果原始残差远大于对偶残差，增加beta
            if residual_ratio > self.residual_ratio_threshold:
                growth_factor = min(self.max_growth_factor, growth_factor * 1.3)
            
            # 如果对偶残差远大于原始残差，减少beta
            elif residual_ratio < 1.0 / self.residual_ratio_threshold:
                growth_factor = max(self.min_growth_factor, growth_factor * 0.7)
        
        # 考虑历史趋势
        if len(self.primal_history) >= 3:
            # 计算最近残差变化趋势
            if self.primal_history[-2] > 1e-10 and self.dual_history[-2] > 1e-10:
                primal_change = self.primal_history[-1] / self.primal_history[-2]
                dual_change = self.dual_history[-1] / self.dual_history[-2]
                
                # 如果残差在下降，减缓调整
                if primal_change < 0.9 and dual_change < 0.9:
                    growth_factor = max(self.min_growth_factor, growth_factor * 0.9)
                # 如果残差在增加，加快调整
                elif primal_change > 1.1 and dual_change > 1.1:
                    growth_factor = min(self.max_growth_factor, growth_factor * 1.1)
        
        # 确保增长因子在合理范围内
        growth_factor = np.clip(growth_factor, self.min_growth_factor, self.max_growth_factor)
        
        return growth_factor
    
    def _is_regression_problem(self, primal_res: float, dual_res: float) -> bool:
        """
        判断是否为回归问题（包含噪声项E）
        
        Args:
            primal_res: 原始残差
            dual_res: 对偶残差
            
        Returns:
            bool: 是否为回归问题
        """
        if primal_res is None or dual_res is None:
            return False
        
        # 回归问题的典型特征：
        # 1. 原始残差和对偶残差都较大
        # 2. 两者数量级相近
        # 3. 原始残差通常小于对偶残差
        
        if primal_res > 0.1 and dual_res > 10.0:
            ratio = primal_res / dual_res
            # 回归问题通常原始残差比对偶残差小1-2个数量级
            if 1e-4 < ratio < 0.1:
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
        if len(self.primal_history) < 5:
            return False
        
        # 计算最近几次迭代的残差变化
        recent_primal = self.primal_history[-5:]
        recent_dual = self.dual_history[-5:]
        
        # 如果残差几乎没有变化，视为停滞
        primal_mean = np.mean(recent_primal)
        dual_mean = np.mean(recent_dual)
        
        if primal_mean > 1e-10 and dual_mean > 1e-10:
            primal_var = np.std(recent_primal) / primal_mean
            dual_var = np.std(recent_dual) / dual_mean
            
            return primal_var < 0.1 and dual_var < 0.1
        
        return False
    
    def _is_near_convergence(self, primal_res: float, dual_res: float) -> bool:
        """
        检测是否接近收敛
        
        Args:
            primal_res: 原始残差
            dual_res: 对偶残差
            
        Returns:
            bool: 是否接近收敛
        """
        if primal_res is None or dual_res is None:
            return False
        
        # 接近收敛的条件：残差已经很小
        if primal_res < 0.01 and dual_res < 0.1:
            return True
        
        # 或者残差在稳定下降
        if len(self.primal_history) >= 3:
            recent_primal = self.primal_history[-3:]
            recent_dual = self.dual_history[-3:]
            
            if all(recent_primal[i] > recent_primal[i+1] for i in range(len(recent_primal)-1)) and \
               all(recent_dual[i] > recent_dual[i+1] for i in range(len(recent_dual)-1)):
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
            'convergence_threshold': self.convergence_threshold,
            'residual_ratio_threshold': self.residual_ratio_threshold,
            'regression_initial_beta': self.regression_initial_beta,
            'regression_growth_factor': self.regression_growth_factor,
            'history_window': self.history_window,
            'initial_phase_iterations': self.initial_phase_iterations,
            'stagnation_threshold': self.stagnation_threshold,
            'dual_residual_threshold': self.dual_residual_threshold
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
            'convergence_threshold', 'residual_ratio_threshold', 
            'regression_initial_beta', 'regression_growth_factor',
            'history_window', 'initial_phase_iterations', 'stagnation_threshold',
            'dual_residual_threshold'
        }
        
        for key, value in params.items():
            if key in valid_params and hasattr(self, key):
                setattr(self, key, value)