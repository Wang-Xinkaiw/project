from strategies.base_strategy import BaseTuningStrategy
import numpy as np

import numpy as np
from typing import Dict, Any, Optional


class BaseTuningStrategy:
    """基础调优策略基类"""

    def __init__(self, initial_parameters: Optional[Dict[str, Any]] = None):
        """初始化策略

        Args:
            initial_parameters: 初始参数字典
        """
        self.parameters = initial_parameters or {}

    def update_parameters(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """根据反馈更新参数

        Args:
            feedback: 包含性能反馈信息的字典

        Returns:
            更新后的参数字典
        """
        raise NotImplementedError("子类必须实现此方法")

    def get_parameters(self) -> Dict[str, Any]:
        """获取当前参数

        Returns:
            当前参数字典
        """
        return self.parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置参数

        Args:
            parameters: 要设置的参数字典
        """
        self.parameters = parameters.copy()


class AdaptiveADMMStrategy(BaseTuningStrategy):
    """自适应ADMM参数调优策略

    该策略根据算法性能反馈自适应调整ADMM参数，特别针对NoneType错误进行优化，
    通过参数验证、边界检查和自适应调整提高算法的稳定性和泛化能力。

    关键特性：
    1. 参数验证和错误处理，防止None值传递
    2. 自适应ρ调整策略，平衡原始和对偶残差
    3. 问题类型感知，针对不同优化问题调整参数
    4. 收敛性监控和自适应调整
    """

    def __init__(self, initial_parameters: Optional[Dict[str, Any]] = None):
        """初始化ADMM调优策略

        Args:
            initial_parameters: 初始参数字典，包含ADMM算法参数
        """
        # 默认参数配置，确保所有参数都有有效默认值
        default_params = {
            'rho': 1.0,                 # ADMM惩罚参数
            'rho_adjustment': True,     # 是否启用自适应ρ调整
            'rho_min': 1e-6,           # ρ最小值
            'rho_max': 1e6,            # ρ最大值
            'tau_incr': 2.0,           # ρ增加倍数
            'tau_decr': 2.0,           # ρ减少倍数
            'mu': 10.0,                # 残差平衡参数
            'max_iter': 1000,          # 最大迭代次数
            'abs_tol': 1e-4,           # 绝对容忍度
            'rel_tol': 1e-2,           # 相对容忍度
            'alpha': 1.0,              # 松弛参数
            'adaptive_tol': True,      # 是否自适应调整容忍度
            'problem_type': 'general', # 问题类型标识
            'verbose': False           # 详细输出
        }

        # 合并用户提供的参数和默认参数
        if initial_parameters:
            # 验证并过滤无效参数
            valid_params = {}
            for key, value in initial_parameters.items():
                if value is not None and key in default_params:
                    valid_params[key] = value
                elif value is not None:
                    # 对于不在默认参数中的键，也允许但记录警告
                    valid_params[key] = value

            # 确保所有必需参数都有值
            for key in default_params:
                if key not in valid_params:
                    valid_params[key] = default_params[key]
        else:
            valid_params = default_params.copy()

        super().__init__(valid_params)

        # 历史性能记录，用于自适应调整
        self.performance_history = {
            'iterations': [],
            'primal_residuals': [],
            'dual_residuals': [],
            'convergence_rates': []
        }

        # 问题类型映射，针对不同类型调整参数
        self.problem_configs = {
            'l1_regularization': {
                'rho': 1.0,
                'alpha': 1.5,
                'adaptive_tol': True
            },
            'elastic_net': {
                'rho': 1.0,
                'alpha': 1.2,
                'adaptive_tol': True
            },
            'l1_regression': {
                'rho': 1.0,
                'alpha': 1.0,
                'adaptive_tol': False
            },
            'elastic_net_regression': {
                'rho': 1.0,
                'alpha': 1.0,
                'adaptive_tol': False
            },
            'low_rank_matrix_completion': {
                'rho': 0.1,
                'alpha': 1.8,
                'adaptive_tol': True
            },
            'low_rank_representation': {
                'rho': 0.1,
                'alpha': 1.8,
                'adaptive_tol': True
            },
            'robust_multi_view_spectral_clustering': {
                'rho': 0.5,
                'alpha': 1.5,
                'adaptive_tol': True
            },
            'general': {
                'rho': 1.0,
                'alpha': 1.0,
                'adaptive_tol': True
            }
        }

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证和清理参数，防止None值

        Args:
            params: 待验证的参数字典

        Returns:
            验证后的参数字典
        """
        validated = {}

        for key, value in params.items():
            # 检查是否为None
            if value is None:
                # 使用当前参数值或默认值
                validated[key] = self.parameters.get(key, 0.0)
                if self.parameters.get('verbose', False):
                    print(f"警告: 参数 {key} 为None，已替换为 {validated[key]}")
            else:
                validated[key] = value

        # 确保关键参数有有效值
        required_params = ['rho', 'max_iter', 'abs_tol', 'rel_tol', 'alpha']
        for param in required_params:
            if param not in validated:
                validated[param] = self.parameters.get(param, 1.0 if param == 'rho' else
                                                      1000 if param == 'max_iter' else
                                                      1e-4 if param == 'abs_tol' else
                                                      1e-2 if param == 'rel_tol' else 1.0)

        # 参数边界检查
        validated['rho'] = np.clip(validated['rho'],
                                  self.parameters.get('rho_min', 1e-6),
                                  self.parameters.get('rho_max', 1e6))
        validated['alpha'] = np.clip(validated['alpha'], 0.5, 2.0)
        validated['max_iter'] = max(10, min(10000, validated['max_iter']))

        return validated

    def _adjust_for_problem_type(self, problem_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """根据问题类型调整参数

        Args:
            problem_type: 问题类型标识
            params: 当前参数

        Returns:
            调整后的参数
        """
        if problem_type in self.problem_configs:
            config = self.problem_configs[problem_type]
            adjusted = params.copy()

            # 合并问题特定配置
            for key in ['rho', 'alpha', 'adaptive_tol']:
                if key in config:
                    adjusted[key] = config[key]

            # 设置问题类型标识
            adjusted['problem_type'] = problem_type

            return adjusted
        else:
            # 未知问题类型，使用通用配置
            params['problem_type'] = 'general'
            return params

    def _adaptive_rho_adjustment(self, primal_residual: float, dual_residual: float,
                                current_rho: float) -> float:
        """自适应调整ρ参数，平衡原始和对偶残差

        Args:
            primal_residual: 原始残差
            dual_residual: 对偶残差
            current_rho: 当前ρ值

        Returns:
            调整后的ρ值
        """
        if not self.parameters.get('rho_adjustment', True):
            return current_rho

        # 避免除以零
        if primal_residual == 0 or dual_residual == 0:
            return current_rho

        mu = self.parameters.get('mu', 10.0)
        tau_incr = self.parameters.get('tau_incr', 2.0)
        tau_decr = self.parameters.get('tau_decr', 2.0)
        rho_min = self.parameters.get('rho_min', 1e-6)
        rho_max = self.parameters.get('rho_max', 1e6)

        # 计算残差比例
        ratio = primal_residual / dual_residual

        # 根据比例调整ρ
        if ratio > mu:
            # 原始残差远大于对偶残差，增加ρ
            new_rho = current_rho * tau_incr
        elif ratio < 1.0 / mu:
            # 对偶残差远大于原始残差，减少ρ
            new_rho = current_rho / tau_decr
        else:
            # 残差平衡，保持ρ不变
            new_rho = current_rho

        # 边界检查
        return np.clip(new_rho, rho_min, rho_max)

    def _adaptive_tolerance_adjustment(self, convergence_rate: float,
                                      current_tol: Dict[str, float]) -> Dict[str, float]:
        """根据收敛速度自适应调整容忍度

        Args:
            convergence_rate: 收敛速度估计
            current_tol: 当前容忍度设置

        Returns:
            调整后的容忍度
        """
        if not self.parameters.get('adaptive_tol', True):
            return current_tol

        adjusted = current_tol.copy()

        # 根据收敛速度调整
        if convergence_rate > 0.9:
            # 收敛缓慢，放松容忍度
            adjusted['abs_tol'] *= 1.5
            adjusted['rel_tol'] *= 1.5
        elif convergence_rate < 0.5:
            # 收敛快速，收紧容忍度以获得更精确解
            adjusted['abs_tol'] *= 0.8
            adjusted['rel_tol'] *= 0.8

        # 边界检查
        adjusted['abs_tol'] = np.clip(adjusted['abs_tol'], 1e-10, 1e-2)
        adjusted['rel_tol'] = np.clip(adjusted['rel_tol'], 1e-10, 1.0)

        return adjusted

    def update_parameters(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """根据性能反馈更新ADMM参数

        该方法分析算法性能反馈，自适应调整参数以提高收敛性和稳定性，
        特别处理NoneType错误和不同问题类型的需求。

        Args:
            feedback: 包含以下键的字典:
                - 'average_iterations': 平均迭代次数
                - 'performance_change': 性能变化百分比
                - 'detailed_problems': 详细问题表现字典
                - 'primal_residuals': 原始残差列表（可选）
                - 'dual_residuals': 对偶残差列表（可选）
                - 'problem_type': 问题类型标识（可选）

        Returns:
            更新后的参数字典
        """
        # 验证反馈数据
        if not isinstance(feedback, dict):
            raise ValueError("反馈必须是字典类型")

        # 获取当前参数并验证
        current_params = self.get_parameters()
        current_params = self._validate_parameters(current_params)

        # 检查是否有错误信息
        error_problems = []
        detailed_problems = feedback.get('detailed_problems', {})

        if detailed_problems:
            for problem_name, problem_info in detailed_problems.items():
                if '错误' in str(problem_info) or 'error' in str(problem_info).lower():
                    error_problems.append(problem_name)

        # 处理NoneType错误 - 这是反馈中的主要问题
        if error_problems:
            if current_params.get('verbose', False):
                print(f"检测到错误问题: {error_problems}")

            # 针对每个错误问题进行参数调整
            for problem in error_problems:
                # 提取问题类型（从问题名称）
                problem_type = problem.lower().replace(' ', '_')

                # 应用问题特定配置
                current_params = self._adjust_for_problem_type(problem_type, current_params)

                # 针对NoneType错误，重置关键参数
                current_params['rho'] = 1.0  # 重置为默认值
                current_params['alpha'] = 1.0  # 重置为默认值

                # 增加最大迭代次数
                current_params['max_iter'] = min(2000, current_params['max_iter'] * 1.5)

                # 收紧容忍度
                current_params['abs_tol'] = max(1e-6, current_params['abs_tol'] * 0.5)
                current_params['rel_tol'] = max(1e-4, current_params['rel_tol'] * 0.5)

        # 处理性能反馈
        avg_iterations = feedback.get('average_iterations')
        performance_change = feedback.get('performance_change', 0.0)

        if avg_iterations is not None:
            # 记录性能历史
            self.performance_history['iterations'].append(avg_iterations)

            # 如果历史记录足够，计算收敛趋势
            if len(self.performance_history['iterations']) >= 3:
                recent_iters = self.performance_history['iterations'][-3:]
                convergence_rate = recent_iters[-1] / max(recent_iters[0], 1)
                self.performance_history['convergence_rates'].append(convergence_rate)

                # 根据收敛趋势调整参数
                if convergence_rate > 1.1:
                    # 发散趋势，调整参数
                    current_params['rho'] *= 1.2
                    current_params['alpha'] *= 0.9
                elif convergence_rate < 0.9:
                    # 良好收敛趋势
                    # 保持参数或微调
                    pass

            # 基于平均迭代次数调整最大迭代次数
            if avg_iterations >= current_params['max_iter'] * 0.9:
                # 接近最大迭代次数，增加上限
                current_params['max_iter'] = min(5000, int(current_params['max_iter'] * 1.2))
            elif avg_iterations <= current_params['max_iter'] * 0.3:
                # 使用较少迭代，可减少上限
                current_params['max_iter'] = max(100, int(current_params['max_iter'] * 0.9))

        # 处理残差信息（如果提供）
        primal_residuals = feedback.get('primal_residuals', [])
        dual_residuals = feedback.get('dual_residuals', [])

        if primal_residuals and dual_residuals:
            # 计算平均残差
            avg_primal = np.mean(primal_residuals) if primal_residuals else 1.0
            avg_dual = np.mean(dual_residuals) if dual_residuals else 1.0

            # 自适应调整ρ
            current_params['rho'] = self._adaptive_rho_adjustment(
                avg_primal, avg_dual, current_params['rho']
            )

            # 自适应调整容忍度
            tol_dict = {
                'abs_tol': current_params.get('abs_tol', 1e-4),
                'rel_tol': current_params.get('rel_tol', 1e-2)
            }

            # 估计收敛速度
            if len(primal_residuals) > 1:
                convergence_rate = primal_residuals[-1] / max(primal_residuals[0], 1e-10)
                adjusted_tol = self._adaptive_tolerance_adjustment(convergence_rate, tol_dict)
                current_params.update(adjusted_tol)

        # 应用问题类型特定调整（如果有指定）
        problem_type = feedback.get('problem_type')
        if problem_type:
            current_params = self._adjust_for_problem_type(problem_type, current_params)

        # 最终验证和边界检查
        current_params = self._validate_parameters(current_params)

        # 更新参数
        self.set_parameters(current_params)

        return current_params

    def get_parameters(self) -> Dict[str, Any]:
        """获取当前参数，确保没有None值

        Returns:
            清理后的参数字典
        """
        return self._validate_parameters(self.parameters.copy())

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置参数，进行验证和清理

        Args:
            parameters: 要设置的参数字典
        """
        validated = self._validate_parameters(parameters)
        self.parameters = validated.copy()

        # 确保关键参数存在
        for key in ['rho', 'max_iter', 'abs_tol', 'rel_tol', 'alpha']:
            if key not in self.parameters:
                self.parameters[key] = 1.0 if key == 'rho' else \
                                     1000 if key == 'max_iter' else \
                                     1e-4 if key == 'abs_tol' else \
                                     1e-2 if key == 'rel_tol' else 1.0