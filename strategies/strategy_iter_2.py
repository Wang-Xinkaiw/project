from strategies.base_strategy import BaseTuningStrategy
import numpy as np

import numpy as np
from typing import Dict, Any, Optional, Union

# 假设BaseTuningStrategy已在其他模块中定义
class BaseTuningStrategy:
    """基类定义，假设已存在"""
    pass


class ADMMStrategy(BaseTuningStrategy):
    """
    ADMM算法的自适应参数调优策略

    改进要点：
    1. 修复类型错误：确保所有数值比较中的参数都是数值类型
    2. 增加参数验证和类型转换
    3. 改进参数更新逻辑，考虑多问题泛化
    4. 添加收敛性、稳定性和效率优化
    """

    def __init__(self, initial_params: Optional[Dict[str, Any]] = None):
        """
        初始化ADMM调优策略

        参数:
            initial_params: 初始参数字典，包含rho, alpha, max_iter等
        """
        # 默认参数设置
        self.default_params = {
            'rho': 1.0,           # ADMM惩罚参数
            'alpha': 1.0,         # 松弛参数（过松弛时为1.5-1.8）
            'max_iter': 1000,     # 最大迭代次数
            'abs_tol': 1e-4,      # 绝对容差
            'rel_tol': 1e-2,      # 相对容差
            'adaptive_rho': True, # 是否自适应调整rho
            'rho_tau': 2.0,       # rho调整系数
            'rho_mu': 10.0,       # 残差平衡阈值
            'rho_min': 1e-6,      # rho最小值
            'rho_max': 1e6,       # rho最大值
        }

        # 当前参数
        self.current_params = self.default_params.copy()

        # 如果提供了初始参数，更新当前参数（需类型转换）
        if initial_params:
            self.set_parameters(initial_params)

        # 性能历史记录
        self.performance_history = []
        self.target_iterations = 500  # 目标迭代次数

        # 自适应调整参数
        self.rho_update_count = 0
        self.max_rho_updates = 10     # 最大rho调整次数

    def _ensure_numeric(self, value: Any, param_name: str) -> Union[float, int]:
        """
        确保参数为数值类型

        参数:
            value: 参数值
            param_name: 参数名称（用于错误信息）

        返回:
            转换后的数值
        """
        try:
            if isinstance(value, (int, float, np.number)):
                return float(value)
            elif isinstance(value, str):
                # 尝试从字符串转换
                return float(value)
            else:
                # 其他类型尝试转换
                return float(value)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"参数 '{param_name}' 无法转换为数值类型: {value}. "
                f"原错误: {str(e)}"
            )

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        验证并转换参数字典中的所有值为数值类型

        参数:
            params: 待验证的参数字典

        返回:
            验证并转换后的参数字典
        """
        validated_params = {}

        for key, value in params.items():
            if key in self.default_params:
                # 确保数值类型
                validated_params[key] = self._ensure_numeric(value, key)

                # 附加验证逻辑
                if key == 'rho' and validated_params[key] <= 0:
                    raise ValueError(f"参数 'rho' 必须为正数，当前值: {validated_params[key]}")
                elif key == 'max_iter' and validated_params[key] < 1:
                    raise ValueError(f"参数 'max_iter' 必须大于等于1，当前值: {validated_params[key]}")
                elif key == 'abs_tol' and validated_params[key] <= 0:
                    raise ValueError(f"参数 'abs_tol' 必须为正数，当前值: {validated_params[key]}")
                elif key == 'rel_tol' and validated_params[key] <= 0:
                    raise ValueError(f"参数 'rel_tol' 必须为正数，当前值: {validated_params[key]}")
                elif key == 'alpha' and validated_params[key] < 1:
                    raise ValueError(f"参数 'alpha' 必须大于等于1，当前值: {validated_params[key]}")

        return validated_params

    def get_parameters(self) -> Dict[str, float]:
        """
        获取当前参数

        返回:
            当前参数字典
        """
        return self.current_params.copy()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置参数

        参数:
            params: 要设置的参数字典
        """
        try:
            validated_params = self._validate_parameters(params)
            self.current_params.update(validated_params)

            # 记录参数更新
            print(f"ADMM参数已更新: {validated_params}")

        except (TypeError, ValueError) as e:
            print(f"参数设置失败: {str(e)}")
            print(f"使用默认参数替代")
            # 失败时使用默认值
            for key in params:
                if key in self.default_params:
                    self.current_params[key] = self.default_params[key]

    def update_parameters(self,
                         performance_metrics: Optional[Dict[str, Any]] = None,
                         problem_type: str = "general") -> Dict[str, float]:
        """
        根据性能指标更新ADMM参数

        改进逻辑：
        1. 根据迭代次数自适应调整rho
        2. 考虑不同问题类型的特性
        3. 确保数值稳定性

        参数:
            performance_metrics: 性能指标字典，包含迭代次数、残差等信息
            problem_type: 问题类型，用于特定调整

        返回:
            更新后的参数字典
        """
        # 确保performance_metrics是字典
        if performance_metrics is None:
            performance_metrics = {}

        # 记录性能历史
        self.performance_history.append(performance_metrics)

        # 限制历史记录长度
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        # 获取当前参数
        params = self.current_params.copy()

        # 检查是否有有效的性能数据
        if not performance_metrics:
            print("无性能数据，使用当前参数")
            return params

        # 提取性能指标，确保数值类型
        try:
            iterations = self._ensure_numeric(
                performance_metrics.get('iterations', params['max_iter']),
                'iterations'
            )

            # 获取残差信息（如果存在）
            primal_residual = self._ensure_numeric(
                performance_metrics.get('primal_residual', 0.0),
                'primal_residual'
            ) if 'primal_residual' in performance_metrics else 0.0

            dual_residual = self._ensure_numeric(
                performance_metrics.get('dual_residual', 0.0),
                'dual_residual'
            ) if 'dual_residual' in performance_metrics else 0.0

        except (TypeError, ValueError) as e:
            print(f"性能指标解析失败: {str(e)}")
            return params

        # 根据问题类型调整策略
        if problem_type in ["l1_regularization", "elastic_net", "l1_regression"]:
            # 稀疏优化问题通常需要较小的rho
            target_iter = 300
            rho_adjust_factor = 1.5
        elif problem_type in ["low_rank_matrix_completion", "low_rank_representation"]:
            # 低秩问题通常需要适中的rho
            target_iter = 400
            rho_adjust_factor = 2.0
        elif problem_type in ["robust_multi_view_spectral_clustering"]:
            # 聚类问题可能需要较大的rho
            target_iter = 600
            rho_adjust_factor = 2.5
        else:
            # 通用问题设置
            target_iter = self.target_iterations
            rho_adjust_factor = 2.0

        # 自适应调整rho（如果启用）
        if params.get('adaptive_rho', True) and self.rho_update_count < self.max_rho_updates:
            # 根据迭代次数调整
            if iterations > target_iter * 1.2:  # 迭代次数过多
                if params['rho'] < params['rho_max'] / rho_adjust_factor:
                    params['rho'] *= rho_adjust_factor
                    print(f"增加rho至 {params['rho']:.4f} (迭代次数: {iterations})")
                    self.rho_update_count += 1
            elif iterations < target_iter * 0.8:  # 迭代次数过少
                if params['rho'] > params['rho_min'] * rho_adjust_factor:
                    params['rho'] /= rho_adjust_factor
                    print(f"减少rho至 {params['rho']:.4f} (迭代次数: {iterations})")
                    self.rho_update_count += 1

            # 基于残差的自适应调整（如果残差信息可用）
            if primal_residual > 0 and dual_residual > 0:
                residual_ratio = primal_residual / dual_residual

                if residual_ratio > params['rho_mu']:
                    # 原始残差远大于对偶残差，增加rho
                    if params['rho'] < params['rho_max'] / params['rho_tau']:
                        params['rho'] *= params['rho_tau']
                        print(f"基于残差增加rho至 {params['rho']:.4f}")
                        self.rho_update_count += 1
                elif residual_ratio < 1 / params['rho_mu']:
                    # 对偶残差远大于原始残差，减少rho
                    if params['rho'] > params['rho_min'] * params['rho_tau']:
                        params['rho'] /= params['rho_tau']
                        print(f"基于残差减少rho至 {params['rho']:.4f}")
                        self.rho_update_count += 1

        # 调整容差参数以提高效率
        if iterations > target_iter * 1.5:  # 收敛过慢
            # 适当放宽容差
            params['abs_tol'] = min(params['abs_tol'] * 2.0, 1e-2)
            params['rel_tol'] = min(params['rel_tol'] * 2.0, 1e-1)
            print(f"放宽容差: abs_tol={params['abs_tol']:.2e}, rel_tol={params['rel_tol']:.2e}")

        # 调整松弛参数alpha（过松弛可加速收敛）
        if iterations > target_iter * 1.3 and params['alpha'] < 1.8:
            params['alpha'] = min(params['alpha'] * 1.1, 1.8)
            print(f"增加松弛参数alpha至 {params['alpha']:.2f}")

        # 根据问题类型调整最大迭代次数
        if problem_type in ["low_rank_matrix_completion", "robust_multi_view_spectral_clustering"]:
            # 复杂问题需要更多迭代
            params['max_iter'] = max(1500, int(iterations * 1.5))
        else:
            params['max_iter'] = max(800, int(iterations * 1.2))

        # 确保参数在合理范围内
        params['rho'] = np.clip(params['rho'], params['rho_min'], params['rho_max'])
        params['alpha'] = np.clip(params['alpha'], 1.0, 2.0)
        params['abs_tol'] = np.clip(params['abs_tol'], 1e-8, 1e-1)
        params['rel_tol'] = np.clip(params['rel_tol'], 1e-6, 1.0)
        params['max_iter'] = max(100, min(params['max_iter'], 10000))

        # 更新当前参数
        self.current_params = params.copy()

        return params

    def reset_adaptation(self) -> None:
        """
        重置自适应调整状态
        """
        self.rho_update_count = 0
        print("ADMM参数自适应调整已重置")

    def get_recommended_parameters(self, problem_type: str = "general") -> Dict[str, float]:
        """
        根据问题类型获取推荐参数

        参数:
            problem_type: 问题类型

        返回:
            推荐参数字典
        """
        recommendations = self.current_params.copy()

        # 基于问题类型的特定推荐
        if problem_type in ["l1_regularization", "elastic_net", "l1_regression"]:
            recommendations.update({
                'rho': 0.1,
                'alpha': 1.0,
                'adaptive_rho': True,
                'max_iter': 1000,
            })
        elif problem_type in ["elastic_net_regression"]:
            recommendations.update({
                'rho': 0.5,
                'alpha': 1.5,
                'adaptive_rho': True,
                'max_iter': 800,
            })
        elif problem_type in ["low_rank_matrix_completion", "low_rank_representation"]:
            recommendations.update({
                'rho': 1.0,
                'alpha': 1.2,
                'adaptive_rho': True,
                'max_iter': 1500,
            })
        elif problem_type in ["robust_multi_view_spectral_clustering"]:
            recommendations.update({
                'rho': 5.0,
                'alpha': 1.0,
                'adaptive_rho': False,  # 复杂问题禁用自适应调整
                'max_iter': 2000,
            })

        return recommendations