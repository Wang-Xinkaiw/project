from strategies.base_strategy import BaseTuningStrategy
import numpy as np

import numpy as np
from typing import Dict, Any, Optional, Union

class BaseTuningStrategy:
    """基类，用于定义参数调优策略的接口"""
    def __init__(self, initial_params: Dict[str, Any] = None):
        self.parameters = initial_params or {}

    def update_parameters(self, performance_info: Dict[str, Any]) -> None:
        """根据性能信息更新参数"""
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        """获取当前参数"""
        return self.parameters

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置参数"""
        self.parameters = params

class RobustADMMStrategy(BaseTuningStrategy):
    """
    鲁棒的ADMM参数调优策略

    针对反馈中的类型错误进行了优化，确保参数类型一致性，并提高算法稳定性。
    主要改进：
    1. 参数类型验证和自动转换
    2. 动态调整rho参数以加速收敛
    3. 自适应停止准则
    4. 错误处理和边界检查
    """

    def __init__(self, initial_params: Dict[str, Any] = None):
        """
        初始化策略

        参数:
            initial_params: 初始参数字典，包含以下键值：
                - rho: 惩罚参数（默认1.0）
                - alpha: 过松弛参数（默认1.5）
                - abs_tol: 绝对容差（默认1e-4）
                - rel_tol: 相对容差（默认1e-2）
                - max_iter: 最大迭代次数（默认1000）
                - adaptive_rho: 是否自适应调整rho（默认True）
                - rho_tau: rho调整因子（默认2.0）
                - rho_min: rho最小值（默认1e-6）
                - rho_max: rho最大值（默认1e6）
        """
        # 默认参数
        default_params = {
            'rho': 1.0,
            'alpha': 1.5,
            'abs_tol': 1e-4,
            'rel_tol': 1e-2,
            'max_iter': 1000,
            'adaptive_rho': True,
            'rho_tau': 2.0,
            'rho_min': 1e-6,
            'rho_max': 1e6,
            'convergence_history': []
        }

        # 合并用户提供的参数
        if initial_params:
            # 确保参数类型正确（解决反馈中的类型错误）
            cleaned_params = self._clean_parameters(initial_params)
            default_params.update(cleaned_params)

        super().__init__(default_params)

        # 初始化内部状态
        self.iteration_count = 0
        self.best_performance = float('inf')
        self.performance_history = []

    def _clean_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理参数，确保所有数值参数都是正确的类型

        参数:
            params: 原始参数字典

        返回:
            清理后的参数字典
        """
        cleaned = {}
        type_mapping = {
            'rho': float,
            'alpha': float,
            'abs_tol': float,
            'rel_tol': float,
            'max_iter': int,
            'adaptive_rho': bool,
            'rho_tau': float,
            'rho_min': float,
            'rho_max': float
        }

        for key, value in params.items():
            if key in type_mapping:
                try:
                    # 显式类型转换，避免字符串与数字的比较错误
                    if type_mapping[key] == bool:
                        # 处理布尔值
                        if isinstance(value, str):
                            cleaned[key] = value.lower() in ('true', '1', 'yes', 't')
                        else:
                            cleaned[key] = bool(value)
                    else:
                        cleaned[key] = type_mapping[key](value)
                except (ValueError, TypeError):
                    # 如果转换失败，使用默认值或保持原值
                    print(f"Warning: Parameter {key} with value {value} cannot be converted to {type_mapping[key]}. Skipping.")
            else:
                # 对于未知参数，直接复制
                cleaned[key] = value

        return cleaned

    def _validate_parameters(self) -> None:
        """
        验证参数的有效性

        抛出:
            ValueError: 如果参数无效
        """
        params = self.parameters

        # 检查rho范围
        if not (params['rho_min'] <= params['rho'] <= params['rho_max']):
            raise ValueError(f"rho must be between {params['rho_min']} and {params['rho_max']}")

        # 检查alpha范围（过松弛参数通常在1.0-1.8之间）
        if not (1.0 <= params['alpha'] <= 1.8):
            raise ValueError(f"alpha must be between 1.0 and 1.8, got {params['alpha']}")

        # 检查容差参数
        if params['abs_tol'] <= 0 or params['rel_tol'] <= 0:
            raise ValueError("Tolerance parameters must be positive")

        # 检查最大迭代次数
        if params['max_iter'] <= 0:
            raise ValueError("max_iter must be positive")

    def _adjust_rho(self, primal_residual: float, dual_residual: float) -> float:
        """
        自适应调整rho参数

        根据原始残差和对偶残差的比例调整rho，以加速收敛

        参数:
            primal_residual: 原始残差
            dual_residual: 对偶残差

        返回:
            调整后的rho值
        """
        if not self.parameters['adaptive_rho']:
            return self.parameters['rho']

        tau = self.parameters['rho_tau']
        rho_min = self.parameters['rho_min']
        rho_max = self.parameters['rho_max']
        current_rho = self.parameters['rho']

        # 避免除零错误
        if dual_residual == 0 or primal_residual == 0:
            return current_rho

        # 计算残差比例
        ratio = primal_residual / (np.sqrt(len(self.performance_history) + 1) * dual_residual)

        # 根据比例调整rho
        if ratio > 10:
            # 原始残差太大，增加rho
            new_rho = current_rho * tau
        elif ratio < 0.1:
            # 对偶残差太大，减小rho
            new_rho = current_rho / tau
        else:
            # 比例合适，保持当前rho
            return current_rho

        # 确保rho在合理范围内
        new_rho = max(rho_min, min(rho_max, new_rho))

        return new_rho

    def update_parameters(self, performance_info: Dict[str, Any]) -> None:
        """
        根据性能信息更新ADMM参数

        参数:
            performance_info: 包含性能信息的字典，应包含：
                - iterations: 当前迭代次数
                - primal_residual: 原始残差（可选）
                - dual_residual: 对偶残差（可选）
                - objective_value: 目标函数值（可选）
                - converged: 是否收敛（可选）
        """
        try:
            # 验证输入类型
            if not isinstance(performance_info, dict):
                raise TypeError("performance_info must be a dictionary")

            # 提取性能信息，确保类型正确
            iterations = int(performance_info.get('iterations', 0))
            primal_residual = float(performance_info.get('primal_residual', 0.0))
            dual_residual = float(performance_info.get('dual_residual', 0.0))
            objective_value = float(performance_info.get('objective_value', 0.0))
            converged = bool(performance_info.get('converged', False))

            # 更新内部状态
            self.iteration_count = iterations
            current_perf = {
                'iterations': iterations,
                'primal_residual': primal_residual,
                'dual_residual': dual_residual,
                'objective_value': objective_value,
                'converged': converged
            }
            self.performance_history.append(current_perf)

            # 限制历史记录长度
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

            # 更新最佳性能
            if objective_value < self.best_performance:
                self.best_performance = objective_value

            # 根据收敛情况调整参数
            if len(self.performance_history) >= 3:
                # 检查最近几次迭代的收敛趋势
                recent_primal = [p['primal_residual'] for p in self.performance_history[-3:]]
                recent_dual = [p['dual_residual'] for p in self.performance_history[-3:]]

                # 计算残差下降率
                primal_decline = (recent_primal[0] - recent_primal[-1]) / max(recent_primal[0], 1e-10)
                dual_decline = (recent_dual[0] - recent_dual[-1]) / max(recent_dual[0], 1e-10)

                # 自适应调整rho
                if self.parameters['adaptive_rho'] and primal_residual > 0 and dual_residual > 0:
                    new_rho = self._adjust_rho(primal_residual, dual_residual)
                    if new_rho != self.parameters['rho']:
                        self.parameters['rho'] = new_rho
                        print(f"Adaptively adjusted rho to {new_rho:.4f}")

                # 如果收敛缓慢，调整容差
                if iterations > self.parameters['max_iter'] * 0.7 and not converged:
                    # 放宽容差以允许更早停止
                    self.parameters['abs_tol'] = min(
                        self.parameters['abs_tol'] * 1.5,
                        1e-2  # 最大容差
                    )
                    print(f"Relaxed tolerance to {self.parameters['abs_tol']:.2e}")

                # 如果收敛过快，收紧容差以提高精度
                if iterations < self.parameters['max_iter'] * 0.3 and converged:
                    self.parameters['abs_tol'] = max(
                        self.parameters['abs_tol'] * 0.8,
                        1e-6  # 最小容差
                    )
                    print(f"Tightened tolerance to {self.parameters['abs_tol']:.2e}")

            # 验证参数有效性
            self._validate_parameters()

            # 更新收敛历史
            conv_info = {
                'iteration': iterations,
                'primal_residual': primal_residual,
                'dual_residual': dual_residual,
                'rho': self.parameters['rho']
            }
            self.parameters['convergence_history'].append(conv_info)

        except KeyError as e:
            print(f"Warning: Missing key in performance_info: {e}")
        except (ValueError, TypeError) as e:
            print(f"Warning: Error processing performance_info: {e}")
            # 回退到安全参数
            self.parameters.update({
                'rho': 1.0,
                'alpha': 1.5,
                'abs_tol': 1e-4,
                'rel_tol': 1e-2
            })

    def get_parameters(self) -> Dict[str, Any]:
        """
        获取当前参数

        返回:
            参数字典，确保所有数值都是正确的类型
        """
        # 返回清理后的参数副本
        return self._clean_parameters(self.parameters.copy())

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置参数

        参数:
            params: 新的参数字典
        """
        try:
            # 清理并验证参数
            cleaned_params = self._clean_parameters(params)

            # 更新参数
            self.parameters.update(cleaned_params)

            # 验证参数有效性
            self._validate_parameters()

            # 重置部分状态
            if 'rho' in params or 'alpha' in params:
                self.performance_history = []
                self.best_performance = float('inf')

        except (ValueError, TypeError) as e:
            print(f"Error setting parameters: {e}")
            # 部分更新，只更新有效的参数
            for key, value in params.items():
                if key in self.parameters:
                    try:
                        self.parameters[key] = self._clean_parameters({key: value})[key]
                    except:
                        pass  # 保持原值

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        获取诊断信息

        返回:
            包含诊断信息的字典
        """
        if not self.performance_history:
            return {}

        recent = self.performance_history[-1]
        return {
            'current_iteration': recent['iterations'],
            'primal_residual': recent['primal_residual'],
            'dual_residual': recent['dual_residual'],
            'current_rho': self.parameters['rho'],
            'performance_history_length': len(self.performance_history),
            'best_objective': self.best_performance
        }