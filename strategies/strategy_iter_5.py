from strategies.base_strategy import BaseTuningStrategy
import numpy as np

import numpy as np
from typing import Dict, Any, Optional, List, Union

class BaseTuningStrategy:
    """基类，定义参数调优策略的接口"""
    def __init__(self, initial_params: Dict[str, Any]):
        self.parameters = initial_params

    def update_parameters(self, feedback: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.parameters.update(params)

class AdaptiveADMMStrategy(BaseTuningStrategy):
    """
    改进的自适应ADMM参数调优策略

    针对反馈中的类型错误进行优化：
    1. 增强参数类型验证和转换
    2. 改进自适应调整逻辑
    3. 添加错误恢复机制

    主要调整参数：
    - rho: ADMM惩罚参数
    - alpha: 过松弛参数 (通常1.0-1.8)
    - max_iter: 最大迭代次数
    - abstol, reltol: 绝对和相对容忍度

    策略特点：
    1. 基于历史性能的自适应调整
    2. 参数边界保护和类型安全
    3. 多问题泛化能力
    4. 收敛性保证机制
    """

    def __init__(self, initial_params: Optional[Dict[str, Any]] = None):
        """
        初始化ADMM调优策略

        Args:
            initial_params: 初始参数，如果为None则使用默认参数
        """
        # 默认参数设置
        default_params = {
            'rho': 1.0,
            'alpha': 1.6,
            'max_iter': 1000,
            'abstol': 1e-4,
            'reltol': 1e-2,
            'adaptive_rho': True,
            'adaptive_alpha': True,
            'rho_tau': 2.0,
            'rho_mu': 10.0,
            'min_rho': 1e-6,
            'max_rho': 1e6,
            'min_alpha': 1.0,
            'max_alpha': 1.99
        }

        # 合并用户提供的参数
        if initial_params:
            # 类型转换和验证
            validated_params = self._validate_parameters(initial_params)
            default_params.update(validated_params)

        super().__init__(default_params)

        # 性能跟踪
        self.iteration_history = []
        self.convergence_history = []
        self.failure_count = 0
        self.success_count = 0

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证并转换参数类型，确保数值类型正确

        Args:
            params: 待验证的参数

        Returns:
            验证后的参数字典
        """
        validated = {}

        # 预定义的参数类型映射
        type_map = {
            'rho': float,
            'alpha': float,
            'max_iter': int,
            'abstol': float,
            'reltol': float,
            'adaptive_rho': bool,
            'adaptive_alpha': bool,
            'rho_tau': float,
            'rho_mu': float,
            'min_rho': float,
            'max_rho': float,
            'min_alpha': float,
            'max_alpha': float
        }

        for key, value in params.items():
            if key in type_map:
                expected_type = type_map[key]
                try:
                    # 尝试转换为目标类型
                    if expected_type == bool:
                        # 处理布尔值
                        if isinstance(value, str):
                            validated[key] = value.lower() in ('true', '1', 'yes', 't')
                        else:
                            validated[key] = bool(value)
                    else:
                        validated[key] = expected_type(value)
                except (ValueError, TypeError):
                    # 转换失败，使用默认值或原值
                    print(f"Warning: Parameter {key} cannot be converted to {expected_type}, using default")
                    continue
            else:
                # 未知参数，保持原样
                validated[key] = value

        return validated

    def _safe_float_comparison(self, a: Any, b: Any) -> bool:
        """
        安全的浮点数比较，避免类型错误

        Args:
            a: 第一个值
            b: 第二个值

        Returns:
            比较结果
        """
        try:
            # 确保两个值都是数值类型
            a_float = float(a) if not isinstance(a, (int, float)) else a
            b_float = float(b) if not isinstance(b, (int, float)) else b
            return a_float < b_float
        except (ValueError, TypeError):
            # 如果转换失败，使用保守策略
            return False

    def _update_rho_adaptive(self, primal_residual: float, dual_residual: float) -> float:
        """
        根据残差自适应调整rho参数

        Args:
            primal_residual: 原始残差
            dual_residual: 对偶残差

        Returns:
            调整后的rho值
        """
        current_rho = self.parameters['rho']
        rho_tau = self.parameters['rho_tau']
        rho_mu = self.parameters['rho_mu']
        min_rho = self.parameters['min_rho']
        max_rho = self.parameters['max_rho']

        # 安全的数值比较
        if self._safe_float_comparison(rho_mu * dual_residual, primal_residual):
            # 原始残差远大于对偶残差，增加rho
            new_rho = current_rho * rho_tau
        elif self._safe_float_comparison(rho_mu * primal_residual, dual_residual):
            # 对偶残差远大于原始残差，减少rho
            new_rho = current_rho / rho_tau
        else:
            new_rho = current_rho

        # 边界保护
        new_rho = max(min_rho, min(max_rho, new_rho))

        return new_rho

    def _update_alpha_adaptive(self, convergence_rate: float) -> float:
        """
        根据收敛速度自适应调整alpha参数

        Args:
            convergence_rate: 收敛速率

        Returns:
            调整后的alpha值
        """
        current_alpha = self.parameters['alpha']
        min_alpha = self.parameters['min_alpha']
        max_alpha = self.parameters['max_alpha']

        # 简单的启发式调整
        if convergence_rate > 0.9:
            # 收敛太快，可能过松弛，稍微减小alpha
            new_alpha = current_alpha * 0.95
        elif convergence_rate < 0.1:
            # 收敛太慢，增加过松弛
            new_alpha = current_alpha * 1.05
        else:
            new_alpha = current_alpha

        # 边界保护
        new_alpha = max(min_alpha, min(max_alpha, new_alpha))

        return new_alpha

    def update_parameters(self, feedback: Dict[str, Any],
                         history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        根据反馈和历史信息更新ADMM参数

        Args:
            feedback: 反馈信息，包含性能指标和错误信息
            history: 历史调优记录

        Returns:
            更新后的参数
        """
        # 验证反馈信息
        if not isinstance(feedback, dict):
            print("Warning: Invalid feedback format")
            return self.get_parameters()

        # 分析反馈信息
        has_error = False
        error_messages = []

        # 检查反馈中的错误信息
        for key, value in feedback.items():
            if '错误' in str(key) or 'error' in str(key).lower() or 'fail' in str(key).lower():
                has_error = True
                error_messages.append(str(value))

        # 处理错误情况
        if has_error:
            self.failure_count += 1
            self.success_count = 0

            # 分析错误类型
            for error_msg in error_messages:
                if "'<' not supported between instances of 'float' and 'str'" in error_msg:
                    # 类型错误，重置参数并加强类型检查
                    print(f"Detected type error: {error_msg}")
                    self.parameters = self._validate_parameters(self.parameters)

                    # 保守调整：减小步长，增加稳定性
                    self.parameters['rho'] = max(0.1, self.parameters['rho'] * 0.5)
                    self.parameters['alpha'] = max(1.0, self.parameters['alpha'] * 0.9)

                elif 'ADMM评估失败' in error_msg:
                    # 其他ADMM失败，尝试更保守的参数
                    self.parameters['rho'] = self.parameters['rho'] * 0.8
                    self.parameters['max_iter'] = min(2000, int(self.parameters['max_iter'] * 1.2))

            # 如果连续失败太多，重置到默认值
            if self.failure_count >= 3:
                print("Multiple failures detected, resetting to conservative parameters")
                self.parameters['rho'] = 1.0
                self.parameters['alpha'] = 1.0
                self.parameters['abstol'] = 1e-3
                self.parameters['reltol'] = 1e-1
                self.failure_count = 0

        else:
            # 成功的情况，基于性能调整
            self.success_count += 1
            self.failure_count = 0

            # 从反馈中提取性能指标
            avg_iterations = feedback.get('平均迭代次数', feedback.get('average_iterations', 1000))
            if isinstance(avg_iterations, str):
                try:
                    avg_iterations = float(avg_iterations)
                except ValueError:
                    avg_iterations = 1000

            improvement = feedback.get('改进', feedback.get('improvement', 0))
            if isinstance(improvement, str):
                try:
                    improvement = float(improvement.replace('%', ''))
                except ValueError:
                    improvement = 0

            # 记录历史
            self.iteration_history.append(avg_iterations)
            if len(self.iteration_history) > 10:
                self.iteration_history.pop(0)

            # 计算平均迭代次数
            if self.iteration_history:
                avg_history = np.mean(self.iteration_history)
            else:
                avg_history = avg_iterations

            # 自适应调整策略
            if self.parameters.get('adaptive_rho', True):
                # 这里简化处理，实际应用中应从反馈获取残差信息
                primal_residual = feedback.get('primal_residual', 1e-3)
                dual_residual = feedback.get('dual_residual', 1e-3)

                # 确保是数值类型
                try:
                    primal_residual = float(primal_residual)
                    dual_residual = float(dual_residual)
                except (ValueError, TypeError):
                    primal_residual = 1e-3
                    dual_residual = 1e-3

                self.parameters['rho'] = self._update_rho_adaptive(primal_residual, dual_residual)

            if self.parameters.get('adaptive_alpha', True):
                # 计算收敛速率（简化）
                convergence_rate = 1000.0 / avg_iterations if avg_iterations > 0 else 0.1
                self.parameters['alpha'] = self._update_alpha_adaptive(convergence_rate)

            # 根据迭代次数调整最大迭代次数
            if avg_iterations > 800:
                self.parameters['max_iter'] = min(5000, int(self.parameters['max_iter'] * 1.2))
            elif avg_iterations < 200:
                self.parameters['max_iter'] = max(500, int(self.parameters['max_iter'] * 0.9))

            # 根据改进情况调整容忍度
            if improvement <= 0:
                # 没有改进，放宽容忍度
                self.parameters['abstol'] = min(1e-2, self.parameters['abstol'] * 1.5)
                self.parameters['reltol'] = min(0.1, self.parameters['reltol'] * 1.5)
            elif improvement > 5:
                # 显著改进，收紧容忍度
                self.parameters['abstol'] = max(1e-6, self.parameters['abstol'] * 0.8)
                self.parameters['reltol'] = max(1e-4, self.parameters['reltol'] * 0.8)

        # 最终验证所有参数
        self.parameters = self._validate_parameters(self.parameters)

        return self.get_parameters()

    def get_parameters(self) -> Dict[str, Any]:
        """获取当前参数"""
        # 返回参数的深拷贝，确保类型安全
        params = super().get_parameters()
        return self._validate_parameters(params)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置参数，进行类型验证"""
        validated_params = self._validate_parameters(params)
        super().set_parameters(validated_params)

        # 更新性能计数器
        if 'reset_counters' in params and params['reset_counters']:
            self.failure_count = 0
            self.success_count = 0
            self.iteration_history = []
            self.convergence_history = []