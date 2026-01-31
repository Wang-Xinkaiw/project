from strategies.base_strategy import BaseTuningStrategy
import numpy as np

import numpy as np

# 假设存在BaseTuningStrategy基类
class BaseTuningStrategy:
    """参数调优策略基类"""
    def __init__(self, initial_params=None):
        if initial_params is None:
            initial_params = {}
        self.params = initial_params

    def update_parameters(self, state_dict):
        """根据状态信息更新参数

        Args:
            state_dict: 包含算法状态的字典，通常包括：
                - primal_residual: 原始残差范数
                - dual_residual: 对偶残差范数
                - iteration: 当前迭代次数
                - primal_tolerance: 原始残差容忍度
                - dual_tolerance: 对偶残差容忍度
                - problem_scale: 问题规模指示器（如变量数量或残差范数初始值）

        Returns:
            dict: 更新后的参数
        """
        raise NotImplementedError

    def get_parameters(self):
        """获取当前参数"""
        return self.params

    def set_parameters(self, params):
        """设置参数"""
        self.params.update(params)


class AdaptiveBetaADMMStrategy(BaseTuningStrategy):
    """ADMM自适应惩罚参数β调整策略

    该策略基于原始残差和对偶残差动态调整惩罚参数β，
    目标是在保持稳定性的同时实现快速收敛。

    数学原理:
    1. 当原始残差远大于对偶残差时，增加β以加强原始可行性
    2. 当对偶残差远大于原始残差时，减少β以加强对偶可行性
    3. 使用平滑调整避免振荡，考虑问题规模实现泛化

    Attributes:
        params (dict): 包含调优参数的字典
            - beta: 当前惩罚参数值
            - beta_min: β的最小允许值
            - beta_max: β的最大允许值
            - mu: 残差比例阈值（通常>1）
            - tau: 增长/缩减因子（通常>1）
            - smoothing_factor: 平滑因子（0-1之间）
            - scale_factor: 问题规模缩放因子
    """

    def __init__(self, initial_params=None):
        """初始化自适应β调整策略

        Args:
            initial_params: 初始参数配置，可包含：
                - beta: 初始惩罚参数（默认1.0）
                - beta_min: β最小值（默认1e-6）
                - beta_max: β最大值（默认1e6）
                - mu: 残差比例阈值（默认10）
                - tau: 调整因子（默认2.0）
                - smoothing_factor: 平滑因子（默认0.5）
                - adaptation_interval: 调整间隔（默认5次迭代）
        """
        if initial_params is None:
            initial_params = {}

        # 设置默认参数
        default_params = {
            'beta': 1.0,
            'beta_min': 1e-6,
            'beta_max': 1e6,
            'mu': 10.0,  # 残差比例阈值
            'tau': 2.0,  # 增长/缩减因子
            'smoothing_factor': 0.5,  # 平滑因子
            'adaptation_interval': 5,  # 调整间隔
            'last_beta_update': 0,  # 上次β更新时间
            'prev_primal_residual': None,  # 上一次原始残差（用于平滑）
            'prev_dual_residual': None,  # 上一次对偶残差（用于平滑）
            'initial_primal_residual': None,  # 初始原始残差（用于缩放）
            'initial_dual_residual': None,  # 初始对偶残差（用于缩放）
        }

        # 合并初始参数
        default_params.update(initial_params)
        super().__init__(default_params)

    def update_parameters(self, state_dict):
        """根据算法状态更新β参数

        更新策略：
        1. 计算归一化残差（考虑问题规模）
        2. 计算残差比例 r/s
        3. 如果 r > μ*s，增加β以加强原始可行性
        4. 如果 s > μ*r，减少β以加强对偶可行性
        5. 应用平滑调整避免振荡
        6. 确保β在合理范围内

        Args:
            state_dict: 包含算法状态的字典

        Returns:
            dict: 包含更新后参数的字典

        Raises:
            ValueError: 如果状态字典缺少必要信息
            KeyError: 如果状态字典键错误
        """
        try:
            # 参数验证
            self._validate_state_dict(state_dict)

            # 获取当前状态
            iteration = state_dict.get('iteration', 0)
            primal_residual = state_dict['primal_residual']
            dual_residual = state_dict['dual_residual']

            # 初始化残差记录（第一次迭代时）
            if self.params['initial_primal_residual'] is None:
                self.params['initial_primal_residual'] = primal_residual
            if self.params['initial_dual_residual'] is None:
                self.params['initial_dual_residual'] = dual_residual

            # 应用指数移动平均平滑残差
            primal_smooth = self._apply_smoothing(
                primal_residual, 'prev_primal_residual'
            )
            dual_smooth = self._apply_smoothing(
                dual_residual, 'prev_dual_residual'
            )

            # 计算归一化残差（考虑问题规模）
            primal_norm = self._normalize_residual(
                primal_smooth,
                self.params['initial_primal_residual'],
                state_dict
            )
            dual_norm = self._normalize_residual(
                dual_smooth,
                self.params['initial_dual_residual'],
                state_dict
            )

            # 只在调整间隔到达时更新β
            if (iteration - self.params['last_beta_update'] >=
                self.params['adaptation_interval']):

                # 计算残差比例（避免除零）
                if dual_norm > 0:
                    residual_ratio = primal_norm / dual_norm
                else:
                    residual_ratio = float('inf') if primal_norm > 0 else 1.0

                # 根据残差比例调整β
                beta = self.params['beta']
                mu = self.params['mu']
                tau = self.params['tau']

                if residual_ratio > mu:
                    # 原始残差过大，增加β以加强原始可行性
                    # β_new = β * τ * min(residual_ratio/mu, τ)
                    # 使用min避免过大的调整
                    adjustment = tau * min(residual_ratio / mu, tau)
                    beta *= adjustment
                elif residual_ratio < 1.0 / mu:
                    # 对偶残差过大，减少β以加强对偶可行性
                    # β_new = β / (τ * min(mu * residual_ratio, τ))
                    adjustment = tau * min(mu * residual_ratio, tau)
                    if adjustment > 0:
                        beta /= adjustment

                # 应用平滑：β_new = α*β_old + (1-α)*β_calculated
                smoothing = self.params['smoothing_factor']
                beta = smoothing * self.params['beta'] + (1 - smoothing) * beta

                # 确保β在合理范围内
                beta = np.clip(
                    beta,
                    self.params['beta_min'],
                    self.params['beta_max']
                )

                # 更新参数
                self.params['beta'] = beta
                self.params['last_beta_update'] = iteration

            return {'beta': self.params['beta']}

        except KeyError as e:
            raise KeyError(f"状态字典缺少必要键: {e}")
        except ValueError as e:
            raise ValueError(f"状态验证失败: {e}")
        except Exception as e:
            # 记录错误但返回当前参数
            print(f"参数更新过程中发生错误: {e}")
            return {'beta': self.params['beta']}

    def get_parameters(self):
        """获取当前所有参数"""
        return self.params.copy()

    def set_parameters(self, params):
        """设置参数

        Args:
            params: 要设置的参数字典
        """
        # 只更新存在的参数，保持其他参数不变
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
            else:
                print(f"警告: 忽略未知参数 '{key}'")

    def _validate_state_dict(self, state_dict):
        """验证状态字典的有效性

        Args:
            state_dict: 要验证的状态字典

        Raises:
            ValueError: 如果状态无效
        """
        required_keys = ['primal_residual', 'dual_residual']
        for key in required_keys:
            if key not in state_dict:
                raise ValueError(f"状态字典必须包含 '{key}'")

        # 验证残差非负
        if state_dict['primal_residual'] < 0:
            raise ValueError("原始残差必须为非负")
        if state_dict['dual_residual'] < 0:
            raise ValueError("对偶残差必须为非负")

        # 验证迭代次数
        iteration = state_dict.get('iteration', 0)
        if iteration < 0:
            raise ValueError("迭代次数必须为非负")

    def _apply_smoothing(self, current_value, prev_key):
        """应用指数移动平均平滑

        Args:
            current_value: 当前值
            prev_key: 存储上一次值的键名

        Returns:
            float: 平滑后的值
        """
        prev_value = self.params[prev_key]

        if prev_value is None:
            # 第一次迭代，直接使用当前值
            smoothed = current_value
        else:
            # 应用平滑: EMA = α * current + (1-α) * previous
            alpha = 0.3  # 平滑系数，可调整
            smoothed = alpha * current_value + (1 - alpha) * prev_value

        # 更新存储的上一次值
        self.params[prev_key] = smoothed

        return smoothed

    def _normalize_residual(self, residual, initial_residual, state_dict):
        """归一化残差，考虑问题规模

        归一化方法：
        1. 如果提供了初始残差，使用初始残差进行缩放
        2. 如果提供了问题规模指示器，考虑规模因素
        3. 否则返回原始残差

        Args:
            residual: 要归一化的残差
            initial_residual: 初始残差
            state_dict: 状态字典

        Returns:
            float: 归一化后的残差
        """
        # 方法1: 使用初始残差归一化
        if initial_residual is not None and initial_residual > 0:
            normalized = residual / initial_residual
        else:
            normalized = residual

        # 方法2: 考虑问题规模（可选）
        problem_scale = state_dict.get('problem_scale')
        if problem_scale is not None and problem_scale > 0:
            # 根据问题规模进行额外调整
            # 例如，对于大规模问题，可能需要更保守的调整
            scale_factor = np.sqrt(problem_scale)  # 平方根缩放是常见选择
            normalized /= scale_factor

        return normalized