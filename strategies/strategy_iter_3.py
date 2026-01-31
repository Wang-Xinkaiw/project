from strategies.base_strategy import BaseTuningStrategy
import numpy as np

import numpy as np
from abc import ABC, abstractmethod

# 基础策略类
class BaseTuningStrategy(ABC):
    """基础参数调优策略抽象基类"""

    @abstractmethod
    def update_parameters(self, feedback):
        """根据反馈信息更新参数

        Args:
            feedback: 包含评估结果的反馈信息字典
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """获取当前参数配置

        Returns:
            dict: 当前参数配置字典
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        """设置参数配置

        Args:
            parameters: 参数配置字典
        """
        pass


class ADMMImprovedStrategy(BaseTuningStrategy):
    """改进的ADMM参数调优策略

    针对反馈中的类型错误问题进行优化，增强参数验证和类型转换。
    支持多种ADMM变体问题的泛化调优。

    主要改进点:
    1. 严格的参数类型验证和转换
    2. 增强的错误处理机制
    3. 自适应的参数调整逻辑
    4. 更好的收敛性保证
    """

    def __init__(self, initial_parameters=None):
        """初始化ADMM改进策略

        Args:
            initial_parameters: 初始参数配置字典，可选
        """
        # 默认参数配置
        self.default_params = {
            'rho': 1.0,  # ADMM惩罚参数
            'alpha': 1.0,  # 松弛参数
            'max_iter': 1000,  # 最大迭代次数
            'abs_tol': 1e-4,  # 绝对容忍误差
            'rel_tol': 1e-2,  # 相对容忍误差
            'adaptive_rho': True,  # 是否自适应调整rho
            'rho_update_factor': 2.0,  # rho更新因子
            'rho_max': 1e6,  # rho最大值
            'rho_min': 1e-6,  # rho最小值
            'convergence_window': 10,  # 收敛判断窗口大小
            'verbose': False  # 详细输出
        }

        # 当前参数配置
        self.current_params = self.default_params.copy()

        # 历史信息记录
        self.iteration_history = []
        self.convergence_history = []
        self.parameter_history = []

        # 如果提供了初始参数，则更新
        if initial_parameters:
            self.set_parameters(initial_parameters)

    def _validate_and_convert_parameters(self, params):
        """验证和转换参数类型

        Args:
            params: 待验证的参数字典

        Returns:
            dict: 验证和转换后的参数字典

        Raises:
            TypeError: 参数类型错误
            ValueError: 参数值错误
        """
        validated_params = {}

        # 参数类型映射和验证规则
        validation_rules = {
            'rho': {'type': (int, float, np.number), 'min': 1e-10, 'max': 1e10},
            'alpha': {'type': (int, float, np.number), 'min': 0.5, 'max': 1.5},
            'max_iter': {'type': int, 'min': 1, 'max': 100000},
            'abs_tol': {'type': (int, float, np.number), 'min': 1e-20, 'max': 1e2},
            'rel_tol': {'type': (int, float, np.number), 'min': 1e-20, 'max': 1e2},
            'adaptive_rho': {'type': bool},
            'rho_update_factor': {'type': (int, float, np.number), 'min': 1.1, 'max': 5.0},
            'rho_max': {'type': (int, float, np.number), 'min': 1e-4, 'max': 1e12},
            'rho_min': {'type': (int, float, np.number), 'min': 1e-12, 'max': 1e4},
            'convergence_window': {'type': int, 'min': 1, 'max': 100},
            'verbose': {'type': bool}
        }

        for key, value in params.items():
            if key not in validation_rules:
                # 对于未知参数，保留原值但记录警告
                if self.current_params.get('verbose', False):
                    print(f"警告: 未知参数 '{key}'，跳过验证")
                validated_params[key] = value
                continue

            rule = validation_rules[key]

            try:
                # 类型转换
                if rule['type'] == bool:
                    # 布尔值转换
                    if isinstance(value, str):
                        if value.lower() in ['true', 't', '1', 'yes', 'y']:
                            converted_value = True
                        elif value.lower() in ['false', 'f', '0', 'no', 'n']:
                            converted_value = False
                        else:
                            raise ValueError(f"无法将字符串 '{value}' 转换为布尔值")
                    else:
                        converted_value = bool(value)

                elif rule['type'] == int:
                    # 整数转换
                    if isinstance(value, str):
                        # 尝试从字符串转换
                        if '.' in value:
                            converted_value = int(float(value))
                        else:
                            converted_value = int(value)
                    else:
                        converted_value = int(value)

                elif isinstance(value, str):
                    # 数值型字符串转换
                    try:
                        converted_value = float(value)
                        # 如果是整数类型的参数，转换为整数
                        if rule['type'] == int:
                            converted_value = int(converted_value)
                    except ValueError:
                        raise TypeError(f"参数 '{key}' 的值 '{value}' 无法转换为数值类型")

                else:
                    # 其他类型，检查类型匹配
                    if not isinstance(value, rule['type']):
                        raise TypeError(f"参数 '{key}' 的类型错误，期望 {rule['type']}，得到 {type(value)}")
                    converted_value = value

                # 数值范围验证
                if 'min' in rule and 'max' in rule:
                    if converted_value < rule['min'] or converted_value > rule['max']:
                        raise ValueError(f"参数 '{key}' 的值 {converted_value} 超出范围 [{rule['min']}, {rule['max']}]")

                validated_params[key] = converted_value

            except (TypeError, ValueError) as e:
                # 参数验证失败，使用默认值
                default_val = self.default_params.get(key)
                print(f"警告: 参数 '{key}' 验证失败: {str(e)}，使用默认值 {default_val}")
                validated_params[key] = default_val

        return validated_params

    def _adaptive_parameter_update(self, performance_metrics):
        """自适应参数更新逻辑

        Args:
            performance_metrics: 性能指标字典
        """
        # 提取性能指标
        avg_iterations = performance_metrics.get('average_iterations', 1000)
        prev_iterations = performance_metrics.get('previous_iterations', 1000)

        # 检查收敛性
        convergence_rate = self._calculate_convergence_rate()

        # 自适应调整rho
        if self.current_params['adaptive_rho']:
            if convergence_rate < 0.1:
                # 收敛缓慢，增加rho
                new_rho = min(self.current_params['rho'] * self.current_params['rho_update_factor'],
                             self.current_params['rho_max'])
                self.current_params['rho'] = new_rho
            elif convergence_rate > 0.9:
                # 收敛过快，减少rho
                new_rho = max(self.current_params['rho'] / self.current_params['rho_update_factor'],
                             self.current_params['rho_min'])
                self.current_params['rho'] = new_rho

        # 根据迭代次数调整
        if avg_iterations > 800:
            # 迭代次数过多，增加rho以加速收敛
            new_rho = min(self.current_params['rho'] * 1.5, self.current_params['rho_max'])
            self.current_params['rho'] = new_rho

            # 适当放宽容忍度
            if self.current_params['abs_tol'] < 1e-3:
                self.current_params['abs_tol'] *= 1.5
                self.current_params['rel_tol'] *= 1.5

        elif avg_iterations < 100:
            # 迭代次数过少，减小rho以提高精度
            new_rho = max(self.current_params['rho'] / 1.5, self.current_params['rho_min'])
            self.current_params['rho'] = new_rho

            # 收紧容忍度
            if self.current_params['abs_tol'] > 1e-6:
                self.current_params['abs_tol'] /= 1.2
                self.current_params['rel_tol'] /= 1.2

    def _calculate_convergence_rate(self):
        """计算收敛率

        Returns:
            float: 收敛率，0到1之间的值
        """
        if len(self.convergence_history) < 2:
            return 0.5

        # 计算最近几次迭代的收敛速度
        window_size = min(self.current_params['convergence_window'], len(self.convergence_history))
        recent_values = self.convergence_history[-window_size:]

        # 计算收敛率（值越小收敛越快）
        if len(recent_values) >= 2:
            convergence_rate = np.mean(np.diff(recent_values) / recent_values[:-1])
            return abs(convergence_rate)

        return 0.5

    def update_parameters(self, feedback):
        """根据反馈信息更新参数

        Args:
            feedback: 包含评估结果的反馈信息字典
        """
        try:
            # 解析反馈信息
            if not feedback or not isinstance(feedback, dict):
                print("警告: 无效的反馈信息，跳过参数更新")
                return

            # 记录性能指标
            avg_iterations = feedback.get('average_iterations', 1000.0)
            self.iteration_history.append(avg_iterations)

            # 提取详细问题表现
            problem_performance = feedback.get('detailed_problem_performance', {})

            # 分析错误信息
            error_problems = []
            for problem, result in problem_performance.items():
                if '错误' in str(result) or '失败' in str(result):
                    error_problems.append(problem)

                    # 检查是否是类型错误
                    if 'not supported between instances of' in str(result) and 'float' in str(result) and 'str' in str(result):
                        print(f"检测到类型错误问题: {problem}")

            # 如果有类型错误，重置参数为默认值
            if error_problems:
                print(f"检测到错误的问题: {error_problems}")

                # 对于类型错误，确保参数是正确类型
                for key in ['rho', 'alpha', 'max_iter', 'abs_tol', 'rel_tol']:
                    if key in self.current_params:
                        # 强制转换为正确类型
                        if key in ['max_iter', 'convergence_window']:
                            self.current_params[key] = int(self.current_params[key])
                        else:
                            self.current_params[key] = float(self.current_params[key])

                # 调整参数以应对不同类型的问题
                self._adjust_for_problem_type(error_problems[0])

            # 更新参数历史
            self.parameter_history.append(self.current_params.copy())

            # 自适应参数更新
            performance_metrics = {
                'average_iterations': avg_iterations,
                'previous_iterations': self.iteration_history[-2] if len(self.iteration_history) > 1 else 1000,
                'error_count': len(error_problems)
            }

            self._adaptive_parameter_update(performance_metrics)

            # 确保参数在合理范围内
            self._enforce_parameter_bounds()

            # 记录收敛历史
            if 'convergence_rate' in feedback:
                self.convergence_history.append(feedback['convergence_rate'])

            if self.current_params['verbose']:
                print(f"参数更新完成。当前rho: {self.current_params['rho']:.4f}, "
                      f"平均迭代次数: {avg_iterations}")

        except Exception as e:
            print(f"参数更新过程中发生错误: {str(e)}")
            # 出错时使用默认参数
            self.current_params = self.default_params.copy()

    def _adjust_for_problem_type(self, problem_name):
        """根据问题类型调整参数

        Args:
            problem_name: 问题名称
        """
        # 针对不同问题类型的参数调整
        problem_specific_settings = {
            'l1_regularization': {'rho': 1.0, 'adaptive_rho': True},
            'elastic_net': {'rho': 1.0, 'alpha': 1.2},
            'l1_regression': {'rho': 1.5, 'abs_tol': 1e-5},
            'elastic_net_regression': {'rho': 1.2, 'alpha': 1.1},
            'low_rank_matrix_completion': {'rho': 0.1, 'adaptive_rho': False},
            'low_rank_representation': {'rho': 0.5, 'max_iter': 2000},
            'robust_multi_view_spectral_clustering': {'rho': 2.0, 'adaptive_rho': True}
        }

        if problem_name in problem_specific_settings:
            settings = problem_specific_settings[problem_name]
            self.current_params.update(settings)
            print(f"已应用问题特定参数设置: {problem_name}")

    def _enforce_parameter_bounds(self):
        """确保参数在合理范围内"""
        # 确保数值参数在合理范围内
        self.current_params['rho'] = np.clip(self.current_params['rho'],
                                           self.current_params['rho_min'],
                                           self.current_params['rho_max'])

        self.current_params['alpha'] = np.clip(self.current_params['alpha'], 0.5, 1.5)
        self.current_params['max_iter'] = max(1, min(100000, self.current_params['max_iter']))
        self.current_params['abs_tol'] = max(1e-20, min(1e2, self.current_params['abs_tol']))
        self.current_params['rel_tol'] = max(1e-20, min(1e2, self.current_params['rel_tol']))

    def get_parameters(self):
        """获取当前参数配置

        Returns:
            dict: 当前参数配置字典
        """
        # 返回参数的深拷贝，防止外部修改
        return self.current_params.copy()

    def set_parameters(self, parameters):
        """设置参数配置

        Args:
            parameters: 参数配置字典

        Raises:
            ValueError: 参数设置失败
        """
        if not parameters:
            return

        try:
            # 验证和转换参数
            validated_params = self._validate_and_convert_parameters(parameters)

            # 更新当前参数
            self.current_params.update(validated_params)

            # 确保参数边界
            self._enforce_parameter_bounds()

            if self.current_params['verbose']:
                print(f"参数设置成功: {validated_params}")

        except Exception as e:
            raise ValueError(f"参数设置失败: {str(e)}")

    def reset(self):
        """重置参数到默认值"""
        self.current_params = self.default_params.copy()
        self.iteration_history = []
        self.convergence_history = []
        self.parameter_history = []
        print("参数已重置为默认值")

    def get_statistics(self):
        """获取策略统计信息

        Returns:
            dict: 统计信息字典
        """
        return {
            'parameter_history_count': len(self.parameter_history),
            'iteration_history': self.iteration_history,
            'convergence_history': self.convergence_history,
            'current_parameters': self.current_params.copy()
        }