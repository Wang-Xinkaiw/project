"""
策略评估器模块
负责在真实问题上评估策略性能
"""

import importlib
import sys
import logging
import re
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path
import inspect

class StrategyEvaluator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 配置文件
        """
        self.config = config
        self.evaluator_config = config.get('evaluator', {
            'max_iterations': 500,
            'tolerance': 1e-6
        })
        
        # 验证 tolerance 是否为浮点数
        try:
            tolerance = self.evaluator_config.get('tolerance', 1e-6)
            self.evaluator_config['tolerance'] = float(tolerance)
        except (ValueError, TypeError):
            self.evaluator_config['tolerance'] = 1e-6
            logging.getLogger(__name__).warning(f"配置中的 tolerance 无法转换为浮点数，使用默认值 1e-6")
        
        self.logger = logging.getLogger(__name__)
        
    def _load_strategy_module(self, strategy_path: str):
        """动态加载策略模块 - 增强验证版本"""
        try:
            # 将策略文件路径添加到Python路径
            strategy_dir = Path(strategy_path).parent
            if str(strategy_dir) not in sys.path:
                sys.path.insert(0, str(strategy_dir))
                
            # 动态导入模块
            module_name = Path(strategy_path).stem
            spec = importlib.util.spec_from_file_location(module_name, strategy_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找策略类
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    attr.__name__.endswith('Strategy') and
                    attr.__name__ != 'BaseTuningStrategy'):
                    strategy_class = attr
                    break
                    
            if strategy_class is None:
                self.logger.error("在模块中未找到策略类")
                return None
                
            # 创建策略实例
            strategy_instance = strategy_class()
            
            # 验证策略实例的方法签名
            if not self._validate_strategy_instance(strategy_instance):
                self.logger.error(f"策略类 {strategy_class.__name__} 验证失败")
                return None
                
            self.logger.info(f"成功加载策略: {strategy_class.__name__}")
            return strategy_instance
            
        except Exception as e:
            self.logger.error(f"加载策略模块失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _validate_strategy_instance(self, strategy_instance) -> bool:
        """验证策略实例的方法签名"""
        try:
            # 检查是否有update_parameters方法
            if not hasattr(strategy_instance, 'update_parameters'):
                self.logger.error("策略实例缺少update_parameters方法")
                return False
            
            update_method = getattr(strategy_instance, 'update_parameters')
            
            # 获取方法签名
            sig = inspect.signature(update_method)
            params = list(sig.parameters.keys())
            
            self.logger.info(f"update_parameters方法签名: {params}")
            
            # 检查参数数量（绑定方法的签名中不包含self，所以至少应该有1个参数iteration_state）
            if len(params) < 1:
                self.logger.error(f"参数数量不足: {len(params)}个")
                return False
            
            # 检查是否有iteration_state参数
            if 'iteration_state' not in params:
                self.logger.warning(f"方法签名缺少iteration_state参数: {sig}")
                # 尝试修正：替换方法为包装器
                self._patch_strategy_method(strategy_instance)
                self.logger.info("已自动修正策略方法")
                return True  # 修正后返回True
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证策略实例失败: {e}")
            return False
    
    def _patch_strategy_method(self, strategy_instance):
        """修正策略方法签名 - 增强版，支持更多参数名变体"""
        original_method = strategy_instance.update_parameters
        
        def patched_method(iteration_state):
            """修正后的方法，将iteration_state映射到原有参数"""
            try:
                # 尝试调用原始方法，根据可能的参数名传递数据
                sig = inspect.signature(original_method)
                params = list(sig.parameters.keys())
                
                # 根据参数名构建参数
                call_kwargs = {}
                
                # 检查参数名并映射（扩展的映射表）
                for param_name in params:
                    if param_name == 'self':
                        continue
                    # 残差相关参数
                    elif param_name in ('residuals', 'residual', 'res'):
                        call_kwargs[param_name] = {
                            'primal': iteration_state.get('primal_residual', 0),
                            'dual': iteration_state.get('dual_residual', 0)
                        }
                    # 变量相关参数
                    elif param_name in ('variables', 'vars', 'x'):
                        call_kwargs[param_name] = {}
                    # 参数字典相关
                    elif param_name in ('params', 'parameters', 'p'):
                        call_kwargs[param_name] = {
                            'beta': iteration_state.get('beta', 1.0),
                            'iteration': iteration_state.get('iteration', 0)
                        }
                    # 状态字典相关
                    elif param_name in ('iteration_state', 'state', 's', 'info', 'ctx', 'context'):
                        call_kwargs[param_name] = iteration_state
                    # 迭代次数相关
                    elif param_name in ('iteration', 'iter', 'k', 'step', 'n', 't'):
                        call_kwargs[param_name] = iteration_state.get('iteration', 0)
                    # beta/惩罚参数相关
                    elif param_name in ('beta', 'rho', 'penalty', 'mu', 'lambda_param'):
                        call_kwargs[param_name] = iteration_state.get('beta', 1.0)
                    # 原始残差相关
                    elif param_name in ('primal_residual', 'primal', 'r_primal', 'primal_res', 'r_p'):
                        call_kwargs[param_name] = iteration_state.get('primal_residual', 0)
                    # 对偶残差相关
                    elif param_name in ('dual_residual', 'dual', 'r_dual', 'dual_res', 'r_d'):
                        call_kwargs[param_name] = iteration_state.get('dual_residual', 0)
                    # 目标函数相关
                    elif param_name in ('objective', 'obj', 'cost', 'loss', 'f'):
                        call_kwargs[param_name] = iteration_state.get('objective', 0)
                    # 收敛标志相关
                    elif param_name in ('converged', 'done', 'finished'):
                        call_kwargs[param_name] = iteration_state.get('converged', False)
                    else:
                        # 其他未知参数，尝试从iteration_state获取或设为None
                        call_kwargs[param_name] = iteration_state.get(param_name, None)
                
                # 调用原始方法
                result = original_method(**call_kwargs)
                
                # 确保结果包含beta
                if isinstance(result, dict) and 'beta' in result:
                    return result
                elif isinstance(result, dict):
                    # 结果是字典但没有beta，添加默认值
                    self.logger.warning(f"策略方法未返回beta，使用当前值")
                    result['beta'] = iteration_state.get('beta', 1.0)
                    return result
                elif isinstance(result, (int, float)):
                    # 结果直接是数值，假设是beta值
                    self.logger.warning(f"策略方法返回了数值而非字典，假设为beta值")
                    return {'beta': float(result)}
                else:
                    # 其他情况返回默认值
                    self.logger.warning(f"策略方法返回了意外类型: {type(result)}，使用默认值")
                    return {'beta': iteration_state.get('beta', 1.0)}
                    
            except Exception as e:
                self.logger.error(f"调用修正方法失败: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                return {'beta': iteration_state.get('beta', 1.0)}
        
        # 替换方法
        strategy_instance.update_parameters = patched_method
        
    def _load_problem(self, algorithm_type: str, problem_name: str):
        """加载测试问题"""
        try:
            problem_module_path = self.config['algorithms'][algorithm_type]['problem_module']
            problem_module = importlib.import_module(problem_module_path)
            
            if algorithm_type == 'admm':
                base_name = problem_name
                if problem_name.endswith('_problem'):
                    base_name = problem_name[:-8]
                
                self.logger.info(f"尝试加载ADMM问题: {problem_name} (基础名称: {base_name})")
                
                if hasattr(problem_module, 'get_admm_problem'):
                    try:
                        problem_instance = problem_module.get_admm_problem(base_name, seed=42)
                        if problem_instance:
                            self.logger.info(f"通过get_admm_problem成功加载: {base_name}")
                            return problem_instance
                    except Exception as e:
                        self.logger.warning(f"get_admm_problem失败: {e}")
                
                function_names = [problem_name, f"{base_name}_problem", base_name]
                for func_name in function_names:
                    if hasattr(problem_module, func_name):
                        problem_func = getattr(problem_module, func_name)
                        problem_instance = problem_func()
                        self.logger.info(f"通过函数调用加载问题: {func_name}")
                        return problem_instance
                
                self.logger.error(f"无法找到问题 {problem_name}")
                return None
            else:
                if hasattr(problem_module, problem_name):
                    problem_func = getattr(problem_module, problem_name)
                    return problem_func()
                else:
                    self.logger.error(f"问题 {problem_name} 不存在于模块 {problem_module_path}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"加载问题失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def _run_evaluation(self, strategy, problem, algorithm_type: str) -> Dict[str, Any]:
        """运行单个问题的评估"""
        try:
            if algorithm_type == 'admm':
                return self._evaluate_admm(strategy, problem)
            elif algorithm_type == 'gradient_descent':
                return self._evaluate_gradient_descent(strategy, problem)
            else:
                if hasattr(problem, 'admm_iteration'):
                    return self._evaluate_admm(strategy, problem)
                elif hasattr(problem, 'gradient_descent_step'):
                    return self._evaluate_gradient_descent(strategy, problem)
                else:
                    return self._evaluate_generic(strategy, problem)
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            self.logger.error(traceback_str)
            return {
                'error': f"{str(e)}\n{traceback_str}",
                'iterations': 0,
                'converged': False,
                'final_objective': float('inf')
            }
        
    def _evaluate_admm(self, strategy, problem) -> Dict[str, Any]:
        """评估ADMM策略 - 修改为只使用beta参数"""
        try:
            # 初始化算法参数 - 只使用beta
            beta = self.config['algorithms']['admm']['default_beta']
            
            # 初始化状态
            iteration_state = {
                'iteration': 0,
                'primal_residual': None,
                'dual_residual': None,
                'beta': beta,
                'converged': False
            }
            
            # 运行算法直到收敛或达到最大迭代次数
            iterations = 0
            convergence_history = []
            
            max_iterations = self.evaluator_config.get('max_iterations', 500)
            
            while (not iteration_state.get('converged', False) and 
                iterations < max_iterations):
                
                # 更新参数 - 只获取beta参数
                new_params = strategy.update_parameters(iteration_state)
                
                # 【重要】只使用beta参数，忽略其他参数
                if 'beta' in new_params:
                    iteration_state['beta'] = new_params['beta']
                
                # 执行一次ADMM迭代
                iteration_result = problem.admm_iteration(
                    beta=iteration_state['beta'],  # 只传递beta
                    iteration=iterations
                )
                
                # 获取残差值并安全转换
                primal_res_val = iteration_result.get('primal_residual', 0)
                dual_res_val = iteration_result.get('dual_residual', 0)
                objective_val = iteration_result.get('objective', 0)
                
                # 更新状态
                iteration_state.update({
                    'iteration': iterations,
                    'primal_residual': self._safe_to_float(primal_res_val),
                    'dual_residual': self._safe_to_float(dual_res_val),
                    'objective': self._safe_to_float(objective_val),
                    'converged': iteration_result.get('converged', False)
                })
                
                # 检查收敛
                convergence_criteria = self._check_convergence(iteration_state, problem)
                iteration_state['converged'] = convergence_criteria['converged']
                
                convergence_history.append({
                    'iteration': iterations,
                    'primal_residual': iteration_state['primal_residual'],
                    'dual_residual': iteration_state['dual_residual'],
                    'beta': iteration_state['beta'],
                    'objective': iteration_state['objective']
                })
                
                iterations += 1
            
            # 计算性能指标
            final_objective = iteration_state.get('objective', 0)
            converged = iteration_state.get('converged', False)
            
            return {
                'iterations': iterations,
                'converged': converged,
                'final_objective': final_objective,
                'final_beta': iteration_state['beta'],  # 只记录beta
                'convergence_history': convergence_history[:10]
            }
            
        except Exception as e:
            self.logger.error(f"ADMM评估失败: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            self.logger.error(traceback_str)
            return {
                'error': f"ADMM评估失败: {str(e)}", 
                'iterations': self.evaluator_config.get('max_iterations', 500),
                'converged': False,
                'final_objective': float('inf')
            }
                
    def _evaluate_gradient_descent(self, strategy, problem) -> Dict[str, Any]:
        """评估梯度下降策略"""
        try:
            # 初始化参数
            learning_rate = self.config['algorithms']['gradient_descent'].get('default_lr', 0.01)
            
            # 初始化状态
            iteration_state = {
                'iteration': 0,
                'gradient_norm': None,
                'objective_value': None,
                'learning_rate': learning_rate,
                'converged': False
            }
            
            # 运行迭代
            iterations = 0
            convergence_history = []
            
            max_iterations = self.evaluator_config.get('max_iterations', 500)
            tolerance = self.evaluator_config.get('tolerance', 1e-6)
            
            while not iteration_state['converged'] and iterations < max_iterations:
                # 更新参数
                new_params = strategy.update_parameters(iteration_state)
                if 'learning_rate' in new_params:
                    iteration_state['learning_rate'] = new_params['learning_rate']
                
                # 执行梯度下降步
                if hasattr(problem, 'gradient_descent_step'):
                    iteration_result = problem.gradient_descent_step(
                        lr=iteration_state['learning_rate'],
                        iteration=iterations
                    )
                else:
                    iteration_result = self._default_gradient_descent_step(problem, iteration_state)
                
                # 获取并安全转换数值
                gradient_norm_val = iteration_result.get('gradient_norm', 0)
                objective_val = iteration_result.get('objective', 0)
                
                # 更新状态
                iteration_state.update({
                    'iteration': iterations,
                    'gradient_norm': self._safe_to_float(gradient_norm_val),
                    'objective_value': self._safe_to_float(iteration_result.get('objective_value', objective_val)),
                    'objective': self._safe_to_float(objective_val),
                    'step_size': iteration_result.get('step_size', 0),
                    'converged': iteration_result.get('converged', False)
                })
                
                # 检查收敛
                gradient_norm = iteration_state['gradient_norm']
                if gradient_norm is not None and gradient_norm < tolerance:
                    iteration_state['converged'] = True
                
                # 记录历史
                convergence_history.append({
                    'iteration': iterations,
                    'gradient_norm': iteration_state['gradient_norm'],
                    'learning_rate': iteration_state['learning_rate'],
                    'objective': iteration_state['objective']
                })
                
                iterations += 1
            
            return {
                'iterations': iterations,
                'converged': iteration_state['converged'],
                'final_objective': iteration_state.get('objective', 0),
                'final_learning_rate': iteration_state['learning_rate'],
                'convergence_history': convergence_history[:10]
            }
            
        except Exception as e:
            self.logger.error(f"梯度下降评估失败: {e}")
            return {
                'iterations': max_iterations,
                'converged': False,
                'final_objective': float('inf'),
                'error': str(e)
            }
    
    def _default_gradient_descent_step(self, problem, state):
        """默认梯度下降步实现"""
        import numpy as np
        
        iteration = state['iteration']
        lr = state['learning_rate']
        
        gradient_norm = 1.0 / (1 + iteration * 0.1)
        objective_value = 0.5 * gradient_norm**2
        
        return {
            'gradient_norm': gradient_norm,
            'objective_value': objective_value,
            'objective': objective_value,
            'step_size': lr * gradient_norm,
            'converged': gradient_norm < 1e-6
        }
    
    def evaluate_strategy(self, strategy_path: str, algorithm_type: str, 
                     problem_names: List[str]) -> Dict[str, Any]:
        """
        评估策略性能
        
        Args:
            strategy_path: 策略文件路径
            algorithm_type: 算法类型
            problem_names: 要测试的问题名称列表
            
        Returns:
            每个问题的评估结果
        """
        results = {}
        
        # 加载策略
        strategy = self._load_strategy_module(strategy_path)
        if not strategy:
            for problem_name in problem_names:
                results[problem_name] = {
                    'error': "无法加载策略模块或策略验证失败",
                    'iterations': 0,
                    'converged': False,
                    'final_objective': float('inf')
                }
            return results
            
        # 对每个问题进行评估
        for problem_name in problem_names:
            self.logger.info(f"在问题 {problem_name} 上评估策略")
            
            try:
                # 加载问题
                problem = self._load_problem(algorithm_type, problem_name)
                if not problem:
                    results[problem_name] = {
                        'error': f"无法加载问题 {problem_name}",
                        'iterations': 0,
                        'converged': False,
                        'final_objective': float('inf')
                    }
                    continue
                    
                # 运行算法并评估
                problem_result = self._run_evaluation(
                    strategy=strategy,
                    problem=problem,
                    algorithm_type=algorithm_type
                )
                
                if not isinstance(problem_result, dict):
                    self.logger.error(f"问题 {problem_name} 返回了非字典结果: {type(problem_result)}")
                    results[problem_name] = {
                        'error': f"评估返回了无效类型: {type(problem_result)}",
                        'iterations': 0,
                        'converged': False,
                        'final_objective': float('inf')
                    }
                else:
                    results[problem_name] = problem_result
                    
            except Exception as e:
                self.logger.error(f"评估问题 {problem_name} 时出错: {e}")
                import traceback
                traceback_str = traceback.format_exc()
                self.logger.error(traceback_str)
                results[problem_name] = {
                    'error': f"评估失败: {str(e)}\n{traceback_str}",
                    'iterations': 0,
                    'converged': False,
                    'final_objective': float('inf')
                }
                    
        return results
    
    def _safe_to_float(self, value: Any) -> float:
        """
        安全地将值转换为浮点数
        
        Args:
            value: 要转换的值
            
        Returns:
            转换后的浮点数，如果转换失败则返回无穷大
        """
        if value is None:
            return float('inf')
        
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        
        try:
            return float(value)
        except (ValueError, TypeError):
            if isinstance(value, str):
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                if numbers:
                    try:
                        return float(numbers[0])
                    except (ValueError, TypeError):
                        pass
            
            self.logger.warning(f"无法将值转换为浮点数: {value} (类型: {type(value)})")
            return float('inf')
        
    def _check_convergence(self, state: Dict[str, Any], problem=None) -> Dict[str, Any]:
        """检查收敛条件"""
        tolerance = self.evaluator_config.get('tolerance', 1e-6)
        
        if problem and hasattr(problem, 'params') and 'tol' in problem.params:
            tolerance = problem.params['tol']
        elif problem and hasattr(problem, 'convergence_tolerance'):
            tolerance = problem.convergence_tolerance
        
        # 确保 tolerance 是浮点数
        try:
            tolerance_float = float(tolerance)
        except (ValueError, TypeError):
            tolerance_float = 1e-6
            self.logger.warning(f"无法将 tolerance 转换为浮点数: {tolerance}，使用默认值 1e-6")
        
        primal_residual = self._safe_to_float(state.get('primal_residual'))
        dual_residual = self._safe_to_float(state.get('dual_residual'))
        gradient_norm = self._safe_to_float(state.get('gradient_norm'))
        objective_change = self._safe_to_float(state.get('objective_change'))
        
        converged = False
        convergence_type = 'none'
        
        if primal_residual != float('inf') and dual_residual != float('inf'):
            converged = (primal_residual < tolerance_float and dual_residual < tolerance_float)
            if converged:
                convergence_type = 'admm'
        
        elif gradient_norm != float('inf'):
            converged = gradient_norm < tolerance_float
            if converged:
                convergence_type = 'gradient'
        
        elif objective_change != float('inf'):
            converged = abs(objective_change) < tolerance_float
            if converged:
                convergence_type = 'objective'
        
        elif 'converged' in state:
            converged_val = state['converged']
            if isinstance(converged_val, bool):
                converged = converged_val
            elif isinstance(converged_val, str):
                converged = converged_val.lower() in ['true', 'yes', '1', 't', 'converged']
            else:
                try:
                    converged = bool(converged_val)
                except:
                    converged = False
            convergence_type = 'direct'
        
        return {
            'converged': converged,
            'convergence_type': convergence_type,
            'primal_residual': primal_residual,
            'dual_residual': dual_residual,
            'gradient_norm': gradient_norm,
            'objective_change': objective_change,
            'tolerance_used': tolerance_float  # 添加这一行以便调试
        }