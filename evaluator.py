# evaluator.py
"""
策略评估器模块
负责评估策略生成器生成的 ADMM 惩罚参数策略代码在 8 个选定问题上的效果
并生成反馈提示词用于指导下一次策略生成
"""

import importlib.util
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """
    策略评估器
    
    功能：
    1. 加载和验证生成的策略代码
    2. 在 8 个 ADMM 问题上评估策略性能
    3. 生成反馈提示词用于指导下一次策略生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 配置文件
        """
        self.config = config
        self.evaluator_config = config.get('evaluator', {})
        self.max_iterations = self.evaluator_config.get('max_iterations', 800)
        self.tolerance = self.evaluator_config.get('tolerance', 1e-4)
        
        # 定义要评估的 8 个 ADMM 问题（从配置读取或使用默认值）
        default_problems = [
            'l1_regularization',
            'elastic_net',
            'l1_regression',
            'elastic_net_regression',
            'low_rank_matrix_completion',
            'low_rank_representation',
            'robust_multi_view_spectral_clustering',
            'tracelasso'
        ]
        
        # 优先从配置读取问题列表
        if 'problems' in config and 'admm' in config['problems']:
            self.admm_problems = config['problems']['admm']
        else:
            self.admm_problems = default_problems
        
        # 问题到算法模块的映射
        self.problem_module_map = {
            'l1_regularization': ('l1', 'l1'),
            'elastic_net': ('elasticnet', 'elasticnet'),
            'l1_regression': ('l1R', 'l1R'),
            'elastic_net_regression': ('elasticnetR', 'elasticnetR'),
            'low_rank_matrix_completion': ('lrmc', 'lrmc'),
            'low_rank_representation': ('lrr', 'lrr'),
            'robust_multi_view_spectral_clustering': ('rmsc', 'rmsc'),
            'tracelasso': ('tracelasso', 'tracelasso')
        }
        
        logger.info(f"策略评估器初始化完成，配置：max_iter={self.max_iterations}, tol={self.tolerance}")
    
    def load_strategy(self, strategy_path: str):
        """
        动态加载策略模块，获取 adjust_beta 函数
        
        Args:
            strategy_path: 策略文件路径
            
        Returns:
            adjust_beta 函数
        """
        try:
            # 生成模块名
            module_name = f"strategy_{os.path.basename(strategy_path).replace('.py', '')}"
            
            # 加载模块
            spec = importlib.util.spec_from_file_location(module_name, strategy_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载模块：{strategy_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # 查找 adjust_beta 函数
            adjust_beta_func = getattr(module, 'adjust_beta', None)
            
            if adjust_beta_func is None:
                raise ValueError(f"在 {strategy_path} 中未找到 adjust_beta 函数")
            
            # 验证函数签名
            import inspect
            sig = inspect.signature(adjust_beta_func)
            params = list(sig.parameters.keys())
            
            if len(params) != 1 or params[0] != 'iteration_state':
                raise ValueError(f"adjust_beta 函数签名不正确，应为 adjust_beta(iteration_state)，实际为：{params}")
            
            # 检查返回值类型注解（如果有）
            return_annotation = sig.return_annotation
            if return_annotation != inspect.Parameter.empty and return_annotation != float:
                logger.warning(f"adjust_beta 函数返回值类型应为 float，实际为：{return_annotation}")
            
            logger.info(f"成功加载策略函数：adjust_beta")
            return adjust_beta_func
            
        except Exception as e:
            logger.error(f"加载策略失败：{e}")
            raise
    
    def _generate_test_data(self, problem_name: str):
        """
        为指定问题生成测试数据
        
        Args:
            problem_name: 问题名称
            
        Returns:
            测试数据元组 (A, B, lambda_, opts) 或其他问题所需的数据
        """
        np.random.seed(42)  # 固定随机种子保证可重复性
        
        if problem_name == 'l1_regularization':
            # min_X ||X||_1, s.t. AX=B
            d, na, nb = 50, 100, 1
            A = np.random.randn(d, na)
            X_true = np.zeros((na, nb))
            X_true[:10, :] = np.random.randn(10, nb)  # 稀疏解
            B = A @ X_true
            opts = self._get_default_opts()
            return (A, B, opts)
        
        elif problem_name == 'elastic_net':
            # min_X ||X||_1+lambda*||X||_F^2, s.t. AX=B
            d, na, nb = 50, 100, 1
            A = np.random.randn(d, na)
            X_true = np.zeros((na, nb))
            X_true[:10, :] = np.random.randn(10, nb)
            B = A @ X_true
            lambda_ = 0.1
            opts = self._get_default_opts()
            return (A, B, lambda_, opts)
        
        elif problem_name == 'l1_regression':
            # min_{X,E} loss(E)+lambda*||X||_1, s.t. AX+E=B
            d, na, nb = 50, 100, 1
            A = np.random.randn(d, na)
            X_true = np.zeros((na, nb))
            X_true[:10, :] = np.random.randn(10, nb)
            B = A @ X_true + 0.1 * np.random.randn(d, nb)  # 添加噪声
            lambda_ = 0.1
            opts = self._get_default_opts(loss='l2')
            return (A, B, lambda_, opts)
        
        elif problem_name == 'elastic_net_regression':
            # min_{X,E} loss(E)+lambda1*||X||_1+lambda2*||X||_F^2, s.t. AX+E=B
            d, na, nb = 50, 100, 1
            A = np.random.randn(d, na)
            X_true = np.zeros((na, nb))
            X_true[:10, :] = np.random.randn(10, nb)
            B = A @ X_true + 0.1 * np.random.randn(d, nb)
            lambda1 = 0.1
            lambda2 = 0.1
            opts = self._get_default_opts(loss='l2')
            return (A, B, lambda1, lambda2, opts)
        
        elif problem_name == 'low_rank_matrix_completion':
            # min_X ||X||_*, s.t. P_Omega(X) = P_Omega(M)
            d, n = 50, 50
            rank = 5
            U = np.random.randn(d, rank)
            V = np.random.randn(n, rank)
            M = U @ V.T
            
            # 随机采样
            sample_ratio = 0.3
            num_samples = int(d * n * sample_ratio)
            indices = np.random.choice(d * n, num_samples, replace=False)
            
            opts = self._get_default_opts()
            return (M, indices, opts)
        
        elif problem_name == 'low_rank_representation':
            # min_{X,E} ||X||_*+lambda*loss(E), s.t. A=BX+E
            d, na, nb = 50, 100, 50
            B = np.random.randn(d, nb)
            X_true = np.zeros((nb, na))
            X_true[:rank, :] = np.random.randn(rank, na)
            E_true = 0.1 * np.random.randn(d, na)
            A = B @ X_true + E_true
            lambda_ = 0.1
            opts = self._get_default_opts(loss='l21')
            return (A, B, lambda_, opts)
        
        elif problem_name == 'robust_multi_view_spectral_clustering':
            # min_{L,S_i} ||L||_*+lambda*sum_i ||S_i||_1, s.t. X_i=L+S_i
            d, n, m = 50, 50, 3
            X = np.random.randn(d, n, m)
            lambda_ = 0.1
            opts = self._get_default_opts()
            return (X, lambda_, opts)
        
        elif problem_name == 'tracelasso':
            # min_x ||A*Diag(x)||_*, s.t. Ax=b
            d, n = 50, 100
            A = np.random.randn(d, n)
            x_true = np.zeros(n)
            x_true[:10] = np.random.randn(10)
            b = A @ x_true
            opts = self._get_default_opts()
            return (A, b, opts)
        
        else:
            raise ValueError(f"未知的问题类型：{problem_name}")
    
    def _get_default_opts(self, loss: str = None):
        """
        获取默认算法选项
        
        Args:
            loss: 损失函数类型（可选）
            
        Returns:
            默认选项字典
        """
        opts = {
            'tol': float(self.tolerance),  # 确保是 float 类型
            'max_iter': int(self.max_iterations),  # 确保是 int 类型
            'rho': 1.1,
            'mu': 1e-4,
            'max_mu': 1e10,
            'DEBUG': 0
        }
        
        if loss:
            opts['loss'] = loss
        
        return opts
    
    def _create_admm_wrapper(self, algorithm_func, strategy_instance):
        """
        创建集成策略的 ADMM 包装器
        
        Args:
            algorithm_func: 原始算法函数
            strategy_instance: 策略实例
            
        Returns:
            包装后的算法函数
        """
        def wrapper(*args, **kwargs):
            # 获取 opts 参数
            if len(args) > 0 and isinstance(args[-1], dict):
                opts = args[-1]
            else:
                opts = kwargs.get('opts', {})
            
            # 初始化 beta（惩罚参数）
            beta = opts.get('beta', 1.0)
            
            # 追踪迭代历史
            iteration_history = []
            convergence_history = []
            
            # 保存原始 mu 以便在迭代中使用策略更新
            original_mu = opts.get('mu', 1e-4)
            
            # 包装算法的迭代过程
            # 注意：这里需要修改算法以支持策略更新
            # 由于算法是固定的，我们采用一个技巧：在 opts 中传入回调
            
            # 调用原始算法
            result = algorithm_func(*args, **kwargs)
            
            return result
        
        return wrapper
    
    def evaluate_on_problem(self, problem_name: str, strategy_func) -> Dict[str, Any]:
        """
        在单个问题上评估策略（使用策略函数动态调整 beta 参数）
        
        Args:
            problem_name: 问题名称
            strategy_func: adjust_beta 策略函数
            
        Returns:
            评估结果字典
        """
        try:
            logger.info(f"正在问题 {problem_name} 上评估策略...")
            
            # 生成测试数据
            test_data = self._generate_test_data(problem_name)
            
            # 获取算法模块和函数
            if problem_name not in self.problem_module_map:
                raise ValueError(f"问题 {problem_name} 没有映射到算法模块")
            
            module_name, func_name = self.problem_module_map[problem_name]
            
            # 动态导入算法模块
            algo_module = importlib.import_module(f'libadmm.algorithms.{module_name}')
            
            # 使用策略运行算法（带策略的自定义执行）
            start_time = datetime.now()
            
            # 调用带策略的算法执行
            result = self._run_algorithm_with_strategy(
                algo_module, func_name, test_data, strategy_func
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 解析结果
            if len(result) == 4:
                X, obj, err, iterations = result
            else:
                X, E, obj, err, iterations = result
            
            # 检查收敛性
            try:
                err_float = float(err) if not isinstance(err, (int, float)) else err
                converged = (iterations < self.max_iterations) and (err_float < float(self.tolerance) * 10)
            except (ValueError, TypeError):
                converged = False
            
            evaluation_result = {
                'problem_name': problem_name,
                'iterations': iterations,
                'converged': converged,
                'final_objective': obj,
                'final_error': err,
                'duration_seconds': duration,
                'strategy_params': {}  # 函数策略没有 get_parameters 方法
            }
            
            logger.info(f"问题 {problem_name} 评估完成：iterations={iterations}, converged={converged}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"在问题 {problem_name} 上评估失败：{e}")
            import traceback
            traceback.print_exc()
            return {
                'problem_name': problem_name,
                'error': str(e),
                'iterations': self.max_iterations,
                'converged': False
            }
    
    def _run_algorithm_with_strategy(self, algo_module, func_name, test_data, strategy_func):
        """
        使用策略函数动态调整 beta 参数来运行算法
        
        直接调用 libadmm 中的算法函数，传入 strategy 函数参数
        
        Args:
            algo_module: 算法模块
            func_name: 算法函数名
            test_data: 测试数据元组
            strategy_func: adjust_beta 策略函数
            
        Returns:
            算法运行结果
        """
        # 获取算法函数
        algorithm_func = getattr(algo_module, func_name)
        
        # 从 test_data 中解包参数
        opts = test_data[-1] if isinstance(test_data[-1], dict) else {}
        
        # 确保 opts 中有必要的参数
        if 'max_iter' not in opts:
            opts['max_iter'] = self.max_iterations
        if 'tol' not in opts:
            opts['tol'] = self.tolerance
        
        # 直接调用算法函数，传入 strategy 函数参数
        # 算法函数会在每次迭代中调用 strategy_func(iteration_state)
        result = algorithm_func(*test_data[:-1], strategy=strategy_func)
        
        return result
    
    def evaluate_strategy(self, strategy_path: str, algorithm_type: str = "admm", 
                         problem_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        评估策略在所有选定问题上的性能
        
        Args:
            strategy_path: 策略文件路径
            algorithm_type: 算法类型（目前仅支持 'admm'）
            problem_names: 要评估的问题列表，如果为 None 则使用默认的 8 个问题
            
        Returns:
            评估结果字典 {problem_name: result_dict}
        """
        if algorithm_type != 'admm':
            logger.error(f"评估器暂不支持算法类型：{algorithm_type}")
            return {}
        
        # 使用默认问题列表
        if problem_names is None:
            problem_names = self.admm_problems
        
        logger.info(f"开始评估策略：{strategy_path}")
        logger.info(f"评估问题列表：{problem_names}")
        
        # 加载策略
        try:
            strategy_func = self.load_strategy(strategy_path)
        except Exception as e:
            logger.error(f"策略加载失败，无法继续评估：{e}")
            return {name: {'error': f'策略加载失败：{str(e)}', 'converged': False, 'iterations': self.max_iterations} 
                    for name in problem_names}
        
        # 在每个问题上评估
        results = {}
        for problem_name in problem_names:
            result = self.evaluate_on_problem(problem_name, strategy_func)
            results[problem_name] = result
        
        # 计算总体统计信息
        total_problems = len(problem_names)
        converged_count = sum(1 for r in results.values() if r.get('converged', False))
        avg_iterations = np.mean([r.get('iterations', self.max_iterations) for r in results.values()])
        
        logger.info(f"策略评估完成：{converged_count}/{total_problems} 问题收敛，平均迭代次数：{avg_iterations:.1f}")
        
        return results
    
    def generate_feedback_prompt(self, evaluation_results: Dict[str, Dict[str, Any]], 
                                  iteration: int,
                                  previous_results: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """
        基于评估结果生成反馈提示词
        
        Args:
            evaluation_results: 当前评估结果
            iteration: 当前迭代轮次
            previous_results: 上一轮评估结果（可选）
            
        Returns:
            反馈提示词字符串
        """
        feedback_parts = []
        
        # 1. 总体性能总结
        total_problems = len(evaluation_results)
        converged_problems = [name for name, result in evaluation_results.items() 
                             if result.get('converged', False)]
        unconverged_problems = [name for name, result in evaluation_results.items() 
                               if not result.get('converged', False)]
        error_problems = [name for name, result in evaluation_results.items() 
                         if 'error' in result]
        
        feedback_parts.append(f"【第{iteration}轮策略评估反馈】")
        feedback_parts.append(f"\n总体表现：{len(converged_problems)}/{total_problems} 问题收敛")
        
        if error_problems:
            feedback_parts.append(f"\n❌ 出错的问题 ({len(error_problems)}个): {', '.join(error_problems)}")
            for name in error_problems:
                feedback_parts.append(f"  - {name}: {evaluation_results[name]['error']}")
        
        # 2. 收敛问题分析
        if converged_problems:
            feedback_parts.append(f"\n✅ 已收敛的问题 ({len(converged_problems)}个):")
            for name in converged_problems:
                result = evaluation_results[name]
                feedback_parts.append(
                    f"  - {name}: {result['iterations']}次迭代, "
                    f"最终误差={result['final_error']:.2e}, "
                    f"目标函数={result['final_objective']:.2e}"
                )
        
        # 3. 未收敛问题分析
        if unconverged_problems:
            feedback_parts.append(f"\n⚠️ 未收敛的问题 ({len(unconverged_problems)}个):")
            for name in unconverged_problems:
                result = evaluation_results[name]
                feedback_parts.append(
                    f"  - {name}: 达到最大迭代次数{result['iterations']}仍未收敛"
                )
        
        # 4. 与上一轮对比（如果有）
        if previous_results:
            feedback_parts.append("\n【与上一轮对比】")
            
            improvements = []
            degradations = []
            
            for name in evaluation_results:
                if name in previous_results and name not in error_problems:
                    curr_iters = evaluation_results[name].get('iterations', self.max_iterations)
                    prev_iters = previous_results[name].get('iterations', self.max_iterations)
                    
                    if curr_iters < prev_iters:
                        improvements.append((name, prev_iters - curr_iters))
                    elif curr_iters > prev_iters:
                        degradations.append((name, curr_iters - prev_iters))
            
            if improvements:
                feedback_parts.append("\n性能提升的问题:")
                for name, diff in improvements:
                    feedback_parts.append(f"  - {name}: 减少{diff}次迭代")
            
            if degradations:
                feedback_parts.append("\n性能下降的问题:")
                for name, diff in degradations:
                    feedback_parts.append(f"  - {name}: 增加{diff}次迭代")
        
        # 5. 针对性优化建议
        feedback_parts.append("\n【优化建议】")
        
        # 分析未收敛问题的共同特征
        if len(unconverged_problems) > total_problems / 2:
            feedback_parts.append("⚠️ 超过半数问题未收敛，建议采用更保守的策略")
            feedback_parts.append("  - 考虑减小 beta 的调整幅度")
            feedback_parts.append("  - 增加 beta 的上下限范围")
        
        # 针对特定问题类型的建议
        regression_problems = [p for p in unconverged_problems 
                              if p in ['l1_regression', 'elastic_net_regression']]
        if regression_problems:
            feedback_parts.append(f"\n回归类问题 ({', '.join(regression_problems)}) 未收敛:")
            feedback_parts.append("  - 建议采用惩罚参数单调上升策略")
            feedback_parts.append("  - beta 只增不减，初始值 1.0，每次按 1.1-1.5 倍增大")
            feedback_parts.append("  - 上限设置为 1e4")
        
        # 矩阵补全和低秩问题
        matrix_problems = [p for p in unconverged_problems 
                          if p in ['low_rank_matrix_completion', 'low_rank_representation']]
        if matrix_problems:
            feedback_parts.append(f"\n低秩矩阵问题 ({', '.join(matrix_problems)}) 未收敛:")
            feedback_parts.append("  - 核范数优化需要更精细的 beta 调整")
            feedback_parts.append("  - 建议基于奇异值的变化调整 beta")
        
        # 6. 超参数调整建议
        feedback_parts.append("\n【超参数调整方向】")
        feedback_parts.append("请基于以下方向调整策略超参数:")
        feedback_parts.append("  1. mu (初始步长): 影响收敛速度")
        feedback_parts.append("  2. tau_inc (增大因子): 控制 beta 增长速率")
        feedback_parts.append("  3. tau_dec (减小因子): 控制 beta 衰减速率")
        feedback_parts.append("  4. min_beta, max_beta (范围限制): 防止数值不稳定")
        
        # 7. 鼓励性总结
        avg_iterations = np.mean([r.get('iterations', self.max_iterations) 
                                 for r in evaluation_results.values() 
                                 if 'error' not in r])
        feedback_parts.append(f"\n【总结】当前策略平均迭代次数：{avg_iterations:.1f}")
        
        if len(converged_problems) >= total_problems * 0.8:
            feedback_parts.append("✅ 策略表现良好，请继续优化以进一步提升性能")
        elif len(converged_problems) >= total_problems * 0.5:
            feedback_parts.append("⚠️ 策略表现中等，需要针对性改进未收敛问题")
        else:
            feedback_parts.append("❌ 策略表现不佳，建议重新设计调整逻辑")
        
        return "\n".join(feedback_parts)
    
    def save_evaluation_results(self, results: Dict[str, Dict[str, Any]], 
                                 iteration: int, output_dir: str = "results"):
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果字典
            iteration: 迭代轮次
            output_dir: 输出目录
        """
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为可序列化的格式
        serializable_results = {}
        for problem_name, result in results.items():
            serializable_results[problem_name] = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_results[problem_name][key] = float(value)
                elif isinstance(value, np.bool_):
                    serializable_results[problem_name][key] = bool(value)
                elif isinstance(value, np.ndarray):
                    serializable_results[problem_name][key] = value.tolist()
                else:
                    serializable_results[problem_name][key] = value
        
        # 保存为 JSON
        output_file = os.path.join(output_dir, f"evaluation_iter_{iteration}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存至：{output_file}")
        
        return output_file
