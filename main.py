#!/usr/bin/env python3
"""主控制循环模块 - 使用strict模式版本管理整个生成-验证-反馈迭代流程"""
import yaml
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
# import importlib.util # 可能需要，取决于evaluator如何加载策略
from strategy_generator import StrategyGenerator
from evaluator import StrategyEvaluator
from feedback_loop import FeedbackLoop
from advisor import ADMMAdvisor

class EvolutionaryTuningMain:
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化主控制器
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置日志
        self._setup_logging()

        # 初始化组件
        self.generator = StrategyGenerator(self.config)
        self.evaluator = StrategyEvaluator(self.config)
        self.feedback_loop = FeedbackLoop(self.config)
        
        # 初始化千问指导者
        if self.config.get('advisor', {}).get('enabled', False):
            self.advisor = ADMMAdvisor(self.config)
        else:
            self.advisor = None
        self.last_advisor_guidance = None  # 保存最近一次的指导建议

        # 状态跟踪
        self.iteration = 0
        self.best_strategy = None
        self.best_performance = float('inf')
        self.history = []
        self.no_improve_rounds = 0  # 连续无性能改进的轮数，用于早停
        self.severe_degradation = False  # 标记上一轮是否性能严重恶化
        self.last_rejected_strategy = None  # 记录上一轮被拒绝的策略信息
        
        # "智能调用千问API"新增：连续无改进轮次计数器
        self.consecutive_no_improvement = 0  # 连续无有效改进的轮次
        self.last_significant_improvement_iter = 0  # 上次显著改进的轮次
        self.advisor_call_history = []  # 千问API调用历史记录
        
        # 加载智能调用配置
        self.smart_call_config = {
            'enabled': self.config.get('advisor', {}).get('smart_call_enabled', True),
            'no_improvement_threshold': self.config.get('advisor', {}).get('no_improvement_threshold', 50),
            'min_performance_change': self.config.get('advisor', {}).get('min_performance_change', 0.01),
            'call_history_enabled': self.config.get('advisor', {}).get('call_history_enabled', True),
            'call_history_file': self.config.get('advisor', {}).get('call_history_file', 'advisor_call_history.json')
        }
        
        # "混合模式"新增：策略尝试历史记录（用于避免重复）
        self.strategy_attempts_history = []  # 记录所有尝试过的策略摘要

        # 创建输出目录
        os.makedirs("strategies", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def _setup_logging(self):
        """设置日志系统"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        """运行主循环"""
        self.logger.info("启动进化自适应调参框架（strict模式）")

        # 检查API配置
        api_key = self.config['api']['api_key']
        default_api_key = "your_deepseek_api_key"
        if api_key == default_api_key or not api_key or not api_key.startswith("sk-"):
            self.logger.error(f"API密钥配置有问题: {api_key[:10]}..." if api_key else "API密钥为空")
            self.logger.error("请确保在config.yaml中配置正确的DeepSeek API密钥")
            self.logger.error("您可以使用: https://platform.deepseek.com/api_keys 获取API密钥")
            self.logger.error("当前config.yaml中的api_key需要替换为您的实际API密钥")
            return
        else:
            self.logger.info("API密钥配置正确")

        # 检查其他API参数
        api_config = self.config['api']
        self.logger.info(f"使用API配置: model={api_config.get('model')}, temperature={api_config.get('temperature')}")

        while not self._should_terminate():
            self.iteration += 1
            self.logger.info(f"开始第 {self.iteration} 轮迭代")

            # 获取反馈信息 (注意：这里使用的是上一轮的结果来生成本轮的反馈)
            # 在第一轮，history 为空，所以会使用初始 prompt
            feedback_for_generation = self._get_feedback_for_generation()

            # 生成策略
            strategy_code = self.generator.generate_strategy(
                algorithm_type="admm",
                feedback=feedback_for_generation
            )
            if not strategy_code:
                self.logger.error("策略生成失败，跳过此轮迭代。")
                continue # 跳过本次迭代，进入下一轮

            # 保存策略
            strategy_path = f"strategies/strategy_iter_{self.iteration}.py"
            with open(strategy_path, 'w', encoding='utf-8') as f:
                f.write(strategy_code)
            self.logger.info(f"生成的策略已保存至: {strategy_path}")

            # 从配置中获取ADMM问题列表
            if 'problems' in self.config and 'admm' in self.config['problems']:
                problem_names = self.config['problems']['admm']
                self.logger.info(f"从配置中加载ADMM问题: {problem_names}")
            else:
                # 如果没有配置，使用默认的7个问题
                problem_names = [
                    "l1_regularization",
                    "elastic_net",
                    "l1_regression",
                    "elastic_net_regression",
                    "low_rank_matrix_completion",
                    "low_rank_representation",
                    "robust_multi_view_spectral_clustering"
                ]
                self.logger.warning(f"使用默认ADMM问题列表: {problem_names}")

            # 评估策略
            try:
                performance_results = self.evaluator.evaluate_strategy(
                    strategy_path=strategy_path,
                    algorithm_type="admm",
                    problem_names=problem_names
                )
                # 新增：详细打印每个问题的迭代次数和收敛情况，方便调试
                for problem_name, result in performance_results.items():
                    if isinstance(result, dict):
                        iters = result.get('iterations')
                        conv = result.get('converged')
                        self.logger.info(f"问题 {problem_name}: iterations={iters}, converged={conv}")
            except Exception as e:
                self.logger.error(f"策略评估失败: {e}")
                performance_results = {name: {'error': f'评估失败: {str(e)}', 'iterations': 1000, 'converged': False} for name in problem_names}

            # === 智能调用千问指导者分析 ===
            if self.advisor and self.advisor.enabled:
                should_call_advisor = self._should_call_advisor(avg_performance)
                
                if should_call_advisor:
                    try:
                        self.logger.info(f"触发千问指导者分析 (连续无改进: {self.consecutive_no_improvement}/{self.smart_call_config['no_improvement_threshold']})")
                        
                        call_start_time = datetime.now()
                        self.last_advisor_guidance = self.advisor.analyze_evaluation_results(
                            evaluation_results=performance_results,
                            iteration=self.iteration,
                            history=self.history
                        )
                        call_end_time = datetime.now()
                        call_duration = (call_end_time - call_start_time).total_seconds()
                        
                        self.advisor.save_analysis(self.last_advisor_guidance, self.iteration)
                        
                        # 记录调用历史
                        self._record_advisor_call(
                            iteration=self.iteration,
                            reason=f"连续无改进达到阈值({self.consecutive_no_improvement}轮)",
                            duration=call_duration,
                            guidance_summary=self.last_advisor_guidance[:200] if self.last_advisor_guidance else ""
                        )
                        
                        self.logger.info("千问指导者分析完成")
                    except Exception as e:
                        self.logger.error(f"千问指导者分析失败: {e}")
                        self.last_advisor_guidance = None
                else:
                    self.logger.debug(f"跳过千问指导者分析 (连续无改进: {self.consecutive_no_improvement}/{self.smart_call_config['no_improvement_threshold']})")

            # 计算总体性能指标
            avg_performance = self._calculate_average_performance(performance_results)
            
            # 「混合模式」新增：记录此次尝试到历史
            self._record_strategy_attempt(strategy_path, strategy_code, avg_performance, performance_results)

            # 精英保留机制：检测性能是否严重恶化
            performance_degradation_ratio = 0.0
            if self.best_performance < float('inf') and self.best_performance > 0:
                performance_degradation_ratio = (avg_performance - self.best_performance) / self.best_performance
            
            # 如果性能恶化超过50%，标记为严重恶化
            self.severe_degradation = performance_degradation_ratio > 0.5
            if self.severe_degradation:
                self.logger.warning(f"检测到性能严重恶化: {self.best_performance:.2f} -> {avg_performance:.2f} (恶化 {performance_degradation_ratio*100:.1f}%)")
                self.logger.warning("下一轮将强制基于历史最优策略进行改进")

            # 更新最佳策略 & 统计连续无改进轮数（用于早停）
            # 性能改进或持平时都收录
            if avg_performance <= self.best_performance:
                # 计算性能改进比例
                if self.best_performance < float('inf') and self.best_performance > 0:
                    improvement_ratio = (self.best_performance - avg_performance) / self.best_performance
                else:
                    improvement_ratio = 1.0  # 首次改进，视为100%改进
                
                self.best_performance = avg_performance
                self.best_strategy = {
                    'path': strategy_path,
                    'performance': avg_performance,
                    'results': performance_results
                }
                self.no_improve_rounds = 0
                self.last_rejected_strategy = None  # 清除拒绝记录
                
                # "智能调用千问API"：检查是否为显著改进
                if improvement_ratio >= self.smart_call_config['min_performance_change']:
                    self.consecutive_no_improvement = 0  # 重置连续无改进计数
                    self.last_significant_improvement_iter = self.iteration
                    self.logger.info(f"发现新最佳策略，性能: {avg_performance:.2f} (改进 {improvement_ratio*100:.2f}%)")
                else:
                    # 性能改进但不够显著，仍然计数
                    self.logger.info(f"发现新最佳策略，性能: {avg_performance:.2f} (改进 {improvement_ratio*100:.2f}%，未达到显著改进阈值)")
                
                # 只有性能改进时才记录到历史
                self.history.append({
                    'iteration': self.iteration,
                    'strategy_path': strategy_path,
                    'performance': avg_performance,
                    'detailed_results': performance_results
                })
                
                # 生成反馈 (注意：这里是为下一轮迭代准备的反馈)
                feedback = self.feedback_loop.generate_feedback(
                    performance_results=performance_results,
                    history=self.history,
                    iteration=self.iteration
                )
                self.feedback_loop.save_feedback(feedback, self.iteration)
            else:
                # 性能变差，不收录这个策略，记录拒绝信息供下一轮参考
                self.last_rejected_strategy = {
                    'iteration': self.iteration,
                    'strategy_path': strategy_path,
                    'performance': avg_performance,
                    'best_performance': self.best_performance,
                    'degradation_ratio': (avg_performance - self.best_performance) / self.best_performance if self.best_performance > 0 else 0,
                    'detailed_results': performance_results
                }
                self.logger.warning(f"性能变差 ({self.best_performance:.2f} -> {avg_performance:.2f})，不收录此策略，继续迭代")
                
                # "智能调用千问API"：性能变差，增加连续无改进计数
                self.consecutive_no_improvement += 1

            self.logger.info(f"第 {self.iteration} 轮迭代完成，平均性能: {avg_performance:.2f}")

        # 输出最终结果
        self._output_final_results()

    def _get_feedback_for_generation(self) -> str:
        """
        获取用于策略生成的反馈信息
        """
        # 这个方法是在新的一轮迭代开始前被调用的。
        # 所以 self.iteration 是当前轮次（比如1, 2, 3...）
        # self.history 包含了截至上一轮的所有历史信息。
        
        if self.iteration == 1:
            # 在第一轮迭代开始前，history 仍然是空的 []。
            # 所以我们提供初始提示。
            initial_prompt = self.feedback_loop.get_initial_prompt("admm")
            strict_requirements = "\n\n【strict模式要求】\n1. 请生成一个继承自 `BaseTuningStrategy` 的Python类。\n2. 实现 `update_parameters` 方法，签名必须为 `update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]`。\n3. 该方法只能调整 `beta` 参数，返回 `{'beta': new_beta_value}`。\n4. 保持其他ADMM参数不变。\n"
            return initial_prompt + strict_requirements
        else:
            # 在第二轮及以后迭代开始前，history 至少包含一轮的结果。
            # 我们获取上一轮的结果来指导本轮的生成。
            # 因为 self.iteration 已经自增了，所以 history[-1] 就是上一轮的结果
            if not self.history:
                 # 防御性检查，理论上不会执行到这里，因为 iteration > 1 时 history 不应为空
                 self.logger.warning("History is unexpectedly empty when iteration > 1.")
                 return self.feedback_loop.get_initial_prompt("admm")
            
            # 检查是否有被拒绝的策略，生成警告提示
            rejected_warning = ""
            if self.last_rejected_strategy:
                rej = self.last_rejected_strategy
                rej_perf = rej.get('performance', 0)
                best_perf = rej.get('best_performance', 0)
                degradation = rej.get('degradation_ratio', 0) * 100
                rej_path = rej.get('strategy_path', '')
                
                rejected_warning = (
                    "\n\n【警告：上一轮策略已被拒绝】\n"
                    f"上一轮生成的策略性能下降，已被丢弃不予收录。\n"
                    f"• 当前最佳性能: {best_perf:.2f} 次迭代\n"
                    f"• 被拒绝策略性能: {rej_perf:.2f} 次迭代 (恶化 {degradation:.1f}%)\n"
                    f"• 拒绝原因: 性能不如当前最佳策略\n\n"
                    "【改进建议】\n"
                    "1. 请避免与上一轮被拒绝策略类似的优化方向\n"
                    "2. 必须基于当前最佳策略进行小幅改进\n"
                    "3. 尝试不同的调参方向，而不是继续之前失败的思路\n"
                )
                
                # 如果被拒绝的策略文件存在，显示其代码供参考（作为反面案例）
                if rej_path and os.path.exists(rej_path):
                    try:
                        with open(rej_path, 'r', encoding='utf-8') as f:
                            rej_code = f.read()
                        rejected_warning += (
                            "【被拒绝的策略代码（反面案例，避免类似设计）】\n"
                            "```python\n" + rej_code + "\n```\n"
                        )
                    except Exception as e:
                        self.logger.warning(f"读取被拒绝策略代码失败: {e}")
            
            current_result = self.history[-1]['detailed_results'] # 上一轮的详细结果
            previous_result = self.history[-2]['detailed_results'] if len(self.history) > 1 else None # 上上轮的结果

            # 基于评估结果生成文字反馈
            performance_feedback = self.feedback_loop.format_feedback(current_result, previous_result)

            # 分析未收敛的问题，生成针对性优化建议
            unconverged_analysis = self._analyze_unconverged_problems(current_result)

            # 追加策略代码，帮助模型做有针对性的改进
            strategy_sections = []

            # 精英保留机制：如果上一轮性能严重恶化，强制基于最优策略改进
            use_best_as_base = getattr(self, 'severe_degradation', False)
            
            if use_best_as_base and self.best_strategy:
                # 性能严重恶化，强制基于最优策略
                best_strategy_path = self.best_strategy.get('path')
                if best_strategy_path and os.path.exists(best_strategy_path):
                    try:
                        with open(best_strategy_path, 'r', encoding='utf-8') as f:
                            best_code = f.read()
                        best_perf = self.best_strategy.get('performance')
                        perf_str = f"{best_perf:.2f}" if isinstance(best_perf, (int, float)) else str(best_perf)
                        
                        strategy_sections.append(
                            "\n\n【警告：性能严重恶化，强制回退到最优策略】\n"
                            f"上一轮策略性能严重恶化，必须基于历史最优策略（平均迭代 {perf_str} 次）进行小幅改进。\n"
                            "绝对禁止大幅重构或使用上一轮的失败策略。\n\n"
                            "【必须基于以下最优策略进行改进】\n"
                            "```python\n" + best_code + "\n```\n\n"
                            "【改进约束】\n"
                            "1. 保留上述最优策略的90%以上的代码\n"
                            "2. 只允许修改 1-2 个超参数的值\n"
                            "3. 不允许改变核心算法逻辑\n"
                        )
                    except Exception as e:
                        self.logger.warning(f"读取最优策略代码失败: {e}")
            else:
                # 正常情况：提供上一轮策略和最优策略
                
                # 上一轮使用的策略
                last_strategy_path = self.history[-1].get('strategy_path')
                if last_strategy_path and os.path.exists(last_strategy_path):
                    try:
                        with open(last_strategy_path, 'r', encoding='utf-8') as f:
                            last_code = f.read()
                        last_perf = self.history[-1].get('performance')
                        last_perf_str = f"{last_perf:.2f}" if isinstance(last_perf, (int, float)) else str(last_perf)
                        strategy_sections.append(
                            f"\n\n【上一轮调参策略代码】(平均迭代 {last_perf_str} 次)\n"
                            "下面是上一轮使用的策略实现，请在理解其 beta 调整逻辑的基础上进行小幅改进：\n"
                            "```python\n" + last_code + "\n```"
                        )
                    except Exception as e:
                        self.logger.warning(f"读取上一轮策略代码失败: {e}")

                # 历史最优策略
                best_strategy_path = self.best_strategy.get('path') if self.best_strategy else None
                if best_strategy_path and os.path.exists(best_strategy_path) and best_strategy_path != last_strategy_path:
                    try:
                        with open(best_strategy_path, 'r', encoding='utf-8') as f:
                            best_code = f.read()
                        best_perf = self.best_strategy.get('performance')
                        perf_str = f"{best_perf:.2f}" if isinstance(best_perf, (int, float)) else str(best_perf)
                        strategy_sections.append(
                            f"\n\n【历史最优调参策略（必须参考）】(平均迭代 {perf_str} 次)\n"
                            "这是目前性能最好的策略。请重点对比上一轮策略与该最优策略的差异，\n"
                            "并在最优策略的基础上设计改进方案：\n"
                            "```python\n" + best_code + "\n```"
                        )
                    except Exception as e:
                        self.logger.warning(f"读取历史最优策略代码失败: {e}")

            # 「混合模式」新增：添加历史策略摘要
            history_summary = self._generate_history_summary()

            # 融合千问指导者建议
            advisor_section = ""
            if hasattr(self, 'last_advisor_guidance') and self.last_advisor_guidance:
                advisor_section = (
                    "\n\n【千问专家指导建议】\n"
                    "以下是ADMM优化专家(Qwen3-235B-A22B)给出的深度分析和优化建议，请优先参考：\n\n"
                    f"{self.last_advisor_guidance}\n\n"
                    "【请优先按照上述专家建议调整策略参数】\n"
                )

            return rejected_warning + performance_feedback + unconverged_analysis + advisor_section + history_summary + "".join(strategy_sections)


    def _should_terminate(self) -> bool:
        """检查是否满足终止条件"""
        termination_config = self.config['termination']

        # 检查最大迭代次数
        if self.iteration >= termination_config['max_iterations']:
            self.logger.info(f"达到最大迭代次数: {termination_config['max_iterations']}")
            return True

        # 检查性能阈值 (如果平均迭代次数低于某个值，认为达标)
        if len(self.history) >= 1: # 可以根据需要调整轮次
            recent_perf = [h['performance'] for h in self.history[-1:]] # 检查最近一轮
            avg_recent = sum(recent_perf) / len(recent_perf)
            # 假设如果平均迭代次数少于500次就算达标 (可根据需要调整)
            performance_target = self.config.get('termination', {}).get('performance_target', 500)
            self.logger.info(f"当前平均性能: {avg_recent:.1f}, 目标: {performance_target}")
            if avg_recent <= performance_target:
                self.logger.info(f"达到性能目标: 最近平均{avg_recent:.1f}次迭代 <= {performance_target}")
                return True

        # 检查改进停滞（基于 patience 的早停机制）
        patience = termination_config['patience']
        if getattr(self, 'no_improve_rounds', 0) >= patience:
            self.logger.info(f"连续 {patience} 轮无性能改进，触发早停")
            return True
        return False

    def _analyze_unconverged_problems(self, results: Dict[str, Any]) -> str:
        """
        分析未收敛的问题，生成针对性的优化建议
        
        Args:
            results: 评估结果字典
            
        Returns:
            包含未收敛问题分析的字符串
        """
        if not results:
            return ""
        
        unconverged_problems = []
        converged_problems = []
        error_problems = []
        
        for problem_name, result in results.items():
            if not isinstance(result, dict):
                continue
                
            if 'error' in result:
                error_problems.append(problem_name)
            elif result.get('converged', False):
                converged_problems.append({
                    'name': problem_name,
                    'iterations': result.get('iterations', 0),
                    'final_objective': result.get('final_objective', 'N/A')
                })
            else:
                # 未收敛的问题
                unconverged_problems.append({
                    'name': problem_name,
                    'iterations': result.get('iterations', 0),
                    'final_primal_res': result.get('convergence_history', [{}])[-1].get('primal_residual', 'N/A') if result.get('convergence_history') else 'N/A',
                    'final_dual_res': result.get('convergence_history', [{}])[-1].get('dual_residual', 'N/A') if result.get('convergence_history') else 'N/A',
                    'final_beta': result.get('final_beta', 'N/A'),
                    'final_objective': result.get('final_objective', 'N/A')
                })
        
        if not unconverged_problems:
            if converged_problems:
                return f"\n\n【收敛情况】所有 {len(converged_problems)} 个问题均已收敛，请进一步优化收敛速度。\n"
            return ""
        
        # 构建未收敛问题的详细分析
        analysis = "\n\n【重点优化目标 - 未收敛问题分析】\n"
        analysis += f"共有 {len(unconverged_problems)} 个问题未收敛，需要重点优化：\n\n"
        
        # 分类分析未收敛问题
        high_primal_res = []  # 原始残差较大
        high_dual_res = []    # 对偶残差较大
        both_high = []        # 两者都大
        
        for prob in unconverged_problems:
            name = prob['name']
            iters = prob['iterations']
            primal = prob['final_primal_res']
            dual = prob['final_dual_res']
            beta = prob['final_beta']
            obj = prob['final_objective']
            
            # 格式化数值
            primal_str = f"{primal:.2e}" if isinstance(primal, (int, float)) else str(primal)
            dual_str = f"{dual:.2e}" if isinstance(dual, (int, float)) else str(dual)
            beta_str = f"{beta:.4f}" if isinstance(beta, (int, float)) else str(beta)
            obj_str = f"{obj:.2e}" if isinstance(obj, (int, float)) else str(obj)
            
            analysis += f"  • {name}:\n"
            analysis += f"    - 迭代次数: {iters} (达到上限未收敛)\n"
            analysis += f"    - 最终原始残差: {primal_str}\n"
            analysis += f"    - 最终对偶残差: {dual_str}\n"
            analysis += f"    - 最终 beta 值: {beta_str}\n"
            analysis += f"    - 最终目标函数: {obj_str}\n\n"
            
            # 分类
            try:
                if isinstance(primal, (int, float)) and isinstance(dual, (int, float)):
                    if primal > 1e-4 and dual > 1e-4:
                        both_high.append(name)
                    elif primal > 1e-4:
                        high_primal_res.append(name)
                    elif dual > 1e-4:
                        high_dual_res.append(name)
            except:
                pass
        
        # 生成针对性优化建议
        analysis += "【针对性优化建议】\n"
        
        if both_high:
            analysis += f"1. 问题 {', '.join(both_high)} 的原始残差和对偶残差都较大:\n"
            analysis += "   - 建议采用更保守的 beta 调整策略，避免剧烈波动\n"
            analysis += "   - 可以试着增大 beta 的初始值或调整调整比率\n\n"
        
        if high_primal_res:
            analysis += f"2. 问题 {', '.join(high_primal_res)} 的原始残差较大（约束违反）:\n"
            analysis += "   - 原始残差大表示 x 和 z 还未充分一致\n"
            analysis += "   - 建议增大 beta 值以加强惩罚，促使原始约束满足\n"
            analysis += "   - 可以试着调整 tau_inc 参数使 beta 更快增长\n\n"
        
        if high_dual_res:
            analysis += f"3. 问题 {', '.join(high_dual_res)} 的对偶残差较大（吶朗日乃次优化）:\n"
            analysis += "   - 对偶残差大表示制的变化还未稳定\n"
            analysis += "   - 建议减小 beta 值以缓解过度惩罚\n"
            analysis += "   - 可以试着调整 tau_dec 参数使 beta 更快衰减\n\n"
        
        if not (both_high or high_primal_res or high_dual_res):
            analysis += "  - 残差已较小但未达到收敛阈值，建议微调 beta 的范围限制或调整更新频率\n\n"
        
        # 添加统计信息
        if converged_problems:
            analysis += f"【已收敛问题】共 {len(converged_problems)} 个: "
            analysis += ", ".join([f"{p['name']}({p['iterations']}次)" for p in converged_problems])
            analysis += "\n"
        
        return analysis

    # ============== 「混合模式」新增方法 ==============
    
    def _record_strategy_attempt(self, strategy_path: str, strategy_code: str, 
                                  avg_performance: float, results: Dict[str, Any]):
        """
        记录策略尝试到历史，用于避免重复生成类似策略
        """
        # 提取策略特征
        features = self._extract_strategy_features(strategy_code)
        
        # 统计收敛情况
        converged_count = sum(1 for r in results.values() 
                             if isinstance(r, dict) and r.get('converged', False))
        total_count = len(results)
        
        # 记录摘要
        attempt_record = {
            'iteration': self.iteration,
            'performance': avg_performance,
            'converged_ratio': f"{converged_count}/{total_count}",
            'features': features,
            'is_best': avg_performance <= self.best_performance,
            'improvement': ((self.best_performance - avg_performance) / self.best_performance * 100) 
                          if self.best_performance > 0 and self.best_performance < float('inf') else 0
        }
        
        self.strategy_attempts_history.append(attempt_record)
        
        # 只保留最近20条记录，避免历史过长
        if len(self.strategy_attempts_history) > 20:
            self.strategy_attempts_history = self.strategy_attempts_history[-20:]
    
    def _extract_strategy_features(self, code: str) -> Dict[str, Any]:
        """
        从策略代码中提取关键特征，用于比较策略相似性
        """
        import re
        
        features = {
            'has_residual_ratio': 'primal' in code.lower() and 'dual' in code.lower(),
            'has_adaptive_tau': 'tau' in code.lower(),
            'has_momentum': 'momentum' in code.lower() or 'history' in code.lower(),
            'has_clipping': 'clip' in code.lower() or 'min_beta' in code.lower(),
            'has_smoothing': 'smooth' in code.lower() or 'alpha' in code.lower(),
        }
        
        # 提取关键超参数值
        mu_match = re.search(r'self\.mu\s*=\s*([\d.]+)', code)
        if mu_match:
            features['mu'] = float(mu_match.group(1))
        
        tau_inc_match = re.search(r'tau_inc\s*=\s*([\d.]+)', code)
        if tau_inc_match:
            features['tau_inc'] = float(tau_inc_match.group(1))
            
        tau_dec_match = re.search(r'tau_dec\s*=\s*([\d.]+)', code)
        if tau_dec_match:
            features['tau_dec'] = float(tau_dec_match.group(1))
        
        # 提取beta范围
        min_beta_match = re.search(r'min_beta\s*=\s*([\d.e\-+]+)', code)
        max_beta_match = re.search(r'max_beta\s*=\s*([\d.e\-+]+)', code)
        if min_beta_match:
            features['min_beta'] = min_beta_match.group(1)
        if max_beta_match:
            features['max_beta'] = max_beta_match.group(1)
        
        return features
    
    def _generate_history_summary(self) -> str:
        """
        生成历史策略尝试摘要，告诉AI哪些方向已经试过
        """
        if len(self.strategy_attempts_history) < 2:
            return ""
        
        summary = "\n\n【历史策略尝试摘要 - 避免重复】\n"
        summary += f"已尝试 {len(self.strategy_attempts_history)} 轮策略，以下是近期尝试记录：\n\n"
        
        # 分类：成功的 vs 失败的
        successful = [a for a in self.strategy_attempts_history if a.get('is_best')]
        failed = [a for a in self.strategy_attempts_history if not a.get('is_best')]
        
        # 显示失败的尝试（这些方向要避免）
        if failed:
            summary += "❌ 以下方向已尝试但效果不佳，请避免类似设计：\n"
            for i, attempt in enumerate(failed[-5:], 1):  # 只显示最近5个
                features = attempt.get('features', {})
                perf = attempt.get('performance', 0)
                conv_ratio = attempt.get('converged_ratio', 'N/A')
                
                # 生成特征描述
                feature_desc = []
                if features.get('mu'):
                    feature_desc.append(f"mu={features['mu']}")
                if features.get('tau_inc'):
                    feature_desc.append(f"tau_inc={features['tau_inc']}")
                if features.get('tau_dec'):
                    feature_desc.append(f"tau_dec={features['tau_dec']}")
                if features.get('has_momentum'):
                    feature_desc.append("使用动量")
                if features.get('has_smoothing'):
                    feature_desc.append("使用平滑")
                
                feature_str = ", ".join(feature_desc) if feature_desc else "基础策略"
                summary += f"   {i}. 第{attempt['iteration']}轮: 性能={perf:.1f}, 收敛={conv_ratio}, 特征=[{feature_str}]\n"
            summary += "\n"
        
        # 显示成功的尝试（这些方向可以参考）
        if successful:
            summary += "✅ 以下方向效果较好，可以参考并微调：\n"
            for i, attempt in enumerate(successful[-3:], 1):  # 只显示最近3个
                features = attempt.get('features', {})
                perf = attempt.get('performance', 0)
                improvement = attempt.get('improvement', 0)
                
                feature_desc = []
                if features.get('mu'):
                    feature_desc.append(f"mu={features['mu']}")
                if features.get('tau_inc'):
                    feature_desc.append(f"tau_inc={features['tau_inc']}")
                if features.get('tau_dec'):
                    feature_desc.append(f"tau_dec={features['tau_dec']}")
                
                feature_str = ", ".join(feature_desc) if feature_desc else "基础策略"
                summary += f"   {i}. 第{attempt['iteration']}轮: 性能={perf:.1f}, 改进={improvement:+.1f}%, 特征=[{feature_str}]\n"
            summary += "\n"
        
        # 生成建议
        summary += "【基于历史的建议】\n"
        
        # 分析失败策略的共同特征
        if len(failed) >= 2:
            failed_mus = [f['features'].get('mu') for f in failed if f['features'].get('mu')]
            if failed_mus:
                avg_failed_mu = sum(failed_mus) / len(failed_mus)
                summary += f"- 失败策略平均mu={avg_failed_mu:.1f}，建议尝试不同的mu值\n"
        
        # 分析成功策略的共同特征
        if successful:
            success_mus = [s['features'].get('mu') for s in successful if s['features'].get('mu')]
            if success_mus:
                avg_success_mu = sum(success_mus) / len(success_mus)
                summary += f"- 成功策略平均mu={avg_success_mu:.1f}，建议在此基础上微调\n"
        
        summary += "- 请尝试与失败策略不同的参数组合\n"
        summary += "- 优先在成功策略基础上做小幅调整\n"
        
        return summary

    def _calculate_average_performance(self, results: Dict[str, Any]) -> float:
        """计算平均性能"""
        if not results:
            return float('inf')
        total_iterations = 0
        count = 0
        for problem_name, result in results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    # 如果评估出错，给予一个很大的惩罚值
                    total_iterations += self.config.get('evaluator', {}).get('max_iterations', 1000) * 2
                    count += 1
                else:
                    iterations = result.get('iterations', 0)
                    converged = result.get('converged', False)
                    if not converged:
                        # 如果未收敛，也给予惩罚
                        iterations = max(iterations, self.config.get('evaluator', {}).get('max_iterations', 1000))
                    total_iterations += iterations
                    count += 1
            else:
                # 如果结果格式不对，也给予惩罚
                total_iterations += self.config.get('evaluator', {}).get('max_iterations', 1000) * 3
                count += 1
        return total_iterations / count if count > 0 else float('inf')

    # ============== "智能调用千问API"新增方法 ==============
    
    def _should_call_advisor(self, current_performance: float) -> bool:
        """
        判断是否应该调用千问指导者
        
        Args:
            current_performance: 当前策略的性能
            
        Returns:
            是否应该调用千问指导者
        """
        # 如果未启用智能调用模式，则每次都调用
        if not self.smart_call_config.get('enabled', True):
            return True
        
        # 如果连续无改进次数达到阈值，触发调用
        if self.consecutive_no_improvement >= self.smart_call_config['no_improvement_threshold']:
            self.logger.info(f"连续无改进次数({self.consecutive_no_improvement})达到阈值({self.smart_call_config['no_improvement_threshold']})，触发千问指导者分析")
            return True
        
        # 如果是第一次迭代，不调用（因为没有历史数据）
        if self.iteration == 1:
            return False
        
        # 其他情况不调用
        return False
    
    def _record_advisor_call(self, iteration: int, reason: str, 
                            duration: float, guidance_summary: str):
        """
        记录千问API调用历史
        
        Args:
            iteration: 迭代轮次
            reason: 调用原因
            duration: 调用耗时（秒）
            guidance_summary: 指导建议摘要
        """
        call_record = {
            'iteration': iteration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'reason': reason,
            'duration_seconds': round(duration, 2),
            'consecutive_no_improvement': self.consecutive_no_improvement,
            'best_performance': self.best_performance,
            'guidance_summary': guidance_summary
        }
        
        self.advisor_call_history.append(call_record)
        
        # 如果启用了调用历史保存，则保存到文件
        if self.smart_call_config.get('call_history_enabled', True):
            self._save_advisor_call_history()
    
    def _save_advisor_call_history(self):
        """保存千问API调用历史到文件"""
        if not self.smart_call_config.get('call_history_enabled', True):
            return
        
        history_file = self.smart_call_config.get('call_history_file', 'advisor_call_history.json')
        
        try:
            import json
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'call_history': self.advisor_call_history,
                    'total_calls': len(self.advisor_call_history),
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"千问API调用历史已保存到: {history_file}")
        except Exception as e:
            self.logger.error(f"保存千问API调用历史失败: {e}")

    def _output_final_results(self):
        """输出最终结果"""
        self.logger.info("=" * 50)
        self.logger.info("进化调参完成")
        self.logger.info(f"总迭代轮次: {self.iteration}")
        if self.best_strategy:
            self.logger.info(f"最佳策略路径: {self.best_strategy['path']}")
            self.logger.info(f"最佳性能 (平均迭代次数): {self.best_strategy['performance']:.2f}")
        else:
            self.logger.info("未找到满足条件的最佳策略。")

        # 保存结果到文件
        results_file = f"results/final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'best_strategy': self.best_strategy,
                'history': self.history,
                'config': self.config,
                'final_iteration': self.iteration,
                'advisor_call_history': self.advisor_call_history,
                'total_advisor_calls': len(self.advisor_call_history),
                'consecutive_no_improvement': self.consecutive_no_improvement
            }, f, indent=2, default=str, ensure_ascii=False)
        self.logger.info(f"结果已保存到: {results_file}")
        
        # 输出千问API调用统计
        if self.advisor_call_history:
            self.logger.info("=" * 50)
            self.logger.info("千问API调用统计")
            self.logger.info(f"总调用次数: {len(self.advisor_call_history)}")
            self.logger.info(f"连续无改进轮次: {self.consecutive_no_improvement}")
            if self.advisor_call_history:
                total_duration = sum(call['duration_seconds'] for call in self.advisor_call_history)
                self.logger.info(f"总调用耗时: {total_duration:.2f}秒")
                self.logger.info(f"平均调用耗时: {total_duration/len(self.advisor_call_history):.2f}秒")

if __name__ == "__main__":
    main = EvolutionaryTuningMain("config.yaml")
    main.run()