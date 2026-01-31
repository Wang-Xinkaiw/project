"""反馈循环模块负责将评估结果组织成反馈信息"""
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import os

class FeedbackLoop:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化反馈循环
        Args:
            config: 配置文件
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_feedback(self, performance_results: Dict[str, Any], history: List[Dict[str, Any]], iteration: int) -> str:
        """
        根据评估结果生成反馈信息
        Args:
            performance_results: 当前轮次的性能结果
            history: 历史迭代记录
            iteration: 当前迭代轮次
        Returns:
            格式化后的反馈信息
        """
        if iteration == 1:
            # 第一轮反馈
            feedback = self._generate_first_feedback(performance_results)
        else:
            # 后续轮次反馈，与历史比较
            previous_results = history[-2]['detailed_results'] if len(history) > 1 else None
            feedback = self._generate_comparative_feedback(
                performance_results, previous_results, history, iteration
            )

        return feedback

    def format_feedback(self, current_results: Dict[str, Any], previous_results: Any = None) -> str:
        """
        格式化反馈信息
        Args:
            current_results: 当前结果（字典）
            previous_results: 上一轮结果
        Returns:
            格式化后的反馈字符串
        """
        feedback_parts = []

        # 检查是否有方法签名错误或其他关键错误
        signature_errors, other_errors = self._categorize_errors(current_results)
        if signature_errors:
            feedback_parts.append("【严重错误】检测到方法签名问题：")
            feedback_parts.extend(signature_errors)
            feedback_parts.append("")
        if other_errors:
             feedback_parts.append("【其他错误】:")
             feedback_parts.extend(other_errors)
             feedback_parts.append("")

        # 总体性能概览
        avg_iterations = self._calculate_average_iterations(current_results)
        feedback_parts.append(f"当前策略在测试集上的平均迭代次数: {avg_iterations:.1f}")

        # 检查previous_results类型
        if previous_results is not None and isinstance(previous_results, dict):
            prev_avg = self._calculate_average_iterations(previous_results)
            if prev_avg > 0:
                improvement = ((prev_avg - avg_iterations) / prev_avg * 100)
                feedback_parts.append(f"相比上一轮改进: {improvement:+.1f}%")
            else:
                feedback_parts.append("这是第一轮评估，或者上一轮数据不可用")

        # 详细问题分析
        feedback_parts.append("\n详细问题表现:")
        for problem_name, result in current_results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    error_msg = result['error']
                    # 提取错误关键信息
                    if "missing" in error_msg and "argument" in error_msg:
                        feedback_parts.append(f" {problem_name}: ❌ 方法签名错误 - {error_msg[:100]}...")
                    else:
                        feedback_parts.append(f" {problem_name}: ❌ 运行时错误 - {error_msg[:100]}...")
                else:
                    iterations = result.get('iterations', 0)
                    converged = result.get('converged', False)
                    status = "✓" if converged else "✗"
                    objective_val = result.get('objective', 'N/A')
                    if isinstance(objective_val, (int, float)):
                        objective_str = f"{objective_val:.2e}"
                    else:
                        objective_str = str(objective_val)
                    feedback_parts.append(f" {problem_name}: {iterations}次迭代 {status} (Objective: {objective_str})")
            else:
                feedback_parts.append(f" {problem_name}: ❌ 无效结果类型 - {type(result).__name__}")

        return "\n".join(feedback_parts)

    def _categorize_errors(self, results: Dict[str, Any]) -> tuple[list[str], list[str]]:
        """分类错误"""
        signature_errs = []
        other_errs = []
        for problem_name, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                 error_msg = result['error']
                 if "missing" in error_msg and "argument" in error_msg:
                     signature_errs.append(f"- {problem_name}: {error_msg}")
                 else:
                     other_errs.append(f"- {problem_name}: {error_msg}")
        return signature_errs, other_errs


    def get_initial_prompt(self, algorithm_type: str) -> str:
        # 此方法现在主要用于 main.py 获取初始提示语句，但核心逻辑已移到 system_prompt
        # 保留此方法以防 main.py 需要
        if algorithm_type == 'admm':
            return """
【任务】生成一个ADMM参数自适应调整策略。
【要求】
1. 继承 `BaseTuningStrategy` 基类。
2. 实现 `update_parameters`, `get_parameters`, `set_parameters` 方法。
3. `update_parameters` 方法签名必须为 `update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]`。
4. 该方法必须返回一个包含 'beta' 键的字典，用于更新惩罚参数。
5. 策略应旨在优化ADMM的收敛速度和稳定性，仅调整 beta。
"""
        return ""

    def save_feedback(self, feedback: str, iteration: int):
        """保存反馈信息到文件"""
        feedback_dir = "feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{feedback_dir}/feedback_iter_{iteration}_{timestamp}.txt"
        # 修复编码问题
        feedback = feedback.replace('•', '*').replace('✅', 'V').replace('❌', 'X')

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"迭代轮次: {iteration}\n")
                f.write(f"生成时间: {timestamp}\n")
                f.write("=" * 50 + "\n")
                f.write(feedback)
        except UnicodeEncodeError as e:
            self.logger.error(f"编码错误: {e}")
            with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(f"迭代轮次: {iteration}\n")
                f.write(f"生成时间: {timestamp}\n")
                f.write("=" * 50 + "\n")
                f.write(feedback)
        self.logger.info(f"反馈已保存到: {filename}")

    def _generate_first_feedback(self, results: Dict[str, Any]) -> str:
        """生成第一轮反馈"""
        feedback = "初始策略评估结果:\n\n"
        successful = 0
        total = 0
        signature_errors = 0

        for problem_name, result in results.items():
            total += 1
            if isinstance(result, dict):
                if 'error' not in result and result.get('converged', False):
                    successful += 1
                # 检查方法签名错误
                if 'error' in result and ("missing" in result['error'] and "argument" in result['error']):
                    signature_errors += 1
                iterations = result.get('iterations', 0)
                converged = result.get('converged', False)
                status = "成功" if converged else "失败"
                feedback += f"{problem_name}: {iterations}次迭代，{status}\n"
                if 'error' in result:
                    error_msg = result['error']
                    if "missing" in error_msg and "argument" in error_msg:
                        feedback += f"  - 错误: ❌ 方法签名不正确 - {error_msg}\n"
                    else:
                        feedback += f"  - 错误: {error_msg[:100]}...\n"
            else:
                feedback += f"{problem_name}: ❌ 无效结果类型\n"

        success_rate = (successful / total * 100) if total > 0 else 0
        feedback += f"\n总体成功率: {success_rate:.1f}% ({successful}/{total})\n"

        # 分析问题并给出具体建议
        if signature_errors > 0:
            feedback += "\n【严重问题】检测到方法签名错误:\n"
            feedback += "1. `update_parameters` 方法必须使用正确签名: `update_parameters(self, iteration_state: Dict[str, Any])`\n"
            feedback += "2. 方法只接受一个参数: `iteration_state` 字典\n"
            feedback += "3. 不允许有其他参数，如 `residuals`, `variables`, `params` 等\n"
            feedback += "4. 返回的字典必须包含 `'beta'` 键\n"
        elif success_rate < 50:
            feedback += "\n分析: 初始策略收敛性较差，建议:\n"
            feedback += "1. 检查 `beta` 调整逻辑的合理性，是否过于激进或保守。\n"
            feedback += "2. 考虑更保守的 `beta` 调整策略，例如基于残差比值的平滑调整。\n"
            feedback += "3. 增加对异常状态（如残差突然增大）的容错处理。\n"
            feedback += "4. 参考标准ADMM策略中 `beta` 的调整公式。\n"
        else:
            feedback += "\n分析: 初始策略表现尚可，优化方向:\n"
            feedback += "1. 进一步优化 `beta` 调整策略以加速收敛。\n"
            feedback += "2. 提高 `beta` 调整策略在不同问题上的泛化能力。\n"
            feedback += "3. 平衡 `beta` 调整的探索与利用，避免过早陷入局部最优。\n"

        # 强调只调整β的要求
        feedback += "\n【重要要求】请确保新策略：\n"
        feedback += "1. `update_parameters` 方法签名: `update_parameters(self, iteration_state: Dict[str, Any])`\n"
        feedback += "2. **只调整惩罚参数 `beta`**\n"
        feedback += "3. 保持ADMM其他参数（如 `rho`, `tau`, `max_iterations`, `tolerance` 等）与标准版本一致，这些参数不由本策略调整。\n"
        feedback += "4. `update_parameters` 方法只返回包含 `'beta'` 键的字典，例如 `{'beta': new_beta_value, ...}`。\n"
        feedback += "5. 类必须继承 `BaseTuningStrategy` 基类。\n"
        feedback += "6. 包含必要的 `import` 语句 (`numpy`, `typing`, `BaseTuningStrategy`)。\n"

        return feedback

    def _generate_comparative_feedback(self, current_results: Dict[str, Any], previous_results: Dict[str, Any], history: List[Dict[str, Any]], iteration: int) -> str:
        """生成比较性反馈"""
        feedback = f"第 {iteration} 轮策略改进分析:\n\n"

        # 检查是否有方法签名错误
        signature_errors, other_errors = self._categorize_errors(current_results)
        if signature_errors:
            feedback += "【严重问题】检测到方法签名错误:\n"
            for err in signature_errors:
                feedback += f"{err}\n"
            feedback += "\n"
        if other_errors:
            feedback += "【其他错误】:\n"
            for err in other_errors:
                 feedback += f"{err}\n"
            feedback += "\n"

        # 计算性能变化
        current_avg = self._calculate_average_iterations(current_results)
        previous_avg = self._calculate_average_iterations(previous_results) if previous_results else current_avg
        improvement = ((previous_avg - current_avg) / previous_avg * 100) if previous_avg > 0 else 0
        feedback += f"平均迭代次数变化: {previous_avg:.1f} → {current_avg:.1f} "
        feedback += f"({improvement:+.1f}%)\n\n"

        # 分析各问题表现
        feedback += "各问题表现对比:\n"
        improvement_problems = []
        degradation_problems = []
        error_problems = []

        for problem_name, result in current_results.items():
            if isinstance(result, dict) and 'error' in result:
                error_problems.append(problem_name)
                feedback += f"- {problem_name}: ❌ 评估错误\n"
                continue

            if problem_name in (previous_results or {}):
                prev_result = previous_results[problem_name]
                if isinstance(prev_result, dict) and 'error' not in prev_result:
                    curr_iter = result.get('iterations', 0)
                    prev_iter = prev_result.get('iterations', 0)
                    if prev_iter > 0:
                        change = (prev_iter - curr_iter) / prev_iter * 100
                        change_symbol = "✅" if change > 0 else "❌"
                        feedback += f"- {problem_name}: {prev_iter} → {curr_iter} {change_symbol}{abs(change):.1f}%\n"
                        # 分类记录
                        if curr_iter > prev_iter * 1.2: # 定义变差为迭代次数增加超过20%
                            degradation_problems.append(problem_name)
                        elif curr_iter < prev_iter * 0.8: # 定义改进为迭代次数减少超过20%
                            improvement_problems.append(problem_name)
                    else:
                         feedback += f"- {problem_name}: Previous iter was 0, cannot compare.\n"
                else:
                     feedback += f"- {problem_name}: Previous result invalid for comparison.\n"
            else:
                 feedback += f"- {problem_name}: No previous result for comparison.\n"


        # 总结改进情况
        feedback += f"\n本轮总结:\n"
        feedback += f"- 整体性能: {'改善' if improvement > 0 else '恶化' if improvement < 0 else '持平'}\n"
        feedback += f"- 明显改进的问题: {len(improvement_problems)}个 ({', '.join(improvement_problems)})\n"
        feedback += f"- 明显变差的问题: {len(degradation_problems)}个 ({', '.join(degradation_problems)})\n"
        feedback += f"- 评估出错的问题: {len(error_problems)}个 ({', '.join(error_problems)})\n"

        if error_problems:
            feedback += f"\n存在错误的问题: {', '.join(error_problems)}\n"

        # 识别模式
        patterns = self._identify_patterns(history)
        if patterns:
            feedback += "\n识别到的趋势/模式:\n"
            for pattern in patterns:
                feedback += f"- {pattern}\n"

        # 生成具体改进建议
        suggestions = self._generate_detailed_suggestions(current_results, previous_results, improvement, error_problems, improvement_problems, degradation_problems)
        feedback += "\n具体改进建议:\n"
        for i, suggestion in enumerate(suggestions, 1):
            feedback += f"{i}. {suggestion}\n"

        # 强调技术约束
        feedback += "\n【再次强调技术约束】请确保新策略：\n"
        feedback += "1. `update_parameters` 方法签名必须为: `update_parameters(self, iteration_state: Dict[str, Any])`\n"
        feedback += "2. **只调整惩罚参数 `beta`**，绝对不能调整 `rho`, `tau`, `max_iterations`, `tolerance` 等其他ADMM参数。\n"
        feedback += "3. 保持与标准ADMM算法的兼容性。\n"
        feedback += "4. 返回的参数字典必须包含 `'beta'` 键，例如 `{'beta': new_beta_value}`。\n"
        feedback += "5. 继承 `BaseTuningStrategy` 基类。\n"
        feedback += "6. 包含必要的 `import` 语句。\n"

        return feedback

    def _calculate_average_iterations(self, results: Any) -> float:
        """计算平均迭代次数"""
        if not isinstance(results, dict):
            self.logger.warning(f"results不是字典类型: {type(results)}")
            return float('inf')
        total_iterations = 0
        count = 0
        for problem_name, result in results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    # 如果出错，给予一个很大的惩罚值
                    total_iterations += self.config.get('evaluator', {}).get('max_iterations', 1000)
                    count += 1
                else:
                    iterations = result.get('iterations', 0)
                    total_iterations += iterations
                    count += 1
            else:
                self.logger.warning(f"问题 {problem_name} 的结果不是字典: {type(result)}")
                total_iterations += self.config.get('evaluator', {}).get('max_iterations', 1000) * 2 # 更大的惩罚
                count += 1
        return total_iterations / count if count > 0 else float('inf')

    def _identify_patterns(self, history: List[Dict[str, Any]]) -> List[str]:
        """识别历史模式"""
        patterns = []
        if len(history) < 3:
            return patterns

        # 检查性能趋势
        performances = [h['performance'] for h in history[-5:]]
        if len(performances) >= 3:
            improvements = [performances[i-1] - performances[i] for i in range(1, len(performances))]
            avg_improvement = sum(improvements) / len(improvements)
            if avg_improvement > 5: # 设定一个阈值认为是持续改进
                patterns.append("策略在最近几轮持续显著改进 (平均迭代次数减少 > 5)")
            elif avg_improvement < -5: # 设定一个阈值认为是持续退化
                patterns.append("策略在最近几轮出现显著退化 (平均迭代次数增加 > 5)")
            elif abs(avg_improvement) < 1:
                patterns.append("策略性能在最近几轮趋于稳定 (平均迭代次数变化 < 1)")

        # 检查错误问题
        recent_error_counts = [len([r for r in h.get('detailed_results', {}).values() if isinstance(r, dict) and 'error' in r]) for h in history[-3:]]
        if any(c > 0 for c in recent_error_counts):
             patterns.append("近期策略生成或评估过程中频繁出现错误")


        return patterns

    def _generate_detailed_suggestions(self, current_results: Dict[str, Any], previous_results: Dict[str, Any], overall_improvement: float, error_problems: list, improvement_problems: list, degradation_problems: list) -> List[str]:
        """生成详细改进建议"""
        suggestions = []

        # 检查是否有方法签名错误
        if error_problems:
            suggestions.append("首要任务：解决导致评估错误的问题，特别是方法签名错误。")
            if any("missing" in current_results[p]['error'] for p in error_problems if 'error' in current_results.get(p, {})):
                 suggestions.append("  - 重新检查 `update_parameters` 的方法签名是否完全符合 `update_parameters(self, iteration_state: Dict[str, Any])`。")
                 suggestions.append("  - 确保返回字典包含 `'beta'` 键。")

        # 分析整体性能变化
        if overall_improvement < -5: # 整体变差且超过阈值
            suggestions.append("整体性能下降明显，建议：")
            suggestions.append("  - 回退到上一个表现较好的策略，或者基于上一个策略进行小幅修改，避免大幅变动。")
            suggestions.append("  - 检查新策略中是否有过于激进的 `beta` 调整逻辑，导致算法不稳定。")

        # 分析具体问题
        if degradation_problems:
            suggestions.append(f"针对变差的问题 ({', '.join(degradation_problems)})，分析其特点：")
            suggestions.append("  - 这些问题是否具有相似的数学性质？例如，都是低秩问题、稀疏问题还是回归问题？")
            suggestions.append("  - 调整策略可能对这类问题不适用，需要更具针对性的设计，或加入问题类型判断逻辑。")

        if improvement_problems:
            suggestions.append(f"对于改进的问题 ({', '.join(improvement_problems)})，尝试提炼成功经验：")
            suggestions.append("  - 其 `beta` 调整逻辑有何共同点？是否可以推广到其他问题？")

        # 通用建议
        if not error_problems: # 如果当前没有错误，再提高级建议
            suggestions.append("考虑引入更复杂的 `beta` 调整机制，例如：")
            suggestions.append("  - 基于历史收敛速度的动态调整。")
            suggestions.append("  - 基于原始/对偶残差比值的非线性调整（而非简单的阈值判断）。")
            suggestions.append("  - 引入平滑系数，避免 `beta` 值剧烈波动。")
            suggestions.append("  - 尝试模拟退火或学习率衰减的思想，早期大胆调整，后期精细微调。")

        return suggestions