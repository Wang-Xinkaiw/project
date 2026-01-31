# advisor.py
"""
千问指导者模块 - 使用Qwen3-235B-A22B作为ADMM优化专家
分析每轮评估结果，为DeepSeek提供优化方向建议
"""

import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from http import HTTPStatus

logger = logging.getLogger(__name__)


class ADMMAdvisor:
    """ADMM算法优化指导者，使用千问模型进行深度分析"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化指导者
        
        Args:
            config: 配置字典，需包含advisor节
        """
        self.config = config
        self.advisor_config = config.get('advisor', {})
        self.enabled = False
        
        # 检查是否启用
        if not self.advisor_config.get('enabled', False):
            logger.info("指导者模块未启用")
            return
        
        # 验证API密钥
        api_key = self.advisor_config.get('api_key', '')
        if not api_key or api_key == 'your_dashscope_api_key':
            logger.error("千问API密钥未配置，指导者模块将被禁用")
            return
        
        # 初始化DashScope
        try:
            import dashscope
            dashscope.api_key = api_key
            self.dashscope = dashscope
            self.enabled = True
            logger.info(f"千问指导者模块初始化成功，模型: {self.advisor_config.get('model', 'qwen3-235b-a22b')}")
        except ImportError:
            logger.error("dashscope库未安装，请运行: pip install dashscope")
            return
        except Exception as e:
            logger.error(f"初始化DashScope失败: {e}")
            return
        
        # 配置参数
        self.model = self.advisor_config.get('model', 'qwen3-235b-a22b')
        self.temperature = self.advisor_config.get('temperature', 0.3)
        self.max_tokens = self.advisor_config.get('max_tokens', 4000)
        self.timeout = self.advisor_config.get('timeout', 60)
        self.save_reports = self.advisor_config.get('save_reports', True)
        self.reports_dir = self.advisor_config.get('reports_dir', 'advisor_reports')
        
        # 创建报告目录
        if self.save_reports:
            os.makedirs(self.reports_dir, exist_ok=True)
        
        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建千问的系统提示词"""
        return """你是ADMM(交替方向乘子法)算法优化专家，专门分析惩罚参数beta的自适应调整策略。

【ADMM算法背景】
ADMM通过迭代更新原始变量x、辅助变量z和拉格朗日乘子y来求解优化问题：
- 原始残差(primal_residual): ||Ax - z||，衡量原始约束的违反程度
- 对偶残差(dual_residual): ||beta*(z_new - z_old)||，衡量对偶变量的变化幅度
- Beta参数: 惩罚系数，平衡原始/对偶残差的收敛速度
  - beta过大：原始残差收敛快，但对偶残差可能振荡
  - beta过小：对偶残差收敛快，但原始残差收敛慢

【7个测试问题】
1. l1_regularization - L1范数最小化：min ||X||_1 s.t. AX=B
2. elastic_net - 弹性网络：min ||X||_1 + λ||X||²_F s.t. AX=B
3. l1_regression - L1正则化回归：min λ||X||_1 + Loss(AX+E-B)
4. elastic_net_regression - 弹性网络回归：双正则化+误差项
5. low_rank_matrix_completion - 低秩矩阵补全：核范数最小化
6. low_rank_representation - 低秩表示：子空间聚类问题
7. robust_multi_view_spectral_clustering - 鲁棒多视图谱聚类：多视图融合

【你的分析任务】
1. 分析7个问题的整体收敛情况（收敛率、平均迭代次数）
2. 识别残差模式：
   - 原始残差大、对偶残差小 → beta应增大
   - 对偶残差大、原始残差小 → beta应减小
   - 两者都大 → 需要更保守的调整策略
3. 评估当前beta调整策略的有效性
4. 给出具体、可操作的优化建议

【关键参数说明】
- mu: 残差比阈值，控制何时调整beta（常见值：2-20）
- tau_inc: beta增大比率（常见值：1.5-2.5）
- tau_dec: beta减小比率（常见值：1.5-2.5）
- min_beta/max_beta: beta的取值范围限制

【输出格式要求】
请按以下结构输出分析，保持简洁专业：

## 整体诊断
[7个问题的收敛率、平均迭代次数、主要瓶颈]

## 关键问题分析
[针对未收敛或迭代次数最多的问题，分析残差模式和可能原因]

## 策略评估
[当前beta调整策略的优点和不足]

## 优化建议
[具体的参数调整方向，如：
- 调整mu值到XX
- 将tau_inc/tau_dec改为XX
- 调整beta范围到[XX, XX]
- 引入XX机制（如平滑、动量等）]

## 风险提示
[可能导致性能恶化的操作]"""

    def analyze_evaluation_results(self, evaluation_results: Dict[str, Any], 
                                    iteration: int, 
                                    history: List[Dict[str, Any]]) -> str:
        """
        分析评估结果并生成优化建议
        
        Args:
            evaluation_results: 7个问题的评估结果
            iteration: 当前迭代轮次
            history: 历史迭代记录
            
        Returns:
            千问生成的分析和建议文本
        """
        if not self.enabled:
            return ""
        
        start_time = time.time()
        
        # 构建用户提示词
        user_prompt = self._build_user_prompt(evaluation_results, iteration, history)
        
        # 调用千问API
        guidance = self._call_qwen_api(user_prompt)
        
        duration = time.time() - start_time
        logger.info(f"千问指导者分析完成，耗时: {duration:.2f}秒")
        
        return guidance
    
    def _build_user_prompt(self, results: Dict[str, Any], 
                           iteration: int, 
                           history: List[Dict[str, Any]]) -> str:
        """构建发送给千问的用户提示词"""
        prompt_parts = []
        
        # 1. 当前轮次信息
        prompt_parts.append(f"【第 {iteration} 轮评估结果分析】\n")
        
        # 2. 格式化7个问题的详细结果
        prompt_parts.append(self._format_results_for_analysis(results))
        
        # 3. 提取收敛模式统计
        patterns = self._extract_convergence_patterns(results)
        prompt_parts.append(self._format_patterns(patterns))
        
        # 4. 历史上下文
        if history:
            prompt_parts.append(self._build_historical_context(history, iteration))
        
        # 5. 分析请求
        prompt_parts.append("\n请根据以上数据进行深度分析，给出针对性的优化建议。")
        
        return "\n".join(prompt_parts)
    
    def _format_results_for_analysis(self, results: Dict[str, Any]) -> str:
        """格式化评估结果为易读文本"""
        lines = ["【各问题详细结果】\n"]
        
        for problem_name, result in results.items():
            if not isinstance(result, dict):
                lines.append(f"  {problem_name}: 无效结果\n")
                continue
            
            if 'error' in result:
                lines.append(f"  {problem_name}: 错误 - {result['error'][:100]}\n")
                continue
            
            iterations = result.get('iterations', 0)
            converged = result.get('converged', False)
            final_beta = result.get('final_beta', 'N/A')
            final_obj = result.get('final_objective', 'N/A')
            
            status = "已收敛" if converged else "未收敛"
            
            # 获取最终残差
            conv_history = result.get('convergence_history', [])
            if conv_history:
                last_record = conv_history[-1]
                primal_res = last_record.get('primal_residual', 'N/A')
                dual_res = last_record.get('dual_residual', 'N/A')
            else:
                primal_res = 'N/A'
                dual_res = 'N/A'
            
            # 格式化数值
            beta_str = f"{final_beta:.4f}" if isinstance(final_beta, (int, float)) else str(final_beta)
            obj_str = f"{final_obj:.2e}" if isinstance(final_obj, (int, float)) else str(final_obj)
            primal_str = f"{primal_res:.2e}" if isinstance(primal_res, (int, float)) else str(primal_res)
            dual_str = f"{dual_res:.2e}" if isinstance(dual_res, (int, float)) else str(dual_res)
            
            lines.append(f"  {problem_name}:")
            lines.append(f"    - 状态: {status}, 迭代次数: {iterations}")
            lines.append(f"    - 最终beta: {beta_str}")
            lines.append(f"    - 原始残差: {primal_str}, 对偶残差: {dual_str}")
            lines.append(f"    - 目标函数值: {obj_str}\n")
        
        return "\n".join(lines)
    
    def _extract_convergence_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """提取收敛模式统计"""
        patterns = {
            'total_problems': 0,
            'converged_count': 0,
            'unconverged_count': 0,
            'error_count': 0,
            'avg_iterations': 0,
            'high_primal_residual': [],  # 原始残差较大的问题
            'high_dual_residual': [],    # 对偶残差较大的问题
            'both_high': [],              # 两者都大的问题
            'iterations_list': []
        }
        
        for problem_name, result in results.items():
            if not isinstance(result, dict):
                continue
            
            patterns['total_problems'] += 1
            
            if 'error' in result:
                patterns['error_count'] += 1
                continue
            
            iterations = result.get('iterations', 0)
            patterns['iterations_list'].append(iterations)
            
            if result.get('converged', False):
                patterns['converged_count'] += 1
            else:
                patterns['unconverged_count'] += 1
                
                # 分析残差模式
                conv_history = result.get('convergence_history', [])
                if conv_history:
                    last = conv_history[-1]
                    primal = last.get('primal_residual', 0)
                    dual = last.get('dual_residual', 0)
                    
                    threshold = 1e-4
                    if isinstance(primal, (int, float)) and isinstance(dual, (int, float)):
                        if primal > threshold and dual > threshold:
                            patterns['both_high'].append(problem_name)
                        elif primal > threshold:
                            patterns['high_primal_residual'].append(problem_name)
                        elif dual > threshold:
                            patterns['high_dual_residual'].append(problem_name)
        
        if patterns['iterations_list']:
            patterns['avg_iterations'] = sum(patterns['iterations_list']) / len(patterns['iterations_list'])
        
        return patterns
    
    def _format_patterns(self, patterns: Dict[str, Any]) -> str:
        """格式化收敛模式统计"""
        lines = ["\n【收敛模式统计】"]
        
        total = patterns['total_problems']
        converged = patterns['converged_count']
        unconverged = patterns['unconverged_count']
        errors = patterns['error_count']
        avg_iter = patterns['avg_iterations']
        
        lines.append(f"  - 总问题数: {total}")
        lines.append(f"  - 已收敛: {converged}/{total} ({converged/total*100:.1f}%)" if total > 0 else "  - 已收敛: 0")
        lines.append(f"  - 未收敛: {unconverged}")
        lines.append(f"  - 出错: {errors}")
        lines.append(f"  - 平均迭代次数: {avg_iter:.1f}")
        
        if patterns['high_primal_residual']:
            lines.append(f"  - 原始残差较大的问题: {', '.join(patterns['high_primal_residual'])}")
            lines.append("    (建议: 增大beta以加强约束惩罚)")
        
        if patterns['high_dual_residual']:
            lines.append(f"  - 对偶残差较大的问题: {', '.join(patterns['high_dual_residual'])}")
            lines.append("    (建议: 减小beta以缓解过度惩罚)")
        
        if patterns['both_high']:
            lines.append(f"  - 两种残差都大的问题: {', '.join(patterns['both_high'])}")
            lines.append("    (建议: 采用更保守的调整策略)")
        
        return "\n".join(lines)
    
    def _build_historical_context(self, history: List[Dict[str, Any]], current_iteration: int) -> str:
        """构建历史上下文信息"""
        if not history:
            return ""
        
        lines = ["\n【历史性能趋势】"]
        
        # 取最近5轮
        recent = history[-5:] if len(history) >= 5 else history
        
        for record in recent:
            iter_num = record.get('iteration', '?')
            perf = record.get('performance', 0)
            perf_str = f"{perf:.1f}" if isinstance(perf, (int, float)) else str(perf)
            lines.append(f"  第{iter_num}轮: 平均迭代 {perf_str} 次")
        
        # 计算趋势
        if len(recent) >= 2:
            perfs = [r.get('performance', float('inf')) for r in recent]
            if perfs[-1] < perfs[0]:
                trend = "改善"
                change = (perfs[0] - perfs[-1]) / perfs[0] * 100 if perfs[0] > 0 else 0
            elif perfs[-1] > perfs[0]:
                trend = "恶化"
                change = (perfs[-1] - perfs[0]) / perfs[0] * 100 if perfs[0] > 0 else 0
            else:
                trend = "持平"
                change = 0
            
            lines.append(f"\n  整体趋势: {trend} ({change:+.1f}%)")
        
        return "\n".join(lines)
    
    def _call_qwen_api(self, user_prompt: str, retries: int = 3) -> str:
        """
        调用千问API
        
        Args:
            user_prompt: 用户提示词
            retries: 重试次数
            
        Returns:
            千问的响应文本
        """
        from dashscope import Generation
        
        for attempt in range(retries):
            try:
                logger.info(f"调用千问API (尝试 {attempt + 1}/{retries})...")
                
                response = Generation.call(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    result_format='message',
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                if response.status_code == HTTPStatus.OK:
                    content = response.output.choices[0].message.content
                    logger.info("千问API调用成功")
                    return content
                else:
                    logger.warning(f"千问API返回错误: {response.code} - {response.message}")
                    if attempt < retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        
            except Exception as e:
                logger.error(f"千问API调用异常 (尝试 {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        # 所有重试失败，返回降级建议
        logger.warning("千问API调用失败，使用降级建议")
        return self._get_fallback_guidance()
    
    def _get_fallback_guidance(self) -> str:
        """返回降级建议（API不可用时使用）"""
        return """## 整体诊断
千问API暂时不可用，以下是基于规则的基础建议。

## 优化建议
1. 检查未收敛问题的残差比（primal_residual / dual_residual）：
   - 比值 > 10：增大beta，建议tau_inc设为2.0
   - 比值 < 0.1：减小beta，建议tau_dec设为2.0
   - 比值在0.1-10之间：保持当前策略

2. 如果连续多轮性能退化：
   - 回退到历史最优策略
   - 减小调整幅度（降低tau_inc和tau_dec）

3. 通用建议：
   - 保持mu在5-15范围内
   - beta范围限制在[1e-4, 1e4]
   - 避免大幅修改核心逻辑

## 风险提示
- 避免同时修改多个参数
- 不要完全重写策略代码"""
    
    def save_analysis(self, analysis: str, iteration: int):
        """
        保存分析报告
        
        Args:
            analysis: 分析内容
            iteration: 迭代轮次
        """
        if not self.save_reports or not self.enabled:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.reports_dir}/advisor_iter_{iteration}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"千问指导者分析报告\n")
                f.write(f"迭代轮次: {iteration}\n")
                f.write(f"生成时间: {timestamp}\n")
                f.write(f"模型: {self.model}\n")
                f.write("=" * 60 + "\n\n")
                f.write(analysis)
            
            logger.info(f"指导者分析报告已保存: {filename}")
        except Exception as e:
            logger.error(f"保存分析报告失败: {e}")
