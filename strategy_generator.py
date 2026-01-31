# strategy_generator.py
import openai
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StrategyGenerator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略生成器
        Args:
            config: 从 config.yaml 加载的配置
        """
        api_config = config['api']
        self.client = openai.OpenAI(
            api_key=api_config['api_key'],
            base_url=api_config.get('base_url', 'https://api.deepseek.com')
        )
        self.model = api_config.get('model', 'deepseek-coder')
        self.temperature = api_config.get('temperature', 0.7)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        """
        构建固定的系统提示，定义任务和约束
        """
        return """你是一个专业的Python程序员，专门编写ADMM算法的参数自适应调优策略。

【核心原则 - 渐进式改进】
你的任务是对已有策略进行**小幅、渐进式的改进**，而不是重新设计整个算法。
- 每次只修改1-2个超参数的值，或者调整一小段逻辑
- 保留已有策略中表现良好的部分
- 避免大幅度重构或完全重写

【强制要求 - 必须严格遵守】

1. 类必须继承 BaseTuningStrategy
2. update_parameters方法签名必须完全匹配：
   def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
   
3. 【禁止】使用以下参数名作为update_parameters的参数：
   - residuals, residual, res
   - variables, vars, x
   - params, parameters, p
   - beta, rho, penalty
   - iteration, iter, k
   
   所有这些信息都必须从 iteration_state 字典中获取！

4. iteration_state字典包含的键：
   - 'iteration': 当前迭代次数
   - 'primal_residual': 原始残差
   - 'dual_residual': 对偶残差  
   - 'beta': 当前beta值
   - 'objective': 目标函数值
   - 'converged': 是否收敛

5. 返回值必须是包含'beta'键的字典：{'beta': new_beta_value}

【正确示例】
```python
from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class MyADMMStrategy(BaseTuningStrategy):
    def __init__(self):
        self.min_beta = 1e-6
        self.max_beta = 1e6
        self.mu = 10.0
        self.tau_inc = 2.0
        self.tau_dec = 2.0
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从iteration_state获取所有需要的信息
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        current_beta = iteration_state.get('beta', 1.0)
        
        # 计算新beta（示例：基于残差比的调整）
        if primal_res is None or dual_res is None:
            return {'beta': current_beta}
        
        if dual_res > 1e-10:
            ratio = primal_res / dual_res
            if ratio > self.mu:
                new_beta = current_beta * self.tau_inc
            elif ratio < 1.0 / self.mu:
                new_beta = current_beta / self.tau_dec
            else:
                new_beta = current_beta
        else:
            new_beta = current_beta
            
        new_beta = np.clip(new_beta, self.min_beta, self.max_beta)
        return {'beta': float(new_beta)}
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'min_beta': self.min_beta, 
            'max_beta': self.max_beta,
            'mu': self.mu,
            'tau_inc': self.tau_inc,
            'tau_dec': self.tau_dec
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        if 'min_beta' in params:
            self.min_beta = params['min_beta']
        if 'max_beta' in params:
            self.max_beta = params['max_beta']
        if 'mu' in params:
            self.mu = params['mu']
```

【错误示例 - 绝对禁止】
```python
# 错误：update_parameters使用了错误的参数
def update_parameters(self, residuals, beta):  # 错误！不允许这样的签名
    ...

# 错误：使用多个参数
def update_parameters(self, primal_residual, dual_residual, beta):  # 错误！
    ...
```

请直接输出符合上述要求的Python代码，代码必须用```python和```包裹。
"""

    def _validate_and_fix_code(self, code: str) -> str:
        """
        验证并修正生成的代码
        
        Args:
            code: 生成的策略代码
            
        Returns:
            验证/修正后的代码
        """
        # 1. 确保有正确的导入语句
        if 'from strategies.base_strategy import BaseTuningStrategy' not in code:
            code = 'from strategies.base_strategy import BaseTuningStrategy\n' + code
            logger.info("自动添加BaseTuningStrategy导入语句")
        
        if 'from typing import Dict, Any' not in code:
            # 检查是否有其他形式的typing导入
            if 'from typing import' in code:
                # 已有typing导入，检查是否包含Dict和Any
                if 'Dict' not in code or 'Any' not in code:
                    logger.warning("typing导入可能不完整，建议检查")
            else:
                code = 'from typing import Dict, Any\n' + code
                logger.info("自动添加typing导入语句")
        
        # 2. 检测并修正错误的方法签名
        wrong_signature_patterns = [
            r'def update_parameters\s*\(\s*self\s*,\s*residuals[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*residual[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*params[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*parameters[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*variables[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*vars[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*beta[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*rho[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*state\s*[^i][^)]*\)',  # state但不是iteration_state
            r'def update_parameters\s*\(\s*self\s*,\s*primal[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*dual[^)]*\)',
        ]
        
        correct_signature = 'def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:'
        
        for pattern in wrong_signature_patterns:
            if re.search(pattern, code):
                logger.warning(f"检测到错误的方法签名模式: {pattern}")
                # 替换整行到冒号
                code = re.sub(pattern + r'[^:]*:', correct_signature, code)
                logger.info("已自动修正方法签名")
        
        # 3. 验证是否有正确的签名
        if 'def update_parameters(self, iteration_state' not in code:
            # 尝试更宽松的匹配
            if 'def update_parameters' in code:
                logger.warning("代码中存在update_parameters方法但签名可能不正确")
            else:
                logger.error("代码中未找到update_parameters方法")
        
        # 4. 检查是否有numpy导入（如果代码中使用了np）
        if 'np.' in code and 'import numpy' not in code:
            code = 'import numpy as np\n' + code
            logger.info("自动添加numpy导入语句")
        
        return code

    def generate_strategy(self, algorithm_type: str, feedback: str = "") -> Optional[str]:
        """
        生成新的调优策略代码
        Args:
            algorithm_type: 算法类型 ('admm', 'gradient_descent', etc.)
            feedback: 来自反馈循环的评估结果和建议
        Returns:
            生成的Python策略代码字符串，如果失败则返回None
        """
        if algorithm_type != 'admm':
            logger.error(f"策略生成器尚未支持非ADMM算法: {algorithm_type}")
            return None

        # 构建用户消息，结合反馈
        user_message_content = f"""
当前任务：基于已有策略进行渐进式改进，优化ADMM算法的自适应惩罚参数(beta)调整策略。

{feedback}

【改进要求】
1. 必须基于上面提供的策略代码进行修改，不要从头重写
2. 每次只做一个小的改动，例如：
   - 调整某个超参数的值（如 mu, tau_inc, tau_dec 等）
   - 修改残差比较的阈值
   - 调整 beta 的范围限制
   - 添加平滑系数或改进更新公式
3. 保留历史最优策略中有效的核心逻辑
4. 避免大幅重构或完全不同的设计

请输出改进后的完整Python代码，代码必须用```python和```包裹。
"""

        try:
            logger.info(f"正在调用DeepSeek API生成策略 (Algorithm: {algorithm_type})...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message_content}
                ],
                temperature=self.temperature,
                # max_tokens=2048, # 可选，限制回复长度
            )

            generated_text = response.choices[0].message.content
            logger.debug(f"API原始响应:\n{generated_text}")

            # 尝试从响应中提取Python代码块
            # 优先匹配 ```python ... ```
            code_match = re.search(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
            if not code_match:
                # 如果没找到，尝试匹配 ``` ... ```
                code_match = re.search(r'```\n(.*?)\n```', generated_text, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1).strip()
                # 验证并修正代码
                extracted_code = self._validate_and_fix_code(extracted_code)
                logger.info("成功从API响应中提取并验证Python代码块。")
                return extracted_code
            else:
                logger.warning("API响应中未找到明确的Python代码块标记 (```python ... ``` 或 ``` ... ```)，返回完整响应。")
                # 如果没有代码块标记，则假设整个响应就是代码（风险较高）
                raw_code = generated_text.strip()
                # 同样进行验证修正
                return self._validate_and_fix_code(raw_code)

        except openai.AuthenticationError as e:
            logger.error(f"DeepSeek API 认证失败: {e}")
            logger.error("请检查 config.yaml 中的 api_key 是否正确配置。")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"DeepSeek API 连接失败: {e}")
            logger.error("请检查网络连接和 API base_url 是否正确。")
            return None
        except openai.RateLimitError as e:
            logger.error(f"DeepSeek API 速率限制错误: {e}")
            logger.error("请检查API配额或稍后重试。")
            return None
        except Exception as e:
            logger.error(f"调用DeepSeek API生成策略时发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- 示例用法 ---
# config = {...} # 从 config.yaml 加载
# generator = StrategyGenerator(config)
# new_strategy_code = generator.generate_strategy(algorithm_type="admm", feedback="上次策略收敛太慢...")
# print(new_strategy_code)