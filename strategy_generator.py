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
        构建系统提示，专业指导 DeepSeek 生成 ADMM 惩罚参数调整策略
        """
        return """你是一名 ADMM 算法优化专家，专门编写惩罚参数β(beta) 的自适应调整策略。

【核心任务】
生成一个 Python 类，用于在 ADMM 算法的每次迭代中动态调整惩罚参数β的值。

【强制技术要求】
1. 类继承：必须继承 BaseTuningStrategy 基类
2. 方法签名：update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]
3. 参数获取：必须从 iteration_state 字典获取信息，禁止使用其他参数名
4. 返回值：必须返回 {'beta': new_beta_value} 格式的字典
5. 代码格式：必须用 ```python 和 ``` 包裹完整代码

【iteration_state 字典的键】
- iteration: int，当前迭代次数
- primal_residual: float，原始残差范数（约束违反程度）
- dual_residual: float，对偶残差范数（最优性条件违反程度）
- beta: float，当前惩罚参数值
- objective: float，目标函数值
- converged: bool，是否已收敛

【问题特性说明】
1. l1_regularization, elastic_net：标准稀疏优化问题
2. l1_regression, elastic_net_regression：含噪声项 E 的回归问题
   - 建议采用β单调递增策略（只增不减）
   - 初始β=0.1~1.0，增长率 1.1~1.5，上限 1e4
3. low_rank_matrix_completion, low_rank_representation：低秩矩阵问题
   - β影响核范数惩罚强度，需平滑调整
4. robust_multi_view_spectral_clustering：多视图聚类
   - β需协调多个视图的平衡
5. tracelasso：Trace Lasso 问题
   - β需考虑设计矩阵的相关性

【正确示例 - 标准 ADMM 策略】
```python
from strategies.base_strategy import BaseTuningStrategy
from typing import Dict, Any
import numpy as np

class MyADMMStrategy(BaseTuningStrategy):
    def __init__(self):
        # 超参数配置
        self.min_beta = 1e-6      # beta 下界
        self.max_beta = 1e6       # beta 上界
        self.mu = 10.0            # 残差平衡阈值
        self.tau_inc = 2.0        # beta 增大因子
        self.tau_dec = 2.0        # beta 减小因子
        
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        # 从 iteration_state 获取状态信息
        primal_res = iteration_state.get('primal_residual', 1.0)
        dual_res = iteration_state.get('dual_residual', 1.0)
        current_beta = iteration_state.get('beta', 1.0)
        
        # 处理 None 值
        if primal_res is None or dual_res is None:
            return {'beta': current_beta}
        
        # 基于残差比的自适应调整
        if dual_res > 1e-10:
            ratio = primal_res / dual_res
            if ratio > self.mu:
                # 原始残差大，增大 beta
                new_beta = current_beta * self.tau_inc
            elif ratio < 1.0 / self.mu:
                # 对偶残差大，减小 beta
                new_beta = current_beta / self.tau_dec
            else:
                # 残差平衡，保持 beta
                new_beta = current_beta
        else:
            new_beta = current_beta
        
        # 限制 beta 范围
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
        if 'tau_inc' in params:
            self.tau_inc = params['tau_inc']
        if 'tau_dec' in params:
            self.tau_dec = params['tau_dec']
```

【错误示例 - 绝对禁止】
```python
# ❌ 错误 1：使用错误的参数名
def update_parameters(self, residuals, beta):  # 禁止！

# ❌ 错误 2：使用多个独立参数
def update_parameters(self, primal_residual, dual_residual, beta):  # 禁止！

# ❌ 错误 3：缺少返回值
def update_parameters(self, iteration_state):
    new_beta = 1.0  # 没有 return {'beta': ...}

# ❌ 错误 4：返回格式错误
def update_parameters(self, iteration_state):
    return new_beta  # 禁止！必须返回字典 {'beta': value}
```

【输出要求】
1. 只输出完整的 Python 代码，不要解释说明
2. 代码必须用 ```python 和 ``` 包裹
3. 必须包含所有必要的 import 语句
4. 必须实现 update_parameters, get_parameters, set_parameters 三个方法
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
            logger.info("自动添加 BaseTuningStrategy 导入语句")
        
        if 'from typing import Dict, Any' not in code:
            if 'from typing import' in code:
                if 'Dict' not in code or 'Any' not in code:
                    logger.warning("typing 导入可能不完整，建议检查")
            else:
                code = 'from typing import Dict, Any\n' + code
                logger.info("自动添加 typing 导入语句")
        
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
            r'def update_parameters\s*\(\s*self\s*,\s*state\s*[^i][^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*primal[^)]*\)',
            r'def update_parameters\s*\(\s*self\s*,\s*dual[^)]*\)',
        ]
        
        correct_signature = 'def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:'
        
        for pattern in wrong_signature_patterns:
            if re.search(pattern, code):
                logger.warning(f"检测到错误的方法签名模式：{pattern}")
                code = re.sub(pattern + r'[^:]*:', correct_signature, code)
                logger.info("已自动修正方法签名")
        
        # 3. 验证是否有正确的签名
        if 'def update_parameters(self, iteration_state' not in code:
            if 'def update_parameters' in code:
                logger.warning("代码中存在 update_parameters 方法但签名可能不正确")
            else:
                logger.error("代码中未找到 update_parameters 方法")
        
        # 4. 检查是否有 numpy 导入（如果代码中使用了 np）
        if 'np.' in code and 'import numpy' not in code:
            code = 'import numpy as np\n' + code
            logger.info("自动添加 numpy 导入语句")
        
        return code

    def generate_strategy(self, algorithm_type: str, feedback: str = "") -> Optional[str]:
        """
        生成新的调优策略代码
        Args:
            algorithm_type: 算法类型 ('admm', 'gradient_descent', etc.)
            feedback: 来自 evaluator 的评估反馈
        Returns:
            生成的 Python 策略代码字符串，如果失败则返回 None
        """
        if algorithm_type != 'admm':
            logger.error(f"策略生成器尚未支持非 ADMM 算法：{algorithm_type}")
            return None

        # 构建用户消息，结合 evaluator 的反馈
        user_message_content = f"""【任务】基于评估反馈，生成改进的 ADMM 惩罚参数β自适应调整策略。

{feedback}

【重点优化方向】
1. 调整超参数（initial_beta, min_beta, max_beta, mu, tau_inc, tau_dec 等）
2. 改进 beta 调整逻辑（基于残差比、历史迭代信息等）
3. 针对未收敛问题设计特殊策略
4. 提高策略的鲁棒性和泛化能力

【输出要求】
输出完整的 Python 策略代码，用 ```python 包裹。
"""

        try:
            logger.info(f"正在调用 DeepSeek API 生成策略 (Algorithm: {algorithm_type})...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message_content}
                ],
                temperature=self.temperature,
            )

            generated_text = response.choices[0].message.content
            logger.debug(f"API 原始响应:\n{generated_text}")

            # 提取 Python 代码块
            code_match = re.search(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\n(.*?)\n```', generated_text, re.DOTALL)
            
            if code_match:
                extracted_code = code_match.group(1).strip()
                extracted_code = self._validate_and_fix_code(extracted_code)
                logger.info("成功从 API 响应中提取并验证 Python 代码块。")
                return extracted_code
            else:
                logger.warning("API 响应中未找到代码块标记，返回完整响应。")
                return self._validate_and_fix_code(generated_text.strip())

        except openai.AuthenticationError as e:
            logger.error(f"DeepSeek API 认证失败：{e}")
            logger.error("请检查 config.yaml 中的 api_key 是否正确配置。")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"DeepSeek API 连接失败：{e}")
            return None
        except openai.RateLimitError as e:
            logger.error(f"DeepSeek API 速率限制错误：{e}")
            return None
        except Exception as e:
            logger.error(f"调用 DeepSeek API 生成策略时发生未知错误：{e}")
            import traceback
            traceback.print_exc()
            return None
