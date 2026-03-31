# strategy_generator.py
"""
ADMM 惩罚参数策略生成器模块

功能：
1. 调用 DeepSeek API 生成 adjust_beta 函数
2. 验证和修正生成的代码
3. 提供详细的生成要求和示例
"""
import openai
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StrategyGenerator:
    """
    策略生成器 - 使用 DeepSeek API 生成 ADMM 惩罚参数调整策略
    
    工作流程：
    1. 构建系统提示词（包含技术要求和示例）
    2. 接收 evaluator 的反馈
    3. 调用 DeepSeek API 生成改进的策略代码
    4. 验证并修正生成的代码
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略生成器
        
        从配置文件加载 API 设置，创建 OpenAI 客户端，
        构建系统提示词用于指导代码生成。
        
        Args:
            config (dict): 配置字典，必须包含 'api' 节
                - api.api_key: DeepSeek API 密钥
                - api.base_url: API 基础 URL（可选，默认 DeepSeek 官方）
                - api.model: 模型名称（可选，默认 'deepseek-coder'）
                - api.temperature: 生成温度（可选，默认 0.7）
                
        Attributes:
            client: OpenAI 客户端实例
            model (str): 使用的模型名称
            temperature (float): 生成温度（0-2，越高越随机）
            system_prompt (str): 系统提示词
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
        构建系统提示词，专业指导 DeepSeek 生成 ADMM 惩罚参数调整策略函数
        
        详细说明：
        1. 核心任务：生成 adjust_beta(iteration_state) -> float 函数
        2. 强制技术要求：函数签名、参数获取、返回值格式
        3. iteration_state 字典的键说明
        4. 8 个测试问题的特性说明
        5. 正确示例和错误示例
        6. 输出格式要求
        
        Returns:
            str: 完整的系统提示词
        """
        return """你是一名 ADMM 算法优化专家，专门编写惩罚参数β(beta) 的自适应调整策略函数。

【核心任务】
生成一个 Python 函数，用于在 ADMM 算法的每次迭代中动态调整惩罚参数β的值。

【强制技术要求】
1. 函数签名：def adjust_beta(iteration_state: Dict[str, Any]) -> float:
2. 参数获取：必须从 iteration_state 字典获取信息，禁止使用其他参数名
3. 返回值：必须直接返回 float 类型的 new_beta 值（不是字典）
4. 代码格式：必须用 ```python 和 ``` 包裹完整代码

【iteration_state 字典的键】
- iteration: int，当前迭代次数
- primal_residual: float，原始残差范数（约束违反程度，||Ax-B||）
- dual_residual: float，对偶残差范数（最优性条件违反程度，||x-z||）
- beta: float，当前惩罚参数值（也称为 mu 或 ρ）
- objective: float，目标函数值
- converged: bool，是否已收敛

【关键收敛原理 - 必须理解】
ADMM 算法的收敛性依赖于原始残差和对偶残差的平衡：
1. 当 primal_residual >> dual_residual 时：说明约束满足度差，需要增大 beta 加强惩罚
2. 当 dual_residual >> primal_residual 时：说明对偶变量变化慢，需要减小 beta 缓解过度惩罚
3. 理想状态：两个残差以相似速率下降，保持平衡

【问题特性与推荐策略】
1. l1_regularization, elastic_net：标准稀疏优化问题
   - 推荐使用残差比策略或单调递增策略
   - beta 范围：[1e-6, 1e6]

2. l1_regression, elastic_net_regression：含噪声项的回归问题（最难收敛）
   - 必须采用 beta 单调递增策略（只增不减）
   - 初始 beta=0.1~1.0，增长率 1.05~1.2，上限 1e4
   - 原因：回归问题需要逐步加强惩罚以逼近期望解

3. low_rank_matrix_completion, low_rank_representation：低秩矩阵问题
   - beta 影响核范数惩罚强度，需平滑调整
   - 避免大幅度跳变，建议使用温和的调整因子（1.5-2.0）

4. robust_multi_view_spectral_clustering：多视图聚类
   - beta 需协调多个视图的平衡
   - 建议使用残差比策略

5. tracelasso：Trace Lasso 问题
   - beta 需考虑设计矩阵的相关性
   - 建议使用保守的调整策略

【推荐策略模板 1 - 残差比策略（通用型）】
```python
from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    \"\"\"
    基于残差比的自适应 beta 调整策略
    
    核心思想：根据原始残差和对偶残差的比例动态调整 beta
    - 当 primal/dual > 10 时：增大 beta（加强惩罚）
    - 当 primal/dual < 0.1 时：减小 beta（缓解惩罚）
    - 当 0.1 <= primal/dual <= 10 时：保持 beta（平衡状态）
    \"\"\"
    primal_res = iteration_state.get('primal_residual', 1.0)
    dual_res = iteration_state.get('dual_residual', 1.0)
    current_beta = iteration_state.get('beta', 1.0)
    
    # 防止除零和无效值
    if primal_res is None or dual_res is None or dual_res < 1e-10:
        return current_beta
    
    # 计算残差比
    ratio = primal_res / dual_res
    
    # 超参数
    mu = 10.0           # 平衡阈值
    tau_inc = 2.0       # 增大因子
    tau_dec = 2.0       # 减小因子
    min_beta = 1e-6     # beta 下界
    max_beta = 1e6      # beta 上界
    
    # 根据残差比调整 beta
    if ratio > mu:
        new_beta = current_beta * tau_inc
    elif ratio < 1.0 / mu:
        new_beta = current_beta / tau_dec
    else:
        new_beta = current_beta
    
    # 限制范围
    return float(np.clip(new_beta, min_beta, max_beta))
```

【推荐策略模板 2 - 单调递增策略（回归问题专用）】
```python
from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    \"\"\"
    单调递增的 beta 调整策略（专为回归问题设计）
    
    核心思想：beta 只增不减，逐步加强惩罚力度
    - 初始 beta 较小（0.1-1.0）
    - 每轮以固定增长率（1.05-1.2）递增
    - 达到上限后保持不变
    \"\"\"
    current_beta = iteration_state.get('beta', 1.0)
    iteration = iteration_state.get('iteration', 0)
    
    # 超参数
    growth_rate = 1.1      # 增长率（1.05-1.2）
    max_beta = 1e4         # 上限
    min_beta = 0.1         # 初始值
    
    # 单调递增
    new_beta = current_beta * growth_rate
    
    # 限制范围
    return float(np.clip(new_beta, min_beta, max_beta))
```

【推荐策略模板 3 - 迭代感知策略（进阶）】
```python
from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    \"\"\"
    考虑迭代次数的自适应策略
    
    核心思想：
    - 前期（iteration<100）：采用激进的调整策略
    - 后期（iteration>=100）：采用保守的调整策略，避免震荡
    \"\"\"
    primal_res = iteration_state.get('primal_residual', 1.0)
    dual_res = iteration_state.get('dual_residual', 1.0)
    current_beta = iteration_state.get('beta', 1.0)
    iteration = iteration_state.get('iteration', 0)
    
    if primal_res is None or dual_res is None or dual_res < 1e-10:
        return current_beta
    
    ratio = primal_res / dual_res
    
    # 根据迭代次数调整策略
    if iteration < 100:
        # 前期：激进调整
        mu = 5.0
        tau_inc = 3.0
        tau_dec = 3.0
    else:
        # 后期：保守调整
        mu = 10.0
        tau_inc = 1.5
        tau_dec = 1.5
    
    if ratio > mu:
        new_beta = current_beta * tau_inc
    elif ratio < 1.0 / mu:
        new_beta = current_beta / tau_dec
    else:
        new_beta = current_beta
    
    return float(np.clip(new_beta, 1e-6, 1e6))
```

【错误示例 - 绝对禁止】
```python
# ❌ 错误 1：定义类而不是函数
class BetaAdjuster:  # 禁止！需要的是函数不是类

# ❌ 错误 2：使用错误的参数名
def adjust_beta(residuals, beta):  # 禁止！

# ❌ 错误 3：使用多个独立参数
def adjust_beta(primal_residual, dual_residual, beta):  # 禁止！

# ❌ 错误 4：返回类型错误
def adjust_beta(iteration_state):
    return {'beta': new_beta}  # 禁止！必须直接返回 float

# ❌ 错误 5：没有返回值
def adjust_beta(iteration_state):
    new_beta = 1.0  # 没有 return

# ❌ 错误 6：除零错误
def adjust_beta(iteration_state):
    ratio = primal_res / dual_res  # 没有检查 dual_res 是否为 0

# ❌ 错误 7：beta 范围过大导致数值不稳定
new_beta = current_beta * 100  # 禁止！增长因子过大
```

【优化建议 - 提高收敛率】
1. 避免过大的调整因子（建议 1.5-3.0，不要超过 5.0）
2. 必须处理除零情况（dual_res < 1e-10）
3. 必须处理 None 值
4. beta 范围建议：[1e-6, 1e6]，回归问题 [0.1, 1e4]
5. 对于回归问题，优先使用单调递增策略
6. 考虑迭代次数，后期采用更保守的策略

【输出要求】
1. 只输出完整的 Python 函数代码，不要解释说明
2. 代码必须用 ```python 和 ``` 包裹
3. 必须包含所有必要的 import 语句（typing, numpy）
4. 必须实现 adjust_beta 函数，签名严格符合要求
5. 优先使用上述推荐策略模板，或在其基础上改进
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
        if 'from typing import Dict, Any' not in code:
            if 'from typing import' in code:
                if 'Dict' not in code or 'Any' not in code:
                    logger.warning("typing 导入可能不完整，建议检查")
            else:
                code = 'from typing import Dict, Any\n' + code
                logger.info("自动添加 typing 导入语句")
        
        # 2. 检测并修正错误的函数签名
        wrong_signature_patterns = [
            r'def adjust_beta\s*\(\s*residuals[^)]*\)',
            r'def adjust_beta\s*\(\s*residual[^)]*\)',
            r'def adjust_beta\s*\(\s*params[^)]*\)',
            r'def adjust_beta\s*\(\s*parameters[^)]*\)',
            r'def adjust_beta\s*\(\s*variables[^)]*\)',
            r'def adjust_beta\s*\(\s*vars[^)]*\)',
            r'def adjust_beta\s*\(\s*beta[^)]*\)',
            r'def adjust_beta\s*\(\s*rho[^)]*\)',
            r'def adjust_beta\s*\(\s*state\s*[^i][^)]*\)',
            r'def adjust_beta\s*\(\s*primal[^)]*\)',
            r'def adjust_beta\s*\(\s*dual[^)]*\)',
            r'def update_parameters.*',  # 禁止使用 update_parameters
        ]
        
        correct_signature = 'def adjust_beta(iteration_state: Dict[str, Any]) -> float:'
        
        for pattern in wrong_signature_patterns:
            if re.search(pattern, code):
                logger.warning(f"检测到错误的方法签名模式：{pattern}")
                code = re.sub(pattern + r'[^:]*:', correct_signature, code)
                logger.info("已自动修正函数签名")
        
        # 3. 验证是否有正确的签名
        if 'def adjust_beta(self, iteration_state' not in code and 'def adjust_beta(iteration_state' not in code:
            if 'def adjust_beta' in code:
                logger.warning("代码中存在 adjust_beta 函数但签名可能不正确")
            else:
                logger.error("代码中未找到 adjust_beta 函数")
        
        # 4. 检查是否有 numpy 导入（如果代码中使用了 np）
        if 'np.' in code and 'import numpy' not in code:
            code = 'import numpy as np\n' + code
            logger.info("自动添加 numpy 导入语句")
        
        # 5. 检查是否错误地定义了类
        if re.search(r'class\s+\w+.*:', code):
            logger.warning("检测到类定义，策略应该是函数而不是类")
        
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
