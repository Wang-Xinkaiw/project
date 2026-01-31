# 进化自适应调参策略框架
凯凯凯凯凯凯
## 🎯 项目概述

本框架实现了一个基于"生成-验证-反馈"循环的进化自适应调参策略系统。通过DeepSeek Coder生成初始调参策略，在真实优化问题上验证其性能，根据结果反馈优化策略，并迭代多轮以获得最优调参策略。

## ✨ 核心特性

- **自动化迭代**：完整的生成-验证-反馈循环
- **多算法支持**：可扩展支持ADMM、梯度下降等多种优化算法
- **真实评估**：在真实或仿真问题上进行策略评估
- **智能反馈**：基于性能数据的自动化分析与建议生成
- **可扩展架构**：模块化设计，易于添加新算法和测试问题

## 📁 文件结构
evolutionary-tuning-framework/
├── config.yaml # 配置文件
├── main.py # 主控制循环
├── strategy_generator.py # 策略生成器模块
├── evaluator.py # 策略评估器模块
├── feedback_loop.py # 反馈循环模块
├── problems/ # 测试问题目录
│ ├── init.py
│ ├── admm_problems.py # ADMM测试问题
│ ├── gradient_descent_problems.py # 梯度下降测试问题
│ └── base_problem.py # 问题基类
├── strategies/ # 策略存储目录
│ ├── init.py
│ ├── base_strategy.py # 策略基类
│ └── (自动生成的策略文件)
├── logs/ # 日志目录
├── results/ # 结果存储目录
├── feedback/ # 反馈信息目录
├── requirements.txt # 依赖包列表
└── README.md # 项目说明文档

text

## 🚀 快速开始

### 环境配置

1. **安装Python依赖**
   ```bash
   pip install -r requirements.txt
配置API密钥
编辑 config.yaml 文件：

yaml
api:
  api_key: "your_deepseek_api_key_here"  # 替换为你的API密钥
准备测试问题

在 problems/ 目录中添加你的测试问题

或使用现有的ADMM测试问题

运行框架
直接运行

bash
python main.py
自定义配置运行

python
# 在main.py中修改
main = EvolutionaryTuningMain("custom_config.yaml")
main.run()
🔧 配置说明
API配置
yaml
api:
  base_url: "https://api.deepseek.com"
  api_key: "your_api_key"  # TODO: 替换为你的API密钥
  model: "deepseek-coder"   # 使用的模型
算法配置
yaml
algorithms:
  admm:
    base_class: "ADMMAlgorithm"
    problem_module: "problems.admm_problems"
    default_beta: 1.0
    
  gradient_descent:
    base_class: "GradientDescentAlgorithm"
    problem_module: "problems.gradient_descent_problems"
    default_lr: 0.01
终止条件
yaml
termination:
  max_iterations: 50          # 最大迭代轮次
  performance_threshold: 0.2  # 性能提升阈值(20%)
  patience: 10                # 连续无改进容忍轮次
🔄 核心流程
迭代循环
text
初始化配置
    ↓
生成初始策略 (DeepSeek Coder)
    ↓
在测试问题上评估策略
    ↓
分析结果，生成反馈
    ↓
基于反馈生成改进策略
    ↓
重复评估与改进
    ↓
达到终止条件 → 输出最佳策略
策略接口
所有策略必须继承 BaseTuningStrategy 并实现以下方法：

python
class BaseTuningStrategy:
    def update_parameters(self, iteration_state: dict) -> dict:
        """
        根据迭代状态更新参数
        
        Args:
            iteration_state: 包含算法当前状态的字典
                - iteration: 当前迭代次数
                - 算法特定状态（如残差、梯度等）
                
        Returns:
            更新后的参数字典
        """
        pass
📈 扩展指南
添加新算法
创建算法实现

python
# problems/new_algorithm_problems.py
class NewAlgorithmProblem:
    def solve(self, strategy, max_iterations=1000):
        # 实现算法求解逻辑
        pass
更新配置文件

yaml
algorithms:
  new_algorithm:
    base_class: "NewAlgorithmProblem"
    problem_module: "problems.new_algorithm_problems"
添加测试问题

python
# 在对应的问题模块中添加问题函数
def test_problem_1():
    return NewAlgorithmProblem(...)
添加新测试问题
在对应算法的问题模块中添加

python
# problems/admm_problems.py
def new_admm_problem():
    # 返回问题实例
    return ADMMProblem(...)
更新配置文件

yaml
problems:
  admm:
    - "l1_regularization"
    - "elastic_net"
    - "new_admm_problem"  # 添加新问题
🐛 故障排除
常见问题
API调用失败

检查API密钥配置

验证网络连接

检查API配额和限制

策略生成失败

检查提示词格式

验证代码提取逻辑

查看日志中的错误信息

评估失败

检查测试问题实现

验证策略类接口

查看具体错误堆栈

调试建议
启用详细日志

python
logging.basicConfig(level=logging.DEBUG)
逐步测试

单独测试策略生成

单独测试问题求解

逐步增加迭代轮次

检查中间结果

查看生成的策略代码

检查每轮的评估结果

分析反馈信息

📊 输出结果
生成文件
strategies/: 每轮生成的策略代码

results/: 最终结果和性能数据

feedback/: 每轮的反馈分析

logs/: 运行日志

结果分析
框架运行结束后会生成：

最佳策略的代码文件

性能对比报告

收敛曲线图（如配置）

详细的迭代历史

🤝 贡献指南
Fork项目

创建功能分支

提交更改

推送到分支

创建Pull Request

📄 许可证
本项目采用MIT许可证。详见LICENSE文件。

🙏 致谢
本框架基于以下研究工作：

Li et al. "Optimization Problem Solving Can Transition to Evolutionary Agentic Workflows"

相关优化算法文献

DeepSeek Coder模型