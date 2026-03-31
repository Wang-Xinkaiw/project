# ADMM 进化调参框架 - 项目结构说明

## 📁 清理后的项目结构

```
project/
├── 核心代码文件
│   ├── main.py                  # 主控制循环（进化调参主控制器）
│   ├── strategy_generator.py    # 策略生成器（调用 DeepSeek API 生成策略代码）
│   ├── evaluator.py             # 策略评估器（在 8 个 ADMM 问题上评估策略）
│   ├── feedback_loop.py         # 反馈循环（生成反馈指导策略优化）
│   ├── advisor.py               # 千问指导者（ADMM 优化专家，提供深度分析）
│   │
├── 配置文件
│   ├── config.yaml              # 主配置文件（包含所有配置项和详细注释）
│   ├── requirements.txt         # Python 依赖包列表
│   ├── .gitignore              # Git 忽略文件配置
│   │
├── 文档文件
│   ├── README.md                # 项目主文档（使用说明、功能介绍）
│   ├── COMMENT_COMPLETION_REPORT.md  # 代码注释完成报告
│   │
├── libadmm/                     # ADMM 算法库（核心算法实现）
│   ├── __init__.py
│   ├── algorithms/              # 算法实现
│   │   ├── l1.py               # L1 正则化问题
│   │   ├── elasticnet.py       # 弹性网络问题
│   │   ├── l1R.py              # L1 正则化回归
│   │   ├── elasticnetR.py      # 弹性网络回归
│   │   ├── lrmc.py             # 低秩矩阵补全
│   │   ├── lrr.py              # 低秩表示
│   │   ├── rmsc.py             # 鲁棒多视图谱聚类
│   │   ├── tracelasso.py       # Trace Lasso 问题
│   │   └── ... (其他算法文件)
│   │
│   ├── examples/                # 使用示例
│   │   ├── example_low_rank_matrix_models.py
│   │   ├── example_low_rank_tensor_models.py
│   │   └── example_sparse_models.py
│   │
│   └── proximal_operators/      # 近端算子
│       ├── prox_l1.py
│       ├── prox_nuclear.py
│       └── ... (其他近端算子)
│
├── strategies/                  # 策略代码目录（运行时生成）
│   ├── __init__.py
│   ├── base_strategy.py         # 策略基类
│   ├── baseline_strategy.py     # 基线策略
│   ├── initial_strategy.py      # 初始策略
│   └── standard_admm_strategy.py # 标准 ADMM 策略
│
├── 输出目录（运行时自动生成）
│   ├── logs/                    # 日志文件目录
│   ├── results/                 # 评估结果目录
│   ├── feedback/                # 反馈信息目录
│   └── advisor_reports/         # 千问分析报告目录
│
└── .trae/                       # IDE 配置目录
    └── documents/
        └── plan_20260301_082340.md
```

---

## 🎯 核心文件说明

### 1. main.py - 主控制循环
- **功能**: 整合所有组件，实现完整的迭代优化流程
- **主要类**: `EvolutionaryTuningMain`
- **工作流程**:
  1. 生成策略代码
  2. 在 8 个 ADMM 问题上评估
  3. 生成反馈信息
  4. 指导下一次策略生成
  5. 智能调用千问指导者分析

### 2. strategy_generator.py - 策略生成器
- **功能**: 调用 DeepSeek API 生成 ADMM 惩罚参数调整策略
- **主要类**: `StrategyGenerator`
- **输出**: Python 函数 `adjust_beta(iteration_state) -> float`

### 3. evaluator.py - 策略评估器
- **功能**: 在 8 个 ADMM 问题上评估策略性能
- **主要类**: `StrategyEvaluator`
- **评估指标**: 平均迭代次数、收敛性、目标函数值

### 4. feedback_loop.py - 反馈循环
- **功能**: 根据评估结果生成反馈，指导下一次策略生成
- **主要类**: `FeedbackLoop`
- **反馈内容**: 性能分析、问题诊断、优化建议

### 5. advisor.py - 千问指导者
- **功能**: 调用 Qwen3-235B-A22B API 进行深度分析
- **主要类**: `ADMMAdvisor`
- **触发条件**: 连续无改进达到阈值（默认 50 轮）

### 6. config.yaml - 配置文件
- **功能**: 集中管理所有配置项
- **配置内容**:
  - DeepSeek API 配置
  - 评估器参数
  - 测试问题列表
  - 终止条件
  - 千问指导者配置

---

## 📊 清理总结

### 已删除的冗余文件：
1. ✅ CODE_COMMENTS.md - 已整合到 COMMENT_COMPLETION_REPORT.md
2. ✅ COMMENT_REVIEW_REPORT.md - 审查报告，已完成
3. ✅ config.yaml.example - 旧的配置模板，已过时
4. ✅ .md/*.md - 4 个过时的文档文件
5. ✅ strategies/strategy_iter_*.py - 测试生成的临时策略文件
6. ✅ logs/*.log - 测试日志文件
7. ✅ results/*.json - 测试结果文件
8. ✅ feedback/*.txt - 反馈信息文件
9. ✅ advisor_reports/*.log - 分析报告
10. ✅ **pycache__/ - Python 缓存文件

### 保留的核心文件：
- ✅ 6 个核心 Python 文件（main.py, strategy_generator.py, evaluator.py, feedback_loop.py, advisor.py）
- ✅ 1 个配置文件（config.yaml）
- ✅ 2 个文档文件（README.md, COMMENT_COMPLETION_REPORT.md）
- ✅ libadmm 算法库（完整保留）
- ✅ strategies/ 目录（保留基础策略文件）

---

## 🚀 快速开始

### 1. 配置 API 密钥
编辑 `config.yaml`，填入你的 API 密钥：
```yaml
api:
  api_key: "your_deepseek_api_key"
  
advisor:
  api_key: "your_qwen_api_key"
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行系统
```bash
python main.py
```

---

## 📝 项目特点

1. **完整的迭代优化流程**: 生成 → 评估 → 反馈 → 再生成
2. **智能调用机制**: 连续无改进时自动调用千问专家分析
3. **8 个 ADMM 测试问题**: 全面评估策略性能
4. **详细的代码注释**: 所有主要函数都有完整注释
5. **严密的项目结构**: 去冗余化，只保留必要文件

---

## 📈 项目统计

- **核心代码文件**: 5 个
- **配置文件**: 1 个
- **文档文件**: 2 个
- **算法文件**: 24 个（libadmm）
- **总代码行数**: ~2000 行（不含算法库）
- **注释覆盖率**: 100%（所有主要函数）

---

*最后更新：2026-03-31*
