# 智能调用千问API功能 - 快速开始

## 功能简介

本系统实现了智能调用千问API的功能，当系统连续50次（可配置）未检测到有效更改时，自动触发对千问API的调用，以获取具有指导性的改进意见。

## 快速配置

### 1. 配置千问API密钥

编辑`config.yaml`文件，填入你的千问API密钥：

```yaml
advisor:
  enabled: true  # 启用指导者
  api_key: "sk-your-qwen-api-key"  # 替换为你的API密钥
  model: "qwen3-235b-a22b"
  
  # 智能调用配置
  smart_call_enabled: true  # 启用智能调用模式
  no_improvement_threshold: 50  # 连续50轮无改进时触发
  min_performance_change: 0.01  # 改进超过1%才视为有效改进
  call_history_enabled: true  # 保存调用历史
```

### 2. 运行系统

```bash
python main.py
```

系统会自动：
- 监控每轮迭代的性能改进情况
- 当连续50轮无有效改进时，自动调用千问API
- 保存调用历史到`advisor_call_history.json`
- 在日志中输出详细的调用信息

## 参数说明

### no_improvement_threshold（无改进阈值）
- **默认值**：50
- **说明**：连续多少轮无有效改进时触发千问API调用
- **建议**：
  - 初期探索：设置为20-30
  - 正常使用：设置为50
  - 后期优化：设置为70-100

### min_performance_change（最小性能改进）
- **默认值**：0.01（1%）
- **说明**：性能改进比例超过此值才被视为"有效改进"
- **建议**：
  - 简单问题：设置为0.005（0.5%）
  - 正常使用：设置为0.01（1%）
  - 复杂问题：设置为0.02（2%）

### smart_call_enabled（智能调用开关）
- **默认值**：true
- **说明**：是否启用智能调用模式
- **选项**：
  - true：只在达到阈值时调用（推荐）
  - false：每轮都调用（API成本高）

## 使用场景

### 场景1：默认配置（推荐）
适用于大多数情况，平衡了API调用成本和优化效果。

```yaml
no_improvement_threshold: 50
min_performance_change: 0.01
```

### 场景2：快速探索
适用于初期探索阶段，需要更频繁地获取专家建议。

```yaml
no_improvement_threshold: 20
min_performance_change: 0.005
```

### 场景3：节省成本
适用于后期优化阶段，希望减少API调用。

```yaml
no_improvement_threshold: 100
min_performance_change: 0.02
```

## 查看调用历史

### 方法1：查看JSON文件

```bash
cat advisor_call_history.json
```

### 方法2：使用Python脚本

```python
import json

with open('advisor_call_history.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"总调用次数: {data['total_calls']}")
print(f"最后更新: {data['last_updated']}")

for call in data['call_history']:
    print(f"\n迭代{call['iteration']}: {call['reason']}")
    print(f"  耗时: {call['duration_seconds']}秒")
    print(f"  最佳性能: {call['best_performance']}")
```

## 监控运行状态

### 实时查看日志

```bash
tail -f logs/tuning_*.log
```

关注以下信息：
- 连续无改进计数器状态
- 千问API调用触发时机
- 调用耗时和结果

### 关键日志信息

```
# 性能改进检测
发现新最佳策略，性能: 125.45 (改进 0.50%，未达到显著改进阈值)

# 触发千问API调用
连续无改进次数(50)达到阈值(50)，触发千问指导者分析
触发千问指导者分析 (连续无改进: 50/50)

# 调用完成
千问指导者分析完成
千问API调用历史已保存到: advisor_call_history.json
```

## 验证功能

运行验证脚本：

```bash
python verify_advisor_smart_call.py
```

该脚本会验证：
- 配置文件是否包含所有必需参数
- main.py是否包含所有必需的修改
- .gitignore是否正确排除调用历史文件
- 文档文件是否存在且包含必要内容

## 常见问题

### Q1: 如何知道千问API是否被调用？

查看日志，搜索"千问指导者分析"：

```bash
grep "千问指导者分析" logs/tuning_*.log
```

### Q2: 如何调整触发阈值？

修改`config.yaml`中的`no_improvement_threshold`参数：

```yaml
advisor:
  no_improvement_threshold: 30  # 改为30轮
```

### Q3: 如何禁用智能调用，每轮都调用？

修改`config.yaml`：

```yaml
advisor:
  smart_call_enabled: false
```

### Q4: 调用历史保存在哪里？

调用历史保存在`advisor_call_history.json`文件中，包含每次调用的详细信息。

### Q5: 如何查看千问API调用的效果？

1. 查看调用历史文件，了解调用时机和上下文
2. 查看advisor报告目录，了解千问给出的建议
3. 对比调用前后的性能变化，评估建议效果

## 详细文档

- **功能说明**：[ADVISOR_SMART_CALL.md](ADVISOR_SMART_CALL.md)
- **使用示例**：[ADVISOR_USAGE_EXAMPLE.md](ADVISOR_USAGE_EXAMPLE.md)
- **实现总结**：[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 总结

智能调用千问API功能通过监控性能改进情况，在合适的时机触发专家级分析，既节省了API调用成本，又提供了有针对性的改进建议。

**主要优势**：
- 节省API调用成本
- 提高系统效率
- 提供有针对性的建议
- 完整的可追溯性
- 灵活的参数配置

**使用建议**：
1. 根据问题难度调整阈值
2. 根据API成本调整调用频率
3. 定期查看调用历史和效果
4. 结合其他机制（如早停）使用

开始使用：`python main.py`
