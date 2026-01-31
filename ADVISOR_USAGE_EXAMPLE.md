# 智能调用千问API功能 - 使用示例

## 快速开始

### 1. 配置文件设置

在`config.yaml`中配置千问API和智能调用参数：

```yaml
# 千问指导者配置
advisor:
  enabled: true  # 启用指导者
  api_key: "sk-your-qwen-api-key"  # 替换为你的API密钥
  model: "qwen3-235b-a22b"
  
  # 智能调用配置
  smart_call_enabled: true  # 启用智能调用模式
  no_improvement_threshold: 50  # 连续50轮无改进时触发
  min_performance_change: 0.01  # 改进超过1%才视为有效改进
  call_history_enabled: true  # 保存调用历史
  call_history_file: "advisor_call_history.json"
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

## 使用场景

### 场景1: 默认配置（推荐）

适用于大多数情况，平衡了API调用成本和优化效果。

```yaml
advisor:
  smart_call_enabled: true
  no_improvement_threshold: 50
  min_performance_change: 0.01
```

**特点**：
- 每50轮无改进时调用一次
- 改进超过1%才视为有效
- 适中的API调用频率

### 场景2: 快速探索模式

适用于初期探索阶段，需要更频繁地获取专家建议。

```yaml
advisor:
  smart_call_enabled: true
  no_improvement_threshold: 20  # 降低阈值
  min_performance_change: 0.005  # 降低改进阈值
```

**特点**：
- 每20轮无改进时调用一次
- 改进超过0.5%就视为有效
- 更频繁的API调用
- 更快获得专家建议

### 场景3: 节省成本模式

适用于后期优化阶段，性能提升空间有限，希望减少API调用。

```yaml
advisor:
  smart_call_enabled: true
  no_improvement_threshold: 100  # 提高阈值
  min_performance_change: 0.02  # 提高改进阈值
```

**特点**：
- 每100轮无改进时调用一次
- 改进超过2%才视为有效
- 显著减少API调用
- 节省API成本

### 场景4: 传统模式（每轮都调用）

适用于需要每轮都获取专家建议的情况。

```yaml
advisor:
  smart_call_enabled: false  # 禁用智能调用
```

**特点**：
- 每轮都调用千问API
- 与原行为一致
- API调用成本较高
- 获得最频繁的专家建议

## 日志示例

### 正常运行日志

```
2026-01-31 12:34:56 - INFO - 启动进化自适应调参框架（strict模式）
2026-01-31 12:34:56 - INFO - API密钥配置正确
2026-01-31 12:34:56 - INFO - 使用API配置: model=deepseek-chat, temperature=0.0
2026-01-31 12:34:56 - INFO - 开始第 1 轮迭代
2026-01-31 12:35:10 - INFO - 第 1 轮迭代完成，平均性能: 150.23
2026-01-31 12:35:10 - INFO - 发现新最佳策略，性能: 150.23 (改进 100.00%)
...
2026-01-31 12:45:30 - INFO - 第 30 轮迭代完成，平均性能: 125.45
2026-01-31 12:45:30 - INFO - 发现新最佳策略，性能: 125.45 (改进 0.50%，未达到显著改进阈值)
2026-01-31 12:45:30 - DEBUG - 跳过千问指导者分析 (连续无改进: 30/50)
...
2026-01-31 12:55:45 - INFO - 第 51 轮迭代完成，平均性能: 125.40
2026-01-31 12:55:45 - INFO - 连续无改进次数(50)达到阈值(50)，触发千问指导者分析
2026-01-31 12:55:45 - INFO - 触发千问指导者分析 (连续无改进: 50/50)
2026-01-31 12:55:45 - INFO - 调用千问API (尝试 1/3)...
2026-01-31 12:56:00 - INFO - 千问API调用成功
2026-01-31 12:56:00 - INFO - 千问指导者分析完成
2026-01-31 12:56:00 - INFO - 千问API调用历史已保存到: advisor_call_history.json
```

### 最终统计日志

```
==================================================
进化调参完成
总迭代轮次: 100
最佳策略路径: strategies/strategy_iter_75.py
最佳性能 (平均迭代次数): 120.35
结果已保存到: results/final_results_20260131_125600.json
==================================================
千问API调用统计
总调用次数: 2
连续无改进轮次: 25
总调用耗时: 30.45秒
平均调用耗时: 15.23秒
```

## 调用历史文件示例

`advisor_call_history.json`文件内容：

```json
{
  "call_history": [
    {
      "iteration": 51,
      "timestamp": "2026-01-31 12:55:45",
      "reason": "连续无改进达到阈值(50轮)",
      "duration_seconds": 15.23,
      "consecutive_no_improvement": 50,
      "best_performance": 125.40,
      "guidance_summary": "## 整体诊断\n7个问题的整体收敛情况..."
    },
    {
      "iteration": 78,
      "timestamp": "2026-01-31 13:10:20",
      "reason": "连续无改进达到阈值(50轮)",
      "duration_seconds": 15.22,
      "consecutive_no_improvement": 50,
      "best_performance": 122.80,
      "guidance_summary": "## 整体诊断\n7个问题的整体收敛情况..."
    }
  ],
  "total_calls": 2,
  "last_updated": "2026-01-31 13:10:20"
}
```

## 监控和分析

### 1. 实时监控

查看日志文件，关注以下信息：
- 连续无改进计数器状态
- 千问API调用触发时机
- 调用耗时和结果

```bash
tail -f logs/tuning_*.log
```

### 2. 调用历史分析

使用Python脚本分析调用历史：

```python
import json

# 读取调用历史
with open('advisor_call_history.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计信息
total_calls = data['total_calls']
total_duration = sum(call['duration_seconds'] for call in data['call_history'])
avg_duration = total_duration / total_calls if total_calls > 0 else 0

print(f"总调用次数: {total_calls}")
print(f"总耗时: {total_duration:.2f}秒")
print(f"平均耗时: {avg_duration:.2f}秒")

# 查看每次调用的详细信息
for call in data['call_history']:
    print(f"\n迭代{call['iteration']}: {call['reason']}")
    print(f"  耗时: {call['duration_seconds']}秒")
    print(f"  最佳性能: {call['best_performance']}")
```

### 3. 性能趋势分析

结合调用历史和最终结果，分析千问建议的效果：

```python
import json

# 读取最终结果
with open('results/final_results_*.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 读取调用历史
with open('advisor_call_history.json', 'r', encoding='utf-8') as f:
    call_history = json.load(f)

# 分析千问建议前后的性能变化
for call in call_history['call_history']:
    iter_before = call['iteration']
    perf_before = call['best_performance']
    
    # 查找调用后的最佳性能
    perf_after = perf_before
    for record in results['history']:
        if record['iteration'] > iter_before:
            perf_after = min(perf_after, record['performance'])
    
    improvement = (perf_before - perf_after) / perf_before * 100 if perf_before > 0 else 0
    print(f"迭代{call['iteration']}: 改进 {improvement:.2f}%")
```

## 参数调优建议

### 1. 根据问题难度调整

**简单问题**（快速收敛）：
```yaml
no_improvement_threshold: 30  # 降低阈值
min_performance_change: 0.005  # 降低改进阈值
```

**复杂问题**（慢速收敛）：
```yaml
no_improvement_threshold: 70  # 提高阈值
min_performance_change: 0.02  # 提高改进阈值
```

### 2. 根据API成本调整

**预算充足**：
```yaml
no_improvement_threshold: 20  # 降低阈值，更频繁调用
```

**预算有限**：
```yaml
no_improvement_threshold: 100  # 提高阈值，减少调用
```

### 3. 根据优化阶段调整

**初期探索**：
```yaml
no_improvement_threshold: 20
min_performance_change: 0.005
```

**中期优化**：
```yaml
no_improvement_threshold: 50
min_performance_change: 0.01
```

**后期精调**：
```yaml
no_improvement_threshold: 100
min_performance_change: 0.02
```

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

### Q3: 如何查看调用历史？

查看`advisor_call_history.json`文件，或使用Python脚本：
```python
import json
with open('advisor_call_history.json', 'r') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
```

### Q4: 如何禁用智能调用，每轮都调用？

修改`config.yaml`：
```yaml
advisor:
  smart_call_enabled: false
```

### Q5: 如何验证功能是否正常工作？

运行测试脚本：
```bash
python test_advisor_smart_call.py
```

## 总结

智能调用千问API功能通过监控性能改进情况，在合适的时机触发专家级分析，既节省了API调用成本，又提供了有针对性的改进建议。通过灵活的参数配置，可以适应不同的应用场景和需求。

详细文档请参考：[ADVISOR_SMART_CALL.md](ADVISOR_SMART_CALL.md)
