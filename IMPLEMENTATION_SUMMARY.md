# 智能调用千问API功能 - 实现总结

## 修改概述

本次修改成功实现了智能调用千问API的功能，当系统连续50次（可配置）未检测到有效更改时，自动触发对千问API的调用，以获取具有指导性的改进意见。

## 修改文件清单

### 1. config.yaml
**修改内容**：添加智能调用千问API的配置参数

```yaml
advisor:
  # 原有配置...
  
  # 新增：智能调用配置
  smart_call_enabled: true  # 是否启用智能调用模式
  no_improvement_threshold: 50  # 连续无改进的轮次阈值
  min_performance_change: 0.01  # 判断为"有效改进"的最小性能变化比例（1%）
  call_history_enabled: true  # 是否保存调用历史记录
  call_history_file: "advisor_call_history.json"  # 调用历史记录文件路径
```

### 2. main.py
**修改内容**：实现智能调用千问API的核心逻辑

#### 新增属性
- `consecutive_no_improvement`: 连续无有效改进的轮次
- `last_significant_improvement_iter`: 上次显著改进的轮次
- `advisor_call_history`: 千问API调用历史记录
- `smart_call_config`: 智能调用配置字典

#### 新增方法
- `_should_call_advisor(current_performance)`: 判断是否应该调用千问指导者
- `_record_advisor_call(iteration, reason, duration, guidance_summary)`: 记录千问API调用历史
- `_save_advisor_call_history()`: 保存千问API调用历史到文件

#### 修改逻辑
1. **初始化阶段**：
   - 加载智能调用配置
   - 初始化连续无改进计数器
   - 初始化调用历史记录

2. **性能评估阶段**：
   - 计算性能改进比例
   - 判断是否为显著改进（>= min_performance_change）
   - 更新连续无改进计数器

3. **千问API调用阶段**：
   - 使用`_should_call_advisor()`判断是否应该调用
   - 达到阈值时触发调用
   - 记录调用历史

4. **结果输出阶段**：
   - 在最终结果中包含调用历史
   - 输出调用统计信息

### 3. .gitignore
**修改内容**：排除千问API调用历史文件

```
# 新增排除项
advisor_reports/
advisor_call_history.json
```

### 4. config.yaml.example（新建）
**内容**：配置文件模板，包含智能调用配置参数

### 5. ADVISOR_SMART_CALL.md（新建）
**内容**：智能调用千问API功能的详细说明文档

包含以下章节：
- 功能概述
- 核心特性
- 工作流程
- 配置参数详解
- 调用历史记录格式
- 日志示例
- 优势与价值
- 使用建议
- 注意事项
- 未来扩展

### 6. ADVISOR_USAGE_EXAMPLE.md（新建）
**内容**：智能调用千问API功能的使用示例文档

包含以下章节：
- 快速开始
- 使用场景（4种典型场景）
- 日志示例
- 调用历史文件示例
- 监控和分析
- 参数调优建议
- 常见问题

### 7. verify_advisor_smart_call.py（新建）
**内容**：验证智能调用千问API功能的测试脚本

测试内容：
- 配置文件验证
- main.py修改验证
- .gitignore修改验证
- 文档文件验证
- config.yaml.example验证

## 核心功能实现

### 1. 连续无改进检测

```python
# 计算性能改进比例
if self.best_performance < float('inf') and self.best_performance > 0:
    improvement_ratio = (self.best_performance - avg_performance) / self.best_performance
else:
    improvement_ratio = 1.0  # 首次改进，视为100%改进

# 判断是否为显著改进
if improvement_ratio >= self.smart_call_config['min_performance_change']:
    self.consecutive_no_improvement = 0  # 重置计数器
    self.last_significant_improvement_iter = self.iteration
else:
    # 性能改进但不够显著，仍然计数
    pass

# 性能变差，增加计数器
self.consecutive_no_improvement += 1
```

### 2. 阈值触发机制

```python
def _should_call_advisor(self, current_performance: float) -> bool:
    """判断是否应该调用千问指导者"""
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
```

### 3. 调用历史记录

```python
def _record_advisor_call(self, iteration: int, reason: str, 
                        duration: float, guidance_summary: str):
    """记录千问API调用历史"""
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
```

### 4. 智能调用逻辑

```python
# 智能调用千问指导者分析
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
```

## 配置参数说明

### smart_call_enabled
- **类型**：布尔值
- **默认值**：true
- **作用**：是否启用智能调用模式
  - true：只在达到阈值时调用
  - false：每轮都调用（与原行为一致）

### no_improvement_threshold
- **类型**：整数
- **默认值**：50
- **作用**：连续无改进的轮次阈值
  - 当连续无改进轮次达到此值时，触发千问API调用
  - 可以根据实际情况调整，建议范围：20-100

### min_performance_change
- **类型**：浮点数
- **默认值**：0.01（1%）
- **作用**：判断为"有效改进"的最小性能变化比例
  - 性能改进比例 >= 此值时，视为有效改进，重置计数器
  - 性能改进比例 < 此值时，仍视为无效改进，计数器+1
  - 可以根据实际情况调整，建议范围：0.005-0.02（0.5%-2%）

### call_history_enabled
- **类型**：布尔值
- **默认值**：true
- **作用**：是否保存调用历史记录
  - true：保存调用历史到JSON文件
  - false：不保存到文件，但仍在内存中记录

### call_history_file
- **类型**：字符串
- **默认值**："advisor_call_history.json"
- **作用**：调用历史记录文件的路径
  - 相对于项目根目录
  - 文件格式为JSON

## 调用历史记录格式

```json
{
  "call_history": [
    {
      "iteration": 51,
      "timestamp": "2026-01-31 12:34:56",
      "reason": "连续无改进达到阈值(50轮)",
      "duration_seconds": 15.23,
      "consecutive_no_improvement": 50,
      "best_performance": 123.45,
      "guidance_summary": "## 整体诊断\n7个问题的整体收敛情况..."
    }
  ],
  "total_calls": 1,
  "last_updated": "2026-01-31 12:34:56"
}
```

## 日志输出示例

```
2026-01-31 12:34:56 - INFO - 发现新最佳策略，性能: 123.45 (改进 0.50%，未达到显著改进阈值)
2026-01-31 12:35:10 - INFO - 第 51 轮迭代完成，平均性能: 123.45
2026-01-31 12:35:10 - INFO - 连续无改进次数(50)达到阈值(50)，触发千问指导者分析
2026-01-31 12:35:10 - INFO - 触发千问指导者分析 (连续无改进: 50/50)
2026-01-31 12:35:11 - INFO - 调用千问API (尝试 1/3)...
2026-01-31 12:35:25 - INFO - 千问API调用成功
2026-01-31 12:35:25 - INFO - 千问指导者分析完成
2026-01-31 12:35:25 - INFO - 千问API调用历史已保存到: advisor_call_history.json
```

## 最终结果输出

系统会在最终结果中包含千问API调用统计：

```
==================================================
千问API调用统计
总调用次数: 2
连续无改进轮次: 25
总调用耗时: 30.45秒
平均调用耗时: 15.23秒
```

## 优势与价值

### 1. 节省API调用成本
- 只在真正需要时调用，避免不必要的API调用
- 显著降低API调用次数和成本

### 2. 提高系统效率
- 减少等待API响应的时间
- 加快整体迭代速度

### 3. 提供有针对性的建议
- 在性能停滞时提供专家级分析
- 帮助突破优化瓶颈

### 4. 完整的可追溯性
- 详细的调用历史记录
- 便于后续分析和优化

### 5. 灵活的配置
- 所有参数可配置
- 适应不同的应用场景

## 使用方法

### 1. 配置参数

在`config.yaml`中配置智能调用参数：

```yaml
advisor:
  smart_call_enabled: true
  no_improvement_threshold: 50
  min_performance_change: 0.01
  call_history_enabled: true
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

### 3. 查看调用历史

查看`advisor_call_history.json`文件，或使用Python脚本：

```python
import json
with open('advisor_call_history.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
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
- config.yaml.example是否包含智能调用配置

## 注意事项

1. **API密钥安全**：
   - 不要将真实的API密钥提交到版本控制系统
   - 使用config.yaml.example作为模板
   - 将config.yaml添加到.gitignore

2. **网络连接**：
   - 确保网络连接稳定
   - 系统已实现重试机制（最多3次）

3. **调用频率限制**：
   - 注意API的调用频率限制
   - 避免短时间内大量调用

4. **参数调优**：
   - 根据问题难度调整阈值
   - 根据API成本调整调用频率
   - 定期查看调用历史和效果

## 文档清单

1. **ADVISOR_SMART_CALL.md**：功能说明文档
   - 详细的功能介绍
   - 工作流程说明
   - 配置参数详解
   - 使用建议和注意事项

2. **ADVISOR_USAGE_EXAMPLE.md**：使用示例文档
   - 快速开始指南
   - 多种使用场景
   - 日志和文件示例
   - 常见问题解答

3. **verify_advisor_smart_call.py**：验证脚本
   - 自动验证所有修改
   - 检查配置和代码
   - 确保功能正常

## 总结

本次修改成功实现了智能调用千问API的功能，主要特点包括：

1. **智能触发机制**：基于连续无改进计数，在合适的时机触发调用
2. **灵活的参数配置**：所有关键参数都可通过配置文件调整
3. **完整的调用历史记录**：记录每次调用的详细信息，便于后续分析
4. **详细的日志输出**：提供清晰的调用过程和状态信息
5. **易于维护**：代码结构清晰，文档完善，便于后续扩展

该功能既节省了API调用成本，又提供了有针对性的改进建议，为进化调参过程提供了强有力的支持。
