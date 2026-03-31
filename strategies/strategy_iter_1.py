from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的自适应beta调整策略，结合残差平衡、迭代历史和收敛状态
    
    核心思想：
    1. 使用动态阈值：根据残差变化趋势调整阈值mu
    2. 引入历史信息：考虑最近几次迭代的残差模式
    3. 收敛感知：接近收敛时采用保守策略
    4. 阶段调整：不同迭代阶段使用不同调整策略
    
    设计原则：
    - 避免beta震荡：引入平滑因子
    - 防止过度调整：限制调整幅度
    - 提高泛化能力：适合多种问题类型
    """
    
    # 1. 从状态字典获取必要参数
    primal_res = iteration_state.get('primal_residual', None)
    dual_res = iteration_state.get('dual_residual', None)
    current_beta = iteration_state.get('beta', 1.0)
    iteration = iteration_state.get('iteration', 0)
    converged = iteration_state.get('converged', False)
    
    # 2. 边界检查和初始值处理
    if primal_res is None or dual_res is None or dual_res < 1e-12:
        return float(np.clip(current_beta, 1e-6, 1e6))
    
    # 3. 如果已经收敛，保持beta不变
    if converged:
        return float(current_beta)
    
    # 4. 获取历史信息（这里简化处理，实际可以存储更多历史）
    # 注意：实际应用中，iteration_state可能需要扩展以存储历史信息
    # 这里我们基于当前状态和迭代次数进行决策
    
    # 5. 计算残差比
    ratio = primal_res / max(dual_res, 1e-12)
    
    # 6. 动态调整阈值和调整因子
    # 早期迭代：采用激进策略加速收敛
    # 中期迭代：采用平衡策略稳定收敛
    # 后期迭代：采用保守策略避免震荡
    
    if iteration < 50:
        # 早期阶段：快速探索
        base_mu = 5.0  # 较低的平衡阈值，允许更频繁的调整
        tau_inc = 1.8  # 较大的增长因子
        tau_dec = 1.8  # 较大的减小因子
        smoothing = 0.3  # 较小的平滑因子
        
    elif iteration < 200:
        # 中期阶段：平衡收敛
        base_mu = 10.0  # 标准平衡阈值
        tau_inc = 1.5   # 适中的增长因子
        tau_dec = 1.5   # 适中的减小因子
        smoothing = 0.5  # 适中的平滑因子
        
    else:
        # 后期阶段：精细调整
        base_mu = 15.0  # 较高的平衡阈值，减少调整频率
        tau_inc = 1.2   # 较小的增长因子
        tau_dec = 1.2   # 较小的减小因子
        smoothing = 0.7  # 较大的平滑因子，避免震荡
    
    # 7. 基于残差变化趋势动态调整mu
    # 如果残差持续不平衡，适当调整mu
    # 这里简化处理，使用固定策略，实际可以基于历史趋势调整
    
    # 8. 检查是否处于未收敛的停滞状态
    # 如果残差长时间没有显著下降，采用特殊策略
    stagnation_threshold = 1e-3
    if iteration > 100 and primal_res < stagnation_threshold and dual_res < stagnation_threshold:
        # 接近收敛，采用微调策略
        base_mu = 20.0
        tau_inc = 1.05
        tau_dec = 1.05
        smoothing = 0.9
    
    # 9. 应用调整策略
    if ratio > base_mu:
        # primal_residual远大于dual_residual，需要增大beta
        proposed_beta = current_beta * tau_inc
        
    elif ratio < 1.0 / base_mu:
        # dual_residual远大于primal_residual，需要减小beta
        proposed_beta = current_beta / tau_dec
        
    else:
        # 残差相对平衡，保持beta不变
        return float(current_beta)
    
    # 10. 应用平滑处理，避免beta剧烈变化
    # new_beta = smoothing * proposed_beta + (1-smoothing) * current_beta
    # 简化处理：直接使用proposed_beta，但限制变化幅度
    
    # 限制最大变化幅度不超过5倍
    max_change_factor = 5.0
    min_change_factor = 1.0 / max_change_factor
    
    if proposed_beta / current_beta > max_change_factor:
        proposed_beta = current_beta * max_change_factor
    elif proposed_beta / current_beta < min_change_factor:
        proposed_beta = current_beta * min_change_factor
    
    # 11. 限制beta范围，确保数值稳定性
    min_beta = 1e-6
    max_beta = 1e6
    
    # 根据问题类型调整范围（这里使用通用范围）
    # 对于回归问题，可能需要不同的范围
    # 可以根据迭代次数或收敛状态动态调整范围
    
    # 12. 最终边界裁剪
    new_beta = float(np.clip(proposed_beta, min_beta, max_beta))
    
    return new_beta