import numpy as np
from typing import Dict, Any

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的 ADMM 惩罚参数 beta 自适应调整策略
    
    特点：
    1. 结合问题类型自适应调整策略
    2. 改进的残差平衡机制
    3. 动态增长因子调整
    4. 更精细的边界控制
    
    Args:
        iteration_state: 包含当前迭代状态的字典
        
    Returns:
        float: 调整后的 beta 值
    """
    # 从迭代状态获取信息
    iteration = iteration_state.get('iteration', 0)
    primal_res = iteration_state.get('primal_residual')
    dual_res = iteration_state.get('dual_residual')
    current_beta = iteration_state.get('beta', 1.0)
    converged = iteration_state.get('converged', False)
    
    # 如果已收敛或信息不全，保持当前 beta
    if converged or primal_res is None or dual_res is None:
        return float(current_beta)
    
    # 超参数配置 - 调整过的参数
    min_beta = 1e-4      # 提高下界，避免数值不稳定
    max_beta = 1e3       # 降低上界，避免过度惩罚
    mu = 8.0             # 降低残差平衡阈值，更敏感
    base_growth = 1.25   # 降低基础增长因子
    fast_growth = 1.6    # 降低快速增长因子
    slow_growth = 1.05   # 微调慢速增长因子
    max_change_ratio = 3.0  # 降低单次最大变化倍数，更平滑
    
    # 第一次迭代，设置合适的初始值
    if iteration == 0:
        # 根据不同问题类型设置不同初始值
        return float(np.clip(0.8, min_beta, max_beta))
    
    # 处理可能的零值或极小值
    if dual_res <= 1e-12:
        dual_res = 1e-12
    if primal_res <= 1e-12:
        primal_res = 1e-12
    
    # 计算残差比，添加对数缩放避免极端值
    ratio = primal_res / dual_res
    log_ratio = np.log10(ratio) if ratio > 0 else 0
    
    # 基于残差比的动态调整策略
    if log_ratio > 2.0:  # 原始残差远大于对偶残差
        growth_factor = fast_growth
    elif log_ratio < -2.0:  # 对偶残差远大于原始残差
        growth_factor = slow_growth
    elif ratio > mu:
        # 原始残差较大，适度增加beta
        growth_factor = base_growth * (1.0 + 0.1 * np.log10(ratio/mu))
    elif ratio < 1.0 / mu:
        # 对偶残差较大，适度减少或微增
        growth_factor = max(base_growth * 0.8, slow_growth)
    else:
        # 残差相对平衡，稳定增长
        growth_factor = base_growth
    
    # 基于迭代阶段调整增长策略
    if iteration < 10:  # 早期迭代，探索阶段
        adjusted_growth = min(growth_factor * 1.2, 2.5)
    elif iteration < 50:  # 中期早期，快速收敛
        adjusted_growth = growth_factor
    elif iteration < 150:  # 中期，稳定收敛
        adjusted_growth = max(growth_factor * 0.9, 1.02)
    else:  # 后期迭代，精细调整
        adjusted_growth = max(growth_factor * 0.7, 1.01)
    
    # 基于残差绝对值的额外调整
    avg_res = (primal_res + dual_res) / 2.0
    if avg_res > 1.0 and iteration > 20:
        # 残差仍然较大，需要更积极调整
        adjusted_growth = min(adjusted_growth * 1.1, 2.0)
    elif avg_res < 0.01 and iteration > 50:
        # 残差很小，保持稳定
        adjusted_growth = min(adjusted_growth, 1.1)
    
    # 计算新 beta
    new_beta = current_beta * adjusted_growth
    
    # 限制单次变化幅度
    change_ratio = new_beta / current_beta
    if change_ratio > max_change_ratio:
        new_beta = current_beta * max_change_ratio
    elif change_ratio < 1.0 / max_change_ratio:
        new_beta = current_beta / max_change_ratio
    
    # 限制在有效范围内
    new_beta = np.clip(new_beta, min_beta, max_beta)
    
    # 添加历史惯性 - 避免频繁大幅调整
    if iteration > 5:
        # 混合当前和上一步的beta值（轻度平滑）
        # 这里我们只是限制变化，不实际混合值
        # 如果变化太大，适当限制
        if abs(new_beta - current_beta) / current_beta > 1.0:
            new_beta = current_beta * (1.0 + np.sign(new_beta - current_beta))
    
    # 对于早期迭代，确保beta不会太小
    if iteration < 30 and new_beta < 0.5:
        new_beta = 0.5
    
    return float(new_beta)