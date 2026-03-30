import numpy as np
from typing import Dict, Any

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    自适应调整 ADMM 惩罚参数 beta 的策略函数
    
    特点：
    1. 针对含噪声回归问题采用单调递增策略
    2. 结合残差平衡机制进行适度调整
    3. 添加安全边界和平滑限制
    4. 基于迭代阶段自适应调整增长率
    
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
    
    # 超参数配置
    min_beta = 1e-6      # beta 下界
    max_beta = 1e4       # beta 上界（针对含噪声回归问题）
    mu = 10.0            # 残差平衡阈值
    base_growth = 1.3    # 基础增长因子
    fast_growth = 1.8    # 快速增长因子（原始残差过大时）
    slow_growth = 1.1    # 慢速增长因子（对偶残差过大时）
    max_change_ratio = 5.0  # 单次最大变化倍数
    
    # 第一次迭代，使用较小初始值
    if iteration == 0:
        return float(min(max(0.5, min_beta), max_beta))
    
    # 处理可能的零值
    if dual_res <= 1e-12:
        dual_res = 1e-12
    
    # 计算残差比
    ratio = primal_res / dual_res
    
    # 基于残差比选择增长因子
    if ratio > mu * 5:  # 原始残差远大于对偶残差
        growth_factor = fast_growth
    elif ratio < 1.0 / (mu * 5):  # 对偶残差远大于原始残差
        growth_factor = slow_growth
    elif ratio > mu:  # 原始残差较大
        growth_factor = min(base_growth * 1.2, fast_growth)
    elif ratio < 1.0 / mu:  # 对偶残差较大
        growth_factor = max(base_growth * 0.8, slow_growth)
    else:  # 残差相对平衡
        growth_factor = base_growth
    
    # 基于迭代阶段调整增长策略
    if iteration < 20:  # 早期迭代，快速逼近
        # 稍微加快增长，但限制单次最大变化
        adjusted_growth = min(growth_factor * 1.1, 2.0)
    elif iteration > 100:  # 后期迭代，精细调整
        # 减慢增长，避免振荡
        adjusted_growth = max(growth_factor * 0.7, 1.05)
    else:  # 中期迭代
        adjusted_growth = growth_factor
    
    # 计算新 beta（单调递增策略）
    new_beta = current_beta * adjusted_growth
    
    # 限制单次变化幅度
    change_ratio = new_beta / current_beta
    if change_ratio > max_change_ratio:
        new_beta = current_beta * max_change_ratio
    elif change_ratio < 1.0 / max_change_ratio:
        new_beta = current_beta / max_change_ratio
    
    # 限制在有效范围内
    new_beta = np.clip(new_beta, min_beta, max_beta)
    
    # 确保 beta 不会太小（对于含噪声问题特别重要）
    if iteration > 10 and new_beta < 0.1:
        new_beta = 0.1
    
    return float(new_beta)