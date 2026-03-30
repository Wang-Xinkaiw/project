import numpy as np
from typing import Dict, Any

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的 ADMM 惩罚参数 beta 自适应调整策略
    
    特点：
    1. 简化调整逻辑，提高稳定性
    2. 针对不同问题阶段采用不同策略
    3. 更平滑的beta变化控制
    4. 修复了可能导致运行时错误的变量访问问题
    
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
    
    # 超参数配置 - 简化并调整
    min_beta = 1e-3      # 提高下界，避免数值不稳定
    max_beta = 1e4       # 恢复较高的上界，适应更多问题
    mu = 5.0             # 残差平衡阈值
    tau_inc = 1.3        # beta 增大因子
    tau_dec = 1.2        # beta 减小因子
    stability_threshold = 30  # 稳定阶段开始的迭代次数
    
    # 首次迭代，设置合适的初始值
    if iteration == 0:
        # 使用适中的初始beta值
        return float(np.clip(0.5, min_beta, max_beta))
    
    # 处理可能的零值或极小值
    if dual_res <= 1e-12:
        dual_res = 1e-12
    if primal_res <= 1e-12:
        primal_res = 1e-12
    
    # 计算残差比例
    ratio = primal_res / dual_res if dual_res > 0 else 1.0
    
    # 基于迭代阶段和残差比例的策略
    if iteration < 10:
        # 早期探索阶段：缓慢调整，寻找方向
        if ratio > 50:
            new_beta = current_beta * 1.5
        elif ratio < 0.02:
            new_beta = current_beta / 1.2
        elif ratio > mu:
            new_beta = current_beta * 1.2
        elif ratio < 1.0 / mu:
            new_beta = current_beta / 1.1
        else:
            new_beta = current_beta * 1.05  # 缓慢增长
        
    elif iteration < 50:
        # 中期快速收敛阶段：基于残差比例调整
        if ratio > mu:
            new_beta = current_beta * tau_inc
        elif ratio < 1.0 / mu:
            new_beta = current_beta / tau_dec
        else:
            # 残差平衡，适度增长
            new_beta = current_beta * 1.1
    
    elif iteration < 150:
        # 中期稳定阶段：更保守的调整
        if ratio > 2.0 * mu:
            new_beta = current_beta * 1.2
        elif ratio < 0.5 / mu:
            new_beta = current_beta / 1.1
        elif ratio > mu:
            new_beta = current_beta * 1.05
        elif ratio < 1.0 / mu:
            new_beta = current_beta / 1.05
        else:
            new_beta = current_beta  # 保持稳定
    
    else:
        # 后期精细调整阶段：非常保守
        if ratio > 3.0 * mu:
            new_beta = current_beta * 1.1
        elif ratio < 0.33 / mu:
            new_beta = current_beta / 1.05
        else:
            new_beta = current_beta  # 基本保持稳定
    
    # 基于绝对残差大小的额外调整
    avg_res = (primal_res + dual_res) / 2.0
    
    if iteration > 20 and avg_res > 10.0:
        # 残差仍然很大，需要更积极的调整
        new_beta = min(new_beta * 1.5, current_beta * 2.0)
    
    elif iteration > 50 and avg_res < 0.001:
        # 残差很小，进入精细调整阶段
        new_beta = current_beta  # 保持稳定
    
    # 限制单次变化幅度（不超过2倍）
    max_change = 2.0
    min_change = 1.0 / max_change
    change_ratio = new_beta / current_beta
    
    if change_ratio > max_change:
        new_beta = current_beta * max_change
    elif change_ratio < min_change:
        new_beta = current_beta * min_change
    
    # 确保beta在有效范围内
    new_beta = np.clip(new_beta, min_beta, max_beta)
    
    # 对于早期迭代，确保beta不会太小
    if iteration < 20 and new_beta < 0.1:
        new_beta = max(new_beta, 0.1)
    
    # 对于回归类问题，beta应单调递增
    # 注意：这里我们不直接判断问题类型，而是通过迭代行为推断
    # 如果迭代次数较多且残差下降缓慢，倾向于保持增长
    if iteration > 100 and avg_res > 0.1:
        # 确保不减少
        new_beta = max(new_beta, current_beta)
    
    return float(new_beta)