from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的ADMM惩罚参数自适应调整策略
    
    核心改进：
    1. 针对elastic_net_regression回归问题，采用更强的单调递增策略
    2. 降低前期调整的激进程度，避免震荡
    3. 简化回归问题检测逻辑
    4. 提高对低秩问题的鲁棒性
    """
    # 从状态字典中提取参数
    iteration = iteration_state.get('iteration', 0)
    primal_res = iteration_state.get('primal_residual')
    dual_res = iteration_state.get('dual_residual')
    current_beta = iteration_state.get('beta', 1.0)
    converged = iteration_state.get('converged', False)
    objective = iteration_state.get('objective', None)
    
    # 如果已收敛，不再调整beta
    if converged:
        return float(current_beta)
    
    # 处理缺失值或无效值
    if primal_res is None or dual_res is None:
        return float(current_beta)
    
    # ==================== 回归问题检测与专用策略 ====================
    # 简化的回归问题检测：基于残差特征
    is_regression_problem = False
    
    # 检查残差特征：回归问题通常原始残差远大于对偶残差
    if iteration > 3:  # 给几次迭代让残差稳定
        if dual_res > 0 and primal_res > 10 * dual_res:
            # 原始残差远大于对偶残差，可能是回归问题
            is_regression_problem = True
            
        # 额外检测：如果对偶残差非常小且原始残差保持较大
        if dual_res < 1e-8 and primal_res > 1e-3:
            is_regression_problem = True
    
    # 弹性网络回归问题专用策略（强单调递增）
    if is_regression_problem:
        # 回归问题专用参数
        min_beta_reg = 0.5  # 提高初始最小值
        max_beta_reg = 1e4
        
        # 更激进的增长率，确保收敛
        if iteration < 50:
            growth_rate = 1.5  # 前期快速增加
        elif iteration < 200:
            growth_rate = 1.2  # 中期适度增加
        else:
            growth_rate = 1.1  # 后期缓慢增加
            
        # 确保初始beta不低于最小值
        if iteration <= 1:
            new_beta = max(current_beta, min_beta_reg)
        else:
            new_beta = current_beta * growth_rate
            
        # 应用上限
        new_beta = min(new_beta, max_beta_reg)
        return float(new_beta)
    
    # ==================== 非回归问题的自适应策略 ====================
    # 处理除零情况
    if dual_res < 1e-10:
        return float(current_beta)
    
    # 计算残差比
    ratio = primal_res / dual_res
    
    # 根据不同迭代阶段调整策略参数
    if iteration < 20:
        # 前期：温和调整，避免过度震荡
        mu = 8.0          # 平衡阈值
        tau_inc = 1.8     # 减小增长因子
        tau_dec = 1.8     # 减小减少因子
        min_beta = 1e-4
        max_beta = 1e5
        max_change = 2.5   # 最大单次变化倍数
    elif iteration < 80:
        # 中期：平衡调整
        mu = 10.0
        tau_inc = 1.6
        tau_dec = 1.6
        min_beta = 1e-5
        max_beta = 1e6
        max_change = 2.0
    else:
        # 后期：保守调整，促进收敛
        mu = 12.0
        tau_inc = 1.4
        tau_dec = 1.4
        min_beta = 1e-6
        max_beta = 1e6
        max_change = 1.8
    
    # 根据残差比调整beta
    if ratio > mu:
        # 原始残差过大，需要增大惩罚
        new_beta = current_beta * tau_inc
    elif ratio < 1.0 / mu:
        # 对偶残差过大，需要减小惩罚
        new_beta = current_beta / tau_dec
    else:
        # 平衡状态：轻微调整
        if primal_res > 1e-3 and iteration > 10:
            # 残差仍然较大，轻微增大beta
            new_beta = current_beta * 1.05
        else:
            # 接近收敛，保持beta不变
            new_beta = current_beta
    
    # 应用单次变化限制，避免剧烈震荡
    change_factor = new_beta / current_beta
    if change_factor > max_change:
        new_beta = current_beta * max_change
    elif change_factor < 1.0 / max_change:
        new_beta = current_beta / max_change
    
    # 确保beta在合理范围内
    new_beta = np.clip(new_beta, min_beta, max_beta)
    
    return float(new_beta)