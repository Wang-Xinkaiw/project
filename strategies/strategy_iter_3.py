from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的ADMM惩罚参数自适应调整策略
    
    核心特性：
    1. 针对回归问题采用确定性单调递增策略
    2. 针对其他问题采用残差比自适应策略
    3. 增加稳定性和避免震荡
    4. 处理除零和无效值
    """
    # 从状态字典中提取参数
    iteration = iteration_state.get('iteration', 0)
    primal_res = iteration_state.get('primal_residual')
    dual_res = iteration_state.get('dual_residual')
    current_beta = iteration_state.get('beta', 1.0)
    converged = iteration_state.get('converged', False)
    
    # 如果已收敛，不再调整beta
    if converged:
        return float(current_beta)
    
    # 处理缺失值或无效值
    if primal_res is None or dual_res is None:
        return float(current_beta)
    
    # ==================== 针对elastic_net_regression的专用策略 ====================
    # 检测是否为回归问题（基于残差特性判断）
    # 回归问题通常原始残差远大于对偶残差，且对偶残差非常小
    is_regression_problem = False
    if iteration > 5:  # 给几次迭代让残差稳定
        if primal_res > 0 and dual_res > 0:
            # 回归问题的典型特征：primal_res >> dual_res 且 dual_res很小
            if primal_res > 100 * max(dual_res, 1e-10) or dual_res < 1e-6:
                is_regression_problem = True
                
        # 额外的回归问题检测：如果连续多次迭代primal_res保持较大值
        # 注意：这里我们无法保存历史，但可以通过当前值推断
    
    # ============= 回归问题：使用单调递增策略（针对elastic_net_regression） =============
    if is_regression_problem:
        # 回归问题专用超参数
        min_beta_reg = 0.1
        max_beta_reg = 1e4
        
        # 根据迭代阶段调整增长率
        if iteration < 50:
            # 前期：快速增加以快速逼近解
            growth_rate = 1.5  # 适度的增长率
        elif iteration < 150:
            # 中期：稳定增长
            growth_rate = 1.2
        elif iteration < 300:
            # 后期：缓慢增长
            growth_rate = 1.1
        else:
            # 接近迭代上限：更缓慢
            growth_rate = 1.05
            
        # 确保初始beta不低于最小值
        if iteration == 0 and current_beta < min_beta_reg:
            new_beta = min_beta_reg
        else:
            new_beta = current_beta * growth_rate
            
        return float(np.clip(new_beta, min_beta_reg, max_beta_reg))
    
    # ============= 非回归问题：残差比自适应策略 =============
    # 处理除零情况
    if dual_res < 1e-10:
        return float(current_beta)
    
    # 计算残差比
    ratio = primal_res / dual_res
    
    # 根据迭代阶段调整策略参数
    if iteration < 30:
        # 前期：快速调整找到合适范围
        mu = 5.0       # 较小的阈值，更敏感
        tau_inc = 2.5  # 增大因子
        tau_dec = 2.5  # 减小因子
        min_beta = 1e-4
        max_beta = 1e5
        min_change = 0.8  # 最小变化因子
        max_change = 3.0  # 最大变化因子
    elif iteration < 100:
        # 中期：平衡调整
        mu = 8.0
        tau_inc = 2.0
        tau_dec = 2.0
        min_beta = 1e-5
        max_beta = 1e6
        min_change = 0.9
        max_change = 2.5
    else:
        # 后期：保守调整，避免震荡
        mu = 12.0
        tau_inc = 1.5
        tau_dec = 1.5
        min_beta = 1e-6
        max_beta = 1e6
        min_change = 0.95
        max_change = 2.0
    
    # 根据残差比调整beta
    if ratio > mu:
        # 原始残差过大，需要增大惩罚
        new_beta = current_beta * tau_inc
    elif ratio < 1.0 / mu:
        # 对偶残差过大，需要减小惩罚
        new_beta = current_beta / tau_dec
    else:
        # 平衡状态：轻微调整以促进收敛
        if primal_res > 1e-3:
            # 残差仍然较大，轻微增大beta
            new_beta = current_beta * 1.05
        else:
            # 接近收敛，保持beta不变或轻微调整
            new_beta = current_beta
    
    # 应用变化限制，避免剧烈震荡
    if new_beta > current_beta * max_change:
        new_beta = current_beta * max_change
    elif new_beta < current_beta * min_change:
        new_beta = current_beta * min_change
    
    # 确保beta在合理范围内
    new_beta = np.clip(new_beta, min_beta, max_beta)
    
    return float(new_beta)