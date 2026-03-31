from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    基于残差比和迭代次数的自适应beta调整策略
    
    核心思想：
    1. 对回归问题采用单调递增策略
    2. 对非回归问题采用残差比自适应策略
    3. 基于迭代次数调整策略的激进程度
    4. 避免震荡，保持数值稳定性
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
    
    # 检测问题类型：回归问题需要单调递增策略
    # 通过观察残差特征来推断是否为回归问题
    is_regression_problem = False
    if iteration > 10 and primal_res is not None and dual_res is not None:
        # 回归问题通常原始残差远大于对偶残差
        if primal_res > 10 * max(dual_res, 1e-10):
            is_regression_problem = True
    
    # 回归问题：单调递增策略
    if is_regression_problem:
        # 回归问题超参数
        min_beta_reg = 0.1
        max_beta_reg = 1e4
        
        # 根据迭代次数调整增长率
        if iteration < 50:
            growth_rate = 1.2  # 前期快速增长
        elif iteration < 200:
            growth_rate = 1.1  # 中期正常增长
        else:
            growth_rate = 1.05  # 后期缓慢增长
        
        new_beta = current_beta * growth_rate
        return float(np.clip(new_beta, min_beta_reg, max_beta_reg))
    
    # 非回归问题：残差比自适应策略
    # 处理除零情况
    if dual_res < 1e-10:
        return float(current_beta)
    
    # 计算残差比
    ratio = primal_res / dual_res
    
    # 根据迭代次数调整策略参数
    if iteration < 50:
        # 前期：快速调整，寻找合适范围
        mu = 5.0      # 阈值较小，更敏感
        tau_inc = 2.5
        tau_dec = 2.5
        min_beta = 1e-5
        max_beta = 1e5
    elif iteration < 200:
        # 中期：平衡调整
        mu = 10.0
        tau_inc = 2.0
        tau_dec = 2.0
        min_beta = 1e-6
        max_beta = 1e6
    else:
        # 后期：保守调整，避免震荡
        mu = 15.0
        tau_inc = 1.5
        tau_dec = 1.5
        min_beta = 1e-6
        max_beta = 1e6
    
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
            # 接近收敛，保持beta不变
            new_beta = current_beta
    
    # 确保beta在合理范围内
    new_beta = np.clip(new_beta, min_beta, max_beta)
    
    # 防止beta变化过于剧烈
    max_change_factor = 3.0
    if new_beta > current_beta * max_change_factor:
        new_beta = current_beta * max_change_factor
    elif new_beta < current_beta / max_change_factor:
        new_beta = current_beta / max_change_factor
    
    return float(new_beta)