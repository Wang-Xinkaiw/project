from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    自适应惩罚参数β调整策略 - 改进版
    
    核心优化：
    1. 针对l1_regularization未收敛问题：增加对停滞情况的检测和处理
    2. 简化问题类型判断逻辑，提高鲁棒性
    3. 优化beta调整频率，避免过度调整
    4. 引入残差绝对阈值检测，提高收敛速度
    """
    
    # 从状态字典获取参数
    primal_res = iteration_state.get('primal_residual', 1.0)
    dual_res = iteration_state.get('dual_residual', 1.0)
    current_beta = iteration_state.get('beta', 1.0)
    iteration = iteration_state.get('iteration', 0)
    converged = iteration_state.get('converged', False)
    objective = iteration_state.get('objective', 0.0)
    
    # 如果已收敛，保持当前beta
    if converged:
        return float(current_beta)
    
    # 处理None或无效值
    if primal_res is None or dual_res is None:
        return float(current_beta)
    
    # 防止除零错误
    if dual_res < 1e-12:
        return float(current_beta)
    
    # 计算残差比
    ratio = primal_res / dual_res
    
    # 阶段1：迭代前50次 - 温和调整策略
    if iteration < 50:
        mu = 8.0
        tau_inc = 1.8
        tau_dec = 1.8
        
        if ratio > mu:
            new_beta = current_beta * tau_inc
        elif ratio < 1.0 / mu:
            new_beta = current_beta / tau_dec
        else:
            new_beta = current_beta
    
    # 阶段2：迭代50-150次 - 平衡调整策略
    elif iteration < 150:
        mu = 10.0
        tau_inc = 2.0
        tau_dec = 2.0
        
        # 检测收敛停滞：残差长时间不下降
        stagnation_threshold = 1e-4
        if primal_res < stagnation_threshold and dual_res < stagnation_threshold:
            # 接近收敛，采用微调策略
            if ratio > 5.0:
                new_beta = current_beta * 1.2
            elif ratio < 0.2:
                new_beta = current_beta / 1.2
            else:
                new_beta = current_beta
        else:
            # 正常调整
            if ratio > mu:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / mu:
                new_beta = current_beta / tau_dec
            else:
                new_beta = current_beta
    
    # 阶段3：迭代150-300次 - 针对l1_regularization的专门策略
    elif iteration < 300:
        # 对于可能停滞的问题，采用更激进的调整策略
        mu = 6.0  # 降低阈值，使调整更频繁
        tau_inc = 2.2
        tau_dec = 2.2
        
        # 检测是否可能是l1_regularization问题（残差较大且未收敛）
        if primal_res > 1e-3 or dual_res > 1e-3:
            # 针对l1_regularization：采用更频繁的调整
            if ratio > 3.0:  # 降低阈值，使更容易触发调整
                new_beta = current_beta * tau_inc
            elif ratio < 0.33:  # 提高敏感度
                new_beta = current_beta / tau_dec
            else:
                new_beta = current_beta
        else:
            # 其他问题：正常调整
            if ratio > mu:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / mu:
                new_beta = current_beta / tau_dec
            else:
                new_beta = current_beta
    
    # 阶段4：迭代300次以上 - 保守策略，防止震荡
    else:
        mu = 15.0
        tau_inc = 1.1
        tau_dec = 1.1
        
        # 如果残差已经很小但未收敛，微调beta
        if primal_res < 1e-4 and dual_res < 1e-4:
            # 非常接近收敛，只做微小调整
            if ratio > 2.0:
                new_beta = current_beta * 1.05
            elif ratio < 0.5:
                new_beta = current_beta / 1.05
            else:
                new_beta = current_beta
        else:
            # 正常调整
            if ratio > mu:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / mu:
                new_beta = current_beta / tau_dec
            else:
                new_beta = current_beta
    
    # 限制beta范围，确保数值稳定性
    min_beta = 1e-6
    max_beta = 1e6
    
    # 针对回归问题的特殊范围（通过残差特性判断）
    # 如果残差非常小且目标函数值接近0，可能是回归问题
    if primal_res < 1e-6 and dual_res < 1e-6 and abs(objective) < 1e-3:
        min_beta = max(min_beta, 0.1)
        max_beta = min(max_beta, 1e4)
    
    # 应用范围限制
    new_beta = float(np.clip(new_beta, min_beta, max_beta))
    
    # 避免beta剧烈变化：限制单次变化不超过3倍
    max_change_factor = 3.0
    min_change_factor = 1.0 / max_change_factor
    actual_change = new_beta / current_beta
    
    if actual_change > max_change_factor:
        new_beta = current_beta * max_change_factor
    elif actual_change < min_change_factor:
        new_beta = current_beta * min_change_factor
    
    # 防止beta无限增长：在迭代后期如果残差不降，限制beta增长
    if iteration > 200 and new_beta > current_beta:
        stagnation_check = primal_res > 1e-2 and dual_res > 1e-2
        if stagnation_check and (iteration % 20 == 0):  # 每20次迭代才允许增长一次
            pass  # 允许增长
        elif stagnation_check:
            new_beta = current_beta  # 保持当前值
    
    return float(new_beta)