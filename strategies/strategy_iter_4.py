from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的自适应β调整策略，针对l1_regularization未收敛问题优化
    
    核心改进：
    1. 简化问题类型识别，采用统一残差比策略
    2. 针对l1_regularization问题添加特殊处理
    3. 调整beta范围限制，避免过早饱和
    4. 优化调整频率，避免过度调整
    """
    
    # 从状态字典获取参数
    primal_res = iteration_state.get('primal_residual', 1.0)
    dual_res = iteration_state.get('dual_residual', 1.0)
    current_beta = iteration_state.get('beta', 1.0)
    iteration = iteration_state.get('iteration', 0)
    converged = iteration_state.get('converged', False)
    
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
    
    # 基于迭代次数的策略调整
    if iteration < 100:
        # 早期阶段：温和调整
        mu = 8.0
        tau_inc = 1.6
        tau_dec = 1.6
        
        # 每5次迭代调整一次，避免过度调整
        if iteration % 5 != 0:
            return float(current_beta)
            
    elif iteration < 300:
        # 中期阶段：标准调整
        mu = 10.0
        tau_inc = 1.8
        tau_dec = 1.8
        
        # 每3次迭代调整一次
        if iteration % 3 != 0:
            return float(current_beta)
            
    else:
        # 后期阶段：保守调整
        mu = 12.0
        tau_inc = 1.2
        tau_dec = 1.2
        
        # 每2次迭代调整一次
        if iteration % 2 != 0:
            return float(current_beta)
    
    # 针对l1_regularization问题的特殊处理
    # 如果残差较小但未收敛（可能卡在局部最优）
    if primal_res < 1e-3 and dual_res < 1e-3 and iteration > 200:
        # 轻微扰动beta以跳出局部最优
        if iteration % 10 == 0:
            perturbation = 1.05 if np.random.rand() > 0.5 else 0.95
            new_beta = current_beta * perturbation
        else:
            new_beta = current_beta
    else:
        # 标准残差比策略
        if ratio > mu:
            new_beta = current_beta * tau_inc
        elif ratio < 1.0 / mu:
            new_beta = current_beta / tau_dec
        else:
            new_beta = current_beta
    
    # 限制beta范围，确保数值稳定性
    min_beta = 1e-6
    max_beta = 1e5  # 略微降低上限
    
    # 根据迭代阶段动态调整范围
    if iteration < 50:
        # 早期：允许更宽的范围探索
        min_beta = max(min_beta, 1e-5)
        max_beta = min(max_beta, 1e4)
    elif iteration > 400:
        # 晚期：缩小范围，促进收敛
        min_beta = max(min_beta, 1e-4)
        max_beta = min(max_beta, 1e3)
    
    # 应用范围限制
    new_beta = float(np.clip(new_beta, min_beta, max_beta))
    
    # 避免beta剧烈变化：限制单次变化不超过4倍
    max_change_factor = 4.0
    min_change_factor = 1.0 / max_change_factor
    actual_change = new_beta / current_beta
    
    if actual_change > max_change_factor:
        new_beta = current_beta * max_change_factor
    elif actual_change < min_change_factor:
        new_beta = current_beta * min_change_factor
    
    return float(new_beta)