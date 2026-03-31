from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的自适应β调整策略 - 专注于解决l1_regularization收敛问题
    
    核心改进：
    1. 针对l1_regularization问题优化beta调整逻辑
    2. 引入残差下降速度检测机制
    3. 优化beta调整频率，避免过早饱和
    4. 更精细的收敛阶段管理
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
    
    # 初始化历史变量（使用状态字典中的历史记录）
    history_key = 'residual_history'
    if history_key not in iteration_state:
        iteration_state[history_key] = []
    
    residual_history = iteration_state[history_key]
    residual_history.append((primal_res, dual_res))
    
    # 仅保留最近10次迭代的历史
    if len(residual_history) > 10:
        residual_history.pop(0)
    
    # 第一阶段：迭代前100次，采用温和调整策略
    if iteration < 100:
        # 针对l1_regularization问题：前期采用更积极的调整策略
        if iteration < 30:
            mu = 5.0
            tau_inc = 1.8
            tau_dec = 1.8
        else:
            mu = 8.0
            tau_inc = 1.6
            tau_dec = 1.6
        
        # 残差比调整策略
        if ratio > mu:
            new_beta = current_beta * tau_inc
        elif ratio < 1.0 / mu:
            new_beta = current_beta / tau_dec
        else:
            new_beta = current_beta
        
        # 针对l1_regularization：避免beta增长过快
        max_beta_growth = 3.0
        if new_beta / current_beta > max_beta_growth:
            new_beta = current_beta * max_beta_growth
    
    # 第二阶段：迭代100-300次，检测收敛停滞
    elif iteration < 300:
        # 检测残差下降速度
        if len(residual_history) >= 5:
            recent_primal = [r[0] for r in residual_history[-5:]]
            primal_change = np.log10(recent_primal[0]) - np.log10(recent_primal[-1])
            
            # 如果残差下降缓慢（小于0.1个数量级），调整策略
            if primal_change < 0.1:
                # 轻微调整beta以打破停滞
                if ratio > 2.0:
                    new_beta = current_beta * 1.3
                elif ratio < 0.5:
                    new_beta = current_beta / 1.3
                else:
                    # 随机微小扰动（±5%）打破停滞
                    perturbation = np.random.uniform(0.95, 1.05)
                    new_beta = current_beta * perturbation
            else:
                # 正常残差比调整策略
                mu = 10.0
                tau_inc = 1.5
                tau_dec = 1.5
                
                if ratio > mu:
                    new_beta = current_beta * tau_inc
                elif ratio < 1.0 / mu:
                    new_beta = current_beta / tau_dec
                else:
                    new_beta = current_beta
        else:
            # 正常残差比调整策略
            mu = 10.0
            tau_inc = 1.5
            tau_dec = 1.5
            
            if ratio > mu:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / mu:
                new_beta = current_beta / tau_dec
            else:
                new_beta = current_beta
    
    # 第三阶段：迭代300次以上，保守策略
    else:
        # 后期保守策略，专注于精细调整
        mu = 15.0
        tau_inc = 1.2
        tau_dec = 1.2
        
        # 针对小残差的特殊处理
        if primal_res < 1e-3 and dual_res < 1e-3:
            # 当两个残差都很小时，微调beta
            if ratio > 2.0:
                new_beta = current_beta * 1.1
            elif ratio < 0.5:
                new_beta = current_beta / 1.1
            else:
                new_beta = current_beta
        else:
            # 正常残差比调整
            if ratio > mu:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / mu:
                new_beta = current_beta / tau_dec
            else:
                new_beta = current_beta
        
        # 在后期阶段，每10次迭代才调整一次beta，避免震荡
        if iteration % 10 != 0:
            return float(current_beta)
    
    # 限制beta范围，确保数值稳定性
    min_beta = 1e-6
    max_beta = 1e6
    
    # 对于回归问题（通过目标函数值判断），调整范围
    is_regression = (objective > 0 and primal_res > 1e-3)
    if is_regression:
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
    
    # 如果beta接近上限且残差仍然较大，增加调整频率
    if new_beta > max_beta * 0.9 and primal_res > 1e-2:
        # 每2次迭代就调整一次beta
        if iteration % 2 == 0:
            new_beta = new_beta * 1.05
    
    return float(new_beta)