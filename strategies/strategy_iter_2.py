from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    改进的ADMM自适应β调整策略
    
    核心改进：
    1. 针对回归问题采用单调递增策略（只增不减）
    2. 针对标准稀疏问题采用残差比策略
    3. 引入问题类型识别机制
    4. 优化收敛速度和稳定性
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
    
    # 第一阶段：迭代前50次，采用温和调整策略
    if iteration < 50:
        # 回归问题：采用温和的单调递增策略
        if objective == 0.0 and iteration > 10:  # 通过目标函数值识别回归问题
            # 回归问题专用：只增不减策略
            growth_rate = 1.1  # 温和增长
            max_beta = 1e4     # 回归问题上限
            new_beta = current_beta * growth_rate
            return float(np.clip(new_beta, 0.1, max_beta))
        else:
            # 标准问题：残差比策略
            mu = 8.0
            tau_inc = 1.8
            tau_dec = 1.8
            if ratio > mu:
                new_beta = current_beta * tau_inc
            elif ratio < 1.0 / mu:
                new_beta = current_beta / tau_dec
            else:
                new_beta = current_beta
    
    # 第二阶段：迭代50-200次，针对未收敛问题加强调整
    elif iteration < 200:
        # 检查是否为回归问题（弹性网络回归、l1回归等）
        is_regression_problem = (objective == 0.0 and primal_res < 1e-3)
        
        if is_regression_problem:
            # 回归问题：采用更强的单调递增策略
            growth_rate = 1.5  # 较强的增长
            max_beta = 1e4
            new_beta = current_beta * growth_rate
            return float(np.clip(new_beta, 0.5, max_beta))
        else:
            # 标准问题：动态调整策略
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
    
    # 第三阶段：迭代200次以上，采用保守策略
    else:
        # 后期保守策略，避免震荡
        mu = 15.0
        tau_inc = 1.2
        tau_dec = 1.2
        
        # 针对回归问题的最终加强策略
        if objective == 0.0:
            # 回归问题：继续温和增长
            growth_rate = 1.05
            max_beta = 1e4
            new_beta = current_beta * growth_rate
            return float(np.clip(new_beta, 1.0, max_beta))
        
        # 标准问题：残差比策略
        if ratio > mu:
            new_beta = current_beta * tau_inc
        elif ratio < 1.0 / mu:
            new_beta = current_beta / tau_dec
        else:
            new_beta = current_beta
    
    # 限制beta范围，确保数值稳定性
    min_beta = 1e-6
    max_beta = 1e6
    
    # 根据问题类型微调范围
    if objective == 0.0:  # 回归问题
        min_beta = max(min_beta, 0.1)  # 提高回归问题的下限
        max_beta = min(max_beta, 1e4)  # 降低回归问题的上限
    
    # 应用范围限制
    new_beta = float(np.clip(new_beta, min_beta, max_beta))
    
    # 避免beta剧烈变化：限制单次变化不超过5倍
    max_change_factor = 5.0
    min_change_factor = 1.0 / max_change_factor
    actual_change = new_beta / current_beta
    
    if actual_change > max_change_factor:
        new_beta = current_beta * max_change_factor
    elif actual_change < min_change_factor:
        new_beta = current_beta * min_change_factor
    
    return float(new_beta)