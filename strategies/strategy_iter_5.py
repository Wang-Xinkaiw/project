from typing import Dict, Any
import numpy as np

def adjust_beta(iteration_state: Dict[str, Any]) -> float:
    """
    自适应β调整策略 - 优化回归问题收敛性
    
    核心改进：
    1. 增强对回归问题的识别和针对性处理
    2. 采用更激进的单调递增策略处理elastic_net_regression
    3. 保持对非回归问题的平衡调整
    4. 提高数值稳定性和鲁棒性
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
    
    # ==================== 回归问题检测与专用策略 ====================
    # 改进的回归问题检测：更准确地识别弹性网络回归问题
    is_regression_problem = False
    
    # 检测回归问题的特征模式
    if iteration > 5:  # 给几次迭代让残差稳定
        # 特征1：原始残差持续远大于对偶残差（比例超过100）
        if dual_res > 0 and primal_res > 100 * dual_res:
            is_regression_problem = True
        
        # 特征2：原始残差持续较大，对偶残差持续很小
        if dual_res < 1e-8 and primal_res > 1e-2:
            is_regression_problem = True
        
        # 特征3：原始残差下降缓慢，对偶残差快速下降
        # 通过残差变化率判断（这里简化处理，实际应用中可能需要历史信息）
    
    # 回归问题专用策略：强单调递增
    if is_regression_problem:
        # 针对elastic_net_regression的强化策略
        min_beta_reg = 0.5    # 初始最小值
        max_beta_reg = 5e3    # 提高上限以增强惩罚
        
        # 更激进的增长策略
        if iteration < 30:
            growth_rate = 1.8  # 前期快速增加
        elif iteration < 150:
            growth_rate = 1.4  # 中期适度增加
        else:
            growth_rate = 1.2  # 后期缓慢增加
        
        # 确保初始beta足够大
        if iteration <= 2:
            new_beta = max(current_beta, min_beta_reg)
        else:
            new_beta = current_beta * growth_rate
        
        # 应用上下限
        new_beta = max(min(new_beta, max_beta_reg), min_beta_reg)
        return float(new_beta)
    
    # ==================== 非回归问题的自适应策略 ====================
    # 处理除零情况
    if dual_res < 1e-12:
        return float(current_beta)
    
    # 计算残差比
    ratio = primal_res / dual_res
    
    # 动态调整策略参数
    if iteration < 15:
        # 前期：温和调整，避免过度震荡
        mu = 7.0          # 平衡阈值
        tau_inc = 1.7     # 增长因子
        tau_dec = 1.7     # 减少因子
        min_beta = 1e-4
        max_beta = 1e5
        max_change = 2.3   # 最大单次变化倍数
    elif iteration < 60:
        # 中期：平衡调整
        mu = 9.0
        tau_inc = 1.5
        tau_dec = 1.5
        min_beta = 1e-5
        max_beta = 1e6
        max_change = 2.0
    else:
        # 后期：保守调整，促进收敛
        mu = 12.0
        tau_inc = 1.3
        tau_dec = 1.3
        min_beta = 1e-6
        max_beta = 1e6
        max_change = 1.6
    
    # 根据残差比调整beta
    if ratio > mu:
        # 原始残差过大，需要增大惩罚
        new_beta = current_beta * tau_inc
    elif ratio < 1.0 / mu:
        # 对偶残差过大，需要减小惩罚
        new_beta = current_beta / tau_dec
    else:
        # 平衡状态：根据残差绝对大小轻微调整
        if primal_res > 1e-2 and iteration > 5:
            # 残差仍然较大，轻微增大beta
            new_beta = current_beta * 1.08
        elif primal_res < 1e-6 and dual_res < 1e-6:
            # 接近收敛，保持beta稳定
            new_beta = current_beta
        else:
            # 正常状态，轻微调整以促进平衡
            if primal_res > dual_res:
                new_beta = current_beta * 1.02
            else:
                new_beta = current_beta * 0.98
    
    # 应用单次变化限制，避免剧烈震荡
    if current_beta > 0:
        change_factor = new_beta / current_beta
        if change_factor > max_change:
            new_beta = current_beta * max_change
        elif change_factor < 1.0 / max_change:
            new_beta = current_beta / max_change
    
    # 确保beta在合理范围内
    new_beta = np.clip(new_beta, min_beta, max_beta)
    
    return float(new_beta)