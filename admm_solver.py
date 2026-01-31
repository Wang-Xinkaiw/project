# admm_solver.py
"""
ADMM求解器 - 实现论文公式17的标准ADMM迭代
与策略类解耦，便于测试不同调参策略
"""

import numpy as np
from typing import Dict, Tuple, Callable


class ADMMSolver:
    """
    通用ADMM求解器
    
    实现论文公式17:
        x₁^{k+1} = argmin_{x₁} L_β(x₁, x₂^k, λ^k)
        x₂^{k+1} = argmin_{x₂} L_β(x₁^{k+1}, x₂, λ^k)
        λ^{k+1} = λ^k - τβ(A₁x₁^{k+1} + A₂x₂^{k+1} - b)
    """
    
    def __init__(self, 
                 strategy,
                 tau: float = 1.0,
                 max_iter: int = 1000,
                 tol: float = 1e-6):
        """
        初始化ADMM求解器
        
        参数:
            strategy: 参数调整策略实例
            tau: 对偶更新步长τ (论文公式17中的τ)
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.strategy = strategy
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        
        # 验证τ的范围 (论文建议τ∈(0, (1+√5)/2) ≈ 2.618)
        if tau <= 0 or tau >= 2.618:
            print(f"警告: 对偶步长τ={tau}，建议范围(0, 2.618)")
    
    def solve(self,
              f1_solver: Callable,
              f2_solver: Callable,
              A1: np.ndarray,
              A2: np.ndarray,
              b: np.ndarray,
              x1_0: np.ndarray,
              x2_0: np.ndarray,
              lambda_0: np.ndarray) -> Tuple:
        """
        执行ADMM求解
        
        参数:
            f1_solver: x₁子问题求解器，函数签名(x₂, λ, β) -> x₁
            f2_solver: x₂子问题求解器，函数签名(x₁, λ, β) -> x₂
            A1, A2, b: 约束参数
            x1_0, x2_0, lambda_0: 初始值
            
        返回:
            x1_opt, x2_opt, lambda_opt, history: 最终解和迭代历史
        """
        # 初始化
        x1 = x1_0.copy()
        x2 = x2_0.copy()
        lambda_val = lambda_0.copy()
        
        # 获取初始参数
        params = self.strategy.get_parameters()
        beta = params.get('beta', 1.0)
        
        # 迭代历史
        history = {
            'primal_res': [],
            'dual_res': [],
            'beta_values': [],
            'objective': [],
            'iterations': 0
        }
        
        print(f"\n开始ADMM求解 (τ={self.tau}, 最大迭代={self.max_iter})")
        print(f"初始参数: β={beta:.4f}")
        
        # 主迭代循环
        for k in range(self.max_iter):
            # 保存x2的旧值（用于计算对偶残差）
            x2_old = x2.copy()
            
            # 获取当前β
            current_beta = self.strategy.get_parameters().get('beta', beta)
            
            # 论文公式17: 更新x₁和x₂
            x1 = f1_solver(x2, lambda_val, current_beta)
            x2 = f2_solver(x1, lambda_val, current_beta)
            
            # 计算原始残差 r^k = A₁x₁^k + A₂x₂^k - b
            primal_res = A1 @ x1 + A2 @ x2 - b
            primal_norm = np.linalg.norm(primal_res)
            
            # 计算对偶残差 s^k = β^k A₁^T A₂ (x₂^{k+1} - x₂^k)
            dual_res = current_beta * A1.T @ A2 @ (x2 - x2_old)
            dual_norm = np.linalg.norm(dual_res)
            
            # 论文公式17: 更新拉格朗日乘子
            lambda_val = lambda_val - self.tau * current_beta * primal_res
            
            # 构建迭代状态
            iteration_state = {
                'iteration': k,
                'primal_residual': primal_norm,
                'dual_residual': dual_norm,
                'x1': x1.copy(),
                'x2': x2.copy(),
                'lambda': lambda_val.copy(),
                'beta': current_beta
            }
            
            # 使用策略更新参数
            new_params = self.strategy.update_parameters(iteration_state)
            
            # 记录历史
            history['primal_res'].append(primal_norm)
            history['dual_res'].append(dual_norm)
            history['beta_values'].append(new_params.get('beta', current_beta))
            history['iterations'] = k + 1
            
            # 改进的收敛检查
            # 论文中通常检查原始和对偶残差，但实际中可以使用相对收敛
            norm_b = np.linalg.norm(b)
            primal_rel = primal_norm / max(1.0, norm_b)
            dual_rel = dual_norm / max(1.0, np.linalg.norm(lambda_val))
            
            # 每50次迭代打印进度
            if (k % 50 == 0) or (k == self.max_iter - 1):
                adjustment = new_params.get('adjustment_type', 'keep')
                print(f"迭代 {k+1:4d}: 原始残差={primal_norm:.2e}, "
                      f"对偶残差={dual_norm:.2e}, β={current_beta:.4f}, 调整={adjustment}")
            
            # 收敛条件：原始残差和对偶残差都小于容差
            if primal_norm < self.tol and dual_norm < self.tol:
                print(f"\n在第 {k+1} 次迭代收敛!")
                break
        
        print(f"\n迭代完成! 最终β={self.strategy.get_parameters()['beta']:.4f}")
        
        return x1, x2, lambda_val, history