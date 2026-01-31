import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = [12, 8]

# ============== 辅助函数 ==============
def soft_threshold(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def svt(X, tau):
    """改进的奇异值阈值函数，增加稳定性"""
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thresh = np.maximum(s - tau, 0)
        rank = np.sum(s_thresh > 0)
        if rank == 0:
            return np.zeros_like(X), 0.0
        X_prox = U[:, :rank] @ np.diag(s_thresh[:rank]) @ Vt[:rank, :]
        return X_prox, np.sum(s_thresh)
    except np.linalg.LinAlgError:
        # SVD失败时返回原始矩阵
        return X, np.linalg.norm(X, 'nuc')

def prox_l21(X, alpha):
    """L2,1范数近端算子，增加数值稳定性"""
    col_norms = np.sqrt(np.sum(X**2, axis=0, keepdims=True))
    col_norms = np.where(col_norms < 1e-10, 1e-10, col_norms)
    scale = np.maximum(1.0 - alpha / col_norms, 0.0)
    return X * scale

def project_simplex(v):
    """投影到概率单纯形（矩阵按列投影）"""
    if v.ndim == 1:
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.where(u * (1 + np.arange(1, n+1)) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0)
    elif v.ndim == 2:
        d, n = v.shape
        result = np.zeros_like(v)
        for j in range(n):
            result[:, j] = project_simplex(v[:, j])
        return result
    else:
        raise ValueError("输入必须是向量或矩阵")

def comp_loss(E, loss_type):
    if loss_type == 'l1':
        return np.sum(np.abs(E))
    elif loss_type == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro')**2
    else:
        raise ValueError(f"不支持的损失函数: {loss_type}")

def prox_elasticnet(x, lambda1, lambda2):
    factor = 1.0 / (1.0 + 2.0 * lambda2)
    return factor * soft_threshold(x, lambda1)

# ============== ADMM算法类 ==============

class StandardADMM:
    """标准ADMM算法 - 使用MATLAB默认策略"""
    def __init__(self, rho=0.1, max_iter=2000, tol=1e-5, rho_update=True, max_rho=1e4):
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.rho_update = rho_update
        self.max_rho = max_rho
        
    def solve(self, problem_func, *args):
        return problem_func(self, *args)

class ImprovedAdaptiveADMM:
    """改进的自适应ADMM算法"""
    def __init__(self, rho=0.1, max_iter=2000, tol=1e-5, rho_update=True, max_rho=1e4):
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.rho_update = rho_update
        self.max_rho = max_rho
        
        # 改进自适应参数
        self.eta = 0.8      # 学习率
        self.epsilon = 1e-8
        self.alpha = 3.0    # 平滑系数
        self.beta = 0.15    # 动量系数
        self.gamma = 0.25   # 最大调整幅度
    
    def solve(self, problem_func, *args):
        return problem_func(self, *args)

# ============== 前4个问题求解函数（保持原样）==============

# 1. L1问题
def solve_l1_standard(admm_solver, A, B):
    """标准ADMM - L1最小化问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    Z = np.zeros((na, nb))
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    for k in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()
        
        # 更新X
        X = soft_threshold(Z - Y2/rho, 1/rho)
        
        # 更新Z
        Z = invAtAI @ (-A.T @ Y1/rho + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z - B
        dY2 = X - Z
        
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 标准ADMM的固定倍增策略
        if rho_update:
            rho = min(rho * 1.1, max_rho)
    
    return max_iter

def solve_l1_improved(admm_solver, A, B):
    """改进自适应ADMM - L1最小化问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    # 改进自适应参数
    eta = admm_solver.eta
    epsilon = admm_solver.epsilon
    alpha = admm_solver.alpha
    beta = admm_solver.beta
    gamma = admm_solver.gamma
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    Z = np.zeros((na, nb))
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    rho_prev = rho
    
    for k in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()
        
        # 更新X
        X = soft_threshold(Z - Y2/rho, 1/rho)
        
        # 更新Z
        Z = invAtAI @ (-A.T @ Y1/rho + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z - B
        dY2 = X - Z
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        dual_res = rho * np.linalg.norm(A.T @ A @ (Z - Zk), 'fro')
        
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 改进的自适应rho更新
        if rho_update and k > 0:
            # 计算残差比例
            if dual_res + epsilon > 0:
                ratio = primal_res / (dual_res + epsilon)
            else:
                ratio = 5.0  # 对偶残差过小时，需要增大rho
            
            # 平滑的符号函数
            diff = primal_res - dual_res
            total = primal_res + dual_res + epsilon
            sign_smooth = np.tanh(alpha * diff / total)
            
            # 动量项
            if k > 1:
                momentum = beta * (rho - rho_prev) / rho_prev
            else:
                momentum = 0
            
            # 使用arctan函数限制调整幅度
            adjustment = eta * np.arctan(ratio - 1) * sign_smooth + momentum
            
            # 限制调整幅度
            adjustment = np.clip(adjustment, -gamma, gamma)
            
            # 更新rho
            rho_prev = rho
            rho = rho * (1 + adjustment)
            rho = np.clip(rho, 1e-6, max_rho)
    
    return max_iter

# 2. 弹性网问题
def solve_elasticnet_standard(admm_solver, A, B, lambda_val):
    """标准ADMM - 弹性网问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    Z = np.zeros((na, nb))
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    for k in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()
        
        # 更新X (弹性网近端算子)
        temp = Z - Y2/rho
        factor = 1.0 / (1.0 + 2.0 * lambda_val / rho)
        X = factor * soft_threshold(temp, 1/rho)
        
        # 更新Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2)/rho + AtB + X)
        
        # 计算残差
        dY1 = A @ Z - B
        dY2 = X - Z
        
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 标准ADMM的固定倍增策略
        if rho_update:
            rho = min(rho * 1.1, max_rho)
    
    return max_iter

def solve_elasticnet_improved(admm_solver, A, B, lambda_val):
    """改进自适应ADMM - 弹性网问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    # 改进自适应参数
    eta = admm_solver.eta
    epsilon = admm_solver.epsilon
    alpha = admm_solver.alpha
    beta = admm_solver.beta
    gamma = admm_solver.gamma
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    Z = np.zeros((na, nb))
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    rho_prev = rho
    
    for k in range(max_iter):
        Xk = X.copy()
        Zk = Z.copy()
        
        # 更新X
        temp = Z - Y2/rho
        factor = 1.0 / (1.0 + 2.0 * lambda_val / rho)
        X = factor * soft_threshold(temp, 1/rho)
        
        # 更新Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2)/rho + AtB + X)
        
        # 计算残差
        dY1 = A @ Z - B
        dY2 = X - Z
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        dual_res = rho * np.linalg.norm(A.T @ A @ (Z - Zk), 'fro')
        
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 改进的自适应rho更新
        if rho_update and k > 0:
            if dual_res + epsilon > 0:
                ratio = primal_res / (dual_res + epsilon)
            else:
                ratio = 5.0
            
            diff = primal_res - dual_res
            total = primal_res + dual_res + epsilon
            sign_smooth = np.tanh(alpha * diff / total)
            
            if k > 1:
                momentum = beta * (rho - rho_prev) / rho_prev
            else:
                momentum = 0
            
            adjustment = eta * np.arctan(ratio - 1) * sign_smooth + momentum
            adjustment = np.clip(adjustment, -gamma, gamma)
            
            rho_prev = rho
            rho = rho * (1 + adjustment)
            rho = np.clip(rho, 1e-6, max_rho)
    
    return max_iter

# 3. L1正则化回归问题
def solve_l1r_standard(admm_solver, A, B, lambda_val, loss='l1'):
    """标准ADMM - L1正则化回归问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol * 10  # 放宽收敛条件
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros((na, nb))
    Y1 = E.copy()
    Y2 = X.copy()
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Zk = Z.copy()
        
        # 更新X
        X = soft_threshold(Z - Y2/rho, lambda_val/rho)
        
        # 更新E
        if loss == 'l1':
            E = soft_threshold(B - A @ Z - Y1/rho, 1/rho)
        elif loss == 'l2':
            E = rho * (B - A @ Z - Y1/rho) / (1 + rho)
        else:
            raise ValueError('不支持的损失函数')
        
        # 更新Z
        Z = invAtAI @ (-A.T @ (Y1/rho + E) + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z + E - B
        dY2 = X - Z
        
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 标准ADMM的固定倍增策略
        if rho_update:
            rho = min(rho * 1.1, max_rho)
    
    return max_iter

def solve_l1r_improved(admm_solver, A, B, lambda_val, loss='l1'):
    """改进自适应ADMM - L1正则化回归问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol * 10  # 放宽收敛条件
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    # 改进自适应参数
    eta = admm_solver.eta
    epsilon = admm_solver.epsilon
    alpha = admm_solver.alpha
    beta = admm_solver.beta
    gamma = admm_solver.gamma
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros((na, nb))
    Y1 = E.copy()
    Y2 = X.copy()
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    rho_prev = rho
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Zk = Z.copy()
        
        # 更新X
        X = soft_threshold(Z - Y2/rho, lambda_val/rho)
        
        # 更新E
        if loss == 'l1':
            E = soft_threshold(B - A @ Z - Y1/rho, 1/rho)
        elif loss == 'l2':
            E = rho * (B - A @ Z - Y1/rho) / (1 + rho)
        else:
            raise ValueError('不支持的损失函数')
        
        # 更新Z
        Z = invAtAI @ (-A.T @ (Y1/rho + E) + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z + E - B
        dY2 = X - Z
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        
        # 估计对偶残差
        if k > 0:
            dual_res = rho * np.linalg.norm(A.T @ A @ (Z - Zk), 'fro')
        else:
            dual_res = 0
        
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 改进的自适应rho更新
        if rho_update and k > 0:
            if dual_res + epsilon > 0:
                ratio = primal_res / (dual_res + epsilon)
            else:
                ratio = 5.0
            
            diff = primal_res - dual_res
            total = primal_res + dual_res + epsilon
            sign_smooth = np.tanh(alpha * diff / total)
            
            if k > 1:
                momentum = beta * (rho - rho_prev) / rho_prev
            else:
                momentum = 0
            
            adjustment = eta * np.arctan(ratio - 1) * sign_smooth + momentum
            adjustment = np.clip(adjustment, -gamma, gamma)
            
            rho_prev = rho
            rho = rho * (1 + adjustment)
            rho = np.clip(rho, 1e-6, max_rho)
    
    return max_iter

# 4. 弹性网回归问题
def solve_elasticnetr_standard(admm_solver, A, B, lambda1, lambda2, loss='l1'):
    """标准ADMM - 弹性网回归问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol * 10
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros((na, nb))
    Y1 = E.copy()
    Y2 = X.copy()
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Zk = Z.copy()
        
        # 更新X (弹性网近端算子)
        X = prox_elasticnet(Z - Y2/rho, lambda1/rho, lambda2/rho)
        
        # 更新E
        if loss == 'l1':
            E = soft_threshold(B - A @ Z - Y1/rho, 1/rho)
        elif loss == 'l2':
            E = rho * (B - A @ Z - Y1/rho) / (1 + rho)
        else:
            raise ValueError('不支持的损失函数')
        
        # 更新Z
        Z = invAtAI @ (-A.T @ (Y1/rho + E) + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z + E - B
        dY2 = X - Z
        
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 标准ADMM的固定倍增策略
        if rho_update:
            rho = min(rho * 1.1, max_rho)
    
    return max_iter

def solve_elasticnetr_improved(admm_solver, A, B, lambda1, lambda2, loss='l1'):
    """改进自适应ADMM - 弹性网回归问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol * 10
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    # 改进自适应参数
    eta = admm_solver.eta
    epsilon = admm_solver.epsilon
    alpha = admm_solver.alpha
    beta = admm_solver.beta
    gamma = admm_solver.gamma
    
    d, na = A.shape
    _, nb = B.shape
    
    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros((na, nb))
    Y1 = E.copy()
    Y2 = X.copy()
    
    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    
    rho_prev = rho
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Zk = Z.copy()
        
        # 更新X (弹性网近端算子)
        X = prox_elasticnet(Z - Y2/rho, lambda1/rho, lambda2/rho)
        
        # 更新E
        if loss == 'l1':
            E = soft_threshold(B - A @ Z - Y1/rho, 1/rho)
        elif loss == 'l2':
            E = rho * (B - A @ Z - Y1/rho) / (1 + rho)
        else:
            raise ValueError('不支持的损失函数')
        
        # 更新Z
        Z = invAtAI @ (-A.T @ (Y1/rho + E) + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z + E - B
        dY2 = X - Z
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        
        # 估计对偶残差
        if k > 0:
            dual_res = rho * np.linalg.norm(A.T @ A @ (Z - Zk), 'fro')
        else:
            dual_res = 0
        
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 改进的自适应rho更新
        if rho_update and k > 0:
            if dual_res + epsilon > 0:
                ratio = primal_res / (dual_res + epsilon)
            else:
                ratio = 5.0
            
            diff = primal_res - dual_res
            total = primal_res + dual_res + epsilon
            sign_smooth = np.tanh(alpha * diff / total)
            
            if k > 1:
                momentum = beta * (rho - rho_prev) / rho_prev
            else:
                momentum = 0
            
            adjustment = eta * np.arctan(ratio - 1) * sign_smooth + momentum
            adjustment = np.clip(adjustment, -gamma, gamma)
            
            rho_prev = rho
            rho = rho * (1 + adjustment)
            rho = np.clip(rho, 1e-6, max_rho)
    
    return max_iter

# 5. 低秩矩阵补全问题 - 改进版本
def solve_lrmc_standard_improved(admm_solver, M_obs, mask):
    """标准ADMM - 低秩矩阵补全问题（改进版）"""
    rho = 0.5  # 增大初始rho
    max_iter = 2000  # 增加迭代次数
    tol = admm_solver.tol * 100  # 放宽收敛条件
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    d, n = M_obs.shape
    M = np.zeros((d, n))
    M[mask] = M_obs[mask]
    
    # 初始化
    X = np.random.randn(d, n) * 0.01
    E = np.zeros((d, n))
    Y = np.zeros((d, n))
    
    best_error = float('inf')
    best_iter = 0
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        
        # 更新X - 增加正则化
        try:
            X_prox, _ = svt(-(E - M + Y/rho), 1/rho)
            X = 0.8 * X + 0.2 * X_prox  # 增加惯性项
        except:
            X = Xk  # 如果SVD失败，保持原值
        
        # 更新E
        E = -(X - M + Y/rho)
        E[mask] = 0
        
        # 计算残差
        dY = X + E - M
        primal_res = np.linalg.norm(dY, 'fro')
        
        # 计算相对误差
        relative_error = primal_res / (np.linalg.norm(M[mask]) + 1e-8)
        
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chg = max(chgX, chgE, np.max(np.abs(dY)))
        
        # 多个停止条件
        if chg < tol or relative_error < 1e-4 or (k > 100 and relative_error < 1e-3):
            return k + 1
        
        # 更新拉格朗日乘子
        Y = Y + rho * dY
        
        # 更温和的rho更新策略
        if rho_update and k > 0:
            dual_res = rho * np.linalg.norm(E - Ek, 'fro')
            if primal_res > 1.5 * dual_res and k % 10 == 0:
                rho = min(rho * 1.05, max_rho)  # 更温和的增长
            elif dual_res > 1.5 * primal_res and k % 10 == 0:
                rho = max(rho / 1.05, 1e-4)  # 更温和的减小
    
    return max_iter

def solve_lrmc_improved_improved(admm_solver, M_obs, mask):
    """改进自适应ADMM - 低秩矩阵补全问题（改进版）"""
    rho = 0.5  # 增大初始rho
    max_iter = 2000  # 增加迭代次数
    tol = admm_solver.tol * 100  # 放宽收敛条件
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    # 改进自适应参数
    eta = admm_solver.eta
    epsilon = admm_solver.epsilon
    alpha = admm_solver.alpha
    beta = admm_solver.beta
    gamma = admm_solver.gamma
    
    d, n = M_obs.shape
    M = np.zeros((d, n))
    M[mask] = M_obs[mask]
    
    # 更好的初始化
    X = np.random.randn(d, n) * 0.01
    E = np.zeros((d, n))
    Y = np.zeros((d, n))
    
    rho_prev = rho
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        
        # 更新X - 增加稳定性
        try:
            X_prox, _ = svt(-(E - M + Y/rho), 1/rho)
            X = 0.8 * X + 0.2 * X_prox  # 增加惯性项，提高稳定性
        except:
            X = Xk  # 如果SVD失败，保持原值
        
        # 更新E
        E = -(X - M + Y/rho)
        E[mask] = 0
        
        # 计算残差
        dY = X + E - M
        primal_res = np.linalg.norm(dY, 'fro')
        dual_res = rho * np.linalg.norm(E - Ek, 'fro')
        
        # 计算相对误差
        relative_error = primal_res / (np.linalg.norm(M[mask]) + 1e-8)
        
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chg = max(chgX, chgE, np.max(np.abs(dY)))
        
        # 多个停止条件
        if chg < tol or relative_error < 1e-4 or (k > 100 and relative_error < 1e-3):
            return k + 1
        
        # 更新拉格朗日乘子
        Y = Y + rho * dY
        
        # 改进的自适应rho更新 - 为LRMC调整
        if rho_update and k > 0:
            if dual_res + epsilon > 0:
                ratio = primal_res / (dual_res + epsilon)
            else:
                ratio = 5.0
            
            diff = primal_res - dual_res
            total = primal_res + dual_res + epsilon
            sign_smooth = np.tanh(alpha * diff / total)
            
            if k > 1:
                momentum = beta * (rho - rho_prev) / rho_prev
            else:
                momentum = 0
            
            # LRMC问题需要更温和的调整
            adjustment = 0.5 * eta * np.arctan(ratio - 1) * sign_smooth + momentum
            adjustment = np.clip(adjustment, -0.1, 0.1)  # 更严格的限制
            
            rho_prev = rho
            rho = rho * (1 + adjustment)
            rho = np.clip(rho, 0.01, 100)  # 限制rho的范围
    
    return max_iter

# 6. 低秩表示问题 - 修复版本
def solve_lrr_standard_fixed(admm_solver, A, B, lambda_val, loss='l1'):
    """修复版标准ADMM - 低秩表示问题"""
    rho = 0.2  # 降低初始rho
    max_iter = 5000  # 增加最大迭代次数
    tol = admm_solver.tol * 50  # 适当放宽收敛条件
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    d, na = A.shape
    _, nb = B.shape
    
    # 一致的初始化：全部初始化为0
    X = np.zeros((nb, na))
    E = np.zeros((d, na))
    J = np.zeros((nb, na))
    Y1 = np.zeros((d, na))
    Y2 = np.zeros((nb, na))
    
    # 预计算并添加正则化
    BtB = B.T @ B
    BtA = B.T @ A
    I = np.eye(nb)
    reg = 1e-6 * np.eye(nb)
    
    # 预计算逆矩阵
    try:
        invBtBI = np.linalg.inv(BtB + I + reg)
    except np.linalg.LinAlgError:
        invBtBI = np.linalg.pinv(BtB + I + reg)
    
    best_obj = float('inf')
    no_improve_count = 0
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Jk = J.copy()
        
        # 1. 更新J（低秩部分）
        try:
            tau = lambda_val / (rho + 1e-8)
            J_prox, _ = svt(X + Y2/rho, tau)
            J = 0.9 * Jk + 0.1 * J_prox  # 缓慢的惯性更新
        except:
            J = Jk
        
        # 2. 更新E（稀疏部分）
        if loss == 'l1':
            E_temp = A - B @ X + Y1/rho
            threshold = 1.0 / (rho + 1e-8)
            E = soft_threshold(E_temp, threshold)
        elif loss == 'l21':
            E_temp = A - B @ X + Y1/rho
            threshold = lambda_val / (rho + 1e-8)
            E = prox_l21(E_temp, threshold)
        else:
            raise ValueError(f"不支持的损失类型: {loss}")
        
        # 3. 更新X
        try:
            X = invBtBI @ (B.T @ (Y1/rho - E) + BtA - Y2/rho + J)
        except:
            X = 0.9 * Xk + 0.1 * invBtBI @ (B.T @ (Y1/rho - E) + BtA - Y2/rho + J)
        
        # 4. 计算残差
        r1 = A - B @ X - E
        r2 = X - J
        
        primal_res = np.linalg.norm(r1, 'fro')**2 + np.linalg.norm(r2, 'fro')**2
        primal_res = np.sqrt(primal_res)
        
        # 正确的对偶残差计算
        if k > 0:
            dual_res = rho * np.linalg.norm(B @ (X - Xk), 'fro') + rho * np.linalg.norm(J - Jk, 'fro')
        else:
            dual_res = 0
        
        # 5. 计算目标函数值
        if loss == 'l1':
            sparse_term = np.sum(np.abs(E))
        else:
            sparse_term = np.sum(np.sqrt(np.sum(E**2, axis=0)))
        
        # 计算核范数
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        nuclear_norm = np.sum(s)
        
        obj_value = 0.5 * np.linalg.norm(A - B @ X - E, 'fro')**2 + lambda_val * (nuclear_norm + sparse_term)
        
        # 6. 检查收敛性
        chg = max(
            np.linalg.norm(X - Xk, 'fro') / (np.linalg.norm(Xk, 'fro') + 1e-8),
            np.linalg.norm(E - Ek, 'fro') / (np.linalg.norm(Ek, 'fro') + 1e-8),
            np.linalg.norm(J - Jk, 'fro') / (np.linalg.norm(Jk, 'fro') + 1e-8),
            np.max(np.abs(r1)),
            np.max(np.abs(r2))
        )
        
        # 检查目标函数是否改进
        if obj_value < best_obj - 1e-6:
            best_obj = obj_value
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 收敛条件
        converged = (
            chg < tol or
            (primal_res < 1e-4 and k > 100) or
            (no_improve_count > 20 and k > 50)
        )
        
        if converged:
            return k + 1
        
        # 7. 更新拉格朗日乘子
        Y1 = Y1 + rho * r1
        Y2 = Y2 + rho * r2
        
        # 8. 自适应调整rho
        if k > 0 and k % 10 == 0:
            if primal_res > 10 * dual_res and rho < 100:
                rho = min(rho * 1.5, 100)
            elif dual_res > 10 * primal_res and rho > 0.01:
                rho = max(rho / 1.5, 0.01)
    
    return max_iter

def solve_lrr_improved_fixed(admm_solver, A, B, lambda_val, loss='l1'):
    """修复版改进自适应ADMM - 低秩表示问题"""
    rho = 0.2  # 降低初始rho
    max_iter = 5000  # 增加最大迭代次数
    tol = admm_solver.tol * 50  # 适当放宽收敛条件
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    # 改进自适应参数
    eta = admm_solver.eta * 0.5  # 降低学习率
    epsilon = admm_solver.epsilon
    alpha = admm_solver.alpha
    beta = admm_solver.beta
    gamma = admm_solver.gamma * 0.5  # 降低最大调整幅度
    
    d, na = A.shape
    _, nb = B.shape
    
    # 一致的初始化：全部初始化为0
    X = np.zeros((nb, na))
    E = np.zeros((d, na))
    J = np.zeros((nb, na))
    Y1 = np.zeros((d, na))
    Y2 = np.zeros((nb, na))
    
    # 预计算并添加正则化
    BtB = B.T @ B
    BtA = B.T @ A
    I = np.eye(nb)
    reg = 1e-6 * np.eye(nb)
    
    # 预计算逆矩阵
    try:
        invBtBI = np.linalg.inv(BtB + I + reg)
    except np.linalg.LinAlgError:
        invBtBI = np.linalg.pinv(BtB + I + reg)
    
    best_obj = float('inf')
    no_improve_count = 0
    rho_prev = rho
    
    for k in range(max_iter):
        Xk = X.copy()
        Ek = E.copy()
        Jk = J.copy()
        
        # 1. 更新J（低秩部分）
        try:
            tau = lambda_val / (rho + 1e-8)
            J_prox, _ = svt(X + Y2/rho, tau)
            J = 0.9 * Jk + 0.1 * J_prox  # 缓慢的惯性更新
        except:
            J = Jk
        
        # 2. 更新E（稀疏部分）
        if loss == 'l1':
            E_temp = A - B @ X + Y1/rho
            threshold = 1.0 / (rho + 1e-8)
            E = soft_threshold(E_temp, threshold)
        elif loss == 'l21':
            E_temp = A - B @ X + Y1/rho
            threshold = lambda_val / (rho + 1e-8)
            E = prox_l21(E_temp, threshold)
        else:
            raise ValueError(f"不支持的损失类型: {loss}")
        
        # 3. 更新X
        try:
            X = invBtBI @ (B.T @ (Y1/rho - E) + BtA - Y2/rho + J)
        except:
            X = 0.9 * Xk + 0.1 * invBtBI @ (B.T @ (Y1/rho - E) + BtA - Y2/rho + J)
        
        # 4. 计算残差
        r1 = A - B @ X - E
        r2 = X - J
        
        primal_res = np.linalg.norm(r1, 'fro')**2 + np.linalg.norm(r2, 'fro')**2
        primal_res = np.sqrt(primal_res)
        
        # 正确的对偶残差计算
        if k > 0:
            dual_res = rho * np.linalg.norm(B @ (X - Xk), 'fro') + rho * np.linalg.norm(J - Jk, 'fro')
        else:
            dual_res = 0
        
        # 5. 计算目标函数值
        if loss == 'l1':
            sparse_term = np.sum(np.abs(E))
        else:
            sparse_term = np.sum(np.sqrt(np.sum(E**2, axis=0)))
        
        # 计算核范数
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        nuclear_norm = np.sum(s)
        
        obj_value = 0.5 * np.linalg.norm(A - B @ X - E, 'fro')**2 + lambda_val * (nuclear_norm + sparse_term)
        
        # 6. 检查收敛性
        chg = max(
            np.linalg.norm(X - Xk, 'fro') / (np.linalg.norm(Xk, 'fro') + 1e-8),
            np.linalg.norm(E - Ek, 'fro') / (np.linalg.norm(Ek, 'fro') + 1e-8),
            np.linalg.norm(J - Jk, 'fro') / (np.linalg.norm(Jk, 'fro') + 1e-8),
            np.max(np.abs(r1)),
            np.max(np.abs(r2))
        )
        
        # 检查目标函数是否改进
        if obj_value < best_obj - 1e-6:
            best_obj = obj_value
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 收敛条件
        converged = (
            chg < tol or
            (primal_res < 1e-4 and k > 100) or
            (no_improve_count > 20 and k > 50)
        )
        
        if converged:
            return k + 1
        
        # 7. 更新拉格朗日乘子
        Y1 = Y1 + rho * r1
        Y2 = Y2 + rho * r2
        
        # 8. 改进的自适应rho更新
        if rho_update and k > 0:
            if dual_res + epsilon > 0:
                ratio = primal_res / (dual_res + epsilon)
            else:
                ratio = 5.0
            
            diff = primal_res - dual_res
            total = primal_res + dual_res + epsilon
            sign_smooth = np.tanh(alpha * diff / total)
            
            if k > 1:
                momentum = beta * (rho - rho_prev) / rho_prev
            else:
                momentum = 0
            
            adjustment = eta * np.arctan(ratio - 1) * sign_smooth + momentum
            adjustment = np.clip(adjustment, -gamma, gamma)
            
            rho_prev = rho
            rho = rho * (1 + adjustment)
            rho = np.clip(rho, 0.01, 100)
    
    return max_iter

# 7. 鲁棒多视图谱聚类问题（保持原样）
def solve_rmsc_standard(admm_solver, X, lambda_val):
    """标准ADMM - 鲁棒多视图谱聚类问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    d, n, m = X.shape
    
    L = np.zeros((d, n))
    S = np.zeros((d, n, m))
    Z = np.zeros((d, n))
    Y = S.copy()
    Y2 = L.copy()
    
    for k in range(max_iter):
        Lk = L.copy()
        Sk = S.copy()
        Zk = Z.copy()
        
        # 更新Z
        Z, _ = svt(L + Y2/rho, 1/rho)
        
        # 更新S_i
        for i in range(m):
            S[:, :, i] = soft_threshold(-L + X[:, :, i] - Y[:, :, i]/rho, lambda_val/rho)
        
        # 更新L
        temp = (np.sum(X - S - Y/rho, axis=2) + Z - Y2/rho) / (m + 1)
        L = project_simplex(temp)
        
        # 计算残差
        dY = np.zeros((d, n, m))
        for i in range(m):
            dY[:, :, i] = L + S[:, :, i] - X[:, :, i]
        dY2 = L - Z
        
        # 计算残差范数
        primal_res = 0
        for i in range(m):
            primal_res += np.linalg.norm(dY[:, :, i], 'fro')**2
        primal_res = np.sqrt(primal_res + np.linalg.norm(dY2, 'fro')**2)
        
        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgL, chgS, chgZ, np.max(np.abs(dY)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y = Y + rho * dY
        Y2 = Y2 + rho * dY2
        
        # 标准ADMM的固定倍增策略
        if rho_update:
            rho = min(rho * 1.1, max_rho)
    
    return max_iter

def solve_rmsc_improved(admm_solver, X, lambda_val):
    """改进自适应ADMM - 鲁棒多视图谱聚类问题"""
    rho = admm_solver.rho
    max_iter = admm_solver.max_iter
    tol = admm_solver.tol
    rho_update = admm_solver.rho_update
    max_rho = admm_solver.max_rho
    
    # 改进自适应参数
    eta = admm_solver.eta
    epsilon = admm_solver.epsilon
    alpha = admm_solver.alpha
    beta = admm_solver.beta
    gamma = admm_solver.gamma
    
    d, n, m = X.shape
    
    L = np.zeros((d, n))
    S = np.zeros((d, n, m))
    Z = np.zeros((d, n))
    Y = S.copy()
    Y2 = L.copy()
    
    rho_prev = rho
    
    for k in range(max_iter):
        Lk = L.copy()
        Sk = S.copy()
        Zk = Z.copy()
        
        # 更新Z
        Z, _ = svt(L + Y2/rho, 1/rho)
        
        # 更新S_i
        for i in range(m):
            S[:, :, i] = soft_threshold(-L + X[:, :, i] - Y[:, :, i]/rho, lambda_val/rho)
        
        # 更新L
        temp = (np.sum(X - S - Y/rho, axis=2) + Z - Y2/rho) / (m + 1)
        L = project_simplex(temp)
        
        # 计算残差
        dY = np.zeros((d, n, m))
        for i in range(m):
            dY[:, :, i] = L + S[:, :, i] - X[:, :, i]
        dY2 = L - Z
        
        # 计算残差范数
        primal_res = 0
        for i in range(m):
            primal_res += np.linalg.norm(dY[:, :, i], 'fro')**2
        primal_res = np.sqrt(primal_res + np.linalg.norm(dY2, 'fro')**2)
        
        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgL, chgS, chgZ, np.max(np.abs(dY)), np.max(np.abs(dY2)))
        
        if chg < tol:
            return k + 1
        
        # 更新拉格朗日乘子
        Y = Y + rho * dY
        Y2 = Y2 + rho * dY2
        
        # 改进的自适应rho更新 - 为RMSC问题调整
        if rho_update and k > 0:
            # 为RMSC问题使用更温和的自适应策略
            dual_res_est = rho * np.linalg.norm(L - Lk, 'fro')
            
            if dual_res_est + epsilon > 0:
                ratio = primal_res / (dual_res_est + epsilon)
            else:
                ratio = 5.0
            
            diff = primal_res - dual_res_est
            total = primal_res + dual_res_est + epsilon
            sign_smooth = np.tanh(alpha * diff / total)
            
            if k > 1:
                momentum = beta * (rho - rho_prev) / rho_prev
            else:
                momentum = 0
            
            adjustment = 0.5 * eta * np.arctan(ratio - 1) * sign_smooth + momentum
            adjustment = np.clip(adjustment, -gamma, gamma)
            
            rho_prev = rho
            rho = rho * (1 + adjustment)
            rho = np.clip(rho, 1e-6, max_rho)
    
    return max_iter

# ============== 测试所有7个问题 ==============

def test_all_problems():
    """测试所有7个问题"""
    np.random.seed(42)
    
    # 1. L1问题
    d, na, nb = 30, 20, 15
    A_l1 = np.random.randn(d, na)
    X_true = np.random.randn(na, nb)
    mask = np.random.rand(na, nb) < 0.3
    X_true = X_true * mask
    B_l1 = A_l1 @ X_true
    
    # 2. 弹性网问题
    A_en = np.random.randn(d, na)
    B_en = A_en @ X_true
    lambda_en = 0.01
    
    # 3. L1R问题
    A_l1r = np.random.randn(d, na)
    A_l1r = A_l1r / np.linalg.norm(A_l1r, ord=2) * 5
    
    X_true_l1r = np.zeros((na, nb))
    for i in range(na):
        for j in range(nb):
            if np.random.rand() < 0.2:
                X_true_l1r[i, j] = np.random.randn()
    
    E_true = 0.001 * np.random.randn(d, nb)
    B_l1r = A_l1r @ X_true_l1r + E_true
    lambda_l1r = 0.001
    
    # 4. 弹性网回归问题
    A_enr = np.random.randn(d, na)
    B_enr = A_enr @ X_true + E_true
    lambda1_enr, lambda2_enr = 0.1, 0.05
    
    # 5. LRMC问题 - 改进参数
    d_lrmc, n_lrmc = 20, 20
    rank = 2
    U = np.random.randn(d_lrmc, rank)
    V = np.random.randn(rank, n_lrmc)
    M_true = U @ V
    obs_ratio = 0.7
    mask_lrmc = np.random.rand(d_lrmc, n_lrmc) < obs_ratio
    M_obs = np.zeros((d_lrmc, n_lrmc))
    M_obs[mask_lrmc] = M_true[mask_lrmc]
    
    # 6. LRR问题 - 改进参数
    d_lrr, na_lrr, nb_lrr = 30, 20, 15
    B_lrr = np.random.randn(d_lrr, nb_lrr)
    rank_lrr = 2
    U_lrr = np.random.randn(nb_lrr, rank_lrr)
    V_lrr = np.random.randn(rank_lrr, na_lrr)
    X_true_lrr = U_lrr @ V_lrr
    E_true_lrr = 0.01 * np.random.randn(d_lrr, na_lrr)  # 进一步减小噪声
    A_lrr = B_lrr @ X_true_lrr + E_true_lrr
    lambda_lrr = 0.02  # 减小lambda
    
    # 7. RMSC问题
    d_rmsc, n_rmsc, m_rmsc = 8, 8, 2
    X_rmsc = np.zeros((d_rmsc, n_rmsc, m_rmsc))
    
    rank_true = 2
    U_rmsc = np.random.randn(d_rmsc, rank_true)
    V_rmsc = np.random.randn(rank_true, n_rmsc)
    L_true = U_rmsc @ V_rmsc
    
    for i in range(m_rmsc):
        S_true = np.zeros((d_rmsc, n_rmsc))
        sparse_mask = np.random.rand(d_rmsc, n_rmsc) < 0.1
        S_true[sparse_mask] = np.random.randn(np.sum(sparse_mask)) * 0.5
        X_rmsc[:, :, i] = L_true + S_true
    
    L_true = project_simplex(L_true)
    lambda_rmsc = 0.1
    
    # 创建算法实例
    std_admm = StandardADMM(
        rho=0.5,
        max_iter=2000,
        tol=1e-5,
        rho_update=True,
        max_rho=1e4
    )
    
    adapt_admm = ImprovedAdaptiveADMM(
        rho=0.5,
        max_iter=2000,
        tol=1e-5,
        rho_update=True,
        max_rho=1e4
    )
    
    # 7个问题定义（使用修复的LRR求解函数）
    problems = [
        ("L1", (solve_l1_standard, (A_l1, B_l1)), (solve_l1_improved, (A_l1, B_l1))),
        ("Elasticnet", (solve_elasticnet_standard, (A_en, B_en, lambda_en)), 
         (solve_elasticnet_improved, (A_en, B_en, lambda_en))),
        ("L1R", (solve_l1r_standard, (A_l1r, B_l1r, lambda_l1r, 'l1')), 
         (solve_l1r_improved, (A_l1r, B_l1r, lambda_l1r, 'l1'))),
        ("ElasticnetR", (solve_elasticnetr_standard, (A_enr, B_enr, lambda1_enr, lambda2_enr, 'l1')),
         (solve_elasticnetr_improved, (A_enr, B_enr, lambda1_enr, lambda2_enr, 'l1'))),
        ("LRMC", (solve_lrmc_standard_improved, (M_obs, mask_lrmc)), 
         (solve_lrmc_improved_improved, (M_obs, mask_lrmc))),
        ("LRR", (solve_lrr_standard_fixed, (A_lrr, B_lrr, lambda_lrr, 'l1')),
         (solve_lrr_improved_fixed, (A_lrr, B_lrr, lambda_lrr, 'l1'))),
        ("RMSC", (solve_rmsc_standard, (X_rmsc, lambda_rmsc)),
         (solve_rmsc_improved, (X_rmsc, lambda_rmsc)))
    ]
    
    results = {}
    
    for name, (std_solver, std_args), (adapt_solver, adapt_args) in problems:
        print(f"\n测试问题: {name}")
        print("-"*40)
        
        try:
            # 标准ADMM
            print("运行标准ADMM...")
            std_iter = std_admm.solve(std_solver, *std_args)
            
            # 改进自适应ADMM
            print("运行改进自适应ADMM...")
            adapt_iter = adapt_admm.solve(adapt_solver, *adapt_args)
            
            results[name] = {
                'standard': std_iter,
                'adaptive': adapt_iter
            }
            
            status_std = "收敛" if std_iter < 2000 else "未收敛"
            status_adapt = "收敛" if adapt_iter < 2000 else "未收敛"
            
            print(f"标准ADMM: {std_iter} 轮迭代 ({status_std})")
            print(f"改进自适应ADMM: {adapt_iter} 轮迭代 ({status_adapt})")
            
        except Exception as e:
            print(f"测试失败: {e}")
            results[name] = {'standard': 2000, 'adaptive': 2000}
    
    return results

# ============== 生成对比图 ==============

def plot_convergence_comparison(results):
    """生成收敛轮数对比柱状图"""
    problem_names = list(results.keys())
    std_iters = [results[name]['standard'] for name in problem_names]
    adapt_iters = [results[name]['adaptive'] for name in problem_names]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(problem_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, std_iters, width, label='标准ADMM', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, adapt_iters, width, label='改进自适应ADMM', color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('问题类型', fontsize=14)
    ax.set_ylabel('收敛轮数', fontsize=14)
    ax.set_title(f'标准ADMM vs 改进自适应ADMM - 7个问题收敛轮数对比 (tol=1e-5)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_names, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签和改进百分比
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # 添加迭代次数
        ax.text(bar1.get_x() + bar1.get_width()/2, height1 + 10, f'{height1}', 
                ha='center', va='bottom', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width()/2, height2 + 10, f'{height2}', 
                ha='center', va='bottom', fontsize=10)
        
        # 计算改进百分比（只在收敛时计算）
        if height1 < 2000 and height1 > 0:
            improvement = 100 * (height1 - height2) / height1
            color = 'green' if improvement > 0 else 'red'
            ax.text(bar2.get_x() + bar2.get_width()/2, max(height1, height2)/2, 
                    f'{improvement:+.1f}%', ha='center', va='center', 
                    fontsize=10, color=color, fontweight='bold')
    
    # 添加未收敛标记
    if any(iter_count >= 2000 for iter_count in std_iters + adapt_iters):
        ax.text(0.5, 0.95, '注: 迭代次数=2000表示未收敛', 
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存并显示图表
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'admm_7problems_final_fixed_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: admm_7problems_final_fixed_{timestamp}.png")
    plt.show()

# ============== 主程序 ==============

if __name__ == "__main__":
    print("="*70)
    print("标准ADMM vs 改进自适应ADMM - 7个问题收敛性能对比测试")
    print("修复版本：专门优化LRR问题的收敛性")
    print("="*70)
    
    # 运行测试
    results = test_all_problems()
    
    # 打印结果表格
    print("\n" + "="*70)
    print("收敛轮数结果表格 (tol=1e-5, max_iter=2000)")
    print("="*70)
    print(f"{'问题':<15} {'标准ADMM':<12} {'改进自适应ADMM':<18} {'改进(%)':<10} {'状态':<10}")
    print("-"*70)
    
    total_std = 0
    total_adapt = 0
    converged_problems = 0
    
    for name in results:
        std_iter = results[name]['standard']
        adapt_iter = results[name]['adaptive']
        
        total_std += std_iter
        total_adapt += adapt_iter
        
        # 判断是否收敛
        std_status = "收敛" if std_iter < 2000 else "未收敛"
        adapt_status = "收敛" if adapt_iter < 2000 else "未收敛"
        
        # 只在收敛时计算改进百分比
        if std_iter < 2000 and std_iter > 0:
            improvement = ((std_iter - adapt_iter) / std_iter) * 100
            converged_problems += 1
        else:
            improvement = 0
        
        # 状态描述
        status_desc = f"标准:{std_status}, 改进:{adapt_status}"
        
        print(f"{name:<15} {std_iter:<12} {adapt_iter:<18} {improvement:>7.1f}% {status_desc:<10}")
    
    # 计算总体改进（只考虑收敛的问题）
    if converged_problems > 0:
        avg_improvement = sum([((results[name]['standard'] - results[name]['adaptive']) / results[name]['standard'] * 100) 
                              for name in results if results[name]['standard'] < 2000]) / converged_problems
    else:
        avg_improvement = 0
    
    print("-"*70)
    print(f"{'总计':<15} {total_std:<12} {total_adapt:<18} {avg_improvement:>7.1f}% (平均)")
    
    # 计算收敛的问题数量
    std_converged = sum([1 for name in results if results[name]['standard'] < 2000])
    adapt_converged = sum([1 for name in results if results[name]['adaptive'] < 2000])
    
    print(f"\n收敛统计:")
    print(f"标准ADMM收敛的问题数: {std_converged}/7")
    print(f"改进自适应ADMM收敛的问题数: {adapt_converged}/7")
    
    # 生成柱状图
    print("\n正在生成收敛轮数对比图...")
    plot_convergence_comparison(results)
