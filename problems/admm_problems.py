"""ADMM测试问题集合 - 基于7个MATLAB ADMM求解器代码实现"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .base_problem import ADMMProblemBase

# ============== 辅助函数 ==============

def soft_threshold(x, alpha):
    """L1范数近端算子"""
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def svt(X, tau):
    """改进的奇异值阈值函数 - 用于核范数正则化，增加稳定性"""
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
    """L2,1范数近端算子 - 用于行稀疏正则化"""
    col_norms = np.sqrt(np.sum(X**2, axis=0, keepdims=True))
    col_norms = np.where(col_norms < 1e-10, 1e-10, col_norms)
    scale = np.maximum(1.0 - alpha / col_norms, 0.0)
    return X * scale

def project_simplex_vector(v):
    """将单个向量投影到概率单纯形"""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * (1 + np.arange(1, n+1)) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def project_simplex_columns(L):
    """对矩阵的每一列进行概率单纯形投影"""
    d, n = L.shape
    L_proj = np.zeros_like(L)
    for j in range(n):
        L_proj[:, j] = project_simplex_vector(L[:, j])
    return L_proj

# ============== 7个ADMM问题实现 ==============

class L1MinimizationProblem(ADMMProblemBase):
    def __init__(self, d: int = 50, na: int = 40, nb: int = 30, 
                 sparsity: float = 0.3, seed: int = 42):
        super().__init__("l1_minimization")
        self.d = d
        self.na = na
        self.nb = nb
        self.sparsity = sparsity
        self.seed = seed
        self.reset(seed)
    
    def _generate_data(self):
        """生成问题数据"""
        # 生成矩阵A
        self.data['A'] = np.random.randn(self.d, self.na)
        
        # 生成稀疏的真实解
        X_true = np.random.randn(self.na, self.nb)
        mask = np.random.rand(self.na, self.nb) < self.sparsity
        X_true = X_true * mask
        
        self.data['B'] = self.data['A'] @ X_true
        self.data['X_true'] = X_true
    
    def initialize_variables(self):
        """初始化ADMM变量"""
        A = self.data['A']
        B = self.data['B']
        d, na = A.shape
        _, nb = B.shape
        
        self.variables = {
            'X': np.zeros((na, nb)),
            'Z': np.zeros((na, nb)),
            'Y1': np.zeros((d, nb)),
            'Y2': np.zeros((na, nb))
        }
        
        # 预计算
        self.data['AtB'] = A.T @ B
        I = np.eye(na)
        self.data['invAtAI'] = np.linalg.inv(A.T @ A + I) @ I
    
    def compute_objective(self) -> float:
        """计算L1范数目标函数值"""
        X = self.variables['X']
        return np.sum(np.abs(X))
    
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        objective = self.compute_objective()
        X = self.variables['X']
        Z = self.variables['Z']
        
        return {
            'objective': objective,
            'X_norm': np.linalg.norm(X, 'fro'),
            'Z_norm': np.linalg.norm(Z, 'fro'),
            'sparsity': np.mean(np.abs(X) < 1e-6),
            'converged': self.converged
        }
    
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """执行一次ADMM更新"""
        A = self.data['A']
        B = self.data['B']
        invAtAI = self.data['invAtAI']
        AtB = self.data['AtB']
        
        X = self.variables['X']
        Z = self.variables['Z']
        Y1 = self.variables['Y1']
        Y2 = self.variables['Y2']
        
        X_old = X.copy()
        Z_old = Z.copy()
        
        # 更新X (L1范数近端算子)
        X = soft_threshold(Z - Y2/beta, 1/beta)
        
        # 更新Z
        Z = invAtAI @ (-A.T @ Y1/beta + AtB + Y2/beta + X)
        
        # 计算残差
        dY1 = A @ Z - B
        dY2 = X - Z
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + 
                           np.linalg.norm(dY2, 'fro')**2)
        dual_res = beta * np.linalg.norm(A.T @ A @ (Z - Z_old), 'fro')
        
        # 检查收敛 - 统一使用原始残差和对偶残差
        tol = 1e-5
        converged = (primal_res < tol) and (dual_res < tol)
        
        # 更新拉格朗日乘子
        Y1 = Y1 + beta * dY1
        Y2 = Y2 + beta * dY2
        
        # 更新变量
        self.variables['X'] = X
        self.variables['Z'] = Z
        self.variables['Y1'] = Y1
        self.variables['Y2'] = Y2
        
        # 返回迭代结果
        return {
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'objective': self.compute_objective(),
            'converged': converged
        }

class ElasticNetProblem(ADMMProblemBase):
    """
    弹性网问题: min_X ||X||₁ + λ||X||_F², s.t. AX = B
    基于MATLAB代码: elasticnet.m
    """
    
    def __init__(self, d: int = 50, na: int = 40, nb: int = 30,
                 sparsity: float = 0.3, lambda_val: float = 0.1, seed: int = 42):
        super().__init__("elastic_net")
        self.d = d
        self.na = na
        self.nb = nb
        self.sparsity = sparsity
        self.lambda_val = lambda_val
        self.seed = seed
        self.reset(seed)
    
    def _generate_data(self):
        """生成问题数据"""
        # 生成矩阵A
        self.data['A'] = np.random.randn(self.d, self.na)
        
        # 生成稀疏的真实解
        X_true = np.random.randn(self.na, self.nb)
        mask = np.random.rand(self.na, self.nb) < self.sparsity
        X_true = X_true * mask
        
        self.data['B'] = self.data['A'] @ X_true
        self.data['X_true'] = X_true
    
    def initialize_variables(self):
        """初始化变量"""
        A = self.data['A']
        B = self.data['B']
        d, na = A.shape
        _, nb = B.shape
        
        self.variables = {
            'X': np.zeros((na, nb)),
            'Z': np.zeros((na, nb)),
            'Y1': np.zeros((d, nb)),
            'Y2': np.zeros((na, nb))
        }
        
        # 预计算
        self.data['AtB'] = A.T @ B
        I = np.eye(na)
        self.data['invAtAI'] = np.linalg.inv(A.T @ A + I) @ I
    
    def compute_objective(self) -> float:
        """计算目标函数值"""
        X = self.variables['X']
        return np.sum(np.abs(X)) + self.lambda_val * np.sum(X**2)
    
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        objective = self.compute_objective()
        X = self.variables['X']
        
        return {
            'objective': objective,
            'X_norm': np.linalg.norm(X, 'fro'),
            'l1_term': np.sum(np.abs(X)),
            'l2_term': self.lambda_val * np.sum(X**2),
            'sparsity': np.mean(np.abs(X) < 1e-6),
            'converged': self.converged
        }
    
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """执行一次ADMM更新"""
        A = self.data['A']
        B = self.data['B']
        invAtAI = self.data['invAtAI']
        AtB = self.data['AtB']
        
        X = self.variables['X']
        Z = self.variables['Z']
        Y1 = self.variables['Y1']
        Y2 = self.variables['Y2']
        
        X_old = X.copy()
        Z_old = Z.copy()
        
        # 更新X (弹性网近端算子)
        temp = Z - Y2/beta
        factor = 1.0 / (1.0 + 2.0 * self.lambda_val / beta)
        X = factor * soft_threshold(temp, 1/beta)
        
        # 更新Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2)/beta + AtB + X)
        
        # 计算残差
        dY1 = A @ Z - B
        dY2 = X - Z
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        dual_res = beta * np.linalg.norm(A.T @ A @ (Z - Z_old), 'fro')
        
        # 检查收敛 - 统一使用原始残差和对偶残差
        tol = 1e-5
        converged = (primal_res < tol) and (dual_res < tol)
        
        # 更新拉格朗日乘子
        Y1 = Y1 + beta * dY1
        Y2 = Y2 + beta * dY2
        
        # 更新变量
        self.variables['X'] = X
        self.variables['Z'] = Z
        self.variables['Y1'] = Y1
        self.variables['Y2'] = Y2
        
        # 返回迭代结果
        return {
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'objective': self.compute_objective(),
            'converged': converged
        }

class L1RegularizedRegressionProblem(ADMMProblemBase):
    """
    L1正则化回归问题: min_X λ||X||₁ + L(AX - B)
    基于MATLAB代码: l1R.m
    
    ADMM分解:
    min_{X,E,Z} λ||X||₁ + ||E||₁
    s.t. AZ + E = B, X = Z
    
    数学逻辑正确，收敛条件已优化
    """
    
    def __init__(self, d: int = 50, na: int = 40, nb: int = 30,
                 sparsity: float = 0.3, lambda_val: float = 0.1,
                 noise_level: float = 0.1, loss_type: str = 'l1', seed: int = 42):
        super().__init__("l1_regularized_regression")
        self.d = d
        self.na = na
        self.nb = nb
        self.sparsity = sparsity
        self.lambda_val = lambda_val
        self.noise_level = noise_level
        self.loss_type = loss_type
        self.seed = seed
        self.reset(seed)
    
    def _generate_data(self):
        """生成问题数据"""
        # 生成矩阵A
        self.data['A'] = np.random.randn(self.d, self.na)
        
        # 生成稀疏的真实解
        X_true = np.random.randn(self.na, self.nb)
        mask = np.random.rand(self.na, self.nb) < self.sparsity
        X_true = X_true * mask
        
        # 添加噪声
        E_true = self.noise_level * np.random.randn(self.d, self.nb)
        self.data['B'] = self.data['A'] @ X_true + E_true
        self.data['X_true'] = X_true
        self.data['E_true'] = E_true
    
    def initialize_variables(self):
        """初始化变量"""
        A = self.data['A']
        B = self.data['B']
        d, na = A.shape
        _, nb = B.shape
        
        self.variables = {
            'X': np.zeros((na, nb)),
            'E': np.zeros((d, nb)),
            'Z': np.zeros((na, nb)),
            'Y1': np.zeros((d, nb)),
            'Y2': np.zeros((na, nb))
        }
        
        # 预计算
        self.data['AtB'] = A.T @ B
        I = np.eye(na)
        self.data['invAtAI'] = np.linalg.inv(A.T @ A + I)
    
    def compute_objective(self) -> float:
        """计算目标函数值"""
        X = self.variables['X']
        E = self.variables['E']
        
        l1_term = self.lambda_val * np.sum(np.abs(X))
        
        if self.loss_type == 'l1':
            loss_term = np.sum(np.abs(E))
        elif self.loss_type == 'l2':
            loss_term = 0.5 * np.sum(E**2)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
        
        return l1_term + loss_term
    
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        objective = self.compute_objective()
        X = self.variables['X']
        E = self.variables['E']
        
        return {
            'objective': objective,
            'X_norm': np.linalg.norm(X, 'fro'),
            'E_norm': np.linalg.norm(E, 'fro'),
            'l1_term': self.lambda_val * np.sum(np.abs(X)),
            'loss_term': np.sum(np.abs(E)) if self.loss_type == 'l1' else 0.5 * np.sum(E**2),
            'sparsity': np.mean(np.abs(X) < 1e-6),
            'converged': self.converged
        }
    
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """执行一次ADMM更新
        
        数学推导:
        - X更新: prox_{(λ/beta)|||·||_1}(Z - Y2/beta)
        - E更新: prox_{(1/beta)|||·||_1}(B - AZ - Y1/beta) (L1损失)
        - Z更新: (A^TA + I)^{-1}(A^T(B - E - Y1/beta) + X + Y2/beta)
        
        beta由外部策略控制
        """
        A = self.data['A']
        B = self.data['B']
        invAtAI = self.data['invAtAI']
        AtB = self.data['AtB']
        
        # 使用外部传入的beta（由策略控制）
        rho = beta
        
        X = self.variables['X']
        E = self.variables['E']
        Z = self.variables['Z']
        Y1 = self.variables['Y1']
        Y2 = self.variables['Y2']
        
        X_old = X.copy()
        E_old = E.copy()
        Z_old = Z.copy()
        
        # 更新X (L1范数近端算子)
        X = soft_threshold(Z - Y2/rho, self.lambda_val/rho)
        
        # 更新E (根据损失函数类型)
        if self.loss_type == 'l1':
            E = soft_threshold(B - A @ Z - Y1/rho, 1/rho)
        elif self.loss_type == 'l2':
            E = rho * (B - A @ Z - Y1/rho) / (1 + rho)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
        
        # 更新Z
        Z = invAtAI @ (-A.T @ (Y1/rho + E) + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z + E - B
        dY2 = X - Z
        
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        dual_res = rho * np.linalg.norm(A.T @ A @ (Z - Z_old), 'fro')
        
        # 检查收敛 - 统一使用原始残差和对偶残差
        tol = 1e-5
        converged = (primal_res < tol) and (dual_res < tol)
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 更新变量
        self.variables['X'] = X
        self.variables['E'] = E
        self.variables['Z'] = Z
        self.variables['Y1'] = Y1
        self.variables['Y2'] = Y2
        
        # 返回迭代结果
        return {
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'objective': self.compute_objective(),
            'converged': converged
        }

class ElasticNetRegressionProblem(ADMMProblemBase):
    """
    弹性网回归问题: min_X λ1||X||₁ + λ2||X||_F² + L(AX - B)
    基于MATLAB代码: elasticnetR.m
    
    ADMM分解:
    min_{X,E,Z} λ1||X||₁ + λ2||X||_F² + ||E||₁
    s.t. AZ + E = B, X = Z
    
    数学逻辑正确：
    - X更新使用弹性网近端算子: (1/(1+2λ2/β)) * S_{λ1/β}(·)
    - E更新使用L1近端算子
    - Z更新是二次问题的解析解
    """
    
    def __init__(self, d: int = 50, na: int = 40, nb: int = 30,
                 sparsity: float = 0.3, lambda1: float = 0.1, lambda2: float = 0.05,
                 noise_level: float = 0.1, loss_type: str = 'l1', seed: int = 42):
        super().__init__("elastic_net_regression")
        self.d = d
        self.na = na
        self.nb = nb
        self.sparsity = sparsity
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.noise_level = noise_level
        self.loss_type = loss_type
        self.seed = seed
        self.reset(seed)
    
    def _generate_data(self):
        """生成问题数据"""
        # 生成矩阵A
        self.data['A'] = np.random.randn(self.d, self.na)
        
        # 生成稀疏的真实解
        X_true = np.random.randn(self.na, self.nb)
        mask = np.random.rand(self.na, self.nb) < self.sparsity
        X_true = X_true * mask
        
        # 添加噪声
        E_true = self.noise_level * np.random.randn(self.d, self.nb)
        self.data['B'] = self.data['A'] @ X_true + E_true
        self.data['X_true'] = X_true
        self.data['E_true'] = E_true
    
    def initialize_variables(self):
        """初始化变量"""
        A = self.data['A']
        B = self.data['B']
        d, na = A.shape
        _, nb = B.shape
        
        self.variables = {
            'X': np.zeros((na, nb)),
            'E': np.zeros((d, nb)),
            'Z': np.zeros((na, nb)),
            'Y1': np.zeros((d, nb)),
            'Y2': np.zeros((na, nb))
        }
        
        # 预计算
        self.data['AtB'] = A.T @ B
        I = np.eye(na)
        self.data['invAtAI'] = np.linalg.inv(A.T @ A + I)
    
    def compute_objective(self) -> float:
        """计算目标函数值"""
        X = self.variables['X']
        E = self.variables['E']
        
        l1_term = self.lambda1 * np.sum(np.abs(X))
        l2_term = self.lambda2 * np.sum(X**2)
        
        if self.loss_type == 'l1':
            loss_term = np.sum(np.abs(E))
        elif self.loss_type == 'l2':
            loss_term = 0.5 * np.sum(E**2)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
        
        return l1_term + l2_term + loss_term
    
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        objective = self.compute_objective()
        X = self.variables['X']
        E = self.variables['E']
        
        return {
            'objective': objective,
            'X_norm': np.linalg.norm(X, 'fro'),
            'E_norm': np.linalg.norm(E, 'fro'),
            'l1_term': self.lambda1 * np.sum(np.abs(X)),
            'l2_term': self.lambda2 * np.sum(X**2),
            'loss_term': np.sum(np.abs(E)) if self.loss_type == 'l1' else 0.5 * np.sum(E**2),
            'sparsity': np.mean(np.abs(X) < 1e-6),
            'converged': self.converged
        }
    
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """执行一次ADMM更新
        
        数学推导:
        - X更新: (1/(1+2λ2/beta)) * prox_{(λ1/beta)|||·||_1}(Z - Y2/beta)
          这是弹性网近端算子的标准形式
        - E更新: prox_{(1/beta)|||·||_1}(B - AZ - Y1/beta)
        - Z更新: (A^TA + I)^{-1}(A^T(B - E - Y1/beta) + X + Y2/beta)
        
        beta由外部策略控制
        """
        A = self.data['A']
        B = self.data['B']
        invAtAI = self.data['invAtAI']
        AtB = self.data['AtB']
        
        # 使用外部传入的beta（由策略控制）
        rho = beta
        
        X = self.variables['X']
        E = self.variables['E']
        Z = self.variables['Z']
        Y1 = self.variables['Y1']
        Y2 = self.variables['Y2']
        
        X_old = X.copy()
        E_old = E.copy()
        Z_old = Z.copy()
        
        # 更新X (弹性网近端算子) - 按照112(1).py: prox_elasticnet(Z - Y2/rho, lambda1/rho, lambda2/rho)
        temp = Z - Y2/rho
        factor = 1.0 / (1.0 + 2.0 * self.lambda2 / rho)
        X = factor * soft_threshold(temp, self.lambda1/rho)
        
        # 更新E (根据损失函数类型)
        if self.loss_type == 'l1':
            E = soft_threshold(B - A @ Z - Y1/rho, 1/rho)
        elif self.loss_type == 'l2':
            E = rho * (B - A @ Z - Y1/rho) / (1 + rho)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
        
        # 更新Z
        Z = invAtAI @ (-A.T @ (Y1/rho + E) + AtB + Y2/rho + X)
        
        # 计算残差
        dY1 = A @ Z + E - B
        dY2 = X - Z
        
        primal_res = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        dual_res = rho * np.linalg.norm(A.T @ A @ (Z - Z_old), 'fro')
        
        # 检查收敛 - 统一使用原始残差和对偶残差
        tol = 1e-5
        converged = (primal_res < tol) and (dual_res < tol)
        
        # 更新拉格朗日乘子
        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        
        # 更新变量
        self.variables['X'] = X
        self.variables['E'] = E
        self.variables['Z'] = Z
        self.variables['Y1'] = Y1
        self.variables['Y2'] = Y2
        
        # 返回迭代结果
        return {
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'objective': self.compute_objective(),
            'converged': converged
        }

class LowRankMatrixCompletionProblem(ADMMProblemBase):
    """
    低秩矩阵补全问题: min_X ||X||_* + λ||P_Ω(X - M)||_1
    基于MATLAB代码: lrmc.m
    
    ADMM分解:
    min_{X,E} ||X||_* + λ||E||_1
    s.t. P_Ω(X - M) = E (只在观测位置)
    
    其中E只在观测位置有定义，表示稀疏器差
    """
    
    def __init__(self, d: int = 30, n: int = 30, rank: int = 5,
                 observation_ratio: float = 0.5, lambda_val: float = 0.1, seed: int = 42):
        super().__init__("low_rank_matrix_completion")
        self.d = d
        self.n = n
        self.rank = rank
        self.observation_ratio = observation_ratio
        self.lambda_val = lambda_val  # 稀疏误差的权重
        self.seed = seed
        self.reset(seed)
    
    def _generate_data(self):
        """生成问题数据"""
        # 生成低秩矩阵
        U = np.random.randn(self.d, self.rank)
        V = np.random.randn(self.rank, self.n)
        self.data['M_true'] = U @ V
        
        # 创建观测掩码
        self.data['mask'] = np.random.rand(self.d, self.n) < self.observation_ratio
        self.data['M_obs'] = np.zeros((self.d, self.n))
        self.data['M_obs'][self.data['mask']] = self.data['M_true'][self.data['mask']]
    
    def initialize_variables(self):
        """初始化变量"""
        M_obs = self.data['M_obs']
        mask = self.data['mask']
        d, n = M_obs.shape
        
        # 初始化X为观测值（在观测位置）
        X = np.zeros((d, n))
        X[mask] = M_obs[mask]
        
        # E只在观测位置有意义，但我们存储为完整矩阵
        self.variables = {
            'X': X,
            'E': np.zeros((d, n)),  # E[mask]表示观测位置的误差
            'Y': np.zeros((d, n))   # 只在观测位置有意义的对偶变量
        }
        
        # 跟踪最佳误差和无改进计数
        self._best_error = float('inf')
        self._no_improve_count = 0
    
    def compute_objective(self) -> float:
        """计算目标函数值"""
        X = self.variables['X']
        E = self.variables['E']
        mask = self.data['mask']
        
        # 计算核范数
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        nuclear_norm = np.sum(s)
        
        # 计算观测误差的L1范数 (只考虑观测位置)
        l1_norm = self.lambda_val * np.sum(np.abs(E[mask]))
        
        return nuclear_norm + l1_norm
    
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        objective = self.compute_objective()
        X = self.variables['X']
        
        # 计算矩阵的秩
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        rank = np.sum(s > 1e-6)
        
        # 计算重构误差
        M_true = self.data['M_true']
        reconstruction_error = np.linalg.norm(X - M_true, 'fro') / (np.linalg.norm(M_true, 'fro') + 1e-8)
        
        return {
            'objective': objective,
            'X_norm': np.linalg.norm(X, 'fro'),
            'rank': rank,
            'reconstruction_error': reconstruction_error,
            'converged': self.converged
        }
    
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """执行一次ADMM更新 - 修复版本
        
        问题: min_X ||X||_* + λ||P_Ω(X - M)||_1
        
        ADMM分解 (E只在观测位置有定义):
        min_{X,E} ||X||_* + λ||E||_1
        s.t. P_Ω(X) - P_Ω(M) = E
        
        更新步骤:
        1. X更新: prox_{(1/β)|||·||_*}(·) 考虑观测位置的约束
        2. E更新: prox_{(λ/β)|||·||_1}(P_Ω(X) - P_Ω(M) + Y/β)
        3. Y更新: Y = Y + β * (P_Ω(X) - P_Ω(M) - E)
        """
        M_obs = self.data['M_obs']
        mask = self.data['mask']
        
        X = self.variables['X']
        E = self.variables['E']
        Y = self.variables['Y']
        
        X_old = X.copy()
        E_old = E.copy()
        
        # 1. 更新X (核范数近端算子)
        # X子问题: min_X ||X||_* + (β/2)||P_Ω(X) - (P_Ω(M) + E - Y/β)||_F²
        # 这是一个带观测约束的核范数最小化问题
        # 近似解: 先在观测位置设置目标值，然后做SVT
        
        # 构建目标矩阵: 在观测位置使用 M_obs + E - Y/β
        target = X_old.copy()
        target[mask] = M_obs[mask] + E[mask] - Y[mask]/beta
        
        # SVT操作
        try:
            X_prox, _ = svt(target, 1.0/beta)
            # 惯性项提高稳定性
            X = 0.7 * X_old + 0.3 * X_prox
        except:
            X = X_old
        
        # 2. 更新E (只在观测位置，L1范数近端算子)
        # E子问题: min_E λ||E||_1 + (β/2)||E - (P_Ω(X) - P_Ω(M) + Y/β)||_F²
        # 解: E = prox_{(λ/β)|||·||_1}(P_Ω(X - M) + Y/β)
        E_target = np.zeros_like(E)
        E_target[mask] = X[mask] - M_obs[mask] + Y[mask]/beta
        E = soft_threshold(E_target, self.lambda_val/beta)
        E[~mask] = 0  # 非观测位置E始终为0
        
        # 3. 计算约束违反 (只在观测位置)
        constraint_violation = np.zeros_like(X)
        constraint_violation[mask] = X[mask] - M_obs[mask] - E[mask]
        
        primal_res = np.linalg.norm(constraint_violation[mask])
        dual_res = beta * np.linalg.norm((E - E_old)[mask])
        
        # 计算相对误差
        M_obs_norm = np.linalg.norm(M_obs[mask]) + 1e-8
        relative_error = primal_res / M_obs_norm
        
        # 检查收敛 - 统一使用原始残差和对偶残差
        tol = 1e-5
        converged = (primal_res < tol) and (dual_res < tol)
        
        # 4. 更新拉格朗日乘子 (只在观测位置)
        Y[mask] = Y[mask] + beta * constraint_violation[mask]
        
        # 更新变量
        self.variables['X'] = X
        self.variables['E'] = E
        self.variables['Y'] = Y
        
        return {
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'relative_error': relative_error,
            'objective': self.compute_objective(),
            'converged': converged
        }

class LowRankRepresentationProblem(ADMMProblemBase):
    """
    低秩表示问题: min_Z ||Z||_* + λ||E||_{2,1}, s.t. A = BZ + E
    基于MATLAB代码: lrr.m
    """
    
    def __init__(self, d: int = 50, na: int = 40, nb: int = 30,
                 rank: int = 5, lambda_val: float = 0.5, 
                 loss_type: str = 'l21', noise_level: float = 0.1, seed: int = 42):
        super().__init__("low_rank_representation")
        self.d = d
        self.na = na
        self.nb = nb
        self.rank = rank
        self.lambda_val = lambda_val
        self.loss_type = loss_type
        self.noise_level = noise_level
        self.seed = seed
        self.reset(seed)
    
    def _generate_data(self):
        """生成问题数据"""
        # 生成字典B和低秩表示矩阵Z
        self.data['B'] = np.random.randn(self.d, self.nb)
        
        # 生成低秩的Z
        U = np.random.randn(self.nb, self.rank)
        V = np.random.randn(self.rank, self.na)
        self.data['Z_true'] = U @ V
        
        # 添加噪声/异常E
        if self.loss_type == 'l1':
            E_true = self.noise_level * np.random.randn(self.d, self.na)
        elif self.loss_type == 'l21':
            # 列稀疏的E
            E_true = np.zeros((self.d, self.na))
            sparse_cols = np.random.rand(self.na) < 0.3
            for i in range(self.na):
                if sparse_cols[i]:
                    E_true[:, i] = self.noise_level * np.random.randn(self.d)
        else:  # l2
            E_true = self.noise_level * np.random.randn(self.d, self.na)
        
        self.data['A'] = self.data['B'] @ self.data['Z_true'] + E_true
        self.data['E_true'] = E_true
    
    def initialize_variables(self):
        """初始化变量 - 改进版本，增加正则化"""
        A = self.data['A']
        B = self.data['B']
        d, na = A.shape
        _, nb = B.shape
        
        # 一致的初始化：全部初始化为0
        self.variables = {
            'X': np.zeros((nb, na)),  # 表示矩阵Z
            'E': np.zeros((d, na)),   # 误差/异常
            'J': np.zeros((nb, na)),  # 辅助变量用于核范数
            'Y1': np.zeros((d, na)),
            'Y2': np.zeros((nb, na))
        }
        
        # 预计算并添加正则化
        self.data['BtB'] = B.T @ B
        self.data['BtA'] = B.T @ A
        I = np.eye(nb)
        reg = 1e-6 * np.eye(nb)  # 添加正则化项
        
        # 预计算逆矩阵，使用pinv作为备选
        try:
            self.data['invBtBI'] = np.linalg.inv(self.data['BtB'] + I + reg)
        except np.linalg.LinAlgError:
            self.data['invBtBI'] = np.linalg.pinv(self.data['BtB'] + I + reg)
        
        # 跟踪最佳目标函数值和无改进计数
        self._best_obj = float('inf')
        self._no_improve_count = 0
    
    def compute_objective(self) -> float:
        """计算目标函数值"""
        X = self.variables['X']
        E = self.variables['E']
        
        # 计算核范数
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        nuclear_norm = np.sum(s)
        
        # 计算误差项
        if self.loss_type == 'l1':
            error_term = np.sum(np.abs(E))
        elif self.loss_type == 'l21':
            col_norms = np.sqrt(np.sum(E**2, axis=0))
            error_term = np.sum(col_norms)
        else:  # l2
            error_term = 0.5 * np.sum(E**2)
        
        return nuclear_norm + self.lambda_val * error_term
    
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        objective = self.compute_objective()
        X = self.variables['X']
        E = self.variables['E']
        
        # 计算矩阵的秩
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        rank = np.sum(s > 1e-6)
        
        # 计算重构误差
        A = self.data['A']
        B = self.data['B']
        reconstruction_error = np.linalg.norm(A - B @ X - E, 'fro')
        
        return {
            'objective': objective,
            'X_norm': np.linalg.norm(X, 'fro'),
            'E_norm': np.linalg.norm(E, 'fro'),
            'rank': rank,
            'reconstruction_error': reconstruction_error,
            'converged': self.converged
        }
    
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """执行一次ADMM更新 - 修复版本，增加惯性项和多重收敛条件"""
        A = self.data['A']
        B = self.data['B']
        invBtBI = self.data['invBtBI']
        BtA = self.data['BtA']
        
        X = self.variables['X']
        E = self.variables['E']
        J = self.variables['J']
        Y1 = self.variables['Y1']
        Y2 = self.variables['Y2']
        
        X_old = X.copy()
        E_old = E.copy()
        J_old = J.copy()
        
        # 1. 更新J (核范数近端算子) - 增加惯性项
        try:
            tau = self.lambda_val / (beta + 1e-8)
            J_prox, _ = svt(X + Y2/beta, tau)
            J = 0.9 * J_old + 0.1 * J_prox  # 缓慢的惯性更新
        except:
            J = J_old
        
        # 2. 更新E (根据损失函数类型)
        if self.loss_type == 'l1':
            E_temp = A - B @ X + Y1/beta
            threshold = 1.0 / (beta + 1e-8)
            E = soft_threshold(E_temp, threshold)
        elif self.loss_type == 'l21':
            E_temp = A - B @ X + Y1/beta
            threshold = self.lambda_val / (beta + 1e-8)
            E = prox_l21(E_temp, threshold)
        elif self.loss_type == 'l2':
            E = beta * (A - B @ X + Y1/beta) / (self.lambda_val + beta)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
        
        # 3. 更新X
        try:
            X = invBtBI @ (B.T @ (Y1/beta - E) + BtA - Y2/beta + J)
        except:
            X = 0.9 * X_old + 0.1 * invBtBI @ (B.T @ (Y1/beta - E) + BtA - Y2/beta + J)
        
        # 4. 计算残差
        r1 = A - B @ X - E
        r2 = X - J
        primal_res = np.sqrt(np.linalg.norm(r1, 'fro')**2 + np.linalg.norm(r2, 'fro')**2)
        
        # 正确的对偶残差计算
        if iteration > 0:
            dual_res = beta * np.linalg.norm(B @ (X - X_old), 'fro') + beta * np.linalg.norm(J - J_old, 'fro')
        else:
            dual_res = 0
        
        # 5. 检查收敛 - 统一使用原始残差和对偶残差
        tol = 1e-5
        converged = (primal_res < tol) and (dual_res < tol)
        
        # 6. 更新拉格朗日乘子
        Y1 = Y1 + beta * r1
        Y2 = Y2 + beta * r2
        
        # 更新变量
        self.variables['X'] = X
        self.variables['E'] = E
        self.variables['J'] = J
        self.variables['Y1'] = Y1
        self.variables['Y2'] = Y2
        
        # 返回迭代结果
        return {
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'objective': self.compute_objective(),
            'converged': converged
        }

class RobustMultiViewSpectralClusteringProblem(ADMMProblemBase):
    """
    鲁棒多视图谱聚类问题: min_L,S_i ∑_i ||S_i||_1 + λ||L||_*, 
                         s.t. X_i = L + S_i, L⪰0, L1=1, L≥0
    基于MATLAB代码: rmsc.m (修正版本)
    """
    
    def __init__(self, d: int = 20, n: int = 20, m: int = 3,
                 lambda_val: float = 0.2, noise_level: float = 0.1, seed: int = 42):
        super().__init__("robust_multi_view_spectral_clustering")
        self.d = d
        self.n = n
        self.m = m
        self.lambda_val = lambda_val
        self.noise_level = noise_level
        self.seed = seed
        self.reset(seed)
    
    def _generate_data(self):
        """生成问题数据"""
        # 生成共同的低秩矩阵L (概率单纯形约束)
        L_true = np.random.rand(self.d, self.n)
        # 归一化每一列到概率单纯形
        for j in range(self.n):
            L_true[:, j] = project_simplex_vector(L_true[:, j])
        
        # 生成多视图数据
        self.data['X'] = np.zeros((self.d, self.n, self.m))
        for i in range(self.m):
            # 添加稀疏异常/噪声
            S_true = np.zeros((self.d, self.n))
            sparse_mask = np.random.rand(self.d, self.n) < 0.1  # 10%稀疏异常
            S_true[sparse_mask] = self.noise_level * np.random.randn(np.sum(sparse_mask))
            
            self.data['X'][:, :, i] = L_true + S_true
        
        self.data['L_true'] = L_true
    
    def initialize_variables(self):
        """初始化变量 - 与112(1).py一致"""
        d, n, m = self.d, self.n, self.m
        
        # 与112(1).py一致：全部初始化为零
        self.variables = {
            'L': np.zeros((d, n)),
            'S': np.zeros((d, n, m)),
            'Z': np.zeros((d, n)),
            'Y': np.zeros((d, n, m)),
            'Y2': np.zeros((d, n))
        }
    
    def compute_objective(self) -> float:
        """计算目标函数值"""
        L = self.variables['L']
        S = self.variables['S']
        
        # 计算核范数
        _, s, _ = np.linalg.svd(L, full_matrices=False)
        nuclear_norm = np.sum(s)
        
        # 计算稀疏项
        sparse_term = np.sum(np.abs(S))
        
        return nuclear_norm + self.lambda_val * sparse_term
    
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        objective = self.compute_objective()
        L = self.variables['L']
        S = self.variables['S']
        
        # 计算矩阵的秩
        _, s, _ = np.linalg.svd(L, full_matrices=False)
        rank = np.sum(s > 1e-6)
        
        # 计算重构误差
        X = self.data['X']
        reconstruction_error = 0
        for i in range(self.m):
            reconstruction_error += np.linalg.norm(X[:, :, i] - L - S[:, :, i], 'fro')**2
        reconstruction_error = np.sqrt(reconstruction_error / self.m)
        
        # 检查概率单纯形约束
        simplex_violation = 0
        for j in range(self.n):
            col = L[:, j]
            simplex_violation += abs(np.sum(col) - 1) + np.sum(np.minimum(0, col))
        simplex_violation /= self.n
        
        return {
            'objective': objective,
            'L_norm': np.linalg.norm(L, 'fro'),
            'S_norm': np.linalg.norm(S, 'fro'),
            'rank': rank,
            'reconstruction_error': reconstruction_error,
            'simplex_violation': simplex_violation,
            'converged': self.converged
        }
    
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """执行一次ADMM更新
        
        问题形式: min_{L,S_i} Σ_i ||S_i||_1 + λ||L||_*
                  s.t. X_i = L + S_i, L ∈ 概率单纯形
        
        更新步骤:
        - Z更新: svt(L + Y2/beta, 1/beta)
        - S更新: soft_threshold(-L + X[:, :, i] - Y[:, :, i]/beta, lambda_val/beta)
        
        beta由外部策略控制
        """
        X = self.data['X']
        
        # 使用外部传入的beta（由策略控制）
        rho = beta
        
        L = self.variables['L']
        S = self.variables['S']
        Z = self.variables['Z']
        Y = self.variables['Y']
        Y2 = self.variables['Y2']
        
        L_old = L.copy()
        S_old = S.copy()
        Z_old = Z.copy()
        
        # 1. 更新Z (核范数近端算子) - 按照112(1).py
        # 阈值是 1/rho
        Z, _ = svt(L + Y2/rho, 1.0/rho)
        
        # 2. 更新每个视图的S_i (L1范数近端算子) - 按照112(1).py
        # 阈值是 lambda_val/rho
        for i in range(self.m):
            S[:, :, i] = soft_threshold(-L + X[:, :, i] - Y[:, :, i]/rho, self.lambda_val/rho)
        
        # 3. 更新L (带概率单纯形约束)
        temp = (np.sum(X - S - Y/rho, axis=2) + Z - Y2/rho) / (self.m + 1)
        L = project_simplex_columns(temp)
        
        # 4. 计算残差
        dY = np.zeros((self.d, self.n, self.m))
        primal_res_sum = 0
        for i in range(self.m):
            dY[:, :, i] = L + S[:, :, i] - X[:, :, i]
            primal_res_sum += np.linalg.norm(dY[:, :, i], 'fro')**2
        
        dY2 = L - Z
        primal_res = np.sqrt(primal_res_sum + np.linalg.norm(dY2, 'fro')**2)
        
        # 5. 计算对偶残差
        dual_res = rho * np.linalg.norm(L - L_old, 'fro')
        
        # 检查收敛 - 统一使用原始残差和对偶残差
        tol = 1e-5
        converged = (primal_res < tol) and (dual_res < tol)
        
        # 6. 更新拉格朗日乘子
        Y = Y + rho * dY
        Y2 = Y2 + rho * dY2
        
        # 7. 更新变量
        self.variables['L'] = L
        self.variables['S'] = S
        self.variables['Z'] = Z
        self.variables['Y'] = Y
        self.variables['Y2'] = Y2
        
        # 返回迭代结果
        return {
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'objective': self.compute_objective(),
            'converged': converged
        }

# ============== 问题工厂函数 ==============

def l1_regularization(**kwargs):
    """L1正则化问题工厂函数"""
    return L1MinimizationProblem(**kwargs)

def elastic_net(**kwargs):
    """弹性网问题工厂函数"""
    return ElasticNetProblem(**kwargs)

def l1_regression(**kwargs):
    """L1回归问题工厂函数"""
    return L1RegularizedRegressionProblem(**kwargs)

def elastic_net_regression(**kwargs):
    """弹性网回归问题工厂函数"""
    return ElasticNetRegressionProblem(**kwargs)

def low_rank_matrix_completion(**kwargs):
    """低秩矩阵补全问题工厂函数"""
    return LowRankMatrixCompletionProblem(**kwargs)

def low_rank_representation(**kwargs):
    """低秩表示问题工厂函数"""
    return LowRankRepresentationProblem(**kwargs)

def robust_multi_view_spectral_clustering(**kwargs):
    """鲁棒多视图谱聚类问题工厂函数"""
    return RobustMultiViewSpectralClusteringProblem(**kwargs)

# ============== 问题注册表 ==============

PROBLEM_REGISTRY = {
    "l1_regularization": l1_regularization,
    "elastic_net": elastic_net,
    "l1_regression": l1_regression,
    "elastic_net_regression": elastic_net_regression,
    "low_rank_matrix_completion": low_rank_matrix_completion,
    "low_rank_representation": low_rank_representation,
    "robust_multi_view_spectral_clustering": robust_multi_view_spectral_clustering,
}

def create_problem(problem_name: str, **kwargs):
    """
    根据问题名称创建问题实例
    
    Args:
        problem_name: 问题名称，必须是PROBLEM_REGISTRY中的键
        **kwargs: 传递给问题构造函数的参数
        
    Returns:
        问题实例
        
    Raises:
        ValueError: 如果问题名称不在注册表中
    """
    if problem_name not in PROBLEM_REGISTRY:
        raise ValueError(f"未知的问题类型: {problem_name}。可用选项: {list(PROBLEM_REGISTRY.keys())}")
    
    return PROBLEM_REGISTRY[problem_name](**kwargs)

def get_all_problems(**kwargs):
    """获取所有7个问题实例"""
    problems = []
    for problem_name in PROBLEM_REGISTRY:
        problems.append(create_problem(problem_name, **kwargs))
    return problems

def get_problem_info(problem_name: str = None):
    """获取问题信息"""
    if problem_name:
        problem = create_problem(problem_name)
        return {
            "name": problem.name,
            "parameters": problem.get_parameters(),
            "description": f"ADMM求解器：{problem.name}"
        }
    else:
        info = {}
        for name in PROBLEM_REGISTRY:
            problem = create_problem(name)
            info[name] = {
                "name": problem.name,
                "parameters": problem.get_parameters(),
                "description": f"ADMM求解器：{problem.name}"
            }
        return info