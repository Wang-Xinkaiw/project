# admm_solver.py - 保持原样，τ参数已在初始化时固定
class ADMMSolver:
    def __init__(self, 
                 strategy,
                 tau: float = 1.0,  # τ参数在这里固定
                 max_iter: int = 1000,
                 tol: float = 1e-6):
        """
        初始化ADMM求解器
        
        参数:
            strategy: 参数调整策略实例
            tau: 对偶更新步长τ (论文公式17中的τ) - 固定不变
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.strategy = strategy
        self.tau = tau  # τ是固定的
        self.max_iter = max_iter
        self.tol = tol
        
        # 验证τ的范围
        if tau <= 0 or tau >= 2.618:
            print(f"警告: 对偶步长τ={tau}，建议范围(0, 2.618)")