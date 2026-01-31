# gradient_descent_problems.py
from base_problem import GradientDescentProblemBase
import numpy as np
from typing import Dict, Any, Optional

class QuadraticProblem(GradientDescentProblemBase):  # ✅ 继承正确基类
    def __init__(self, dim=10, condition_number=10):
        super().__init__("quadratic_problem")  # 调用基类初始化
        self.dim = dim
        self.condition_number = condition_number
        self.params.update({
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'tolerance': 1e-6
        })
        
        # 重置生成数据
        self.reset()
    
    def _generate_data(self):
        """生成二次问题数据"""
        # 生成条件数控制的Hessian矩阵
        eigenvalues = np.linspace(1, self.condition_number, self.dim)
        Q = np.random.randn(self.dim, self.dim)
        Q, _ = np.linalg.qr(Q)
        self.data['A'] = Q @ np.diag(eigenvalues) @ Q.T
        self.data['b'] = np.random.randn(self.dim)
    
    def initialize_variables(self):
        """初始化变量"""
        self.variables['x'] = np.random.randn(self.dim)
    
    def compute_objective(self):
        """计算二次函数值"""
        x = self.variables['x']
        A = self.data['A']
        b = self.data['b']
        return 0.5 * x.T @ A @ x - b.T @ x
    
    def evaluate_solution(self):
        """评估解质量"""
        return {
            'objective': self.compute_objective(),
            'gradient_norm': np.linalg.norm(self.compute_gradient()),
            'converged': self.converged
        }
    
    def compute_gradient(self):
        """计算梯度"""
        x = self.variables['x']
        A = self.data['A']
        b = self.data['b']
        return A @ x - b
    
    def _gradient_update(self, learning_rate: float, iteration: int) -> Dict[str, Any]:
        """执行一步梯度下降"""
        # 计算梯度
        gradient = self.compute_gradient()
        gradient_norm = np.linalg.norm(gradient)
        
        # 更新参数
        self.variables['x'] = self.variables['x'] - learning_rate * gradient
        
        # 计算目标函数变化
        old_objective = self.compute_objective()
        new_objective = self.compute_objective()
        objective_change = new_objective - old_objective
        
        # 检查收敛
        converged = gradient_norm < self.params['tolerance']
        
        return {
            'gradient_norm': gradient_norm,
            'objective': new_objective,
            'objective_change': objective_change,
            'step_size': learning_rate * gradient_norm,
            'converged': converged
        }