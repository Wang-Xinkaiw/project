"""
优化问题基类 - 用于进化自适应调参策略框架
为不同算法类型提供统一的接口
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging


class BaseOptimizationProblem(ABC):
    """
    优化问题基类
    
    所有优化问题必须继承此类，并实现相应算法的方法
    """
    
    def __init__(self, name: str, algorithm_type: str):
        """
        初始化优化问题
        
        参数:
            name: 问题名称
            algorithm_type: 算法类型 ('admm', 'gradient_descent', 'generic')
        """
        self.name = name
        self.algorithm_type = algorithm_type
        self.logger = logging.getLogger(f"Problem.{algorithm_type}.{name}")
        
        # 问题参数和数据
        self.params: Dict[str, Any] = {}
        self.data: Dict[str, Any] = {}
        
        # 变量和状态
        self.variables: Dict[str, np.ndarray] = {}
        self.solution: Dict[str, Any] = {}
        
        # 收敛历史
        self.convergence_history: List[Dict[str, Any]] = []
        
        # 收敛状态
        self.converged = False
        self.iterations = 0
        self.final_objective = float('inf')
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None, 
              params: Optional[Dict[str, Any]] = None):
        """
        重置问题状态
        
        参数:
            seed: 随机种子
            params: 问题参数
        """
        pass
    
    @abstractmethod
    def initialize_variables(self):
        """初始化变量"""
        pass
    
    @abstractmethod
    def compute_objective(self) -> float:
        """计算当前目标函数值"""
        pass
    
    @abstractmethod
    def evaluate_solution(self) -> Dict[str, Any]:
        """评估当前解的质量"""
        pass
    
    @abstractmethod
    def get_initial_params(self) -> Dict[str, Any]:
        """获取算法初始参数"""
        pass
    
    @abstractmethod
    def get_iteration_state(self) -> Dict[str, Any]:
        """获取当前迭代状态，用于调参策略"""
        pass
    
    def get_problem_info(self) -> Dict[str, Any]:
        """获取问题信息"""
        return {
            'name': self.name,
            'algorithm_type': self.algorithm_type,
            'dimensions': self._get_dimensions(),
            'parameters': self.params,
            'converged': self.converged,
            'iterations': self.iterations,
            'final_objective': self.final_objective
        }
    
    def _get_dimensions(self) -> Dict[str, int]:
        """获取问题维度信息"""
        dims = {}
        for key, value in self.variables.items():
            if isinstance(value, np.ndarray):
                dims[key] = value.shape
            else:
                dims[key] = 1
        return dims
    
    def save_convergence_history(self, iteration: int, 
                               iteration_result: Dict[str, Any]):
        """保存收敛历史"""
        history_entry = {
            'iteration': iteration,
            'timestamp': np.datetime64('now'),
            **iteration_result
        }
        self.convergence_history.append(history_entry)
        
        # 限制历史记录长度，防止内存爆炸
        max_history = self.params.get('max_history', 1000)
        if len(self.convergence_history) > max_history:
            self.convergence_history = self.convergence_history[-max_history:]
    
    def check_convergence(self, iteration_result: Dict[str, Any], 
                         tolerance: float = 1e-6) -> bool:
        """
        检查收敛条件
        
        参数:
            iteration_result: 当前迭代结果
            tolerance: 收敛容忍度
            
        返回:
            是否收敛
        """
        # 【优先】如果子类明确返回了收敛标志，直接使用
        if 'converged' in iteration_result and iteration_result['converged']:
            self.converged = True
            self.final_objective = iteration_result.get('objective', self.compute_objective())
            self.logger.info(f"子类报告收敛: objective={self.final_objective:.6f}")
            return True
        
        # 否则使用残差检查
        primal_res = iteration_result.get('primal_residual', float('inf'))
        dual_res = iteration_result.get('dual_residual', float('inf'))
        gradient_norm = iteration_result.get('gradient_norm', float('inf'))
        objective_change = iteration_result.get('objective_change', float('inf'))
        
        converged = False
        
        # ADMM收敛检查
        if primal_res < tolerance and dual_res < tolerance:
            converged = True
            self.logger.info(f"ADMM收敛: 原始残差={primal_res:.2e}, 对偶残差={dual_res:.2e}")
        
        # 梯度下降收敛检查
        elif gradient_norm < tolerance:
            converged = True
            self.logger.info(f"梯度下降收敛: 梯度范数={gradient_norm:.2e}")
        
        # 目标函数变化收敛检查
        elif abs(objective_change) < tolerance:
            converged = True
            self.logger.info(f"目标函数收敛: 变化={objective_change:.2e}")
        
        # 最大迭代次数检查
        max_iter = self.params.get('max_iterations', 1000)
        if self.iterations >= max_iter:
            self.logger.warning(f"达到最大迭代次数: {max_iter}")
            converged = True  # 强制终止
        
        if converged:
            self.converged = True
            self.final_objective = iteration_result.get('objective', self.compute_objective())
        
        return converged


class ADMMProblemBase(BaseOptimizationProblem):
    """
    ADMM问题基类
    
    提供ADMM算法的通用接口
    """
    
    def __init__(self, name: str):
        super().__init__(name, 'admm')
        
        # ADMM特定参数
        self.params.update({
            'beta': 1.0,          # 惩罚参数
            'tau': 1.0,           # 对偶更新步长
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'rho': 1.2,           # 惩罚参数增长因子
            'mu': 1e-3,           # 增广拉格朗日参数
            'max_rho': 1e10
        })
    
    def reset(self, seed: Optional[int] = None, 
              params: Optional[Dict[str, Any]] = None):
        """重置ADMM问题"""
        if seed is not None:
            np.random.seed(seed)
        
        # 更新参数
        if params:
            self.params.update(params)
        
        # 生成数据
        self._generate_data()
        
        # 初始化变量
        self.initialize_variables()
        
        # 重置状态
        self.converged = False
        self.iterations = 0
        self.final_objective = float('inf')
        self.convergence_history = []
        
        self.logger.info(f"ADMM问题 '{self.name}' 重置完成")
    
    @abstractmethod
    def _generate_data(self):
        """生成问题特定数据"""
        pass
    
    def admm_iteration(self, beta: Optional[float] = None, 
                      iteration: int = 0) -> Dict[str, Any]:
        """
        执行一次ADMM迭代
        
        参数:
            beta: 惩罚参数（如果为None则使用当前参数）
            iteration: 当前迭代次数
            
        返回:
            迭代结果字典
        """
        if beta is None:
            beta = self.params['beta']
        
        # 执行ADMM更新
        try:
            iteration_result = self._admm_update(beta, iteration)
            
            # 更新迭代次数
            self.iterations += 1
            
            # 保存历史
            self.save_convergence_history(iteration, iteration_result)
            
            # 检查收敛
            self.converged = self.check_convergence(
                iteration_result, 
                self.params['tolerance']
            )
            
            # 更新最终目标值
            if self.converged:
                self.final_objective = iteration_result.get('objective', 
                                                          self.compute_objective())
            
            return iteration_result
            
        except Exception as e:
            self.logger.error(f"ADMM迭代失败: {e}")
            return {
                'error': str(e),
                'iteration': iteration,
                'converged': False,
                'objective': float('inf')
            }
    
    @abstractmethod
    def _admm_update(self, beta: float, iteration: int) -> Dict[str, Any]:
        """
        ADMM更新步骤（子类实现）
        
        参数:
            beta: 惩罚参数
            iteration: 当前迭代次数
            
        返回:
            包含更新结果的字典
        """
        pass
    
    def get_initial_params(self) -> Dict[str, Any]:
        """获取ADMM初始参数"""
        return {
            'beta': self.params['beta'],
            'tau': self.params['tau'],
            'max_iterations': self.params['max_iterations'],
            'tolerance': self.params['tolerance']
        }
    
    def get_iteration_state(self) -> Dict[str, Any]:
        """获取ADMM迭代状态"""
        if not self.convergence_history:
            return {
                'iteration': 0,
                'beta': self.params['beta'],
                'converged': False
            }
        
        last_history = self.convergence_history[-1]
        return {
            'iteration': self.iterations,
            'primal_residual': last_history.get('primal_residual', 0.0),
            'dual_residual': last_history.get('dual_residual', 0.0),
            'beta': last_history.get('beta', self.params['beta']),
            'objective': last_history.get('objective', 0.0),
            'converged': self.converged
        }
    
    def compute_residuals(self) -> Tuple[float, float]:
        """
        计算ADMM残差
        
        返回:
            (原始残差, 对偶残差)
        """
        # 子类应实现具体的残差计算
        return 0.0, 0.0


class GradientDescentProblemBase(BaseOptimizationProblem):
    """
    梯度下降问题基类
    
    提供梯度下降算法的通用接口
    """
    
    def __init__(self, name: str):
        super().__init__(name, 'gradient_descent')
        
        # 梯度下降特定参数
        self.params.update({
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'momentum': 0.9,
            'decay_rate': 0.99
        })
    
    def reset(self, seed: Optional[int] = None, 
              params: Optional[Dict[str, Any]] = None):
        """重置梯度下降问题"""
        if seed is not None:
            np.random.seed(seed)
        
        # 更新参数
        if params:
            self.params.update(params)
        
        # 生成数据
        self._generate_data()
        
        # 初始化变量
        self.initialize_variables()
        
        # 重置状态
        self.converged = False
        self.iterations = 0
        self.final_objective = float('inf')
        self.convergence_history = []
        
        self.logger.info(f"梯度下降问题 '{self.name}' 重置完成")
    
    @abstractmethod
    def _generate_data(self):
        """生成问题特定数据"""
        pass
    
    def gradient_descent_step(self, learning_rate: Optional[float] = None,
                            iteration: int = 0) -> Dict[str, Any]:
        """
        执行一步梯度下降
        
        参数:
            learning_rate: 学习率（如果为None则使用当前参数）
            iteration: 当前迭代次数
            
        返回:
            迭代结果字典
        """
        if learning_rate is None:
            learning_rate = self.params['learning_rate']
        
        # 执行梯度下降更新
        try:
            iteration_result = self._gradient_update(learning_rate, iteration)
            
            # 更新迭代次数
            self.iterations += 1
            
            # 保存历史
            self.save_convergence_history(iteration, iteration_result)
            
            # 检查收敛
            self.converged = self.check_convergence(
                iteration_result, 
                self.params['tolerance']
            )
            
            # 更新最终目标值
            if self.converged:
                self.final_objective = iteration_result.get('objective', 
                                                          self.compute_objective())
            
            return iteration_result
            
        except Exception as e:
            self.logger.error(f"梯度下降步失败: {e}")
            return {
                'error': str(e),
                'iteration': iteration,
                'converged': False,
                'objective': float('inf')
            }
    
    @abstractmethod
    def _gradient_update(self, learning_rate: float, 
                        iteration: int) -> Dict[str, Any]:
        """
        梯度更新步骤（子类实现）
        
        参数:
            learning_rate: 学习率
            iteration: 当前迭代次数
            
        返回:
            包含更新结果的字典
        """
        pass
    
    def compute_gradient(self) -> np.ndarray:
        """计算梯度（子类实现）"""
        raise NotImplementedError("子类必须实现compute_gradient方法")
    
    def get_initial_params(self) -> Dict[str, Any]:
        """获取梯度下降初始参数"""
        return {
            'learning_rate': self.params['learning_rate'],
            'momentum': self.params.get('momentum', 0.0),
            'max_iterations': self.params['max_iterations'],
            'tolerance': self.params['tolerance']
        }
    
    def get_iteration_state(self) -> Dict[str, Any]:
        """获取梯度下降迭代状态"""
        if not self.convergence_history:
            return {
                'iteration': 0,
                'learning_rate': self.params['learning_rate'],
                'converged': False
            }
        
        last_history = self.convergence_history[-1]
        return {
            'iteration': self.iterations,
            'gradient_norm': last_history.get('gradient_norm', 0.0),
            'learning_rate': last_history.get('learning_rate', 
                                            self.params['learning_rate']),
            'objective': last_history.get('objective', 0.0),
            'converged': self.converged
        }


class GenericOptimizationProblem(BaseOptimizationProblem):
    """
    通用优化问题基类
    
    提供通用优化算法的接口
    """
    
    def __init__(self, name: str):
        super().__init__(name, 'generic')
        
        # 通用参数
        self.params.update({
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'step_size': 0.01
        })
    
    def reset(self, seed: Optional[int] = None, 
              params: Optional[Dict[str, Any]] = None):
        """重置通用优化问题"""
        if seed is not None:
            np.random.seed(seed)
        
        # 更新参数
        if params:
            self.params.update(params)
        
        # 生成数据
        self._generate_data()
        
        # 初始化变量
        self.initialize_variables()
        
        # 重置状态
        self.converged = False
        self.iterations = 0
        self.final_objective = float('inf')
        self.convergence_history = []
        
        self.logger.info(f"通用优化问题 '{self.name}' 重置完成")
    
    @abstractmethod
    def _generate_data(self):
        """生成问题特定数据"""
        pass
    
    def optimization_step(self, step_params: Optional[Dict[str, Any]] = None,
                         iteration: int = 0) -> Dict[str, Any]:
        """
        执行一步优化
        
        参数:
            step_params: 步长参数
            iteration: 当前迭代次数
            
        返回:
            迭代结果字典
        """
        if step_params is None:
            step_params = {'step_size': self.params['step_size']}
        
        # 执行优化更新
        try:
            iteration_result = self._optimization_update(step_params, iteration)
            
            # 更新迭代次数
            self.iterations += 1
            
            # 保存历史
            self.save_convergence_history(iteration, iteration_result)
            
            # 检查收敛
            self.converged = self.check_convergence(
                iteration_result, 
                self.params['tolerance']
            )
            
            # 更新最终目标值
            if self.converged:
                self.final_objective = iteration_result.get('objective', 
                                                          self.compute_objective())
            
            return iteration_result
            
        except Exception as e:
            self.logger.error(f"优化步失败: {e}")
            return {
                'error': str(e),
                'iteration': iteration,
                'converged': False,
                'objective': float('inf')
            }
    
    @abstractmethod
    def _optimization_update(self, step_params: Dict[str, Any], 
                           iteration: int) -> Dict[str, Any]:
        """
        优化更新步骤（子类实现）
        
        参数:
            step_params: 步长参数
            iteration: 当前迭代次数
            
        返回:
            包含更新结果的字典
        """
        pass
    
    def get_initial_params(self) -> Dict[str, Any]:
        """获取通用优化初始参数"""
        return {
            'step_size': self.params['step_size'],
            'max_iterations': self.params['max_iterations'],
            'tolerance': self.params['tolerance']
        }
    
    def get_iteration_state(self) -> Dict[str, Any]:
        """获取通用优化迭代状态"""
        if not self.convergence_history:
            return {
                'iteration': 0,
                'converged': False
            }
        
        last_history = self.convergence_history[-1]
        return {
            'iteration': self.iterations,
            'objective': last_history.get('objective', 0.0),
            'converged': self.converged
        }