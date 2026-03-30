"""
基础策略基类模块
所有策略都应继承这个基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import inspect
import logging

class BaseTuningStrategy(ABC):
    """
    参数调优策略基类
    
    所有具体策略必须继承此类并实现以下方法
    """
    
    @abstractmethod
    def update_parameters(self, iteration_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于当前迭代状态更新参数
        
        Args:
            iteration_state: 包含当前迭代状态的字典，通常包括:
                - iteration: 当前迭代次数
                - primal_residual: 原始残差 (ADMM)
                - dual_residual: 对偶残差 (ADMM)
                - gradient_norm: 梯度范数 (梯度下降)
                - objective: 目标函数值
                - converged: 是否收敛
                
        Returns:
            更新后的参数字典，例如 {'beta': new_beta} 或 {'learning_rate': new_lr}
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取当前参数值
        
        Returns:
            当前参数字典
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置参数值
        
        Args:
            params: 要设置的参数字典
        """
        pass
    
    @classmethod
    def validate_strategy(cls, strategy_instance) -> tuple[bool, str]:
        """
        验证策略实例是否符合要求
        
        Args:
            strategy_instance: 策略实例
            
        Returns:
            (是否有效，错误信息)
        """
        try:
            # 检查是否是 BaseTuningStrategy 的子类
            if not isinstance(strategy_instance, BaseTuningStrategy):
                return False, "策略必须继承 BaseTuningStrategy"
            
            # 检查 update_parameters 方法签名
            update_method = getattr(strategy_instance, 'update_parameters', None)
            if not update_method:
                return False, "策略缺少 update_parameters 方法"
            
            # 检查方法签名
            sig = inspect.signature(update_method)
            params = list(sig.parameters.keys())
            
            # 对于绑定方法，self 不计入参数，所以应该有 1 个参数 (iteration_state)
            if len(params) < 1:
                return False, f"update_parameters 方法参数不足，应有至少 1 个参数，实际有{len(params)}个：{params}"
            
            # 检查是否包含 iteration_state 参数
            if 'iteration_state' not in params:
                return False, f"update_parameters 方法应包含'iteration_state'参数，实际参数：{params}"
            
            # 测试调用方法（使用虚拟数据）
            test_state = {
                'iteration': 0,
                'primal_residual': 1.0,
                'dual_residual': 1.0,
                'beta': 1.0,
                'objective': 0.0,
                'converged': False
            }
            
            try:
                result = update_method(test_state)
            except Exception as e:
                return False, f"调用 update_parameters 方法失败：{str(e)}"
            
            # 检查返回值
            if not isinstance(result, dict):
                return False, f"update_parameters 方法应返回字典，实际返回{type(result)}"
            
            # 对于 ADMM 策略，检查是否包含 beta
            if hasattr(strategy_instance, '__class__'):
                class_name = strategy_instance.__class__.__name__
                if 'ADMM' in class_name.upper() or 'BETA' in class_name.upper():
                    if 'beta' not in result:
                        return False, "ADMM 策略的 update_parameters 方法返回字典必须包含'beta'键"
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"验证过程中发生错误：{str(e)}"
