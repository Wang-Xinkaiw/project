#!/usr/bin/env python3
"""测试脚本"""

import os
import sys
import yaml

# 确保当前目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import EvolutionaryTuningMain

def test_config():
    """测试配置文件加载"""
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ 配置文件加载成功")
        
        # 检查必要的配置项
        required_keys = ['api', 'evaluator', 'problems']
        for key in required_keys:
            if key not in config:
                print(f"✗ 缺少必要配置项: {key}")
                return False
        print("✓ 配置项检查通过")
        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def test_directories():
    """测试目录创建"""
    directories = ['strategies', 'results', 'feedback', 'logs']
    
    for dir_name in directories:
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✓ 目录 {dir_name} 创建成功")
        except Exception as e:
            print(f"✗ 目录 {dir_name} 创建失败: {e}")
            return False
    
    # 检查是否已存在初始策略
    if not os.path.exists("strategies/initial_strategy.py"):
        print("⚠ 初始策略文件不存在，将创建...")
        # 创建初始策略文件
        with open("strategies/initial_strategy.py", 'w', encoding='utf-8') as f:
            f.write('''# strategies/initial_strategy.py
"""初始测试策略"""

from strategies.base_strategy import BaseTuningStrategy

class InitialADMMStrategy(BaseTuningStrategy):
    """初始ADMM策略 - 用于测试"""
    
    def __init__(self, initial_beta=1.0):
        self.beta = initial_beta
        self.history = []
    
    def update_parameters(self, iteration_state):
        """简单的自适应调整"""
        return {'beta': self.beta, 'adjustment': 'keep'}
    
    def get_parameters(self):
        return {'beta': self.beta}
    
    def set_parameters(self, params):
        if 'beta' in params:
            self.beta = params['beta']
''')
        print("✓ 初始策略文件创建成功")
    
    return True

def test_imports():
    """测试导入"""
    try:
        import numpy as np
        import yaml
        import requests
        
        print("✓ 基本包导入成功")
        
        # 测试导入项目模块
        from strategies.base_strategy import BaseTuningStrategy
        print("✓ 项目模块导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def run_test_iteration():
    """运行一次测试迭代"""
    print("\n" + "="*50)
    print("开始测试迭代")
    print("="*50)
    
    try:
        main = EvolutionaryTuningMain("config.yaml")
        
        # 只运行一轮测试
        main.iteration = 0
        
        # 手动设置一个简单的策略代码
        test_strategy = '''
from strategies.base_strategy import BaseTuningStrategy

class TestADMMStrategy(BaseTuningStrategy):
    """测试策略"""
    
    def __init__(self, initial_beta=1.0):
        self.beta = initial_beta
    
    def update_parameters(self, iteration_state):
        return {'beta': self.beta, 'adjustment': 'keep'}
    
    def get_parameters(self):
        return {'beta': self.beta}
    
    def set_parameters(self, params):
        if 'beta' in params:
            self.beta = params['beta']
    
    def reset(self):
        pass
'''
        
        # 保存测试策略
        strategy_path = "strategies/strategy_test.py"
        with open(strategy_path, 'w', encoding='utf-8') as f:
            f.write(test_strategy)
        
        # 评估策略
        from evaluator import StrategyEvaluator
        evaluator = StrategyEvaluator(main.config)
        
        # 使用简短名称
        test_problems = ['rpca']
        print(f"测试问题: {test_problems}")
        
        results = evaluator.evaluate_strategy(
            strategy_path=strategy_path,
            algorithm_type="admm",
            problem_names=test_problems
        )
        
        print(f"评估结果: {results}")
        
        # 计算性能
        performance = main._calculate_average_performance(results)
        print(f"平均性能: {performance}")
        
        return True
    except Exception as e:
        print(f"✗ 测试迭代失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始系统测试...")
    
    tests = [
        ("配置测试", test_config),
        ("目录测试", test_directories),
        ("导入测试", test_imports),
        ("迭代测试", run_test_iteration)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            print(f"✓ {test_name} 通过")
        else:
            print(f"✗ {test_name} 失败")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ 所有测试通过！")
        print("现在可以运行 main.py")
    else:
        print("✗ 有些测试失败，请查看上面的错误信息")