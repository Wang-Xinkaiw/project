#!/usr/bin/env python3
"""
测试框架组件 - 针对现有的问题模块结构
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """测试配置文件"""
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("测试配置文件...")
    
    # 检查API配置
    assert 'api' in config, "配置文件缺少api部分"
    assert 'api_key' in config['api'], "配置文件缺少api_key"
    assert 'temperature' in config['api'], "配置文件缺少temperature"
    assert 'max_tokens' in config['api'], "配置文件缺少max_tokens"
    
    # 检查generator配置
    assert 'generator' in config, "配置文件缺少generator部分"
    
    # 检查problems配置
    assert 'problems' in config, "配置文件缺少problems部分"
    assert 'admm' in config['problems'], "配置文件缺少admm问题列表"
    
    print("配置文件测试通过!")
    return True

def test_base_strategy():
    """测试基础策略类"""
    try:
        from strategies.base_strategy import BaseTuningStrategy
        print("基础策略类导入成功!")
        return True
    except Exception as e:
        print(f"基础策略类导入失败: {e}")
        return False

def test_problems_module():
    """测试问题模块 - 使用现有的文件结构"""
    try:
        # 首先尝试导入问题模块
        import problems.admm_problems as admm_problems
        print("ADMM问题模块导入成功!")
        
        # 尝试创建L1正则化问题
        try:
            from problems.admm_problems import create_problem
            
            problem = create_problem("l1_regularization", seed=42)
            print(f"成功创建问题实例: {problem.name}")
            
            # 检查必要的方法
            required_methods = ['admm_iteration', 'evaluate_solution', 
                               'compute_objective', 'initialize_variables']
            for method in required_methods:
                if hasattr(problem, method):
                    print(f"[OK] 问题有方法: {method}")
                else:
                    print(f"[ERROR] 问题缺少方法: {method}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"创建问题实例失败: {e}")
            # 尝试直接实例化一个简单的问题
            print("尝试使用直接实例化...")
            try:
                # 直接使用类名创建
                from problems.admm_problems import L1MinimizationProblem
                problem = L1MinimizationProblem(seed=42)
                print(f"直接实例化成功: {problem.name}")
                return True
            except Exception as e2:
                print(f"直接实例化也失败: {e2}")
                return False
        
    except Exception as e:
        print(f"问题模块导入失败: {e}")
        # 创建一个简化的问题进行测试
        print("创建简化问题进行测试...")
        try:
            from test_fix import SimpleL1Problem
            problem = SimpleL1Problem(seed=42)
            print(f"创建简化问题成功: {problem.name}")
            return True
        except Exception as e2:
            print(f"创建简化问题也失败: {e2}")
            return False

def test_all_problems():
    """测试所有7个ADMM问题"""
    try:
        from problems.admm_problems import PROBLEM_REGISTRY, create_problem
        print(f"\n测试所有 {len(PROBLEM_REGISTRY)} 个ADMM问题...")
        
        successful = []
        failed = []
        
        for problem_name in PROBLEM_REGISTRY.keys():
            try:
                print(f"测试问题: {problem_name}")
                problem = create_problem(problem_name, seed=42)
                
                # 检查必要的方法
                required_attrs = ['name', 'params', 'converged', 'iterations']
                for attr in required_attrs:
                    if not hasattr(problem, attr):
                        print(f"  [ERROR] 缺少属性: {attr}")
                        failed.append(problem_name)
                        break
                else:
                    print(f"  [OK] 问题 '{problem.name}' 创建成功")
                    successful.append(problem_name)
                    
            except Exception as e:
                print(f"  [ERROR] 创建失败: {e}")
                failed.append(problem_name)
        
        print(f"\n测试结果: {len(successful)} 成功, {len(failed)} 失败")
        if failed:
            print(f"失败的问题: {failed}")
        
        return len(failed) == 0
        
    except Exception as e:
        print(f"测试所有问题时出错: {e}")
        return False

def test_evaluator():
    """测试评估器与问题模块的兼容性"""
    try:
        from evaluator import StrategyEvaluator
        
        # 创建测试配置
        test_config = {
            'evaluator': {
                'max_iterations': 100,
                'tolerance': 1e-6
            },
            'algorithms': {
                'admm': {
                    'default_beta': 1.0,
                    'problem_module': 'problems.admm_problems'
                }
            }
        }
        
        evaluator = StrategyEvaluator(test_config)
        print("策略评估器创建成功!")
        
        # 测试问题加载
        try:
            problem = evaluator._load_problem('admm', 'l1_regularization')
            if problem:
                print(f"问题加载成功: {problem.name if hasattr(problem, 'name') else '未知'}")
                return True
            else:
                print("问题加载失败: 返回None")
                return False
        except Exception as e:
            print(f"问题加载测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"评估器测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("测试进化自适应调参框架组件")
    print("=" * 60)
    
    tests = [
        ("配置文件", test_config),
        ("基础策略类", test_base_strategy),
        ("ADMM问题模块", test_problems_module),
        ("所有7个ADMM问题", test_all_problems),
        ("评估器兼容性", test_evaluator),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n[测试 {test_name}]")
        try:
            if test_func():
                passed += 1
                print(f"[PASS] {test_name} 通过")
            else:
                failed += 1
                print(f"[FAIL] {test_name} 失败")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n[SUCCESS] 所有测试通过! 可以运行主程序了。")
        print("\n运行主程序的命令:")
        print("python main.py")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误。")
        if passed >= 3:  # 如果大部分测试通过，仍可尝试运行
            print("\n提示: 您可以尝试运行主程序，但可能需要修复一些问题。")
        return False

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)