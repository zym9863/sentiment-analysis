# -*- coding: utf-8 -*-
"""
测试运行脚本
用于运行项目中的所有单元测试
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目路径到系统路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """运行所有测试"""
    print("开始运行所有单元测试...")
    print("=" * 60)
    
    # 设置测试发现器
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    
    # 计算成功率
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"成功率: {success_rate:.1f}%")
    
    # 显示失败和错误详情
    if result.failures:
        print("\n失败的测试:")
        for test, trace in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n出错的测试:")
        for test, trace in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()

def run_specific_test(test_module):
    """运行特定模块的测试"""
    print(f"运行 {test_module} 测试...")
    print("=" * 60)
    
    try:
        # 动态导入测试模块
        module = __import__(test_module)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except ImportError as e:
        print(f"无法导入测试模块 {test_module}: {e}")
        return False

def show_test_coverage():
    """显示测试覆盖的模块"""
    test_files = list(Path(__file__).parent.glob('test_*.py'))
    
    print("可用的测试模块:")
    print("-" * 40)
    for test_file in test_files:
        module_name = test_file.stem
        target_module = module_name.replace('test_', '')
        print(f"  {module_name} -> 测试 {target_module}")
    print()

if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'all':
            success = run_all_tests()
            sys.exit(0 if success else 1)
        elif command == 'list':
            show_test_coverage()
        elif command.startswith('test_'):
            success = run_specific_test(command)
            sys.exit(0 if success else 1)
        else:
            print(f"未知命令: {command}")
            print("可用命令:")
            print("  all     - 运行所有测试")
            print("  list    - 显示可用测试模块")
            print("  test_*  - 运行特定测试模块")
            sys.exit(1)
    else:
        # 默认运行所有测试
        success = run_all_tests()
        sys.exit(0 if success else 1)