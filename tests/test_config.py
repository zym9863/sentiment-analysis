# -*- coding: utf-8 -*-
"""
配置管理系统测试
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
import json
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ConfigManager, DataConfig, ModelConfig, TrainingConfig, SystemConfig, PathConfig

class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_config_initialization(self):
        """测试默认配置初始化"""
        config_manager = ConfigManager()
        
        # 验证各个配置对象被正确创建
        self.assertIsInstance(config_manager.data_config, DataConfig)
        self.assertIsInstance(config_manager.model_config, ModelConfig)
        self.assertIsInstance(config_manager.training_config, TrainingConfig)
        self.assertIsInstance(config_manager.system_config, SystemConfig)
        self.assertIsInstance(config_manager.path_config, PathConfig)
        
        # 验证默认值
        self.assertEqual(config_manager.data_config.batch_size, 16)
        self.assertEqual(config_manager.model_config.model_name, "bert-base-chinese")
        self.assertEqual(config_manager.training_config.learning_rate, 2e-5)
    
    def test_config_validation(self):
        """测试配置验证功能"""
        config_manager = ConfigManager()
        
        # 测试有效配置
        self.assertTrue(config_manager.validate_config())
        
        # 测试无效配置
        config_manager.data_config.batch_size = -1  # 无效批处理大小
        self.assertFalse(config_manager.validate_config())
    
    def test_json_config_file(self):
        """测试JSON配置文件加载和保存"""
        # 创建测试配置
        config_data = {
            'data': {
                'batch_size': 32,
                'max_length': 256
            },
            'model': {
                'model_name': 'test-bert',
                'dropout_rate': 0.5
            },
            'training': {
                'learning_rate': 1e-4,
                'max_epochs': 5
            }
        }
        
        # 保存到临时JSON文件
        json_file = self.temp_dir / 'test_config.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)
        
        # 加载配置
        config_manager = ConfigManager(str(json_file))
        
        # 验证配置被正确加载
        self.assertEqual(config_manager.data_config.batch_size, 32)
        self.assertEqual(config_manager.data_config.max_length, 256)
        self.assertEqual(config_manager.model_config.model_name, 'test-bert')
        self.assertEqual(config_manager.model_config.dropout_rate, 0.5)
        self.assertEqual(config_manager.training_config.learning_rate, 1e-4)
        self.assertEqual(config_manager.training_config.max_epochs, 5)
    
    def test_yaml_config_file(self):
        """测试YAML配置文件加载和保存"""
        # 创建测试配置
        config_data = {
            'data': {
                'batch_size': 64,
                'test_size': 0.3
            },
            'system': {
                'device': 'cuda',
                'log_level': 'DEBUG'
            }
        }
        
        # 保存到临时YAML文件
        yaml_file = self.temp_dir / 'test_config.yaml'
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        # 加载配置
        config_manager = ConfigManager(str(yaml_file))
        
        # 验证配置被正确加载
        self.assertEqual(config_manager.data_config.batch_size, 64)
        self.assertEqual(config_manager.data_config.test_size, 0.3)
        self.assertEqual(config_manager.system_config.device, 'cuda')
        self.assertEqual(config_manager.system_config.log_level, 'DEBUG')
    
    def test_config_save(self):
        """测试配置保存功能"""
        config_manager = ConfigManager()
        
        # 修改一些配置
        config_manager.data_config.batch_size = 128
        config_manager.model_config.dropout_rate = 0.4
        
        # 保存到JSON文件
        json_file = self.temp_dir / 'saved_config.json'
        config_manager.save_to_file(str(json_file))
        
        # 验证文件被创建
        self.assertTrue(json_file.exists())
        
        # 重新加载并验证
        new_config_manager = ConfigManager(str(json_file))
        self.assertEqual(new_config_manager.data_config.batch_size, 128)
        self.assertEqual(new_config_manager.model_config.dropout_rate, 0.4)
    
    def test_environment_variable_loading(self):
        """测试环境变量加载"""
        # 设置测试环境变量
        test_env = {
            'BATCH_SIZE': '256',
            'MODEL_TYPE': 'base',
            'LEARNING_RATE': '5e-5',
            'DEVICE': 'cpu'
        }
        
        # 保存原始环境变量
        original_env = {}
        for key in test_env:
            original_env[key] = os.environ.get(key)
            os.environ[key] = test_env[key]
        
        try:
            # 创建配置管理器
            config_manager = ConfigManager()
            
            # 验证环境变量被正确加载
            self.assertEqual(config_manager.data_config.batch_size, 256)
            self.assertEqual(config_manager.model_config.model_type, 'base')
            self.assertEqual(config_manager.training_config.learning_rate, 5e-5)
            self.assertEqual(config_manager.system_config.device, 'cpu')
            
        finally:
            # 恢复原始环境变量
            for key, value in original_env.items():
                if value is None:
                    if key in os.environ:
                        del os.environ[key]
                else:
                    os.environ[key] = value
    
    def test_create_directories(self):
        """测试目录创建功能"""
        config_manager = ConfigManager()
        config_manager.path_config.project_root = str(self.temp_dir)
        config_manager.path_config.__post_init__()  # 重新计算路径
        
        # 创建目录
        config_manager.create_directories()
        
        # 验证目录被创建
        expected_dirs = [
            config_manager.path_config.data_dir,
            config_manager.path_config.model_dir,
            config_manager.path_config.output_dir,
            config_manager.path_config.log_dir,
            config_manager.path_config.checkpoint_dir
        ]
        
        for dir_path in expected_dirs:
            self.assertTrue(Path(dir_path).exists())
    
    def test_get_all_configs(self):
        """测试获取所有配置的字典表示"""
        config_manager = ConfigManager()
        all_configs = config_manager.get_all_configs()
        
        # 验证返回的字典包含所有配置类别
        expected_keys = ['data', 'model', 'training', 'system', 'paths']
        for key in expected_keys:
            self.assertIn(key, all_configs)
        
        # 验证字典中包含具体配置项
        self.assertIn('batch_size', all_configs['data'])
        self.assertIn('model_name', all_configs['model'])
        self.assertIn('learning_rate', all_configs['training'])
        self.assertIn('device', all_configs['system'])
        self.assertIn('data_dir', all_configs['paths'])


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)