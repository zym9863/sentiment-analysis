# -*- coding: utf-8 -*-
"""
模型训练模块测试
"""

import unittest
import torch
import tempfile
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import ModelTrainer
from models.base_model import create_base_model
from models.improved_model import create_improved_model

class TestModelTrainer(unittest.TestCase):
    """模型训练器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建训练器实例（使用较小的配置用于测试）
        self.trainer = ModelTrainer(
            model_type='base',
            num_classes=2,
            learning_rate=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=1,
            device=torch.device('cpu'),  # 使用CPU进行测试
            save_dir=str(self.temp_dir)
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_creation(self):
        """测试模型创建"""
        # 测试基础模型创建
        base_trainer = ModelTrainer(
            model_type='base',
            device=torch.device('cpu'),
            save_dir=str(self.temp_dir)
        )
        self.assertIsNotNone(base_trainer.model)
        
        # 测试改进模型创建
        improved_trainer = ModelTrainer(
            model_type='improved',
            device=torch.device('cpu'),
            save_dir=str(self.temp_dir)
        )
        self.assertIsNotNone(improved_trainer.model)
    
    def test_optimizer_setup(self):
        """测试优化器设置"""
        optimizer = self.trainer.optimizer
        self.assertIsNotNone(optimizer)
        
        # 验证参数组数量（BERT参数和其他参数分开）
        self.assertEqual(len(optimizer.param_groups), 2)
    
    def test_gpu_memory_monitoring_cpu(self):
        """测试CPU环境下GPU内存监控功能"""
        # 在CPU环境下，GPU监控函数应该安全执行
        self.trainer._log_gpu_memory("测试")
        self.trainer._clear_gpu_cache()
        result = self.trainer._monitor_gpu_memory_threshold()
        self.assertFalse(result)  # CPU环境下应该返回False
    
    @unittest.skipUnless(torch.cuda.is_available(), "需要GPU环境")
    def test_gpu_memory_monitoring_gpu(self):
        """测试GPU环境下内存监控功能"""
        gpu_trainer = ModelTrainer(
            model_type='base',
            device=torch.device('cuda'),
            save_dir=str(self.temp_dir)
        )
        
        # 测试GPU内存监控函数
        gpu_trainer._log_gpu_memory("GPU测试")
        gpu_trainer._clear_gpu_cache()
        
        # 监控阈值测试（通常不会超过阈值）
        result = gpu_trainer._monitor_gpu_memory_threshold(threshold=0.99)
        self.assertIsInstance(result, bool)


class TestModelCreation(unittest.TestCase):
    """模型创建测试类"""
    
    def test_base_model_creation(self):
        """测试基础模型创建"""
        model = create_base_model(
            model_name='bert-base-chinese',
            num_classes=2,
            model_type='enhanced'
        )
        
        self.assertIsNotNone(model)
        
        # 测试模型输出
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            self.assertIn('logits', outputs)
            self.assertEqual(outputs['logits'].shape, (batch_size, 2))
    
    def test_improved_model_creation(self):
        """测试改进模型创建"""
        model = create_improved_model(
            model_name='bert-base-chinese',
            num_classes=2,
            use_contrastive_loss=False  # 简化测试
        )
        
        self.assertIsNotNone(model)
        
        # 测试模型输出
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            self.assertIn('logits', outputs)
            self.assertEqual(outputs['logits'].shape, (batch_size, 2))


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)