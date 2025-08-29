# -*- coding: utf-8 -*-
"""
工具函数模块测试
"""

import unittest
import torch
import tempfile
import os
import sys
from pathlib import Path
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import EarlyStopping, ModelCheckpoint, setup_logging

class TestEarlyStopping(unittest.TestCase):
    """早停机制测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.early_stopping = EarlyStopping(patience=3, verbose=False)
    
    def test_early_stopping_initialization(self):
        """测试早停初始化"""
        self.assertEqual(self.early_stopping.patience, 3)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertFalse(self.early_stopping.early_stop)
        self.assertEqual(self.early_stopping.best_score, None)
    
    def test_early_stopping_improvement(self):
        """测试性能改善情况"""
        # 模拟损失下降
        losses = [1.0, 0.8, 0.6, 0.7, 0.5]
        
        for loss in losses:
            self.early_stopping(loss)
        
        # 验证最后没有触发早停
        self.assertFalse(self.early_stopping.early_stop)
    
    def test_early_stopping_trigger(self):
        """测试早停触发"""
        # 模拟损失先下降后上升
        losses = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for loss in losses:
            self.early_stopping(loss)
            if self.early_stopping.early_stop:
                break
        
        # 验证早停被触发
        self.assertTrue(self.early_stopping.early_stop)


class TestModelCheckpoint(unittest.TestCase):
    """模型检查点测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_path = self.temp_dir / 'test_model.pth'
        self.checkpoint = ModelCheckpoint(
            self.checkpoint_path, 
            verbose=False
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_initialization(self):
        """测试检查点初始化"""
        self.assertEqual(self.checkpoint.checkpoint_path, self.checkpoint_path)
        self.assertEqual(self.checkpoint.best_score, -float('inf'))
    
    def test_checkpoint_saving(self):
        """测试检查点保存"""
        # 创建模拟模型和优化器
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # 模拟多次性能改善
        scores = [0.7, 0.8, 0.9, 0.85, 0.95]
        
        for epoch, score in enumerate(scores):
            self.checkpoint(score, model, optimizer, epoch)
        
        # 验证检查点文件被创建
        self.assertTrue(self.checkpoint_path.exists())
        
        # 验证最佳分数被记录
        self.assertEqual(self.checkpoint.best_score, 0.95)
    
    def test_checkpoint_loading(self):
        """测试检查点加载"""
        # 创建并保存模型
        original_model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(original_model.parameters())
        
        # 保存检查点
        self.checkpoint(0.9, original_model, optimizer, 0)
        
        # 加载检查点
        loaded_checkpoint = torch.load(self.checkpoint_path)
        
        # 验证检查点包含必要信息
        expected_keys = ['model_state_dict', 'optimizer_state_dict', 'score', 'epoch']
        for key in expected_keys:
            self.assertIn(key, loaded_checkpoint)


class TestUtilityFunctions(unittest.TestCase):
    """工具函数测试类"""
    
    def test_setup_logging(self):
        """测试日志设置功能"""
        # 测试默认日志设置
        logger = setup_logging()
        
        # 验证日志器被正确创建
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.INFO)
        
        # 测试自定义级别
        debug_logger = setup_logging(level=logging.DEBUG)
        self.assertEqual(debug_logger.level, logging.DEBUG)
    
    def test_logging_output(self):
        """测试日志输出"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            # 创建带文件输出的日志器
            logger = setup_logging(log_file=log_file)
            
            # 写入测试日志
            test_message = "测试日志消息"
            logger.info(test_message)
            
            # 验证日志文件存在
            self.assertTrue(os.path.exists(log_file))
            
            # 验证日志内容（需要刷新缓冲区）
            logging.shutdown()
            
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
                self.assertIn(test_message, log_content)
                
        finally:
            # 清理日志文件
            if os.path.exists(log_file):
                os.unlink(log_file)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)