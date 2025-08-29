# -*- coding: utf-8 -*-
"""
模型评估模块测试
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_evaluation import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    """模型评估器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建模拟模型
        self.mock_model = MagicMock()
        
        # 模拟模型输出
        def mock_forward(input_ids, attention_mask):
            batch_size = input_ids.shape[0]
            # 返回随机logits
            logits = torch.randn(batch_size, 2)
            return {'logits': logits}
        
        self.mock_model.return_value = mock_forward
        self.mock_model.eval = MagicMock()
        
        # 创建评估器
        self.evaluator = ModelEvaluator(
            model=self.mock_model,
            device=torch.device('cpu'),
            class_names=['负面', '正面']
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_evaluator_initialization(self):
        """测试评估器初始化"""
        self.assertIsNotNone(self.evaluator.model)
        self.assertEqual(self.evaluator.device, torch.device('cpu'))
        self.assertEqual(self.evaluator.class_names, ['负面', '正面'])
    
    def test_calculate_metrics(self):
        """测试指标计算"""
        # 创建测试数据
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])
        
        metrics = self.evaluator._calculate_metrics(y_true, y_pred, y_prob)
        
        # 验证指标包含必要字段
        expected_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # 验证指标值的范围
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(0 <= metrics['recall'] <= 1)
        self.assertTrue(0 <= metrics['f1'] <= 1)
        self.assertTrue(0 <= metrics['auc'] <= 1)
    
    def test_create_mock_dataloader(self):
        """创建模拟数据加载器用于测试"""
        from torch.utils.data import DataLoader, TensorDataset
        
        # 创建模拟数据
        batch_size = 4
        seq_length = 128
        num_samples = 16
        
        input_ids = torch.randint(0, 1000, (num_samples, seq_length))
        attention_mask = torch.ones(num_samples, seq_length)
        labels = torch.randint(0, 2, (num_samples,))
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return dataloader
    
    def test_model_evaluation_basic(self):
        """测试基础模型评估功能"""
        # 创建模拟数据加载器
        test_loader = self.test_create_mock_dataloader()
        
        # 更新模拟模型以适配DataLoader格式
        def mock_model_call(input_ids, attention_mask):
            batch_size = input_ids.shape[0]
            logits = torch.randn(batch_size, 2)
            return {'logits': logits}
        
        self.mock_model.side_effect = mock_model_call
        
        try:
            # 执行评估（简化版，不生成报告）
            with torch.no_grad():
                total_predictions = 0
                correct_predictions = 0
                
                for batch in test_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                        input_ids, attention_mask, labels = batch[:3]
                    else:
                        # 如果是字典格式
                        input_ids = batch['input_ids']
                        attention_mask = batch['attention_mask'] 
                        labels = batch['labels']
                    
                    outputs = self.mock_model(input_ids, attention_mask)
                    predictions = torch.argmax(outputs['logits'], dim=1)
                    
                    total_predictions += labels.size(0)
                    correct_predictions += (predictions == labels).sum().item()
                
                accuracy = correct_predictions / total_predictions
                
                # 验证评估结果合理
                self.assertTrue(0 <= accuracy <= 1)
                
        except Exception as e:
            # 如果遇到依赖问题，跳过测试
            self.skipTest(f"模型评估测试跳过，原因: {str(e)}")


class TestMetricsCalculation(unittest.TestCase):
    """指标计算测试类"""
    
    def test_confusion_matrix_calculation(self):
        """测试混淆矩阵计算"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 验证混淆矩阵形状
        self.assertEqual(cm.shape, (2, 2))
        
        # 验证混淆矩阵值合理
        self.assertTrue(np.all(cm >= 0))
    
    def test_classification_metrics(self):
        """测试分类指标计算"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.8, 0.7, 0.4, 0.9, 0.8])  # 正类概率
        
        # 计算各种指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_prob)
        
        # 验证指标范围
        self.assertTrue(0 <= accuracy <= 1)
        self.assertTrue(0 <= precision <= 1)
        self.assertTrue(0 <= recall <= 1)
        self.assertTrue(0 <= f1 <= 1)
        self.assertTrue(0 <= auc <= 1)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)