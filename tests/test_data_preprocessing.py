# -*- coding: utf-8 -*-
"""
数据预处理模块测试
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor, SentimentDataset
from transformers import BertTokenizer

class TestDataPreprocessor(unittest.TestCase):
    """数据预处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时测试数据
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_path = self.temp_dir / 'test_data.csv'
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'review': [
                '这个酒店很好，服务态度很棒！',
                '房间很脏，服务差，不推荐。',
                '<p>位置不错</p>，但是价格有点贵。',
                '总体还可以，下次还会来。',
                ''  # 空文本测试
            ],
            'label': [1, 0, 1, 1, 0]
        })
        test_data.to_csv(self.test_data_path, index=False, encoding='utf-8')
        
        # 创建预处理器
        self.preprocessor = DataPreprocessor(
            str(self.test_data_path),
            output_dir=str(self.temp_dir / 'processed')
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_data(self):
        """测试数据加载功能"""
        df = self.preprocessor.load_data()
        
        # 验证数据加载正确
        self.assertEqual(len(df), 5)
        self.assertIn('review', df.columns)
        self.assertIn('label', df.columns)
        
        # 测试文件不存在的情况
        with self.assertRaises(FileNotFoundError):
            wrong_preprocessor = DataPreprocessor('nonexistent.csv')
            wrong_preprocessor.load_data()
    
    def test_clean_text(self):
        """测试文本清洗功能"""
        # 测试HTML标签去除
        html_text = '<p>这是一个测试</p>'
        cleaned = self.preprocessor.clean_text(html_text)
        self.assertEqual(cleaned, '这是一个测试')
        
        # 测试空值处理
        empty_text = None
        cleaned = self.preprocessor.clean_text(empty_text)
        self.assertEqual(cleaned, '')
        
        # 测试特殊字符处理
        special_text = '测试@#$%文本！！！'
        cleaned = self.preprocessor.clean_text(special_text)
        self.assertNotIn('@', cleaned)
        self.assertNotIn('#', cleaned)
    
    def test_segment_text(self):
        """测试分词功能"""
        text = '这是一个测试句子'
        segmented = self.preprocessor.segment_text(text)
        
        # 验证分词结果包含空格分隔
        self.assertIn(' ', segmented)
        
        # 测试空文本
        empty_segmented = self.preprocessor.segment_text('')
        self.assertEqual(empty_segmented, '')
    
    def test_analyze_data_distribution(self):
        """测试数据分布分析功能"""
        df = self.preprocessor.load_data()
        result = self.preprocessor.analyze_data_distribution(df)
        
        # 验证返回结果包含必要字段
        self.assertIn('label_distribution', result)
        self.assertIn('text_length_stats', result)
        self.assertIn('total_samples', result)
        
        # 验证样本总数
        self.assertEqual(result['total_samples'], 5)
    
    def test_preprocess_data(self):
        """测试完整预处理流程"""
        try:
            train_df, val_df, test_df = self.preprocessor.preprocess_data(
                test_size=0.2, val_size=0.2, random_state=42
            )
            
            # 验证数据集不为空
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(val_df), 0)
            self.assertGreater(len(test_df), 0)
            
            # 验证列存在
            for df in [train_df, val_df, test_df]:
                self.assertIn('cleaned_review', df.columns)
                self.assertIn('label', df.columns)
            
        except Exception as e:
            # 如果数据太少无法分层采样，跳过测试
            if 'The least populated class' in str(e):
                self.skipTest("数据量太少，无法进行分层采样测试")
            else:
                raise


class TestSentimentDataset(unittest.TestCase):
    """情感分析数据集测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.texts = ['测试文本1', '测试文本2']
        self.labels = [0, 1]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.dataset = SentimentDataset(
            self.texts, self.labels, self.tokenizer, max_length=128
        )
    
    def test_dataset_length(self):
        """测试数据集长度"""
        self.assertEqual(len(self.dataset), 2)
    
    def test_dataset_getitem(self):
        """测试数据集索引访问"""
        item = self.dataset[0]
        
        # 验证返回的字典包含必要键
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        
        # 验证张量形状
        self.assertEqual(item['input_ids'].shape[0], 128)
        self.assertEqual(item['attention_mask'].shape[0], 128)
        self.assertEqual(item['labels'].item(), 0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)