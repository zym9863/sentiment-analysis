# -*- coding: utf-8 -*-
"""
数据预处理模块
负责ChnSentiCorp数据集的加载、清洗、分词和预处理
"""

import pandas as pd
import numpy as np
import re
import jieba
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """情感分析数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # BERT编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_path, output_dir="data/processed"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
    def load_data(self):
        """加载原始数据"""
        logger.info("加载ChnSentiCorp数据集...")
        df = pd.read_csv(self.data_path, encoding='utf-8')
        logger.info(f"数据集大小: {len(df)}")
        return df
    
    def clean_text(self, text):
        """文本清洗"""
        if pd.isna(text):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除特殊字符，保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】\s]', '', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_data_distribution(self, df):
        """分析数据分布"""
        logger.info("分析数据分布...")
        
        # 标签分布
        label_counts = df['label'].value_counts()
        logger.info(f"标签分布:\n{label_counts}")
        
        # 文本长度分布
        df['text_length'] = df['review'].astype(str).apply(len)
        length_stats = df['text_length'].describe()
        logger.info(f"文本长度统计:\n{length_stats}")
        
        # 保存分析结果
        analysis_result = {
            'label_distribution': label_counts.to_dict(),
            'text_length_stats': length_stats.to_dict(),
            'total_samples': len(df)
        }
        
        with open(self.output_dir / 'data_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        return analysis_result
    
    def segment_text(self, text):
        """中文分词"""
        if not text:
            return ""
        
        # 使用jieba分词
        words = jieba.cut(text)
        return ' '.join(words)
    
    def preprocess_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """完整的数据预处理流程"""
        logger.info("开始数据预处理...")
        
        # 加载数据
        df = self.load_data()
        
        # 数据分析
        self.analyze_data_distribution(df)
        
        # 数据清洗
        logger.info("清洗文本数据...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # 去除空文本
        df = df[df['cleaned_review'].str.len() > 0].reset_index(drop=True)
        logger.info(f"清洗后数据集大小: {len(df)}")
        
        # 分词（可选，BERT会自己处理）
        logger.info("进行中文分词...")
        df['segmented_review'] = df['cleaned_review'].apply(self.segment_text)
        
        # 数据集划分
        logger.info("划分训练集、验证集和测试集...")
        
        # 先划分训练集和临时集
        train_df, temp_df = train_test_split(
            df, test_size=test_size+val_size, 
            stratify=df['label'], random_state=random_state
        )
        
        # 再从临时集划分验证集和测试集
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=1-val_ratio, 
            stratify=temp_df['label'], random_state=random_state
        )
        
        logger.info(f"训练集大小: {len(train_df)}")
        logger.info(f"验证集大小: {len(val_df)}")
        logger.info(f"测试集大小: {len(test_df)}")
        
        # 保存处理后的数据
        train_df.to_csv(self.output_dir / 'train.csv', index=False, encoding='utf-8')
        val_df.to_csv(self.output_dir / 'val.csv', index=False, encoding='utf-8')
        test_df.to_csv(self.output_dir / 'test.csv', index=False, encoding='utf-8')
        
        logger.info("数据预处理完成！")
        
        return train_df, val_df, test_df
    
    def create_dataloaders(self, batch_size=16, max_length=512):
        """创建DataLoader"""
        logger.info("创建DataLoader...")
        
        # 加载处理后的数据
        train_df = pd.read_csv(self.output_dir / 'train.csv')
        val_df = pd.read_csv(self.output_dir / 'val.csv')
        test_df = pd.read_csv(self.output_dir / 'test.csv')
        
        # 创建数据集
        train_dataset = SentimentDataset(
            train_df['cleaned_review'].values,
            train_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        val_dataset = SentimentDataset(
            val_df['cleaned_review'].values,
            val_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        test_dataset = SentimentDataset(
            test_df['cleaned_review'].values,
            test_df['label'].values,
            self.tokenizer,
            max_length
        )
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader, test_loader

def main():
    """主函数"""
    # 初始化预处理器
    preprocessor = DataPreprocessor('ChnSentiCorp_htl_all.csv')
    
    # 预处理数据
    train_df, val_df, test_df = preprocessor.preprocess_data()
    
    # 创建DataLoader
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders()
    
    logger.info("数据预处理完成！")

if __name__ == "__main__":
    main()
