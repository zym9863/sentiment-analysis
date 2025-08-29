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

# 导入配置管理器
from .config import get_config

# 获取配置
config = get_config()

# 设置日志
logging.basicConfig(level=getattr(logging, config.system_config.log_level))
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """情感分析数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or config.data_config.max_length
    
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
    
    def __init__(self, data_path=None, output_dir=None):
        self.data_path = data_path or config.data_config.data_path
        self.output_dir = Path(output_dir or config.data_config.processed_data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.model_config.model_name)
        
    def load_data(self):
        """加载原始数据"""
        logger.info("加载ChnSentiCorp数据集...")
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
            logger.info(f"数据集大小: {len(df)}")
            return df
        except FileNotFoundError:
            logger.error(f"数据文件未找到: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("数据文件为空")
            raise
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def clean_text(self, text):
        """文本清洗"""
        if pd.isna(text):
            return ""
        
        try:
            # 去除HTML标签
            text = re.sub(r'<[^>]+>', '', text)
            
            # 去除特殊字符，保留中文、英文、数字和基本标点
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】\s]', '', text)
            
            # 去除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            logger.warning(f"文本清洗失败: {e}")
            return str(text) if text is not None else ""
    
    def analyze_data_distribution(self, df):
        """分析数据分布"""
        logger.info("分析数据分布...")
        
        try:
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
        except Exception as e:
            logger.error(f"数据分布分析失败: {e}")
            return {
                'label_distribution': {},
                'text_length_stats': {},
                'total_samples': 0
            }
    
    def segment_text(self, text):
        """中文分词"""
        if not text:
            return ""
        
        try:
            # 使用jieba分词
            words = jieba.cut(text)
            return ' '.join(words)
        except Exception as e:
            logger.warning(f"分词失败: {e}")
            return str(text)
    
    def preprocess_data(self, test_size=None, val_size=None, random_state=None):
        """完整的数据预处理流程"""
        logger.info("开始数据预处理...")
        
        # 使用配置中的默认值
        test_size = test_size or config.data_config.test_size
        val_size = val_size or config.data_config.val_size
        random_state = random_state or config.data_config.random_state
        
        try:
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
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            raise
    
    def create_dataloaders(self, batch_size=None, max_length=None):
        """创建DataLoader"""
        logger.info("创建DataLoader...")
        
        # 使用配置中的默认值
        batch_size = batch_size or config.data_config.batch_size
        max_length = max_length or config.data_config.max_length
        num_workers = config.system_config.dataloader_num_workers
        
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
            num_workers=num_workers,
            pin_memory=config.system_config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.system_config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.system_config.pin_memory
        )
        
        return train_loader, val_loader, test_loader

def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    向后兼容的数据加载和预处理函数
    返回字典格式的数据，包含文本和标签
    """
    # 创建预处理器实例
    preprocessor = DataPreprocessor(data_path)
    
    # 检查是否已有处理好的数据
    processed_dir = Path("data/processed")
    train_path = processed_dir / 'train.csv'
    val_path = processed_dir / 'val.csv'
    test_path = processed_dir / 'test.csv'
    
    if train_path.exists() and val_path.exists() and test_path.exists():
        logger.info("加载已处理的数据...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        logger.info("重新处理数据...")
        train_df, val_df, test_df = preprocessor.preprocess_data(test_size, val_size, random_state)
    
    # 转换为字典格式
    def df_to_dict_list(df):
        data_list = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        for _, row in df.iterrows():
            text = str(row['cleaned_review'])
            label = int(row['label'])
            
            # BERT编码
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            data_dict = {
                'text': text,
                'label': label,
                'input_ids': encoding['input_ids'].flatten().numpy(),
                'attention_mask': encoding['attention_mask'].flatten().numpy()
            }
            data_list.append(data_dict)
        
        return data_list
    
    train_data = df_to_dict_list(train_df)
    val_data = df_to_dict_list(val_df)
    test_data = df_to_dict_list(test_df)
    
    logger.info(f"数据加载完成 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}")
    
    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, batch_size=16):
    """
    向后兼容的DataLoader创建函数
    """
    # 创建自定义数据集类用于字典格式数据
    class DictDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(item['label'], dtype=torch.long)
            }
    
    # 创建数据集
    train_dataset = DictDataset(train_data)
    val_dataset = DictDataset(val_data)
    test_dataset = DictDataset(test_data)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows系统建议设为0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
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
