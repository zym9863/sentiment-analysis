"""
数据加载模块
负责加载ChnSentiCorp数据集并进行数据划分
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    加载ChnSentiCorp数据集
    
    Args:
        filepath: CSV文件路径
        
    Returns:
        包含label和review列的DataFrame
    """
    df = pd.read_csv(filepath)
    # 移除缺失值
    df = df.dropna(subset=['review', 'label'])
    # 确保label为整数类型
    df['label'] = df['label'].astype(int)
    print(f"数据集加载完成: {len(df)} 条评论")
    print(f"正面评论: {(df['label'] == 1).sum()} 条")
    print(f"负面评论: {(df['label'] == 0).sum()} 条")
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        df: 原始数据集
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子
        
    Returns:
        (训练集, 验证集, 测试集)
    """
    # 先划分出测试集
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # 再从剩余数据中划分验证集
    actual_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=actual_val_size, random_state=random_state, 
        stratify=train_val['label']
    )
    
    print(f"\n数据集划分完成:")
    print(f"  训练集: {len(train)} 条")
    print(f"  验证集: {len(val)} 条")
    print(f"  测试集: {len(test)} 条")
    
    return train, val, test


def get_data_statistics(df: pd.DataFrame) -> dict:
    """
    获取数据集统计信息
    
    Args:
        df: 数据集
        
    Returns:
        统计信息字典
    """
    # 计算文本长度
    df['text_length'] = df['review'].apply(len)
    
    stats = {
        'total_samples': len(df),
        'positive_samples': (df['label'] == 1).sum(),
        'negative_samples': (df['label'] == 0).sum(),
        'positive_ratio': (df['label'] == 1).mean(),
        'avg_text_length': df['text_length'].mean(),
        'max_text_length': df['text_length'].max(),
        'min_text_length': df['text_length'].min(),
        'median_text_length': df['text_length'].median()
    }
    
    return stats
