"""
特征提取模块
支持TF-IDF和词袋模型特征提取
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Tuple, Optional
import joblib


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(
        self,
        method: str = 'tfidf',
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        初始化特征提取器
        
        Args:
            method: 特征提取方法，'tfidf' 或 'bow'（词袋）
            max_features: 最大特征数量
            ngram_range: n-gram范围
            min_df: 最小文档频率
            max_df: 最大文档频率
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        
    def fit(self, texts):
        """
        拟合特征提取器
        
        Args:
            texts: 预处理后的文本列表
        """
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df
            )
        elif self.method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df
            )
        else:
            raise ValueError(f"不支持的特征提取方法: {self.method}")
        
        self.vectorizer.fit(texts)
        print(f"特征提取器拟合完成，特征维度: {len(self.vectorizer.vocabulary_)}")
        
    def transform(self, texts):
        """
        转换文本为特征向量
        
        Args:
            texts: 预处理后的文本列表
            
        Returns:
            特征矩阵
        """
        if self.vectorizer is None:
            raise ValueError("请先调用fit方法拟合特征提取器")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        拟合并转换文本
        
        Args:
            texts: 预处理后的文本列表
            
        Returns:
            特征矩阵
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """获取特征名称列表"""
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath: str):
        """保存特征提取器"""
        joblib.dump(self.vectorizer, filepath)
        print(f"特征提取器已保存到: {filepath}")
        
    def load(self, filepath: str):
        """加载特征提取器"""
        self.vectorizer = joblib.load(filepath)
        print(f"特征提取器已从 {filepath} 加载")
