"""
模型模块
包含传统机器学习和深度学习模型
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from typing import Optional
import joblib


# ==================== 传统机器学习模型 ====================

class TraditionalModels:
    """传统机器学习模型集合"""
    
    @staticmethod
    def get_naive_bayes(alpha: float = 1.0):
        """朴素贝叶斯分类器"""
        return MultinomialNB(alpha=alpha)
    
    @staticmethod
    def get_svm(C: float = 1.0, max_iter: int = 1000):
        """支持向量机（线性核）"""
        return LinearSVC(C=C, max_iter=max_iter, random_state=42)
    
    @staticmethod
    def get_logistic_regression(C: float = 1.0, max_iter: int = 1000):
        """逻辑回归"""
        return LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    
    @staticmethod
    def get_random_forest(n_estimators: int = 100, max_depth: Optional[int] = None):
        """随机森林"""
        return RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42,
            n_jobs=-1
        )


# ==================== 深度学习模型 ====================

class TextCNN(nn.Module):
    """TextCNN模型"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: tuple = (2, 3, 4),
        dropout: float = 0.5,
        num_classes: int = 2
    ):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch_size, num_filters, *)
            pooled = torch.max(conv_out, dim=2)[0]  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # 拼接所有卷积输出
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class LSTMClassifier(nn.Module):
    """LSTM分类器"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        num_classes: int = 2
    ):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        
        return out


class Vocabulary:
    """词汇表类，用于深度学习模型"""
    
    def __init__(self, max_size: int = 10000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.max_size = max_size
        
    def build(self, texts):
        """
        构建词汇表
        
        Args:
            texts: 预处理后的文本列表（空格分隔的词语）
        """
        from collections import Counter
        
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(self.max_size - 2)  # 留出PAD和UNK
        
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"词汇表构建完成，大小: {len(self.word2idx)}")
        
    def text_to_indices(self, text: str, max_len: int = 128):
        """
        将文本转换为索引序列
        
        Args:
            text: 预处理后的文本
            max_len: 最大序列长度
            
        Returns:
            索引列表
        """
        words = text.split()
        indices = [self.word2idx.get(w, 1) for w in words]  # 1是UNK
        
        # 截断或填充
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [0] * (max_len - len(indices))  # 0是PAD
        
        return indices
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """保存词汇表"""
        joblib.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, filepath)
        
    def load(self, filepath: str):
        """加载词汇表"""
        data = joblib.load(filepath)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
