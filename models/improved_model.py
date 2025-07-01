# -*- coding: utf-8 -*-
"""
改进的BERT+BiLSTM+Attention模型
结合BERT、双向LSTM和多头注意力机制的情感分析模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import math
import logging

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        output = self.W_o(context)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ImprovedSentimentModel(nn.Module):
    """改进的情感分析模型: BERT + BiLSTM + Multi-Head Attention"""
    
    def __init__(self, 
                 model_name='bert-base-chinese',
                 num_classes=2,
                 lstm_hidden_size=256,
                 lstm_num_layers=2,
                 num_attention_heads=8,
                 dropout_rate=0.1,
                 use_contrastive_loss=True):
        super(ImprovedSentimentModel, self).__init__()
        
        self.num_classes = num_classes
        self.use_contrastive_loss = use_contrastive_loss
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # 冻结BERT的部分层（可选）
        # self._freeze_bert_layers(freeze_layers=6)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            self.bert_hidden_size,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.lstm_output_size = lstm_hidden_size * 2  # 双向LSTM
        
        # 多头注意力机制
        self.attention = MultiHeadAttention(
            self.lstm_output_size,
            num_attention_heads,
            dropout_rate
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.lstm_output_size)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(self.lstm_output_size)
        self.layer_norm2 = nn.LayerNorm(self.lstm_output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.bert_hidden_size + self.lstm_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Linear(256, num_classes)
        
        # 对比学习用的投影头
        if use_contrastive_loss:
            self.projection_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        
        self._init_weights()
    
    def _freeze_bert_layers(self, freeze_layers=6):
        """冻结BERT的前几层"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    def _init_weights(self):
        """初始化权重"""
        # 初始化特征融合层
        for layer in self.feature_fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # 初始化分类器
        if isinstance(self.classifier, nn.Linear):
            nn.init.xavier_normal_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        
        # 初始化投影头（如果存在）
        if hasattr(self, 'projection_head'):
            for layer in self.projection_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播"""
        batch_size = input_ids.size(0)
        
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        bert_hidden_states = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        bert_pooled_output = bert_outputs.pooler_output      # [batch_size, hidden_size]
        
        # BiLSTM编码
        lstm_output, (hidden, cell) = self.lstm(bert_hidden_states)
        
        # 位置编码
        lstm_output = self.pos_encoding(lstm_output)
        lstm_output = self.layer_norm1(lstm_output)
        
        # 多头注意力
        attention_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output, attention_mask
        )
        
        # 残差连接
        attention_output = lstm_output + self.dropout(attention_output)
        attention_output = self.layer_norm2(attention_output)
        
        # 全局平均池化
        if attention_mask is not None:
            # 考虑padding mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(attention_output.size())
            attention_output = attention_output * mask_expanded
            pooled_attention = attention_output.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_attention = attention_output.mean(dim=1)
        
        # 特征融合
        fused_features = torch.cat([bert_pooled_output, pooled_attention], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        # 准备输出
        outputs = {
            'logits': logits,
            'bert_hidden_states': bert_hidden_states,
            'lstm_output': lstm_output,
            'attention_weights': attention_weights,
            'fused_features': fused_features
        }
        
        # 对比学习的特征
        if self.use_contrastive_loss:
            projection = self.projection_head(fused_features)
            outputs['projection'] = F.normalize(projection, dim=1)
        
        return outputs

class ContrastiveLoss(nn.Module):
    """对比学习损失函数 - 修复了原地操作问题"""
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: 归一化的特征向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]
        """
        batch_size = features.size(0)
        device = features.device
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码（相同标签为正样本）
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_eq = (labels_expanded == labels_expanded.T).float()
        
        # 创建对角线掩码（去除自身）
        identity_mask = torch.eye(batch_size, device=device).bool()
        
        # 使用where操作代替masked_fill避免原地操作
        labels_eq = torch.where(identity_mask, torch.zeros_like(labels_eq), labels_eq)
        
        # 计算exp相似度，避免对角线
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = torch.where(identity_mask, torch.zeros_like(exp_sim), exp_sim)
        
        # 计算正样本和负样本
        pos_sum = (exp_sim * labels_eq).sum(dim=1)
        neg_sum = exp_sim.sum(dim=1)
        
        # 防止除零和log(0)
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        neg_sum = torch.clamp(neg_sum, min=1e-8)
        
        # 计算对比损失
        loss = -torch.log(pos_sum / neg_sum)
        
        # 只对有正样本的样本计算损失
        valid_mask = pos_sum > 1e-8
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

def create_improved_model(model_name='bert-base-chinese', 
                         num_classes=2, 
                         **kwargs):
    """创建改进的模型"""
    model = ImprovedSentimentModel(
        model_name=model_name,
        num_classes=num_classes,
        **kwargs
    )
    
    logger.info(f"Created improved BERT+BiLSTM+Attention model")
    return model

def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }

if __name__ == "__main__":
    # 测试模型
    model = create_improved_model()
    
    # 计算参数数量
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    
    # 测试前向传播
    batch_size = 2
    seq_length = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Attention weights shape: {outputs['attention_weights'].shape}")
        if 'projection' in outputs:
            print(f"Projection shape: {outputs['projection'].shape}")
    
    # 测试对比学习损失
    contrastive_loss = ContrastiveLoss()
    features = torch.randn(4, 64)
    features = F.normalize(features, dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    
    loss = contrastive_loss(features, labels)
    print(f"Contrastive loss: {loss.item():.4f}")
