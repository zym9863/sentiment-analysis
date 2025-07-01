# -*- coding: utf-8 -*-
"""
基础BERT模型
使用预训练的bert-base-chinese进行情感分类
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import logging

logger = logging.getLogger(__name__)

class BaseBertModel(nn.Module):
    """基础BERT情感分析模型"""
    
    def __init__(self, model_name='bert-base-chinese', num_classes=2, dropout_rate=0.1):
        super(BaseBertModel, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        
        # 获取BERT的隐藏层维度
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # 初始化分类器权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS]标记的表示进行分类
        pooled_output = outputs.pooler_output
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'hidden_states': outputs.last_hidden_state,
            'pooled_output': pooled_output
        }
    
    def get_embeddings(self, input_ids, attention_mask=None):
        """获取文本embeddings"""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return outputs.last_hidden_state

class BertForSequenceClassification(nn.Module):
    """BERT序列分类模型 - 增强版"""
    
    def __init__(self, model_name='bert-base-chinese', num_classes=2, 
                 dropout_rate=0.1, use_layer_norm=True):
        super(BertForSequenceClassification, self).__init__()
        
        self.num_classes = num_classes
        self.use_layer_norm = use_layer_norm
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # 层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS]标记的表示
        pooled_output = outputs.pooler_output
        
        # 层归一化
        if self.use_layer_norm:
            pooled_output = self.layer_norm(pooled_output)
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'hidden_states': outputs.last_hidden_state,
            'pooled_output': pooled_output
        }

def create_base_model(model_name='bert-base-chinese', num_classes=2, model_type='simple'):
    """创建基础BERT模型"""
    if model_type == 'simple':
        model = BaseBertModel(model_name, num_classes)
    elif model_type == 'enhanced':
        model = BertForSequenceClassification(model_name, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} BERT model with {num_classes} classes")
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
    model = create_base_model(model_type='enhanced')
    
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
        print(f"Hidden states shape: {outputs['hidden_states'].shape}")
