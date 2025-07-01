# -*- coding: utf-8 -*-
"""
模型训练模块
支持基础BERT模型和改进模型的训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import logging
import json
import os
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# 本地模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import create_base_model
from models.improved_model import create_improved_model, ContrastiveLoss
from src.data_preprocessing import DataPreprocessor
from src.utils import EarlyStopping, ModelCheckpoint, setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, 
                 model_type='improved',
                 model_name='bert-base-chinese',
                 num_classes=2,
                 learning_rate=2e-5,
                 batch_size=16,
                 max_epochs=10,
                 patience=3,
                 device=None,
                 save_dir='models/saved_models',
                 use_contrastive_loss=True,
                 contrastive_weight=0.1):
        
        self.model_type = model_type
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_weight = contrastive_weight
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # 保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        if self.use_contrastive_loss:
            self.contrastive_criterion = ContrastiveLoss()
        
        # 优化器
        self.optimizer = self._setup_optimizer()
        
        # 学习率调度器
        self.scheduler = None
        
        # 早停和模型检查点
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.checkpoint = ModelCheckpoint(
            self.save_dir / f'best_{model_type}_model.pth', 
            verbose=True
        )
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def _create_model(self):
        """创建模型"""
        if self.model_type == 'base':
            model = create_base_model(
                model_name=self.model_name,
                num_classes=self.num_classes,
                model_type='enhanced'
            )
        elif self.model_type == 'improved':
            model = create_improved_model(
                model_name=self.model_name,
                num_classes=self.num_classes,
                use_contrastive_loss=self.use_contrastive_loss
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    def _setup_optimizer(self):
        """设置优化器"""
        # 分别设置BERT和其他层的学习率
        bert_param_ids = set(id(p) for p in self.model.bert.parameters())
        
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if id(param) in bert_param_ids:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': bert_params, 'lr': self.learning_rate * 0.1},  # BERT层使用较小学习率
            {'params': other_params, 'lr': self.learning_rate}
        ], weight_decay=0.01)
        
        return optimizer
    
    def _setup_scheduler(self, train_loader):
        """设置学习率调度器"""
        total_steps = len(train_loader) * self.max_epochs
        
        # 使用余弦退火调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps,
            eta_min=1e-7
        )
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            # 数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # 计算损失
            ce_loss = self.criterion(logits, labels)
            total_loss_batch = ce_loss
            
            # 对比学习损失
            if self.use_contrastive_loss and 'projection' in outputs:
                contrastive_loss = self.contrastive_criterion(
                    outputs['projection'], labels
                )
                total_loss_batch = ce_loss + self.contrastive_weight * contrastive_loss
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # 统计
            total_loss += total_loss_batch.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        # 计算详细指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader):
        """完整的训练流程"""
        logger.info("开始训练...")
        
        # 设置学习率调度器
        self._setup_scheduler(train_loader)
        
        best_val_f1 = 0
        
        for epoch in range(self.max_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.max_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate_epoch(val_loader)
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rate'].append(current_lr)
            
            # 打印结果
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # 模型检查点
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.checkpoint(val_f1, self.model, self.optimizer, epoch)
            
            # 早停检查
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
        
        # 保存训练历史
        self._save_training_history()
        
        logger.info("训练完成！")
        return self.train_history
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = self.save_dir / f'{self.model_type}_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--model_type', type=str, default='improved', 
                       choices=['base', 'improved'], help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning')
    
    args = parser.parse_args()
    
    # 数据预处理
    logger.info("Loading and preprocessing data...")
    preprocessor = DataPreprocessor('ChnSentiCorp_htl_all.csv')
    
    # 检查是否已有处理后的数据
    if not (Path('data/processed/train.csv').exists()):
        preprocessor.preprocess_data()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        batch_size=args.batch_size
    )
    
    # 创建训练器
    trainer = ModelTrainer(
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        use_contrastive_loss=args.use_contrastive
    )
    
    # 开始训练
    history = trainer.train(train_loader, val_loader)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
