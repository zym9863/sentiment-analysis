# -*- coding: utf-8 -*-
"""
工具函数模块
提供训练和评估过程中需要的辅助功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
import os
from pathlib import Path
import json
import pickle
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging(log_level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model=None):
        if self.best is None:
            self.best = val_loss
            if model is not None:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
            if model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping after {self.wait} epochs without improvement")

class ModelCheckpoint:
    """模型检查点"""
    
    def __init__(self, filepath, monitor='val_f1', mode='max', verbose=False):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = None
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best
            self.best = -np.inf
        else:
            self.is_better = lambda current, best: current < best
            self.best = np.inf
    
    def __call__(self, current_value, model, optimizer=None, epoch=None):
        if self.is_better(current_value, self.best):
            if self.verbose:
                print(f"Saving model to {self.filepath}")
            
            self.best = current_value
            
            # 创建保存目录
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best,
                'timestamp': datetime.now().isoformat()
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, self.filepath)

class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """计算各种评估指标"""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # 加权平均指标
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # 宏平均指标
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        }
        
        # AUC（如果提供了概率）
        if y_prob is not None:
            if self.num_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auc'] = auc
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovo')
                metrics['auc'] = auc
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 分类报告
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        metrics['classification_report'] = report
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, figsize=(8, 6)):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_classification_report(self, y_true, y_pred, save_path=None, figsize=(10, 6)):
        """绘制分类报告热力图"""
        from sklearn.metrics import classification_report
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # 提取数值数据
        data = []
        labels = []
        
        for class_name in self.class_names:
            if class_name in report:
                data.append([
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ])
                labels.append(class_name)
        
        data = np.array(data)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            data, annot=True, fmt='.3f', cmap='RdYlBu_r',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=labels,
            vmin=0, vmax=1
        )
        
        plt.title('分类性能指标')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失', marker='o')
    axes[0, 0].plot(history['val_loss'], label='验证损失', marker='s')
    axes[0, 0].set_title('损失变化')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_acc'], label='训练准确率', marker='o')
    axes[0, 1].plot(history['val_acc'], label='验证准确率', marker='s')
    axes[0, 1].set_title('准确率变化')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 0].plot(history['learning_rate'], label='学习率', marker='d')
    axes[1, 0].set_title('学习率变化')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 综合对比
    axes[1, 1].plot(history['train_loss'], label='训练损失', alpha=0.7)
    axes[1, 1].plot(history['val_loss'], label='验证损失', alpha=0.7)
    ax2 = axes[1, 1].twinx()
    ax2.plot(history['train_acc'], label='训练准确率', color='green', alpha=0.7)
    ax2.plot(history['val_acc'], label='验证准确率', color='red', alpha=0.7)
    axes[1, 1].set_title('训练概览')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_model_info(model, save_path):
    """保存模型信息"""
    info = {
        'model_class': model.__class__.__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'model_structure': str(model),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def load_checkpoint(model, checkpoint_path, device='cpu'):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型状态字典和检查点状态字典
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint['model_state_dict']
    
    # 检查是否有缺失的键
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state_dict.keys():
        if key not in checkpoint_state_dict:
            missing_keys.append(key)
    
    for key in checkpoint_state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    
    # 只加载匹配的参数
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() 
                          if k in model_state_dict}
    
    # 使用strict=False来允许部分加载
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # 获取logger实例
    logger = logging.getLogger(__name__)
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
    
    return checkpoint.get('best_score', None), checkpoint.get('epoch', None)

def set_seed(seed=42):
    """设置随机种子"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AttentionVisualizer:
    """注意力权重可视化器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def visualize_attention(self, text, attention_weights, layer_idx=0, head_idx=0, 
                          save_path=None, figsize=(12, 8)):
        """可视化注意力权重"""
        # 分词
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 获取特定层和头的注意力权重
        attention = attention_weights[layer_idx, head_idx].cpu().numpy()
        
        # 截取到实际token长度
        seq_len = min(len(tokens), attention.shape[0])
        attention = attention[:seq_len, :seq_len]
        tokens = tokens[:seq_len]
        
        # 绘制热力图
        plt.figure(figsize=figsize)
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            cbar=True
        )
        
        plt.title(f'注意力权重可视化 (Layer {layer_idx}, Head {head_idx})')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def analyze_model_predictions(model, data_loader, tokenizer, device, num_samples=10):
    """分析模型预测结果"""
    model.eval()
    samples_analyzed = 0
    results = []
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_analyzed >= num_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
            
            for i in range(input_ids.size(0)):
                if samples_analyzed >= num_samples:
                    break
                    
                # 解码文本
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                
                result = {
                    'text': text,
                    'true_label': labels[i].item(),
                    'predicted_label': predictions[i].item(),
                    'confidence': probabilities[i].max().item(),
                    'probabilities': probabilities[i].cpu().numpy().tolist()
                }
                
                results.append(result)
                samples_analyzed += 1
    
    return results
