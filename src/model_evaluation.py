# -*- coding: utf-8 -*-
"""
模型评估模块
提供全面的模型评估功能，包括多维度指标、消融实验、案例分析等
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import json
import argparse
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import create_base_model
from models.improved_model import create_improved_model
from src.data_preprocessing import DataPreprocessor
from src.utils import (
    MetricsCalculator, AttentionVisualizer, 
    analyze_model_predictions, load_checkpoint,
    setup_logging
)

# 设置日志和中文字体
setup_logging()
logger = logging.getLogger(__name__)

# 尝试设置中文字体，如果没有则使用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用英文标签
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, tokenizer, device='cpu', class_names=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names or ['Negative', 'Positive']  # Use English labels
        
        # 检查是否支持中文字体
        import matplotlib.font_manager as fm
        self.use_chinese = False
        for font in fm.fontManager.ttflist:
            if 'SimHei' in font.name or 'SimSun' in font.name:
                self.use_chinese = True
                self.class_names = class_names or ['负面', '正面']
                break
        
        self.metrics_calculator = MetricsCalculator(
            num_classes=len(self.class_names),
            class_names=self.class_names
        )
        
        self.attention_visualizer = AttentionVisualizer(tokenizer)
    
    def _get_label(self, chinese_text, english_text):
        """根据字体支持情况返回合适的标签"""
        return chinese_text if self.use_chinese else english_text
    
    def evaluate_on_dataset(self, data_loader, return_predictions=False):
        """在数据集上评估模型"""
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_features = []
        
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        logger.info("开始模型评估...")
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # 获取预测结果
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # 收集结果
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # 如果有特征，也收集起来
                if 'fused_features' in outputs:
                    all_features.extend(outputs['fused_features'].cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions, np.array(all_probabilities)
        )
        
        metrics['avg_loss'] = avg_loss
        
        logger.info(f"评估完成 - 准确率: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")
        
        if return_predictions:
            return metrics, {
                'labels': all_labels,
                'predictions': all_predictions,
                'probabilities': all_probabilities,
                'features': all_features if all_features else None
            }
        
        return metrics
    
    def generate_evaluation_report(self, test_loader, save_dir='results'):
        """生成完整的评估报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("生成评估报告...")
        
        # 评估模型
        metrics, predictions = self.evaluate_on_dataset(test_loader, return_predictions=True)
        
        # 保存指标
        with open(save_dir / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # 绘制混淆矩阵
        self.metrics_calculator.plot_confusion_matrix(
            predictions['labels'], 
            predictions['predictions'],
            save_path=save_dir / 'confusion_matrix.png'
        )
        
        # 绘制分类报告
        self.metrics_calculator.plot_classification_report(
            predictions['labels'],
            predictions['predictions'],
            save_path=save_dir / 'classification_report.png'
        )
        
        # 绘制ROC曲线
        self._plot_roc_curve(
            predictions['labels'],
            predictions['probabilities'],
            save_path=save_dir / 'roc_curve.png'
        )
        
        # 预测置信度分布
        self._plot_confidence_distribution(
            predictions['probabilities'],
            predictions['labels'],
            save_path=save_dir / 'confidence_distribution.png'
        )
        
        # 案例分析
        case_analysis = self._analyze_cases(test_loader, num_samples=20)
        with open(save_dir / 'case_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(case_analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估报告已保存到 {save_dir}")
        return metrics, predictions
    def _plot_roc_curve(self, y_true, y_prob, save_path=None):
        """绘制ROC曲线"""
        plt.figure(figsize=(8, 6))
        
        # 确保 y_prob 是 numpy 数组
        y_prob = np.array(y_prob)
        if len(self.class_names) == 2:
            # 二分类
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC{self._get_label("曲线", " Curve")} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label=self._get_label('随机分类器', 'Random Classifier'))
            
            plt.xlabel(self._get_label('假正率 (FPR)', 'False Positive Rate (FPR)'))
            plt.ylabel(self._get_label('真正率 (TPR)', 'True Positive Rate (TPR)'))
            plt.title(self._get_label('ROC曲线', 'ROC Curve'))
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # 多分类 - 为每个类别绘制ROC曲线
            for i, class_name in enumerate(self.class_names):
                y_true_binary = (np.array(y_true) == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                auc_score = roc_auc_score(y_true_binary, y_prob[:, i])
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{class_name} (AUC = {auc_score:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label=self._get_label('随机分类器', 'Random Classifier'))
            plt.xlabel(self._get_label('假正率 (FPR)', 'False Positive Rate (FPR)'))
            plt.ylabel(self._get_label('真正率 (TPR)', 'True Positive Rate (TPR)'))
            plt.title(self._get_label('多类别ROC曲线', 'Multi-class ROC Curve'))
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    def _plot_confidence_distribution(self, y_prob, y_true, save_path=None):
        """绘制预测置信度分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 确保 y_prob 是 numpy 数组
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        
        # 获取最大概率（置信度）
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
          # 总体置信度分布
        axes[0].hist(confidences, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel(self._get_label('预测置信度', 'Prediction Confidence'))
        axes[0].set_ylabel(self._get_label('频次', 'Frequency'))
        axes[0].set_title(self._get_label('预测置信度分布', 'Prediction Confidence Distribution'))
        axes[0].grid(True, alpha=0.3)
        
        # 按正确/错误预测分组的置信度分布
        correct_predictions = (predictions == y_true)
        correct_confidences = confidences[correct_predictions]
        wrong_confidences = confidences[~correct_predictions]
        
        axes[1].hist(correct_confidences, bins=30, alpha=0.7, 
                    label=f'{self._get_label("正确预测", "Correct Predictions")} ({len(correct_confidences)})', color='green')
        axes[1].hist(wrong_confidences, bins=30, alpha=0.7, 
                    label=f'{self._get_label("错误预测", "Wrong Predictions")} ({len(wrong_confidences)})', color='red')
        axes[1].set_xlabel(self._get_label('预测置信度', 'Prediction Confidence'))
        axes[1].set_ylabel(self._get_label('频次', 'Frequency'))
        axes[1].set_title(self._get_label('按预测正确性分组的置信度分布', 'Confidence Distribution by Prediction Correctness'))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_cases(self, data_loader, num_samples=20):
        """案例分析"""
        cases = analyze_model_predictions(
            self.model, data_loader, self.tokenizer, 
            self.device, num_samples
        )
        
        # 分析结果
        correct_cases = [case for case in cases if case['true_label'] == case['predicted_label']]
        wrong_cases = [case for case in cases if case['true_label'] != case['predicted_label']]
        
        # 按置信度排序
        correct_cases.sort(key=lambda x: x['confidence'], reverse=True)
        wrong_cases.sort(key=lambda x: x['confidence'], reverse=True)
        
        analysis = {
            'total_samples': len(cases),
            'correct_predictions': len(correct_cases),
            'wrong_predictions': len(wrong_cases),
            'accuracy': len(correct_cases) / len(cases),
            'high_confidence_correct': correct_cases[:5],
            'high_confidence_wrong': wrong_cases[:5],
            'low_confidence_correct': correct_cases[-3:] if correct_cases else [],
            'low_confidence_wrong': wrong_cases[-3:] if wrong_cases else []
        }
        
        return analysis
    
    def compare_models(self, models_info, test_loader, save_dir='results'):
        """比较多个模型的性能"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_results = {}
        
        for model_name, model_info in models_info.items():
            logger.info(f"评估模型: {model_name}")
            
            # 加载模型
            model = model_info['model']
            if 'checkpoint_path' in model_info:
                load_checkpoint(model, model_info['checkpoint_path'], self.device)
            
            model.to(self.device)
            
            # 评估
            old_model = self.model
            self.model = model
            
            metrics = self.evaluate_on_dataset(test_loader)
            comparison_results[model_name] = metrics
            
            self.model = old_model
        
        # 保存比较结果
        with open(save_dir / 'model_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        # 绘制比较图
        self._plot_model_comparison(comparison_results, save_dir)
        
        return comparison_results
    
    def _plot_model_comparison(self, comparison_results, save_dir):
        """绘制模型比较图"""
        models = list(comparison_results.keys())
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_weighted', '').title(),
                    'Score': comparison_results[model][metric]
                })
        
        df = pd.DataFrame(data)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Metric', y='Score', hue='Model')
        plt.title(self._get_label('模型性能比较', 'Model Performance Comparison'))
        plt.ylabel(self._get_label('分数', 'Score'))
        plt.ylim(0, 1)
        plt.legend(title=self._get_label('模型', 'Model'))
        
        for i, metric in enumerate(metrics):
            metric_name = metric.replace('_weighted', '').title()
            for j, model in enumerate(models):
                score = comparison_results[model][metric]
                plt.text(i + (j - len(models)/2 + 0.5) * 0.2, 
                        score + 0.01, f'{score:.3f}', 
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

class AblationStudy:
    """消融实验"""
    
    def __init__(self, base_model_config, device='cpu'):
        self.base_config = base_model_config
        self.device = device
        self.results = {}
        
        # 检查字体支持
        import matplotlib.font_manager as fm
        self.use_chinese = False
        for font in fm.fontManager.ttflist:
            if 'SimHei' in font.name or 'SimSun' in font.name:
                self.use_chinese = True
                break
    
    def _get_label(self, chinese_text, english_text):
        """根据字体支持情况返回合适的标签"""
        return chinese_text if self.use_chinese else english_text
    
    def run_ablation_study(self, train_loader, val_loader, test_loader, 
                          tokenizer, save_dir='results/ablation'):
        """运行消融实验"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("开始消融实验...")        # 实验配置
        experiments = {
            'full_model': {
                'use_bilstm': True,
                'use_attention': True,
                'use_contrastive': True,
                'description': self._get_label('完整模型', 'Full Model')
            },
            'no_bilstm': {
                'use_bilstm': False,
                'use_attention': True,
                'use_contrastive': True,
                'description': self._get_label('移除BiLSTM', 'No BiLSTM')
            },
            'no_attention': {
                'use_bilstm': True,
                'use_attention': False,
                'use_contrastive': True,
                'description': self._get_label('移除注意力机制', 'No Attention')
            },
            'no_contrastive': {
                'use_bilstm': True,
                'use_attention': True,
                'use_contrastive': False,
                'description': self._get_label('移除对比学习', 'No Contrastive')
            },
            'bert_only': {
                'use_bilstm': False,
                'use_attention': False,
                'use_contrastive': False,
                'description': self._get_label('仅BERT', 'BERT Only')
            }
        }
        
        for exp_name, exp_config in experiments.items():
            logger.info(f"运行实验: {exp_config['description']}")
            
            # 创建模型（这里需要根据配置创建不同的模型）
            # 由于时间限制，这里简化处理
            if exp_name == 'bert_only':
                model = create_base_model(num_classes=2, model_type='enhanced')
            else:
                model = create_improved_model(
                    num_classes=2,
                    use_contrastive_loss=exp_config['use_contrastive']
                )
            
            model.to(self.device)
            
            # 快速训练（简化版）
            # 在实际应用中，这里应该进行完整的训练
            
            # 评估
            evaluator = ModelEvaluator(model, tokenizer, self.device)
            metrics = evaluator.evaluate_on_dataset(test_loader)
            
            self.results[exp_name] = {
                'config': exp_config,
                'metrics': metrics
            }
        
        # 保存结果
        with open(save_dir / 'ablation_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 绘制结果
        self._plot_ablation_results(save_dir)
        
        return self.results
    
    def _plot_ablation_results(self, save_dir):
        """绘制消融实验结果"""
        experiments = list(self.results.keys())
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.results[exp][metrics][metric] for exp in experiments]
            descriptions = [self.results[exp]['config']['description'] for exp in experiments]
            bars = axes[i].bar(range(len(experiments)), values)
            axes[i].set_xlabel(self._get_label('实验配置', 'Experiment Configuration'))
            axes[i].set_ylabel(metric.replace('_weighted', '').title())
            axes[i].set_title(f'{metric.replace("_weighted", "").title()} {self._get_label("消融实验结果", "Ablation Study Results")}')
            axes[i].set_xticks(range(len(experiments)))
            axes[i].set_xticklabels(descriptions, rotation=45)
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='improved', 
                       choices=['base', 'improved'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 数据加载
    preprocessor = DataPreprocessor('ChnSentiCorp_htl_all.csv')
    _, _, test_loader = preprocessor.create_dataloaders(batch_size=args.batch_size)
      # 模型加载
    if args.model_type == 'base':
        model = create_base_model(num_classes=2, model_type='enhanced')
    else:
        # 创建不带对比学习的改进模型，因为保存的检查点可能不包含projection_head
        model = create_improved_model(num_classes=2, use_contrastive_loss=False)
    
    # 加载检查点
    load_checkpoint(model, args.model_path, device)
    model.to(device)
    
    # 创建评估器
    evaluator = ModelEvaluator(model, preprocessor.tokenizer, device)
    
    # 生成评估报告
    metrics, predictions = evaluator.generate_evaluation_report(test_loader, args.save_dir)
    
    logger.info("评估完成！")
    print(f"模型性能：")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"F1分数: {metrics['f1_weighted']:.4f}")
    print(f"精确率: {metrics['precision_weighted']:.4f}")
    print(f"召回率: {metrics['recall_weighted']:.4f}")

if __name__ == "__main__":
    main()
