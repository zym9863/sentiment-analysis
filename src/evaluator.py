"""
评估器模块
计算各种评估指标
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from typing import Dict, List, Optional, Tuple


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    评估模型性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（用于计算AUC）
        
    Returns:
        评估指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # 计算AUC
    if y_proba is not None:
        if len(y_proba.shape) == 2:
            y_scores = y_proba[:, 1]  # 取正类的概率
        else:
            y_scores = y_proba
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        metrics['auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    return metrics


def print_evaluation_report(metrics: Dict, model_name: str = "模型"):
    """
    打印评估报告
    
    Args:
        metrics: 评估指标字典
        model_name: 模型名称
    """
    print(f"\n{'='*50}")
    print(f"{model_name} 评估报告")
    print('='*50)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score):  {metrics['f1']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC值:              {metrics['auc']:.4f}")
    
    print(f"\n混淆矩阵:")
    cm = metrics['confusion_matrix']
    print(f"           预测负面  预测正面")
    print(f"实际负面    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"实际正面    {cm[1,0]:5d}    {cm[1,1]:5d}")
    print('='*50)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    获取详细分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        分类报告字符串
    """
    target_names = ['负面评论', '正面评论']
    return classification_report(y_true, y_pred, target_names=target_names)


def compare_models(results: Dict[str, Dict]) -> None:
    """
    比较多个模型的性能
    
    Args:
        results: 模型名称到评估指标的字典
    """
    print("\n" + "="*70)
    print("模型性能对比")
    print("="*70)
    print(f"{'模型名称':<20} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1分数':>10}")
    print("-"*70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:>10.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f}")
    
    print("="*70)
    
    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\n最佳模型（按F1分数）: {best_model[0]} (F1={best_model[1]['f1']:.4f})")
