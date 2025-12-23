"""
可视化模块
数据分析和模型结果可视化
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_data_distribution(labels: List[int], save_path: Optional[str] = None):
    """
    绘制数据分布图
    
    Args:
        labels: 标签列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 饼图
    label_counts = Counter(labels)
    labels_name = ['负面评论', '正面评论']
    sizes = [label_counts[0], label_counts[1]]
    colors = ['#ff6b6b', '#4ecdc4']
    explode = (0.02, 0.02)
    
    axes[0].pie(sizes, explode=explode, labels=labels_name, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
    axes[0].set_title('情感标签分布（饼图）', fontsize=14)
    
    # 柱状图
    x = np.arange(len(labels_name))
    bars = axes[1].bar(x, sizes, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_xlabel('情感类别', fontsize=12)
    axes[1].set_ylabel('样本数量', fontsize=12)
    axes[1].set_title('情感标签分布（柱状图）', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_name)
    
    # 在柱子上显示数值
    for bar, count in zip(bars, sizes):
        height = bar.get_height()
        axes[1].annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_text_length_distribution(lengths: List[int], save_path: Optional[str] = None):
    """
    绘制文本长度分布图
    
    Args:
        lengths: 文本长度列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    axes[0].hist(lengths, bins=50, color='#5dade2', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(lengths), color='red', linestyle='--', label=f'平均值: {np.mean(lengths):.1f}')
    axes[0].axvline(np.median(lengths), color='green', linestyle='--', label=f'中位数: {np.median(lengths):.1f}')
    axes[0].set_xlabel('文本长度（字符数）', fontsize=12)
    axes[0].set_ylabel('频次', fontsize=12)
    axes[0].set_title('文本长度分布（直方图）', fontsize=14)
    axes[0].legend()
    
    # 箱线图
    bp = axes[1].boxplot(lengths, patch_artist=True)
    bp['boxes'][0].set_facecolor('#9b59b6')
    axes[1].set_ylabel('文本长度（字符数）', fontsize=12)
    axes[1].set_title('文本长度分布（箱线图）', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_wordcloud(word_freq: Dict[str, int], title: str = "词云图", 
                   save_path: Optional[str] = None):
    """
    绘制词云图
    
    Args:
        word_freq: 词频字典
        title: 图表标题
        save_path: 保存路径
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("请安装wordcloud: pip install wordcloud")
        return
    
    # 创建词云
    wc = WordCloud(
        font_path='simhei.ttf',  # Windows中文字体
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        colormap='viridis'
    )
    
    # 如果找不到字体文件，尝试其他方法
    try:
        wc.generate_from_frequencies(word_freq)
    except:
        # 如果字体不可用，尝试不指定字体（可能无法显示中文）
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=200,
            colormap='viridis'
        )
        wc.generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    """
    绘制混淆矩阵热力图
    
    Args:
        cm: 混淆矩阵
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'],
                yticklabels=['负面', '正面'],
                annot_kws={'size': 14})
    
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_roc_curves(results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    绘制多个模型的ROC曲线
    
    Args:
        results: 模型名称到评估指标的字典
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (model_name, metrics), color in zip(results.items(), colors):
        if 'fpr' in metrics and 'tpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'], 
                    color=color, lw=2,
                    label=f"{model_name} (AUC = {metrics['auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('ROC曲线对比', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_model_comparison(results: Dict[str, Dict], save_path: Optional[str] = None):
    """
    绘制模型性能对比图
    
    Args:
        results: 模型名称到评估指标的字典
        save_path: 保存路径
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
    
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_names)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics[m] for m in metrics_names]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('评估指标', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('模型性能对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    绘制深度学习训练历史曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, history['train_losses'], 'b-', label='训练损失', marker='o')
    if history['val_losses']:
        axes[0].plot(epochs, history['val_losses'], 'r-', label='验证损失', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('损失', fontsize=12)
    axes[0].set_title('训练/验证损失曲线', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, history['train_accs'], 'b-', label='训练准确率', marker='o')
    if history['val_accs']:
        axes[1].plot(epochs, history['val_accs'], 'r-', label='验证准确率', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('准确率', fontsize=12)
    axes[1].set_title('训练/验证准确率曲线', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def plot_top_words(word_freq: Dict[str, int], top_n: int = 20, 
                   title: str = "高频词统计", save_path: Optional[str] = None):
    """
    绘制高频词柱状图
    
    Args:
        word_freq: 词频字典
        top_n: 显示前n个词
        title: 图表标题
        save_path: 保存路径
    """
    # 获取前top_n个词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [w[0] for w in sorted_words]
    freqs = [w[1] for w in sorted_words]
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(words)))
    bars = plt.barh(range(len(words)), freqs, color=colors)
    
    plt.yticks(range(len(words)), words)
    plt.xlabel('词频', fontsize=12)
    plt.ylabel('词语', fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()  # 频率高的在上面
    
    # 在柱子末端显示数值
    for bar, freq in zip(bars, freqs):
        plt.text(bar.get_width() + max(freqs)*0.01, bar.get_y() + bar.get_height()/2,
                f'{freq}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
