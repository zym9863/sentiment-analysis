# 中文情感分析 - NLP结课作业

基于ChnSentiCorp酒店评论数据集的中文情感分析项目，使用传统机器学习和深度学习方法。

## 项目概述

本项目实现了完整的中文情感分析Pipeline，包括：
- 📊 数据预处理与分析可视化
- 🤖 传统机器学习模型（朴素贝叶斯、SVM、逻辑回归、随机森林）
- 🧠 深度学习模型（TextCNN、LSTM）
- 📈 多维度模型评估与对比

## 项目结构

```
sentiment-analysis/
├── src/                       # 源代码
│   ├── data_loader.py         # 数据加载模块
│   ├── preprocessor.py        # 文本预处理（分词、清洗）
│   ├── feature_extractor.py   # TF-IDF特征提取
│   ├── models.py              # 模型定义
│   ├── trainer.py             # 训练器
│   ├── evaluator.py           # 评估器
│   └── visualizer.py          # 可视化模块
├── outputs/
│   ├── models/                # 保存的模型
│   └── figures/               # 可视化图表
├── ChnSentiCorp_htl_all.csv   # 数据集
├── main.py                    # 主程序
├── pyproject.toml             # 项目配置
└── README.md
```

## 环境配置

### 安装依赖

本项目使用 `uv` 包管理器：

```bash
# 安装uv（如未安装）
pip install uv

# 安装项目依赖
uv sync
```

## 快速开始

### 运行完整训练流程

```bash
uv run python main.py
```

程序将自动完成：
1. 加载并分析数据集
2. 文本预处理（分词、去停用词）
3. 训练传统机器学习模型
4. 训练深度学习模型
5. 生成评估报告和可视化图表

## 数据集

**ChnSentiCorp酒店评论数据集**
- 样本数量：约7,700条评论
- 标签分布：正面评论 / 负面评论
- 数据格式：CSV（label, review）

## 模型介绍

### 传统机器学习

| 模型 | 特点 |
|------|------|
| 朴素贝叶斯 | 基于概率的分类器，训练速度快 |
| SVM | 支持向量机，适合高维特征 |
| 逻辑回归 | 可解释性强，适合二分类 |
| 随机森林 | 集成学习方法，抗过拟合 |

### 深度学习

| 模型 | 特点 |
|------|------|
| TextCNN | 多尺度卷积捕获局部特征 |
| LSTM | 双向LSTM捕获序列依赖 |

## 输出结果

### 生成的可视化图表

- `data_distribution.png` - 数据分布图
- `text_length_distribution.png` - 文本长度分布
- `top_words.png` - 高频词统计
- `wordcloud.png` - 词云图
- `model_comparison.png` - 模型性能对比
- `roc_curves.png` - ROC曲线对比
- `confusion_matrix_best.png` - 最佳模型混淆矩阵
- `textcnn_training.png` - TextCNN训练曲线
- `lstm_training.png` - LSTM训练曲线

### 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- AUC值

## 技术栈

- **数据处理**: pandas, numpy
- **中文分词**: jieba
- **机器学习**: scikit-learn
- **深度学习**: PyTorch
- **可视化**: matplotlib, seaborn, wordcloud

## 参考文献

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory

## License

MIT License
