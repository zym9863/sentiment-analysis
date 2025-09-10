[中文文档](README.md) | [English Version](README_EN.md)

# 基于BERT与注意力机制的中文酒店评论情感分析

## 项目简介

本项目实现了一个基于BERT与注意力机制的中文酒店评论情感分析系统，使用ChnSentiCorp数据集进行训练和评估。项目包含以下创新点：

1. **改进的模型架构**: BERT + BiLSTM + 多头注意力机制
2. **多层次特征融合**: 结合词级和句级特征
3. **对比学习**: 增强模型的表示学习能力
4. **多维度评估**: 包含Precision、Recall、F1、消融实验、案例分析等

## 项目结构

```
├── models/                 # 模型目录
│   ├── base_model.py      # 基础BERT模型
│   └── improved_model.py  # 改进的BERT+BiLSTM+Attention模型
├── src/                   # 源代码
│   ├── config.py          # 配置文件
│   ├── data_preprocessing.py  # 数据预处理
│   ├── model_training.py     # 模型训练
│   ├── model_evaluation.py   # 模型评估
│   └── utils.py             # 工具函数
├── notebooks/             # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_analysis.ipynb
│   └── 04_ablation_study.ipynb
├── tests/                 # 测试文件
│   ├── __init__.py
│   ├── run_tests.py       # 测试运行器
│   ├── test_config.py     # 配置测试
│   ├── test_data_preprocessing.py  # 数据预处理测试
│   ├── test_model_evaluation.py    # 模型评估测试
│   ├── test_model_training.py      # 模型训练测试
│   └── test_utils.py      # 工具函数测试
├── results/               # 实验结果
│   ├── ablation_study_report.md    # 消融实验报告
│   └── evaluation_summary.md       # 评估总结
├── ChnSentiCorp_htl_all.csv        # 数据集文件
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖包
├── intro.ipynb           # 项目介绍笔记本
├── code.md               # 代码文档
├── product.md            # 产品文档
├── LICENSE               # 许可证
├── README.md             # 中文文档
└── README_EN.md          # 英文文档
```

## 环境设置

### 创建Conda虚拟环境
```bash
conda create -n bert_sentiment_analysis python=3.9
conda activate bert_sentiment_analysis
pip install -r requirements.txt
```

## 模型架构

### 基础模型 (bert-base-chinese)
- 使用预训练的BERT模型进行情感分类
- 简单的分类头

### 改进模型 (BERT + BiLSTM + Attention)
- BERT编码器：提取上下文特征
- BiLSTM层：捕获序列依赖关系
- 多头注意力机制：关注重要信息
- 残差连接和层归一化
- 对比学习损失

## 使用方法

1. **数据预处理**:
   ```python
   python src/data_preprocessing.py
   ```

2. **模型训练**:
   ```python
   python src/model_training.py --model_type improved
   ```

3. **模型评估**:
   ```python
   python src/model_evaluation.py --model_path models/saved_models/best_model.pth
   ```

4. **运行测试**:
   ```python
   python tests/run_tests.py
   ```

## 创新点

1. **多层次特征融合**: 结合BERT的深层语义和BiLSTM的序列建模
2. **改进的注意力机制**: 使用多头注意力捕获不同方面的信息
3. **对比学习**: 增强模型的表示学习能力
4. **全面的评估体系**: 多维度指标和详细的案例分析
5. **完整的测试框架**: 包含单元测试和集成测试