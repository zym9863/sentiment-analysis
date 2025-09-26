**当前文档：中文** | [English Version](README_EN.md)

# 基于BERT与注意力机制的中文酒店评论情感分析

## 项目简介

本项目实现了一个针对中文酒店评论的情感分析系统，核心模型基于`bert-base-chinese`，并结合双向LSTM与多头注意力机制进行特征建模。我们在ChnSentiCorp数据集上完成了训练与评估，并补充了对比学习模块与多维度分析工具，旨在提升模型效果与可靠性。

主要特点如下：

- **改进的模型架构**：BERT + BiLSTM + 多头注意力 + 对比学习损失。
- **多层次特征融合**：结合词级语义表示与句级序列建模能力。
- **完善的评估体系**：覆盖Precision、Recall、F1、AUC、消融实验与案例分析。
- **工程化支撑**：提供完整的数据预处理、训练、评估、测试与可视化工具链。

## 最新进展

- **当前版本**：v1.0.1
- **最后更新**：2024年12月
- **状态摘要**：
  - ✅ 基础BERT模型与改进模型均已交付，并通过核心用例验证。
  - ✅ 建立6个模块的单元测试框架，支持全量与单模块执行。
  - 🚧 正在推进GPU内存管理优化与配置管理统一化工作。

## 项目结构

```
├── models/                 # 模型定义
│   ├── base_model.py       # 基础BERT模型
│   └── improved_model.py   # BERT + BiLSTM + Attention改进模型
├── src/                    # 源码实现
│   ├── config.py           # 通用配置
│   ├── data_preprocessing.py   # 数据预处理脚本
│   ├── model_training.py       # 训练脚本
│   ├── model_evaluation.py     # 评估脚本
│   └── utils.py                # 工具函数
├── notebooks/              # 实验记录
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_analysis.ipynb
│   └── 04_ablation_study.ipynb
├── tests/                  # 测试用例
│   ├── run_tests.py
│   ├── test_config.py
│   ├── test_data_preprocessing.py
│   ├── test_model_evaluation.py
│   ├── test_model_training.py
│   └── test_utils.py
├── results/                # 结果与报告
│   ├── ablation_study_report.md
│   └── evaluation_summary.md
├── ChnSentiCorp_htl_all.csv # 数据集
├── config.yaml             # 配置示例
├── requirements.txt        # 依赖列表
├── intro.ipynb             # 项目介绍
├── code.md                 # 代码说明
├── product.md              # 产品规划
├── LICENSE
├── README.md               # 中文文档
└── README_EN.md            # 英文文档

## 快速开始

1. **创建虚拟环境并安装依赖**

   ```bash
   conda create -n bert_sentiment_analysis python=3.9
   conda activate bert_sentiment_analysis
   pip install -r requirements.txt
   ```

2. **数据预处理**

   ```bash
   python src/data_preprocessing.py
   ```

3. **模型训练**

   ```bash
   python src/model_training.py --model_type improved
   ```

4. **模型评估**

   ```bash
   python src/model_evaluation.py --model_path models/saved_models/best_model.pth
   ```

5. **运行测试**

   ```bash
   python tests/run_tests.py
   ```

## 模型架构概览

- **基础模型（bert-base-chinese）**：使用预训练BERT提取上下文特征，配合轻量分类头完成情感预测。
- **改进模型（BERT + BiLSTM + 多头注意力）**：
  - BERT编码器：获取上下文语义表示。
  - BiLSTM层：捕获正反向序列依赖关系。
  - 多头注意力：聚焦关键语义片段。
  - 残差连接与层归一化：提升训练稳定性。
  - 对比学习损失：增强表示区分度。

## 性能指标

| 指标 | 基础BERT | 改进BERT | 提升 |
|------|----------|----------|------|
| 准确率 | 91.38% | 91.44% | +0.06% |
| 精确率 | 89.91% | 89.85% | -0.06% |
| 召回率 | 90.17% | 90.49% | +0.32% |
| F1分数 | 90.04% | 90.16% | +0.12% |
| AUC | 96.26% | 96.42% | +0.17% |

## 创新要点

1. **多层次特征融合**：结合深层语义与序列信息，提升表达能力。
2. **增强型注意力机制**：多头注意力捕获多维度情感线索。
3. **对比学习策略**：提升类间区分度与泛化能力。
4. **全面评估体系**：提供多指标评估、错误分析与置信度分析。
5. **完善的测试体系**：覆盖数据处理、训练、评估等关键模块。

## 路线图概览

- **P0 核心问题**
  - 修复 `src/model_evaluation.py` 中的缩进问题。
  - 为数据预处理、网络请求与文件操作等模块补充异常处理。
- **P1 稳定性提升**
  - 构建GPU显存监控与自动清理机制。
  - 推进配置管理统一化并支持环境变量。
- **中期优化目标**
  - 引入混合精度训练、梯度累积与数据加载优化。
  - 实现推理量化、批量推理与服务化API。
  - 引入模型集成与解释性分析工具（LIME、SHAP、注意力热力图）。