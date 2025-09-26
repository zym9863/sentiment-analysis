# Sentiment Analysis of Chinese Hotel Reviews Based on BERT and Attention

[中文文档](README.md) | **Current document: English**

## Project Overview

This repository delivers a sentiment analysis system tailored for Chinese hotel reviews. The solution is built on `bert-base-chinese`, enhanced with bidirectional LSTMs and multi-head attention to better capture nuanced contextual features. Leveraging the ChnSentiCorp dataset, we augment the training pipeline with contrastive learning and extensive evaluation utilities to improve accuracy and reliability.

Key highlights include:

- **Enhanced architecture**: BERT + BiLSTM + multi-head attention + contrastive learning objective.
- **Hierarchical feature fusion**: Jointly integrates token-level semantics and sentence-level dependencies.
- **Comprehensive evaluation**: Covers Precision, Recall, F1, AUC, ablation studies, and qualitative analyses.
- **Production-ready tooling**: Provides preprocessing, training, evaluation, testing, and visualization scripts end-to-end.

## Latest Updates

- **Current version**: v1.0.1
- **Last updated**: December 2024
- **Status summary**:
  - ✅ Baseline and improved models have been delivered and validated on core use cases.
  - ✅ A six-module unit-test suite now supports both full and single-module executions.
  - 🚧 GPU memory management and unified configuration improvements are in progress.

## Repository Structure

```
├── models/                 # Model definitions
│   ├── base_model.py       # Baseline BERT classifier
│   └── improved_model.py   # BERT + BiLSTM + Attention architecture
├── src/                    # Source code
│   ├── config.py           # Shared configuration
│   ├── data_preprocessing.py   # Data preprocessing pipeline
│   ├── model_training.py       # Training entry point
│   ├── model_evaluation.py     # Evaluation entry point
│   └── utils.py                # Utility helpers
├── notebooks/              # Experiment notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_analysis.ipynb
│   └── 04_ablation_study.ipynb
├── tests/                  # Automated tests
│   ├── run_tests.py
│   ├── test_config.py
│   ├── test_data_preprocessing.py
│   ├── test_model_evaluation.py
│   ├── test_model_training.py
│   └── test_utils.py
├── results/                # Reports and summaries
│   ├── ablation_study_report.md
│   └── evaluation_summary.md
├── ChnSentiCorp_htl_all.csv # Dataset file
├── config.yaml             # Configuration template
├── requirements.txt        # Dependency list
├── intro.ipynb             # Project introduction
├── code.md                 # Code documentation
├── product.md              # Product roadmap
├── LICENSE
├── README.md               # Chinese README
└── README_EN.md            # English README (this file)
```

## Getting Started

1. **Create the environment and install dependencies**

   ```bash
   conda create -n bert_sentiment_analysis python=3.9
   conda activate bert_sentiment_analysis
   pip install -r requirements.txt
   ```

2. **Run data preprocessing**

   ```bash
   python src/data_preprocessing.py
   ```

3. **Train the model**

   ```bash
   python src/model_training.py --model_type improved
   ```

4. **Evaluate the model**

   ```bash
   python src/model_evaluation.py --model_path models/saved_models/best_model.pth
   ```

5. **Execute automated tests**

   ```bash
   python tests/run_tests.py
   ```

## Model Architecture Summary

- **Baseline (`bert-base-chinese`)**: Employs pre-trained BERT representations with a lightweight classification head for sentiment prediction.
- **Enhanced model (BERT + BiLSTM + multi-head attention)**:
  - BERT encoder for contextual embeddings.
  - BiLSTM layer to capture bidirectional sequence dependencies.
  - Multi-head attention to focus on salient semantic components.
  - Residual connections with layer normalization for stable optimization.
  - Contrastive learning auxiliary loss to improve discriminative power.

## Performance Metrics

| Metric | Baseline BERT | Improved BERT | Delta |
|--------|---------------|---------------|-------|
| Accuracy | 91.38% | 91.44% | +0.06% |
| Precision | 89.91% | 89.85% | -0.06% |
| Recall | 90.17% | 90.49% | +0.32% |
| F1 Score | 90.04% | 90.16% | +0.12% |
| AUC | 96.26% | 96.42% | +0.17% |

## Innovation Highlights

1. **Multi-level feature fusion**: Blends deep semantic and sequential cues to strengthen representations.
2. **Advanced attention mechanism**: Multi-head attention captures diverse sentiment indicators.
3. **Contrastive learning strategy**: Improves class separability and generalization.
4. **Comprehensive evaluation**: Includes quantitative metrics, error breakdowns, and confidence analysis.
5. **Robust testing coverage**: Automated tests span preprocessing, training, evaluation, and utilities.

## Roadmap Overview

- **P0 Critical Tasks**
  - Fix indentation issues in `src/model_evaluation.py`.
  - Add defensive exception handling around preprocessing, network, and file I/O operations.
- **P1 Stability Enhancements**
  - Implement GPU memory monitoring and automated cleanup routines.
  - Unify configuration management and introduce environment variable support.
- **Mid-term Goals**
  - Adopt mixed-precision training, gradient accumulation, and data-loading optimizations.
  - Enable inference quantization, batch serving, and API deployment.
  - Deliver model ensemble strategies and interpretability tooling (LIME, SHAP, attention heatmaps).

## Support

- For coordination and feedback channels, refer to the contact details listed in `product.md`.
- For Chinese-language resources, please consult `README.md`.