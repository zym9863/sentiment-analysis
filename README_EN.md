# Sentiment Analysis of Chinese Hotel Reviews Based on BERT and Attention Mechanism

[中文文档](README.md) | [English Version](README_EN.md)

## Project Introduction

This project implements a sentiment analysis system for Chinese hotel reviews based on BERT and attention mechanisms, trained and evaluated on the ChnSentiCorp dataset. The project features the following innovations:

1. **Improved Model Architecture**: BERT + BiLSTM + Multi-head Attention
2. **Multi-level Feature Fusion**: Combines word-level and sentence-level features
3. **Contrastive Learning**: Enhances the model's representation learning ability
4. **Multi-dimensional Evaluation**: Includes Precision, Recall, F1, ablation studies, case analysis, etc.

## Project Structure

```
├── models/                 # Model directory
│   ├── base_model.py      # Basic BERT model
│   └── improved_model.py  # Improved BERT+BiLSTM+Attention model
├── src/                   # Source code
│   ├── config.py          # Configuration file
│   ├── data_preprocessing.py  # Data preprocessing
│   ├── model_training.py     # Model training
│   ├── model_evaluation.py   # Model evaluation
│   └── utils.py             # Utility functions
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_analysis.ipynb
│   └── 04_ablation_study.ipynb
├── tests/                 # Test files
│   ├── __init__.py
│   ├── run_tests.py       # Test runner
│   ├── test_config.py     # Configuration tests
│   ├── test_data_preprocessing.py  # Data preprocessing tests
│   ├── test_model_evaluation.py    # Model evaluation tests
│   ├── test_model_training.py      # Model training tests
│   └── test_utils.py      # Utility function tests
├── results/               # Experimental results
│   ├── ablation_study_report.md    # Ablation study report
│   └── evaluation_summary.md       # Evaluation summary
├── ChnSentiCorp_htl_all.csv        # Dataset file
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
├── intro.ipynb           # Project introduction notebook
├── code.md               # Code documentation
├── product.md            # Product documentation
├── LICENSE               # License
├── README.md             # Chinese documentation
└── README_EN.md          # English documentation
```

## Environment Setup

### Create Conda Virtual Environment
```bash
conda create -n bert_sentiment_analysis python=3.9
conda activate bert_sentiment_analysis
pip install -r requirements.txt
```

## Model Architecture

### Base Model (bert-base-chinese)
- Uses pre-trained BERT for sentiment classification
- Simple classification head

### Improved Model (BERT + BiLSTM + Attention)
- BERT encoder: extracts contextual features
- BiLSTM layer: captures sequence dependencies
- Multi-head attention: focuses on important information
- Residual connections and layer normalization
- Contrastive loss

## Usage

1. **Data Preprocessing**:
   ```python
   python src/data_preprocessing.py
   ```

2. **Model Training**:
   ```python
   python src/model_training.py --model_type improved
   ```

3. **Model Evaluation**:
   ```python
   python src/model_evaluation.py --model_path models/saved_models/best_model.pth
   ```

4. **Run Tests**:
   ```python
   python tests/run_tests.py
   ```

## Innovations

1. **Multi-level Feature Fusion**: Combines deep semantic features from BERT and sequence modeling from BiLSTM
2. **Improved Attention Mechanism**: Uses multi-head attention to capture diverse aspects
3. **Contrastive Learning**: Enhances the model's representation learning ability
4. **Comprehensive Evaluation System**: Multi-dimensional metrics and detailed case analysis
5. **Complete Testing Framework**: Includes unit tests and integration tests