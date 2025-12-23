[English](README-EN.md) | [中文](README.md)

# Chinese Sentiment Analysis - NLP Course Project

A Chinese sentiment analysis project based on the ChnSentiCorp hotel review dataset, using traditional machine learning and deep learning methods.

## Project Overview

This project implements a complete Chinese sentiment analysis pipeline, including:
- 📊 Data preprocessing and analysis visualization
- 🤖 Traditional machine learning models (Naive Bayes, SVM, Logistic Regression, Random Forest)
- 🧠 Deep learning models (TextCNN, LSTM)
- 📈 Multi-dimensional model evaluation and comparison

## Project Structure

```
sentiment-analysis/
├── src/                       # Source code
│   ├── data_loader.py         # Data loading module
│   ├── preprocessor.py        # Text preprocessing (tokenization, cleaning)
│   ├── feature_extractor.py   # TF-IDF feature extraction
│   ├── models.py              # Model definitions
│   ├── trainer.py             # Trainer
│   ├── evaluator.py           # Evaluator
│   └── visualizer.py          # Visualization module
├── outputs/
│   ├── models/                # Saved models
│   └── figures/               # Visualization charts
├── ChnSentiCorp_htl_all.csv   # Dataset
├── main.py                    # Main program
├── pyproject.toml             # Project configuration
└── README.md
```

## Environment Setup

### Install Dependencies

This project uses the `uv` package manager:

```bash
# Install uv (if not installed)
pip install uv

# Install project dependencies
uv sync
```

## Quick Start

### Run Complete Training Pipeline

```bash
uv run python main.py
```

The program will automatically:
1. Load and analyze the dataset
2. Text preprocessing (tokenization, stopword removal)
3. Train traditional machine learning models
4. Train deep learning models
5. Generate evaluation reports and visualization charts

## Dataset

**ChnSentiCorp Hotel Review Dataset**
- Sample size: ~7,700 reviews
- Label distribution: Positive reviews / Negative reviews
- Data format: CSV (label, review)

## Model Introduction

### Traditional Machine Learning

| Model | Characteristics |
|-------|-----------------|
| Naive Bayes | Probability-based classifier, fast training |
| SVM | Support Vector Machine, suitable for high-dimensional features |
| Logistic Regression | High interpretability, suitable for binary classification |
| Random Forest | Ensemble learning method, resistant to overfitting |

### Deep Learning

| Model | Characteristics |
|-------|-----------------|
| TextCNN | Multi-scale convolution for capturing local features |
| LSTM | Bidirectional LSTM for capturing sequential dependencies |

## Output Results

### Generated Visualization Charts

- `data_distribution.png` - Data distribution chart
- `text_length_distribution.png` - Text length distribution
- `top_words.png` - Top frequent words statistics
- `wordcloud.png` - Word cloud
- `model_comparison.png` - Model performance comparison
- `roc_curves.png` - ROC curves comparison
- `confusion_matrix_best.png` - Best model confusion matrix
- `textcnn_training.png` - TextCNN training curve
- `lstm_training.png` - LSTM training curve

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- AUC

## Tech Stack

- **Data Processing**: pandas, numpy
- **Chinese Tokenization**: jieba
- **Machine Learning**: scikit-learn
- **Deep Learning**: PyTorch
- **Visualization**: matplotlib, seaborn, wordcloud

## References

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory

## License

MIT License
