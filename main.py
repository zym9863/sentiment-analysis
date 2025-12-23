"""
中文情感分析 - 主程序
基于ChnSentiCorp酒店评论数据集
"""
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_dataset, split_dataset, get_data_statistics
from src.preprocessor import preprocess_texts, get_word_frequencies
from src.feature_extractor import FeatureExtractor
from src.models import TraditionalModels, TextCNN, LSTMClassifier, Vocabulary
from src.trainer import TraditionalTrainer, DeepLearningTrainer
from src.evaluator import evaluate_model, print_evaluation_report, compare_models
from src.visualizer import (
    plot_data_distribution,
    plot_text_length_distribution,
    plot_wordcloud,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    plot_training_history,
    plot_top_words
)


def ensure_dirs():
    """确保输出目录存在"""
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)


def run_data_analysis(df, processed_texts):
    """运行数据分析和可视化"""
    print("\n" + "="*60)
    print("数据分析与可视化")
    print("="*60)
    
    # 数据统计
    stats = get_data_statistics(df)
    print(f"\n数据集统计信息:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  正面样本: {stats['positive_samples']} ({stats['positive_ratio']*100:.1f}%)")
    print(f"  负面样本: {stats['negative_samples']} ({(1-stats['positive_ratio'])*100:.1f}%)")
    print(f"  平均文本长度: {stats['avg_text_length']:.1f}")
    print(f"  最长文本: {stats['max_text_length']}")
    print(f"  最短文本: {stats['min_text_length']}")
    
    # 可视化
    labels = df['label'].tolist()
    plot_data_distribution(labels, save_path='outputs/figures/data_distribution.png')
    
    lengths = [len(text) for text in df['review']]
    plot_text_length_distribution(lengths, save_path='outputs/figures/text_length_distribution.png')
    
    # 词频统计与可视化
    word_freq = get_word_frequencies(processed_texts, top_n=100)
    plot_top_words(word_freq, top_n=20, save_path='outputs/figures/top_words.png')
    
    try:
        plot_wordcloud(word_freq, title="酒店评论词云", save_path='outputs/figures/wordcloud.png')
    except Exception as e:
        print(f"词云生成失败（可能缺少中文字体）: {e}")


def train_traditional_models(X_train, y_train, X_test, y_test):
    """训练传统机器学习模型"""
    print("\n" + "="*60)
    print("传统机器学习模型训练")
    print("="*60)
    
    results = {}
    
    # 模型配置
    models = {
        '朴素贝叶斯': TraditionalModels.get_naive_bayes(),
        'SVM': TraditionalModels.get_svm(),
        '逻辑回归': TraditionalModels.get_logistic_regression(),
        '随机森林': TraditionalModels.get_random_forest(n_estimators=100)
    }
    
    for name, model in models.items():
        print(f"\n--- 训练 {name} ---")
        trainer = TraditionalTrainer(model)
        trainer.train(X_train, y_train)
        
        # 预测
        y_pred = trainer.predict(X_test)
        y_proba = trainer.predict_proba(X_test)
        
        # 评估
        metrics = evaluate_model(y_test, y_pred, y_proba)
        print_evaluation_report(metrics, name)
        results[name] = metrics
        
        # 保存模型
        trainer.save(f'outputs/models/{name.replace(" ", "_")}.pkl')
    
    return results


def train_deep_learning_models(train_texts, train_labels, val_texts, val_labels, 
                                test_texts, test_labels, vocab):
    """训练深度学习模型"""
    print("\n" + "="*60)
    print("深度学习模型训练")
    print("="*60)
    
    results = {}
    max_len = 128
    
    # 将文本转换为索引序列
    X_train = np.array([vocab.text_to_indices(t, max_len) for t in train_texts])
    X_val = np.array([vocab.text_to_indices(t, max_len) for t in val_texts])
    X_test = np.array([vocab.text_to_indices(t, max_len) for t in test_texts])
    
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)
    
    # TextCNN
    print("\n--- 训练 TextCNN ---")
    textcnn = TextCNN(vocab_size=len(vocab), embedding_dim=128, num_filters=64)
    textcnn_trainer = DeepLearningTrainer(textcnn, learning_rate=1e-3)
    textcnn_trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=10, batch_size=64, early_stopping=3
    )
    
    y_pred = textcnn_trainer.predict(X_test)
    y_proba = textcnn_trainer.predict_proba(X_test)
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print_evaluation_report(metrics, 'TextCNN')
    results['TextCNN'] = metrics
    
    # 绘制训练历史
    history = textcnn_trainer.get_training_history()
    plot_training_history(history, save_path='outputs/figures/textcnn_training.png')
    
    textcnn_trainer.save('outputs/models/textcnn.pth')
    
    # LSTM
    print("\n--- 训练 LSTM ---")
    lstm = LSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64)
    lstm_trainer = DeepLearningTrainer(lstm, learning_rate=1e-3)
    lstm_trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=10, batch_size=64, early_stopping=3
    )
    
    y_pred = lstm_trainer.predict(X_test)
    y_proba = lstm_trainer.predict_proba(X_test)
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print_evaluation_report(metrics, 'LSTM')
    results['LSTM'] = metrics
    
    # 绘制训练历史
    history = lstm_trainer.get_training_history()
    plot_training_history(history, save_path='outputs/figures/lstm_training.png')
    
    lstm_trainer.save('outputs/models/lstm.pth')
    
    return results


def main():
    """主函数"""
    print("="*60)
    print("中文情感分析 - 基于ChnSentiCorp酒店评论数据集")
    print("="*60)
    
    # 确保目录存在
    ensure_dirs()
    
    # 1. 加载数据
    print("\n[1/6] 加载数据集...")
    df = load_dataset('ChnSentiCorp_htl_all.csv')
    
    # 2. 数据划分
    print("\n[2/6] 划分数据集...")
    train_df, val_df, test_df = split_dataset(df)
    
    # 3. 文本预处理
    print("\n[3/6] 文本预处理...")
    train_texts = preprocess_texts(train_df['review'].tolist())
    val_texts = preprocess_texts(val_df['review'].tolist())
    test_texts = preprocess_texts(test_df['review'].tolist())
    
    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()
    test_labels = test_df['label'].tolist()
    
    # 4. 数据分析可视化
    print("\n[4/6] 数据分析与可视化...")
    all_processed_texts = train_texts + val_texts + test_texts
    run_data_analysis(df, all_processed_texts)
    
    # 5. 特征提取（用于传统机器学习）
    print("\n[5/6] 特征提取...")
    feature_extractor = FeatureExtractor(method='tfidf', max_features=5000)
    X_train = feature_extractor.fit_transform(train_texts)
    X_val = feature_extractor.transform(val_texts)
    X_test = feature_extractor.transform(test_texts)
    
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)
    
    feature_extractor.save('outputs/models/tfidf_vectorizer.pkl')
    
    # 6. 模型训练与评估
    print("\n[6/6] 模型训练与评估...")
    
    # 传统机器学习模型
    traditional_results = train_traditional_models(X_train, y_train, X_test, y_test)
    
    # 构建词汇表（用于深度学习）
    vocab = Vocabulary(max_size=10000)
    vocab.build(train_texts)
    vocab.save('outputs/models/vocabulary.pkl')
    
    # 深度学习模型
    dl_results = train_deep_learning_models(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        vocab
    )
    
    # 合并结果
    all_results = {**traditional_results, **dl_results}
    
    # 模型对比
    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    compare_models(all_results)
    
    # 可视化对比
    plot_model_comparison(all_results, save_path='outputs/figures/model_comparison.png')
    plot_roc_curves(all_results, save_path='outputs/figures/roc_curves.png')
    
    # 绘制最佳模型的混淆矩阵
    best_model_name = max(all_results.items(), key=lambda x: x[1]['f1'])[0]
    plot_confusion_matrix(
        all_results[best_model_name]['confusion_matrix'],
        save_path='outputs/figures/confusion_matrix_best.png'
    )
    
    print("\n" + "="*60)
    print("所有实验完成！")
    print(f"模型保存在: outputs/models/")
    print(f"可视化结果保存在: outputs/figures/")
    print("="*60)


if __name__ == "__main__":
    main()
