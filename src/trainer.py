"""
训练器模块
负责模型训练流程
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from typing import Optional, List, Tuple
from tqdm import tqdm
import joblib


class TraditionalTrainer:
    """传统机器学习模型训练器"""
    
    def __init__(self, model: BaseEstimator):
        self.model = model
        
    def train(self, X_train, y_train):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        print(f"开始训练 {type(self.model).__name__} 模型...")
        self.model.fit(X_train, y_train)
        print("训练完成!")
        
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率（如果模型支持）"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # 对于SVM等，使用decision_function
            scores = self.model.decision_function(X)
            # 简单归一化到0-1
            return np.column_stack([-scores, scores])
        return None
    
    def save(self, filepath: str):
        """保存模型"""
        joblib.dump(self.model, filepath)
        print(f"模型已保存到: {filepath}")
        
    def load(self, filepath: str):
        """加载模型"""
        self.model = joblib.load(filepath)
        print(f"模型已从 {filepath} 加载")


class DeepLearningTrainer:
    """深度学习模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: Optional[str] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def _create_dataloader(self, X, y, batch_size: int, shuffle: bool = True):
        """创建数据加载器"""
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
        epochs: int = 10,
        batch_size: int = 64,
        early_stopping: int = 3
    ):
        """
        训练模型
        
        Args:
            X_train: 训练数据（索引序列）
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            early_stopping: 早停耐心值
        """
        train_loader = self._create_dataloader(X_train, y_train, batch_size)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练 {type(self.model).__name__}，设备: {self.device}")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            self.train_losses.append(avg_train_loss)
            self.train_accs.append(train_acc)
            
            # 验证阶段
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        print(f"早停触发，在epoch {epoch+1}停止训练")
                        break
            else:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}")
        
        print("训练完成!")
        
    def _evaluate(self, dataloader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(dataloader), correct / total
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """预测概率"""
        self.model.eval()
        X_tensor = torch.LongTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def get_training_history(self):
        """获取训练历史"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save(self.model.state_dict(), filepath)
        print(f"模型已保存到: {filepath}")
        
    def load(self, filepath: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"模型已从 {filepath} 加载")
