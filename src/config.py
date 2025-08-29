# -*- coding: utf-8 -*-
"""
统一配置管理系统
用于管理项目中所有模块的配置参数
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """数据相关配置"""
    data_path: str = "ChnSentiCorp_htl_all.csv"
    processed_data_dir: str = "data/processed"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    max_length: int = 512
    batch_size: int = 16
    num_workers: int = 2

@dataclass 
class ModelConfig:
    """模型相关配置"""
    model_name: str = "bert-base-chinese"
    model_type: str = "improved"  # 'base' or 'improved'
    num_classes: int = 2
    dropout_rate: float = 0.3
    hidden_size: int = 768
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.1

@dataclass
class TrainingConfig:
    """训练相关配置"""
    learning_rate: float = 2e-5
    bert_learning_rate: float = 1e-5  # BERT层使用更小的学习率
    weight_decay: float = 0.01
    max_epochs: int = 10
    patience: int = 3
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 0
    scheduler_type: str = "cosine"  # 'cosine', 'linear', 'plateau'
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50

@dataclass
class SystemConfig:
    """系统相关配置"""
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    mixed_precision: bool = False
    dataloader_num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42
    deterministic: bool = True
    
    # GPU内存管理
    gpu_memory_threshold: float = 0.9
    clear_cache_frequency: int = 50  # 每N个batch清理一次缓存
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class PathConfig:
    """路径相关配置"""
    project_root: str = "."
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "results"
    log_dir: str = "logs"
    checkpoint_dir: str = "models/saved_models"
    
    def __post_init__(self):
        """确保所有路径都是绝对路径"""
        project_root = Path(self.project_root).resolve()
        self.data_dir = str(project_root / self.data_dir)
        self.model_dir = str(project_root / self.model_dir)
        self.output_dir = str(project_root / self.output_dir)
        self.log_dir = str(project_root / self.log_dir)
        self.checkpoint_dir = str(project_root / self.checkpoint_dir)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.system_config = SystemConfig()
        self.path_config = PathConfig()
        
        # 如果提供了配置文件，加载配置
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        else:
            # 从环境变量加载配置
            self.load_from_env()
    
    def load_from_file(self, config_file: str):
        """从文件加载配置"""
        config_path = Path(config_file)
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 更新各个配置类
            if 'data' in config_data:
                self._update_config(self.data_config, config_data['data'])
            if 'model' in config_data:
                self._update_config(self.model_config, config_data['model'])
            if 'training' in config_data:
                self._update_config(self.training_config, config_data['training'])
            if 'system' in config_data:
                self._update_config(self.system_config, config_data['system'])
            if 'paths' in config_data:
                self._update_config(self.path_config, config_data['paths'])
            
            logger.info(f"成功从 {config_file} 加载配置")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.info("使用默认配置")
    
    def load_from_env(self):
        """从环境变量加载配置"""
        # 数据配置
        if os.getenv('DATA_PATH'):
            self.data_config.data_path = os.getenv('DATA_PATH')
        if os.getenv('BATCH_SIZE'):
            self.data_config.batch_size = int(os.getenv('BATCH_SIZE'))
        
        # 模型配置
        if os.getenv('MODEL_NAME'):
            self.model_config.model_name = os.getenv('MODEL_NAME')
        if os.getenv('MODEL_TYPE'):
            self.model_config.model_type = os.getenv('MODEL_TYPE')
        
        # 训练配置
        if os.getenv('LEARNING_RATE'):
            self.training_config.learning_rate = float(os.getenv('LEARNING_RATE'))
        if os.getenv('MAX_EPOCHS'):
            self.training_config.max_epochs = int(os.getenv('MAX_EPOCHS'))
        
        # 系统配置
        if os.getenv('DEVICE'):
            self.system_config.device = os.getenv('DEVICE')
        if os.getenv('LOG_LEVEL'):
            self.system_config.log_level = os.getenv('LOG_LEVEL')
    
    def _update_config(self, config_obj, config_dict):
        """更新配置对象"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建完整配置字典
        config_data = {
            'data': asdict(self.data_config),
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'system': asdict(self.system_config),
            'paths': asdict(self.path_config)
        }
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            logger.info(f"配置已保存到 {config_file}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置的字典表示"""
        return {
            'data': asdict(self.data_config),
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'system': asdict(self.system_config),
            'paths': asdict(self.path_config)
        }
    
    def create_directories(self):
        """创建必要的目录"""
        paths_to_create = [
            self.path_config.data_dir,
            self.path_config.model_dir,
            self.path_config.output_dir,
            self.path_config.log_dir,
            self.path_config.checkpoint_dir,
            self.data_config.processed_data_dir
        ]
        
        for path in paths_to_create:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"创建目录: {path}")
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 验证数据配置
            assert 0 < self.data_config.test_size < 1, "test_size 必须在 (0, 1) 范围内"
            assert 0 < self.data_config.val_size < 1, "val_size 必须在 (0, 1) 范围内"
            assert self.data_config.batch_size > 0, "batch_size 必须大于 0"
            assert self.data_config.max_length > 0, "max_length 必须大于 0"
            
            # 验证模型配置
            assert self.model_config.num_classes > 1, "num_classes 必须大于 1"
            assert 0 <= self.model_config.dropout_rate <= 1, "dropout_rate 必须在 [0, 1] 范围内"
            assert self.model_config.model_type in ['base', 'improved'], "model_type 必须是 'base' 或 'improved'"
            
            # 验证训练配置
            assert self.training_config.learning_rate > 0, "learning_rate 必须大于 0"
            assert self.training_config.max_epochs > 0, "max_epochs 必须大于 0"
            assert self.training_config.patience > 0, "patience 必须大于 0"
            
            # 验证系统配置
            assert self.system_config.device in ['auto', 'cuda', 'cpu'], "device 必须是 'auto', 'cuda' 或 'cpu'"
            assert 0 < self.system_config.gpu_memory_threshold <= 1, "gpu_memory_threshold 必须在 (0, 1] 范围内"
            
            logger.info("配置验证通过")
            return True
            
        except AssertionError as e:
            logger.error(f"配置验证失败: {e}")
            return False
        except Exception as e:
            logger.error(f"配置验证过程中出错: {e}")
            return False

# 全局配置实例
config = ConfigManager()

def get_config(config_file: Optional[str] = None) -> ConfigManager:
    """获取配置管理器实例"""
    global config
    if config_file:
        config = ConfigManager(config_file)
    return config

def load_config(config_file: str) -> ConfigManager:
    """加载指定的配置文件"""
    return ConfigManager(config_file)