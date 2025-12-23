"""
文本预处理模块
负责中文文本清洗、分词和停用词过滤
"""
import re
import jieba
from typing import List, Optional


# 中文停用词列表
STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '她', '他', '它', '们', '来', '去', '过', '吗', '呢', '吧',
    '啊', '呀', '哦', '哈', '嗯', '哎', '唉', '喂', '嘿', '诶', '啦', '呐', '哇',
    '而', '或', '但', '还', '因为', '所以', '如果', '虽然', '但是', '然后', '就是',
    '可以', '这样', '那样', '怎么', '什么', '为什么', '多少', '几', '哪', '哪里',
    '这里', '那里', '什么样', '怎么样', '如何', '能', '能够', '可能', '应该', '必须',
    '只', '只是', '只有', '才', '更', '最', '非常', '特别', '真', '太', '比较',
    '一些', '一点', '一下', '一直', '一起', '一样', '每', '各', '该', '其', '其他',
    '之', '与', '及', '等', '这个', '那个', '这些', '那些', '某', '某个', '某些',
    '把', '被', '给', '让', '向', '从', '到', '对', '对于', '关于', '根据', '按照',
    '通过', '以', '以及', '并', '并且', '而且', '或者', '不但', '不仅', '不过',
    '当', '当时', '时', '时候', '之后', '之前', '以后', '以前', '现在', '目前',
    '已经', '曾经', '将', '将要', '正在', '一定', '肯定', '大概', '也许', '可能',
])


def clean_text(text: str) -> str:
    """
    清洗文本
    
    Args:
        text: 原始文本
        
    Returns:
        清洗后的文本
    """
    if not isinstance(text, str):
        return ""
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 移除URL
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # 移除邮箱
    text = re.sub(r'\S+@\S+', '', text)
    
    # 移除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊符号（保留中文、英文、数字）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    return text.strip()


def tokenize(text: str, use_stopwords: bool = True) -> List[str]:
    """
    对文本进行分词
    
    Args:
        text: 输入文本
        use_stopwords: 是否过滤停用词
        
    Returns:
        分词后的词语列表
    """
    # 使用jieba分词
    words = jieba.cut(text)
    
    # 过滤空白词和停用词
    if use_stopwords:
        words = [w for w in words if w.strip() and w not in STOPWORDS]
    else:
        words = [w for w in words if w.strip()]
    
    return words


def preprocess_text(text: str, use_stopwords: bool = True) -> str:
    """
    完整的文本预处理流程
    
    Args:
        text: 原始文本
        use_stopwords: 是否过滤停用词
        
    Returns:
        预处理后的文本（空格分隔的词语）
    """
    # 清洗文本
    cleaned = clean_text(text)
    
    # 分词
    tokens = tokenize(cleaned, use_stopwords)
    
    # 用空格连接
    return ' '.join(tokens)


def preprocess_texts(texts: List[str], use_stopwords: bool = True) -> List[str]:
    """
    批量预处理文本
    
    Args:
        texts: 文本列表
        use_stopwords: 是否过滤停用词
        
    Returns:
        预处理后的文本列表
    """
    from tqdm import tqdm
    
    processed = []
    for text in tqdm(texts, desc="预处理文本"):
        processed.append(preprocess_text(text, use_stopwords))
    
    return processed


def get_word_frequencies(texts: List[str], top_n: int = 100) -> dict:
    """
    统计词频
    
    Args:
        texts: 预处理后的文本列表
        top_n: 返回前n个高频词
        
    Returns:
        词频字典
    """
    from collections import Counter
    
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    word_freq = Counter(all_words)
    return dict(word_freq.most_common(top_n))
