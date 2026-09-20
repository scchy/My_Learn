# python3 
# Create Date: 2026-09-16
# Author: Scc_hy X Claude
# Func: 智能分块策略优化
# ======================================================================================================


import re
import math
from cv2 import merge
import numpy as np 
from typing import Callable, List 


# 1- 句子切分 - 标题缺失时最小结构单元
_SENT_SPLIT = re.compile(r'(?<=[。！？；!?.])\s*')


def split_sentences(text: str):
    """按中英文标点切句；空行视为段落强边界，先分段再切句。"""
    text = re.sub(r'\n+', '\n', text.strip())
    sents = []
    for para in text.split('\n'):
        if para.strip():
            sents.extend(s for s in _SENT_SPLIT.split(para.strip()) if s.strip())
    return sents


def semantic_chunk(
    text: str, 
    embed_fn: Callable[[List[str]]],
    percentile: float = 25.0, 
    abs_floor: float = 0.5,
    min_size: int =3,
    max_size: int = 12,
    smooth_window: int = 2
):
    """基于相邻句嵌入相似度周边检查语义边界，将无法结构化文档切为主题内聚的块
    Args:
        text (str): 文本
        embed_fn (Callable[[List[str]], np.ndarray]): 嵌入函数
        percentile (float, optional): 相似度低于全篇 百分位数 的视为候选边界. Defaults to 25.0.
        abs_floor (float, optional): 绝对下限：全篇同主题时防止相对阈值强切. Defaults to 0.5.
        min_size (int, optional): 最小块大小. Defaults to 3.
        max_size (int, optional): 最大块大小. Defaults to 12.
        smooth_window (int, optional): 平滑窗口大小. Defaults to 2.
    """
    sents = split_sentences(text)
    N = len(sents)
    if N < min_size:
        return [text]
    # a- 批量编码： 只做一次模型推理
    embs = embed_fn(sents)
    embs = embs /  (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    
    # b- 相邻句余弦相似度
    sims = np.sum(embs[:-1] * embs[1:], axis=1)
    
    # c- 平滑窗口
    offset = 0
    if smooth_window > 0 and N - 1 > 2 * smooth_window:
        k = np.ones(2 * smooth_window + 1, dtype=np.float32) / (2 * smooth_window + 1)
        sims = np.convolve(sims, k, mode='valid')
        offset = smooth_window

    # d- 自适应阈值：相对分位数 + 绝对下限取更严格，骤降处即主题转换点
    effective = min(np.percentile(sims, percentile), abs_floor)
    cuts = [i + 1 + offset for i in range(len(sims)) if sims[i] < effective]
    
    # e- 尺寸约束：超长块均分、过短块回填；检查不到有边界时退化为固定大小块
    bounds = [0] + cuts + [N]
    chunks = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        k = math.ceil((e - s) / max_size)
        step = math.ceil((e - s) / k)
        for i in range(s, e, step):
            chunks.append((i, min(i + step, e)))
        
    merged = []
    for s, e in chunks:
        if (e - s < min_size and 
            merged and
            merged[-1][1] - merged[-1][0] + (e - s) <= max_size):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [''.join(sents[a:b]) for a, b in merged]



if __name__ == '__main__':
    novel_text = 'asdasda asxa'
    from sentence_transformers import SentenceTransformer 
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embed_fn = lambda sents: model.encode(sents, batch_size=64)
    chunks = semantic_chunk(novel_text, embed_fn)  # 小说/法条/聊天记录等无标题文档
    
