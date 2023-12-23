# python3
# Create date: 2023-12-20
# Author: Scc_hy
# Chapter: 11- Future Directions
# ===================================================================================


# 1. Scaling Transformers
# -----------------------------------------------------
info_ = """
where the approach of encoding human knowledge within AI systems was ultimately outdone by increased computations.
在人工智能系统中对人类知识进行编码的方法最终因计算量的增加而被超越.

We have to learn the bitter lesson that building in how we think we think does not work in the long run...
There are now signs that a similar lesson is at play with transformers; while many of the early BERT and GPT descendants focused on tweaking 
the arcgitecture or pretraining objectives, the best-performing models in mid-2021, like GPT-3, are essentially basic scaled-up versions of the 
original models without many architectural modifications. 
the amount of compute and training data must also be scaled in tandem to train these monster.

"""

## 1.1 Scaling Laws
# *****************************************************
info_11 = """
1. The relationship of performance and scale
    尽管许多 NLP 研究人员专注于架构调整或超参数优化（例如调整层数或注意力头）以提高固定数据集上的性能，但缩放定律的含义是，
    实现更好模型的更有效途径是专注于 同时增加 N、C 和 D
2. Smooth power laws
    这些幂律的一个重要特征是，可以推断损失曲线的早期部分，以预测如果训练进行更长时间的话，近似损失会是多少
3. Sample efficiency
    大型模型能够通过较少的训练步骤达到与小型模型相同的性能。 通过比较损失曲线在一定数量的训练步骤中趋于稳定的区域可以看出这一点，
    这表明与简单地扩大模型相比，性能回报递减。
目前尚不清楚幂律缩放是否是 Transformer 语言模型的通用属性。 
目前，我们可以使用缩放定律作为工具来推断大型、昂贵的模型，而无需显式地训练它们。
 然而，扩展并不像听起来那么容易。 
 现在让我们看看绘制这一前沿时出现的一些挑战。
"""

## 1.2 Challenges with Scaling
# *****************************************************
info_ = """
a few of the biggest challenges:
    1. Infrastructure - 工程
    2. Cost - 成本
    3. Dataset curation - 数据质量(控制性别倾向、种族歧视、个人信息实用许可等)
    4. Model evaluation - 模型评估
    5. Deployment - 模型部署

two community-led projects that aim to produce and probe large language models in
the open:
    1. BigScience :a one-year-long research workshop (2021 - 2022)
        research questions surrounding these models (capabilities,
        limitations, potential improvements, bias, ethics, 
        environmental impact, role in
        the general AI/cognitive research landscape)
    2. EleutherAI: 
        This is a decentralized collective of volunteer researchers, engineers, and developers 
            focused on AI alignment, scaling, and open source AI research.
        One of its aims is to train and open-source a GPT-3-sized model, 
            released: GPT-Neo and GPT-J
"""

## 1.3 Attention Please! (making self-attention more efficient)
# *********************************************************
info = """
some method:
    1. Low Rank / Kernels
    2. Memory
    3. Recurrence
    5. Fixed/Factorized/Random Patterns
    6. Learnable Patterns
"""

## 1.4 Sparse Attention
# *********************************************************
info = """
simply limit the number of query-key pairs 
a. global: defines a few special tokens in the sequence that are allowed to attend to all other tokens.
b. band: computes attention over a diagonal band
c. dilated: Sjips some query-key pairs by using a dilated window with gaps
d. random
e. block local: Divides the sequence into blocks and restricts attention within thhese blocks

In practice, most transformer models with sparse attention use a mix of the atomic sparsity patterns
see: Figure 11-6. Sparse attention patterns for recent transformer models
"""

## 1.5 Linearized Attention
# *********************************************************
info = """
$y_i = \sum_{j} \frac{sim(Q_i, K_j)}{\sum_k{Q_i}{K_k}}V_j

The trick behind linearized attention mechanisms is to express the similarity function
as a kernel function that decomposes the operation into two pieces:

sim(Q_i, K_j) = \phi(Q_i)^T\phi(K_j)
"""

# 2.Going Beyond Text
# -----------------------------------------------------
info = """
1. Human reporting bias
    文本中事件的频率可能无法代表其真实频率。一个只根据互联网上的文本训练的模型可能会有一个非常扭曲的世界形象。
2. Common sense
    常识是人类推理的基本品质，但很少被写下来。因此，基于文本训练的语言模型可能知道世界上的许多事实，但缺乏基本的常识推理。
3. Facts
    概率语言模型无法以可靠的方式存储事实，并且可能产生事实错误的文本。类似地，这种模型可以检测命名实体，但无法直接访问有关它们的信息
4. Modality
    语言模型无法连接到其他可以解决先前问题的模式，例如音频或视觉信号或表格数据。
"""
    
## 2.1 Vision
# *********************************************************
info = """
    IGPT: uses the GPT architecture and autoregressive pretraining objective to predict
the next pixel values
    ViT: Vision Transformer a BERT-style take on transformers for vision
        First the image is split into smaller patches, and each of these patches is embedded with a linear projection
        The results strongly resemble the
        token embeddings in BERT, and what follows is virtually identical. The patch embeddings
        are combined with position embeddings and then fed through an ordinary
        transformer encoder. During pretraining some of the patches are masked or distorted,
        and the objective is to predict the average color of the masked patch.
"""
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline # load a ViT model

image = Image.open('images/doge.jpg')
plt.imshow(image)
plt.axis('off')
plt.show()

img_clf = pipeline('image-classification')
preds = img_clf(image)
preds_df = pd.DataFrame(preds)
preds_df

Tables : TAPAS (short for Table Parser)

table_qa = pipeline("table-question-answering")


# 3. Multimodal Transformers
# ====================================================
## 3.1 Speech-to-Text (ASR: autimatic speech recognition)
# *********************************************************
info = """
    wave2vec 2.0:use a transformer layer in combination with a CNN
"""

from datasets import load_dataset    
import soundfile as sf

def map_to_array(batch):
    speech, _ = sf.read(batch['file'])
    batch['speech'] = speech
    return batch


ds = load_dataset('superb', 'asr', split='validation[:1]')
print(ds[0])
ds = ds.map(map_to_array)
asr = pipeline("automatic-speech-recognition")
# from IPython.display import Audio
# display(Audio(ds[0]['speech'], rate=16000))
pred = asr(ds[0]["speech"])
print(pred)
# {'text': 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO
# WELCOME HIS GOSPEL'}

## 3.2 Vision and Text
# *********************************************************

info = """
VisualQA, LayoutLM, DALL'E, CLIP

Models such as LXMERT and VisualBERT use vision models like ResNets to extract
features from the pictures and then use transformer encoders to combine them with
the natural questions and predict an answer

    LayoutLM:  Analyzing scanned business documents like receipts, invoices, or reports is another
        area where extracting visual and layout information can be a useful way to recognize
        text fields of interest
"""


# 4. Where to from Here ?
# ====================================================






