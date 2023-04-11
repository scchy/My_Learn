# python3
# Create Date: 2023-04-08
# Author: Scc_hy
# Func: dataset
# tip: pip install datasets
# =================================================================================


from datasets import list_datasets, load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datasets
datasets.__version__

# 查看有多少数据
all_datasets = list_datasets()
print(f'There are {len(all_datasets)} datasets currently available on the Hub')

# 加载emotion数据
emotions = load_dataset('emotion')
train_ds = emotions['train']
# {'text': 'i didnt feel humiliated', 'label': 0}
print(train_ds[0])
# {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None)}
print(train_ds.features)
# {'text': ['i didnt feel humiliated', 
# 'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake', 
# 'im grabbing a minute to post i feel greedy wrong'], 
# 'label': [0, 0, 3]}
print(train_ds[:3])


# 可以通过set_format 直接转成pd.DataFrame
emotions.set_format(type='pandas')
df = emotions['train'][:]
df.head()

def label_int2str(r):
    return emotions['train'].features['label'].int2str(r)

df['label_name'] = df['label'].map(label_int2str)
df.head()

# -----------------------------------------------------
# 一、Looking at the Class Distribution
# -----------------------------------------------------
df['label_name'].value_counts(ascending=True).plot.barh()
plt.title('Frequency of Classes')
plt.show()


# 类不平衡处理方法
# ------------------------------
# - 对较少的类进行过采样
# - 对其他类进行下采样
# - 获取更多的 数据
# ------------------------------

# 句子长度限制
# DistilBERT token最大长度512，所以需要观察下句子的长度分布
df['Words Per Tweet'] = df['text'].str.split().apply(len)
df.boxplot('Words Per Tweet', by='label_name', grid=False, showfliers=False, color='black')
plt.xlabel('')
plt.show()


# -----------------------------------------------------
# 二、 From Text to Tokens
# 1- Subword Tokenization
#   WordPiece -> BERT & DistilBERT tokenizers
# -----------------------------------------------------
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
text = 'Tokenizing text is a core task of NLP'
encoded_text = tokenizer(text)
# ['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '[SEP]']
# 1- [CLS] -> 开始； [SEP] -> 结束
# 2- 都变小写了
# 3- tokenizing -> 'token', '##izing'; nlp -> 'nl', '##p'   ##指明前面不是空白的
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# [CLS] tokenizing text is a core task of nlp [SEP]
print(tokenizer.convert_tokens_to_string(tokens))
# (30522, 512)
tokenizer.vocab_size, tokenizer.model_max_length


# 2- Tokenizing the Whole DataSet
# [PAD]: 0, [UNK]: 100, [CLS]: 101, [SEP]: 102, [MASK]: 103
# --------------------------------
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


emotions.set_format()
print(tokenize_batch(emotions['train'][:2]))
emotions_encoded = emotions.map(tokenize_batch, batched=True, batch_size=None)
# ['text', 'label', 'input_ids', 'attention_mask']
emotions_encoded['train'].column_names


# -----------------------------------------------------
# 三、 Training a Text Classifier
#  - Feature extraction: 不动预训练的权重，仅仅调整分类层的权重
#  - Fine-tuning: 同时也调整预训练层的权重
# -----------------------------------------------------

# 1- ** Feature extraction **
# -----------------------------------

from transformers import AutoModel
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained('distilbert-base-uncased')

# sample test
text = 'This is a test'
ipt = tokenizer(text, return_tensors='pt')
ipts = {k:v.to(device) for k, v in ipt.items()}
with torch.no_grad():
    out = model(**ipts)


# tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids)
# 返回 [batch_size, n_tokens, hidden_dim] = (1, 6, 768)
# ['[CLS]', 'this', 'is', 'a', 'test', '[SEP]']  对于分类仅仅需要拿取第一个就行
print(out.last_hidden_state[:, 0].shape)


def extract_hidden_states(batch):
    ipts = {k:v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        out = model(**ipts).last_hidden_state
    return {'hidden_state': out[:,0].cpu().numpy()}


emotions_encoded = emotions.map(tokenize_batch, batched=True, batch_size=1000)
emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

emotions_hidden['train'].column_names

import pickle 
with open('emotions.pkl', 'wb') as f:
    pickle.dump(emotions_hidden, f)
    
with open('emotions.pkl', 'rb') as f:
    emotions_hidden = pickle.load(f)


# 1-1 Creating a feature matrix
# --------------------------
x_tr = np.array(emotions_hidden['train']['hidden_state'])
x_val = np.array(emotions_hidden['validation']['hidden_state'])
y_tr = np.array(emotions_hidden['train']['label'])
y_val = np.array(emotions_hidden['validation']['label'])


# 1-2 Visualizing the training set
# --------------------------
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

x_sc = MinMaxScaler().fit_transform(x_tr)
mapper = UMAP(n_components=2, metric='cosine').fit(x_sc)
df_emb = pd.DataFrame(mapper.embedding_, columns=['X', 'Y'])
df_emb['label'] = y_tr

# plot
fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()
cmaps = ['Greys', 'Blues', 'Oranges', 'Reds', 'Purples', 'Greens']
labels = emotions['train'].features['label'].names

for idx, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f'label == {idx}')
    axes[idx].hexbin(
        df_emb_sub['X'], df_emb_sub['Y'], cmap=cmap, gridsize=20, linewidths=(0,)
    )
    axes[idx].set_title(label)
    axes[idx].set_xticks([]), axes[idx].set_yticks([])

plt.tight_layout()
plt.show()


# 1-3 Training a simple classifier
# --------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(x_tr, y_tr)
lr_clf.score(x_val, y_val)

dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(x_tr, y_tr)
dummy_clf.score(x_val, y_val)


def plot_confusion_matrix(y_pred, y_true, labels):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
    plt.title('Normalized confusion matrix')
    plt.show()
    

y_pred = lr_clf.predict(x_val)
plot_confusion_matrix(y_pred, y_val, labels)


# 2- ** Fine-Tuning Transformers **
# -----------------------------------
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch
from huggingface_hub import login
from transformers import Trainer, TrainingArguments
from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer
from torch.nn.functional import cross_entropy

model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


emotions = load_dataset('emotion')
emotions.set_format()
emotions_encoded = emotions.map(tokenize_batch, batched=True, batch_size=None)


# 2-1 加载模型
# AutoModelForSequenceClassification has a classification head on top of the pretrained model outputs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=6)
         .to(device)
)

# 2-2 Defining the performance metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}


batch_size = 64
logging_step = len(emotions_encoded['train']) // batch_size
model_name = f'{model_ckpt}-finetuned-emotion'
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01, # Adam
    evaluation_strategy='epoch',
    disable_tqdm=False,
    logging_steps=logging_step,
    push_to_hub=False,
    log_level='error'
)

# 2-3 train & metrics
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded['train'],
    eval_dataset=emotions_encoded['validation']
)
trainer.train()

preds_output = trainer.predict(emotions_encoded['validation'])
preds_output.metrics

y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_pred, y_val, labels)

# 2-4 error analysis
def forward_pass_with_label(batch):
    ipts = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**ipts)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch['label'].to(device), reduction='none')
    return {
        'loss': loss.cpu().numpy(),
        'predicted_label': pred_label.cpu().numpy()
    }
        

emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# 计算损失
emotions_encoded['validation'] = emotions_encoded['validation'].map(
    forward_pass_with_label, batched=True, batch_size=16
)

# DataFrame -> texts, losses, predicted/true labels
emotions_encoded.set_format('pandas')
cols = ['text', 'label', 'predicted_label', 'loss']
df_test = emotions_encoded['validation'][:][cols]
df_test['label'] = df_test['label'].map(label_int2str)
df_test['predicted_label'] = df_test['predicted_label'].map(label_int2str)
df_test.sort_values('loss', ascending=False).head(10)


# 2-5 save and sharing the model
trainer.push_to_hub(commit_message='Training completed!')
from transformers import pipeline
model_id = f'transformersbook/{model_ckpt}-finetuned-emotion'
clf = pipeline('text-classification', model=model_id)
customer_tweet = 'I saw a movie today and it was really good.'
preds = clf(customer_tweet, return_all_scores=True)
preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100*preds_df['score'], color='c0')
plt.title(f'{customer_tweet}')
plt.ylabel('Class Probability (%)')
plt.show()



