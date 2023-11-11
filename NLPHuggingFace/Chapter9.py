# python3
# Create date: 2023-10-07
# Author: Scc_hy
# Chapter: 9- Dealing With Few to No Labels
# ====================================================================================

__doc__ = """
Several technique that can be used to improve model performance in the absence of large amounts of labeled data.
1. Do you have labeled data ?
- Yes -> 2
- No -> Zero-shot Learning

2. How many labels ?
- A lot -> Fine-tune model
- A few -> 3

3. Do you have unlabeled data ?
- No  -> Embedding lookup / Few-shot learning
- Yes -> Domain adaptation / UDA/UST 
"""

import pandas as pd 
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import (
    DataCollatorForLanguageModeling, set_seed, AutoModelForMaskedLM
)
import torch
from scipy.special import expit as sigmoid

# 1st、 Preparing the Data
# -----------------------------------------------------------------
# almost 10,000 issues in the dataset
dataset_url = "https://git.io/nlp-with-transformers"
df_issues = pd.read_json(dataset_url, lines=True)
print(f'df.shape = f{df_issues.shape}')

cols = ["url", "id", "title", "user", "labels", "state", "created_at", "body"]
df_issues.loc[2, cols].to_frame()
df_issues["labels"] = df_issues["labels"].apply(lambda x: [meta["name"] for meta in x])
df_issues[["labels"]].head()
# observe 
df_issues['labels'].apply(lambda x: len(x)).value_counts().to_frame().T 
df_counts = df_issues['labels'].explode().value_counts()
df_counts.to_frame().head(8).T

# focus problems 
label_map = {
    "Core: Tokenization": "tokenization",
    "New model": "new model",
    "Core: Modeling": "model training",
    "Usage": "usage",
    "Core: Pipeline": "pipeline",
    "TensorFlow": "tensorflow or tf",
    "PyTorch": "pytorch",
    "Examples": "examples",
    "Documentation": "documentation"
}

def filter_labels(x):
    return [label_map[label] for label in x if label in label_map]

df_issues["labels"] = df_issues["labels"].apply(filter_labels)
all_labels = list(label_map.values())
df_counts = df_issues["labels"].explode().value_counts()
df_counts.to_frame().T

df_issues["split"] = "unlabeled"
mask = df_issues["labels"].apply(lambda x: len(x)) > 0
df_issues.loc[mask, "split"] = "labeled"
# split labeled & unlabeled 
df_issues["split"].value_counts().to_frame()

# look at an example
for column in ["title", "body", "labels"]:
    print(f"{column}: {df_issues[column].iloc[26][:500]}\n")

df_issues["text"] = df_issues.apply(lambda x: x["title"] + "\n\n" + x["body"], axis=1)
len_before = len(df_issues)
df_issues = df_issues.drop_duplicates(subset="text")
print(f"Removed {(len_before-len(df_issues))/len_before:.2%} duplicates.")


df_issues["text"].str.split().apply(len).hist(bins=np.linspace(0, 500, 50), grid=False, edgecolor="C0")
plt.title("Words per issue")
plt.xlabel("Number of words")
plt.ylabel("Number of issues")
plt.show()

# 2nd、 Creating Training Sets
# Creating training and validation sets is a bit trickier for multlilabel problems because
# there is no guaranteed balance for all labels.
# -----------------------------------------------------------------
mlb = MultiLabelBinarizer()
mlb.fit([all_labels])
mlb.transform([['tokenization', 'new model'], ['pytorch']])
# array([[0, 0, 0, 1, 0, 0, 0, 1, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0]])

def balanced_split(df, test_size=0.5):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    labels = mlb.transform(df['labels'])
    ind_tr, _, ind_te, _ = iterative_train_test_split(ind, labels, test_size)
    return df.iloc[ind_tr[:, 0]], df.iloc[ind_te[:, 0]]



df_clean = df_issues[['text', 'labels', 'split']].reset_index(drop=True).copy(deep=True)
df_unsup = df_clean.loc[df_clean['split'] == 'unlabeled', ['text', 'labels']].reset_index(drop=True)
df_sup = df_clean.loc[df_clean['split'] == 'labeled', ['text', 'labels']].reset_index(drop=True)

np.random.seed(0)
df_train, df_tmp = balanced_split(df_sup, test_size=0.5)
df_valid, df_test = balanced_split(df_tmp, test_size=0.5)

ds = DatasetDict({
    'train' : Dataset.from_pandas(df_train.reset_index(drop=True)),
    'valid' : Dataset.from_pandas(df_valid.reset_index(drop=True)),
    'test' : Dataset.from_pandas(df_test.reset_index(drop=True)),
    'unsup' : Dataset.from_pandas(df_sup.reset_index(drop=True)),
})

## Create Training Slices
all_indices = np.expand_dims(np.arange(len(ds['train'])), axis=1)
indices_pool = all_indices
labels = mlb.transform(ds['train']['labels'])
train_samples = [8, 16, 32, 64, 128]
train_slices, last_k = [], 0

for i, k in enumerate(train_samples):
    # Split off samples necessary to fill the gap to the next split size
    indices_pool, labels, new_slice, _ = iterative_train_test_split(
        indices_pool, labels, (k - last_k)/len(labels)
    )
    last_k = k
    if i == 0:
        train_slices.append(new_slice)
    else:
        train_slices.append(np.concatenate((train_slices[-1], new_slice)))


# Add full dataset as last slice
train_slices.append(all_indices)
train_samples.append(len(ds['train']))
train_slices = [np.squeeze(s) for s in train_slices]
# Note that this iterative approach only approximately splits the samples to the desired
# size, since it is not always possible to find a balanced split at a given split size:



# 3rd、 Implementing a Naive Bayesline
# Whenever you start a new NLP project, it's always a good idea to implement 
# a set of strong baselines. Two main reason:
## 1. 基于正则表达式、手工规则或非常简单的模型的基线可能已经很好地解决了这个问题。在这些情况下，没有理由推出像transformer这样的大炮，因为在生产环境中部署和维护transformer通常更复杂。
## 2. 基线在您探索更复杂的模型时提供了快速检查。例如，假设您对BERT进行了大规模训练，并在验证集上获得了80%的准确率。你可能会把它当作一个硬数据集，然后到此为止。
#       但是，如果你知道像逻辑回归这样的简单分类器可以获得95%的准确率呢？这会引起您的怀疑，并促使您调试您的模型。
# -----------------------------------------------------------------

def prepare_labels(batch):
    batch['label_ids'] = mlb.transform(batch['labels'])
    return batch

ds = ds.map(prepare_labels, batched=True)

macro_scores, micro_scores = defaultdict(list), defaultdict(list)

for tr_s in train_slices:
    ds_tr_sample = ds['train'].select(tr_s)
    y_tr = np.array(ds_tr_sample['label_ids'])
    y_te = np.array(ds['test']['label_ids'])
    # Use a simple count vectorizer to encode our texts as token counts
    count_vect = CountVectorizer()
    x_tr_counts = count_vect.fit_transform(ds_tr_sample['text'])
    x_te_counts = count_vect.transform(ds['test']['text'])
    # Create and train our model
    clf_ = BinaryRelevance(classifier=MultinomialNB())
    clf_.fit(x_tr_counts, y_tr)
    # Generate predictions and evaluate
    y_pred_te = clf_.predict(x_te_counts)
    clf_report = classification_report(y_te, y_pred_te, target_names=mlb.classes_, zero_division=0, output_dict=True)
    # store metrics
    macro_scores['Naive Bayes'].append(clf_report['macro avg']['f1-score'])
    micro_scores['Naive Bayes'].append(clf_report['micro avg']['f1-score'])


# plot
def plot_metrics(micro_scores, macro_scores, sample_sizes, current_model):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for run in micro_scores.keys():
        axes[0].plot(sample_sizes, micro_scores[run], label=run, linewidth=2, linestyle=None if run == current_model else "dashed")
        axes[1].plot(sample_sizes, macro_scores[run], label=run, linewidth=2, linestyle=None if run == current_model else "dashed")

    axes[0].set_title("Micro F1 scores")
    axes[1].set_title("Macro F1 scores")
    axes[0].set_ylabel("Test set F1 score")
    axes[0].legend(loc="lower right")
    for ax in axes:
        ax.set_xlabel("Number of training samples")
        ax.set_xscale("log")
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels(sample_sizes)
        ax.minorticks_off()
    plt.tight_layout()
    plt.show()

plot_metrics(micro_scores, macro_scores, train_samples, "Naive Bayes")


# Working with No Labeled Data
# ****************************
from transformers import pipeline
import os
os.environ['CURL_CA_BUNDLE'] = ''

pipe = pipeline('fill-mask', model='bert-base-uncased')
movie_desc = "The main characters of the movie madacasar are a lion, a zebra, a giraffa, and a hippo. "
prompt = "The movies is about [MASK]. "

out = pipe(movie_desc + prompt) # , targets=["animals", "cars"])

for ele in out:
    print(f'Token {ele["token_str"]}: \t{ele["score"]:.3f}%')


# Using the masked language model for classification is a nice trick,
pipe = pipeline("zero-shot-classification", device=0)
sample = ds["train"][0]
print(f"Labels: {sample['labels']}")
output = pipe(sample["text"], all_labels, multi_label=True)
print(output["sequence"][:400])
print("\nPredictions:")
for label, score in zip(output["labels"], output["scores"]):
    print(f"{label}, {score:.2f}")

# it might work much better for some domains than others, depending on how close they are to the training data.

def zero_shot_pipeline(example):
    out = pipe(example['text'], all_labels, multi_label=True)
    example['predicted_labels'] = out['labels']
    example['scores'] = out['scores']
    return example


ds_zero_shot = ds["valid"].map(zero_shot_pipeline)
# the next step is to determine which set of labels should be assigned to each example. There are a few options we experiment with
# 1- Define a threshold and select all labels above the threshold
# 2- Pick the Top K labels with k highest scores

def get_preds(example, threshold=None, topk=None):
    preds = []
    if threshold:
        for label, score in zip(example["predicted_labels"], example["scores"]):
            if score >= threshold:
                preds.append(label)
    elif topk:
        for i in range(topk):
            preds.append(example["predicted_labels"][i])
    else:
        raise ValueError("Set either `threshold` or `topk`.")
    return {"pred_label_ids": list(np.squeeze(mlb.transform([preds])))}


def get_clf_report(ds):
    y_true = np.array(ds["label_ids"])
    y_pred = np.array(ds["pred_label_ids"])
    return classification_report(
        y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True
    )


macros, micros = [], []
topks = [1, 2, 3, 4]
for topk in topks:
    ds_zero_shot = ds_zero_shot.map(get_preds, batched=False, fn_kwargs={'topk': topk})
    clf_report = get_clf_report(ds_zero_shot)
    micros.append(clf_report['micro avg']['f1-score'])
    macros.append(clf_report['macro avg']['f1-score'])
    
plt.plot(topks, micros, label='Micro F1')
plt.plot(topks, macros, label='Macro F1')
plt.xlabel("Top-k")
plt.ylabel("F1-score")
plt.legend(loc='best')
plt.show()


desc = """
the best results are obtained by selecting the label with the highest score per example (top 1).
This is perhaps not so surprising, given that most of the examples in our datasets have only one label
"""


macros, micros = [], []
thresholds = np.linspace(0.01, 1, 100)
for threshold in thresholds:
    ds_zero_shot = ds_zero_shot.map(get_preds,
    fn_kwargs={"threshold": threshold})
    clf_report = get_clf_report(ds_zero_shot)
    micros.append(clf_report["micro avg"]["f1-score"])
    macros.append(clf_report["macro avg"]["f1-score"])
plt.plot(thresholds, micros, label="Micro F1")
plt.plot(thresholds, macros, label="Macro F1")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.legend(loc="best")
plt.show()

best_t, best_micro = thresholds[np.argmax(micros)], np.max(micros)
print(f'Best threshold (micro): {best_t} with F1-score {best_micro:.2f}.')
best_t, best_macro = thresholds[np.argmax(macros)], np.max(macros)
print(f'Best threshold (macros): {best_t} with F1-score {best_macro:.2f}.')

# Best threshold (micro): 0.75 with F1-score 0.46.
# Best threshold (macros): 0.72 with F1-score 0.42.
# Since the top-1 method performs best, let’s use this to compare zero-shot classification
## against Naive Bayes on the test set:

ds_zero_shot = ds['test'].map(zero_shot_pipeline)
ds_zero_shot = ds_zero_shot.map(get_preds, fn_kwargs={'topk': 1})
clf_report = get_clf_report(ds_zero_shot)
for train_slice in train_slices:
    macro_scores['Zero Shot'].append(clf_report['macro avg']['f1-score'])
    micro_scores['Zero Shot'].append(clf_report['micro avg']['f1-score'])

plot_metrics(micro_scores, macro_scores, train_samples, "Zero Shot")

res_ = """
1. If we have less than 50 labeled samples, the zero-shot pipeline handily outperforms the baseline.
2. Even above 50 samples, the performance of the zero-shot pipeline is superior when considering the
micro and macro F_1 score. The results for the micro F1 score tell us that the baseline performs well
on the frequent classes, while the zero-shot pipeline excels at those since it does not require any examples to learn from.


If you find it difficult to get good results on your own dataset, here are a few things
you can do to improve the zero-shot pipeline:

- The way the pipeline works makes it very sensitive to the names of the labels. If the names 
dont make such sense or are not easily connected to the texts, the pipeline will likely perform poorly.
Either try using different names or use several names in parallel and aggregate the in extra step.
- Another thing you can improve is the form of the hypothesis. By default it is `hypothesis="This is example is about {}"`
, but you can pass any other text to the pipeline.Depending on the use case, this might improve the performance. 
"""


# Working with Few Lables
# ****************************
# data augmentation that can help us multiply the title labeled data that we have.

## Data Augmentation
# ------------------
info = """
Back translation
    Take a text in the source language, translate it into one or more target languages
    using machine translation, and then translate it back to the source language. Back
    translation tends to works best for high-resource languages or corpora that don’t
    contain too many domain-specific words

Token perturbations
    Given a text from the training set, randomly choose the perform simple transformations tends to works 
    best for high-resource languages or corpora that dont contain too many domain-specific words.
    
https://amitness.com/2020/05/data-augmentation-for-nlp/

We’ll use the ContextualWordEmbsAug augmenter from NlpAug to leverage the contextual
word embeddings of DistilBERT for our synonym replacements. Let’s start
with a simple example:
"""
from transformers import set_seed
import nlpaug.augmenter.word as naw

set_seed(3)
aug = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased',
    device='cpu', action='substitute'
)
text = "Transformers are the most popular toys"
print(f"Original text: {text}")
print(f"Augmented text: {aug.augment(text)}")

def augment_text(batch, transformations_per_examples=1):
    text_aug, label_ids = [], []
    for text, labels in zip(batch['text'], batch['label_ids']):
        text_aug += [text]
        label_ids += [labels]
        for _ in range(transformations_per_examples):
            text_aug += [aug.augment(text)]
            label_ids += [labels]
    return {'text': text_aug, 'label_ids': label_ids}



for tr_s in train_slices:
    ds_tr_sample = ds['train'].select(tr_s)
    # Aug
    ds_tr_sample = ds_tr_sample.map(augment_text, batched=True, remove_columns=ds_tr_sample.column_names).shuffle(seed=42)
    y_tr = np.array(ds_tr_sample['label_ids'])
    y_te = np.array(ds['test']['label_ids'])
    # Use a simple count vectorizer to encode our texts as token counts
    count_vect = CountVectorizer()
    x_tr_counts = count_vect.fit_transform(ds_tr_sample['text'])
    x_te_counts = count_vect.transform(ds['test']['text'])
    # Create and train our model
    clf_ = BinaryRelevance(classifier=MultinomialNB())
    clf_.fit(x_tr_counts, y_tr)
    # Generate predictions and evaluate
    y_pred_te = clf_.predict(x_te_counts)
    clf_report = classification_report(y_te, y_pred_te, target_names=mlb.classes_, zero_division=0, output_dict=True)
    # store metrics
    macro_scores['Naive Bayes + Aug'].append(clf_report['macro avg']['f1-score'])
    micro_scores['Naive Bayes + Aug'].append(clf_report['micro avg']['f1-score'])


plot_metrics(micro_scores, macro_scores, train_samples, "Naive Bayes + Aug")

## Using Embeddings as a Lookup Table
# ------------------------------------------
model_ckpt = 'miguelvictor/python-gpt2-large'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)


def mean_pooling(model_output, attention_mask):
    token_embedding = model_output[0]
    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embedding.size())
        .float()
    )
    sum_embedding = torch.sum(token_embedding * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embedding / sum_mask


def embed_text(examples):
    ipts = tokenizer(examples['text'], padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**ipts)
    
    pooled_embeds = mean_pooling(model_output, ipts['attention_mask'])
    return {'embedding': pooled_embeds.cpu().numpy()}


tokenizer.pad_token = tokenizer.eos_token
embs_train = ds["train"].map(embed_text, batched=True, batch_size=16)
embs_valid = ds["valid"].map(embed_text, batched=True, batch_size=16)
embs_test = ds["test"].map(embed_text, batched=True, batch_size=16)
# you can think of this as a search engine for embeddings,
embs_train.add_faiss_index("embedding")


i, k = 0, 3 # Select the first query and 3 nearest neighbors
rn, nl = "\r\n\r\n", "\n"

query = np.array(embs_valid[i]['embedding'], dtype=np.float32)
scores, samples = embs_train.get_nearest_examples('embedding', query, k=k)
print(f"QUERY LABELS: {embs_valid[i]['labels']}")
print(f"QUERY TEXT:\n{embs_valid[i]['text'][:200].replace(rn, nl)} [...]\n")
print("="*50)
print(f"Retrieved documents:")
for score, label, text in zip(scores, samples["labels"], samples["text"]):
    print("="*50)
    print(f"TEXT:\n{text[:200].replace(rn, nl)} [...]")
    print(f"SCORE: {score:.2f}")
    print(f"LABELS: {label}")


def get_sample_preds(sample, m):
    """
    decide to use the labels with appeared m times 
    """
    return (np.sum(sample['label_ids'], axis=0) >= m).astype(int)


def find_best_k_m(ds_train, valuid_queries, valid_labels, max_k=17):
    max_k = min(len(ds_train), max_k)
    perf_micro = np.zeros((max_k, max_k))
    perf_macro = np.zeros((max_k, max_k))
    for k in range(1, max_k):
        # try severel times
        for m in range(1, k+1):
            _, samples = ds_train.get_nearest_examples_batch(
                'embedding',
                valuid_queries,
                k=k
            )
            y_pred = [get_sample_preds(s, m) for s in samples]
            clf_report = classification_report(
                valid_labels,
                y_pred,
                target_names=mlb.classes_,
                zero_division=0,
                output_dict=True
            )
            perf_micro[k, m] = clf_report['micro_avg']['f1-score']
            perf_macro[k, m] = clf_report['marco_avg']['f1-score']
    return perf_micro, perf_macro


valid_labels = np.array(embs_valid["label_ids"])
valid_queries = np.array(embs_valid["embedding"], dtype=np.float32)
perf_micro, perf_macro = find_best_k_m(embs_train, valid_queries, valid_labels)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
ax0.imshow(perf_micro)
ax1.imshow(perf_macro)
ax0.set_title("micro scores")
ax0.set_ylabel("k")
ax1.set_title("macro scores")
for ax in [ax0, ax1]:
    ax.set_xlim([0.5, 17 - 0.5])
    ax.set_ylim([17 - 0.5, 0.5])
    ax.set_xlabel("m")
plt.show()

k, m = np.unravel_index(perf_micro.argmax(), perf_micro.shape)
print(f'Best k: {k}, best m: {m}')

# use k=15, m=5
embs_train.drop_index("embedding")
test_labels = np.array(embs_test["label_ids"])
test_queries = np.array(embs_test["embedding"], dtype=np.float32)
for train_slice in train_slices:
    # Create a Faiss index from training slice
    embs_train_tmp = embs_train.select(train_slice)
    embs_train_tmp.add_faiss_index("embedding")
    # Get best k, m values with validation set
    perf_micro, _ = find_best_k_m(embs_train_tmp, valid_queries, valid_labels)
    k, m = np.unravel_index(perf_micro.argmax(), perf_micro.shape)
    # Get predictions on test set
    _, samples = embs_train_tmp.get_nearest_examples_batch(
        "embedding",
        test_queries,
        k=int(k) # 15
    )
    # m = 5
    y_pred = np.array([get_sample_preds(s, m) for s in samples])
    clf_report = classification_report(
        test_labels, 
        y_pred,
        target_names=mlb.classes_, 
        zero_division=0, 
        output_dict=True
    )
    macro_scores["Embedding"].append(clf_report["macro avg"]["f1-score"])
    micro_scores["Embedding"].append(clf_report["micro avg"]["f1-score"])

plot_metrics(micro_scores, macro_scores, train_samples, "Embedding")

## Fine-Tuning a Vanilla Transformer
# ------------------------------------------
model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenize(batch['text'], truncation=True, max_length=128)


ds_enc = ds.map(tokenize, batched=True)
ds_enc = ds_enc.remove_columns(['labels', 'text'])
ds_enc.set_format('torch')
ds_enc = ds_enc.map(lambda x: {'label_ids_f': x['label_ids'].to(torch.float)}, remove_columns=['label_ids'])
ds_enc.rename_column('label_ids_f', 'label_ids')

training_args_fine_tune = TrainingArguments(
    output_dir='/home/scc/sccWork/localModels/chapter9_fine_tune_model',
    num_train_epochs=20,
    learning_rate=3e-5,
    lr_scheduler_type='constant',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    weight_decay=0.0,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_emd=True,
    metric_for_best_model='micro f1', # We need the F1-score to choose the best model
    save_total_limit=1,
    log_level='error'    
)


def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = sigmoid(pred.predictions)
    y_pred = (y_pred > 0.5).astype(float)
    clf_dict = classification_report(
        y_true, y_pred, target_names=all_labels, zero_division=0, output_dict=True
    )
    return {"micro f1": clf_dict["micro avg"]["f1-score"],
            "macro f1": clf_dict["macro avg"]["f1-score"]}


config = AutoConfig.from_pretrained(model_ckpt)
config.num_labels = len(all_labels)
config.problem_type = 'multi_label_classification'

for tr_s in train_slices:
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,
        config=config)
    trainer = Trainer(
        model=model, tokenizer=tokenize,
        args=training_args_fine_tune,
        compute_metrics=compute_metrics,
        train_dataset=ds_enc['train'].select(tr_s),
        eval_dataset=ds_enc['valid']
    )
    
    trainer.train()
    pred = trainer.predict(ds_enc["test"])
    metrics = compute_metrics(pred)
    macro_scores["Fine-tune (vanilla)"].append(metrics["macro f1"])
    micro_scores["Fine-tune (vanilla)"].append(metrics["micro f1"])


plot_metrics(micro_scores, macro_scores, train_samples, "Fine-tune (vanilla)")
info = """
fine tune vanilla BERT model
before this the behavior is a bit erratic, which is again due to training a model on a
small sample where some labels can be unfavorably unbalances.
"""

## In-Context and Few-Shot Learning with Prompts
# -------------------------------------------------- 
prompt = """\
Translate English to French:
thanks =>
"""

# Leveraging Unlabeled Data
# ****************************
info_ = """
domain adaptation (which we also saw for question answering in Chapter 7).
Instead of retraining the language model from scratch, we can continue training it
on data from our domain.

In this step we use the classic language model objective of predicting masked words,
which means we dont need any labeled data. 
After that we can load the adapted model as a classifier and fine-tune it, 
thus leveraging the unlabeled data.
"""
## Fine-Tuning a Language Model
# -------------------------------------------------- 

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=128, return_special_tokens_mask=True)

ds_mlm = ds.map(tokenize, batched=True)
ds_mlm = ds_mlm.rename_column(['labels', 'text', 'label_ids'])

info_ = """
masks random tokens and creates labels for these sequences

A much more elegant solution is to use data collator. Remember that the data collator is 
the function that builds the bridge between the dataset and the model call.
A batch is sampled from dataset, and the data collator prepares the elements in the batch to 
feed them to model.

"""
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15
)
set_seed(3)
data_collator.return_tensors = 'np'
ipts = tokenizer('Transformers are awesome!', return_tensors='np')
opts = data_collator([{'input_ids': ipts['input_ids'][0]}])
pd.DataFrame({
    'Original tokens': tokenize.convert_ids_to_tokens(ipts['input_ids'][0]),
    'Masked tokens': tokenize.convert_ids_to_tokens(opts['iput_ids'][0]),
    'Original input_ids': ipts['input_ids'][0],
    'Masked input_ids': opts['iput_ids'][0],
    'Labels': opts['labels'][0] # the entries containing -100 are ignored when calculating the loss
}).T

data_collator.return_tensors = 'pt'
training_args = TrainingArguments(
    output_dir=f'{model_ckpt}-issues-128',
    per_device_train_batch_size=32,
    logging_strategy='epoch',
    save_strategy='no',
    num_train_epochs=16,
    push_to_hub=True,
    log_level='error',
    report_to='none'
)
trainer = Trainer(
    model=AutoModelForMaskedLM.from_pretrained('bert-base-uncased'),
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds_mlm['unsup'],
    eval_dataset=ds_mlm['train']
)
trainer.train()
trainer.push_to_hub('Training complete!')
# plot history
df_log = pd.DataFrame(trainer.state.log_history)
df_log.dropna(subset=['evel_loss']).reset_index()['eval_loss'].plot(label='Validation')
df_log.dropna(subset=['loss']).reset_index()['loss'].plot(label='Train')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


## Fine-Tuning a Classifier
# -------------------------------------------------- 
model_ckpt = f'{model_ckpt}-issues-128'
config = AutoConfig.from_pretrained(model_ckpt)
config.num_labels = len(all_labels)
config.problem_type = 'multi_label_classification'

for tr_s in train_slices:
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args_fine_tune,
        compute_metrics=compute_metrics,
        train_dataset=ds_enc['train'].select(tr_s)
        eval_dataset=ds_enc['vaild']
    )
    trainer.train()
    pred = trainer.predict(ds_enc['test'])
    metrics = compute_metrics(pred)
    macro_scores['Fine-tune (DA)'].append(metrics['macro f1'])
    micro_scores['Fine-tune (DA)'].append(metrics['micro f1'])

plot_metrics(micro_scores, macro_scores, train_samples, "Fine-tune (DA)")
summary = """
This highlights that domain adaption can provice a slight boost to the model's performance with 
unlabeled dara and little effort. Naturally, the more unlabeled data and the less labeled data you have,
the more impact you will get with this method.
Before we conclude this chapter, we'll show you a few more tricks for taking advantage of unlabeled data.
"""

## Advanced Methods
# --------------------------------------------------
info_ = """
there are sophisticated methods than can leverage unlabeled data even further.

1. Unsupervised data augmentation (UDA)
    - loss = Supervised Cross-entropy Loss + Unsupervised Consistency Loss
        - Unsupervised Consistency Loss = Loss(P(y|x), P(y|\hat{ x })
        - x: unlabeled data
        - \hat{ x }: ublabeled dara Augmentations (Back translation / RandAugment / TF-IDF word replacement)
    - The performance if this approach is quite impressive:
        with a handful of labeled examples, BERT models trained with UDA get similar performance to models trained
        on thousands of examples.
2. Uncertainty-aware self-training (UST)
    - train a teacher model on the labeled data and then use that model to create pseudo-labels on the unlabeled data.
    then a student is trained on the pseudo-labeled data, and after training it becomes the teacher for the next iteration.
    - pseudo-label generation:
        - same input is fed several times through the mdoel with dropout turned on.
        - the variance in the predictions gives a proxy for the certainty of the model on a specific sample.
        - uncertainty measure the pesudo-labels are then sampled using a method called Bayesian Active Learning by Disagreement (BALD)

"""



