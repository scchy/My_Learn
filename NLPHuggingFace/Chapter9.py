# python3
# Create date: 2023-10-07
# Author: Scc_hy
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

