# python3
# Create Date: 2023-04-29
# Author: Scc_hy
# Func: 文本生成
# kaggle: kaggle上运行会更快 https://www.kaggle.com/code/scchuy/nlptransformers-chapter5-textgeneration
# =================================================================================
import torch
from torch import nn 
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import pandas as pd
import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ----------------------------------
# 一、贪婪搜索Decoding
# ----------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'gpt2' # 'gpt2-xl' 会更大
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


input_text = 'Transformers are the'
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'].to(device)
iter_list = []
n_steps = 8
choice_per_step = 5

with torch.no_grad():
    for step in tqdm(range(n_steps)):
        iter_dict = dict()
        iter_dict['Input'] = tokenizer.decode(input_ids[0])
        output = model(input_ids=input_ids)
        next_token_logits = output.logits[0, -1, :]
        print(f'Step - {step} output.logits.shape=', output.logits.shape)
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
        for idx in range(choice_per_step):
            token_id = sorted_ids[idx]
            token_prob = next_token_probs[token_id].detach().cpu().numpy()
            token_choice = (
                f'{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)'
            )
            iter_dict[f'Choice {idx+1}'] = token_choice
        
        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
        iter_list.append(iter_dict)

pd.DataFrame(iter_list)

# 也可以用 max_new_tokens 作为参数直接输出最大可能的值
input_text = 'Transformers are the'
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'].to(device)
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))

max_length = 128
input_text = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
print(tokenizer.decode(output_greedy[0]))

res_point = """
the main backwards with greedy search decoding : it tends to produce repetitive output sequence, 
which is certainly undesirable in a news article.
--- It's  a common problem with greedy search algorithims
"""

# ----------------------------------
# 二、Beam Search Decoding
# ----------------------------------

def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label


@torch.no_grad()
def sequence_logprob(model, labels, input_len=0):
    output = model(labels)
    log_probs = log_probs_from_logits(
        output.logits[:, :-1, :], labels[:, 1:] # start 2nd
    )
    seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob


logp_org = sequence_logprob(model, output_greedy, input_len=len(input_ids[0]))
print(tokenizer.decode(output_greedy[0]))
print(f'log_prob: {logp_org:.2f}')

output_beam = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False) # no_repeat_ngram_size=2)
logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
print(tokenizer.decode(output_beam[0]))
print(f'log_prob: {logp:.2f}')


point_res = """
no_repeat_ngram_size parameter that tracks which n-grams have been seen 
and sets the next token probability to zero if it would produce a previously seen n-gram    
"""

# ----------------------------------
# 三、Sampling Method  exp(Z/T) / \sum exp(Z/T)
# T 
# - T<1: the distribution becomes peaked around the origin and the rare tokens are suppressed.
# - T>1: the distribution flattens out and each token becomes equally likely
# ----------------------------------
output_temp = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.5, top_k=0)

# ----------------------------------
# 四、 TopK and Nucleus Sampling
# Top-k and nucleus (top-p) sampling: 
#   the basic idea is to restrict the number of possible
#   tokens we can sample from at each timestep.
# - TopK: avoid the low-probability choices by only sampling from the k tokens with the highest probability
# - TopP: dynamic cut 
# - TopK + TopP: choosing tokens with a probability mass of 90%, from a pool of at most 50 tokens.
# ----------------------------------
output_topk = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50)
print(tokenizer.decode(output_topk[0]))


output_topk = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=0.9)
print(tokenizer.decode(output_topk[0]))

output_topk = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.9)
print(tokenizer.decode(output_topk[0]))


