# python3
# Create Date: 2023-04-29
# Author: Scc_hy
# Func: 文本生成
# kaggle: kaggle上运行会更快 https://www.kaggle.com/code/scchuy/nlptransformers-chapter5-textgeneration
# =================================================================================
import torch
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
