# python3
# Create Date: 2026-03-19
# Author: Scc_hy 
# =========================================================================================

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from modelscope.hub.snapshot_download import snapshot_download


# 指定模型ID
model_name = "Qwen/Qwen3-0.6B"
out_dir = os.path.join(os.environ['SCC_DISK'], 'model_weight')
print(f'{out_dir=}')
model_id = os.path.join(out_dir, model_name)
if not os.path.exists(model_id):
    snapshot_download(model_id=model_name, cache_dir=out_dir)


def test_temperature(model, tokenizer):
    # 准备对话输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，3句话之内简单介绍下Agent"}
    ]
    # 使用分词器的模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print(f"编码前输入:{messages}\n编码后的输入文本:")
    print(model_inputs)

    # 使用模型生成回答
    # max_new_tokens 控制了模型最多能生成多少个新的Token
    for t in [2.0, 1.0, 0.6, 0.2]:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            generation_config=GenerationConfig(
                temperature=t,
                top_p=0.9,
                do_sample=True
            )
        )

        # 将生成的 Token ID 截取掉输入部分
        # parsing thinking content
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        print(f"\n模型的回答 [{t=}]:")
        print(f"=====================================")
        print("thinking content:", thinking_content)
        print("content:", content)


def model_respone(model, tokenizer, prompt="你好，3句话之内简单介绍下Agent"):
    # 准备对话输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # 使用分词器的模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # 使用模型生成回答
    # max_new_tokens 控制了模型最多能生成多少个新的Token
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        generation_config=GenerationConfig(
            temperature=0.6,
            top_p=0.9,
            do_sample=True
        )
    )

    # 将生成的 Token ID 截取掉输入部分
    # parsing thinking content
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print(f"=====================================")
    print("thinking content:", thinking_content)
    print("content:", content)
    return thinking_content, content
    
    
def zero_few_cot_ask(model, tokenizer):
    zero_shot = """
请解答这道数学题：
小明有 24 颗糖，给了小红 1/3，又给了小刚剩下的一半，最后还剩多少颗？
直接给出答案。/no_think
    """
    few_shot = """
请按步骤解答数学题：

【示例1】
题目：小明有 10 个苹果，吃掉 3 个，又买了 5 个，现在有几个？
解答：
1. 原有 10 个
2. 吃掉 3 个：10 - 3 = 7 个
3. 又买 5 个：7 + 5 = 12 个
答案：12 个

【示例2】
题目：一本书 120 页，第一天看 1/4，第二天看剩下的 1/3，还剩多少页？
解答：
1. 总页数 120 页
2. 第一天看：120 × 1/4 = 30 页，剩 90 页
3. 第二天看：90 × 1/3 = 30 页，剩 60 页
答案：60 页

【待解答】
题目：小明有 24 颗糖，给了小红 1/3，又给了小刚剩下的一半，最后还剩多少颗？
解答：/no_think
    """
    cot = """
请一步一步思考并解答这道题。在得出最终答案前，先解释你的推理过程。
题目：小明有 24 颗糖，给了小红 1/3，又给了小刚剩下的一半，最后还剩多少颗？
思考过程：/think
    """
    print('\n>>>>>> zero_shot:', zero_shot)
    _, _ = model_respone(model, tokenizer, prompt=zero_shot)
    
    print('\n>>>>>> few_shot:', few_shot)
    _, _ = model_respone(model, tokenizer, prompt=few_shot)
    
    print('\n>>>>>> CoT:', cot)
    _, _ = model_respone(model, tokenizer, prompt=cot)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 加载模型，并将其移动到指定设备
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    print("模型和分词器加载完成！")
    test_temperature(model, tokenizer)
    print("--"*30)
    zero_few_cot_ask(model, tokenizer)
    


