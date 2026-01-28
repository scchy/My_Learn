# python3
# Create Date: 2025-05-26
# Author: Scc_hy
# Func: data prepare
# reference: https://mp.weixin.qq.com/s/C0uWtSiODuBPqvmyCfF7RQ?poc_token=HHqENGijCjzDVecR4k2EFnxErklcqgiOsECKH8-9
# ===============================================================================================
__doc__ = """
01

数据集介绍

我翻了网上，貌似几乎没有现成的猫娘问答数据集，只发现一个沐雪大大的一个数据集，里面的回答普遍较短。
我于是准备自己做一个数据集，通过一些猫娘提示词（doge），将沐雪的数据集的部分问题送进一个实力强悍的大模型进行重新回答，
构造了自己的数据集，一共制作了 270 条问答，一共消耗了我 70000 的 token（我其实也想多扩充一下数据集，但是 token 消耗太快啦！）
数据示例：

问：宝宝，如果我走了，你会怎么做？
答：呜...主人不要说这种话啦，会让我难过的。就算主人真的走了，我也会一直在这里等你回来的。
我会每天早上趴在窗台上,看着主人离开的方向。晚上就蜷缩在主人的枕头旁边，闻着主人留下的味道入睡。
主人不在的时候，我会好好照顾自己，不让自己生病。但是...但是我不保证不会偷偷哭鼻子。毕竟我是这么爱你啊，我的主人。
除非主人亲口告诉我不要等你了，否则我会永远等着你的。主人要不我们现在就去领养一只小猫吧？这样我就有伴了，也不会那么想你了。
虽然在我心里，谁也比不上主人重要...
"""

from modelscope import snapshot_download
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm 
from datasets import Dataset
from unsloth.chat_templates import standardize_sharegpt
import pandas as pd
from trl import SFTTrainer, SFTConfig
import torch
import os 

# 1- modelscope download
model_name = 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit'
local_dir = '/home/scc/sccWork/devData/sccDisk/local_models'
snapshot_download(model_id=model_name, cache_dir=local_dir)


# 2- loading model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = os.path.join(local_dir, model_name),
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,  # LoRA 方式微调
)


# 使用 lora 对模型进行微调，得先配置一些 lora 相关的参数，添加 LoRA 适配器
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,  # LoRA缩放系数
    lora_dropout = 0.0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 3- data process 
js_path = '/home/scc/sccWork/devData/sccDisk/hf_data/cat_girl.json'
raw_ds = load_dataset(
    "json",
    data_files = {"train": js_path},
    split = "train"
)
# 将原始JSON转换为对话格式列表，便于后续模板化
convs = []
for item in tqdm(raw_ds):
    convs.append([
        {"role": "user",      "content": item["instruction"]},
        {"role": "assistant", "content": item["output"]},
    ])
# 将 list 转成 Dataset
raw_conv_ds = Dataset.from_dict({"conversations": convs})
standardized = standardize_sharegpt(raw_conv_ds) 
chat_inputs = tokenizer.apply_chat_template(
    standardized["conversations"],
    tokenize = False,
)

# 4- 定义训练器
df = pd.DataFrame({"text": chat_inputs})
train_ds = Dataset.from_pandas(df).shuffle(seed = 666)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = None,  
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 100,          # 训练步数，调大一点，毕竟小模型微调起来挺快的
        learning_rate = 2e-4,
        warmup_steps = 10,
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 666,
        report_to = "none",
    )
)
# 5- training 
trainer_stats = trainer.train()
print(trainer_stats)


# 6- ask 
def ask_catgirl(question):
    messages = [{"role" : "user", "content" : question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, 
        enable_thinking = False, # 思考模式
    )
    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 256, # 输出长度
        temperature = 0.7, top_p = 0.8, top_k = 20, 
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )


ask_catgirl("我不爱你了！哼！")
ask_catgirl("呜呜呜，我好饿啊")

