
from lmdeploy import turbomind as tm
from datetime import datetime 
from tqdm.auto import tqdm
import sys
from pynvml import (
    nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo, 
    nvmlDeviceGetName,  nvmlShutdown
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def cuda_mem():
    # 21385MiB / 81920MiB
    fill = 21385
    n = datetime.now()
    nvmlInit()
    # 创建句柄
    handle = nvmlDeviceGetHandleByIndex(0)
    # 获取信息
    info = nvmlDeviceGetMemoryInfo(handle)
    # 获取gpu名称
    gpu_name = nvmlDeviceGetName(handle)
    # 查看型号、显存、温度、电源
    print("[ {} ]-[ GPU{}: {}".format(n, 0, gpu_name), end="    ")
    print("总共显存: {:.3}G".format((info.total // 1048576) / 1024), end="    ")
    print("空余显存: {:.3}G".format((info.free // 1048576) / 1024), end="    ")
    model_use = (info.used  // 1048576) - fill
    print("模型使用显存: {:.3}G({}MiB)".format( model_use / 1024, model_use))
    nvmlShutdown()


def tm_chat(model, query):
    generator = model.create_instance()
    # process query
    # query = "你是谁"
    prompt = model.model.get_prompt(query)
    input_ids = model.tokenizer.encode(prompt)
    # inference
    for outputs in generator.stream_infer(
            session_id=0,
            input_ids=[input_ids]):
        res, tokens = outputs[0]

    response = model.tokenizer.decode(res.tolist())
    return response


def hf_model(query_list):
    model_name_or_path = "/root/share/temp/model_repos/internlm-chat-7b/"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    model = model.eval()
    cuda_mem()
    for q in tqdm(query_list):
        res = model.chat(tokenizer, q)
        print(f'res={res}')
        cuda_mem()
    print('**'*25)
    cuda_mem()


def main():
    print('**'*25)
    q_list = [
         '你是谁？',
        'what is the dose for vit b12 tabet?',
        'how long do opioid withdraws last',
        'why did my doctor give me levetiracetam'
    ]
    model_path = sys.argv[1]
    if model_path == 'HF':
        hf_model(q_list)
        return None
    tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')
    cuda_mem()
    for q in tqdm(q_list):
        res = tm_chat(tm_model, q)
        print(f'res={res}')
        cuda_mem()
    print('**'*25)
    cuda_mem()


if __name__ == '__main__':
    main()
