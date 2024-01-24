
# 1- 基础作业

使用 OpenCompass 评测 InternLM2-Chat-7B 模型在 C-Eval 数据集上的性能

1. 模型下载
```python
# /root/opencompass/InternLM/Shanghai_AI_Laboratory/internlm2-chat-7b
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(model_id='Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='./InternLM')
```
2. 开启评测 `nohup sh eval.sh > __eval.log &`
```shell
# eval.sh 

cd /root/opencompass/opencompass
model_path=/root/opencompass/InternLM/Shanghai_AI_Laboratory/internlm2-chat-7b
python run.py --datasets ceval_gen \
--hf-path ${model_path} \
--tokenizer-path ${model_path} \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--model-kwargs device_map='auto' trust_remote_code=True \
--max-seq-len 2048 \
--max-out-len 16 \
--batch-size 4  \
--num-gpus 1  \
--debug

```
3. 评测结果 同时 和 `internlm-chat-7B`进行了比较
```text
[2024-01-21 13:54:08,025] [INFO] [real_accelerator.py:161:get_accelerator] Setting ds_accelerator to cuda (auto detect)
01/21 13:54:18 - OpenCompass - INFO - Task [opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-7b/ceval-physician]: {'accuracy': 51.02040816326531}
01/21 13:54:18 - OpenCompass - INFO - time elapsed: 6.51s
01/21 13:54:19 - OpenCompass - DEBUG - An `DefaultSummarizer` instance is built from registry, and its implementation can be found in opencompass.summarizers.default
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-7b
----------------------------------------------  ---------  -------------  ------  -------------------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                                     47.37
.... 具体见下表
01/21 13:54:19 - OpenCompass - INFO - write summary to /root/opencompass/opencompass/outputs/default/20240121_132344/summary/summary_20240121_132344.txt
01/21 13:54:19 - OpenCompass - INFO - write csv to /root/opencompass/opencompass/outputs/default/20240121_132344/summary/summary_20240121_132344.csv
```
|dataset                                        | version    | metric         | mode    |   <font color=darkred>**internlm2-chat-7b**</font> | internlm-chat-7b |
|---------------------------------------------- | ---------  | -------------  | ------  | ------------------  | ------------------  |
|ceval-computer_network                         | db9ce2     | accuracy       | gen     |     <font color=darkred>**47.37 ↑**</font>| 31.58
|ceval-operating_system                         | 1c2571     | accuracy       | gen     |     <font color=darkred>**57.89 ↑**</font>| 36.84
|ceval-computer_architecture                    | a74dad     | accuracy       | gen     |     <font color=darkred>**42.86 ↑**</font>| 28.57
|ceval-college_programming                      | 4ca32a     | accuracy       | gen     |     <font color=darkred>**51.35 ↑**</font>| 32.43
|ceval-college_physics                          | 963fa8     | accuracy       | gen     |     <font color=darkred>**36.84 ↑**</font>| 26.32
|ceval-college_chemistry                        | e78857     | accuracy       | gen     |     <font color=darkred>**33.33 ↑**</font>| 16.67
|ceval-advanced_mathematics                     | ce03e2     | accuracy       | gen     |      15.79 | <font color=darkred>**21.05 ↑**</font>
|ceval-probability_and_statistics               | 65e812     | accuracy       | gen     |      27.78 | <font color=darkred>**38.89 ↑**</font>
|ceval-discrete_mathematics                     | e894ae     | accuracy       | gen     |     <font color=darkred>**18.75 ↑**</font>| 18.75
|ceval-electrical_engineer                      | ae42b9     | accuracy       | gen     |     <font color=darkred>**40.54 ↑**</font>| 35.14
|ceval-metrology_engineer                       | ee34ea     | accuracy       | gen     |     <font color=darkred>**58.33 ↑**</font>| 50
|ceval-high_school_mathematics                  | 1dc5bf     | accuracy       | gen     |     <font color=darkred>**44.44 ↑**</font>| 22.22
|ceval-high_school_physics                      | adf25f     | accuracy       | gen     |     <font color=darkred>**47.37 ↑**</font>| 31.58
|ceval-high_school_chemistry                    | 2ed27f     | accuracy       | gen     |     <font color=darkred>**52.63 ↑**</font>| 15.79
|ceval-high_school_biology                      | 8e2b9a     | accuracy       | gen     |     26.32| <font color=darkred>**36.84 ↑**</font>
|ceval-middle_school_mathematics                | bee8d5     | accuracy       | gen     |     <font color=darkred>**26.32 ↑**</font>| 26.32
|ceval-middle_school_biology                    | 86817c     | accuracy       | gen     |     <font color=darkred>**66.67 ↑**</font>| 61.9
|ceval-middle_school_physics                    | 8accf6     | accuracy       | gen     |     57.89| <font color=darkred>**63.16 ↑**</font>
|ceval-middle_school_chemistry                  | 167a15     | accuracy       | gen     |     <font color=darkred>**95 ↑**</font>| 60
|ceval-veterinary_medicine                      | b4e08d     | accuracy       | gen     |     39.13| <font color=darkred>**47.83 ↑**</font>
|ceval-college_economics                        | f3f4e6     | accuracy       | gen     |     <font color=darkred>**47.27 ↑**</font>| 41.82
|ceval-business_administration                  | c1614e     | accuracy       | gen     |     <font color=darkred>**51.52 ↑**</font>| 33.33
|ceval-marxism                                  | cf874c     | accuracy       | gen     |     <font color=darkred>**84.21 ↑**</font>| 68.42
|ceval-mao_zedong_thought                       | 51c7a4     | accuracy       | gen     |     <font color=darkred>**70.83 ↑**</font>| 70.83
|ceval-education_science                        | 591fee     | accuracy       | gen     |     <font color=darkred>**72.41 ↑**</font>| 58.62
|ceval-teacher_qualification                    | 4e4ced     | accuracy       | gen     |     <font color=darkred>**79.55 ↑**</font>| 70.45
|ceval-high_school_politics                     | 5c0de2     | accuracy       | gen     |     21.05| <font color=darkred>**26.32 ↑**</font>
|ceval-high_school_geography                    | 865461     | accuracy       | gen     |     <font color=darkred>**47.37 ↑**</font>| 47.37
|ceval-middle_school_politics                   | 5be3e7     | accuracy       | gen     |     42.86| <font color=darkred>**52.38 ↑**</font>
|ceval-middle_school_geography                  | 8a63be     | accuracy       | gen     |     <font color=darkred>**58.33 ↑**</font>| 58.33
|ceval-modern_chinese_history                   | fc01af     | accuracy       | gen     |     65.22| <font color=darkred>**73.91 ↑**</font>
|ceval-ideological_and_moral_cultivation        | a2aa4a     | accuracy       | gen     |     <font color=darkred>**89.47 ↑**</font>| 63.16
|ceval-logic                                    | f5b022     | accuracy       | gen     |     <font color=darkred>**54.55 ↑**</font>| 31.82
|ceval-law                                      | a110a1     | accuracy       | gen     |     <font color=darkred>**41.67 ↑**</font>| 25
|ceval-chinese_language_and_literature          | 0f8b68     | accuracy       | gen     |     <font color=darkred>**56.52 ↑**</font>| 30.43
|ceval-art_studies                              | 2a1300     | accuracy       | gen     |     <font color=darkred>**69.7 ↑**</font>| 60.61
|ceval-professional_tour_guide                  | 4e673e     | accuracy       | gen     |     <font color=darkred>**86.21 ↑**</font>| 62.07
|ceval-legal_professional                       | ce8787     | accuracy       | gen     |     <font color=darkred>**43.48 ↑**</font>| 39.13
|ceval-high_school_chinese                      | 315705     | accuracy       | gen     |     <font color=darkred>**68.42 ↑**</font>| 63.16
|ceval-high_school_history                      | 7eb30a     | accuracy       | gen     |     <font color=darkred>**75 ↑**</font>| 70
|ceval-middle_school_history                    | 48ab4a     | accuracy       | gen     |     <font color=darkred>**68.18 ↑**</font>| 59.09
|ceval-civil_servant                            | 87d061     | accuracy       | gen     |     <font color=darkred>**55.32 ↑**</font>| 53.19
|ceval-sports_science                           | 70f27b     | accuracy       | gen     |     <font color=darkred>**73.68 ↑**</font>| 52.63
|ceval-plant_protection                         | 8941f9     | accuracy       | gen     |     <font color=darkred>**77.27 ↑**</font>| 59.09
|ceval-basic_medicine                           | c409d6     | accuracy       | gen     |     <font color=darkred>**63.16 ↑**</font>| 47.37
|ceval-clinical_medicine                        | 49e82d     | accuracy       | gen     |     <font color=darkred>**45.45 ↑**</font>| 40.91
|ceval-urban_and_rural_planner                  | 95b885     | accuracy       | gen     |     <font color=darkred>**58.7 ↑**</font>| 45.65
|ceval-accountant                               | 002837     | accuracy       | gen     |     <font color=darkred>**44.9 ↑**</font>| 26.53
|ceval-fire_engineer                            | bc23f5     | accuracy       | gen     |     <font color=darkred>**38.71 ↑**</font>| 22.58
|ceval-environmental_impact_assessment_engineer | c64e2d     | accuracy       | gen     |     45.16 | <font color=darkred>**64.52 ↑**</font>
|ceval-tax_accountant                           | 3a5e3c     | accuracy       | gen     |     <font color=darkred>**51.02 ↑**</font>| 34.69
|ceval-physician                                | 6e277d     | accuracy       | gen     |     <font color=darkred>**51.02 ↑**</font>| 40.82
|ceval-stem                                     | -          | naive_average  | gen     |     <font color=darkred>**44.33 ↑**</font>| 35.09
|ceval-social-science                           | -          | naive_average  | gen     |     <font color=darkred>**57.54 ↑**</font>| 52.79
|ceval-humanities                               | -          | naive_average  | gen     |     <font color=darkred>**65.31 ↑**</font>| 52.58
|ceval-other                                    | -          | naive_average  | gen     |     <font color=darkred>**54.94 ↑**</font>| 44.36
|ceval-hard                                     | -          | naive_average  | gen     |     <font color=darkred>**34.62 ↑**</font>| 23.91
|ceval                                          | -          | naive_average  | gen     |     <font color=darkred>**53.55 ↑**</font>| 44.16




# 2-进阶作业

使用 OpenCompass 评测 InternLM2-Chat-7B 模型使用 LMDeploy 0.2.0 部署后在 C-Eval 数据集上的性能

1. `lmdeploy`安装
```shell
pip install packaging
# 使用 flash_attn 的预编译包解决安装过慢问题
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install 'lmdeploy[all]==0.2.0'

# check
pip list | grep lmdeploy
# lmdeploy                      0.2.0

# 查看支持模型
lmdeploy list | grep internlm
spt="
internlm
internlm-20b
internlm-chat
internlm-chat-20b
internlm-chat-7b
internlm-chat-7b-8k
internlm2-20b
internlm2-7b
internlm2-chat-20b
internlm2-chat-7b
"
```
2. 模型转换 -> `lmdeploy TurboMind`
```shell
model_path=/root/opencompass/InternLM/Shanghai_AI_Laboratory/internlm2-chat-7b
dst_path=/root/opencompass/internlm2_chat_7b_workspace

# 采用离线转换
lmdeploy convert internlm2-chat-7b  ${model_path} --dst-path ${dst_path}
```
3. 配置模型 
   - 修改配置 `vi /root/opencompass/internlm2_chat_7b_workspace/triton_models/weights/config.ini`
```shell
--max-seq-len 2048 --> session_len = 2048
--max-out-len 16 \
--batch-size 4   --> max_batch_size = 4
``` 
   - 配置模型: 在`/root/opencompass/opencompass/configs/`下增加py文件
```python
# /root/opencompass/opencompass/configs/eval_internlm2_chat_7b_turbomind.py
from opencompass.models.turbomind import TurboMindModel
from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel

with read_base():
    # ceval_gen
    from .ceval_gen_5f30c7 import ceval_datasets 
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = [*ceval_datasets]


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)


model_path = '/root/opencompass/internlm2_chat_7b_workspace/triton_models'
models = [
    dict(
        type=TurboMindModel,
        abbr='internlm2-chat-7b-turbomind',
        path=model_path,
        tis_addr='0.0.0.0:33337',
        max_out_len=16,
        max_seq_len=2048,
        batch_size=4,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
```
1. C-Eval评测 
   - 开启服务 todo: 服务无法访问的问题
```shell
lmdeploy serve triton_client "localhost:33337"

./internlm2_chat_7b_workspace \
--model-name internlm2-chat-7b \
--backend turbomind \
--server-name 0.0.0.0 --server-port 33337 \
--max-batch-size 4 \
--tp 1
```
   - 开始评测`nohup sh eval_new.sh > __eval_new.log &`
```shell
# eval_new.sh
cd /root/opencompass/opencompass
python run.py configs/eval_internlm2_chat_7b_turbomind.py --debug
```

