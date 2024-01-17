# 一、基础作业

> 使用 LMDeploy 以本地对话、网页Gradio、API服务中的一种方式部署 InternLM-Chat-7B 模型，生成 300 字的小故事（需截图）

![deploy_hw1](./pic/deploy_HW1.jpg)


# 二、进阶作业 


## 2.1 自我认知小助手模型量化部署
将第四节课训练自我认知小助手模型使用 LMDeploy 量化部署到 OpenXLab 平台。  
在第四节中已经将Xtun微调的参数转成adapter，并和原来的模型进行合并。
- 因为过拟合所以将造的数据和MedQA2019数据进行混合，再微调-最终解决过拟合问题
  - 产出文件模型文件在：`/root/personal_assistant/hf_merge`
- 进行基本量化流程
  - 统计maxmin
  - 模型量化
  - 模型转换
  - 模型上传到 `modelscope`
- 模型部署
  - 部署github: [LLM_W4A16_myAssistant](https://github.com/scchy/LLM_W4A16_myAssistant)
  - 部署应用openxlab地址 [https://openxlab.org.cn/apps/detail/Scchy/LLM_scc_Assistant](https://openxlab.org.cn/apps/detail/Scchy/LLM_scc_Assistant)

![pic](./pic/deploy_HW_selfassistant.jpg)


## 2.2 量化比对
对internlm-chat-7b <font color=darkred>模型进行量化，并同时使用KV Cache量化</font>，使用量化后的模型完成API服务的部署。
   - 分别对比模型量化前后和 KV Cache 量化前后的显存大小（将 bs设置为 1 和 max len 设置为512）。
```shell
# 0- 离线模型装换
lmdeploy convert internlm-chat-7b  \
    /root/share/temp/model_repos/internlm-chat-7b/ \
    --dst_path ./workspace_7bOrg

# 1. 计算 minmax
lmdeploy lite calibrate \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --calib_dataset "c4" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_maxmin_info

# 2-1 KV Cache 量化
lmdeploy lite kv_qparams \
  --work_dir ./quant_maxmin_info  \
  --turbomind_dir workspace/triton_models/weights/ \
  --kv_sym False \
  --num_tp 1
# 修改配置 quant_policy=4


# 2-2 量化权重模型
lmdeploy lite auto_awq \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_maxmin_info

# 2-2-F  转换模型的layout
lmdeploy convert  internlm-chat-7b ./quant_maxmin_info \
    --model-format awq \
    --group-size 128 \
    --dst_path ./workspace_w4a16


# 2-3 量化模型 + KV Cache
lmdeploy lite kv_qparams \
  --work_dir ./quant_maxmin_info  \
  --turbomind_dir ./workspace_w4a16/triton_models/weights/ \
  --kv_sym False \
  --num_tp 1
```
- [ ] 如何比对
  - [ ] 比对参数设置`xx_workspace/triton_models/weights/config.ini` :`max_batch_size = 1` 和 `session_len = 512` 和 `quant_policy=4 & 0`
  - [ ] 问题：你是谁
- [ ] 如何启动
  - [X] 启动原始的模型（离线转换后）`lmdeploy serve gradio ./workspace_7bOrg`
  - [X] kv量化后的模型可以直接在 workspace 启动 `lmdeploy serve gradio ./workspace`
  - [X] 量化后的模型可以直接在 对应workspace 启动 `lmdeploy serve gradio ./workspace_w4a16`
  - [X] 量化后的模型+KV可以直接在 对应workspace 启动 `lmdeploy serve gradio ./workspace_w4a16`


| 模型类型 | 配置 | 操作截图 |  模型大小&变化 |
|-|-|-|-|
| internlm-chat-7b | `max_batch_size = 1` 和 `session_len = 512` 和 `quant_policy=0` | ![base](./pic/deploy_HW2_base.jpg) | 14654 MiB -> 14726 MiB (72 MiB)|
| internlm-chat-7b + KV Cache| `max_batch_size = 1` 和 `session_len = 512` 和 `quant_policy=4` |![kv](./pic/deploy_HW2_kv.jpg) | 14622 MiB -> 14630iB (8 MiB)|
| internlm-chat-7b-Qunt| `max_batch_size = 1` 和 `session_len = 512` 和 `quant_policy=0` |![q](./pic/deploy_HW2_q.jpg) | 5692 MiB -> 5760 MiB (68 MiB)| 
| internlm-chat-7b-Qunt + KV Cache | `max_batch_size = 1` 和 `session_len = 512` 和 `quant_policy=4` |![q](./pic/deploy_HW2_q_kv.jpg) | 5660 MiB -> 5696 MiB (36 MiB)   **因为推理用FP16所以会额外多占一些显存**| 



## 2.3 在任务数据集上进行测试比对

在自己的任务数据集上任取若干条进行Benchmark测试，测试方向包括：
1. TurboMind推理+Python代码集成
2. 在（1）的基础上采用W4A16量化
3. 在（1）的基础上开启KV Cache量化
4. 在（2）的基础上开启KV Cache量化
5. 使用Huggingface推理



