reference Github: [https://github.com/InternLM/tutorial/blob/main/helloworld/hello_world.md](https://github.com/InternLM/tutorial/blob/main/helloworld/hello_world.md)


# 1- 环境配置

之前都是用科学上网在`huggingFace`进行的模型下载，同时还需要进行一些配置

```python
import os
os.environ['CURL_CA_BUNDLE'] = ''
```

在本次的学习中发现可以[**设置镜像**](https://hf-mirror.com/)或者是通过`modelscope`(`pip install modelscope`)进行下载

```python
# huggingface-下载
## 安装依赖: pip install -U huggingface_hub
## 直接设置环境变量: export HF_ENDPOINT=https://hf-mirror.com
import os 
from huggingface_hub import hf_hub_download 
hf_hub_download(repo_id="internlm/internlm-7b", filename="config.json")


# modelscope 下载
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/root/model', revision='v1.0.3')

```
下载完之后可以直接用`transformers` 无缝衔接
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

download_dir = "xxx/internlm-chat-7b"

model = (
        AutoModelForCausalLM.from_pretrained(download_dir, trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
tokenizer = AutoTokenizer.from_pretrained(download_dir, trust_remote_code=True)
```

同时补齐了信息差，知道了[国内镜像合集网址:MirrorZ Help](https://help.mirrors.cernet.edu.cn/)

# 2- InterLM-Chat-7B Demo尝试

> 主要的项目GitHub: [https://github.com/InternLM/InternLM](https://github.com/InternLM/InternLM)

## 2.1 终端demo
其实进行demo尝试相对比较简单，主要是模型下载和GPU显存(`20G : 1/4的A100-80G`)的限制比较大。只用`transformers.AutoModelForCausalLM`加载进行尝试就行。



```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```

## 2.2 web demo

进行web demo，主要是对终端demo进行一层`streamlit`的封装，同时通过ssh将端口映射到本地，资源占用的时服务器的资源。
从教程中学习到了[`@st.cache_resource`](https://docs.streamlit.io/library/api-reference/performance/st.cache_resource)装饰器的用法，这个在笔者之前的`streamlit`项目中没有用到过, 后续可以在自己项目中尝试用在保持`database`的连接上。

```python
@st.cache_resource
def load_model():
    model = (
        AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
    return model, tokenizer
```


# 3- Lagent 智能体工具调用 Demo尝试

> 主要的项目GitHub: [https://github.com/InternLM/lagent](https://github.com/InternLM/lagent)

在第一节中已经了解到：**大模型的局限性，需要Agent去引导优化**, 这次demo尝试加深了对这个句话的理解。

在本次Demo中调用`lagent`，去解决数学问题： 已知 2x+3=10，求x ,此时 `InternLM-Chat-7B` 模型理解题意生成解此题的 Python 代码，Lagent 调度送入 Python 代码解释器求出该问题的解。

主要步骤如下：
1. 模型初始化`init_model`(基于选择的name)
    - `model = HFTransformerCasualLM('/root/model/Shanghai_AI_Laboratory/internlm-chat-7b')`
2. 构建lagent `initialize_chatbot`
    - `chatbot = ReAct(llm=model, action_executor=ActionExecutor(actions=PythonInterpreter()))`
3. 用户输入调用`chatbot`
    - `agent_return = chatbot.chat(user_input)`
4. 解析返回结果并展示（最后保存历史信息）
    -  `render_assistant(agent_return)`
    -  action解析展示如下
```python
    def render_assistant(self, agent_return):
        with st.chat_message('assistant'):
            for action in agent_return.actions:
                if (action):
                    self.render_action(action)
            st.markdown(agent_return.response)
            
    def render_action(self, action):
        with st.expander(action.type, expanded=True):
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>插    件</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.type + '</span></p>',
                unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>思考步骤</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.thought + '</span></p>',
                unsafe_allow_html=True)
            if (isinstance(action.args, dict) and 'text' in action.args):
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行内容</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                    unsafe_allow_html=True)
                st.markdown(action.args['text'])
            self.render_action_results(action)
            
    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if (isinstance(action.result, dict)):
            st.markdown(
                "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行结果</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                unsafe_allow_html=True)
            if 'text' in action.result:
                st.markdown(
                    "<p style='text-align: left;'>" + action.result['text'] +
                    '</p>',
                    unsafe_allow_html=True)
            if 'image' in action.result:
                image_path = action.result['image']
                image_data = open(image_path, 'rb').read()
                st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)
```

简单的代码可以如下
```python
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
from lagent.agents.react import ReAct
from lagent.llms import GPTAPI
from lagent.llms.huggingface import HFTransformerCasualLM


# init_model
model = HFTransformerCasualLM('/root/model/Shanghai_AI_Laboratory/internlm-chat-7b')

# initialize_chatbot
chatbot = ReAct(llm=model, action_executor=ActionExecutor(actions=PythonInterpreter()))
agent_return = chatbot.chat(user_input)
```

# 4- 浦语·灵笔图文理解创作 Demo尝试

> 主要的项目GitHub: [https://github.com/InternLM/InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)
> 这里的模型也是不一样: `InternLM-XComposer`是基于`InternLM`研发的视觉-语言大模型

`InternLM-XComposer`是提供出色的图文理解和创作能力，具有多项优势：
- 图文交错创作: `InternLM-XComposer`可以为用户打造图文并貌的专属文章。这一能力由以下步骤实现：
    1. 理解用户指令，创作符合要求的长文章。
    2. 智能分析文章，自动规划插图的理想位置，确定图像内容需求。
    3. 多层次智能筛选，从图库中锁定最完美的图片。
- 基于丰富多模态知识的图文理解:    `InternLM-XComposer`设计了高效的训练策略，为模型注入海量的多模态概念和知识数据，赋予其强大的图文理解和对话能力。

- 杰出性能: `InternLM-XComposer`在多项视觉语言大模型的主流评测上均取得了最佳性能，包括MME Benchmark (英文评测), MMBench (英文评测), Seed-Bench (英文评测), CCBench(中文评测), MMBench-CN (中文评测).

## 4.1 生成文章
1. 模型和token初始化
2. 模型调用
    - 快速使用的话，可以直接`llm_model.generate` 
    - 进行复杂的一些操作可以 `llm_model.internlm_model.generate`
        - 细节看笔者简单修改的`generate`函数

核心示例code: 
```python
import torch
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModel
from examples.utils import auto_configure_device_map


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[:, -len(stop):])).item():
                return True
        return False


folder = 'internlm/internlm-xcomposer-7b'
device = 'cuda'

# 1- init model and tokenizer
llm_model = AutoModel.from_pretrained(folder, trust_remote_code=True).cuda().eval()
if args.num_gpus > 1:
    from accelerate import dispatch_model
    device_map = auto_configure_device_map(args.num_gpus)
    model = dispatch_model(model, device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
llm_model.internlm_tokenizer = tokenizer
llm_model.tokenizer = tokenizer


# 2 封装generate
def generate(llm_model, text, random, beam, max_length, repetition, use_inputs=False):
    """生成文章封装
    llm_model:  AutoModel.from_pretrained 加载的 internlm/internlm-xcomposer-7b
    random:  采样
    beam: beam search 数量
    max_length: 文章最大长度
    repetition: repetition_penalty
    """
    device = 'cuda'
    # stop critria
    stop_words_ids = [
        torch.tensor([103027]).to(device),  
        torch.tensor([103028]).to(device),  
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)])
    # 输入tokens
    input_tokens = llm_model.internlm_tokenizer(
        text, return_tensors="pt",
        add_special_tokens=True).to(llm_model.device)
    # 输入生成图像的embeds
    img_embeds = llm_model.internlm_model.model.embed_tokens(
        input_tokens.input_ids)
    inputs = input_tokens.input_ids if use_inputs else None
    # 模型推理
    with torch.no_grad():
        with llm_model.maybe_autocast():
            outputs = llm_model.internlm_model.generate(
                inputs=inputs,
                inputs_embeds=img_embeds,  #  生成配图
                stopping_criteria=stopping_criteria,
                do_sample=random,
                num_beams=beam,
                max_length=max_length,
                repetition_penalty=float(repetition),
            )
    # decode及输出
    output_text = llm_model.internlm_tokenizer.decode(
        outputs[0][1:], add_special_tokens=False)
    output_text = output_text.split('<TOKENS_UNUSED_1>')[0]
    return output_text



# 调用
## 生成小文章
text = '请介绍下爱因斯坦的生平'
- 直接调用: 
response = llm_model.generate(text)
print(f'User: {text}')
print(f'Bot: {response}')
- 封装调用:
generate(llm_model, text,
        random=False,
        beam=3,
        max_length=300,
        repetition=5.,
        use_inputs=True
)

```
## 4.2 多模态对话

主要用 gradio 搭建web(可以阅读【[知乎 Gradio：轻松实现AI算法可视化部署](https://zhuanlan.zhihu.com/p/374238080)】)

1. 模型和token初始化
2. 模型调用
    - 快速使用的话，可以直接`llm_model.chat(text=text, image=image, history=None)` 
    - 进行复杂的一些操作可以 `llm_model.internlm_model.generate`
        - 这部分embeding比较复杂(笔者梳理的`chat_answer`示意了主要流程，但还是存在bug)

```python
import torch
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModel
from examples.utils import auto_configure_device_map


# 模型初始化同上
state = CONV_VISION_7132_v2.copy()
def chat_answer(llm_model, state, text, image):
    """
    state: 起到history的作用
    text: 输入的提问内容
    image: 图片
    """
    # image 需要读取
    # image = gr.State()

    device = 'cuda'
    # stop critria
    stop_words_ids = [
        torch.tensor([103027]).to(device),  
        torch.tensor([103028]).to(device),  
    ]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    # 输入处理
    img_list = []
    state.append_message(state.roles[0], text)
    with torch.no_grad():
        image_pt = llm_model.vis_processor(image).unsqueeze(0).to(0)
        image_emb = llm_model.encode_img(image_pt)
    img_list.append(image_emb)

    # 生成内容的embedding
    prompt = state.get_prompt()
    prompt_segs = prompt.split('<Img><ImageHere></Img>')
    seg_tokens = [
        llm_model.internlm_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    seg_embs = [
        llm_model.internlm_model.model.embed_tokens(seg_t) for seg_t in seg_tokens
    ]
    mixed_embs = [
        emb for pair in zip(seg_embs[:-1], img_list) for emb in pair
    ] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    embs = mixed_embs

    # 模型推理
    outputs = llm_model.internlm_model.generate(
        inputs_embeds=embs,
        max_new_tokens=300,
        stopping_criteria=stopping_criteria,
        num_beams=3,
        #temperature=float(temperature),
        do_sample=False,
        repetition_penalty=float(0.5),
        bos_token_id=llm_model.internlm_tokenizer.bos_token_id,
        eos_token_id=llm_model.internlm_tokenizer.eos_token_id,
        pad_token_id=llm_model.internlm_tokenizer.pad_token_id,
    )
    # decode输出
    output_token = outputs[0]
    if output_token[0] == 0:
        output_token = output_token[1:]
    output_text = llm_model.internlm_tokenizer.decode(
        output_token, add_special_tokens=False)
    print(output_text)
    output_text = output_text.split('<TOKENS_UNUSED_1>')[
        0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    output_text = output_text.replace("<s>", "")
    return output_text
# 图文对话
## 1st return
image = 'examples/images/aiyinsitan.jpg'
text = '图片里面的是谁？'
- 直接调用: 
response, history = llm_model.chat(text=text, image=image, history=None)
print(f'User: {text}')
print(f'Bot: {response}')
- 封装调用:
output_text = chat_answer(llm_model, state, text, image)


## 2nd turn
text = '他有哪些成就?'
- 直接调用: 
response, history = llm_model.chat(text=text, image=None, history=history)
print(f'User: {text}')
print(f'Bot: {response}')
- 封装调用:
output_text = chat_answer(llm_model, state, text, image)
```