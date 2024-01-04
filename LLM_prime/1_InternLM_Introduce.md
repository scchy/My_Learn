reference video: [bilibili-书生·浦语大模型全链路开源体系](https://www.bilibili.com/video/BV1Rc411b7ns/)
reference Github: [https://github.com/InternLM/tutorial](https://github.com/InternLM/tutorial)

# 一、从模型到应用

```mermaid
graph LR

S1(模型选型-评测) --> SA{业务场景是否复杂}
SA -->|是|SA2{算力足够吗}
SA2 -->|是|SA2Y(继续训练/全参数微调)
SA2 -->|否|SA2N(部分参数微调)
SA2N -->SA3{是否需要环境交互}
SA2Y -->SA3
SA3 -->|是|SA3Y(构建智能体)

SA3 -->|否|M
SA3Y --> M
M --> F(模型部署)

SA -->|否|M(模型评测)
```

# 二、预训练`InternLM-Train`
并行训练
- 高可拓展 8~千卡，千卡加速 92%
- 极致优化`Hybrid Zero` 3600 tokens/sec/gpu, 加速50%
- 兼容主流： 无缝接入Huggingface
- 开箱即用：修改配置即可训练


# 三、微调`XTuner`

- 增量续训练
  - 使用场景：让base模型学习新的垂直领域知识
  - 训练数据：文章、书籍、代码等
- 有监督微调 `SFT`
  - 让模型学会理解和遵循各种指令
  - 训练数据：高质量的对话、问答数据
- `XTuner`
  - 优化QLoRA可以在8G显存的GPU上微调 InternLM-7B模型

# 四、评测`OpenCompass`
> 6大维度，80+评测集，40万+评测题目  
> 唯一国内开发者主要开发的大模型评测工具  
> Meta推荐的评测工具之一

- 学科
- 语音
- 知识
- 理解
- 推理
- 安全

# 五、部署`LMDeploy`

- 技术挑战
  - 低存储设备如何部署
  - 推理：如何加速；如何解决动态shape；如何有效管理和利用内存
  - 服务：提升系统整体吞吐量；降低请求的平均响应时间
- 部署方案
  - 模型并行
  - 低比特量化
  - Attention优化
  - 计算和访问优化
  - Continuous Batching

# 六、应用——智能体`Lagent`,`AgentLego`

大模型的局限性，需要Agent去引导优化

```mermaid
graph LR

LLM1(PythonExecutor) --> LLMA
LLM2(Search) --> LLMA
LLM3(Calender) --> LLMA
LLM4(More..) --> LLMA

LLMA(Action Executor) --> LLMO(LLM) --> PA(Planning & Action)
PA -->|Act|LLMA

PA --> PA1(Plan-Act Iteration)
PA --> PA2(Plan-then-Act)
PA --> PA3(Reflexion)
PA --> PA4(Exploration)

HM1(Human Feadback) --> LLMO
HM2(Human Instrcution) --> LLMO
HM3(Observations) --> LLMO
```

`Lagent`支持多种类型的智能体能力、支持多种大语言模型

多模态智能体工具箱`AgentLego`
- 丰富的工具集合
- 支持多个主流智能体系统，`LangChain`,`Transformers Agents`,`Lagent`
- 灵活的多模态工具调用接口
- 一键式远程工具部署
