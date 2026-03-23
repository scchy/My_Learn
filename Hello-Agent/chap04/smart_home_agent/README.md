# 智能家居Agent - Smart Home Agent

基于 ReAct + LLM Function Calling + 三层反射机制的智能家居控制系统。

## 📋 项目概述

本项目实现了一个具备自我学习和优化能力的智能家居Agent，采用三层反射架构：

```
用户输入 → ReAct执行(LLM Function Calling) → 【L1即时调整】 → 执行 → 观察 → 【L2任务反思】 → 响应用户
                                                   ↓
                                             【L3策略优化】(定时触发)
```

## ✨ 核心特性

1. **LLM驱动的意图理解**: 使用OpenAI Function Calling实现智能动作规划
2. **L1即时调整**: 基于习惯库和短期记忆实时调整动作
3. **L2任务反思**: 任务结束后异步分析执行情况，提取新规则
4. **L3策略优化**: 定期批量分析行为模式，优化长期策略
5. **Fallback机制**: LLM不可用时自动切换本地规则匹配

## 🏗️ 架构设计

### 三层反射机制

| 层级 | 触发时机 | 核心功能 | 延迟 |
|------|----------|----------|------|
| **L1 即时调整** | 每个Action执行前 | 查询习惯库和短期记忆，实时调整动作 | < 50ms |
| **L2 任务反思** | 任务结束后异步 | 评估执行质量，分析失败原因，生成改进规则 | 异步 |
| **L3 策略优化** | 每周/每月定时 | 批量分析长期行为，挖掘模式，修正控制策略 | 离线 |

### 核心组件

```
SmartHomeAgent
├── LLM Client              # OpenAI/兼容API客户端
├── Function Tools          # 设备控制函数定义
├── HabitDatabase          # 习惯数据库存储长期规则
├── ShortTermMemory        # 短期记忆(TTL过期机制)
├── DeviceSimulator        # 设备执行模拟器
└── ReflectionEngine       # 反射引擎(L2/L3)
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- asyncio
- openai (可选，用于LLM模式)

```bash
pip install openai
```

### 运行演示

```bash
cd Hello-Agent/chap04/smart_home_agent
python agent.py
```

### 基础使用

```python
import asyncio
from agent import SmartHomeAgent

async def main():
    # 方式1: 仅使用本地规则(无需API Key)
    agent = SmartHomeAgent(user_id="user_001", use_local_only=True)
    
    # 方式2: 使用OpenAI API
    agent = SmartHomeAgent(
        user_id="user_001",
        api_key="your-openai-api-key",
        llm_model="gpt-3.5-turbo"
    )
    
    # 方式3: 使用兼容API (如Qwen、GLM等)
    agent = SmartHomeAgent(
        user_id="user_001",
        api_key="your-api-key",
        base_url="https://api.example.com/v1",
        llm_model="qwen-turbo"
    )
    
    # 处理用户请求
    response = await agent.handle_request("我要睡觉了")
    print(response)  # 输出: 已完成所有操作(3个设备)，祝您生活愉快！
    
    # 执行周度维护(触发L3)
    agent.weekly_maintenance()
    
    # 查看统计
    stats = agent.get_statistics()
    print(f"执行次数: {stats['total_executions']}")

asyncio.run(main())
```

## 🤖 LLM Function Calling

### Function Tool定义

```python
tools = [{
    "type": "function",
    "function": {
        "name": "control_device",
        "description": "控制智能家居设备",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "enum": ["ac", "light", "tv", "curtain", "music"],
                    "description": "设备类型"
                },
                "operation": {
                    "type": "string",
                    "description": "操作类型"
                },
                "params": {
                    "type": "object",
                    "description": "操作参数"
                }
            },
            "required": ["device", "operation", "params"]
        }
    }
}]
```

### 调用流程

```
用户输入: "我要睡觉了"
    ↓
LLM理解意图 → 调用 control_device(ac, set, {temp:24, mode:"sleep"})
           → 调用 control_device(light, turn_off, {on:false})
           → 调用 control_device(curtain, close, {open:false})
    ↓
L1层调整 → 根据习惯优化参数 → 执行
```

## 📖 核心概念

### 1. 意图识别与动作规划

Agent通过两种方式进行动作规划：

**LLM方式** (推荐):
```python
# LLM自动理解复杂意图
await agent.handle_request("我有点热，想睡觉")
# LLM规划:
# 1. 空调: 23度睡眠模式 (理解"热"+"睡觉")
# 2. 灯光: 关闭
# 3. 窗帘: 关闭
```

**本地规则方式** (Fallback):
```python
# 基于关键词匹配
if any(kw in intent for kw in ["睡觉", "sleep"]):
    actions = [空调26度睡眠模式, 关灯, 关窗帘]
```

### 2. 习惯规则

习惯规则定义用户的行为模式，支持自动学习：

```python
HabitRule(
    rule_id="rule_001",
    trigger={
        "intent_keywords": ["睡觉", "睡眠"],
        "time_range": "21-24",
        "season": ["spring", "summer"],
        "device": "ac"
    },
    action={"device": "ac", "temp": 24, "mode": "sleep"},
    confidence=0.85,
    reasoning="用户习惯夜间睡眠模式温度设为24度"
)
```

### 3. 反射学习

#### L1 - 即时调整示例

```
用户输入: "我要睡觉了"
计划动作: 空调设置26度 (LLM规划)
L1匹配规则: "夜间睡眠偏好24度(置信度0.85)"
调整后: 空调设置24度
执行结果: 成功
```

#### L2 - 任务反思示例

```
任务执行后分析:
- 用户满意度: False (有覆盖行为)
- 分析: 用户连续3次将温度从26调至24度
- 生成规则: 新增习惯规则，默认24度
- 初始置信度: 0.6
```

#### L3 - 策略优化示例

```
周度分析:
- 分析7天日志数据
- 发现模式: "工作日22点睡觉, 周末23点"
- 策略提案: 更新默认睡眠时间
- 高置信度规则: 自动应用
- 中置信度规则: 建议用户确认
```

## 📁 文件结构

```
smart_home_agent/
├── agent.py          # 主代码文件
├── README.md         # 本文档
└── [暂无其他文件]
```

## 🔧 API参考

### SmartHomeAgent

#### `__init__(user_id, llm_client, llm_model, use_local_only, api_key, base_url)`

初始化Agent实例

**参数:**
- `user_id`: 用户ID
- `llm_client`: 可选，自定义LLM客户端
- `llm_model`: LLM模型名称，默认 "gpt-3.5-turbo"
- `use_local_only`: 仅使用本地规则，默认 False
- `api_key`: OpenAI API Key
- `base_url`: 自定义API基础URL

**示例:**
```python
# 纯本地模式
agent = SmartHomeAgent(use_local_only=True)

# OpenAI模式
agent = SmartHomeAgent(api_key="sk-xxx")

# 兼容API模式
agent = SmartHomeAgent(
    api_key="xxx",
    base_url="https://api.qwen.ai/v1",
    llm_model="qwen-turbo"
)
```

#### `async handle_request(user_input: str) -> str`

处理用户请求，返回响应消息

#### `weekly_maintenance()`

执行周度维护，触发L3策略优化

#### `get_statistics() -> Dict`

获取运行统计数据

### 数据模型

#### `Action`
```python
@dataclass
class Action:
    device: str           # 设备名称: ac/light/tv/curtain/music
    operation: str        # 操作类型: turn_on/turn_off/set/dim/open/close/play
    params: Dict          # 操作参数
```

#### `Context`
```python
@dataclass
class Context:
    hour: int             # 当前小时(0-23)
    season: str           # 季节: spring/summer/autumn/winter
    weather: str          # 天气
    user_location: str    # 用户位置
    temperature: float    # 环境温度
```

## 📝 支持的场景

| 关键词 | 场景 | 执行动作 |
|--------|------|----------|
| 睡觉/睡眠/sleep | 睡眠模式 | 空调24度睡眠模式 + 关灯 + 关窗帘 |
| 看电影/movie | 观影模式 | 开电视 + 灯光30%暖色 |
| 起床/wake | 起床模式 | 开窗帘50% + 灯光80% + 空调24度 |
| 音乐/music | 音乐模式 | 播放音乐(音量40%) |

## 🔬 演示输出示例

```
============================================================
智能家居Agent演示
============================================================

>>> 演示1: 睡眠场景
============================================================
[Agent] 收到请求: '我要睡觉了'
============================================================
[ReAct] 规划了 3 个动作

[L1] 处理意图: '我要睡觉了', 计划动作: ac.set(temp=26, mode=sleep, on=True)
[L1] 应用习惯规则: rule_001
[Device] ac: set -> {'temp': 24, 'mode': 'sleep', 'on': True}

[L1] 处理意图: '我要睡觉了', 计划动作: light.turn_off(on=False)
[L1] 无匹配规则，使用默认
[Device] light: turn_off -> {'on': False}

[Agent] 响应: 已完成所有操作(3个设备)，祝您生活愉快！

==================================================
[L2 任务反思] 开始分析...
[L2] 评估结果: 满意度=True, 覆盖次数=0
==================================================

==================================================
[L3 策略优化] 分析时间窗口: 7d...
[L3] 发现模式: 空调温度偏好_hour_22, 置信度=0.85
[L3] 自动应用策略: 空调温度偏好_hour_22
==================================================
```

## 🎯 扩展开发

### 添加新的设备类型

```python
class DeviceType(Enum):
    # 现有类型...
    FAN = "fan"  # 新增风扇
    HUMIDIFIER = "humidifier"  # 新增加湿器
```

然后在Function Tool定义中添加新设备：

```python
"device": {
    "enum": ["ac", "light", "tv", "curtain", "music", "fan", "humidifier"],
    ...
}
```

### 自定义LLM客户端

```python
from openai import OpenAI

# 自定义客户端配置
client = OpenAI(
    api_key="your-key",
    base_url="https://custom-api.com/v1",
    timeout=30,
    max_retries=3
)

agent = SmartHomeAgent(user_id="user_001", llm_client=client)
```

### 集成其他LLM框架

```python
class CustomLLMClient:
    def __init__(self, model_name):
        self.model = load_model(model_name)
    
    def chat_completion_create(self, model, messages, tools, tool_choice):
        # 实现自定义调用逻辑
        response = self.model.generate(messages, tools)
        return response

# 使用自定义客户端
agent = SmartHomeAgent(
    user_id="user_001",
    llm_client=CustomLLMClient("custom-model")
)
```

## 🔒 环境变量配置

```bash
# .env 文件示例
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
LLM_MODEL=gpt-3.5-turbo
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

agent = SmartHomeAgent(
    user_id="user_001",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo")
)
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📄 许可

MIT License
