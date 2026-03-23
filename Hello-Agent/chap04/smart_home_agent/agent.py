"""
智能家居Agent - 基于ReAct + 三层反射机制的智能家居控制系统

架构:
  用户输入 → ReAct执行 → 【L1即时调整】 → 执行 → 观察 → 【L2任务反思】 → 响应用户
                                    ↓
                              【L3策略优化】(定时触发)

核心特性:
1. LLM驱动的意图理解和动作规划 (Function Calling)
2. L1即时调整: 基于习惯和短期记忆实时调整动作
3. L2任务反思: 任务结束后异步分析执行情况
4. L3策略优化: 定期批量分析优化长期策略
"""
import sys
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union


# ==================== 数据模型 ====================

class DeviceType(Enum):
    """设备类型枚举"""
    AC = "ac"           # 空调
    LIGHT = "light"     # 灯光
    TV = "tv"           # 电视
    CURTAIN = "curtain" # 窗帘
    MUSIC = "music"     # 音乐


@dataclass
class Context:
    """环境上下文"""
    hour: int
    season: str
    weather: str
    user_location: str
    weekday: int = field(default_factory=lambda: datetime.now().weekday())
    temperature: float = 25.0
    
    def to_dict(self) -> Dict:
        return {
            "hour": self.hour,
            "season": self.season,
            "weather": self.weather,
            "user_location": self.user_location,
            "weekday": self.weekday,
            "temperature": self.temperature
        }


@dataclass
class Action:
    """动作定义"""
    device: str
    operation: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "device": self.device,
            "operation": self.operation,
            "params": self.params
        }
    
    def __str__(self) -> str:
        params_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.device}.{self.operation}({params_str})"


@dataclass
class HabitRule:
    """习惯规则"""
    rule_id: str
    trigger: Dict[str, Any]
    action: Dict[str, Any]
    confidence: float = 0.5
    frequency: int = 0
    last_triggered: Optional[str] = None
    source: str = "system"
    reasoning: str = ""
    user_confirmed: bool = False
    
    def matches(self, intent: str, context: Context) -> bool:
        """检查规则是否匹配当前情境"""
        if "intent_keywords" in self.trigger:
            if not any(kw in intent for kw in self.trigger["intent_keywords"]):
                return False
        
        if "time_range" in self.trigger:
            # 简单的时间范围检查
            time_range = self.trigger["time_range"]
            if isinstance(time_range, str):
                start_h, end_h = map(int, time_range.replace(":", "").split("-"))
                if not (start_h <= context.hour <= end_h):
                    return False
        
        if "hour" in self.trigger and context.hour != self.trigger["hour"]:
            return False
            
        if "season" in self.trigger:
            seasons = self.trigger["season"]
            if isinstance(seasons, list):
                if context.season not in seasons:
                    return False
            elif context.season != seasons:
                return False
        
        if "device" in self.trigger:
            if self.trigger["device"] != self.action.get("device"):
                return False
                
        return True


@dataclass
class LogEntry:
    """执行日志条目"""
    timestamp: str
    context: Context
    planned_action: Action
    adjusted_action: Optional[Action]
    reflection_note: str
    execution_result: str
    user_override: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "context": self.context.to_dict(),
            "planned": self.planned_action.to_dict(),
            "adjusted": self.adjusted_action.to_dict() if self.adjusted_action else None,
            "reflection": self.reflection_note,
            "result": self.execution_result,
            "user_override": self.user_override
        }


# ==================== 核心组件 ====================

class HabitDatabase:
    """习惯数据库 - 存储和管理用户习惯规则"""
    
    def __init__(self):
        self.rules: List[HabitRule] = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认规则"""
        default_rules = [
            HabitRule(
                rule_id="rule_001",
                trigger={
                    "intent_keywords": ["睡觉", "睡眠", "rest", "sleep"],
                    "time_range": "21-24",
                    "season": ["spring", "summer"],
                    "device": "ac"
                },
                action={"device": "ac", "temp": 24, "mode": "sleep"},
                confidence=0.85,
                frequency=12,
                reasoning="用户习惯夜间睡眠模式温度设为24度"
            ),
            HabitRule(
                rule_id="rule_002",
                trigger={
                    "intent_keywords": ["看电影", "观影", "movie"],
                    "time_range": "19-23",
                    "device": "light"
                },
                action={"device": "light", "brightness": 30, "color": "warm"},
                confidence=0.75,
                frequency=8,
                reasoning="观影时偏好低亮度暖色灯光"
            ),
            HabitRule(
                rule_id="rule_003",
                trigger={
                    "intent_keywords": ["起床", "wake"],
                    "hour": 7,
                    "device": "curtain"
                },
                action={"device": "curtain", "operation": "open", "percent": 50},
                confidence=0.9,
                frequency=30,
                reasoning="早晨7点自动打开窗帘50%"
            )
        ]
        self.rules.extend(default_rules)
    
    def query(self, intent: str, context: Context,
              device: Optional[str] = None, confidence_threshold: float = 0.7) -> List[HabitRule]:
        """查询匹配的习惯规则"""
        matching_rules = []
        for rule in self.rules:
            if rule.matches(intent, context) and rule.confidence >= confidence_threshold:
                if device is None or rule.action.get("device") == device:
                    matching_rules.append(rule)
        return matching_rules
    
    def add_rule(self, trigger: Dict, old_default: Dict, new_default: Dict,
                 confidence: float, reasoning: str, source: str = "system"):
        """添加新规则"""
        rule_id = f"rule_{len(self.rules) + 1:03d}"
        new_rule = HabitRule(
            rule_id=rule_id,
            trigger=trigger,
            action=new_default,
            confidence=confidence,
            source=source,
            reasoning=reasoning
        )
        self.rules.append(new_rule)
        print(f"[HabitDB] 新增规则: {rule_id}, 置信度={confidence:.2f}")
        return rule_id
    
    def get_all_rules(self, user_id: Optional[str] = None) -> List[HabitRule]:
        """获取所有规则"""
        return self.rules
    
    def archive_stale_rules(self, threshold_days: int = 30):
        """清理过期规则"""
        # 简化实现: 移除低频规则
        self.rules = [r for r in self.rules if r.frequency > 0 or r.source == "system"]
        print(f"[HabitDB] 已清理过期规则，剩余 {len(self.rules)} 条")
    
    def update_confidence(self, rule_id: str, delta: float):
        """更新规则置信度"""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.confidence = min(1.0, max(0.0, rule.confidence + delta))
                rule.frequency += 1
                rule.last_triggered = datetime.now().isoformat()
                break


class ShortTermMemory:
    """短期记忆 - 存储最近的用户覆盖行为"""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl  # 过期时间(秒)
        self.memory: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
    
    def store(self, key: str, value: Any):
        """存储记忆"""
        self.memory[key] = value
        self.timestamps[key] = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """获取记忆，自动检查过期"""
        if key in self.memory:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.memory[key]
            else:
                # 过期清理
                del self.memory[key]
                del self.timestamps[key]
        return None
    
    def has_recent_override(self, context: Context) -> bool:
        """检查是否有最近的覆盖记录"""
        location_key = f"override_{context.user_location}"
        return self.get(location_key) is not None
    
    def record_override(self, context: Context, action: Action, new_params: Dict):
        """记录用户覆盖行为"""
        key = f"override_{context.user_location}_{action.device}"
        self.store(key, {
            "context": context,
            "action": action,
            "new_params": new_params,
            "timestamp": datetime.now().isoformat()
        })


class DeviceSimulator:
    """设备模拟器 - 模拟设备执行"""
    
    def __init__(self):
        self.devices: Dict[str, Dict] = {}
        self._init_devices()
    
    def _init_devices(self):
        """初始化设备状态"""
        self.devices = {
            "ac": {"on": False, "temp": 26, "mode": "cool"},
            "light": {"on": False, "brightness": 100, "color": "white"},
            "tv": {"on": False, "volume": 30, "channel": "HDMI1"},
            "curtain": {"open": False, "percent": 0},
            "music": {"playing": False, "volume": 50, "playlist": "relax"}
        }
    
    def execute(self, action: Action) -> Dict:
        """执行设备动作"""
        device = action.device
        if device not in self.devices:
            return {"status": "error", "message": f"未知设备: {device}"}
        
        # 更新设备状态
        for key, value in action.params.items():
            self.devices[device][key] = value
        
        result = {
            "status": "success",
            "device": device,
            "operation": action.operation,
            "state": self.devices[device].copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"[Device] {device}: {action.operation} -> {action.params}")
        return result
    
    def get_state(self, device: str) -> Dict:
        """获取设备状态"""
        return self.devices.get(device, {}).copy()


class ReflectionEngine:
    """反射引擎 - 实现L2和L3反射"""
    
    def __init__(self, habit_db: HabitDatabase, llm_client=None):
        self.habit_db = habit_db
        self.llm_client = llm_client
    
    def l2_task_reflection(self, session_trace: List[LogEntry]) -> Dict:
        """
        L2: 任务级反思
        评估本次任务质量，生成改进策略
        """
        print("\n" + "="*50)
        print("[L2 任务反思] 开始分析...")
        
        if not session_trace:
            return {"success": True, "message": "空轨迹，无需分析"}
        
        # 1. 多维度评估
        total_actions = len(session_trace)
        override_count = sum(1 for log in session_trace if log.user_override)
        
        evaluation = {
            "success": total_actions > 0,
            "user_satisfaction": override_count == 0,
            "efficiency": total_actions,
            "override_rate": override_count / total_actions if total_actions > 0 else 0,
            "total_actions": total_actions
        }
        
        print(f"[L2] 评估结果: 满意度={evaluation['user_satisfaction']}, 覆盖次数={override_count}")
        
        # 2. 分析覆盖情况并提取规则
        for log in session_trace:
            if log.user_override:
                self._extract_rule_from_override(log)
        
        print("="*50 + "\n")
        return evaluation
    
    def _extract_rule_from_override(self, log: LogEntry):
        """从用户覆盖中提取规则"""
        if not log.user_override:
            return
        
        # 模拟规则提取
        changed_to = log.user_override.get("changed_to")
        if changed_to:
            trigger = {
                "intent_keywords": ["睡觉"],  # 简化
                "time_range": f"{log.context.hour}-{log.context.hour}",
                "device": log.planned_action.device
            }
            
            # 计算新默认值(简化版)
            old_params = log.planned_action.params
            new_params = old_params.copy()
            if "temp" in old_params:
                new_params["temp"] = changed_to
            
            self.habit_db.add_rule(
                trigger=trigger,
                old_default=old_params,
                new_default=new_params,
                confidence=0.6,
                reasoning=f"用户将温度从{old_params.get('temp')}改为{changed_to}",
                source="l2_reflection"
            )
    
    def l3_strategy_reflection(self, logs: List[LogEntry], time_window: str = "7d") -> List[Dict]:
        """
        L3: 策略级反思
        深度分析长期行为，修正控制策略
        """
        print("\n" + "="*50)
        print(f"[L3 策略优化] 分析时间窗口: {time_window}...")
        
        if not logs:
            print("[L3] 无日志数据，跳过分析")
            return []
        
        # 1. 简单的模式挖掘
        patterns = self._mine_patterns(logs)
        
        # 2. 生成策略提案
        proposals = []
        for pattern in patterns:
            if pattern["confidence"] > 0.8 and pattern["frequency"] >= 3:
                proposals.append({
                    "type": "update_default",
                    "scenario": pattern["scenario"],
                    "old_default": pattern["old_default"],
                    "new_default": pattern["new_default"],
                    "confidence": pattern["confidence"],
                    "evidence": pattern["instances"]
                })
                print(f"[L3] 发现模式: {pattern['scenario']}, 置信度={pattern['confidence']:.2f}")
        
        # 3. 应用策略
        applied = self._apply_proposals(proposals)
        
        print(f"[L3] 应用了 {len(applied)} 个策略优化")
        print("="*50 + "\n")
        
        return proposals
    
    def _mine_patterns(self, logs: List[LogEntry]) -> List[Dict]:
        """挖掘行为模式"""
        patterns = []
        
        # 分析温度偏好模式
        temp_preferences = {}
        for log in logs:
            if log.planned_action.device == "ac" and "temp" in log.planned_action.params:
                hour = log.context.hour
                key = f"hour_{hour}"
                if key not in temp_preferences:
                    temp_preferences[key] = []
                temp_preferences[key].append(log.planned_action.params["temp"])
        
        # 统计高频模式
        for key, temps in temp_preferences.items():
            if len(temps) >= 3:
                avg_temp = sum(temps) / len(temps)
                patterns.append({
                    "scenario": f"空调温度偏好_{key}",
                    "old_default": 26,
                    "new_default": round(avg_temp),
                    "confidence": min(0.95, 0.6 + len(temps) * 0.05),
                    "frequency": len(temps),
                    "instances": temps
                })
        
        return patterns
    
    def _apply_proposals(self, proposals: List[Dict]) -> List[Dict]:
        """应用策略提案"""
        applied = []
        for p in proposals:
            if p["confidence"] > 0.9:
                # 高置信度自动应用
                applied.append(p)
                print(f"[L3] 自动应用策略: {p['scenario']}")
            elif p["confidence"] > 0.8:
                # 中置信度建议确认
                print(f"[L3] 建议确认: {p['scenario']}")
        return applied


# ==================== L1层函数 ====================

def l1_check_and_act(
    user_intent: str,
    planned_action: Action,
    context: Context,
    habit_db: HabitDatabase,
    short_term_memory: ShortTermMemory,
    device_simulator: DeviceSimulator
) -> Dict:
    """
    L1: 即时调整层
    时机：每个Action前
    核心：查习惯库 → 调整当前动作
    
    示例日志格式:
    {
        "timestamp": "2026-03-23T18:30:00",
        "context": {...},
        "planned": {"device": "ac", "temp": 26, ...},
        "adjusted": {"device": "ac", "temp": 24, ...},
        "reflection": "L1应用: 夜间睡眠模式历史偏好24度(置信度0.85)",
        "result": {...},
        "user_override": {...}
    }
    """
    print(f"\n[L1] 处理意图: '{user_intent}', 计划动作: {planned_action}")
    
    # 1. 查短期记忆
    if short_term_memory.has_recent_override(context):
        override = short_term_memory.get(f"override_{context.user_location}_{planned_action.device}")
        if override:
            adjusted_action = apply_short_term_adjustment(planned_action, override)
            reflection_note = f"L1应用: 短期记忆覆盖"
            print(f"[L1] 应用短期记忆调整")
        else:
            adjusted_action = planned_action
            reflection_note = "L1: 短期记忆已过期"
    else:
        # 2. 查长期习惯库
        habits = habit_db.query(
            intent=user_intent,
            context=context,
            device=planned_action.device,
            confidence_threshold=0.7
        )
        
        # 3. 应用最强匹配
        if habits:
            best_match = max(habits, key=lambda h: h.confidence)
            adjusted_action = apply_habit(planned_action, best_match)
            reflection_note = f"L1应用: {best_match.reasoning}(置信度{best_match.confidence:.2f})"
            habit_db.update_confidence(best_match.rule_id, 0.01)
            print(f"[L1] 应用习惯规则: {best_match.rule_id}")
        else:
            adjusted_action = planned_action
            reflection_note = "L1无匹配，使用默认"
            print(f"[L1] 无匹配规则，使用默认")
    
    # 4. 执行并记录
    result = device_simulator.execute(adjusted_action)
    
    # 5. 检测用户覆盖(模拟)
    user_override = simulate_user_override(context, adjusted_action)
    if user_override:
        short_term_memory.record_override(context, adjusted_action, user_override)
        print(f"[L1] 检测到用户覆盖: {user_override}")
    
    log_entry = LogEntry(
        timestamp=datetime.now().isoformat(),
        context=context,
        planned_action=planned_action,
        adjusted_action=adjusted_action,
        reflection_note=reflection_note,
        execution_result=result["status"],
        user_override=user_override
    )
    
    return {
        "log_entry": log_entry,
        "result": result,
        "user_override": user_override
    }


def apply_short_term_adjustment(action: Action, override: Dict) -> Action:
    """应用短期记忆调整"""
    new_params = action.params.copy()
    new_params.update(override.get("new_params", {}))
    return Action(
        device=action.device,
        operation=action.operation,
        params=new_params
    )


def apply_habit(action: Action, habit: HabitRule) -> Action:
    """应用习惯规则到动作"""
    new_params = action.params.copy()
    habit_action = habit.action
    
    # 合并习惯参数
    for key, value in habit_action.items():
        if key != "device":
            new_params[key] = value
    
    return Action(
        device=action.device,
        operation=action.operation,
        params=new_params
    )


def simulate_user_override(context: Context, action: Action) -> Optional[Dict]:
    """模拟用户覆盖行为(用于演示)"""
    # 10%概率用户会覆盖
    if random.random() < 0.1:
        if "temp" in action.params:
            return {
                "occurred": True,
                "changed_to": action.params["temp"] - 2,
                "after_seconds": random.randint(60, 300)
            }
    return None


def now() -> str:
    """获取当前时间"""
    return datetime.now().isoformat()


# ==================== 智能体主类 ====================

class SmartHomeAgent:
    """
    智能家居Agent主类
    
    实现ReAct循环 + 三层反射机制:
    - L1: 动作前即时调整
    - L2: 任务后异步反思
    - L3: 定期策略优化
    
    LLM集成:
    - 支持OpenAI/兼容API的Function Calling
    - 可配置使用本地规则作为fallback
    """
    
    def __init__(
        self,
        user_id: str = "user_001",
        llm_model: str = "qwen2.5-7B",
        use_local_only: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.user_id = user_id
        self.habit_db = HabitDatabase()
        self.short_term_memory = ShortTermMemory(ttl=3600)  # 1小时过期
        self.device_simulator = DeviceSimulator()
        
        self.session_logs: List[LogEntry] = []
        self.all_logs: List[LogEntry] = []
        self.llm_model = llm_model
        self.use_local_only = use_local_only

        if api_key is None:
            base_url = "http://localhost:8088/v1"
            api_key = "sk-test"
        # LLM配置
        self.llm_client = self._init_llm_client(api_key, base_url)            
        self.reflection = ReflectionEngine(self.habit_db, self.llm_client)
    
    def _init_llm_client(self, api_key: str, base_url: Optional[str] = None):
        """初始化LLM客户端"""
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"[Agent] LLM客户端初始化成功")
        except ImportError:
            print("[警告] 未安装openai库，使用本地规则模式。安装: pip install openai")
            sys.exit()
        except Exception as e:
            print(f"[警告] LLM客户端初始化失败: {e}")
            sys.exit()
        return llm_client
    
    def plan_actions(self, user_input: str) -> List[Action]:
        """
        ReAct: 使用LLM + Function Calling 规划动作序列
        
        如果配置了llm_client，使用Function Calling
        否则使用本地规则匹配作为fallback
        """
        # 优先使用LLM进行规划
        if self.llm_client is not None and not self.use_local_only:
            try:
                return self._plan_with_llm(user_input)
            except Exception as e:
                print(f"[LLM规划失败，使用本地规则] {e}")
                return self._plan_with_local_rules(user_input)
        else:
            return self._plan_with_local_rules(user_input)
    
    def _plan_with_llm(self, user_input: str) -> List[Action]:
        """使用LLM Function Calling规划动作"""
        # Function tool definitions for smart home control
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "control_device",
                    "description": "控制智能家居设备，根据用户意图执行相应的设备操作",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device": {
                                "type": "string",
                                "enum": ["ac", "light", "tv", "curtain", "music"],
                                "description": "设备类型: ac(空调), light(灯光), tv(电视), curtain(窗帘), music(音乐)"
                            },
                            "operation": {
                                "type": "string",
                                "description": "操作类型: turn_on, turn_off, set, dim, open, close, play"
                            },
                            "params": {
                                "type": "object",
                                "description": "操作参数，根据设备类型不同而变化"
                            }
                        },
                        "required": ["device", "operation", "params"]
                    }
                }
            }
        ]
        
        # System prompt for smart home agent
        system_prompt = """你是一个智能家居控制助手。请分析用户的自然语言指令，并调用control_device函数来控制相应的设备。

可用设备及其参数说明:
1. ac (空调): params包含 temp(温度, 16-30), mode(模式: cool/heat/sleep), on(开关: true/false)
2. light (灯光): params包含 brightness(亮度, 0-100), color(颜色: white/warm/cold), on(开关: true/false)
3. tv (电视): params包含 on(开关: true/false), input(输入源: HDMI1/HDMI2/TV), volume(音量, 0-100)
4. curtain (窗帘): params包含 open(开关: true/false), percent(开合百分比, 0-100)
5. music (音乐): params包含 playing(播放状态: true/false), volume(音量, 0-100), playlist(播放列表: relax/rock/pop/jazz)

场景示例:
- 睡觉: 设置空调睡眠模式24度，关闭灯光，关闭窗帘
- 看电影: 打开电视，调暗灯光至30%暖色
- 起床: 打开窗帘50%，开启灯光80%白光，空调24度制冷
- 音乐: 播放音乐，音量40%，放松播放列表

注意:
- 可以调用多个control_device来完成一个场景
- 根据用户意图推断最佳参数
- 如果用户意图不明确，询问用户
"""
        
        context_info = f"""
当前环境信息:
- 时间: {datetime.now().strftime('%H:%M')}
- 季节: {self.get_context().season}
- 用户位置: {self.get_context().user_location}
"""
        
        messages = [
            {"role": "system", "content": system_prompt + context_info},
            {"role": "user", "content": user_input}
        ]
        
        # Call LLM with function calling
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        actions = []
        message = response.choices[0].message
        print(f'_plan_with_llm -> {message=}\n')
        
        # Parse function calls from response
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "control_device":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        action = Action(
                            device=args["device"],
                            operation=args["operation"],
                            params=args.get("params", {})
                        )
                        actions.append(action)
                        print(f"[LLM规划] {action}")
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"[LLM解析错误] {e}, args: {tool_call.function.arguments}")
                        continue
        
        # If no function calls but has content, use local rules as fallback
        if not actions and message.content:
            print(f"[LLM回复] {message.content}")
            return self._plan_with_local_rules(user_input)
        
        return actions if actions else self._plan_with_local_rules(user_input)
    
    def _plan_with_local_rules(self, user_input: str) -> List[Action]:
        """
        本地规则匹配 - 作为LLM的fallback
        """
        actions = []
        intent = user_input.lower()
        
        # 睡眠场景
        if any(kw in intent for kw in ["睡觉", "睡眠", "sleep", "rest"]):
            actions.append(Action(
                device="ac",
                operation="set",
                params={"temp": 26, "mode": "sleep", "on": True}
            ))
            actions.append(Action(
                device="light",
                operation="turn_off",
                params={"on": False}
            ))
            actions.append(Action(
                device="curtain",
                operation="close",
                params={"open": False, "percent": 0}
            ))
        
        # 观影场景
        elif any(kw in intent for kw in ["看电影", "movie", "tv"]):
            actions.append(Action(
                device="tv",
                operation="turn_on",
                params={"on": True, "input": "HDMI1"}
            ))
            actions.append(Action(
                device="light",
                operation="dim",
                params={"brightness": 30, "color": "warm", "on": True}
            ))
        
        # 起床场景
        elif any(kw in intent for kw in ["起床", "wake", "morning"]):
            actions.append(Action(
                device="curtain",
                operation="open",
                params={"open": True, "percent": 50}
            ))
            actions.append(Action(
                device="light",
                operation="turn_on",
                params={"brightness": 80, "color": "white", "on": True}
            ))
            actions.append(Action(
                device="ac",
                operation="set",
                params={"temp": 24, "mode": "cool", "on": True}
            ))
        
        # 音乐场景
        elif any(kw in intent for kw in ["音乐", "music", "听歌"]):
            actions.append(Action(
                device="music",
                operation="play",
                params={"playing": True, "volume": 40, "playlist": "relax"}
            ))
        
        # 默认响应
        else:
            actions.append(Action(
                device="light",
                operation="turn_on",
                params={"on": True}
            ))
        
        print(f"[本地规则规划] 匹配到 {len(actions)} 个动作")
        return actions
    
    def get_context(self) -> Context:
        """获取当前环境上下文"""
        now = datetime.now()
        hour = now.hour
        
        # 简化季节判断
        month = now.month
        if month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        elif month in [9, 10, 11]:
            season = "autumn"
        else:
            season = "winter"
        
        return Context(
            hour=hour,
            season=season,
            weather="clear",
            user_location="bedroom",
            temperature=25.0
        )
    
    def generate_response(self, results: List[Dict]) -> str:
        """生成用户响应"""
        success_count = sum(1 for r in results if r["result"]["status"] == "success")
        total = len(results)
        
        if success_count == total:
            return f"已完成所有操作({total}个设备)，祝您生活愉快！"
        else:
            return f"部分操作完成({success_count}/{total})，请检查设备状态。"
    
    def get_trace(self) -> List[LogEntry]:
        """获取当前会话轨迹"""
        return self.session_logs
    
    async def handle_request(self, user_input: str) -> str:
        """
        处理用户请求 (ReAct主循环)
        
        流程:
        1. 规划动作
        2. L1调整并执行每个动作
        3. 响应用户
        4. 异步L2反思
        """
        print(f"\n{'='*60}")
        print(f"[Agent] 收到请求: '{user_input}'")
        print(f"{'='*60}")
        
        # 清空会话日志
        self.session_logs = []
        
        # ===== ReAct 主循环 =====
        actions = self.plan_actions(user_input)
        print(f"[ReAct] 规划了 {len(actions)} 个动作")
        
        results = []
        context = self.get_context()
        
        for action in actions:
            # L1: 每步检查调整
            l1_result = l1_check_and_act(
                user_intent=user_input,
                planned_action=action,
                context=context,
                habit_db=self.habit_db,
                short_term_memory=self.short_term_memory,
                device_simulator=self.device_simulator
            )
            
            results.append(l1_result)
            self.session_logs.append(l1_result["log_entry"])
            self.all_logs.append(l1_result["log_entry"])
        
        # 响应用户（不等待L2）
        response = self.generate_response(results)
        print(f"[Agent] 响应: {response}")
        
        # L2: 异步任务反思
        asyncio.create_task(
            self._async_l2_reflection()
        )
        
        return response
    
    async def _async_l2_reflection(self):
        """异步执行L2反思"""
        await asyncio.sleep(0.1)  # 模拟异步延迟
        self.reflection.l2_task_reflection(self.session_logs)
    
    def weekly_maintenance(self):
        """定时触发L3策略优化"""
        print(f"\n{'='*60}")
        print("[Agent] 执行周度维护...")
        
        # 过滤最近7天的日志
        week_ago = datetime.now() - timedelta(days=7)
        recent_logs = [
            log for log in self.all_logs 
            if datetime.fromisoformat(log.timestamp) > week_ago
        ]
        
        self.reflection.l3_strategy_reflection(
            logs=recent_logs,
            time_window="7d"
        )
        
        # 清理过期规则
        self.habit_db.archive_stale_rules(threshold_days=30)
        
        print("[Agent] 周度维护完成")
        print(f"{'='*60}\n")
    
    def get_statistics(self) -> Dict:
        """获取Agent运行统计"""
        total_logs = len(self.all_logs)
        override_count = sum(1 for log in self.all_logs if log.user_override)
        
        return {
            "total_executions": total_logs,
            "user_overrides": override_count,
            "override_rate": override_count / total_logs if total_logs > 0 else 0,
            "habit_rules": len(self.habit_db.rules),
            "short_term_memory_items": len(self.short_term_memory.memory)
        }


# ==================== 演示和测试 ====================

async def demo():
    """演示Agent功能"""
    print("\n" + "="*60)
    print("智能家居Agent演示")
    print("="*60)
    
    agent = SmartHomeAgent(user_id="demo_user")
    
    # 演示1: 睡眠场景
    print("\n>>> 演示1: 睡眠场景")
    await agent.handle_request("我要睡觉了")
    await asyncio.sleep(0.5)
    
    # 演示2: 观影场景
    print("\n>>> 演示2: 观影场景")
    await agent.handle_request("我想看电影")
    await asyncio.sleep(0.5)
    
    # 演示3: 起床场景
    print("\n>>> 演示3: 起床场景")
    await agent.handle_request("起床了")
    await asyncio.sleep(0.5)
    
    # 演示4: 音乐场景
    print("\n>>> 演示4: 音乐场景")
    await agent.handle_request("播放音乐")
    await asyncio.sleep(0.5)
    
    # 演示5: 再次睡眠(测试习惯学习)
    print("\n>>> 演示5: 再次睡眠(测试习惯学习)")
    await agent.handle_request("准备睡觉")
    await asyncio.sleep(0.5)
    
    # 执行周度维护
    agent.weekly_maintenance()
    
    # 输出统计
    stats = agent.get_statistics()
    print("\n" + "="*60)
    print("运行统计:")
    print(f"  - 总执行次数: {stats['total_executions']}")
    print(f"  - 用户覆盖次数: {stats['user_overrides']}")
    print(f"  - 覆盖率: {stats['override_rate']:.2%}")
    print(f"  - 习惯规则数: {stats['habit_rules']}")
    print("="*60)
    
    # 输出所有规则
    print("\n当前习惯规则:")
    for rule in agent.habit_db.rules:
        print(f"  [{rule.rule_id}] {rule.reasoning} (置信度: {rule.confidence:.2f})")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo())
