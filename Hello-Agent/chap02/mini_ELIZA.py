# python3
# Create Date: 2026-03-16
# Func: ELIZA
# Learning-Url: https://datawhalechina.github.io/hello-agents/#/en/chapter2/Chapter2-History-of-Agents

# 记住用户在对话中提到的关键信息（如姓名、年龄、职业)
#   step1: 提取关键信息 （正则方式提取）
#   step2: 关键信息保存
#   step3: 基于一些问题组合回答
# ----------------------------------------------------------------------------------------------------------------------------


import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class UserProfile:
    """用户档案数据结构"""
    name: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    hobbies: List[str] = None
    family_members: Dict[str, str] = None  # 关系 -> 名字
    mood_history: List[Tuple[str, str]] = None  # (时间, 情绪)
    topics_discussed: List[str] = None
    extracted_facts: Dict[str, str] = None  # 其他提取的事实
    
    def __post_init__(self):
        if self.hobbies is None:
            self.hobbies = []
        if self.family_members is None:
            self.family_members = {}
        if self.mood_history is None:
            self.mood_history = []
        if self.topics_discussed is None:
            self.topics_discussed = []
        if self.extracted_facts is None:
            self.extracted_facts = {}
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def get_completeness_score(self):
        """计算档案完整度（0-100）"""
        score = 0
        if self.name: score += 20
        if self.age: score += 10
        if self.occupation: score += 20
        if self.location: score += 10
        if self.hobbies: score += 15
        if self.family_members: score += 15
        if self.mood_history: score += 10
        return min(score, 100)
    
    def get_summary(self):
        """生成用户档案摘要"""
        parts = []
        if self.name:
            parts.append(f"名字是{self.name}")
        if self.age:
            parts.append(f"{self.age}岁")
        if self.occupation:
            parts.append(f"职业是{self.occupation}")
        if self.location:
            parts.append(f"住在{self.location}")
        if self.hobbies:
            parts.append(f"喜欢{', '.join(self.hobbies[:2])}")
        return "；".join(parts) if parts else "还没有太多信息"


class ContextualELIZA:
    """带上下文记忆的 ELIZA 聊天机器人"""
    
    # 信息提取正则模式（按优先级排序，更具体的模式在前）
    EXTRACTION_PATTERNS = {
        "name": [
            r'(?:my name is|call me)\s+["\']?([a-zA-Z]+)["\']?',
            r'(?:i am|i\'m)\s+([a-zA-Z]{2,20})(?:\s+and|\s*,|\s*\.|\s*!|$)',
        ],
        "age": [
            r'(?:i am|i\'m)\s+(\d{1,3})\s*(?:years? old|y\.o\.?|岁)',
            r'\b(\d{1,3})\s*(?:years? old|岁)\b',
        ],
        "occupation": [
            r'(?:i work as|i am|i\'m)\s+(?:a|an)\s+([a-zA-Z\s]{3,30}?)(?:\.|\!|$|\s+(?:at|in|for))',
            r'(?:my job is|i work in)\s+([a-zA-Z\s]{3,30}?)(?:\.|\!|$)',
        ],
        "location": [
            r'(?:i live in|i\'m from|i come from)\s+([a-zA-Z\s,]{2,30}?)(?:\.|\!|$)',
            r'(?:live|living)\s+in\s+([a-zA-Z\s,]{2,30}?)(?:\.|\!|$)',
        ],
        "hobby": [
            r'(?:i like|i enjoy|i love|my hobby is)\s+([a-zA-Z\s]+?)(?:\.|\!|$|\s+and)',
            r'(?:interested in|passionate about)\s+([a-zA-Z\s]+?)(?:\.|\!|$)',
        ],
        "family": [
            r'(?:my|our)\s+(mother|father|mom|dad|sister|brother|wife|husband|partner)\s+(?:is|name is)\s+([a-zA-Z]+)',
            r'([a-zA-Z]+)\s+is\s+my\s+(mother|father|mom|dad|sister|brother|wife|husband|partner)',
        ],
        "mood": [
            r'(?:i feel|i am feeling|feeling)\s+(happy|sad|angry|excited|worried|stressed|tired|great|upset|depressed|anxious)',
            r'(?:i am|i\'m)\s+(happy|sad|angry|excited|worried|stressed|tired|great|upset|depressed|anxious)',
        ]
    }
    
    # 代词转换规则
    PRONOUN_SWAP = {
        "i": "you", "you": "I", "me": "you", "my": "your", "your": "my",
        "am": "are", "are": "am", "was": "were", "were": "was",
        "i'd": "you would", "i've": "you have", "i'll": "you will",
        "you'd": "I would", "you've": "I have", "you'll": "I will",
        "yours": "mine", "mine": "yours", "myself": "yourself", "yourself": "myself"
    }
    
    # 主题关键词
    TOPIC_KEYWORDS = {
        "family": ["mother", "father", "parent", "sister", "brother", "family", "mom", "dad", "家人", "父母", "妈妈", "爸爸"],
        "work": ["work", "job", "career", "boss", "office", "company", "工作", "职业", "老板", "同事"],
        "study": ["study", "learn", "school", "class", "university", "college", "学习", "学校", "大学", "考试"],
        "health": ["health", "sick", "doctor", "hospital", "exercise", "健康", "生病", "医生", "锻炼"],
        "relationship": ["friend", "girlfriend", "boyfriend", "relationship", "marriage", "朋友", "女朋友", "男朋友", "关系"],
        "finance": ["money", "finance", "debt", "salary", "income", "expensive", "钱", "工资", "收入"],
        "hobby": ["hobby", "interest", "like", "enjoy", "爱好", "兴趣", "喜欢"],
    }
    
    # 组合回应触发词
    COMPOSITION_TRIGGERS = [
        "what do you know about me", "tell me about myself", 
        "基于你知道的", "关于我", "给我建议", "你怎么看",
        "analyze me", "my profile", "综合建议", "整体情况",
        "你了解我什么", "记得我吗"
    ]

    # 常见非名字词汇（用于过滤）
    NON_NAME_WORDS = {'am', 'are', 'was', 'were', 'is', 'be', 'been', 'being',
                      'happy', 'sad', 'angry', 'tired', 'stressed', 'worried',
                      'excited', 'great', 'fine', 'good', 'bad', 'okay', 'ok',
                      'years', 'old', 'from', 'live', 'work'}

    def __init__(self):
        self.user_profile = UserProfile()
        self.conversation_history: List[Dict] = []
        self.rules = self._init_rules()
    
    def _init_rules(self):
        """初始化带优先级的规则库 (pattern, priority, responses, context_type)"""
        return [
            # 高优先级：具体信息提取
            (r'my name is\s+(\w+)', 10, [
                "Nice to meet you, {name}! How can I help you today?",
                "Hello {name}, what brings you here?",
                "{name}, that's a nice name. Tell me more about yourself."
            ], "name"),
            
            (r'i am\s+(\d+)\s*years? old', 10, [
                "Being {age} is a wonderful age. What are your goals at this stage of life?",
                "At {age}, you have so much ahead of you. What concerns you most?",
                "How do you feel about being {age}?"
            ], "age"),
            
            (r'i\s+(?:work as|am)\s+(?:a|an)\s+([a-zA-Z\s]{3,30}?)(?:\.|\!|$|\s+(?:at|in|for))', 9, [
                "Being a {occupation} must be interesting. How do you feel about your work?",
                "What challenges do you face as a {occupation}?",
                "Does working as a {occupation} bring you satisfaction?"
            ], "occupation"),
            
            (r'i live in\s+([\w\s,]+)', 9, [
                "How do you like living in {location}?",
                "What's life like in {location}?",
                "Do you feel at home in {location}?"
            ], "location"),
            
            (r'i\s+(?:like|enjoy|love)\s+([\w\s]+)', 8, [
                "What do you enjoy most about {hobby}?",
                "How long have you been interested in {hobby}?",
                "Does {hobby} help you relax?"
            ], "hobby"),
            
            (r'my\s+(mother|father|sister|brother|wife|husband|partner)\s+(?:is|name is)\s+(\w+)', 8, [
                "Tell me more about your {relation}.",
                "How is your relationship with your {relation}?",
                "What does your {relation} mean to you?"
            ], "family"),
            
            # 情绪相关
            (r'i feel\s+(sad|depressed|down|upset)', 7, [
                "I'm sorry to hear you're feeling {mood}. What's causing this?",
                "When did you start feeling {mood}?",
                "Have you talked to anyone about feeling {mood}?"
            ], "mood"),
            
            (r'i feel\s+(happy|excited|great|wonderful)', 7, [
                "That's wonderful that you feel {mood}! What's making you feel this way?",
                "It's great to hear you're {mood}. Tell me more!",
                "What contributed to you feeling {mood}?"
            ], "mood"),
            
            # 标准 ELIZA 模式
            (r'i need\s+(.*)', 5, [
                "Why do you need {0}?",
                "Would it really help you to get {0}?",
                "Are you sure you need {0}?"
            ], None),
            
            (r'why don\'t you\s+(.*)\?', 5, [
                "Do you really think I don't {0}?",
                "Perhaps eventually I will {0}.",
                "Do you really want me to {0}?"
            ], None),
            
            (r'why can\'t i\s+(.*)\?', 5, [
                "Do you think you should be able to {0}?",
                "If you could {0}, what would you do?",
                "I don't know -- why can't you {0}?"
            ], None),
            
            (r'i am\s+(.*)', 5, [
                "Did you come to me because you are {0}?",
                "How long have you been {0}?",
                "How do you feel about being {0}?"
            ], None),
            
            (r'(?:what|who)\s+(?:is|are)\s+(?:my|your)\s+(.*)', 4, [
                "Let's discuss why you're asking about {0}.",
                "What do you think about {0}?",
                "Why is {0} important to you right now?"
            ], None),
            
            # 主题相关
            (r'.*\b(mother|mom|mum|妈妈|母亲)\b.*', 3, [
                "Tell me more about your mother.",
                "What was your relationship with your mother like?",
                "How do you feel about your mother?"
            ], "family_topic"),
            
            (r'.*\b(father|dad|daddy|爸爸|父亲)\b.*', 3, [
                "Tell me more about your father.",
                "How did your father make you feel?",
                "What has your father taught you?"
            ], "family_topic"),
            
            (r'.*\b(work|job|career|boss|colleague|工作|职业)\b.*', 3, [
                "How do you feel about your work?",
                "What challenges are you facing in your career?",
                "Does your job bring you satisfaction?"
            ], "work_topic"),
            
            (r'.*\b(study|learn|school|class|university|college|学习|学校)\b.*', 3, [
                "What are you currently learning?",
                "How do you feel about your studies?",
                "What subjects interest you the most?"
            ], "study_topic"),
            
            (r'.*\b(friend|buddy|pal|social|朋友)\b.*', 3, [
                "Tell me more about your friends.",
                "How do your friends influence you?",
                "What do you value most in friendship?"
            ], "social_topic"),
            
            (r'.*\b(stress|tired|busy|anxious|worried|overwhelmed|压力|累|焦虑)\b.*', 3, [
                "What is causing you to feel this way?",
                "How do you usually cope with stress?",
                "Have you talked to anyone about these feelings?"
            ], "stress_topic"),
            
            # 通配符规则（最低优先级）
            (r'.*', 1, [
                "Please tell me more.",
                "Let's change focus a bit... Tell me about your family.",
                "Can you elaborate on that?",
                "I see. Go on.",
                "Interesting. Tell me more about that.",
                "How does that make you feel?"
            ], None)
        ]
    
    def swap_pronouns(self, phrase: str):
        """对输入短语中的代词进行第一/第二人称转换"""
        words = phrase.lower().split()
        swapped_words = []
        for word in words:
            # 处理标点
            clean_word = word.strip(".,!?;:'\"")
            suffix = word[len(clean_word):] if len(word) > len(clean_word) else ""
            swapped = self.PRONOUN_SWAP.get(clean_word, clean_word)
            swapped_words.append(swapped + suffix)
        return " ".join(swapped_words)
     
    def extract_information(self, user_input: str):
        """从用户输入中提取关键信息"""
        extracted = {}
        user_input_lower = user_input.lower()
        
        # 提取姓名（只在未设置时提取，避免覆盖）
        if not self.user_profile.name:
            for pattern in self.EXTRACTION_PATTERNS["name"]:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    name = match.group(1).capitalize()
                    # 过滤非名字词汇
                    if name.lower() not in self.NON_NAME_WORDS and len(name) >= 2:
                        self.user_profile.name = name
                        extracted["name"] = name
                        break
        
        # 提取年龄（只在未设置时提取）
        if not self.user_profile.age:
            for pattern in self.EXTRACTION_PATTERNS["age"]:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    age = int(match.group(1))
                    if 1 <= age <= 120:  # 合理的年龄范围
                        self.user_profile.age = age
                        extracted["age"] = age
                        break
        
        # 提取职业（只在未设置时提取）
        if not self.user_profile.occupation:
            for pattern in self.EXTRACTION_PATTERNS["occupation"]:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    occupation = match.group(1).strip().lower()
                    # 过滤无效职业（排除形容词和常见非职业词汇）
                    invalid_occupation_words = {'very', 'really', 'quite', 'so', 'too', 'stressed', 'tired', 'busy', 
                                                'happy', 'sad', 'demanding', 'hard', 'difficult', 'easy', 'good', 'bad'}
                    if occupation not in self.NON_NAME_WORDS and occupation not in invalid_occupation_words and len(occupation) >= 3:
                        self.user_profile.occupation = occupation
                        extracted["occupation"] = occupation
                        break
        
        # 提取位置（只在未设置时提取）
        if not self.user_profile.location:
            for pattern in self.EXTRACTION_PATTERNS["location"]:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    location = match.group(1).strip()
                    if len(location) >= 2:
                        self.user_profile.location = location
                        extracted["location"] = location
                        break
        
        # 提取爱好（可以添加多个）
        for pattern in self.EXTRACTION_PATTERNS["hobby"]:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                hobby = match.group(1).strip().lower()
                if hobby not in self.NON_NAME_WORDS and len(hobby) >= 3:
                    if hobby not in self.user_profile.hobbies:
                        self.user_profile.hobbies.append(hobby)
                    extracted["hobby"] = hobby
                    break
        
        # 提取家庭成员
        for pattern in self.EXTRACTION_PATTERNS["family"]:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                if "is my" in user_input_lower:
                    name = match.group(1).capitalize()
                    relation = match.group(2).lower()
                else:
                    relation = match.group(1).lower()
                    name = match.group(2).capitalize()
                self.user_profile.family_members[relation] = name
                extracted["family"] = (relation, name)
                break
        
        # 提取情绪（每次都可以记录）
        for pattern in self.EXTRACTION_PATTERNS["mood"]:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                mood = match.group(1).lower()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.user_profile.mood_history.append((timestamp, mood))
                extracted["mood"] = mood
                break
        
        # 记录讨论主题
        topics = self._identify_topics(user_input_lower)
        if topics:
            for topic in topics:
                if topic not in self.user_profile.topics_discussed:
                    self.user_profile.topics_discussed.append(topic)
        
        return extracted
    
    def _identify_topics(self, text: str):
        """识别讨论主题"""
        topics = []
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        return topics

    def _should_compose_response(self, user_input: str):
        """判断是否触发组合回答"""
        input_lower = user_input.lower()
        
        # 检查显式触发词
        for trigger in self.COMPOSITION_TRIGGERS:
            if trigger in input_lower:
                return True
        
        # 检查档案完整度是否足够高（>=50）且用户询问建议
        if self.user_profile.get_completeness_score() >= 50:
            advice_keywords = ["advice", "suggest", "recommend", "what should", "how to", "建议", "怎么办"]
            if any(kw in input_lower for kw in advice_keywords):
                return True
        
        return False

    def generate_memory_response(self):
        """基于记忆生成个性化回应"""
        profile = self.user_profile
        
        # 如果信息很少，返回 None
        if profile.get_completeness_score() < 20:
            return None
        
        # 构建基于记忆的回应
        memory_parts = []
        if profile.name:
            memory_parts.append(f"{profile.name}")
        
        if profile.occupation and profile.hobbies:
            return f"{profile.name}，作为一名{profile.occupation}，同时又是{profile.hobbies[0]}的爱好者，你对这个问题有什么看法呢？"
        
        if profile.age and profile.location:
            return f"{profile.name}，在{profile.location}生活的{profile.age}岁年轻人，能分享一下你的想法吗？"
        
        if profile.occupation:
            return f"{profile.name}，从你的职业{profile.occupation}角度来看，你怎么看这个问题？"
        
        if profile.hobbies:
            return f"{profile.name}，作为{profile.hobbies[0]}的爱好者，你觉得呢？"
        
        if profile.name:
            return f"{profile.name}，根据我对你的了解，你觉得呢？"
        
        return None

    def generate_profile_summary(self):
        """生成用户档案总结"""
        profile = self.user_profile
        
        if profile.get_completeness_score() == 0:
            return "我们刚开始对话，我还不太了解你呢。多告诉我一些关于你的事情吧！"
        
        parts = []
        
        if profile.name:
            parts.append(f"你的名字是{profile.name}")
        if profile.age:
            parts.append(f"{profile.age}岁")
        if profile.occupation:
            parts.append(f"职业是{profile.occupation}")
        if profile.location:
            parts.append(f"住在{profile.location}")
        if profile.hobbies:
            parts.append(f"喜欢{', '.join(profile.hobbies[:3])}")
        if profile.family_members:
            family_str = "，".join([f"{rel}是{name}" for rel, name in list(profile.family_members.items())[:2]])
            parts.append(f"家人情况：{family_str}")
        
        summary = "。".join(parts)
        
        # 添加建议
        if profile.get_completeness_score() >= 60:
            summary += f"\n\n根据我对你的了解，你似乎是一个对生活有追求的人。"
        
        return summary

    def format_response(self, template: str, match_groups: tuple, extracted: Dict):
        """格式化回应模板，支持多种变量替换"""
        response = template
        
        # 替换匹配组 {0}, {1}, ...
        for i, group in enumerate(match_groups):
            swapped = self.swap_pronouns(group) if group else ""
            response = response.replace(f"{{{i}}}", swapped)
        
        # 替换记忆变量 {name}, {age}, {occupation}, {location}, {hobby}, {mood}, {relation}
        profile = self.user_profile
        
        # 使用字典映射简化替换逻辑
        replacements = {
            "{name}": profile.name,
            "{age}": str(profile.age) if profile.age else None,
            "{occupation}": profile.occupation,
            "{location}": profile.location,
            "{hobby}": extracted.get("hobby") or (profile.hobbies[0] if profile.hobbies else None),
            "{mood}": extracted.get("mood"),
            "{relation}": extracted.get("family", [None, None])[0] if "family" in extracted else None,
        }
        
        for placeholder, value in replacements.items():
            if placeholder in response and value:
                response = response.replace(placeholder, value)
        
        return response

    def respond(self, user_input: str):
        """根据规则库生成响应"""
        # 保存对话历史
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user": user_input
        })
        
        # 步骤1: 提取关键信息
        extracted = self.extract_information(user_input)
        
        # 步骤2: 检查是否触发组合回答（询问关于用户的信息）
        if self._should_compose_response(user_input):
            return self.generate_profile_summary()
        
        # 步骤3: 随机尝试使用记忆生成回应（15%概率，当有足够信息且没有特定匹配时）
        # 这样不会干扰正常的规则匹配，只是偶尔增加个性化
        
        # 步骤4: 按优先级排序规则并匹配
        sorted_rules = sorted(self.rules, key=lambda x: x[1], reverse=True)
        
        for pattern, priority, responses, context_type in sorted_rules:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                # 选择回应模板
                template = random.choice(responses)
                
                # 格式化回应
                response = self.format_response(template, match.groups(), extracted)
                
                # 保存并返回
                self.conversation_history[-1]["bot"] = response
                return response
        
        # 如果没有匹配任何特定规则，使用最后的通配符规则
        default_response = random.choice(self.rules[-1][2])
        self.conversation_history[-1]["bot"] = default_response
        return default_response
    
    def get_profile(self):
        """获取用户档案"""
        return self.user_profile.to_dict()
    
    def get_history(self):
        """获取对话历史"""
        return self.conversation_history
    
    def clear_memory(self):
        """清空记忆"""
        self.user_profile = UserProfile()
        self.conversation_history = []


def run_demo():
    """运行演示测试"""
    print("=" * 60)
    print("ELIZA Chatbot with Context Memory - Demo")
    print("=" * 60)
    
    eliza = ContextualELIZA()
    
    # 测试用例
    test_cases = [
        ("自我介绍", "My name is Alice"),
        ("年龄", "I am 25 years old"),
        ("职业", "I work as a software engineer"),
        ("位置", "I live in Beijing"),
        ("爱好", "I enjoy playing guitar and hiking"),
        ("家庭成员", "My mother is Mary"),
        ("情绪", "I feel happy today"),
        ("询问关于自己", "What do you know about me?"),
        ("工作话题", "My job is stressful"),
        ("通用话题", "I don't know what to do"),
        ("组合", "基于你知道的给我一些建议")
    ]
    
    for topic, user_input in test_cases:
        response = eliza.respond(user_input)
        print(f"\n[{topic}]")
        print(f"  You: {user_input}")
        print(f"  ELIZA: {response}")
    
    print("\n" + "=" * 60)
    print("用户档案:")
    print("=" * 60)
    import json
    print(json.dumps(eliza.get_profile(), indent=2, ensure_ascii=False))


def run_interactive():
    """运行交互式对话"""
    print("=" * 60)
    print("ELIZA Chatbot with Context Memory")
    print("=" * 60)
    print("Commands:")
    print("  'quit/exit/bye' - 结束对话")
    print("  'profile' - 查看已记忆的信息")
    print("  'history' - 查看对话历史")
    print("  'clear' - 清空记忆")
    print("=" * 60)
    
    eliza = ContextualELIZA()
    
    print("\nELIZA: Hello! I'm ELIZA. How can I help you today?")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "bye", "再见", "拜拜"]:
                name = eliza.user_profile.name
                if name:
                    print(f"ELIZA: Goodbye, {name}! It was nice talking to you.")
                else:
                    print("ELIZA: Goodbye! It was nice talking to you.")
                break
            
            if user_input.lower() == "profile":
                import json
                print("\n--- 用户档案 ---")
                print(json.dumps(eliza.get_profile(), indent=2, ensure_ascii=False))
                print(f"完整度: {eliza.user_profile.get_completeness_score()}%")
                continue
            
            if user_input.lower() == "history":
                print("\n--- 对话历史 ---")
                for entry in eliza.get_history():
                    print(f"[{entry['timestamp']}] You: {entry['user']}")
                    if 'bot' in entry:
                        print(f"  ELIZA: {entry['bot']}")
                continue
            
            if user_input.lower() == "clear":
                eliza.clear_memory()
                print("ELIZA: Memory cleared. Let's start fresh!")
                continue
            
            response = eliza.respond(user_input)
            print(f"ELIZA: {response}")
            
        except KeyboardInterrupt:
            print("\nELIZA: Goodbye!")
            break
        except Exception as e:
            print(f"ELIZA: Sorry, something went wrong. ({e})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_interactive()
