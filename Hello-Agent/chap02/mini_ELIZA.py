# python3
# Create Date: 2026-03-16
# Func: ELIZA
# Learning-Url: https://datawhalechina.github.io/hello-agents/#/en/chapter2/Chapter2-History-of-Agents
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


# 定义信息提取正则模式
extraction_patterns = {
    "name": [
        r'(?:my name is|i am|call me) (\w+)',
        r'(?:^|\s)i\'m (\w+)(?:\s|$|\.|\!)'
    ],
    "age": [
        r'(?:i am|i\'m) (\d+) (?:years? old|y\.o\.?)',
        r'(\d+) years? old'
    ],
    "occupation": [
        r'(?:i work as|i am|i\'m) (?:a|an) ([\w\s]+?)(?:\.|\!|$|\s+(?:at|in|for))',
        r'(?:my job is|i work in) ([\w\s]+?)(?:\.|\!|$)'
    ],
    "location": [
        r'(?:i live in|i\'m from|i come from) ([\w\s,]+?)(?:\.|\!|$)',
        r'(?:live|living) in ([\w\s,]+?)(?:\.|\!|$)'
    ],
    "hobby": [
        r'(?:i like|i enjoy|i love|my hobby is) ([\w\s]+?)(?:\.|\!|$|and)',
        r'(?:interested in|passionate about) ([\w\s]+?)(?:\.|\!|$)'
    ],
    "family": [
        r'(?:my|our) (mother|father|mom|dad|sister|brother|wife|husband|partner)(?:\s+(?:is|name is))?\s+(\w+)',
        r'(\w+) is my (mother|father|mom|dad|sister|brother|wife|husband|partner)'
    ],
    "mood": [
        r'(?:i feel|i am feeling|feeling) (\w+)',
        r'(?:i am|i\'m) (happy|sad|angry|excited|worried|stressed|tired|great)'
    ]
}

def _init_rules():
    """初始化带优先级的规则库 (pattern, priority, responses, context_aware)"""
    return [
        # 高优先级：具体信息提取
        (r'my name is (\w+)', 10, [
            "Nice to meet you, {0}. How can I help you today?",
            "Hello {0}, what brings you here?",
            "{0}, that's a nice name. Tell me more about yourself."
        ], "name"),
        
        (r'i am (\d+) years? old', 10, [
            "Being {0} is a wonderful age. What are your goals at this stage of life?",
            "At {0}, you have so much ahead of you. What concerns you most?",
            "How do you feel about being {0}?"
        ], "age"),
        
        (r'i (?:work as|am) (?:a|an) ([\w\s]+)', 9, [
            "Being a {0} must be interesting. How do you feel about your work?",
            "What challenges do you face as a {0}?",
            "Does working as a {0} bring you satisfaction?"
        ], "occupation"),
        
        (r'i live in ([\w\s,]+)', 9, [
            "How do you like living in {0}?",
            "What's life like in {0}?",
            "Do you feel at home in {0}?"
        ], "location"),
        
        (r'i (?:like|enjoy|love) ([\w\s]+)', 8, [
            "What do you enjoy most about {0}?",
            "How long have you been interested in {0}?",
            "Does {0} help you relax?"
        ], "hobby"),
        
        (r'my (mother|father|sister|brother|wife|husband|partner) (?:is|name is) (\w+)', 8, [
            "Tell me more about your {0}.",
            "How is your relationship with {0}?",
            "What does {0} mean to you?"
        ], "family"),
        
        # 情绪相关
        (r'i feel (sad|depressed|down|upset)', 7, [
            "I'm sorry to hear you're feeling {0}. What's causing this?",
            "When did you start feeling {0}?",
            "Have you talked to anyone about feeling {0}?"
        ], "mood"),
        
        (r'i feel (happy|excited|great|wonderful)', 7, [
            "That's wonderful that you feel {0}! What's making you feel this way?",
            "It's great to hear you're {0}. Tell me more!",
            "What contributed to you feeling {0}?"
        ], "mood"),
        
        # 标准 ELIZA 模式
        (r'i need (.*)', 5, [
            "Why do you need {0}?",
            "Would it really help you to get {0}?",
            "Are you sure you need {0}?"
        ], None),
        
        (r'why don\'t you (.*)\?', 5, [
            "Do you really think I don't {0}?",
            "Perhaps eventually I will {0}.",
            "Do you really want me to {0}?"
        ], None),
        
        (r'why can\'t i (.*)\?', 5, [
            "Do you think you should be able to {0}?",
            "If you could {0}, what would you do?",
            "I don't know -- why can't you {0}?"
        ], None),
        
        (r'i am (.*)', 5, [
            "Did you come to me because you are {0}?",
            "How long have you been {0}?",
            "How do you feel about being {0}?"
        ], None),
        
        (r'(?:what|who) (?:is|are) (?:my|your) (.*)', 4, [
            "Let's discuss why you're asking about {0}.",
            "What do you think about {0}?",
            "Why is {0} important to you right now?"
        ], None),
        
        # 上下文感知规则（使用已记忆的信息）
        (r'(?:what|how) about (?:my|me|i)', 3, [
            "Earlier you mentioned you're {0}. How does that relate?",
            "Considering your work as {1}, what do you think?",
            "As someone who enjoys {2}, you might have insights on this."
        ], "context"),
        
        # 主题相关
        (r'.*\b(mother|mom|mum)\b.*', 3, [
            "Tell me more about your mother.",
            "What was your relationship with your mother like?",
            "How do you feel about your mother?"
        ], "family_topic"),
        
        (r'.*\b(father|dad|daddy)\b.*', 3, [
            "Tell me more about your father.",
            "How did your father make you feel?",
            "What has your father taught you?"
        ], "family_topic"),
        
        (r'.*\b(work|job|career|boss|colleague)\b.*', 3, [
            "How do you feel about your work?",
            "What challenges are you facing in your career?",
            "Does your job bring you satisfaction?"
        ], "work_topic"),
        
        (r'.*\b(study|learn|school|class|university|college)\b.*', 3, [
            "What are you currently learning?",
            "How do you feel about your studies?",
            "What subjects interest you the most?"
        ], "study_topic"),
        
        (r'.*\b(friend|buddy|pal|social)\b.*', 3, [
            "Tell me more about your friends.",
            "How do your friends influence you?",
            "What do you value most in friendship?"
        ], "social_topic"),
        
        (r'.*\b(stress|tired|busy|anxious|worried|overwhelmed)\b.*', 3, [
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
# 定义规则库:模式(正则表达式) -> 响应模板列表
rules = _init_rules()
# 定义代词转换规则
pronoun_swap = {
    "i": "you", 
    "you": "i", 
    "me": "you", 
    "my": "your",
    "am": "are", 
    
    "are": "am", 
    "was": "were",
    "i'd": "you would",
    "i've": "you have", 
    "i'll": "you will", 
    "yours": "mine",
    
    "mine": "yours"
}


def swap_pronouns(phrase):
    """
    对输入短语中的代词进行第一/第二人称转换
    """
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)


def _identify_topics(text: str):
    """识别讨论主题"""
    topics = []
    topic_keywords = {
        "family": ["mother", "father", "parent", "sister", "brother", "family"],
        "work": ["work", "job", "career", "boss", "office", "company"],
        "study": ["study", "learn", "school", "class", "university", "exam"],
        "health": ["health", "sick", "doctor", "hospital", "exercise"],
        "relationship": ["friend", "girlfriend", "boyfriend", "relationship", "marriage"],
        "finance": ["money", "finance", "debt", "salary", "income", "expensive"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)
    
    return topics


# 记住用户在对话中提到的关键信息（如姓名、年龄、职业)
#   step1: 提取关键信息 （正则方式提取）
#   step2: 关键信息保存
#   step3: 基于一些问题组合回答
def extract_information(user_profile, user_input):
    """从用户输入中提取关键信息"""
    extracted = {}
    user_input_lower = user_input.lower()
    
    # 提取姓名
    for pattern in extraction_patterns["name"]:
        match = re.search(pattern, user_input_lower)
        if match:
            name = match.group(1).capitalize()
            user_profile.name = name
            extracted["name"] = name
            break
    
    # 提取年龄
    for pattern in extraction_patterns["age"]:
        match = re.search(pattern, user_input_lower)
        if match:
            age = int(match.group(1))
            user_profile.age = age
            extracted["age"] = age
            break
    
    # 提取职业
    for pattern in extraction_patterns["occupation"]:
        match = re.search(pattern, user_input_lower)
        if match:
            occupation = match.group(1).strip()
            user_profile.occupation = occupation
            extracted["occupation"] = occupation
            break
    
    # 提取位置
    for pattern in extraction_patterns["location"]:
        match = re.search(pattern, user_input_lower)
        if match:
            location = match.group(1).strip()
            user_profile.location = location
            extracted["location"] = location
            break
    
    # 提取爱好
    for pattern in extraction_patterns["hobby"]:
        match = re.search(pattern, user_input_lower)
        if match:
            hobby = match.group(1).strip()
            if hobby not in user_profile.hobbies:
                user_profile.hobbies.append(hobby)
            extracted["hobby"] = hobby
            break
    
    # 提取家庭成员
    for pattern in extraction_patterns["family"]:
        match = re.search(pattern, user_input_lower)
        if match:
            if "is my" in user_input_lower:
                name = match.group(1)
                relation = match.group(2)
            else:
                relation = match.group(1)
                name = match.group(2)
            user_profile.family_members[relation] = name.capitalize()
            extracted["family"] = (relation, name.capitalize())
            break

    # 记录讨论主题
    topics = _identify_topics(user_input_lower)
    if topics:
        for topic in topics:
            if topic not in user_profile.topics_discussed:
                user_profile.topics_discussed.append(topic)
    
    return extracted


def generate_context_response(user_profile):
    """基于记忆生成上下文相关的回应"""
    facts = []
    
    if user_profile.occupation:
        facts.append(f"working as a {user_profile.occupation}")
    if user_profile.hobbies:
        facts.append(f"interested in {random.choice(user_profile.hobbies)}")
    if user_profile.location:
        facts.append(f"living in {user_profile.location}")
    
    if len(facts) >= 2:
        return f"As someone who is {facts[0]} and {facts[1]}, how do you see this situation?"
    elif facts:
        return f"Given that you are {facts[0]}, what are your thoughts on this?"
    
    return None


composition_triggers = [
    "what do you know about me", "tell me about myself", 
    "基于你知道的", "关于我", "给我建议", "你怎么看",
    "analyze me", "my profile", "综合建议", "整体情况"
]


def _should_compose_response(user_profile, user_input: str):
    """判断是否触发组合回答"""
    input_lower = user_input.lower()
    
    # 检查显式触发词
    for trigger in composition_triggers:
        if trigger in input_lower:
            return True
    
    # 检查档案完整度是否足够高（>=50）且用户询问建议
    if user_profile.get_completeness_score() >= 50:
        advice_keywords = ["advice", "suggest", "recommend", "what should", "how to", "建议", "怎么办"]
        if any(kw in input_lower for kw in advice_keywords):
            return True
    
    return False


def respond(user_profile, user_input, his, composer=None):
    """
    根据规则库生成响应
    """
    # 保存对话历史
    his.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": user_input
    })

    # 步骤1: 提取关键信息
    extracted = extract_information(user_profile, user_input)
    
    # 步骤2: 检查是否触发组合回答
    if _should_compose_response(user_input) or \
        (composer and user_profile.get_completeness_score() >= 40):

        composed = composer.compose()
        if composed:
            return composed

    # 步骤3: 按优先级排序规则并匹配
    sorted_rules = sorted(rules, key=lambda x: x[1], reverse=True)
    
    for pattern, priority, responses, context_type in sorted_rules:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # 处理上下文感知规则
            if context_type == "context":
                context_response = generate_context_response(user_profile)
                if context_response:
                    his[-1]["bot"] = context_response
                continue
            
            # 捕获匹配到的部分
            captured_group = match.group(1) if match.groups() else ''
            # 进行代词转换
            swapped_group = swap_pronouns(captured_group)
            # 从模板中随机选择一个并格式化
            template = random.choice(responses).format(swapped_group)
            # 格式化回应
            if "{0}" in template:
                context_response = template.format(swapped_group)
            elif "{1}" in template and len(match.groups()) > 1:
                context_response = template.format(match.group(1), match.group(2))
            else:
                context_response = template

            if user_profile.name and random.random() < 0.3:
                response = f"{user_profile.name}, {context_response[0].lower()}{context_response[1:]}"
            
            return context_response
    # 如果没有匹配任何特定规则，使用最后的通配符规则
    return random.choice(rules[-1][2])



# 主聊天循环
if __name__ == '__main__':
    his = []
    user_profile = UserProfile()
    # Test
    test_ask_dict = {
        "工作/职业": "I am tired from work",
        "学习/学校": "I study online",
        "爱好/兴趣": "I like to play guitar",
        "朋友/社交": "My friend helps me",
        "情绪/压力": "I am happy"
    }
    for tp, a in test_ask_dict.items():
        response = respond(user_profile, a, his)
        print(f'[ {tp} ] TEST-ASK: {a} -> Therapist: {response}\n')

    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        
        response = respond(user_profile, user_input, his)
        print(f"Therapist: {response}")

    print(user_profile.to_dict())
