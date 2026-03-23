# python3
# Create Date: 2026-03-22
# Author: Scc_hy
# Func: reflection
#       执行 -> 反思 -> 优化
# =========================================================================================

doc = """
1. 执行 (Execution)：
    首先，智能体使用我们熟悉的方法（如 ReAct 或 Plan-and-Solve）尝试完成任务，生成一个初步的解决方案或行动轨迹。
    这可以看作是“初稿”。
2. 反思 (Reflection)：
    接着，智能体进入反思阶段。它会调用一个独立的、或者带有特殊提示词的大语言模型实例，来扮演一个“评审员”的角色。
    这个“评审员”会审视第一步生成的“初稿”，并从多个维度进行评估，例如：
        事实性错误：是否存在与常识或已知事实相悖的内容？
        逻辑漏洞：推理过程是否存在不连贯或矛盾之处？
        效率问题：是否有更直接、更简洁的路径来完成任务？
        遗漏信息：是否忽略了问题的某些关键约束或方面？ 根据评估，它会生成一段结构化的反馈 (Feedback)，指出具体的问题所在和改进建议。
3. 优化 (Refinement)：
    最后，智能体将“初稿”和“反馈”作为新的上下文，再次调用大语言模型，
    要求它根据反馈内容对初稿进行修正，生成一个更完善的“修订稿”。

我们将引入记忆管理机制，因为reflection通常对应着信息的存储和提取，如果上下文足够长的情况，
想让“评审员”直接获取所有的信息然后进行反思往往会传入很多冗余信息。这一步实践我们主要完成代码生成与迭代优化。


成本与收益：
1. 主要成本
    模型调用开销增加:这是最直接的成本。每进行一轮迭代，至少需要额外调用两次大语言模型（一次用于反思，一次用于优化）。如果迭代多轮，API 调用成本和计算资源消耗将成倍增加。
    任务延迟显著提高:Reflection 是一个串行过程，每一轮的优化都必须等待上一轮的反思完成。这使得任务的总耗时显著延长，不适合对实时性要求高的场景。
    提示工程复杂度上升:如我们的案例所示，Reflection 的成功在很大程度上依赖于高质量、有针对性的提示词。为“执行”、“反思”、“优化”等不同阶段设计和调试有效的提示词，需要投入更多的开发精力。

2. 核心收益
    解决方案质量的跃迁: 最大的收益在于，它能将一个“合格”的初始方案，迭代优化成一个“优秀”的最终方案。这种从功能正确到性能高效、从逻辑粗糙到逻辑严谨的提升，在很多关键任务中是至关重要的。
    鲁棒性与可靠性增强: 通过内部的自我纠错循环，智能体能够发现并修复初始方案中可能存在的逻辑漏洞、事实性错误或边界情况处理不当等问题，从而大大提高了最终结果的可靠性。

综上所述，Reflection 机制是一种典型的“以成本换质量”的策略。它非常适合那些对最终结果的质量、准确性和可靠性有极高要求，且对任务完成的实时性要求相对宽松的场景。例如:
- 生成关键的业务代码或技术报告。
- 在科学研究中进行复杂的逻辑推演。
- 需要深度分析和规划的决策支持系统。

"""
import sys 
from os.path import dirname
from typing import List, Dict, Any, Optional
CUR_DIR = dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
from HelloAgentLLM import HelloAgentsLLM, ToolExecutor, search


class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    def __init__(self):
        """
        初始化一个空列表来存储所有记录。
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self):
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self):
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        如果不存在，则返回 None。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None



INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")
        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")
            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)
            
            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)      
          
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str):
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text



if __name__ == '__main__':
    reflect_agent = ReflectionAgent(
        HelloAgentsLLM()
    )
    reflect_agent.run('编写一个Python函数，找出1到n之间所有的素数 (prime numbers)')

    res = '''

--- 开始处理任务 ---
任务: 编写一个Python函数，找出1到n之间所有的素数 (prime numbers)

--- 正在进行初始尝试 ---
🧠 正在调用 qwen2.5-7B 模型...
✅ 大语言模型响应成功:
```python
def find_primes(n: int) -> list:
    """
    返回从1到n之间所有的素数列表。

    参数:
    n (int): 上限值，寻找1到n之间的素数。

    返回:
    list: 包含所有素数的列表。
    """
    def is_prime(num: int) -> bool:
        """判断一个数是否为素数"""
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True

    primes = [num for num in range(1, n + 1) if is_prime(num)]
    return primes
```
📝 记忆已更新，新增一条 'execution' 记录。

--- 第 1/3 轮迭代 ---

-> 正在进行反思...
🧠 正在调用 qwen2.5-7B 模型...
✅ 大语言模型响应成功:
当前代码的时间复杂度较高，因为对于每个数字`num`，`is_prime`函数会进行`sqrt(num)`次除法操作。整体时间复杂度为O(n * sqrt(n))。

可以考虑使用埃拉托斯特尼筛法（Sieve of Eratosthenes）来优化此问题。这种方法的时间复杂度为O(n log log n)，明显优于当前实现。

具体改进如下：

```python
def find_primes(n: int) -> list:
    """
    返回从1到n之间所有的素数列表。

    参数:
    n (int): 上限值，寻找1到n之间的素数。

    返回:
    list: 包含所有素数的列表。
    """
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not prime numbers
    for current in range(2, int(n ** 0.5) + 1):
        if sieve[current]:
            for multiple in range(current * current, n + 1, current):
                sieve[multiple] = False
    primes = [num for num, is_prime in enumerate(sieve) if is_prime]
    return primes
```

这个改进版本使用了一个布尔数组`sieve`来标记哪些数字是素数。首先将所有数字标记为素数，然后从2开始遍历，将所有2的倍数标记为非素数。这样可以显著减少计算量。
📝 记忆已更新，新增一条 'reflection' 记录。

-> 正在进行优化...
🧠 正在调用 qwen2.5-7B 模型...
✅ 大语言模型响应成功:
```python
def find_primes(n: int) -> list:
    """
    返回从1到n之间所有的素数列表。

    参数:
    n (int): 上限值，寻找1到n之间的素数。

    返回:
    list: 包含所有素数的列表。
    """
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not prime numbers
    for current in range(2, int(n ** 0.5) + 1):
        if sieve[current]:
            for multiple in range(current * current, n + 1, current):
                sieve[multiple] = False
    primes = [num for num, is_prime in enumerate(sieve) if is_prime]
    return primes
```
📝 记忆已更新，新增一条 'execution' 记录。

--- 第 2/3 轮迭代 ---

-> 正在进行反思...
🧠 正在调用 qwen2.5-7B 模型...
✅ 大语言模型响应成功:
当前代码实现的是埃拉托斯特尼筛法（Sieve of Eratosthenes），这是一种高效的查找素数的方法。该方法的时间复杂度为O(n log log n)，在查找大量素数时表现良好。

无需改进。当前实现已经较为优化，但在某些场景下可以考虑进一步优化内存使用，例如只存储奇数索引，以减少一半的空间占用。但这种优化对整体时间复杂度影响不大。
📝 记忆已更新，新增一条 'reflection' 记录。

✅ 反思认为代码已无需改进，任务完成。

--- 任务完成 ---
最终生成的代码:
```python
def find_primes(n: int) -> list:
    """
    返回从1到n之间所有的素数列表。

    参数:
    n (int): 上限值，寻找1到n之间的素数。

    返回:
    list: 包含所有素数的列表。
    """
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not prime numbers
    for current in range(2, int(n ** 0.5) + 1):
        if sieve[current]:
            for multiple in range(current * current, n + 1, current):
                sieve[multiple] = False
    primes = [num for num, is_prime in enumerate(sieve) if is_prime]
    return primes
```
'''  
    