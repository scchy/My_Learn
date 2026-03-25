# 新增文件：error_recovery.py
# Author: Scc_hy
# Create Date: 2026-03-23
# Func: 智能体的错误恢复与容错策略
# ========================================================================

import re
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field


class ErrorType(Enum):
    """错误类型分类"""
    TOOL_NOT_FOUND = auto()        # 工具不存在
    TOOL_EXECUTION = auto()        # 工具执行错误
    INVALID_PARAMETERS = auto()    # 参数无效
    NETWORK_ERROR = auto()         # 网络错误
    TIMEOUT = auto()               # 超时
    PERMISSION_DENIED = auto()     # 权限不足
    RESOURCE_UNAVAILABLE = auto()  # 资源不可用
    UNKNOWN = auto()               # 未知错误


@dataclass
class ErrorRecord:
    """错误记录数据结构"""
    tool_name: str
    error_type: ErrorType
    error_message: str
    step: int
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """恢复动作数据结构"""
    action: str  # "retry" | "fallback" | "escalate" | "skip"
    feedback_message: str  # 插入到 messages 中的纠错提示
    excluded_tools: List[str] = field(default_factory=list)  # 本次要屏蔽的工具
    suggested_alternative: Optional[str] = None  # 建议的替代工具
    wait_time: float = 0.0  # 重试前的等待时间（秒）


class ErrorAnalyzer:
    """错误分析器：分析错误类型和原因"""
    
    # 错误模式匹配规则
    ERROR_PATTERNS = {
        ErrorType.TOOL_NOT_FOUND: [
            r"not found",
            r"does not exist",
            r"no such tool",
            r"工具.*不存在",
            r"找不到工具",
        ],
        ErrorType.INVALID_PARAMETERS: [
            r"invalid parameter",
            r"missing required",
            r"type error",
            r"参数.*无效",
            r"缺少.*参数",
        ],
        ErrorType.TOOL_EXECUTION: [
            r"execution failed",
            r"runtime error",
            r"执行.*失败",
            r"运行.*错误",
        ],
        ErrorType.NETWORK_ERROR: [
            r"connection",
            r"network",
            r"unreachable",
            r"连接.*失败",
            r"网络.*错误",
        ],
        ErrorType.TIMEOUT: [
            r"timeout",
            r"timed out",
            r"超时",
        ],
        ErrorType.PERMISSION_DENIED: [
            r"permission",
            r"access denied",
            r"unauthorized",
            r"权限",
            r"拒绝访问",
        ],
        ErrorType.RESOURCE_UNAVAILABLE: [
            r"resource",
            r"unavailable",
            r"资源.*不可用",
        ],
    }
    
    @classmethod
    def analyze(cls, error: Exception, tool_name: str) -> ErrorType:
        """分析错误类型"""
        error_str = str(error).lower()
        
        for error_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_str, re.IGNORECASE):
                    return error_type
        
        return ErrorType.UNKNOWN
    
    @classmethod
    def get_error_suggestion(cls, error_type: ErrorType, tool_name: str) -> str:
        """根据错误类型获取建议"""
        suggestions = {
            ErrorType.TOOL_NOT_FOUND: f"工具 '{tool_name}' 不存在，请检查工具名称或使用可用工具列表中的工具。",
            ErrorType.INVALID_PARAMETERS: f"工具 '{tool_name}' 的参数格式不正确，请检查参数类型和必填项。",
            ErrorType.TOOL_EXECUTION: f"工具 '{tool_name}' 执行失败，可能是输入数据有问题，请尝试修改输入。",
            ErrorType.NETWORK_ERROR: f"网络连接问题导致无法调用 '{tool_name}'，请检查网络或稍后重试。",
            ErrorType.TIMEOUT: f"工具 '{tool_name}' 调用超时，请尝试简化查询或分批处理。",
            ErrorType.PERMISSION_DENIED: f"没有权限调用工具 '{tool_name}'，请使用其他可用工具。",
            ErrorType.RESOURCE_UNAVAILABLE: f"工具 '{tool_name}' 所需的资源暂时不可用，请稍后重试。",
            ErrorType.UNKNOWN: f"调用工具 '{tool_name}' 时发生未知错误，请尝试其他方法。",
        }
        return suggestions.get(error_type, suggestions[ErrorType.UNKNOWN])


class ErrorRecoveryStrategy:
    """可插拔的容错策略"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.error_history: List[ErrorRecord] = []
        self.retry_count: int = 0
        self.max_retries: int = max_retries
        self.backoff_factor: float = backoff_factor
        self.excluded_tools: set = set()  # 持续屏蔽的工具
        self.fallback_handler: Optional[Callable] = None  # 降级处理函数
    
    def on_tool_error(
            self, 
            error: Exception, 
            tool_name: str,
            messages: list, 
            available_tools: Optional[List[str]] = None
        ) -> RecoveryAction:
        """
        处理工具错误，返回恢复动作
        
        Args:
            error: 异常对象
            tool_name: 出错的工具名
            messages: 当前对话历史
            available_tools: 可用工具列表（用于推荐替代工具）
            
        Returns:
            RecoveryAction: 恢复动作配置
        """
        self.retry_count += 1
        
        # 分析错误类型
        error_type = ErrorAnalyzer.analyze(error, tool_name)
        
        # 记录错误
        record = ErrorRecord(
            tool_name=tool_name,
            error_type=error_type,
            error_message=str(error),
            step=len(messages),
            retry_count=self.retry_count
        )
        self.error_history.append(record)
        
        # 检查是否超过最大重试次数
        if self.retry_count >= self.max_retries:
            return self._handle_max_retries_exceeded(tool_name, error_type, available_tools)
        
        # 根据错误类型制定恢复策略
        return self._get_recovery_action(error_type, tool_name, available_tools)
    
    def _get_recovery_action(self, error_type: ErrorType, tool_name: str,
                             available_tools: Optional[List[str]] = None) -> RecoveryAction:
        """根据错误类型获取恢复动作"""
        
        # 工具不存在：屏蔽该工具并推荐替代
        if error_type == ErrorType.TOOL_NOT_FOUND:
            self.excluded_tools.add(tool_name)
            alternative = self._find_alternative_tool(tool_name, available_tools or [])
            return RecoveryAction(
                action="retry",
                feedback_message=ErrorAnalyzer.get_error_suggestion(error_type, tool_name),
                excluded_tools=[tool_name],
                suggested_alternative=alternative
            )
        
        # 参数错误：重试并提示修正参数
        elif error_type == ErrorType.INVALID_PARAMETERS:
            return RecoveryAction(
                action="retry",
                feedback_message=ErrorAnalyzer.get_error_suggestion(error_type, tool_name),
                excluded_tools=[]
            )
        
        # 网络错误：指数退避后重试
        elif error_type == ErrorType.NETWORK_ERROR:
            wait_time = self.backoff_factor ** self.retry_count
            return RecoveryAction(
                action="retry",
                feedback_message=f"{ErrorAnalyzer.get_error_suggestion(error_type, tool_name)} 等待 {wait_time:.1f} 秒后重试...",
                excluded_tools=[],
                wait_time=wait_time
            )
        
        # 超时：重试并建议简化
        elif error_type == ErrorType.TIMEOUT:
            return RecoveryAction(
                action="retry",
                feedback_message=ErrorAnalyzer.get_error_suggestion(error_type, tool_name),
                excluded_tools=[]
            )
        
        # 权限错误：永久屏蔽该工具
        elif error_type == ErrorType.PERMISSION_DENIED:
            self.excluded_tools.add(tool_name)
            alternative = self._find_alternative_tool(tool_name, available_tools or [])
            return RecoveryAction(
                action="retry",
                feedback_message=ErrorAnalyzer.get_error_suggestion(error_type, tool_name),
                excluded_tools=[tool_name],
                suggested_alternative=alternative
            )
        
        # 执行错误：重试一次，然后降级
        elif error_type == ErrorType.TOOL_EXECUTION:
            if self.retry_count >= 2:
                return RecoveryAction(
                    action="fallback",
                    feedback_message=f"工具 '{tool_name}' 多次执行失败，将使用备用方案直接回答。",
                    excluded_tools=[tool_name]
                )
            return RecoveryAction(
                action="retry",
                feedback_message=ErrorAnalyzer.get_error_suggestion(error_type, tool_name),
                excluded_tools=[]
            )
        
        # 未知错误：默认重试
        else:
            return RecoveryAction(
                action="retry",
                feedback_message=ErrorAnalyzer.get_error_suggestion(error_type, tool_name),
                excluded_tools=[]
            )
    
    def _handle_max_retries_exceeded(self, tool_name: str, error_type: ErrorType,
                                     available_tools: Optional[List[str]] = None) -> RecoveryAction:
        """处理超过最大重试次数的情况"""
        self.excluded_tools.add(tool_name)
        
        # 如果有可用的降级处理函数
        if self.fallback_handler:
            return RecoveryAction(
                action="escalate",
                feedback_message=f"多次尝试调用工具 '{tool_name}' 失败，将升级到人工处理。",
                excluded_tools=[tool_name]
            )
        
        # 否则使用 fallback 策略
        return RecoveryAction(
            action="fallback",
            feedback_message=f"多次尝试失败，已屏蔽工具 '{tool_name}'。建议基于已有信息直接回答，或使用其他工具。",
            excluded_tools=[tool_name],
            suggested_alternative=self._find_alternative_tool(tool_name, available_tools or [])
        )
    
    def _find_alternative_tool(self, failed_tool: str, 
                               available_tools: List[str]) -> Optional[str]:
        """寻找替代工具"""
        if not available_tools:
            return None
        
        # 简单的相似度匹配（基于前缀或子串）
        for tool in available_tools:
            if tool != failed_tool and tool not in self.excluded_tools:
                # 检查是否有共同前缀
                if tool.startswith(failed_tool[:3]) or failed_tool.startswith(tool[:3]):
                    return tool
        
        # 返回第一个可用的替代工具
        for tool in available_tools:
            if tool not in self.excluded_tools and tool != failed_tool:
                return tool
        
        return None
    
    def reset(self):
        """重置错误状态（开始新对话时调用）"""
        self.retry_count = 0
        self.error_history.clear()
        self.excluded_tools.clear()
    
    def get_excluded_tools(self) -> List[str]:
        """获取当前被屏蔽的工具列表"""
        return list(self.excluded_tools)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要统计"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        for record in self.error_history:
            error_type = record.error_type.name
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "retry_count": self.retry_count,
            "excluded_tools": list(self.excluded_tools),
            "error_breakdown": error_counts,
            "most_problematic_tool": max(
                set(r.tool_name for r in self.error_history),
                key=lambda t: sum(1 for r in self.error_history if r.tool_name == t)
            ) if self.error_history else None
        }
    
    def set_fallback_handler(self, handler: Callable):
        """设置降级处理函数"""
        self.fallback_handler = handler


class ReActAgentWithRecovery:
    """集成错误恢复的 ReAct Agent（示例）"""
    
    def __init__(self, agent, recovery_strategy: Optional[ErrorRecoveryStrategy] = None):
        self.agent = agent
        self.recovery = recovery_strategy or ErrorRecoveryStrategy()
        self.available_tools = []
    
    def execute_tool_with_recovery(self, tool_name: str, tool_input: Dict,
                                   tool_func: Callable, messages: list) -> Any:
        """
        带错误恢复的工具执行
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数
            tool_func: 工具函数
            messages: 当前对话历史
            
        Returns:
            工具执行结果或错误信息
        """
        try:
            # 尝试执行工具
            result = tool_func(**tool_input)
            # 成功后重置重试计数
            self.recovery.retry_count = 0
            return result
            
        except Exception as e:
            print(f"⚠️ 工具 '{tool_name}' 执行出错: {e}")
            
            # 获取可用工具列表
            if hasattr(self.agent, 'tool_executor'):
                self.available_tools = list(self.agent.tool_executor.tools.keys())
            
            # 调用错误恢复策略
            action = self.recovery.on_tool_error(e, tool_name, messages, self.available_tools)
            
            print(f"🔧 恢复策略: {action.action}")
            print(f"💡 提示: {action.feedback_message}")
            
            if action.suggested_alternative:
                print(f"🔄 建议替代工具: {action.suggested_alternative}")
            
            # 等待（如果需要）
            if action.wait_time > 0:
                import time
                time.sleep(action.wait_time)
            
            # 根据动作类型处理
            if action.action == "retry":
                # 返回错误信息，让 Agent 重试
                return f"错误: {e}\n提示: {action.feedback_message}"
            
            elif action.action == "fallback":
                # 降级处理：直接返回降级提示
                return f"[FALLBACK] {action.feedback_message}"
            
            elif action.action == "escalate":
                # 升级到人工处理
                if self.recovery.fallback_handler:
                    return self.recovery.fallback_handler(tool_name, tool_input, e)
                return f"[ESCALATE] {action.feedback_message}"
            
            elif action.action == "skip":
                # 跳过此工具
                return f"[SKIPPED] 工具 '{tool_name}' 已被跳过。"
            
            else:
                return f"错误: {e}"
    
    def get_filtered_tools(self, tools: List[Dict]) -> List[Dict]:
        """过滤掉被屏蔽的工具"""
        excluded = set(self.recovery.get_excluded_tools())
        return [t for t in tools if t.get("function", {}).get("name") not in excluded]


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    # 创建错误恢复策略
    recovery = ErrorRecoveryStrategy(max_retries=3)
    
    # 设置降级处理函数
    def fallback_handler(tool_name, tool_input, error):
        return f"[人工介入] 工具 {tool_name} 持续失败，需要人工处理。"
    
    recovery.set_fallback_handler(fallback_handler)
    
    # 模拟错误处理
    try:
        # 模拟工具调用失败
        raise Exception("Tool not found: search_data")
    except Exception as e:
        action = recovery.on_tool_error(
            e, 
            "search_data",
            messages=[{"role": "user", "content": "搜索天气"}],
            available_tools=["search", "calculate", "weather_lookup"]
        )
        print(f"动作: {action.action}")
        print(f"反馈: {action.feedback_message}")
        print(f"替代工具: {action.suggested_alternative}")
    
    # 获取错误摘要
    summary = recovery.get_error_summary()
    print(f"\n错误摘要: {summary}")


if __name__ == "__main__":
    example_usage()
