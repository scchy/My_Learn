"""
Pi Agent Reference Implementation
==================================
A minimal but correct Python implementation of a coding agent,
covering all core mechanisms of the Pi Agent.

Modules:
- llm.py    : Unified LLM client with Delta-based streaming
- tools.py  : Tool registration, schema generation, safe execution
- agent.py  : ReAct agent loop with steering & parallel tools
- context.py: Three-tier context compression
- session.py: Session persistence with tree structure
- cli.py    : CLI interface (typer + rich)
"""

from pi_agent_reference.llm import LLMClient, Message, Delta, LLMError
from pi_agent_reference.tools import ToolRegistry, ToolResult
from pi_agent_reference.agent import Agent, AgentConfig
from pi_agent_reference.context import ContextCompressor, CompressionConfig
from pi_agent_reference.session import SessionStore, SessionNode

__all__ = [
    "LLMClient", "Message", "Delta", "LLMError",
    "ToolRegistry", "ToolResult",
    "Agent", "AgentConfig",
    "ContextCompressor", "CompressionConfig",
    "SessionStore", "SessionNode",
]
