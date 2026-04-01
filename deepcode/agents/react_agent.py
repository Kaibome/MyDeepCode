#!/usr/bin/env python
# coding: utf-8
"""
Agent layer built on LangChain / LangGraph.

Mirrors the original deepcode agent hierarchy:
  BaseAgent  – LLM + MCP tool management
  ReActAgent – tool-calling loop (ReAct pattern)
  ChatAgent  – simple single-turn LLM chat
"""

import json
import logging
from typing import Dict, AsyncIterator, Any, List, Union, Optional

from pydantic import BaseModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from deepcode.utils.mcp_tool_manager import MCPToolManager, create_mcp_tools_from_config

logger = logging.getLogger(__name__)


def _create_chat_model(
    model_provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """Instantiate the appropriate LangChain chat model."""
    provider = (model_provider or "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        extra: Dict[str, Any] = {}
        if api_base:
            extra["base_url"] = api_base
        if api_key:
            extra["api_key"] = api_key
        return ChatOpenAI(model=model_name, **extra, **kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        extra = {}
        if api_key:
            extra["api_key"] = api_key
        return ChatAnthropic(model=model_name, **extra, **kwargs)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        extra = {}
        if api_key:
            extra["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(model=model_name, **extra, **kwargs)

    else:
        from langchain_openai import ChatOpenAI

        extra = {}
        if api_base:
            extra["base_url"] = api_base
        if api_key:
            extra["api_key"] = api_key
        return ChatOpenAI(model=model_name, **extra, **kwargs)


class AgentConfig(BaseModel):
    name: str
    llm_config: dict
    system_prompt: str = "你是一个小助手。"
    server_names: Optional[List[str]] = None


class BaseAgent:
    """
    Base agent wrapping a LangChain chat model and MCP tools.

    Replaces the original openjiuwen-based BaseAgent.
    """

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.name = agent_config.name
        self.system_prompt = agent_config.system_prompt

        llm_cfg = dict(agent_config.llm_config)
        self.model_name = llm_cfg.pop("model_name", None)
        model_provider = llm_cfg.pop("model_provider", "openai")
        api_key = llm_cfg.pop("api_key", None)
        api_base = llm_cfg.pop("api_base", None)

        self._llm: BaseChatModel = _create_chat_model(
            model_provider=model_provider,
            model_name=self.model_name or "gpt-4o",
            api_key=api_key,
            api_base=api_base,
        )

        self._tools: List[BaseTool] = []
        self._tool_names: List[str] = []
        self._mcp_managers: List[MCPToolManager] = []

    def _log_llm_request(
        self,
        messages: List[BaseMessage],
        tools: Optional[List] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "event": "llm_request",
            "agent_name": self.name,
            "model_name": self.model_name or "gpt-4o",
            "messages": [_message_to_dict(m) for m in messages],
            "tools": _serialise_tools(tools),
        }
        if extra:
            payload["extra"] = extra
        logger.info(
            f"Start call llm request: {json.dumps(payload, ensure_ascii=False, default=str)}"
        )

    def _log_llm_response(
        self,
        response: AIMessage,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "event": "llm_response",
            "agent_name": self.name,
            "model_name": self.model_name or "gpt-4o",
            "response": _message_to_dict(response),
        }
        if extra:
            payload["extra"] = extra
        logger.info(
            f"End call llm, response: {json.dumps(payload, ensure_ascii=False, default=str)}"
        )

    async def add_mcp_servers(
        self,
        server_configs: List[Dict[str, Any]],
    ):
        """
        Connect to MCP tool servers and register their tools.

        Each entry in *server_configs* should have keys:
          server_name, command, args, env (optional), client_type (ignored).
        """
        for cfg in server_configs:
            server_name = cfg.get("server_name", "unknown")
            command = cfg.get("command") or cfg.get("params", {}).get("command", "python")
            args = cfg.get("args") or cfg.get("params", {}).get("args", [])
            env = cfg.get("env") or cfg.get("params", {}).get("env")

            # Handle StdioServerParameters-like objects
            params = cfg.get("params")
            if params and hasattr(params, "command"):
                command = params.command
                args = list(params.args) if params.args else []
                env = dict(params.env) if params.env else None

            mgr = await create_mcp_tools_from_config(
                server_name=server_name,
                command=command,
                args=args,
                env=env,
            )
            self._mcp_managers.append(mgr)
            new_tools = mgr.get_tools()
            self._tools.extend(new_tools)
            self._tool_names.extend([t.name for t in new_tools])
            logger.info(
                f"[{self.name}] Added {len(new_tools)} tools from MCP server '{server_name}'"
            )

        if not self._tools:
            raise ValueError("No tools loaded from any MCP server")

    # Legacy compatibility alias
    async def add_mcps(self, mcp_configs):
        """Compatibility wrapper accepting ToolServerConfig-like objects."""
        converted = []
        for cfg in mcp_configs:
            if hasattr(cfg, "server_name"):
                converted.append({
                    "server_name": cfg.server_name,
                    "params": cfg.params,
                    "client_type": getattr(cfg, "client_type", "stdio"),
                })
            else:
                converted.append(cfg)
        await self.add_mcp_servers(converted)

    async def execute_mcp_tool(self, tool_name: str, inputs: dict) -> Any:
        """Invoke a loaded MCP tool by name."""
        for tool in self._tools:
            if tool.name == tool_name:
                return await tool.ainvoke(inputs)
        raise ValueError(f"{tool_name} tool not found")

    def call_llm(
        self,
        model_name: Optional[str],
        messages: Union[List[BaseMessage], List[Dict], str],
        tools: Optional[List] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """
        Synchronous LLM invocation (mirrors the original API).

        *tools* can be a list of LangChain ``BaseTool`` objects or
        OpenAI-format tool dicts.
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        elif messages and isinstance(messages[0], dict):
            messages = _dicts_to_messages(messages)

        llm = self._llm
        lc_tools = None
        if tools:
            lc_tools = _normalise_tools(tools)
            llm = llm.bind_tools(lc_tools)

        self._log_llm_request(
            messages=messages,
            tools=lc_tools,
            extra={
                "temperature": temperature,
                "top_p": top_p,
                "kwargs": kwargs,
            },
        )
        response = llm.invoke(messages)
        self._log_llm_response(response)
        return response

    async def cleanup(self):
        """Disconnect all MCP servers."""
        for mgr in self._mcp_managers:
            await mgr.disconnect()
        self._mcp_managers.clear()
        self._tools.clear()
        self._tool_names.clear()


class ReActAgent(BaseAgent):
    """
    ReAct agent: iterates tool calls until the LLM produces a final answer.
    """

    async def ainvoke(self, inputs: Dict, runtime=None) -> Dict:
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        query = inputs.get("query", "")
        if not query:
            raise ValueError("No query provided")

        messages.append(HumanMessage(content=query))

        llm = self._llm
        if self._tools:
            llm = llm.bind_tools(self._tools)

        self._log_llm_request(
            messages=messages,
            tools=self._tools,
            extra={"stage": "initial"},
        )
        response: AIMessage = llm.invoke(messages)
        self._log_llm_response(response, extra={"stage": "initial"})
        messages.append(response)

        while response.tool_calls:
            for tc in response.tool_calls:
                tc_args = tc["args"] if isinstance(tc, dict) else tc.args
                tc_name = tc["name"] if isinstance(tc, dict) else tc.name
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

                result = await self.execute_mcp_tool(tc_name, tc_args)
                messages.append(
                    ToolMessage(tool_call_id=tc_id, content=str(result))
                )

            self._log_llm_request(
                messages=messages,
                tools=self._tools,
                extra={"stage": "tool_followup"},
            )
            response = llm.invoke(messages)
            self._log_llm_response(response, extra={"stage": "tool_followup"})
            messages.append(response)

        return {
            "content": response.content,
            "history": self._format_tool_use_response(messages),
        }

    @staticmethod
    def _format_tool_use_response(messages: List[BaseMessage]) -> str:
        response = ["# 当前工具调用历史："]
        tool_call_history: Dict[str, Dict] = {}

        for message in messages[:-1]:
            if isinstance(message, AIMessage) and message.tool_calls:
                for tc in message.tool_calls:
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    tool_call_history[tc_id] = {
                        "tool_name": tc_name,
                        "arguments": json.dumps(tc_args, ensure_ascii=False),
                    }
            elif isinstance(message, ToolMessage):
                tid = message.tool_call_id
                if tid in tool_call_history:
                    tool_call_history[tid]["result"] = message.content

        idx = 1
        for tc_id, info in tool_call_history.items():
            tool_name = info["tool_name"]
            arguments = info["arguments"]
            result = info.get("result", "")
            response.append(
                f"{idx}. 调用工具：{tool_name}， 工具参数{arguments}\n工具结果：{result}"
            )
            idx += 1

        if messages and isinstance(messages[-1], AIMessage):
            content = messages[-1].content
            response.append(f"\n\n# Agent最终回复\n{content}")

        return "\n".join(response)


class ChatAgent(BaseAgent):
    """Simple single-turn chat agent (no tool calling)."""

    async def ainvoke(self, inputs: Dict, runtime=None) -> Dict:
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        query = inputs.get("query", "")
        if not query:
            raise ValueError("No query provided")
        messages.append(HumanMessage(content=query))

        self._log_llm_request(messages=messages, extra={"stage": "chat"})
        response: AIMessage = await self._llm.ainvoke(messages)
        self._log_llm_response(response, extra={"stage": "chat"})
        return {"content": response.content}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dicts_to_messages(dicts: List[Dict]) -> List[BaseMessage]:
    """Convert a list of ``{"role": ..., "content": ...}`` dicts to messages."""
    mapping = {
        "system": SystemMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "assistant": AIMessage,
        "ai": AIMessage,
        "tool": ToolMessage,
    }
    result = []
    for d in dicts:
        role = d.get("role", "user")
        cls = mapping.get(role, HumanMessage)
        if cls is ToolMessage:
            result.append(cls(content=d.get("content", ""), tool_call_id=d.get("tool_call_id", "")))
        else:
            result.append(cls(content=d.get("content", "")))
    return result


def _normalise_tools(tools: List) -> List:
    """
    Accept either LangChain BaseTool objects or OpenAI-format dicts
    and return a list suitable for ``model.bind_tools()``.
    """
    if not tools:
        return []
    if isinstance(tools[0], BaseTool):
        return tools
    # OpenAI-format dicts are already accepted by bind_tools
    return tools


def _message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert LangChain messages into JSON-serialisable dicts."""
    role = "user"
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"

    result: Dict[str, Any] = {
        "type": message.__class__.__name__,
        "role": role,
        "content": message.content,
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
    }

    if isinstance(message, AIMessage):
        result["tool_calls"] = message.tool_calls or []
    if isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id

    return result


def _serialise_tools(tools: Optional[List]) -> List[Dict[str, Any]]:
    """Convert tools list into JSON-serialisable metadata."""
    if not tools:
        return []

    serialised: List[Dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            serialised.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                }
            )
        elif isinstance(tool, dict):
            serialised.append(tool)
        else:
            serialised.append({"value": str(tool)})
    return serialised
