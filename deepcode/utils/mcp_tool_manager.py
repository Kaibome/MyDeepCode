"""
Centralized MCP tool management using langchain-mcp-adapters.

``MCPToolManager`` remains an async context manager for app lifecycle; internally
``MultiServerMCPClient`` (langchain-mcp-adapters >= 0.1.0) is used via
``await client.get_tools()`` rather than ``async with`` on the client.
"""

import os
import logging
from typing import Dict, List, Optional, Any

import yaml
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def load_mcp_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load MCP server definitions from the project config file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, returning empty MCP config")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config.get("mcp", {}).get("servers", {})


def build_server_params(
    server_name: str,
    server_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a single server config dict into the format expected by
    ``MultiServerMCPClient`` (``StdioConnection`` kwargs).
    """
    return {
        "command": server_cfg.get("command", "python"),
        "args": server_cfg.get("args", []),
        "env": {**os.environ, **(server_cfg.get("env") or {})},
    }


class MCPToolManager:
    """
    Manages MCP stdio server connections and exposes them as LangChain tools.

    Usage::

        async with MCPToolManager(server_names=["filesystem", "brave"]) as mgr:
            tools = mgr.get_tools()
            # pass *tools* to a LangChain/LangGraph agent
    """

    def __init__(
        self,
        server_names: Optional[List[str]] = None,
        config_path: str = "config.yaml",
        extra_servers: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._config_path = config_path
        self._requested_names = server_names
        self._extra_servers = extra_servers or {}
        self._client: Optional[MultiServerMCPClient] = None
        self._tools: List[BaseTool] = []

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """Start MCP server processes and retrieve their tool lists."""
        all_server_cfgs = load_mcp_config(self._config_path)
        all_server_cfgs.update(self._extra_servers)

        if self._requested_names is not None:
            selected = {
                name: cfg
                for name, cfg in all_server_cfgs.items()
                if name in self._requested_names
            }
        else:
            selected = all_server_cfgs

        if not selected:
            logger.warning("No MCP servers selected – tool list will be empty")
            return

        connection_dict: Dict[str, Any] = {}
        for name, cfg in selected.items():
            connection_dict[name] = {
                "transport": "stdio",
                **build_server_params(name, cfg),
            }

        self._client = MultiServerMCPClient(connection_dict)
        self._tools = await self._client.get_tools()
        logger.info(
            f"MCPToolManager connected {len(selected)} servers, "
            f"{len(self._tools)} tools available"
        )

    async def disconnect(self):
        """Release references; MultiServerMCPClient no longer uses a client-level async exit."""
        if self._client is not None:
            self._client = None
            self._tools = []

    def get_tools(self, tool_names: Optional[List[str]] = None) -> List[BaseTool]:
        """
        Return loaded LangChain tools, optionally filtered by name.

        Args:
            tool_names: If provided, only return tools whose names are in
                        this list.
        """
        if tool_names is None:
            return list(self._tools)
        name_set = set(tool_names)
        return [t for t in self._tools if t.name in name_set]

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Return a single tool by name, or ``None``."""
        for t in self._tools:
            if t.name == tool_name:
                return t
        return None

    @property
    def tool_names(self) -> List[str]:
        return [t.name for t in self._tools]


async def create_mcp_tools_from_config(
    server_name: str,
    command: str,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
) -> "MCPToolManager":
    """
    Convenience helper: create an MCPToolManager for a single dynamically-
    defined server (e.g. a tool path from an env-var).  Caller is responsible
    for calling ``await mgr.disconnect()`` when done.
    """
    extra = {
        server_name: {
            "command": command,
            "args": args,
            "env": env or {},
        }
    }
    mgr = MCPToolManager(
        server_names=[server_name],
        extra_servers=extra,
    )
    await mgr.connect()
    return mgr
