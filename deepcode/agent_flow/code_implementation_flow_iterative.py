#!/usr/bin/env python
# coding: utf-8
"""
Code Implementation Flow - Iterative Dialogue Architecture

Built on LangChain / LangGraph, replacing the openjiuwen-based implementation.

Adopts an iterative dialogue pattern where the LLM autonomously drives
the implementation process:
  - LLM Decision: LLM controls the implementation pace and order
  - Tool Execution: Controller passively executes tool calls
  - Feedback Loop: detailed feedback after each tool call
  - Memory Optimization: token-count-based intelligent memory management

Key features:
  1. LLM-driven: Agent decides next action autonomously
  2. Tool feedback: guidance feedback after each tool call
  3. Progress tracking: real-time file implementation progress
  4. Loop detection: prevents getting stuck in analysis loops
  5. Memory compression: automatic dialogue history compression
"""

import asyncio
import os
import time
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from deepcode.agents.react_agent import ReActAgent, AgentConfig
from deepcode.prompts.sys_prompts import (
    PURE_CODE_IMPLEMENTATION_SYSTEM_PROMPT_INDEX,
)

_PROJECT_ENV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)
load_dotenv(dotenv_path=_PROJECT_ENV_PATH)
logger = logging.getLogger(__name__)


# ============================================================
# Part 1: Progress Tracking Agent
# ============================================================

class Plan:
    """
    File implementation progress tracker.

    Tracks implemented files, detects analysis loops, manages technical
    decision records, and provides progress statistics.
    """

    TRACKED_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
        ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx",
        ".go", ".rs", ".php", ".rb", ".pl", ".lua",
        ".r", ".kt", ".scala", ".vue",
        ".html", ".css", ".scss", ".sass", ".less",
        ".json", ".yaml", ".yml", ".toml", ".xml",
        ".ini", ".cfg", ".env",
        ".md", ".rst", ".txt",
        ".sh", ".bash", ".zsh", ".bat", ".ps1", ".cmd",
        ".sql", ".db", ".dockerfile", ".gitignore",
        ".lock", ".sum", ".mod",
    }

    EXCLUDED_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
        ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
    }

    EXCLUDED_DIRECTORIES = {
        "__pycache__", ".pyc", "node_modules", ".git",
        ".vscode", ".idea", "dist", "build", "output",
        ".egg-info", "venv", ".venv", "env", ".env",
        "target", "bin", "obj", ".next", ".nuxt",
    }

    def __init__(self, mcp_agent, allow_read_ops: bool = True):
        self.mcp_agent = mcp_agent
        self.allow_read_ops = allow_read_ops

        self.completed_files: List[Dict[str, Any]] = []
        self.files_count = 0
        self.unique_files: set = set()
        self.file_summaries: Dict[str, str] = {}

        self.planned_files: List[str] = []
        self.implemented_files: set = set()

        self.implementation_summary: Dict[str, Any] = {
            "completed_files": self.completed_files,
        }

        self.recent_tool_calls: List[str] = []
        self.max_read_streak = 5
        self.max_read_without_write = 5

        self.tech_decisions: List[Dict] = []
        self.constraints: List[Dict] = []
        self.architecture_notes: List[Dict] = []

    async def process_tool_execution(self, tool_calls: List[Dict]) -> List[Dict]:
        results = []
        read_tools = {"read_file", "read_code_mem", "list_files", "list_directory"}

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_input = tool_call.get("input", {})
            tool_call_id = tool_call.get("id", f"call_{len(results)}")

            base_tool_name = tool_name.split("-")[-1] if "-" in tool_name else tool_name
            logger.info(f"[Tracker] Processing tool: {tool_name}")
            self._track_tool_pattern(base_tool_name)

            try:
                if not self.allow_read_ops and base_tool_name in read_tools:
                    logger.warning(f"[Tracker] Read tool '{tool_name}' blocked")
                    results.append({
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "content": "[Access Denied] Read operations disabled. Use write_file.",
                        "is_error": False,
                    })
                    continue

                if base_tool_name == "read_file":
                    file_path = tool_input.get("file_path") or tool_input.get("path")
                    if file_path and file_path in self.file_summaries:
                        summary = self.file_summaries[file_path]
                        results.append({
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "content": f"[Memory Optimized Content]\nSummary of {file_path}:\n{summary}",
                            "is_error": False,
                        })
                        continue

                if base_tool_name == "write_file" and "path" in tool_input and "file_path" not in tool_input:
                    tool_input["file_path"] = tool_input.pop("path")

                result = await self.mcp_agent.execute_mcp_tool(
                    tool_name=tool_name, inputs=tool_input
                )

                if base_tool_name == "write_file":
                    self._track_file_write(tool_input, result)

                results.append({
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "content": str(result),
                    "is_error": False,
                })

            except Exception as e:
                logger.error(f"[Tracker] Tool failed: {e}")
                results.append({
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "content": f"Error: {e}",
                    "is_error": True,
                })

        return results

    def set_planned_files(self, planned_files: List[str]):
        self.planned_files = planned_files

    def register_file_summary(self, file_path: str, summary: str):
        self.file_summaries[file_path] = summary

    def get_knowledge_base_text(self) -> str:
        if not self.file_summaries:
            return "No files implemented yet."
        kb = "## Implemented Files Knowledge Base:\n"
        for path, summary in self.file_summaries.items():
            kb += f"\n### {path}\n{summary}\n"
        return kb

    def _track_file_write(self, tool_input: Dict, result: Any):
        file_path = tool_input.get("path", "") or tool_input.get("file_path", "")
        if file_path and file_path not in self.unique_files:
            self.files_count += 1
            self.unique_files.add(file_path)
            self.implemented_files.add(self._normalize_path(file_path))
            content_length = len(str(tool_input.get("content", "")))
            self.completed_files.append({
                "file": file_path,
                "timestamp": time.time(),
                "content_length": content_length,
                "iteration": self.files_count,
                "size": content_length,
            })
            logger.info(f"[Tracker] File implemented: {file_path} (Total: {self.files_count})")

    def _track_tool_pattern(self, tool_name: str):
        self.recent_tool_calls.append(tool_name)
        max_limit = max(self.max_read_streak, self.max_read_without_write)
        if len(self.recent_tool_calls) > max_limit:
            self.recent_tool_calls.pop(0)

    def is_stuck_in_analysis(self) -> bool:
        read_tools = {"read_file", "read_code_mem", "list_files", "list_directory", "search_code_references"}
        recent = self.recent_tool_calls[-self.max_read_streak:]
        if len(recent) >= self.max_read_streak:
            return set(recent).issubset(read_tools)
        return False

    def _normalize_path(self, file_path: str) -> str:
        return file_path.replace("\\", "/").strip("/").lower()

    def _fuzzy_match_file(self, target_file: str, implemented_file: str) -> bool:
        t_name = target_file.split("/")[-1]
        i_name = implemented_file.split("/")[-1]
        if t_name == i_name:
            return True
        if target_file in implemented_file or implemented_file in target_file:
            return True
        if implemented_file.endswith(target_file) or target_file.endswith(implemented_file):
            return True
        return False

    def check_implementation_complete(self) -> tuple:
        if not self.planned_files:
            return False, []
        normalized = [self._normalize_path(f) for f in self.planned_files]
        unimplemented = []
        for planned in normalized:
            if not any(self._fuzzy_match_file(planned, impl) for impl in self.implemented_files):
                unimplemented.append(planned)
        return len(unimplemented) == 0, unimplemented

    def scan_generated_files(self, output_dir: str) -> set:
        generated = set()
        if not os.path.exists(output_dir):
            return generated
        for root, dirs, files in os.walk(output_dir):
            dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRECTORIES]
            for f in files:
                ext = "." + f.split(".")[-1] if "." in f else ""
                if ext in self.EXCLUDED_EXTENSIONS:
                    continue
                if ext in self.TRACKED_EXTENSIONS or f in [".gitignore", "Dockerfile"]:
                    rel = os.path.relpath(os.path.join(root, f), output_dir)
                    generated.add(self._normalize_path(rel))
        return generated

    def check_completion_by_directory_scan(self, output_dir: str) -> tuple:
        if not self.planned_files:
            return False, [], 0
        generated = self.scan_generated_files(output_dir)
        normalized = [self._normalize_path(f) for f in self.planned_files]
        unimplemented = [
            p for p in normalized
            if not any(self._fuzzy_match_file(p, g) for g in generated)
        ]
        return len(unimplemented) == 0, unimplemented, len(generated)

    def get_loop_break_guidance(self) -> str:
        recent = self.recent_tool_calls[-self.max_read_streak:]
        return (
            f"ANALYSIS LOOP DETECTED - ACTION REQUIRED\n\n"
            f"Problem: You've been analyzing for {len(recent)} consecutive ops.\n"
            f"Recent: {' -> '.join(recent)}\n\n"
            f"SOLUTION: Use write_file NOW to create a new code file.\n"
            f"Files implemented: {self.files_count}\n"
            f"CRITICAL: Your next response MUST use write_file!"
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_files": self.files_count,
            "unique_files_count": len(self.unique_files),
            "files_implemented_count": self.files_count,
            "completed_files": [f["file"] for f in self.completed_files],
            "recent_tool_pattern": self.recent_tool_calls,
            "tech_decisions_count": len(self.tech_decisions),
            "is_stuck": self.is_stuck_in_analysis(),
            "completed_files_list": [f["file"] for f in self.completed_files],
        }

    def get_completed_files_list(self) -> List[str]:
        return [f["file"] for f in self.implementation_summary["completed_files"]]

    def reset_tracking(self):
        self.completed_files = []
        self.files_count = 0
        self.unique_files = set()
        self.recent_tool_calls = []
        self.implemented_files = set()


# ============================================================
# Part 2: Memory Management
# ============================================================

class DialogueMemoryManager:
    """
    Dialogue history memory manager.

    Uses simple list-based state management (replacing ContextEngine).
    """

    def __init__(self, agent_id: str = "iterative_code_agent", progress_tracker=None):
        self.compression_count = 0
        self.progress_tracker = progress_tracker
        self.current_round_tool_results: List[Dict] = []
        self.last_write_file_success: bool = False
        self._stored_messages: List[Dict] = []

    def estimate_tokens(self, text: str) -> int:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return (chinese_chars // 2) + (other_chars // 4)

    def estimate_messages_tokens(self, messages: List[Dict]) -> int:
        return sum(self.estimate_tokens(str(m.get("content", ""))) for m in messages)

    def should_trigger_memory_optimization(self, messages: List[Dict], files_count: int) -> bool:
        if self.last_write_file_success:
            logger.info("[Memory] write_file success detected, triggering optimization")
            return True
        return False

    def should_trigger_emergency_compression(self, messages: List[Dict]) -> bool:
        return len(messages) > 50

    def apply_memory_optimization(
        self,
        system_message: str,
        messages: List[Dict],
        files_implemented_count: int,
    ) -> List[Dict]:
        if len(messages) < 2:
            return messages

        unimplemented = self._get_unimplemented_files()
        progress_summary = self._generate_progress_summary(files_implemented_count)

        initial_plan = ""
        for msg in messages[:3]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "Code Reproduction Plan" in content or "file_structure" in content.lower():
                    initial_plan = content
                    break
        if not initial_plan and len(messages) > 1:
            initial_plan = messages[1].get("content", "")

        tool_results_text = self._get_current_round_tool_results()
        knowledge_base = self._get_knowledge_base()

        new_messages = [
            {"role": "system", "content": f"{system_message}\n\n{progress_summary}"},
            {
                "role": "user",
                "content": (
                    f"{initial_plan}\n\n"
                    f"**当前未实现文件:**\n{unimplemented}\n\n"
                    f"{knowledge_base}\n\n"
                    f"**本轮工具执行结果:**\n{tool_results_text}\n\n"
                    f"**下一步行动:**\n"
                    f"Continue implementing the remaining files from the plan above.\n"
                    f"Use write_file to create the next file.\n"
                ),
            },
        ]

        self.compression_count += 1
        self._stored_messages = list(new_messages)

        logger.info(
            f"[Memory] Compressed {len(messages)} -> {len(new_messages)} messages"
        )
        return new_messages

    def _get_unimplemented_files(self) -> str:
        if not self.progress_tracker:
            return "Unable to determine"
        is_complete, unimplemented = self.progress_tracker.check_implementation_complete()
        if is_complete:
            return "All planned files implemented!"
        if not unimplemented:
            return "No planned files info available"
        file_list = "\n".join([f"  - {f}" for f in unimplemented[:20]])
        if len(unimplemented) > 20:
            file_list += f"\n  ... and {len(unimplemented) - 20} more"
        return f"Total: {len(unimplemented)} remaining\n{file_list}"

    def _generate_progress_summary(self, files_count: int) -> str:
        if not self.progress_tracker:
            return f"**Progress:** {files_count} files completed"
        total = len(self.progress_tracker.planned_files) if self.progress_tracker.planned_files else 0
        if total > 0:
            pct = (files_count / total) * 100
            return f"**Progress:** {files_count}/{total} files ({pct:.1f}%)"
        return f"**Progress:** {files_count} files completed"

    def _get_current_round_tool_results(self) -> str:
        if not self.current_round_tool_results:
            return "No tool results"
        parts = []
        for i, r in enumerate(self.current_round_tool_results, 1):
            content = r.get("content", "")
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            parts.append(f"{i}. {r.get('tool_name', 'unknown')}: {content}")
        return "\n".join(parts)

    def _get_knowledge_base(self) -> str:
        if not self.progress_tracker:
            return ""
        summaries = self.progress_tracker.file_summaries
        if not summaries:
            return "**Knowledge Base:** No files implemented yet."
        parts = ["**Below is the Knowledge Base of Implemented Files:**"]
        for fp, summary in summaries.items():
            parts.append(f"### {fp}\n```\n{summary}\n```\n")
        return "\n".join(parts)

    def record_tool_result(self, tool_name: str, content: str):
        self.current_round_tool_results.append({"tool_name": tool_name, "content": content})
        if len(self.current_round_tool_results) > 5:
            self.current_round_tool_results = self.current_round_tool_results[-5:]

    def clear_tool_results(self):
        self.current_round_tool_results = []
        self.last_write_file_success = False

    def mark_write_file_success(self):
        self.last_write_file_success = True

    def store_message(self, message: Dict):
        self._stored_messages.append(message)

    def get_stored_messages(self, num: int = -1) -> List[Dict]:
        if num == -1:
            return list(self._stored_messages)
        return list(self._stored_messages[-num:])

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_compressions": self.compression_count,
            "stored_messages_count": len(self._stored_messages),
            "context_engine_enabled": False,
        }


# ============================================================
# Part 3: JSON Repair Utility
# ============================================================

class AdvancedJsonRepairer:
    @staticmethod
    def fix_malformed_json(json_text: str, tool_identifier: str = "") -> Dict:
        cleaned = json_text.strip()
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        repaired = AdvancedJsonRepairer._auto_close_brackets(cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        if tool_identifier == "write_file":
            return AdvancedJsonRepairer._fix_write_file_json(repaired)

        return {"error": "JSON parsing failed", "raw_content": json_text}

    @staticmethod
    def _auto_close_brackets(text: str) -> str:
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        quote_count = text.count('"')
        result = text
        if quote_count % 2 != 0:
            result += '"'
        result += ']' * max(open_brackets, 0)
        result += '}' * max(open_braces, 0)
        return result

    @staticmethod
    def _fix_write_file_json(text: str) -> Dict:
        try:
            data = json.loads(text)
        except Exception:
            data = {}
        if "path" not in data:
            m = re.search(r'"path"\s*:\s*"([^"]+)"', text)
            data["path"] = m.group(1) if m else "unknown.py"
        if "content" not in data:
            m = re.search(r'"content"\s*:\s*"([^"]+)', text, re.DOTALL)
            data["content"] = m.group(1) if m else "# Content extraction failed"
        return data


# ============================================================
# Part 4: Feedback Generator
# ============================================================

class IterativeFeedbackGenerator:
    @staticmethod
    def generate_success_feedback(files_count: int) -> str:
        return (
            f"File implementation completed successfully!\n"
            f"Progress: {files_count} files implemented\n"
            f"Next: Check if ALL files from the plan are implemented.\n"
            f"If ALL done: test and say 'implementation complete'.\n"
            f"If MORE needed: continue with write_file."
        )

    @staticmethod
    def generate_error_feedback() -> str:
        return (
            "Error detected during implementation.\n"
            "Review the error, fix the issue, and continue."
        )

    @staticmethod
    def generate_no_tools_warning(files_count: int) -> str:
        return (
            f"No tool calls detected!\n"
            f"Progress: {files_count} files implemented\n"
            f"You MUST use tools. Use write_file to implement the next file.\n"
            f"CRITICAL: Your next response MUST include tool calls!"
        )

    @staticmethod
    def compile_feedback(tool_results: List[Dict], guidance: str) -> str:
        parts = []
        if tool_results:
            parts.append("## Tool Execution Results:\n")
            for r in tool_results:
                icon = "ERROR" if r.get("is_error") else "OK"
                parts.append(f"### [{icon}] {r.get('tool_name', 'unknown')}")
                parts.append(f"```\n{r.get('content', '')}\n```\n")
        if guidance:
            parts.append(guidance)
        return "\n\n".join(parts)

    @staticmethod
    def check_for_errors(tool_results: List[Dict]) -> bool:
        for r in tool_results:
            if r.get("is_error"):
                return True
            content = str(r.get("content", "")).lower()
            if any(kw in content for kw in ("error", "failed", "exception")):
                return True
        return False


# ============================================================
# Part 5: Tool Filter
# ============================================================

class EssentialToolFilter:
    CORE_TOOLS = {"write_file", "search_code_references"}

    @staticmethod
    def filter_tool_definitions(all_tools: List, logger_instance) -> List:
        filtered = []
        for tool in all_tools:
            name = tool.name if hasattr(tool, "name") else tool.get("name", "")
            base_name = name.split("-")[-1] if name else ""
            if base_name in EssentialToolFilter.CORE_TOOLS or name in EssentialToolFilter.CORE_TOOLS:
                filtered.append(tool)
        logger_instance.info(f"Tool filtering: {len(filtered)}/{len(all_tools)} active")
        return filtered


# ============================================================
# Part 6: Iterative Controller
# ============================================================

class IterativeCodeFlow:
    def __init__(self):
        self._llm_config = {
            "model_provider": os.getenv("LLM_MODEL_PROVIDER"),
            "api_key": os.getenv("LLM_API_KEY"),
            "api_base": os.getenv("LLM_API_BASE"),
            "model_name": os.getenv("LLM_MODEL_NAME"),
        }
        self._agent: Optional[ReActAgent] = None
        self._progress_tracker: Optional[Plan] = None
        self._memory_manager: Optional[DialogueMemoryManager] = None
        self._feedback_generator = IterativeFeedbackGenerator()
        self._json_repairer = AdvancedJsonRepairer()
        self._allow_read_ops: bool = True

        self._completion_claim_count: int = 0
        self._max_completion_claims: int = 5
        self._output_dir: Optional[str] = None

    @staticmethod
    def _resolve_tool_script_path(env_value: Optional[str], default_filename: str) -> Optional[str]:
        """
        Resolve MCP tool script path from env with robust fallbacks:
        1) explicit env absolute/relative
        2) package-local tools directory
        3) workspace-root deepcode/tools
        """
        candidate_paths: List[str] = []
        if env_value:
            candidate_paths.append(env_value)

        package_tools_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "tools")
        )
        candidate_paths.append(os.path.join(package_tools_dir, default_filename))

        workspace_tools_dir = os.path.abspath(
            os.path.join(os.getcwd(), "deepcode", "tools")
        )
        candidate_paths.append(os.path.join(workspace_tools_dir, default_filename))

        for path in candidate_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        return None

    async def initialize(self):
        logger.info("[IterativeFlow] Initializing agent and tools…")

        self._agent = ReActAgent(
            AgentConfig(
                name="IterativeCodeAgent",
                llm_config=dict(self._llm_config),
                system_prompt=PURE_CODE_IMPLEMENTATION_SYSTEM_PROMPT_INDEX,
            )
        )

        mcp_servers = []
        impl_path = self._resolve_tool_script_path(
            os.getenv("CODE_IMPLEMENTATION_SERVER_PATH"),
            "code_implementation_server.py",
        )
        if impl_path:
            mcp_servers.append({
                "server_name": "code-implementation",
                "command": "python",
                "args": [impl_path],
            })
        else:
            logger.warning(
                "[IterativeFlow] code_implementation_server.py not found; "
                "set CODE_IMPLEMENTATION_SERVER_PATH in .env"
            )

        ref_path = self._resolve_tool_script_path(
            os.getenv("CODE_REFERENCE_INDEXER_PATH"),
            "code_reference_indexer.py",
        )
        if ref_path:
            mcp_servers.append({
                "server_name": "code-reference",
                "command": "python",
                "args": [ref_path],
            })
        else:
            logger.warning(
                "[IterativeFlow] code_reference_indexer.py not found; "
                "set CODE_REFERENCE_INDEXER_PATH in .env"
            )

        if mcp_servers:
            await self._agent.add_mcp_servers(mcp_servers)
            logger.info(
                f"[IterativeFlow] {len(self._agent._tools)} tools active"
            )
            logger.info(
                f"[IterativeFlow] Tool names: {[t.name for t in self._agent._tools]}"
            )
        else:
            raise RuntimeError(
                "No MCP servers available for IterativeCodeFlow. "
                "Please verify CODE_IMPLEMENTATION_SERVER_PATH and CODE_REFERENCE_INDEXER_PATH."
            )

        self._progress_tracker = Plan(self._agent, allow_read_ops=self._allow_read_ops)
        self._memory_manager = DialogueMemoryManager(
            agent_id="iterative_code_agent",
            progress_tracker=self._progress_tracker,
        )
        logger.info("[IterativeFlow] Agent initialized successfully")

    async def execute(
        self,
        plan_file_path: str,
        target_directory: Optional[str] = None,
        enable_read_tools: bool = True,
    ) -> Dict[str, Any]:
        self._allow_read_ops = enable_read_tools
        self._completion_claim_count = 0
        start_time = time.time()

        try:
            plan_content = Path(plan_file_path).read_text(encoding="utf-8")
            base_dir = target_directory or str(Path(plan_file_path).parent)
            output_dir = os.path.join(base_dir, "generate_code")
            self._output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

            logger.info("=" * 60)
            logger.info(f"[IterativeFlow] Plan: {plan_file_path}")
            logger.info(f"[IterativeFlow] Output: {output_dir}")

            await self._set_workspace(output_dir)

            planned_files = self._extract_files_from_plan(plan_content)
            self._progress_tracker.set_planned_files(planned_files)

            initial_message = self._build_initial_prompt(plan_content, output_dir)
            result = await self._iterative_loop(initial_message)

            elapsed = time.time() - start_time
            stats = self._progress_tracker.get_statistics()
            return {
                "status": "success",
                "output_directory": output_dir,
                "statistics": stats,
                "duration_seconds": round(elapsed, 2),
                "result": result,
            }
        except Exception as e:
            logger.error(f"[IterativeFlow] Execution failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        finally:
            await self._cleanup_resources()

    @staticmethod
    def _strip_assistant_tool_calls_when_no_tools_executed(msg: AIMessage) -> AIMessage:
        """
        OpenAI-compatible APIs require: an assistant message with ``tool_calls`` must be
        immediately followed by tool messages for each ``tool_call_id``. If this round
        does not execute tools, strip tool-call metadata so a following ``HumanMessage``
        (e.g. nudge/warning) does not trigger HTTP 400.
        """
        extra = getattr(msg, "additional_kwargs", None) or {}
        has_extra = "tool_calls" in extra or "function_call" in extra
        if not msg.tool_calls and not msg.invalid_tool_calls and not has_extra:
            return msg
        add_kw = dict(extra)
        add_kw.pop("tool_calls", None)
        add_kw.pop("function_call", None)
        return msg.model_copy(
            update={
                "tool_calls": [],
                "invalid_tool_calls": [],
                "additional_kwargs": add_kw,
            }
        )

    async def _iterative_loop(self, initial_message: str) -> str:
        max_iterations = 800
        max_time = 7200
        iteration = 0
        start_time = time.time()
        no_tool_rounds = 0
        max_no_tool_rounds = 8

        messages: List[BaseMessage] = [
            SystemMessage(content=self._agent.system_prompt),
            HumanMessage(content=initial_message),
        ]

        llm = self._agent._llm
        if self._agent._tools:
            llm = llm.bind_tools(self._agent._tools)

        while iteration < max_iterations:
            iteration += 1
            elapsed = time.time() - start_time
            if elapsed > max_time:
                logger.warning(f"[IterativeFlow] Time limit ({max_time}s) reached")
                break

            logger.info(f"[IterativeFlow] Iteration {iteration}/{max_iterations}")

            # 1. Call LLM with retry
            response = None
            retry_delay = 1
            for attempt in range(3):
                try:
                    response = llm.invoke(messages)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"[IterativeFlow] LLM call failed (attempt {attempt+1}): {e}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"[IterativeFlow] LLM call failed after 3 attempts: {e}")

            if response is None:
                break

            response_text = response.content or "[Empty response]"
            messages.append(response)
            self._memory_manager.store_message({"role": "assistant", "content": response_text})

            if self._check_completion_signal(response_text):
                logger.info("[IterativeFlow] Implementation complete signal detected")
                break

            # Extract tool calls
            tool_calls_raw = response.tool_calls if hasattr(response, "tool_calls") else []
            tool_calls = []
            if tool_calls_raw:
                for tc in tool_calls_raw:
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    tc_id = tc.get("id", f"call_{len(tool_calls)}") if isinstance(tc, dict) else getattr(tc, "id", f"call_{len(tool_calls)}")

                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            tc_args = AdvancedJsonRepairer.fix_malformed_json(tc_args, tc_name)

                    tool_calls.append({"id": tc_id, "name": tc_name, "input": tc_args})

            # If we will not run tools this round, remove any tool_calls left on the
            # assistant message — otherwise the next invoke() fails with 400.
            if not tool_calls and isinstance(messages[-1], AIMessage):
                stripped = self._strip_assistant_tool_calls_when_no_tools_executed(
                    messages[-1]
                )
                if stripped is not messages[-1]:
                    logger.warning(
                        "[IterativeFlow] Stripped assistant tool_calls (no tools executed "
                        "this round; required for valid chat history)."
                    )
                messages[-1] = stripped

            # 2. Process tool calls
            if tool_calls:
                tool_results = await self._progress_tracker.process_tool_execution(tool_calls)

                for r in tool_results:
                    self._memory_manager.record_tool_result(r.get("tool_name", ""), r.get("content", ""))

                write_success = False
                for i, call in enumerate(tool_calls):
                    if call.get("name") == "write_file" and not tool_results[i].get("is_error"):
                        write_success = True
                        self._memory_manager.mark_write_file_success()
                        fp = call.get("input", {}).get("file_path") or call.get("input", {}).get("path")
                        content = call.get("input", {}).get("content", "")
                        if fp and content:
                            summary = await self._generate_summary_with_llm(fp, content)
                            self._progress_tracker.register_file_summary(fp, summary)

                has_errors = self._feedback_generator.check_for_errors(tool_results)
                if has_errors:
                    guidance = self._feedback_generator.generate_error_feedback()
                else:
                    guidance = self._feedback_generator.generate_success_feedback(
                        self._progress_tracker.files_count
                    )

                user_feedback = self._feedback_generator.compile_feedback(tool_results, guidance)

                for r in tool_results:
                    messages.append(
                        ToolMessage(
                            tool_call_id=r.get("tool_call_id", ""),
                            content=r.get("content", ""),
                        )
                    )
                messages.append(HumanMessage(content=user_feedback))
                self._memory_manager.store_message({"role": "user", "content": user_feedback})

                # Memory management
                msgs_dict = self._convert_messages_to_dict(messages)
                fc = self._progress_tracker.files_count

                if self._memory_manager.should_trigger_memory_optimization(msgs_dict, fc):
                    msgs_dict = self._memory_manager.apply_memory_optimization(
                        self._agent.system_prompt, msgs_dict, fc
                    )
                    messages = self._convert_dict_to_messages(msgs_dict)
                    self._memory_manager.clear_tool_results()
                elif self._memory_manager.should_trigger_emergency_compression(msgs_dict):
                    msgs_dict = self._memory_manager.apply_memory_optimization(
                        self._agent.system_prompt, msgs_dict, fc
                    )
                    messages = self._convert_dict_to_messages(msgs_dict)
                    self._memory_manager.clear_tool_results()
            else:
                no_tool_rounds += 1
                warning = self._feedback_generator.generate_no_tools_warning(
                    self._progress_tracker.files_count
                )
                messages.append(HumanMessage(content=warning))
                self._memory_manager.store_message({"role": "user", "content": warning})
                logger.warning(
                    f"[IterativeFlow] No tool calls in this round ({no_tool_rounds}/{max_no_tool_rounds})"
                )

                if no_tool_rounds >= max_no_tool_rounds:
                    logger.error(
                        "[IterativeFlow] Stopping due to repeated no-tool responses. "
                        "Likely MCP tool loading or tool-calling incompatibility issue."
                    )
                    break

                msgs_dict = self._convert_messages_to_dict(messages)
                if self._memory_manager.should_trigger_emergency_compression(msgs_dict):
                    msgs_dict = self._memory_manager.apply_memory_optimization(
                        self._agent.system_prompt, msgs_dict, self._progress_tracker.files_count
                    )
                    messages = self._convert_dict_to_messages(msgs_dict)
                    self._memory_manager.clear_tool_results()
            if tool_calls:
                no_tool_rounds = 0

            # Loop detection
            if self._progress_tracker.is_stuck_in_analysis():
                guidance = self._progress_tracker.get_loop_break_guidance()
                messages.append(HumanMessage(content=guidance))

            # Completion check via directory scan
            if self._output_dir:
                is_complete, unimpl, gen_count = self._progress_tracker.check_completion_by_directory_scan(
                    self._output_dir
                )
                if is_complete:
                    logger.info(f"[IterativeFlow] All {gen_count} planned files found!")
                    break
                if iteration % 5 == 0 and unimpl:
                    messages.append(HumanMessage(content=self._generate_progress_reminder(unimpl)))
            else:
                if iteration % 5 == 0:
                    is_complete, unimpl = self._progress_tracker.check_implementation_complete()
                    if is_complete:
                        break
                    if unimpl:
                        messages.append(HumanMessage(content=self._generate_progress_reminder(unimpl)))

        return self._generate_final_report(iteration, time.time() - start_time)

    async def _generate_summary_with_llm(self, file_path: str, content: str) -> str:
        max_len = 12000
        snippet = content[:max_len] + "\n... (truncated)" if len(content) > max_len else content

        prompt = (
            f"You are a code analysis engine.\n"
            f"Target File: {file_path}\n\n"
            f"Analyze the code and generate a concise Interface Summary:\n"
            f"1. Class names and responsibilities\n"
            f"2. Public function signatures and explanations\n"
            f"3. Key global constants\n"
            f"4. No internal implementation details\n\n"
            f"Code:\n```\n{snippet}\n```\n\nOutput the summary directly."
        )

        for attempt in range(3):
            try:
                response = self._agent._llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return self._extract_code_summary_fallback(snippet, file_path)

    def _extract_code_summary_fallback(self, content: str, file_path: str) -> str:
        lines = content.split("\n")
        parts = [f"File: {file_path}"]
        for line in lines:
            s = line.strip()
            if s.startswith("class ") or (s.startswith("def ") and not s.startswith("def _")):
                parts.append(s)
        return "\n".join(parts[:20])

    def _convert_messages_to_dict(self, messages: List) -> List[Dict]:
        result = []
        for m in messages:
            if hasattr(m, "role") and hasattr(m, "content"):
                result.append({"role": getattr(m, "type", "user"), "content": m.content})
            elif isinstance(m, dict):
                result.append(m)
            else:
                role = "user"
                if isinstance(m, SystemMessage):
                    role = "system"
                elif isinstance(m, AIMessage):
                    role = "assistant"
                elif isinstance(m, ToolMessage):
                    role = "tool"
                result.append({"role": role, "content": getattr(m, "content", str(m))})
        return result

    def _convert_dict_to_messages(self, msgs: List[Dict]) -> List[BaseMessage]:
        result = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "tool":
                result.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
            else:
                result.append(HumanMessage(content=content))
        return result

    def _extract_files_from_plan(self, plan_content: str) -> List[str]:
        extensions_pattern = "|".join(
            ext.replace(".", r"\.") for ext in Plan.TRACKED_EXTENSIONS
        )
        files = []
        m = re.search(
            r'file_structure:\s*\|(.*?)(?=\n\n|\n[A-Z]|$)',
            plan_content,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            pattern = rf'[│├└─\s]*([\w/._-]+(?:{extensions_pattern}))'
            files.extend(re.findall(pattern, m.group(1)))

        if not files:
            pattern = rf'\b([\w/._-]+(?:{extensions_pattern}))\b'
            files.extend(re.findall(pattern, plan_content))

        unique = list(set(files))
        filtered = []
        for f in unique:
            if any(d in f for d in Plan.EXCLUDED_DIRECTORIES):
                continue
            ext = "." + f.split(".")[-1] if "." in f else ""
            if ext in Plan.EXCLUDED_EXTENSIONS:
                continue
            if f.startswith(("http://", "https://")):
                continue
            filtered.append(f)

        logger.info(f"[IterativeFlow] Extracted {len(filtered)} files from plan")
        return filtered

    def _build_initial_prompt(self, plan_content: str, output_dir: str) -> str:
        expected = len([
            l for l in plan_content.split("\n")
            if any(ext in l for ext in Plan.TRACKED_EXTENSIONS)
        ])
        return (
            f"**URGENT: Start Writing Code NOW**\n\n"
            f"**Code Reproduction Plan:**\n{plan_content}\n\n"
            f"**Working Directory:** {output_dir}\n\n"
            f"**IMMEDIATE ACTION REQUIRED:**\n"
            f"Do NOT explore or analyze. START WRITING CODE IMMEDIATELY!\n"
            f"1. START with the first file from the plan\n"
            f"2. Use `write_file` tool to create complete implementations\n"
            f"3. Move to next file after each one\n"
            f"4. Continue until ALL files are implemented\n\n"
            f"**Available Tools:**\n"
            f"- `write_file(file_path, content)`: USE THIS NOW\n"
            f"- `search_code_references`: Optional reference lookup\n\n"
            f"**RULES:**\n"
            f"- USE EXACT FILE PATHS from the plan\n"
            f"- NO file exploration\n"
            f"- Each iteration should write at least ONE file\n"
            f"- When all {expected} files are written, say 'implementation complete'\n\n"
            f"**START WRITING THE FIRST FILE NOW!**"
        )

    def _check_completion_signal(self, text: str) -> bool:
        if self._output_dir:
            is_complete, unimpl, gen_count = self._progress_tracker.check_completion_by_directory_scan(
                self._output_dir
            )
            if is_complete:
                return True
        else:
            is_complete, unimpl = self._progress_tracker.check_implementation_complete()
            if is_complete:
                return True

        text_lower = text.lower()
        signals = ["implementation complete", "all files implemented", "implementation finished"]
        if any(s in text_lower for s in signals):
            self._completion_claim_count += 1
            if unimpl:
                if self._completion_claim_count >= self._max_completion_claims:
                    logger.warning(
                        f"[IterativeFlow] Forced completion after {self._completion_claim_count} claims"
                    )
                    return True
                return False
            return True
        else:
            self._completion_claim_count = 0
        return False

    def _generate_progress_reminder(self, unimplemented: List[str]) -> str:
        total = len(self._progress_tracker.planned_files)
        done = len(self._progress_tracker.implemented_files)
        sample = unimplemented[:5]
        return (
            f"**Progress Check** {done}/{total} files ({done*100//total if total else 0}%)\n"
            f"Remaining: {len(unimplemented)}\n"
            + "\n".join(f"  - {f}" for f in sample)
            + ("\n  - ..." if len(unimplemented) > 5 else "")
            + "\nContinue implementing remaining files."
        )

    async def _set_workspace(self, output_dir: str):
        try:
            await self._agent.execute_mcp_tool(
                "code-implementation-set_workspace", {"workspace_path": output_dir}
            )
        except Exception as e:
            # Fallback: some MCP adapters may expose a slightly different prefixed name.
            try:
                tool_names = [t.name for t in self._agent._tools]
                matched = next((n for n in tool_names if n.endswith("set_workspace")), None)
                if matched:
                    await self._agent.execute_mcp_tool(
                        matched, {"workspace_path": output_dir}
                    )
                    logger.info(f"[IterativeFlow] Workspace setup via fallback tool: {matched}")
                    return
            except Exception:
                pass
            logger.warning(f"[IterativeFlow] Workspace setup failed: {e}")

    async def _cleanup_resources(self):
        try:
            if self._agent:
                await self._agent.cleanup()
        except Exception as e:
            logger.warning(f"[IterativeFlow] Cleanup warning: {e}")

    def _generate_final_report(self, iterations: int, elapsed: float) -> str:
        stats = self._progress_tracker.get_statistics()
        mem_stats = self._memory_manager.get_statistics()
        files_list = "\n".join(f"- {f}" for f in stats["completed_files"]) or "No files"
        return (
            f"# Implementation Report (Iterative Mode)\n\n"
            f"## Summary\n"
            f"- Iterations: {iterations}\n"
            f"- Duration: {elapsed:.1f}s\n"
            f"- Files Implemented: {stats['total_files']}\n\n"
            f"## Completed Files\n{files_list}\n\n"
            f"## Statistics\n"
            f"- Unique Files: {stats['unique_files_count']}\n"
            f"- Compressions: {mem_stats['total_compressions']}\n"
        )


# ============================================================
# Entry Point
# ============================================================

async def main():
    controller = IterativeCodeFlow()
    await controller.initialize()

    plan_file = os.getenv("TEST_PLAN_FILE", "initial_plan.txt")
    target_dir = os.getenv("TEST_TARGET_DIR", "test_output")

    if not os.path.exists(plan_file):
        logger.warning(f"Plan file not found: {plan_file}")
        return

    result = await controller.execute(plan_file, target_dir)
    logger.info(f"Final Status: {result['status']}")


if __name__ == "__main__":
    asyncio.run(main())
