#!/usr/bin/env python
# coding: utf-8
"""
Multi-agent research pipeline built on LangChain / LangGraph.

Mirrors the original deepcode MultiAgentResearchFlow with identical phase
structure (Phase 0-9), replacing openjiuwen primitives with LangChain agents
and MCP tool management via langchain-mcp-adapters.
"""

import asyncio
import json
import os
import logging
import yaml
from typing import Tuple, Dict, Any, List, Optional

from dotenv import load_dotenv

from deepcode.agents.react_agent import ReActAgent, AgentConfig
from deepcode.utils.mcp_tool_manager import create_mcp_tools_from_config
from deepcode.prompts.sys_prompts import (
    PAPER_INPUT_ANALYZER_PROMPT,
    PAPER_DOWNLOADER_PROMPT,
    DOCUMENT_SEGMENTATION_PROMPT,
)
from deepcode.prompts.user_prompts import (
    RESOURCE_PROCESSOR_USER_PROMPT,
    ANALYZE_AND_PREPARE_DOCUMENT_USER_PROMPT,
    GET_DOCUMENT_OVERVIEW_USER_PROMPT,
    VALIDATE_SEGMENTATION_QUALITY_USER_PROMPT,
    CONCEPT_ANALYSIS_PROMPT,
    ALGORITHM_ANALYSIS_PROMPT,
    CODE_PLANNING_PROMPT,
    PAPER_REFERENCE_ANALYZER_PROMPT,
    GITHUB_DOWNLOADER_PROMPT,
)
from deepcode.utils.utils import extract_clean_json
from deepcode.tools.pdf_downloader import move_file_to, download_file_to
from deepcode.utils.file_processor import FileProcessor
from deepcode.utils.llm_utils import get_token_limits
from deepcode.agent_flow.agent_aggregation import AgentAggregation
from deepcode.agent_flow.code_implementation_flow_iterative import IterativeCodeFlow

_PROJECT_ENV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)
load_dotenv(dotenv_path=_PROJECT_ENV_PATH)

logger = logging.getLogger(__name__)


class MultiAgentResearchFlow:
    """
    Multi-phase research-to-code pipeline.

    Phases:
      0-1  Workspace setup & input validation
      2    Research analysis & resource processing
      3    Workspace infrastructure synthesis
      3.5  Document segmentation / preprocessing
      4    Code planning orchestration
      5    Reference intelligence  (optional)
      6    Repository acquisition  (optional)
      7    Codebase intelligence    (optional)
      8    Code implementation synthesis
      9    Final status report
    """

    def __init__(self):
        self.enable_index = True
        self.llm_config = {
            "model_provider": os.getenv("LLM_MODEL_PROVIDER"),
            "api_key": os.getenv("LLM_API_KEY"),
            "api_base": os.getenv("LLM_API_BASE"),
            "model_name": os.getenv("LLM_MODEL_NAME"),
        }

    # ------------------------------------------------------------------
    # Agent initialisation helpers
    # ------------------------------------------------------------------

    async def initialize_agents(self):
        await self._initialize_research_analyzer_agent()
        await self._initialize_resource_processor_agent()
        await self._initialize_document_segmentation_agent()
        await self._initialize_code_planning_agent()
        await self._initialize_citation_discovery_agent()
        await self._initialize_github_acquisition_agent()

    async def _initialize_research_analyzer_agent(self):
        self.analyzer_agent = ReActAgent(
            AgentConfig(
                name="ResearchAnalyzerAgent",
                llm_config=dict(self.llm_config),
                system_prompt=PAPER_INPUT_ANALYZER_PROMPT,
            )
        )

    async def _initialize_resource_processor_agent(self):
        self.processor_agent = ReActAgent(
            AgentConfig(
                name="ResourceProcessorAgent",
                llm_config=dict(self.llm_config),
                system_prompt=PAPER_DOWNLOADER_PROMPT,
            )
        )

        pdf_downloader_path = os.getenv("PDF_DOWNLOADER_PATH")
        if pdf_downloader_path and os.path.exists(pdf_downloader_path):
            await self.processor_agent.add_mcp_servers([
                {
                    "server_name": "file-downloader",
                    "command": "python",
                    "args": [pdf_downloader_path],
                }
            ])

    async def _initialize_document_segmentation_agent(self):
        self.document_segmentation_agent = ReActAgent(
            AgentConfig(
                name="DocumentSegmentationCoordinator",
                llm_config=dict(self.llm_config),
                system_prompt=DOCUMENT_SEGMENTATION_PROMPT,
            )
        )

        doc_seg_path = os.getenv("DOCUMENT_SEGMENTATION_PATH")
        if doc_seg_path and os.path.exists(doc_seg_path):
            await self.document_segmentation_agent.add_mcp_servers([
                {
                    "server_name": "document-segmentation-server",
                    "command": "python",
                    "args": [doc_seg_path],
                }
            ])

    async def _initialize_code_planning_agent(self):
        server_names = self._get_server_names()

        self.architecture_agent = ReActAgent(
            AgentConfig(
                name="ArchitectureSpecialist",
                llm_config=dict(self.llm_config),
                system_prompt=CONCEPT_ANALYSIS_PROMPT,
                server_names=server_names,
            )
        )

        self.algorithm_agent = ReActAgent(
            AgentConfig(
                name="AlgorithmSpecialist",
                llm_config=dict(self.llm_config),
                system_prompt=ALGORITHM_ANALYSIS_PROMPT,
                server_names=server_names,
            )
        )

        self.lead_planner_agent = ReActAgent(
            AgentConfig(
                name="LeadArchitect",
                llm_config=dict(self.llm_config),
                system_prompt=CODE_PLANNING_PROMPT,
                server_names=server_names,
            )
        )

        self.planning_engine = AgentAggregation(
            aggregator=self.lead_planner_agent,
            source_agents=[self.architecture_agent, self.algorithm_agent],
        )

    async def _initialize_citation_discovery_agent(self):
        server_names = self._get_server_names()
        self.citation_miner = ReActAgent(
            AgentConfig(
                name="CitationDiscoverySpecialist",
                llm_config=dict(self.llm_config),
                system_prompt=PAPER_REFERENCE_ANALYZER_PROMPT,
                server_names=server_names,
            )
        )

    async def _initialize_github_acquisition_agent(self):
        self.repo_acquisitor = ReActAgent(
            AgentConfig(
                name="RepoAcquisitionSpecialist",
                llm_config=dict(self.llm_config),
                system_prompt=GITHUB_DOWNLOADER_PROMPT,
                server_names=["filesystem"],
            )
        )

        git_tool_path = os.getenv("GITHUB_DOWNLOADER_PATH", "tools/git_command.py")
        if os.path.exists(git_tool_path):
            await self.repo_acquisitor.add_mcp_servers([
                {
                    "server_name": "github-downloader",
                    "command": "python",
                    "args": [git_tool_path],
                    "env": {"PYTHONPATH": "."},
                }
            ])
        else:
            logger.error(f"GitHub tool not found at {git_tool_path}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def ainvoke(self, input_source: str):
        # Phase 0
        workspace_dir = self._prepare_workspace()

        # Phase 1
        self._input_processing_and_validation(input_source)

        # Phase 2
        analysis_result, resource_processing_result = await self._analysis_processing_input(
            input_source
        )

        # Phase 3
        dir_info = await self._process_workspace_infrastructure(
            resource_processing_result, workspace_dir
        )

        # Phase 3.5
        doc_split_result = await self._document_preprocessing_agent(dir_info)

        process_status = doc_split_result.get("status", "")
        if process_status == "success":
            logger.info("Document preprocessing completed successfully")
            logger.info(f"Segmentation enabled: {dir_info.get('use_segmentation', False)}")
            if dir_info.get("segments_ready", False):
                logger.info(f"Segments dir: {doc_split_result.get('segments_dir', 'N/A')}")
        elif process_status == "fallback_to_traditional":
            logger.warning(
                f"Document segmentation failed, falling back: "
                f"{doc_split_result.get('original_error', 'unknown')}"
            )
        else:
            logger.error(
                f"Document preprocessing error: {doc_split_result.get('error_message', 'unknown')}"
            )

        # Phase 4
        await self._orchestrate_code_planning(dir_info)

        # Phase 8: Code Implementation Synthesis
        logger.info("\n" + "=" * 60)
        logger.info("[Phase 8] Starting Code Implementation Synthesis")
        logger.info("=" * 60)

        try:
            code_impl_controller = IterativeCodeFlow()
            await code_impl_controller.initialize()

            impl_result = await code_impl_controller.execute(
                plan_file_path=dir_info["initial_plan_path"],
                target_directory=dir_info["paper_dir"],
                enable_read_tools=True,
            )

            if impl_result["status"] in ("success", "partial"):
                metrics = impl_result.get("metrics", {})
                logger.info(
                    f"Code Implementation completed: {impl_result['status']}, "
                    f"{metrics.get('succeeded', 0)}/{metrics.get('total_files', 0)} files, "
                    f"{metrics.get('duration_seconds', 0)}s"
                )
            else:
                logger.error(f"Code Implementation failed: {impl_result.get('message')}")
        except Exception as e:
            logger.error(f"Failed to execute Code Implementation: {e}", exc_info=True)

        # Phase 9
        logger.info("\n" + "=" * 60)
        logger.info("[Phase 9] Multi-agent research pipeline completed!")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _prepare_workspace(self) -> str:
        workspace_dir = os.path.join(os.getcwd(), "deepcode_lab")
        os.makedirs(workspace_dir, exist_ok=True)
        logger.info(f"Workspace directory: {workspace_dir}")
        logger.info(f"Index {'enabled' if self.enable_index else 'disabled'}")
        return workspace_dir

    def _input_processing_and_validation(self, input_source: str) -> str:
        if input_source.startswith("file://"):
            file_path = input_source[len("file://"):]
            if os.name == "nt":
                file_path = file_path.lstrip("/") if file_path.startswith("/") else file_path
            return file_path
        return input_source

    async def _analysis_processing_input(self, input_source: str) -> Tuple[str, str]:
        supported_extensions = (".pdf", ".docx", ".txt", ".html", ".md")
        supported_prefixes = ("http", "file://")

        if isinstance(input_source, str):
            is_supported_ext = input_source.endswith(supported_extensions)
            is_supported_prefix = input_source.startswith(supported_prefixes)
            if is_supported_ext or is_supported_prefix:
                analysis_result = await self._run_research_analyzer(input_source)
                await asyncio.sleep(5)
                download_result = await self._run_resource_processor(analysis_result)
                return analysis_result, download_result

        return None, input_source

    async def _run_research_analyzer(self, prompt_text: str) -> str:
        if not prompt_text or prompt_text.strip() == "":
            raise ValueError("Empty prompt_text provided to run_research_analyzer")

        result = await self.analyzer_agent.ainvoke({"query": prompt_text})
        raw_result = result.get("content", "")
        if not raw_result:
            raise ValueError("LLM returned empty result")

        try:
            clean_result = extract_clean_json(raw_result)
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}, raw: {raw_result}")
            raise

        if not clean_result or clean_result.strip() == "":
            raise ValueError("JSON extraction resulted in empty output")

        return clean_result

    async def _run_resource_processor(self, analysis_result: str) -> str:
        papers_dir = "./deepcode_lab/papers"
        os.makedirs(papers_dir, exist_ok=True)

        next_id = self._get_next_paper_id(papers_dir)
        paper_dir = os.path.join(papers_dir, str(next_id))
        os.makedirs(paper_dir, exist_ok=True)

        logger.info(f"Paper ID: {next_id}, directory: {paper_dir}")

        try:
            analysis_data = json.loads(analysis_result)
            source_path = analysis_data.get("path") or analysis_data.get("input_path")
            input_type = analysis_data.get("input_type", "unknown")

            logger.info(f"Processing {input_type}: {source_path}")

            direct_result = await self._process_direct_source(
                input_type, source_path, paper_dir, next_id
            )

            if direct_result["success"]:
                resolved_paper_path = self._resolve_paper_path(paper_dir, next_id)
                return json.dumps({
                    "status": "success",
                    "paper_id": next_id,
                    "paper_dir": paper_dir,
                    # Keep both keys for compatibility with downstream processors.
                    "paper_path": resolved_paper_path,
                    "file_path": resolved_paper_path,
                    "message": f"File successfully processed to {paper_dir}",
                    "operation_details": direct_result["details"],
                })
            else:
                logger.info(f"Falling back to LLM agent for: {input_type} - {source_path}")
                context = f"\nPrevious attempt result: {direct_result['details']}"
                message = RESOURCE_PROCESSOR_USER_PROMPT.format(
                    paper_dir=paper_dir,
                    source_path=source_path,
                    input_type=input_type,
                    next_id=next_id,
                    next_id_2=next_id,
                    context=context,
                )
                agent_result = await self.processor_agent.ainvoke({"query": message})
                return agent_result.get("content", "")

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Error processing resource: {e}")
            return json.dumps({
                "status": "partial",
                "paper_id": next_id,
                "paper_dir": paper_dir,
                "message": f"Paper directory created at {paper_dir}, manual file placement may be needed",
            })

    def _get_next_paper_id(self, papers_dir: str) -> int:
        try:
            existing_ids = []
            for d in os.listdir(papers_dir):
                dir_path = os.path.join(papers_dir, d)
                if os.path.isdir(dir_path) and d.isdigit():
                    existing_ids.append(int(d))
            return max(existing_ids) + 1 if existing_ids else 1
        except Exception:
            return 1

    def _resolve_paper_path(self, paper_dir: str, paper_id: int) -> str:
        """
        Resolve the best available paper file path after direct download/copy.
        Prefer converted markdown, then common text-like formats, then PDF.
        """
        candidate_paths = [
            os.path.join(paper_dir, f"{paper_id}.md"),
            os.path.join(paper_dir, f"{paper_id}.markdown"),
            os.path.join(paper_dir, f"{paper_id}.txt"),
            os.path.join(paper_dir, f"{paper_id}.html"),
            os.path.join(paper_dir, f"{paper_id}.htm"),
            os.path.join(paper_dir, f"{paper_id}.pdf"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return path

        if os.path.isdir(paper_dir):
            preferred_exts = (".md", ".markdown", ".txt", ".html", ".htm", ".pdf")
            for ext in preferred_exts:
                for filename in sorted(os.listdir(paper_dir)):
                    if filename.lower().endswith(ext):
                        return os.path.join(paper_dir, filename)

        # Fallback to historical default so callers still receive a deterministic path.
        return os.path.join(paper_dir, f"{paper_id}.md")

    async def _process_direct_source(
        self, input_type: str, source_path: str, paper_dir: str, paper_id: int
    ) -> dict:
        operation_result = None
        success = False

        if input_type == "file" and source_path and os.path.exists(source_path):
            logger.info(f"Direct file copy: {source_path} -> {paper_dir}")
            try:
                operation_result = await move_file_to(
                    source=source_path,
                    destination=paper_dir,
                    filename=f"{paper_id}.pdf",
                )
                success = "[SUCCESS]" in operation_result and "[ERROR]" not in operation_result
            except Exception as e:
                logger.warning(f"Direct file copy failed: {e}")

        elif input_type == "url" and source_path:
            logger.info(f"Direct URL download: {source_path} -> {paper_dir}")
            try:
                operation_result = await download_file_to(
                    url=source_path,
                    destination=paper_dir,
                    filename=f"{paper_id}.pdf",
                )
                success = "[SUCCESS]" in operation_result and "[ERROR]" not in operation_result
            except Exception as e:
                logger.warning(f"Direct download failed: {e}")

        return {"success": success, "details": operation_result}

    async def _process_workspace_infrastructure(
        self, download_result: str, workspace_dir: str
    ):
        file_process_result = await FileProcessor.process_file_input(
            download_result, base_dir=workspace_dir
        )
        paper_dir = file_process_result["paper_dir"]

        logger.info(
            f"Workspace infrastructure synthesized: "
            f"base={workspace_dir or 'auto'}, research={paper_dir}"
        )

        return {
            "paper_dir": paper_dir,
            "standardized_text": file_process_result["standardized_text"],
            "reference_path": os.path.join(paper_dir, "reference.txt"),
            "initial_plan_path": os.path.join(paper_dir, "initial_plan.txt"),
            "download_path": os.path.join(paper_dir, "github_download.txt"),
            "index_report_path": os.path.join(paper_dir, "codebase_index_report.txt"),
            "implementation_report_path": os.path.join(paper_dir, "code_implementation_report.txt"),
            "workspace_dir": workspace_dir,
        }

    async def _document_preprocessing_agent(self, dir_info: Dict[str, str]) -> Dict[str, Any]:
        try:
            logger.info("Starting adaptive document preprocessing…")
            paper_dir = dir_info["paper_dir"]

            md_file_list = [f for f in os.listdir(paper_dir) if f.endswith(".md")]

            if not md_file_list:
                logger.info("No Markdown files found, skipping preprocessing")
                dir_info["segments_ready"] = False
                dir_info["use_segmentation"] = False
                return {
                    "status": "skipped",
                    "reason": "no_markdown_files",
                    "paper_dir": paper_dir,
                    "segments_ready": False,
                    "use_segmentation": False,
                }

            primary_md_path = os.path.join(paper_dir, md_file_list[0])
            doc_content = ""
            try:
                with open(primary_md_path, "rb") as f:
                    file_header = f.read(8)
                    if file_header.startswith(b"%PDF"):
                        raise IOError(
                            f"File {primary_md_path} is actually a PDF. "
                            "Please convert to Markdown."
                        )
                with open(primary_md_path, "r", encoding="utf-8") as f:
                    doc_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read document: {e}")
                dir_info["segments_ready"] = False
                dir_info["use_segmentation"] = False
                return {
                    "status": "error",
                    "error_message": f"Document read failed: {e}",
                    "paper_dir": paper_dir,
                    "segments_ready": False,
                    "use_segmentation": False,
                }

            need_segment, segment_reason = self._should_use_document_segmentation(doc_content)
            logger.info(f"Segmentation decision: {need_segment} – {segment_reason}")
            dir_info["use_segmentation"] = need_segment

            if need_segment:
                logger.info("Enabling intelligent document segmentation…")
                segment_result = await self._prepare_document_segments(paper_dir=paper_dir)

                if segment_result["status"] == "success":
                    dir_info["segments_dir"] = segment_result["segments_dir"]
                    dir_info["segments_ready"] = True
                    return segment_result
                else:
                    logger.warning(
                        f"Segmentation failed: {segment_result.get('error_message')}, "
                        "falling back to traditional mode"
                    )
                    dir_info["segments_ready"] = False
                    dir_info["use_segmentation"] = False
                    return {
                        "status": "fallback_to_traditional",
                        "original_error": segment_result.get("error_message", "unknown"),
                        "paper_dir": paper_dir,
                        "segments_ready": False,
                        "use_segmentation": False,
                    }
            else:
                dir_info["segments_ready"] = False
                return {
                    "status": "traditional",
                    "reason": segment_reason,
                    "paper_dir": paper_dir,
                    "segments_ready": False,
                    "use_segmentation": False,
                    "document_size": len(doc_content),
                }

        except Exception as e:
            logger.error(f"Document preprocessing error: {e}")
            dir_info["segments_ready"] = False
            dir_info["use_segmentation"] = False
            return {
                "status": "error",
                "paper_dir": dir_info["paper_dir"],
                "segments_ready": False,
                "use_segmentation": False,
                "error_message": str(e),
            }

    def _should_use_document_segmentation(
        self,
        document_content: str,
        config_path: str = "config.yaml",
    ) -> Tuple[bool, str]:
        default_config = {"enabled": True, "size_threshold_chars": 50000}
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                seg_config = config.get("document_segmentation", {})
                seg_config = {
                    "enabled": seg_config.get("enabled", default_config["enabled"]),
                    "size_threshold_chars": seg_config.get(
                        "size_threshold_chars",
                        default_config["size_threshold_chars"],
                    ),
                }
            else:
                seg_config = default_config
        except Exception:
            seg_config = default_config

        if not seg_config["enabled"]:
            return False, "Document segmentation disabled in configuration"

        doc_size = len(document_content)
        threshold = seg_config["size_threshold_chars"]

        if doc_size > threshold:
            return True, f"Document size ({doc_size:,} chars) exceeds threshold ({threshold:,} chars)"
        return False, f"Document size ({doc_size:,} chars) below threshold ({threshold:,} chars)"

    async def _prepare_document_segments(
        self, paper_dir: str, force_refresh: bool = False
    ) -> Dict[str, Any]:
        try:
            md_files = [f for f in os.listdir(paper_dir) if f.endswith(".md")]
            if not md_files:
                raise ValueError(f"No markdown file found in {paper_dir}")

            doc_analysis_prompt = ANALYZE_AND_PREPARE_DOCUMENT_USER_PROMPT.format(
                paper_dir=paper_dir,
                force_refresh=force_refresh,
                paper_dir_2=paper_dir,
            )
            await self.document_segmentation_agent.ainvoke({"query": doc_analysis_prompt})

            doc_overview_prompt = GET_DOCUMENT_OVERVIEW_USER_PROMPT.format(paper_dir=paper_dir)
            doc_overview_output = await self.document_segmentation_agent.ainvoke(
                {"query": doc_overview_prompt}
            )

            seg_validate_prompt = VALIDATE_SEGMENTATION_QUALITY_USER_PROMPT.format(
                paper_dir=paper_dir
            )
            seg_validate_output = await self.document_segmentation_agent.ainvoke(
                {"query": seg_validate_prompt}
            )

            segments_dir = os.path.join(paper_dir, "document_segments")
            return {
                "status": "success",
                "paper_dir": paper_dir,
                "segments_dir": segments_dir,
                "segments_available": True,
                "overview_data": {
                    "status": "success",
                    "paper_dir": paper_dir,
                    "overview_result": doc_overview_output,
                },
                "validation_result": seg_validate_output,
            }
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return {
                "status": "error",
                "paper_dir": paper_dir,
                "error_message": str(e),
                "segments_available": False,
            }

    def _get_default_search_server(self, config_path: str = "config.yaml") -> str:
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                return config.get("default_search_server", "brave")
        except Exception:
            pass
        return "brave"

    def _get_server_names(self) -> List[str]:
        search_server = self._get_default_search_server()
        return ["filesystem", search_server]

    async def _orchestrate_code_planning(self, dir_info: Dict[str, str]):
        logger.info("[Phase 4] Synthesizing intelligent code architecture…")

        initial_plan_path = dir_info["initial_plan_path"]
        paper_dir = dir_info["paper_dir"]
        use_segmentation = dir_info.get("use_segmentation", True)

        if os.path.exists(initial_plan_path):
            logger.info(f"Existing plan found at {initial_plan_path}, skipping generation.")
            return

        try:
            plan_content = await self._run_code_analyzer_workflow(paper_dir, use_segmentation)
            if plan_content:
                with open(initial_plan_path, "w", encoding="utf-8") as f:
                    f.write(plan_content)
                logger.info(f"Initial plan saved to {initial_plan_path}")
            else:
                raise ValueError("Code planning returned empty result")
        except Exception as e:
            logger.error(f"Error in code planning orchestration: {e}")
            raise

    def _get_file_size_threshold(self, config_path: str = "config.yaml") -> int:
        default_threshold = 50000
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                return config.get("document_segmentation", {}).get(
                    "size_threshold_chars", default_threshold
                )
        except Exception:
            pass
        return default_threshold

    async def _run_code_analyzer_workflow(
        self, paper_dir: str, use_segmentation: bool = True
    ) -> str:
        base_token_limit, retry_token_limit = get_token_limits()
        file_size_threshold = self._get_file_size_threshold()
        search_server = self._get_default_search_server()
        full_content_loaded = False

        context_instruction = ""
        if use_segmentation:
            segments_dir = os.path.join(paper_dir, "document_segments")
            if os.path.exists(segments_dir) and os.listdir(segments_dir):
                context_instruction = (
                    f"The research paper is extensive. Structured summaries are in: "
                    f"'{segments_dir}'.\n"
                    f"**Action Required**: Read the files in the segments directory."
                )
            else:
                context_instruction = (
                    f"Please use your tools to analyze the paper files in: {paper_dir}"
                )
        else:
            try:
                for filename in os.listdir(paper_dir):
                    if filename.endswith(".md"):
                        file_path = os.path.join(paper_dir, filename)
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        if len(content) < file_size_threshold:
                            context_instruction = (
                                f"Here is the full content of the research paper:\n"
                                f"=== PAPER CONTENT START ===\n{content}\n"
                                f"=== PAPER CONTENT END ===\n"
                            )
                            full_content_loaded = True
                        else:
                            context_instruction = (
                                f"The paper file is at: {file_path}.\n"
                                f"It is too large ({len(content)} chars). "
                                f"**Action Required**: Read it with filesystem tools."
                            )
                        break
                else:
                    context_instruction = (
                        f"Please locate and analyze the research paper in: {paper_dir}"
                    )
            except Exception:
                context_instruction = (
                    f"Please locate and analyze the research paper in: {paper_dir}"
                )

        search_directive = (
            "You have access to external search tools. Use them to:\n"
            " - Verify library versions\n"
            " - Find official repositories\n"
            " - Clarify algorithmic details\n"
            "Do NOT hallucinate API calls or library names."
        )

        base_query = (
            f"Project Workspace Directory: {paper_dir}\n\n"
            f"=== RESEARCH CONTEXT ===\n{context_instruction}\n=== END CONTEXT ===\n\n"
            f"{search_directive}\n\n"
            f"**Mission**: Synthesize the research analysis into a production-ready "
            f"code reproduction blueprint in strict YAML format.\n\n"
            f"**Structure Requirement**: The output must contain exactly these keys:\n"
            f"- file_structure\n- implementation_components\n"
            f"- environment_setup\n- validation_approach\n- implementation_strategy"
        )

        current_max_tokens = base_token_limit
        current_temp = 0.2
        max_attempts = 3
        best_result = ""

        for attempt in range(max_attempts):
            try:
                logger.info(f"Planning attempt {attempt + 1}/{max_attempts}")

                response = await self.planning_engine.ainvoke(
                    inputs={"query": base_query},
                    runtime=None,
                )

                plan_text = response.get("content", "") if isinstance(response, dict) else str(response)
                integrity_score = self.evaluate_plan_completeness(plan_text)

                if integrity_score >= 0.8:
                    logger.info(f"High-quality plan generated (score: {integrity_score:.2f})")
                    return plan_text

                logger.warning(f"Plan incomplete (score: {integrity_score:.2f}), retrying…")
                if len(plan_text) > len(best_result):
                    best_result = plan_text

                if attempt == 0:
                    current_max_tokens = retry_token_limit
                elif attempt == 1:
                    current_max_tokens = int(retry_token_limit * 0.9)
                else:
                    current_max_tokens = int(retry_token_limit * 0.8)
                current_temp = max(current_temp - 0.15, 0.05)

            except Exception as e:
                logger.error(f"Execution failed at attempt {attempt + 1}: {e}")
                continue

        if not best_result:
            raise ValueError("Failed to generate code plan after multiple attempts.")
        return best_result

    def evaluate_plan_completeness(self, content: str) -> float:
        DEFAULT_PLAN_SECTIONS = [
            "file_structure:",
            "implementation_components:",
            "validation_approach:",
            "environment_setup:",
            "implementation_strategy:",
        ]

        if not content or len(content.strip()) < 500:
            return 0.0

        text_lower = content.lower()
        total_score = 0.0

        weights = {
            "sections": 0.5,
            "structure": 0.2,
            "completeness": 0.15,
            "length": 0.15,
        }

        found_count = sum(1 for key in DEFAULT_PLAN_SECTIONS if key in text_lower)
        coverage_ratio = found_count / len(DEFAULT_PLAN_SECTIONS)
        total_score += coverage_ratio * weights["sections"]

        has_start = any(tag in content for tag in ["```yaml", "file_structure:", "paper_info:"])
        tail_content = content[-500:]
        has_end = any(
            tag in tail_content
            for tag in ["```", "validation_approach:", "implementation_strategy:"]
        )

        if has_start and has_end:
            total_score += weights["structure"]
        elif has_start:
            total_score += weights["structure"] / 2

        lines = content.strip().splitlines()
        is_truncated = True
        if lines:
            last_line = lines[-1].strip()
            valid_endings = ("```", ".", ":", "}", "]")
            is_list_item = last_line.startswith(("-", "*"))
            is_short_line = len(last_line) < 80 and not last_line.endswith(",")
            if last_line.endswith(valid_endings) or is_list_item or is_short_line:
                is_truncated = False

        if not is_truncated:
            total_score += weights["completeness"]

        length = len(content)
        if length >= 10000:
            total_score += weights["length"]
        elif length >= 5000:
            total_score += weights["length"] - 0.05
        elif length >= 2000:
            total_score += weights["length"] - 0.1

        return min(total_score, 1.0)

    async def _execute_reference_mining_workflow(self, dir_info: Dict[str, str]) -> str:
        logger.info("[Phase 5] Initiating citation mining…")
        reference_path = dir_info["reference_path"]
        paper_dir = dir_info["paper_dir"]

        if not self.enable_index:
            skip_msg = "Citation mining skipped (Fast Mode)."
            with open(reference_path, "w", encoding="utf-8") as f:
                f.write(skip_msg)
            return skip_msg

        if os.path.exists(reference_path):
            with open(reference_path, "r", encoding="utf-8") as f:
                existing = f.read()
            if len(existing) > 50:
                return existing

        try:
            mining_query = (
                f"Target Directory: {paper_dir}\n"
                f"**Task**: Locate the markdown file and analyze the References section.\n"
                f"**Goal**: Extract 5 most relevant references with GitHub URLs."
            )
            result = await self.citation_miner.ainvoke({"query": mining_query})
            result_content = result.get("content", "") if isinstance(result, dict) else str(result)

            if result_content:
                with open(reference_path, "w", encoding="utf-8") as f:
                    f.write(result_content)
            return result_content
        except Exception as e:
            error_log = f"Error during mining: {e}"
            with open(reference_path, "w", encoding="utf-8") as f:
                f.write(error_log)
            return error_log

    async def _execute_repo_acquisition_workflow(
        self, reference_data: str, dir_map: Dict[str, str]
    ) -> None:
        logger.info("[Phase 6] Automating repository acquisition…")
        download_log = dir_map["download_path"]
        code_base_dir = os.path.join(dir_map["paper_dir"], "code_base")

        if not self.enable_index:
            self._write_log(download_log, "Repository acquisition skipped (Fast Mode).")
            return

        if not reference_data or len(reference_data) < 20 or "skipped" in reference_data.lower():
            self._write_log(download_log, "Reference data invalid or empty.")
            return

        try:
            await asyncio.sleep(2)
            instruction = (
                f"Reference Context:\n{reference_data}\n\n"
                f"**Task**: Clone relevant GitHub repositories.\n"
                f"**Target Directory**: {code_base_dir}\n"
                f"**Constraint**: Skip already-cloned repos."
            )
            response = await self.repo_acquisitor.ainvoke({"query": instruction})
            result_text = response.get("content", "") if isinstance(response, dict) else str(response)
            self._write_log(download_log, result_text)
            self._verify_download_results(code_base_dir)
        except Exception as e:
            self._write_log(download_log, f"Error in repo acquisition: {e}")

    def _verify_download_results(self, target_dir: str) -> bool:
        if not os.path.exists(target_dir):
            return False
        try:
            repos = [
                d for d in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith(".")
            ]
            if repos:
                logger.info(f"Download verified: {len(repos)} repos ({', '.join(repos[:3])}…)")
                return True
        except Exception:
            pass
        return False

    def _write_log(self, path: str, content: str):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write log to {path}: {e}")
