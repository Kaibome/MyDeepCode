#!/usr/bin/env python
# coding: utf-8
"""
Merged Codebase Indexer — combines CodeIndexer and CodebaseIndexWorkflow.

Built on LangChain (replaces openjiuwen imports), but the core indexing
logic is framework-independent.
"""

import asyncio
import json
import os
import re
import time
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from deepcode.agent_flow.codebase_agent import (
    pre_filter_files,
    analyze_file_content,
    find_relationships,
)
from deepcode.utils.code_indexing_utils import (
    supported_extensions,
    skip_directories,
    FileSummary,
    FileRelationship,
    RepoIndex,
    IndexerConfig,
)

logger = logging.getLogger(__name__)


class MergedCodebaseIndexer:
    """Merged codebase indexer combining CodeIndexer and CodebaseIndexWorkflow."""

    def __init__(
        self,
        code_base_path: Union[str, Path],
        target_structure: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        enable_pre_filtering: bool = True,
        indexer_config_path: Optional[str] = None,
    ):
        self.code_base_path = Path(code_base_path)
        self.output_dir = Path(output_dir) if output_dir else self.code_base_path / "code_indexes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_structure = target_structure
        self.enable_pre_filtering = enable_pre_filtering
        self.indexer_config_path = indexer_config_path

        self.config = IndexerConfig(
            code_base_path=self.code_base_path,
            output_dir=self.output_dir,
        )

        self.verbose_output = True
        self.include_metadata = True
        self.generate_statistics = True
        self.generate_summary = True
        self.index_filename_pattern = "{repo_name}_code_index.json"
        self.summary_filename = "codebase_summary.md"
        self.stats_filename = "codebase_statistics.json"
        self.initial_plan_path = None

        self.min_confidence_score = 0.3
        self.high_confidence_threshold = 0.7
        self.relationship_types = {
            "direct_match": 1.0,
            "partial_match": 0.8,
            "reference": 0.6,
            "utility": 0.4,
        }

        self.enable_concurrent_analysis = True
        self.max_concurrent_files = 5
        self.enable_content_caching = False
        self.max_cache_size = 100
        self.content_cache = {} if self.enable_content_caching else None
        self.request_delay = 0.1

        self.supported_extensions = set(supported_extensions)
        self.skip_directories = set(skip_directories)
        self.max_file_size = 1048576
        self.max_content_length = 3000

        logger.info(f"Initialized MergedCodebaseIndexer: {self.code_base_path}")

    # ------------------------------------------------------------------
    # Plan / config helpers
    # ------------------------------------------------------------------

    def extract_file_tree_from_plan(self, plan_content: str) -> Dict[str, Any]:
        patterns = [
            r"```(?:json|markdown|)\s*(.*?)\s*```",
            r"```(?:json|markdown|)\s*([\s\S]*?)\s*```",
        ]
        tree_content = plan_content
        for p in patterns:
            m = re.search(p, plan_content, re.DOTALL)
            if m:
                tree_content = m.group(1)
                break

        tree_content = re.sub(r"^#.*?\n", "", tree_content, flags=re.MULTILINE)
        tree_content = re.sub(r"^\*\s*", "", tree_content, flags=re.MULTILINE)
        tree_content = re.sub(r"^-\s*", "", tree_content, flags=re.MULTILINE)

        try:
            ft = json.loads(tree_content)
            if isinstance(ft, dict):
                return ft
        except json.JSONDecodeError:
            pass
        return {"name": "codebase", "type": "directory", "children": []}

    def load_target_structure_from_plan(self, plan_path: Union[str, Path]) -> Dict[str, Any]:
        plan_path = Path(plan_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan file not found: {plan_path}")
        with open(plan_path, "r", encoding="utf-8") as f:
            return self.extract_file_tree_from_plan(f.read())

    def load_or_create_indexer_config(
        self, initial_plan_path: Optional[Union[str, Path]] = None
    ) -> IndexerConfig:
        if initial_plan_path:
            self.initial_plan_path = Path(initial_plan_path)
            if self.initial_plan_path.exists():
                self.target_structure = self.load_target_structure_from_plan(self.initial_plan_path)
        return IndexerConfig(
            code_base_path=self.code_base_path,
            output_dir=self.output_dir,
            initial_plan_path=self.initial_plan_path,
            target_structure=self.target_structure,
        )

    # ------------------------------------------------------------------
    # File tree / scanning
    # ------------------------------------------------------------------

    def generate_file_tree(self, repo_path: Path, max_depth: int = 5) -> str:
        tree_lines: List[str] = []

        def _walk(cur: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            try:
                items = sorted(
                    cur.iterdir(),
                    key=lambda x: (x.is_file(), x.name.lower() if x.name else ""),
                )
                items = [
                    i for i in items
                    if not i.name.startswith(".") and i.name not in self.skip_directories
                ]
                for idx, item in enumerate(items):
                    is_last = idx == len(items) - 1
                    tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
                    if item.is_dir():
                        _walk(item, prefix + ("    " if is_last else "│   "), depth + 1)
            except PermissionError:
                tree_lines.append(f"{prefix}├── [Permission Denied]")

        tree_lines.append(f"{repo_path.name}/")
        _walk(repo_path)
        return "\n".join(tree_lines)

    def get_all_repo_files(self, repo_path: Path) -> List[Path]:
        files: List[Path] = []
        try:
            for root, dirs, filenames in os.walk(repo_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in self.skip_directories]
                for fn in filenames:
                    fp = Path(root) / fn
                    if fp.suffix.lower() in self.supported_extensions:
                        files.append(fp)
        except Exception as e:
            logger.error(f"Error traversing {repo_path}: {e}")
        return files

    def filter_files_by_paths(
        self, all_files: List[Path], selected_paths: List[str], repo_path: Path
    ) -> List[Path]:
        if not selected_paths:
            return all_files
        filtered = []
        for fp in all_files:
            rel = str(fp.relative_to(repo_path))
            for sp in selected_paths:
                if (
                    rel == sp
                    or rel.replace("\\", "/") == sp.replace("\\", "/")
                    or sp in rel
                    or rel in sp
                ):
                    filtered.append(fp)
                    break
        return filtered

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    async def _analyze_single(self, fp: Path, idx: int, total: int) -> Tuple:
        if self.verbose_output:
            logger.info(f"Analyzing {idx}/{total}: {fp.name}")
        summary = await analyze_file_content(
            file_path=fp,
            max_file_size=self.max_file_size,
            code_base_path=self.code_base_path,
            enable_content_caching=self.enable_content_caching,
            content_cache=self.content_cache,
            verbose_output=self.verbose_output,
            max_content_length=self.max_content_length,
            max_cache_size=self.max_cache_size,
        )
        rels = await find_relationships(
            file_summary=summary,
            target_structure=self.target_structure,
            min_confidence_score=self.min_confidence_score,
            relationship_types=self.relationship_types,
            verbose_output=self.verbose_output,
        )
        return summary, rels

    async def _process_sequentially(self, files: List[Path]) -> Tuple:
        summaries, rels = [], []
        for i, fp in enumerate(files, 1):
            s, r = await self._analyze_single(fp, i, len(files))
            summaries.append(s)
            rels.extend(r)
            await asyncio.sleep(self.request_delay)
        return summaries, rels

    async def _process_concurrently(self, files: List[Path]) -> Tuple:
        sem = asyncio.Semaphore(self.max_concurrent_files)

        async def _wrapped(fp, idx, total):
            async with sem:
                if idx > 1:
                    await asyncio.sleep(self.request_delay * 0.5)
                return await self._analyze_single(fp, idx, total)

        tasks = [_wrapped(fp, i, len(files)) for i, fp in enumerate(files, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries, rels = [], []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Failed to analyze {files[i]}: {r}")
                summaries.append(
                    FileSummary(
                        file_path=str(files[i].relative_to(self.code_base_path)),
                        file_type="error", main_functions=[], key_concepts=[],
                        dependencies=[], summary=f"Failed: {r}", lines_of_code=0,
                        last_modified="",
                    )
                )
            else:
                s, rel = r
                summaries.append(s)
                rels.extend(rel)
        return summaries, rels

    # ------------------------------------------------------------------
    # Repository processing
    # ------------------------------------------------------------------

    async def process_repository(self, repo_dir: Path) -> RepoIndex:
        repo_name = repo_dir.name
        logger.info(f"Processing repository: {repo_name}")

        file_tree = self.generate_file_tree(repo_dir)
        all_files = self.get_all_repo_files(repo_dir)
        logger.info(f"Found {len(all_files)} files in {repo_name}")

        if self.enable_pre_filtering and self.target_structure:
            selected = await pre_filter_files(
                self.target_structure, file_tree, self.min_confidence_score
            )
        else:
            selected = []

        files_to_analyze = (
            self.filter_files_by_paths(all_files, selected, repo_dir) if selected else all_files
        )

        if self.enable_concurrent_analysis and len(files_to_analyze) > 1:
            summaries, rels = await self._process_concurrently(files_to_analyze)
        else:
            summaries, rels = await self._process_sequentially(files_to_analyze)

        return RepoIndex(
            repo_name=repo_name,
            total_files=len(all_files),
            file_summaries=summaries,
            relationships=rels,
            analysis_metadata={
                "analysis_date": datetime.now().isoformat(),
                "total_relationships_found": len(rels),
                "high_confidence_relationships": len(
                    [r for r in rels if r.confidence_score > self.high_confidence_threshold]
                ),
                "analyzer_version": "2.0.0-langgraph",
                "pre_filtering_enabled": self.enable_pre_filtering,
                "files_before_filtering": len(all_files),
                "files_after_filtering": len(files_to_analyze),
                "filtering_efficiency": round(
                    (1 - len(files_to_analyze) / len(all_files)) * 100, 2
                ) if all_files else 0,
                "min_confidence_score": self.min_confidence_score,
                "high_confidence_threshold": self.high_confidence_threshold,
            },
        )

    def export_index(self, repo_index: RepoIndex, index_file: Path) -> None:
        class _Encoder(json.JSONEncoder):
            def default(self, o):
                if hasattr(o, "__dict__"):
                    return o.__dict__
                return super().default(o)

        index_dict = asdict(repo_index)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_dict, f, indent=2, ensure_ascii=False, cls=_Encoder)
        logger.info(f"Exported index to {index_file}")

    def generate_summary_markdown(self, repo_index: RepoIndex) -> str:
        md = f"# Codebase Intelligence Report: {repo_index.repo_name}\n\n"
        md += f"Total Files: {repo_index.total_files}\n"
        md += f"Analyzed: {len(repo_index.file_summaries)}\n"
        md += f"Relationships: {len(repo_index.relationships)}\n\n"
        for fs in sorted(repo_index.file_summaries, key=lambda x: x.file_path):
            md += f"### {fs.file_path}\n- Type: {fs.file_type}\n- Lines: {fs.lines_of_code}\n"
            md += f"- Summary: {fs.summary[:150]}…\n\n"
        return md

    async def build_all_indexes(self) -> Dict[str, str]:
        if not self.code_base_path.exists():
            raise FileNotFoundError(f"Code base not found: {self.code_base_path}")

        repo_dirs = [
            d for d in self.code_base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if not repo_dirs:
            raise ValueError(f"No repositories in {self.code_base_path}")

        output_files: Dict[str, str] = {}
        for repo_dir in repo_dirs:
            try:
                ri = await self.process_repository(repo_dir)
                fn = self.index_filename_pattern.format(repo_name=ri.repo_name)
                out = self.output_dir / fn
                self.export_index(ri, out)
                output_files[ri.repo_name] = str(out)
            except Exception as e:
                logger.error(f"Failed to process {repo_dir.name}: {e}")
        return output_files

    async def run_indexing_workflow(
        self, initial_plan_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        logger.info("Starting codebase indexing workflow…")
        start = time.time()
        try:
            cfg = self.load_or_create_indexer_config(initial_plan_path)
            self.code_base_path = cfg.code_base_path
            self.output_dir = cfg.output_dir
            self.target_structure = cfg.target_structure

            if not self.code_base_path.exists():
                raise FileNotFoundError(f"Not found: {self.code_base_path}")

            output_files = await self.build_all_indexes()
            elapsed = time.time() - start
            return {
                "success": True,
                "message": "Indexing completed",
                "execution_time": round(elapsed, 2),
                "output_files": output_files,
            }
        except Exception as e:
            logger.error(f"Indexing workflow failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "execution_time": round(time.time() - start, 2),
            }

    @staticmethod
    async def run_codebase_indexing(
        code_base_path, output_dir=None, initial_plan_path=None
    ) -> Dict[str, Any]:
        indexer = MergedCodebaseIndexer(code_base_path, output_dir=output_dir)
        return await indexer.run_indexing_workflow(initial_plan_path)


async def orchestrate_codebase_intelligence_agent(dir_info: Dict[str, Any]) -> Dict[str, Any]:
    paper_dir = dir_info.get("paper_dir")
    if not paper_dir:
        raise ValueError("paper_dir not found in dir_info")

    code_base_dir = os.path.join(paper_dir, "code_base")
    code_base_path = Path(code_base_dir)

    if not code_base_path.exists():
        os.makedirs(code_base_path, exist_ok=True)
        return {"status": "empty", "message": "No repositories found"}

    repo_dirs = [i for i in code_base_path.iterdir() if i.is_dir()]
    if not repo_dirs:
        return {"status": "empty", "message": "No repositories found"}

    all_results = {}
    for rd in repo_dirs:
        try:
            indexer = MergedCodebaseIndexer(rd)
            results = await indexer.run_indexing_workflow(dir_info.get("initial_plan_path"))
            all_results[rd.name] = results
        except Exception as e:
            all_results[rd.name] = {"status": "error", "message": str(e)}

    return {"status": "completed", "repositories": all_results}
