"""
LLM utility functions for DeepCode LangGraph project.

Replaces mcp_agent LLM class imports with LangChain model class selection.
Preserves the same public API as the original deepcode/utils/llm_utils.py.
"""

import os
import logging
from typing import Any, Type, Dict, Tuple

import yaml

logger = logging.getLogger(__name__)


def get_preferred_llm_class(config_path: str = "config.yaml") -> Type[Any]:
    """
    Select the LangChain chat model class based on user preference.

    Priority:
      1. Check config.yaml for llm_provider preference
      2. Verify the preferred provider has an API key (env vars)
      3. Fallback to first available provider

    Returns the LangChain chat model *class* (not an instance).
    """
    from langchain_openai import ChatOpenAI

    try:
        preferred_provider = None
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            preferred_provider = (config.get("llm_provider") or "").strip().lower()

        provider_map = {
            "openai": ("langchain_openai", "ChatOpenAI", "OPENAI_API_KEY"),
            "anthropic": ("langchain_anthropic", "ChatAnthropic", "ANTHROPIC_API_KEY"),
            "google": ("langchain_google_genai", "ChatGoogleGenerativeAI", "GOOGLE_API_KEY"),
        }

        def _load_class(module_name: str, class_name: str):
            import importlib
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name)

        if preferred_provider and preferred_provider in provider_map:
            mod, cls_name, env_key = provider_map[preferred_provider]
            if os.getenv(env_key) or os.getenv("LLM_API_KEY"):
                logger.info(f"Using {cls_name} (preference: {preferred_provider})")
                return _load_class(mod, cls_name)
            else:
                logger.warning(
                    f"Preferred provider '{preferred_provider}' has no API key, checking alternatives"
                )

        for provider, (mod, cls_name, env_key) in provider_map.items():
            if os.getenv(env_key) or os.getenv("LLM_API_KEY"):
                logger.info(f"Using {cls_name} ({provider} key found)")
                return _load_class(mod, cls_name)

        logger.warning("No API keys configured, falling back to ChatOpenAI")
        return ChatOpenAI

    except Exception as e:
        logger.error(f"Error in get_preferred_llm_class: {e}")
        return ChatOpenAI


def get_token_limits(config_path: str = "config.yaml") -> Tuple[int, int]:
    """
    Get token limits from configuration.

    Returns:
        (base_max_tokens, retry_max_tokens)
    """
    default_base = 20000
    default_retry = 15000

    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            openai_config = config.get("openai", {}) or {}
            base_tokens = openai_config.get("base_max_tokens", default_base)
            retry_tokens = openai_config.get("retry_max_tokens", default_retry)
            logger.info(f"Token limits: base={base_tokens}, retry={retry_tokens}")
            return base_tokens, retry_tokens
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return default_base, default_retry
    except Exception as e:
        logger.error(f"Error reading token config: {e}")
        return default_base, default_retry


def get_default_models(config_path: str = "config.yaml") -> Dict[str, str]:
    """
    Get default models from configuration file.

    Returns:
        dict with 'anthropic', 'openai', and 'google' default models.
    """
    defaults = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "google": "gemini-2.0-flash",
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return {
                "anthropic": (config.get("anthropic") or {}).get("default_model", defaults["anthropic"]),
                "openai": (config.get("openai") or {}).get("default_model", defaults["openai"]),
                "google": (config.get("google") or {}).get("default_model", defaults["google"]),
            }
        else:
            logger.warning(f"Config file {config_path} not found, using default models")
            return defaults
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return defaults


def get_document_segmentation_config(
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Get document segmentation configuration from config file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            seg_config = config.get("document_segmentation", {}) or {}
            return {
                "enabled": seg_config.get("enabled", True),
                "size_threshold_chars": seg_config.get("size_threshold_chars", 50000),
            }
        else:
            return {"enabled": True, "size_threshold_chars": 50000}
    except Exception as e:
        logger.error(f"Error reading segmentation config: {e}")
        return {"enabled": True, "size_threshold_chars": 50000}


def should_use_document_segmentation(
    document_content: str, config_path: str = "config.yaml"
) -> Tuple[bool, str]:
    """Determine whether to use document segmentation."""
    seg_config = get_document_segmentation_config(config_path)

    if not seg_config["enabled"]:
        return False, "Document segmentation disabled in configuration"

    doc_size = len(document_content)
    threshold = seg_config["size_threshold_chars"]

    if doc_size > threshold:
        return True, f"Document size ({doc_size:,} chars) exceeds threshold ({threshold:,} chars)"
    return False, f"Document size ({doc_size:,} chars) below threshold ({threshold:,} chars)"


def get_adaptive_agent_config(
    use_segmentation: bool, search_server_names: list = None
) -> Dict[str, list]:
    """Get adaptive agent configuration based on segmentation usage."""
    if search_server_names is None:
        search_server_names = []

    config = {
        "concept_analysis": [],
        "algorithm_analysis": search_server_names.copy(),
        "code_planner": search_server_names.copy(),
    }

    if use_segmentation:
        config["concept_analysis"] = ["document-segmentation"]
        for key in ("algorithm_analysis", "code_planner"):
            if "document-segmentation" not in config[key]:
                config[key].append("document-segmentation")
    else:
        config["concept_analysis"] = ["filesystem"]
        for key in ("algorithm_analysis", "code_planner"):
            if "filesystem" not in config[key]:
                config[key].append("filesystem")

    return config


def get_adaptive_prompts(use_segmentation: bool) -> Dict[str, str]:
    """Get appropriate prompt versions based on segmentation usage."""
    from deepcode.prompts.sys_prompts import (
        PAPER_CONCEPT_ANALYSIS_PROMPT,
        PAPER_ALGORITHM_ANALYSIS_PROMPT,
        CODE_PLANNING_PROMPT,
        PAPER_CONCEPT_ANALYSIS_PROMPT_TRADITIONAL,
        PAPER_ALGORITHM_ANALYSIS_PROMPT_TRADITIONAL,
        CODE_PLANNING_PROMPT_TRADITIONAL,
    )

    if use_segmentation:
        return {
            "concept_analysis": PAPER_CONCEPT_ANALYSIS_PROMPT,
            "algorithm_analysis": PAPER_ALGORITHM_ANALYSIS_PROMPT,
            "code_planning": CODE_PLANNING_PROMPT,
        }
    return {
        "concept_analysis": PAPER_CONCEPT_ANALYSIS_PROMPT_TRADITIONAL,
        "algorithm_analysis": PAPER_ALGORITHM_ANALYSIS_PROMPT_TRADITIONAL,
        "code_planning": CODE_PLANNING_PROMPT_TRADITIONAL,
    }
