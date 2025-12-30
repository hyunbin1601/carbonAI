"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the Hooxi carbon emission management agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    # Model configuration for 5-node structure
    model_classify: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="claude-sonnet-4-5-20250929",
        metadata={
            "description": "Model for classify node (intent understanding). Uses Claude Sonnet 4.5 for better reasoning."
        },
    )

    model_generate: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="claude-haiku-4-5-20251001",
        metadata={
            "description": "Model for generate node (response generation). Uses Claude Haiku 4.5 for speed and cost."
        },
    )

    model_verify: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="claude-sonnet-4-5-20250929",
        metadata={
            "description": "Model for verify node (quality verification). Uses Claude Sonnet 4.5 for better judgment."
        },
    )

    # Legacy model field (for backward compatibility)
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="claude-sonnet-4-5-20250929",
        metadata={
            "description": "Default model (backward compatibility). Uses Claude Sonnet 4.5."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    # Quality verification threshold
    quality_threshold: int = field(
        default=70,
        metadata={
            "description": "Minimum quality score (0-100) required to pass verification. "
            "Responses below this score will be regenerated."
        },
    )

    # Hooxi-specific configuration
    enable_artifacts: bool = field(
        default=True,
        metadata={
            "description": "Enable artifact generation (charts, calculators, documents)."
        },
    )

    chromadb_path: str = field(
        default="./chroma_db",
        metadata={
            "description": "Path to ChromaDB vector store for emission factors."
        },
    )

    brand_color_primary: str = field(
        default="#0D9488",
        metadata={
            "description": "Primary Hooxi brand color (Teal)."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from the current context."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
