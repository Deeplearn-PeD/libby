from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn
from typing import Dict, List, Any


def _default_wiki_base_path() -> str:
    """Prefer the persistent /data volume in container deployments.

    The default home-directory path lives inside the container filesystem,
    which is wiped on every image rebuild — losing all wiki pages and
    knowledge graphs. Standard deployments mount a volume at /data.
    """
    if Path("/data").is_dir():
        return "/data/wikis"
    return str(Path.home() / ".libby" / "wikis")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


    languages: Dict[str, Dict[str, Any]] = {
        "English": {"code": "en_US", "is_default": True},
        "Português": {"code": "pt_BR"},
    }

    models: Dict[str, Dict[str, Any]] = {
        "Llama3": {"code": "llama3.2"},
        "Gemma": {"code": "gemma3"},
        "GLM": {"code": "glm-5-turbo"},
        "Qwen": {"code": "qwen3.5:4b"},
        "Kimi": {"code": "kimi-k2.6"},
        "DeepSeek": {"code": "deepseek-flash", "is_default": True},
    }

    embedding_models: Dict[str, Dict[str, Any]] = {
        "GemmaEmbedding": {"code": "embeddinggemma", "is_default": True},
        "Mxbai": {"code": "mxbai-embed-large"},
        "Gemini": {"code": "gemini-embedding-001"},
    }

    wiki_base_path: str = Field(
        default_factory=lambda: _default_wiki_base_path(),
        description=(
            "Base directory for LLM wikis. Defaults to /data/wikis when a "
            "/data volume exists (container deployments) so wikis survive "
            "container recreation; otherwise ~/.libby/wikis."
        ),
    )

    wiki_auto_ingest: bool = Field(
        default=True,
        description="Automatically ingest documents into the wiki after embedding",
    )

    wiki_model: str = Field(
        default="deepseek-flash",
        description=(
            "Dedicated LLM model for wiki ingest/query/summary. Empty falls "
            "back to the default chat model. Structured-output callers "
            "automatically disable DeepSeek thinking mode (extra_body "
            "thinking:disabled) since thinking rejects pydantic-ai "
            "tool_choice and yields sparse summaries; plain chat keeps "
            "thinking on."
        ),
    )

    wiki_graph_enabled: bool = Field(
        default=True,
        description="Maintain a knowledge graph over wiki pages and embedded chunks",
    )

    wiki_graph_communities: bool = Field(
        default=True,
        description="Detect topic communities (Louvain) in the wiki knowledge graph",
    )

    wiki_graph_resolution: float = Field(
        default=1.0,
        description=(
            "Louvain resolution for community detection; higher values yield "
            "more, smaller communities"
        ),
    )

    wiki_graph_hub_percentile: int = Field(
        default=95,
        description=(
            "Degree percentile above which a page node counts as a super-hub; "
            "super-hubs get a damped degree bonus in query ranking"
        ),
    )

    db_url: str = Field(
        default_factory=lambda: f"sqlite:///{Path.home() / '.libby' / 'data' / 'libby.db'}",
        description="Database URL for chat history",
    )

    embed_db_url: str = Field(
        default_factory=lambda: f"sqlite:///{Path.home() / '.libby' / 'data' / 'embeddings.db'}",
        description="Database URL for embeddings",
    )

    target_postgres_url: str = Field(
        default_factory=lambda: os.getenv("PGURL", ""),
        description="Pre-configured PostgreSQL target URL for migration",
    )

    target_duckdb_path: str = Field(
        default_factory=lambda: str(Path.home() / ".libby" / "data" / "embeddings.duckdb"),
        description="Pre-configured DuckDB target path for migration",
    )

    target_sqlite_path: str = Field(
        default_factory=lambda: str(Path.home() / ".libby" / "data" / "embeddings.db"),
        description="Pre-configured SQLite target path for migration",
    )

    @property
    def default_model(self) -> str:
        for model, details in self.models.items():
            if details.get("is_default"):
                return details["code"]
        return "deepseek-flash"

    @property
    def default_embedding_model(self) -> str:
        for model, details in self.embedding_models.items():
            if details.get("is_default"):
                return details["code"]
        return "embeddinggemma"
