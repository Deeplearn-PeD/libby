from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn
from typing import Dict, List, Any


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    # pgurl: PostgresDsn
    # duckurl: str
    # openai_api_key: str
    # google_api_key: str
    # ollama_key: str
    # language: str

    languages: Dict[str, Dict[str, Any]] = {
        "English": {"code": "en_US", "is_default": True},
        "Português": {"code": "pt_BR"},
    }

    models: Dict[str, Dict[str, Any]] = {
        "Llama3": {"code": "llama3.2", "is_default": True},
        "Gemma": {"code": "gemma3"},
        "ChatGPT": {"code": "gpt-4o"},
        "Qwen": {"code": "qwen3"},
    }

    embedding_models: Dict[str, Dict[str, Any]] = {
        "GemmaEmbedding": {"code": "embeddinggemma", "is_default": True},
        "Mxbai": {"code": "mxbai-embed-large"},
        "Gemini": {"code": "gemini-embedding-001"},
    }

    @property
    def default_model(self) -> str:
        for model, details in self.models.items():
            if details.get("is_default"):
                return details["code"]
        return "llama3.2"

    @property
    def default_embedding_model(self) -> str:
        for model, details in self.embedding_models.items():
            if details.get("is_default"):
                return details["code"]
        return "embeddinggemma"
