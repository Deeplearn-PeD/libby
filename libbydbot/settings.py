from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Any

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    languages: Dict[str, Dict[str, Any]] = {
        "English": {"code": "en_US", "is_default": True},
        "Português": {"code": "pt_BR"}
    }
    
    models: Dict[str, Dict[str, Any]] = {
        "Llama3": {"code": "llama3.2", "is_default": True},
        "Gemma": {"code": "gemma3"},
        "Llama3-vision": {"code": "llama3.2-vision"},
        "ChatGPT": {"code": "gpt-4o"},
        "Qwen": {"code": "qwen3"}
    }

    @property
    def default_model(self) -> str:
        for model, details in self.models.items():
            if details.get("is_default"):
                return details["code"]
        return "llama3.2"

settings = Settings()
