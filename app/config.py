from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Any


class Settings(BaseSettings):
    qdrant_address: str
    qdrant_memory_colname: str
    qdrant_base_colname: str
    max_memory_context: int
    max_doc_context: int
    dense_vec_size: int

    llm_address: str
    llm_model_name: str
    llm_api_key: str | None = None

    emb_address: str
    emb_model_name: str
    emb_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file="./.env",
        env_file_encoding="utf-8"
    )

    def model_post_init(self, __context: Any) -> None:
        if self.llm_api_key is None:
            self.llm_api_key = "sk-no-key-required"
        if self.emb_api_key is None:
            self.emb_api_key = "sk-no-key-required"


@lru_cache
def get_settings():
    return Settings()


