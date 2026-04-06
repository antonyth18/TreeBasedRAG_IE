from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Eywa-AI RAG Backend"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    TREES_DIR: str = os.path.join(BASE_DIR, "my_tree")
    
    # LLM Config
    DEFAULT_LLM_MODEL: str = "qwen3:8b"
    SUMMARY_MODEL: str = "qwen2.5:3b"
    SUMMARY_MAX_TOKENS: int = 128
    SUMMARY_MAX_RETRIES: int = 1
    SUMMARY_RETRY_DELAY: float = 0.5
    SUMMARY_VERIFY_FAITHFULNESS: bool = False
    SUMMARY_MAX_VERIFICATION_RETRIES: int = 1
    OLLAMA_NUM_CTX: int = 8192

    # Web search fallback config
    ENABLE_WEB_SEARCH: bool = True
    WEB_SEARCH_THRESHOLD: float = 0.75
    WEB_SEARCH_N_RESULTS: int = 3
    
    class Config:
        case_sensitive = True

settings = Settings()
