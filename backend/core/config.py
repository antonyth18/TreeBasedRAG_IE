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
    DEFAULT_LLM_MODEL: str = "llama3.2"
    
    class Config:
        case_sensitive = True

settings = Settings()
