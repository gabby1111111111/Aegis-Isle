import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8001)
    api_reload: bool = Field(default=True)

    # Database Configuration
    database_url: str = Field(default="sqlite:///./aegis_isle.db")
    vector_db_type: str = Field(default="qdrant")
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="aegis_isle_documents")

    # LLM Configuration
    llm_provider: str = Field(default="openai")
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default=None)
    
    # Model Settings
    default_llm_model: str = Field(default="gpt-4-1106-preview")
    embedding_model: str = Field(default="text-embedding-ada-002")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)

    # RAG Configuration
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_retrieved_docs: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)

    # Agent Configuration
    max_agent_iterations: int = Field(default=10)
    agent_timeout: int = Field(default=300)
    enable_memory: bool = Field(default=True)
    st_sovits_webhook_url: str = Field(
        default="http://127.0.0.1:8000/api/plugins/companion-link/trigger_call"
    )
    ntfy_topic_ring: str = Field(default="gabby-ring")

    # File Processing
    upload_max_size: str = Field(default="50MB")
    supported_formats: str = Field(default="pdf,docx,txt,md,html")
    ocr_enabled: bool = Field(default=True)
    ocr_language: str = Field(default="eng+chi_sim")

    # Security & Authentication
    secret_key: str = Field(default="change-this-in-production")
    access_token_expire_minutes: int = Field(default=30)
    allowed_hosts: str = Field(default="localhost,127.0.0.1")

    # OAuth2 + RBAC Configuration
    admin_username: str = Field(default="admin")
    admin_password: str = Field(default="admin123")
    jwt_algorithm: str = Field(default="HS256")

    # Audit Logging Configuration
    audit_log_enabled: bool = Field(default=True)
    audit_log_retention_days: int = Field(default=365)
    structured_logging: bool = Field(default=True)
    elk_compatible: bool = Field(default=True)

    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    log_requests: bool = Field(default=True)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Multi-modal
    enable_multimodal: bool = Field(default=True)
    image_processing_enabled: bool = Field(default=True)
    vision_model: str = Field(default="gpt-4-vision-preview")

    # Computed properties
    @property
    def supported_formats_list(self) -> List[str]:
        """Get supported file formats as a list."""
        return [fmt.strip() for fmt in self.supported_formats.split(",")]

    @property
    def allowed_hosts_list(self) -> List[str]:
        """Get allowed hosts as a list."""
        return [host.strip() for host in self.allowed_hosts.split(",")]

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"

    @property
    def uploads_dir(self) -> Path:
        """Get uploads directory."""
        return self.data_dir / "uploads"

    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.project_root / "models"

    @property
    def config_dir(self) -> Path:
        """Get config directory."""
        return self.project_root / "config"


# Global settings instance
settings = Settings()
