"""集中配置：在应用启动前加载环境变量，避免模块导入顺序导致的配置丢失。"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """从环境变量或 backend/.env 读取配置。

    本地启动器使用 uvicorn --env-file .env.local，环境变量优先于 .env。
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str = ""
    openai_api_base: str = ""
    openai_model: str = "deepseek-v4.1-flash"
    openai_enable_thinking: str = ""
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "xiaodi_smart_oa"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_days: int = 30

    # Auth
    secret_key: str = "xiaodi-oa-default-secret-key"

    # HTTP
    allowed_origins: str = "http://localhost:3000"

    # Knowledge base
    kb_top_k: int = 6
    kb_min_score: float = 9.0

    # Chat rate limit
    chat_rate_limit: int = 60
    chat_rate_window_seconds: int = 300

    # Observability
    langsmith_tracing: str = "false"
    langsmith_api_key: str = ""
    langsmith_project: str = "lantu-agent-platform"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @property
    def tracing_enabled(self) -> bool:
        return self.langsmith_tracing.strip().lower() == "true"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
