from urllib.parse import quote

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_", env_file=".env", extra="ignore"
    )

    user: str
    password: SecretStr
    db: str
    host: str
    port: int

    @property
    def dsn(self) -> str:
        credentials = f"{quote(self.user, safe='')}:{quote(self.password.get_secret_value(), safe='')}"
        return f"postgresql://{credentials}@{self.host}:{self.port}/{quote(self.db, safe='')}"


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_", env_file=".env", extra="ignore"
    )

    api_key: SecretStr = Field(min_length=1)
    embedding_model: str = Field("text-embedding-3-small", min_length=1)
    embedding_dimensions: int = Field(1536, gt=0)
    timeout: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0)


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_", env_file=".env", extra="ignore"
    )

    enabled: bool = False
    candidate_pool: int = Field(20, gt=0)
    timeout: float = Field(210.0, gt=0)


class RerankerServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_", env_file=".env", extra="ignore"
    )

    host: str = Field(min_length=1)
    port: int
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
