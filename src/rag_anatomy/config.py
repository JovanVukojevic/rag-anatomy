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
