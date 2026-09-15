"""Application settings, read exclusively from the process environment.

This module never opens, reads, or logs `.env`. If `.env` needs to be loaded
into the process environment for local development, that happens once, in
the CLI entrypoint (ingest/run.py), before Settings() is constructed.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")  # no env_file= here, deliberately

    evds_api_key: SecretStr = Field(alias="EVDS_API_KEY")

    def evds_auth_header(self) -> dict[str, str]:
        """The one place the raw key value is materialized, for the outgoing header only."""
        return {"key": self.evds_api_key.get_secret_value()}


def get_settings() -> Settings:
    return Settings()
