"""Application settings, read exclusively from the process environment.

This module never opens, reads, or logs `.env`. If `.env` needs to be loaded
into the process environment for local development, that happens once, in
the CLI entrypoint (ingest/run.py), before Settings() is constructed.
"""

from pathlib import Path

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


class AgentSettings(BaseSettings):
    """Agent-lane settings: MIA access, model ids, data paths.

    A separate class from `Settings` on purpose. `Settings` requires
    EVDS_API_KEY and there is a test asserting it fails without one, but agent
    code and its tests must construct settings with no EVDS key in the
    environment at all. Keeping two classes costs one extra import and removes
    any need to change the data lane's contract.

    Same discipline as `Settings`: no `env_file`, the process environment is the
    only source. `.env` is loaded once in a CLI/serving entrypoint, never by
    library code and never by a test.
    """

    model_config = SettingsConfigDict(extra="ignore")  # no env_file= here, deliberately

    mia_api_key: SecretStr = Field(alias="MIA_API_KEY")
    mia_base_url: str = Field("https://mia.csp.kloudeks.com/v1", alias="MIA_BASE_URL")
    #: DeepSeek since 22 Sep (docs/decisions.md). Measured on the scenario's
    #: three questions over the 45k-series pool: Qwen3.8-27B took 78 s on the
    #: first turn and returned an empty answer, and misstated a unit on the
    #: second; DeepSeek finished all three (first turn 15 s, answers 2-3 s)
    #: with the right months and units. Qwen is still one env var away:
    #: MIA_CHAT_MODEL=kkbhackathon2026/Qwen3.8-27B.
    mia_chat_model: str = Field(
        "deepseek-ai/DeepSeek-V4.1-Flash", alias="MIA_CHAT_MODEL"
    )
    mia_embed_model: str = Field(
        "kkbhackathon2026/Qwen3-Embedding-8B", alias="MIA_EMBED_MODEL"
    )
    mia_ocr_model: str = Field("kkbhackathon2026/Unlimited-OCR", alias="MIA_OCR_MODEL")

    data_dir: Path = Field(Path("data"), alias="DATA_DIR")

    # The HTTP timeout of a call with no budget of its own (embedding, OCR),
    # and the agent loop's cap on one planning or observing call and on a
    # thinking-on answer attempt, so a slow call leaves the turn time for what
    # follows. A last-chance answer is not held to it: it gets whatever the
    # turn has left. A timed-out request is not retried (llm._is_retryable),
    # so this is no longer multiplied by three: the old "3 x 45 = 135s fits in
    # 180s" ignored both the backoff sleeps and the time the turn had already
    # spent, and one call really did take 138s.
    llm_timeout_s: float = Field(45.0, alias="LLM_TIMEOUT_S")

    @property
    def usage_log_path(self) -> Path:
        return self.data_dir / "logs" / "llm_usage.jsonl"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def adhoc_dir(self) -> Path:
        """Cache for data fetched at demo time; never written back into gold."""
        return self.data_dir / "adhoc"

    def mia_auth_key(self) -> str:
        """The one place the raw MIA key is materialized, for the client only."""
        return self.mia_api_key.get_secret_value()


def get_agent_settings() -> AgentSettings:
    """Build agent settings from the process environment.

    Library code does not call this; it takes an `AgentSettings` parameter so
    tests can pass one in without touching the environment. Only entrypoints
    (the llm CLI, the Chainlit app, the eval runner) call it.
    """
    return AgentSettings()
