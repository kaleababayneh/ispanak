"""Tests for AgentSettings.

The point of these: agent-lane code must be constructible and testable with no
secrets in the environment at all, and the data lane's `Settings` must keep
behaving exactly as it did before AgentSettings was added.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from lakehouse_agent.config import AgentSettings, Settings, get_agent_settings


@pytest.fixture(autouse=True)
def _clean_optional_env(monkeypatch):
    """conftest already strips the API keys; these are the optional overrides."""
    for name in (
        "MIA_BASE_URL",
        "MIA_CHAT_MODEL",
        "MIA_EMBED_MODEL",
        "MIA_OCR_MODEL",
        "DATA_DIR",
        "LLM_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)


def test_agent_settings_builds_without_an_evds_key(monkeypatch):
    """The whole reason AgentSettings exists as a separate class."""
    monkeypatch.setenv("MIA_API_KEY", "test-only-not-a-real-key")
    settings = AgentSettings()
    assert settings.mia_auth_key() == "test-only-not-a-real-key"


def test_data_lane_settings_still_requires_its_key(monkeypatch):
    """Adding AgentSettings must not change the data lane's contract."""
    monkeypatch.setenv("MIA_API_KEY", "test-only-not-a-real-key")
    with pytest.raises(ValidationError):
        Settings()


def test_agent_settings_requires_the_mia_key():
    with pytest.raises(ValidationError):
        AgentSettings()


def test_defaults_match_the_mia_guide(monkeypatch):
    """The model ids are copied verbatim from the MIA guide and Slack. The
    chat model is DeepSeek since 22 Sep (docs/decisions.md); Qwen stays
    reachable through MIA_CHAT_MODEL."""
    monkeypatch.setenv("MIA_API_KEY", "k")
    settings = AgentSettings()
    assert settings.mia_base_url == "https://mia.csp.kloudeks.com/v1"
    assert settings.mia_chat_model == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert settings.mia_embed_model == "kkbhackathon2026/Qwen3-Embedding-8B"
    assert settings.mia_ocr_model == "kkbhackathon2026/Unlimited-OCR"


def test_timeout_default_is_a_cap_that_can_bind(monkeypatch):
    """The timeout caps one call, not three: timeouts are not retried. A cap
    under the shortest attempt the client will start would refuse every call,
    and one over the turn budget would cap nothing."""
    from lakehouse_agent.agent.loop import AgentConfig
    from lakehouse_agent.llm import MIN_ATTEMPT_S

    monkeypatch.setenv("MIA_API_KEY", "k")
    timeout = AgentSettings().llm_timeout_s
    assert MIN_ATTEMPT_S < timeout < AgentConfig().total_budget_s


def test_derived_paths_hang_off_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MIA_API_KEY", "k")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = AgentSettings()
    assert settings.usage_log_path == tmp_path / "logs" / "llm_usage.jsonl"
    assert settings.gold_dir == tmp_path / "gold"
    assert settings.adhoc_dir == tmp_path / "adhoc"


def test_data_dir_defaults_to_relative_data(monkeypatch):
    monkeypatch.setenv("MIA_API_KEY", "k")
    assert AgentSettings().data_dir == Path("data")


def test_settings_never_load_dotenv_implicitly(monkeypatch, tmp_path):
    """A stray .env next to the process must not leak into settings."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MIA_API_KEY=value-from-dotenv\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        AgentSettings()

    monkeypatch.setenv("MIA_API_KEY", "value-from-environment")
    assert AgentSettings().mia_auth_key() == "value-from-environment"


def test_key_is_not_exposed_by_repr_or_dump(monkeypatch):
    """The raw key may only appear via mia_auth_key(), nowhere else."""
    secret = "super-secret-not-real"
    monkeypatch.setenv("MIA_API_KEY", secret)
    settings = AgentSettings()
    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in str(settings.model_dump())
    assert secret not in settings.model_dump_json()
    assert settings.mia_auth_key() == secret


def test_get_agent_settings_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MIA_API_KEY", "k")
    monkeypatch.setenv("MIA_CHAT_MODEL", "custom/model")
    assert get_agent_settings().mia_chat_model == "custom/model"
