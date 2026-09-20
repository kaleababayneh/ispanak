"""Shared fixtures.

The important one is `_no_secrets_in_env`: it runs for every test and strips the
project's API keys out of the environment. A unit test that quietly reaches the
real MIA or EVDS would be slow, flaky and would burn quota we need for the demo,
so the suite is built so that it cannot happen by accident. Tests that need a
key set one explicitly.

The fake-layer fixtures import inside their own bodies rather than at module
level, so this file still collects cleanly in a tree where
`lakehouse_agent.testing` is not present yet.
"""

import pandas as pd
import pytest

from lakehouse_agent.config import AgentSettings


@pytest.fixture(autouse=True)
def _no_secrets_in_env(monkeypatch):
    """No test may inherit a real key from the developer's shell."""
    for name in ("MIA_API_KEY", "EVDS_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def agent_settings(monkeypatch, tmp_path) -> AgentSettings:
    """Settings with a throwaway key and a per-test data directory."""
    monkeypatch.setenv("MIA_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return AgentSettings()


@pytest.fixture
def fake_llm():
    from lakehouse_agent.testing.fake_llm import FakeLLM

    return FakeLLM()


@pytest.fixture
def fake_lake():
    from lakehouse_agent.testing.fake_lake import FakeLake

    return FakeLake()


@pytest.fixture
def scenario_wide(fake_lake) -> pd.DataFrame:
    """The scenario's four series, monthly, 2021-01 .. 2025-12: 60 rows."""
    from lakehouse_agent.testing.fake_lake import (
        CPI_SERIES,
        CREDIT_SERIES,
        HPI_SERIES,
        RATE_SERIES,
    )

    return fake_lake.wide(
        [CREDIT_SERIES, RATE_SERIES, CPI_SERIES, HPI_SERIES], "2021-01", "2025-12"
    )
