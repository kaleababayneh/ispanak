"""Tests for lakehouse_agent.ingest.evds. No network access — every HTTP call
is monkeypatched; JSON fixtures under tests/fixtures/evds/ hold trimmed real
(or, where the real capture was corrupt, hand-written synthetic) EVDS
responses."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from lakehouse_agent.config import Settings
from lakehouse_agent.ingest import evds

FIXTURES = Path(__file__).parent / "fixtures" / "evds"


def _load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _settings(monkeypatch, key: str = "test-only-not-a-real-key") -> Settings:
    monkeypatch.setenv("EVDS_API_KEY", key)
    return Settings()


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Rate-limit and backoff sleeps must not slow the test suite down."""
    monkeypatch.setattr(evds.time, "sleep", lambda *_args, **_kwargs: None)


# --- build_url / redact_url ------------------------------------------------


def test_build_url_is_path_style_not_query_string():
    url = evds.build_url(evds.DATAGROUPS_PATH)
    assert url == "https://evds3.tcmb.gov.tr/igmevdsms-dis/datagroups/mode=0&type=json"
    assert "?" not in url


def test_redact_url_masks_key():
    url = (
        "https://evds3.tcmb.gov.tr/igmevdsms-dis/categories/type=json&key=super-secret"
    )
    assert evds.redact_url(url) == (
        "https://evds3.tcmb.gov.tr/igmevdsms-dis/categories/type=json&key=***"
    )


# --- _authenticated_get: auth, redirects, retries ---------------------------


def test_authenticated_get_sends_key_in_header_not_url(monkeypatch):
    captured = {}

    def fake_get(url, headers, allow_redirects, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["allow_redirects"] = allow_redirects
        return _FakeResponse(200, payload=[{"a": 1}])

    monkeypatch.setattr(evds.requests, "get", fake_get)
    settings = _settings(monkeypatch, key="a-fake-test-key")

    evds._authenticated_get(evds.CATEGORIES_PATH, settings)

    assert captured["headers"] == {"key": "a-fake-test-key"}
    assert "key=" not in captured["url"]
    assert captured["allow_redirects"] is False


def test_authenticated_get_raises_on_redirect_without_retry(monkeypatch):
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return _FakeResponse(302)

    monkeypatch.setattr(evds.requests, "get", fake_get)
    settings = _settings(monkeypatch)

    with pytest.raises(evds.EvdsRedirectError):
        evds._authenticated_get(evds.CATEGORIES_PATH, settings)
    assert len(calls) == 1


def test_authenticated_get_raises_auth_error_on_403_without_retry(monkeypatch):
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return _FakeResponse(403)

    monkeypatch.setattr(evds.requests, "get", fake_get)
    settings = _settings(monkeypatch)

    with pytest.raises(evds.EvdsAuthError):
        evds._authenticated_get(evds.CATEGORIES_PATH, settings)
    assert len(calls) == 1  # 403 is not retried


def test_authenticated_get_retries_retryable_status_then_succeeds(monkeypatch):
    responses = iter(
        [_FakeResponse(500), _FakeResponse(500), _FakeResponse(200, payload=[{"a": 1}])]
    )
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return next(responses)

    monkeypatch.setattr(evds.requests, "get", fake_get)
    settings = _settings(monkeypatch)

    result = evds._authenticated_get(evds.CATEGORIES_PATH, settings)

    assert result == [{"a": 1}]
    assert len(calls) == 3


def test_authenticated_get_gives_up_after_max_retries(monkeypatch):
    def fake_get(*_args, **_kwargs):
        return _FakeResponse(500)

    monkeypatch.setattr(evds.requests, "get", fake_get)
    settings = _settings(monkeypatch)

    with pytest.raises(evds.EvdsRequestError):
        evds._authenticated_get(evds.CATEGORIES_PATH, settings)


# --- fetch_log: real append + compaction ------------------------------------


def test_log_call_appends_one_line_per_call(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    call = evds.ApiCall("https://x/y", "y", 200, 3, "2026-09-13T00:00:00+00:00", "ok")

    evds._log_call(call)
    evds._log_call(call)
    evds._log_call(call)

    lines = evds.FETCH_LOG_JSONL_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["outcome"] == "ok" for line in lines)


def test_compact_fetch_log_builds_parquet_from_jsonl(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    evds._log_call(evds.ApiCall("https://x/a", "a", 200, 1, "t1", "ok"))
    evds._log_call(evds.ApiCall("https://x/b", "b", 403, None, "t2", "auth_error"))

    path = evds.compact_fetch_log()

    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df["outcome"]) == {"ok", "auth_error"}


def test_compact_fetch_log_with_no_jsonl_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = evds.compact_fetch_log()
    assert not path.exists()


# --- column name reversal ----------------------------------------------------


def test_denormalize_response_columns_round_trips_catalog_codes():
    serielist = _load_fixture("serielist_sample.json")
    serie_codes = [row["SERIE_CODE"] for row in serielist]
    assert any(code.count(".") >= 2 for code in serie_codes)  # multi-dot code present

    mangled = {code.replace(".", "_"): [1, 2] for code in serie_codes}
    df = pd.DataFrame({"Tarih": ["2024-1", "2024-2"], **mangled})

    result = evds.denormalize_response_columns(df, serie_codes)

    assert set(result.columns) == {"Tarih", *serie_codes}


# --- build_catalog: assembly + idempotency ----------------------------------


def test_build_catalog_assembles_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    categories = _load_fixture("categories_sample.json")
    datagroups = _load_fixture("datagroups_sample.json")
    istirakbs_series = _load_fixture("serielist_sample.json")
    dbb_series = [
        {
            "SERIE_CODE": "TP.DBB.A1",
            "DATAGROUP_CODE": "bie_dbb",
            "SERIE_NAME": "Test Series",
        }
    ]

    calls = []

    def fake_authenticated_get(path, _settings):
        calls.append(path)
        if path == evds.CATEGORIES_PATH:
            return categories
        if path == evds.DATAGROUPS_PATH:
            return datagroups
        if path == evds.SERIELIST_PATH_TMPL.format(code="bie_istirakbs"):
            return istirakbs_series
        if path == evds.SERIELIST_PATH_TMPL.format(code="bie_dbb"):
            return dbb_series
        raise AssertionError(f"unexpected path requested: {path}")

    monkeypatch.setattr(evds, "_authenticated_get", fake_authenticated_get)
    settings = _settings(monkeypatch)

    catalog_path = evds.build_catalog(settings)

    assert catalog_path == evds.CATALOG_PATH
    catalog = pd.read_parquet(catalog_path)
    expected_codes = {row["SERIE_CODE"] for row in istirakbs_series} | {"TP.DBB.A1"}
    assert set(catalog["SERIE_CODE"]) == expected_codes
    istirakbs_rows = catalog[catalog["DATAGROUP_CODE"] == "bie_istirakbs"]
    assert (istirakbs_rows["TOPIC_TITLE_TR"] == "İŞTİRAKLER").all()

    calls.clear()
    second_path = evds.build_catalog(settings)

    assert second_path == catalog_path
    assert calls == []  # second run makes zero HTTP calls


# --- Settings: never reads .env directly -------------------------------------


def test_settings_reads_only_process_environment_not_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "EVDS_API_KEY=value-from-dotenv-file\n", encoding="utf-8"
    )
    monkeypatch.delenv("EVDS_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings()  # .env is never loaded implicitly -> required field missing

    monkeypatch.setenv("EVDS_API_KEY", "value-from-environment")
    settings = Settings()
    assert settings.evds_api_key.get_secret_value() == "value-from-environment"
