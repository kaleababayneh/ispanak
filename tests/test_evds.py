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


@pytest.fixture(autouse=True)
def _isolated_bronze(monkeypatch, tmp_path):
    """Every test runs in its own directory. The bronze paths are relative,
    and the auth/retry tests used to append their fake calls to the real
    `data/bronze/evds/fetch_log.jsonl` -- 574 lines of it by 22 Sep, mixed in
    with the log of real fetches. Fixtures are read by absolute path."""
    monkeypatch.chdir(tmp_path)


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


# --- _parse_tarih: one case per frequency family ----------------------------


def test_parse_tarih_daily_weekly_business_day_format():
    assert evds._parse_tarih("01-01-2024") == evds.date(2024, 1, 1)
    assert evds._parse_tarih("04-10-2024") == evds.date(2024, 10, 4)


def test_parse_tarih_monthly_format_no_leading_zero():
    assert evds._parse_tarih("2024-1") == evds.date(2024, 1, 1)
    assert evds._parse_tarih("2024-12") == evds.date(2024, 12, 1)


def test_parse_tarih_quarterly_format():
    assert evds._parse_tarih("2024-Q1") == evds.date(2024, 1, 1)
    assert evds._parse_tarih("2024-Q3") == evds.date(2024, 7, 1)


def test_parse_tarih_yearly_format():
    assert evds._parse_tarih("2024") == evds.date(2024, 1, 1)


def test_parse_tarih_rejects_unknown_format():
    with pytest.raises(ValueError):
        evds._parse_tarih("not-a-date")


# --- detect_truncation -------------------------------------------------------


def test_detect_truncation_false_for_weekly_group_overrunning_year_end():
    # 2024 requested; last row spills into the next year's first Friday --
    # that is normal (Friday-anchored weeks don't align to Dec 31), not truncation.
    df = pd.DataFrame({"Tarih": ["05-01-2024", "27-12-2024", "03-01-2025"]})
    assert evds.detect_truncation(df, evds.date(2024, 1, 1)) is False


def test_detect_truncation_true_when_row_count_hits_cap_and_first_date_is_late():
    # Measured incident: 2021-2026 requested, 1000-row cap, response started 2023-12-20.
    df = pd.DataFrame({"Tarih": ["20-12-2023"] * 1000})
    assert evds.detect_truncation(df, evds.date(2021, 1, 1)) is True


def test_detect_truncation_false_for_legitimately_late_starting_series():
    """Regression: TP.HPBITABLO6.3 measured to start 2024-06-01. The old
    "first row far after expected_start" heuristic alone flagged this as
    truncated -- with only a handful of rows (nowhere near the 1000 cap),
    this is just a series that started late that year, not truncation."""
    df = pd.DataFrame({"Tarih": ["01-06-2024", "08-06-2024", "15-06-2024"]})
    assert evds.detect_truncation(df, evds.date(2024, 1, 1)) is False


def test_detect_truncation_true_when_total_count_exceeds_returned_rows():
    """En güçlü sinyal: yanıtın kendi totalCount alanı dönen satır sayısından
    fazlaysa, ilk tarih 1000 satır eşiğine bakmadan kesme kabul edilir."""
    df = pd.DataFrame({"Tarih": ["05-01-2021", "12-01-2021"]})
    assert evds.detect_truncation(df, evds.date(2021, 1, 1), total_count=1500) is True


def test_detect_truncation_false_for_empty_dataframe():
    assert evds.detect_truncation(pd.DataFrame(), evds.date(2024, 1, 1)) is False


# --- fetch_datagroup_year ----------------------------------------------------


def test_fetch_datagroup_year_is_idempotent_skips_http_when_cached(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)
    out_path = evds._datagroup_year_path("bie_kfe", 2024)
    out_path.parent.mkdir(parents=True)
    pd.DataFrame({"Tarih": ["2024-1"], "TP_KFE_TR": ["150.0"]}).to_parquet(out_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("should not hit the network when cached")

    monkeypatch.setattr(evds, "_authenticated_get", fail_if_called)

    result = evds.fetch_datagroup_year("bie_kfe", 2024, settings)

    assert result.outcome == "cached"


def test_fetch_datagroup_year_writes_parquet_on_ok(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)
    rows = [
        {"Tarih": "2024-1", "TP_KFE_TR": "150.0"},
        {"Tarih": "2024-2", "TP_KFE_TR": "151.0"},
    ]
    monkeypatch.setattr(evds, "_authenticated_get", lambda *_a, **_k: rows)

    result = evds.fetch_datagroup_year("bie_kfe", 2024, settings)

    assert result.outcome == "ok"
    df = pd.read_parquet(result.path)
    assert len(df) == 2
    assert "_fetched_at" in df.columns


def test_fetch_datagroup_year_empty_response_writes_no_parquet(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)
    monkeypatch.setattr(evds, "_authenticated_get", lambda *_a, **_k: [])

    result = evds.fetch_datagroup_year("bie_hpbitablo6", 2021, settings)

    assert result.outcome == "empty"
    assert not evds._datagroup_year_path("bie_hpbitablo6", 2021).exists()


def test_an_empty_year_is_remembered_so_a_rerun_makes_no_call(monkeypatch, tmp_path):
    """CLAUDE.md #6: the second run makes zero HTTP calls. An empty year wrote
    nothing, so every rerun asked for it again."""
    settings = _settings(monkeypatch)
    calls = []

    def empty(path, _settings):
        calls.append(path)
        return []

    monkeypatch.setattr(evds, "_authenticated_get", empty)

    evds.fetch_datagroup_year("bie_eski", 2021, settings)
    second = evds.fetch_datagroup_year("bie_eski", 2021, settings)

    assert len(calls) == 1
    assert second.outcome == "cached"
    assert not list(tmp_path.rglob("*.parquet"))  # silver reads parquet only


def test_a_year_that_times_out_is_fetched_again_in_quarters(monkeypatch, tmp_path):
    """The previous full run lost 34 datagroup-years (large daily groups) to
    timeouts that an identical second round did not recover."""
    settings = _settings(monkeypatch)
    calls = []

    def slow_for_a_year(path, _settings):
        calls.append(path)
        if "startDate=01-01-2024&endDate=31-12-2024" in path:
            raise evds.EvdsTimeoutError("slow")
        start = path.split("startDate=")[1].split("&")[0]
        return [{"Tarih": start, "TP_X": "1.0"}]

    monkeypatch.setattr(evds, "_authenticated_get", slow_for_a_year)

    result = evds.fetch_datagroup_year("bie_buyuk", 2024, settings)

    assert result.outcome == "ok"
    assert result.row_count == 4
    folder = evds.DATAGROUPS_DATA_ROOT / "bie_buyuk"
    assert sorted(p.name for p in folder.glob("*.parquet")) == [
        "2024-Q1.parquet",
        "2024-Q2.parquet",
        "2024-Q3.parquet",
        "2024-Q4.parquet",
    ]
    assert "startDate=01-10-2024&endDate=31-12-2024" in calls[-1]

    calls.clear()
    assert evds.fetch_datagroup_year("bie_buyuk", 2024, settings).outcome == "cached"
    assert calls == []


def test_a_quarter_that_times_out_is_fetched_in_months(monkeypatch, tmp_path):
    settings = _settings(monkeypatch)

    def slow_unless_a_month(path, _settings):
        start = path.split("startDate=")[1].split("&")[0]
        end = path.split("endDate=")[1].split("&")[0]
        if start[3:5] != end[3:5]:  # more than one month asked
            raise evds.EvdsTimeoutError("slow")
        return [{"Tarih": start, "TP_X": "1.0"}]

    monkeypatch.setattr(evds, "_authenticated_get", slow_unless_a_month)

    result = evds.fetch_datagroup_year("bie_cok_buyuk", 2024, settings)

    assert result.outcome == "ok"
    names = {p.stem for p in (evds.DATAGROUPS_DATA_ROOT / "bie_cok_buyuk").glob("*")}
    assert names == {f"2024-{m:02d}" for m in range(1, 13)}


def test_a_month_that_still_times_out_is_a_failed_year(monkeypatch, tmp_path):
    settings = _settings(monkeypatch)

    def always_slow(path, _settings):
        raise evds.EvdsTimeoutError("slow")

    monkeypatch.setattr(evds, "_authenticated_get", always_slow)

    result = evds.fetch_datagroup_year("bie_olu", 2024, settings)

    assert result.outcome == "failed"


def test_a_timeout_is_logged_and_raised_as_its_own_error(monkeypatch, tmp_path):
    settings = _settings(monkeypatch)

    def timeout(*_args, **_kwargs):
        raise evds.requests.exceptions.Timeout("read timeout")

    monkeypatch.setattr(evds.requests, "get", timeout)

    with pytest.raises(evds.EvdsTimeoutError):
        evds._authenticated_get("datagroup=bie_x&type=json", settings)

    logged = [
        json.loads(line) for line in evds.FETCH_LOG_JSONL_PATH.read_text().splitlines()
    ]
    assert [entry["outcome"] for entry in logged] == ["timeout"]


def test_fetch_datagroup_year_truncated_response_writes_no_parquet(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)
    payload = {
        "totalCount": 1850,
        "items": [{"Tarih": "20-12-2023", "TP_KTF12": "40.0"}] * 1000,
    }
    monkeypatch.setattr(evds, "_authenticated_get", lambda *_a, **_k: payload)

    result = evds.fetch_datagroup_year("bie_kt100h", 2021, settings)

    assert result.outcome == "truncated"
    assert not evds._datagroup_year_path("bie_kt100h", 2021).exists()


def test_fetch_datagroup_year_legitimately_late_starting_series_writes_parquet(
    monkeypatch, tmp_path
):
    """Regression: TP.HPBITABLO6.3 measured to start 2024-06-01 -- the old
    "first row far after expected_start" heuristic alone flagged this whole
    year as truncated and never wrote its parquet, silently losing every
    real observation. row_count is nowhere near the 1000 cap, so this must
    write successfully."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)
    rows = [
        {"Tarih": "01-06-2024", "TP_HPBITABLO6_3": "100.0"},
        {"Tarih": "08-06-2024", "TP_HPBITABLO6_3": "101.0"},
    ]
    monkeypatch.setattr(evds, "_authenticated_get", lambda *_a, **_k: rows)

    result = evds.fetch_datagroup_year("bie_hpbitablo6", 2024, settings)

    assert result.outcome == "ok"
    assert result.path is not None
    assert evds._datagroup_year_path("bie_hpbitablo6", 2024).exists()


def test_fetch_datagroup_year_network_error_is_caught_and_reported(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)

    def raise_connection_error(*_args, **_kwargs):
        raise evds.requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(evds, "_authenticated_get", raise_connection_error)

    result = evds.fetch_datagroup_year("bie_kfe", 2024, settings)

    assert result.outcome == "failed"


def test_fetch_datagroup_year_403_propagates_and_is_not_swallowed(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)

    def raise_auth_error(*_args, **_kwargs):
        raise evds.EvdsAuthError("nope")

    monkeypatch.setattr(evds, "_authenticated_get", raise_auth_error)

    with pytest.raises(evds.EvdsAuthError):
        evds.fetch_datagroup_year("bie_kfe", 2024, settings)


# --- fetch_datagroups_years: continues past failures, stops on 403 ----------


def test_fetch_datagroups_years_continues_past_failed_groups(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)

    def fake_authenticated_get(path, _settings):
        if "bie_broken" in path:
            raise evds.EvdsTimeoutError("slow")
        return [{"Tarih": "2024-1", "X": "1.0"}]

    monkeypatch.setattr(evds, "_authenticated_get", fake_authenticated_get)

    results = evds.fetch_datagroups_years(["bie_broken", "bie_kfe"], [2024], settings)

    outcomes = {r.datagroup_code: r.outcome for r in results}
    assert outcomes["bie_broken"] == "failed"
    assert outcomes["bie_kfe"] == "ok"


def test_fetch_datagroups_years_stops_immediately_on_403(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = _settings(monkeypatch)
    calls = []

    def fake_authenticated_get(path, _settings):
        calls.append(path)
        raise evds.EvdsAuthError("nope")

    monkeypatch.setattr(evds, "_authenticated_get", fake_authenticated_get)

    with pytest.raises(evds.EvdsAuthError):
        evds.fetch_datagroups_years(["bie_a", "bie_b"], [2024], settings)

    assert calls == [
        "datagroup=bie_a&startDate=01-01-2024&endDate=31-12-2024&type=json"
    ]


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
