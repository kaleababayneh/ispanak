"""Tests for lake/quality.py. No network -- synthetic DataFrames throughout."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from lakehouse_agent.lake import quality

CATALOG_COLUMNS = ["series_id", "name_tr", "source", "freq"]


def _obs(series_id, periods, values, *, fetched_at="2026-09-18T00:00:00+00:00"):
    return pd.DataFrame(
        {
            "series_id": series_id,
            "period": periods,
            "freq": "M",
            "value": values,
            "unit": None,
            "source": "evds",
            "fetched_at": fetched_at,
        }
    )


def _monthly(start: str, n: int):
    return list(pd.date_range(start, periods=n, freq="MS").date)


def _weekly(start: str, n: int):
    return list(pd.date_range(start, periods=n, freq="7D").date)


def test_build_quality_report_monthly_no_gaps():
    periods = _monthly("2021-01-01", 6)
    obs = _obs("evds:X", periods, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    catalog = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "M",
            }
        ]
    )

    report = quality.build_quality_report(obs, catalog)

    row = report.iloc[0]
    assert row["n"] == 6
    assert row["missing"] == 0
    assert row["start"] == periods[0]
    assert row["end"] == periods[-1]


def test_build_quality_report_monthly_counts_gap_in_the_middle():
    periods = _monthly("2021-01-01", 6)
    del periods[3]  # drop April -- a real gap, not a leading/trailing one
    obs = _obs("evds:X", periods, [1.0, 2.0, 3.0, 5.0, 6.0])
    catalog = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "M",
            }
        ]
    )

    report = quality.build_quality_report(obs, catalog)

    assert report.iloc[0]["missing"] == 1


def test_build_quality_report_never_counts_periods_before_series_own_start():
    """TP.HPBITABLO6.3 shape: EVDS bulk fetch returns every week of a
    requested year even before the series existed, as real NaN -- gold's
    observations.parquet carries those pre-start rows. A series starting in
    week 10 of a 20-week bronze fetch must report missing=0, not 9.
    """
    all_weeks = _weekly("2024-01-01", 20)
    values = [np.nan] * 9 + [float(i) for i in range(11)]
    obs = _obs("evds:X", all_weeks, values)
    obs["freq"] = "W"
    catalog = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "W",
            }
        ]
    )

    report = quality.build_quality_report(obs, catalog)

    row = report.iloc[0]
    assert row["n"] == 11
    assert row["missing"] == 0
    assert row["start"] == all_weeks[9]


def test_build_quality_report_weekly_counts_real_gap():
    weeks = _weekly("2024-01-01", 10)
    del weeks[5]
    obs = _obs("evds:X", weeks, list(range(9)))
    obs["freq"] = "W"
    catalog = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "W",
            }
        ]
    )

    report = quality.build_quality_report(obs, catalog)

    assert report.iloc[0]["missing"] == 1


def test_build_quality_report_counts_duplicate_periods():
    periods = _monthly("2021-01-01", 3) + [_monthly("2021-01-01", 3)[0]]  # dupe Jan
    obs = _obs("evds:X", periods, [1.0, 2.0, 3.0, 1.5])
    catalog = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "M",
            }
        ]
    )

    report = quality.build_quality_report(obs, catalog)

    assert report.iloc[0]["duplicates"] == 1


def test_build_quality_report_counts_iqr_outliers():
    periods = _monthly("2021-01-01", 8)
    values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.0, 11.5, 200.0]  # last one way off
    obs = _obs("evds:X", periods, values)
    catalog = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "M",
            }
        ]
    )

    report = quality.build_quality_report(obs, catalog)

    assert report.iloc[0]["outliers"] == 1


def test_build_quality_report_empty_series_does_not_raise():
    obs = pd.DataFrame(columns=["series_id", "period", "value", "fetched_at"])
    catalog = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "M",
            }
        ]
    )

    report = quality.build_quality_report(obs, catalog)

    row = report.iloc[0]
    assert row["n"] == 0
    assert row["missing"] == 0
    assert row["start"] is None


def test_expected_periods_rejects_unsupported_frequency():
    with pytest.raises(ValueError, match="frekans"):
        quality._expected_periods("YILLIK", date(2021, 1, 1), date(2022, 1, 1))


def test_expected_periods_supports_all_contract_frequencies():
    for freq in ("D", "W", "M", "Q", "A"):
        result = quality._expected_periods(freq, date(2021, 1, 1), date(2021, 12, 31))
        assert date(2021, 1, 1) in result


def test_render_quality_markdown_shows_real_dropped_duplicate_count():
    """Regression: quality.py's per-series `duplicates` column is always 0
    (silver already deduped before gold ever saw the data) -- the real
    count now surfaces via the dropped_duplicates parameter."""
    report = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X",
                "source": "evds",
                "freq": "M",
                "start": date(2021, 1, 1),
                "end": date(2021, 6, 1),
                "n": 6,
                "missing": 0,
                "duplicates": 0,
                "outliers": 0,
                "last_updated": "2026-09-18T00:00:00+00:00",
            }
        ]
    )

    markdown = quality.render_quality_markdown(report, dropped_duplicates=3)

    assert "3" in markdown
    assert "mükerrer" in markdown


def test_profile_table_computes_quality_report_from_contracts():
    from lakehouse_agent.contracts import QualityReport

    df = pd.DataFrame(
        {
            "period": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 4, 1)],
            "value": [10.0, 11.0, 12.0],
        }
    )

    report = quality.profile_table(df, table_ref="t01_read_url")

    assert isinstance(report, QualityReport)
    assert report.table_ref == "t01_read_url"
    assert report.rows == 3
    assert report.date_range == ("2024-01", "2024-04")
    assert report.freq == "M"
    assert report.missing_periods == 1


def test_render_quality_markdown_includes_warning_section_when_missing():
    report = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "M",
                "start": date(2021, 1, 1),
                "end": date(2021, 6, 1),
                "n": 5,
                "missing": 1,
                "duplicates": 0,
                "outliers": 0,
                "last_updated": "2026-09-18T00:00:00+00:00",
            }
        ]
    )

    markdown = quality.render_quality_markdown(report)

    assert markdown.index("⚠️") < markdown.index("Kaynak bazında özet")
    assert "evds:X" in markdown
    assert "1 eksik dönem" in markdown


def test_render_quality_markdown_omits_warning_section_when_clean():
    report = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X serisi",
                "source": "evds",
                "freq": "M",
                "start": date(2021, 1, 1),
                "end": date(2021, 6, 1),
                "n": 6,
                "missing": 0,
                "duplicates": 0,
                "outliers": 0,
                "last_updated": "2026-09-18T00:00:00+00:00",
            }
        ]
    )

    markdown = quality.render_quality_markdown(report)

    assert "⚠️" not in markdown
    assert "Kaynak bazında özet" in markdown
    assert "Seri detayları" in markdown


def test_render_quality_markdown_source_summary_aggregates_correctly():
    report = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X",
                "source": "evds",
                "freq": "M",
                "start": date(2021, 1, 1),
                "end": date(2021, 6, 1),
                "n": 6,
                "missing": 0,
                "duplicates": 0,
                "outliers": 1,
                "last_updated": "2026-09-18T00:00:00+00:00",
            },
            {
                "series_id": "bddk:y",
                "name_tr": "Y",
                "source": "BDDK-Aylik",
                "freq": "M",
                "start": date(2021, 1, 1),
                "end": date(2021, 6, 1),
                "n": 6,
                "missing": 2,
                "duplicates": 0,
                "outliers": 0,
                "last_updated": "2026-09-18T00:00:00+00:00",
            },
        ]
    )

    markdown = quality.render_quality_markdown(report)

    assert "| evds | 1 | 6 | 0 | 1 |" in markdown
    assert "| BDDK-Aylik | 1 | 6 | 2 | 0 |" in markdown


def test_write_quality_report_writes_file(tmp_path):
    report = pd.DataFrame(
        [
            {
                "series_id": "evds:X",
                "name_tr": "X",
                "source": "evds",
                "freq": "M",
                "start": date(2021, 1, 1),
                "end": date(2021, 6, 1),
                "n": 6,
                "missing": 0,
                "duplicates": 0,
                "outliers": 0,
                "last_updated": "2026-09-18T00:00:00+00:00",
            }
        ]
    )

    out_path = quality.write_quality_report(report, out_path=tmp_path / "sub" / "dq.md")

    assert out_path.exists()
    assert "evds:X" in out_path.read_text(encoding="utf-8")
