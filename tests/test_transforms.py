"""Tests for lakehouse_agent.analysis.transforms. Pure functions, no
network/model/session -- small hand-computed examples only."""

import logging
import math
from datetime import date

import pandas as pd
import pytest

from lakehouse_agent.analysis import transforms


def _series(periods, values):
    return pd.DataFrame({"period": periods, "value": values})


# --- to_freq ------------------------------------------------------------


def test_to_freq_stock_takes_last_value_in_target_period():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, formula = transforms.to_freq(df, "stock", "QS")

    assert list(result["value"]) == [30.0]
    assert "last" in formula


def test_to_freq_flow_sums_values_in_target_period():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, _ = transforms.to_freq(df, "flow", "QS")

    assert list(result["value"]) == [60.0]


def test_to_freq_rate_averages_values_in_target_period():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, _ = transforms.to_freq(df, "rate", "QS")

    assert result["value"].iloc[0] == 20.0


def test_to_freq_flow_all_nan_window_stays_nan_not_zero():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [float("nan")] * 3
    )

    result, _ = transforms.to_freq(df, "flow", "QS")

    assert math.isnan(result["value"].iloc[0])


def test_to_freq_period_stays_datetime64_not_object():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, _ = transforms.to_freq(df, "stock", "QS")

    assert pd.api.types.is_datetime64_any_dtype(result["period"])


def test_to_freq_output_merges_with_datetime64_period_via_deflate():
    """Regression: to_freq() used to downcast `period` to python `date`
    objects (object dtype) via `.dt.date`; deflate/align_join's
    pd.merge(..., on="period") then raised ValueError when the other side
    was datetime64 -- B'nin ölçtüğü çökme."""
    monthly = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )
    quarterly, _ = transforms.to_freq(monthly, "flow", "QS")
    price_index = pd.DataFrame(
        {"period": pd.to_datetime(["2024-01-01"]), "value": [100.0]}
    )

    result, _ = transforms.deflate(quarterly, price_index)

    assert len(result) == 1
    assert result["value"].iloc[0] == 60.0


def test_to_freq_accepts_contract_freq_values_and_returns_period_first_day():
    """B'nin ölçtüğü çökme: to_freq(df, "rate", "M") -> ValueError ("'M' is
    no longer supported for offsets. Please use 'ME' instead."). Sözleşme
    değerleri (D|W|M|Q|A, contracts.py Freq Literal) birinci sınıf kabul
    edilmeli -- hiçbiri hata vermemeli, hepsinde period dönemin İLK günü
    olmalı (ME/QE/YE değil)."""
    base = list(pd.date_range("2024-01-01", periods=10, freq="D").date)
    df = _series(base, [float(i) for i in range(10)])

    for freq in ("D", "W", "M", "Q", "A"):
        result, _ = transforms.to_freq(df, "rate", freq)
        assert not result.empty, freq
        first_period = result["period"].iloc[0]
        # 2024-01-01 is itself a Monday -- every one of these frequencies'
        # first bucket for this 10-day input starts on that exact date.
        assert first_period.date() == date(2024, 1, 1), freq


def test_to_freq_agg_method_decides_over_kind():
    """A rate EVDS publishes as an end-of-period value is still a rate; its
    month is its last value, not the average (the source's own agg_method)."""
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, formula = transforms.to_freq(df, "rate", "Q", agg_method="last")

    assert list(result["value"]) == [30.0]
    assert "last" in formula


def test_to_freq_puts_a_week_in_the_month_it_was_observed():
    """Silver names a week by its Monday; the value is the Friday's. The week
    of Monday 31 March 2025 was observed on Friday 4 April: April's."""
    df = pd.DataFrame(
        {
            "period": [date(2025, 3, 24), date(2025, 3, 31), date(2025, 4, 7)],
            "obs_date": [date(2025, 3, 28), date(2025, 4, 4), date(2025, 4, 11)],
            "value": [1.0, 2.0, 3.0],
        }
    )

    result, formula = transforms.to_freq(df, "stock", "M")

    by_month = dict(
        zip(result["period"].dt.strftime("%Y-%m"), result["value"], strict=True)
    )
    assert by_month == {"2025-03": 1.0, "2025-04": 3.0}
    assert "gözlem tarihine göre" in formula


def test_to_freq_names_a_week_by_its_monday():
    """pandas' W-MON resample labelled Tuesday..Monday by the closing Monday;
    silver's weeks are Monday..Sunday named by the opening one."""
    days = list(pd.date_range("2024-06-03", periods=14, freq="D").date)  # Monday
    df = _series(days, [1.0] * 14)

    result, _ = transforms.to_freq(df, "flow", "W")

    assert list(result["period"].dt.date) == [date(2024, 6, 3), date(2024, 6, 10)]
    assert list(result["value"]) == [7.0, 7.0]


def test_to_freq_a_gap_is_a_nan_row_not_a_missing_one():
    df = _series([date(2024, 1, 1), date(2024, 3, 1)], [1.0, 3.0])

    result, _ = transforms.to_freq(df, "stock", "M")

    assert len(result) == 3
    assert math.isnan(result["value"].iloc[1])


def test_to_freq_rejects_unknown_kind():
    df = _series([date(2024, 1, 1)], [1.0])
    try:
        transforms.to_freq(df, "bogus", "QS")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- deflate --------------------------------------------------------------


def test_deflate_divides_nominal_by_index_over_base():
    nominal = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 110.0])
    price_index = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 110.0])

    result, _ = transforms.deflate(nominal, price_index, base=100.0)

    assert [round(v, 6) for v in result["value"]] == [100.0, 100.0]


def test_deflate_missing_index_period_yields_nan_not_dropped_row():
    nominal = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 110.0])
    price_index = _series([date(2024, 1, 1)], [100.0])

    result, _ = transforms.deflate(nominal, price_index, base=100.0)

    assert len(result) == 2
    assert math.isnan(result["value"].iloc[1])


def test_deflate_raises_on_frequency_mismatch():
    """B'nin ölçtüğü: aylık nominal + haftalık endeks -> merge sessizce
    9/12 NaN üretiyordu, hata yok. Artık gerçek frekans uyuşmazlığında
    ValueError."""
    nominal = _series([date(2024, m, 1) for m in range(1, 13)], [100.0] * 12)
    price_index = _series(
        list(pd.date_range("2024-01-01", periods=12, freq="W-MON").date),
        [100.0] * 12,
    )

    with pytest.raises(ValueError):
        transforms.deflate(nominal, price_index)


def test_deflate_warns_on_unmatched_periods_same_frequency(caplog):
    nominal = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 110.0])
    price_index = _series([date(2024, 1, 1)], [100.0])

    with caplog.at_level(logging.WARNING, logger="lakehouse_agent.analysis.transforms"):
        result, _ = transforms.deflate(nominal, price_index)

    assert len(result) == 2
    assert math.isnan(result["value"].iloc[1])
    assert any("eşleşmedi" in r.message for r in caplog.records)


# --- rebase ---------------------------------------------------------------


def test_rebase_rescales_so_base_period_equals_base_value():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [50.0, 100.0, 150.0]
    )

    result, _ = transforms.rebase(df, date(2024, 2, 1), base_value=200.0)

    assert list(result["value"]) == [100.0, 200.0, 300.0]


def test_rebase_raises_when_base_period_not_in_series():
    df = _series([date(2024, 1, 1)], [50.0])
    try:
        transforms.rebase(df, date(2024, 6, 1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_rebase_raises_when_base_value_is_zero():
    df = _series([date(2024, 1, 1), date(2024, 2, 1)], [0.0, 50.0])
    with pytest.raises(ValueError):
        transforms.rebase(df, date(2024, 1, 1))


def test_rebase_raises_when_base_value_is_nan():
    df = _series([date(2024, 1, 1), date(2024, 2, 1)], [float("nan"), 50.0])
    with pytest.raises(ValueError):
        transforms.rebase(df, date(2024, 1, 1))


# --- pct_change / yoy / diff ----------------------------------------------


def test_pct_change_matches_hand_computed_values():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [100.0, 110.0, 121.0]
    )

    result, _ = transforms.pct_change(df)

    assert math.isnan(result["value"].iloc[0])
    assert round(result["value"].iloc[1], 6) == 10.0
    assert round(result["value"].iloc[2], 6) == 10.0


def test_yoy_uses_periods_per_year_as_the_lag():
    # periods_per_year=12 (real monthly cadence): Jan2025/Feb2025's true
    # 12-months-back counterpart is Jan2024/Feb2024 -- the calendar-based
    # shift (not a row-position shift) finds them correctly even though
    # Mar-Dec 2024 are entirely absent from this toy series.
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2025, 1, 1), date(2025, 2, 1)],
        [100.0, 110.0, 120.0, 132.0],
    )

    result, _ = transforms.yoy(df, periods_per_year=12)

    assert math.isnan(result["value"].iloc[0])
    assert math.isnan(result["value"].iloc[1])
    assert round(result["value"].iloc[2], 6) == 20.0
    assert round(result["value"].iloc[3], 6) == 20.0


def test_diff_matches_hand_computed_values():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [100.0, 110.0, 90.0]
    )

    result, _ = transforms.diff(df)

    assert math.isnan(result["value"].iloc[0])
    assert result["value"].iloc[1] == 10.0
    assert result["value"].iloc[2] == -20.0


def test_yoy_uses_calendar_shift_not_row_shift_missing_middle_period():
    """B'nin ölçtüğü: eksiksiz seride yoy doğru; 1 ay eksikse pandas'ın
    SATIR kaydırması bir sonraki dönemi yanlış aya eşliyordu. 2021-02 kasıtlı
    eksik: 2022-01'in gerçek eşi (2021-01) var -> doğru; 2022-02'nin gerçek
    eşi (2021-02) eksik -> NaN, eski koddaki gibi 2021-01'e YANLIŞ eşleşip
    13.0 dönmüyor."""
    periods = [
        date(2021, 1, 1),
        date(2021, 3, 1),
        date(2021, 4, 1),
        date(2021, 5, 1),
        date(2021, 6, 1),
        date(2021, 7, 1),
        date(2021, 8, 1),
        date(2021, 9, 1),
        date(2021, 10, 1),
        date(2021, 11, 1),
        date(2021, 12, 1),
        date(2022, 1, 1),
        date(2022, 2, 1),
    ]  # 2021-02 eksik
    values = [
        100.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
        113.0,
    ]
    df = _series(periods, values)

    result, _ = transforms.yoy(df, periods_per_year=12)
    by_period = dict(zip(result["period"], result["value"], strict=True))

    assert round(by_period[date(2022, 1, 1)], 6) == 12.0
    assert math.isnan(by_period[date(2022, 2, 1)])


def test_pct_change_sorts_input_reversed_rows_do_not_flip_the_sign():
    df = _series(
        [date(2024, 3, 1), date(2024, 2, 1), date(2024, 1, 1)],
        [90.0, 100.0, 80.0],
    )

    result, _ = transforms.pct_change(df)
    by_period = dict(zip(result["period"], result["value"], strict=True))

    assert math.isnan(by_period[date(2024, 1, 1)])
    assert round(by_period[date(2024, 2, 1)], 6) == 25.0
    assert round(by_period[date(2024, 3, 1)], 6) == -10.0


# --- align_join ------------------------------------------------------------


def test_align_join_outer_keeps_all_periods_missing_stays_nan():
    a = _series([date(2024, 1, 1), date(2024, 2, 1)], [1.0, 2.0])
    b = _series([date(2024, 2, 1), date(2024, 3, 1)], [20.0, 30.0])

    result, _ = transforms.align_join({"a": a, "b": b}, how="outer")

    assert list(result["period"]) == [
        date(2024, 1, 1),
        date(2024, 2, 1),
        date(2024, 3, 1),
    ]
    assert result["a"].iloc[0] == 1.0
    assert math.isnan(result["a"].iloc[2])
    assert math.isnan(result["b"].iloc[0])
    assert result["b"].iloc[2] == 30.0


def test_align_join_inner_keeps_only_common_periods():
    a = _series([date(2024, 1, 1), date(2024, 2, 1)], [1.0, 2.0])
    b = _series([date(2024, 2, 1), date(2024, 3, 1)], [20.0, 30.0])

    result, _ = transforms.align_join({"a": a, "b": b}, how="inner")

    assert list(result["period"]) == [date(2024, 2, 1)]


# --- change_since -------------------------------------------------------


def test_change_since_measures_every_period_against_one_base():
    df = _series(
        [date(2021, 1, 1), date(2021, 2, 1), date(2021, 3, 1)], [80.0, 100.0, 120.0]
    )

    result, formula = transforms.change_since(df, date(2021, 1, 1))

    # (100/80 - 1) * 100 = 25, (120/80 - 1) * 100 = 50
    assert list(result["value"]) == pytest.approx([0.0, 25.0, 50.0])
    assert "value[2021-01-01]" in formula


def test_change_since_refuses_a_base_period_that_is_not_there():
    df = _series([date(2021, 2, 1)], [100.0])
    with pytest.raises(ValueError, match="seride yok"):
        transforms.change_since(df, date(2021, 1, 1))


# --- ratio --------------------------------------------------------------


def test_ratio_is_a_share_in_percent():
    part = _series([date(2024, 1, 1), date(2024, 2, 1)], [25.0, 30.0])
    whole = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 120.0])

    result, formula = transforms.ratio(part, whole)

    assert list(result["value"]) == pytest.approx([25.0, 25.0])
    assert "* 100" in formula


def test_ratio_leaves_a_missing_or_zero_denominator_missing():
    """NaN, not 0 and not inf: a share that cannot be computed is unknown."""
    part = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [1.0, 2.0, 3.0]
    )
    whole = _series([date(2024, 1, 1), date(2024, 2, 1)], [10.0, 0.0])

    result, _ = transforms.ratio(part, whole)

    assert result["value"].iloc[0] == pytest.approx(10.0)
    assert math.isnan(result["value"].iloc[1])
    assert math.isnan(result["value"].iloc[2])


def test_ratio_refuses_two_different_frequencies():
    monthly = _series([date(2024, 1, 1), date(2024, 2, 1)], [1.0, 2.0])
    weekly = _series([date(2024, 1, 1), date(2024, 1, 8)], [1.0, 2.0])
    with pytest.raises(ValueError, match="farklı frekanslarda"):
        transforms.ratio(monthly, weekly)


# --- mark_periods -------------------------------------------------------


def _scenario_diffs():
    periods = [date(2024, m, 1) for m in (1, 2, 3, 4)]
    rate = _series(periods, [float("nan"), -0.5, -0.2, 0.3])
    credit = _series(periods, [float("nan"), 0.0, 5.0, -1.0])
    return {"faiz_diff": rate, "kredi_diff": credit}


def test_mark_periods_flags_where_every_condition_holds():
    """The scenario's question: the rate fell and credit did not grow."""
    result, formula = transforms.mark_periods(
        _scenario_diffs(), [("faiz_diff", "<", 0), ("kredi_diff", "<=", 0)]
    )

    # Feb: -0.5 < 0 and 0.0 <= 0 -> True. Mar: credit grew -> False.
    # Apr: rate rose -> False.
    assert list(result["value"])[1:] == [True, False, False]
    assert formula == "mark = faiz_diff < 0 AND kredi_diff <= 0"


def test_mark_periods_leaves_an_unevaluable_period_unknown_not_false():
    """The first month of a difference has no previous month. That is not
    "the rate did not fall", and reporting it as False would be a claim."""
    result, _ = transforms.mark_periods(
        _scenario_diffs(), [("faiz_diff", "<", 0), ("kredi_diff", "<=", 0)]
    )

    assert result["value"].iloc[0] is pd.NA


def test_mark_periods_is_false_when_one_condition_is_definitely_false():
    """Kleene AND: unknown and False is False."""
    periods = [date(2024, 1, 1), date(2024, 2, 1)]
    series = {
        "a": _series(periods, [float("nan"), 1.0]),
        "b": _series(periods, [5.0, 5.0]),
    }

    result, _ = transforms.mark_periods(series, [("a", "<", 0), ("b", "<", 0)])

    assert list(result["value"]) == [False, False]


@pytest.mark.parametrize(
    ("conditions", "match"),
    [
        ([], "en az bir"),
        ([("yok", "<", 0)], "bilinmeyen seri"),
        ([("faiz_diff", "=<", 0)], "bilinmeyen karşılaştırma"),
    ],
)
def test_mark_periods_refuses_what_it_cannot_evaluate(conditions, match):
    with pytest.raises(ValueError, match=match):
        transforms.mark_periods(_scenario_diffs(), conditions)


# --- year-to-date cumulative items (BDDK profit and loss) -----------------


def test_decumulate_ytd_turns_running_totals_into_monthly_flows():
    """Hand-computed: 10, 25, 45 year-to-date are flows of 10, 15, 20; the
    next January starts again at its own value."""
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1), date(2025, 1, 1)],
        [10.0, 25.0, 45.0, 7.0],
    )

    result, formula = transforms.decumulate_ytd(df)

    assert list(result["value"]) == [10.0, 15.0, 20.0, 7.0]
    assert "Ocak" in formula


def test_decumulate_ytd_leaves_a_month_after_a_gap_unknown():
    """March minus January is two months' flow, not March's: not guessed."""
    df = _series([date(2024, 1, 1), date(2024, 3, 1)], [10.0, 45.0])

    result, _ = transforms.decumulate_ytd(df)

    assert result["value"].iloc[0] == 10.0
    assert math.isnan(result["value"].iloc[1])


def test_year_to_date_is_the_inverse_and_stops_at_a_missing_month():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1), date(2025, 1, 1)],
        [10.0, 15.0, 20.0, 7.0],
    )

    result, _ = transforms.year_to_date(df)
    assert list(result["value"]) == [10.0, 25.0, 45.0, 7.0]

    gappy = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, None, 20.0]
    )
    result, _ = transforms.year_to_date(gappy)
    assert result["value"].iloc[0] == 10.0
    assert result["value"].iloc[1:].isna().all()


def test_a_total_row_is_recognised_by_its_label_in_either_language():
    from lakehouse_agent.analysis.transforms import is_total_label

    assert is_total_label("TOPLAM / TOTAL")
    assert is_total_label("Genel Toplam")
    assert is_total_label("Grand total")
    assert not is_total_label("Ocak / January")
    assert not is_total_label("Toplamlar Hariç Konut")  # no whole word
    assert not is_total_label(2024)
