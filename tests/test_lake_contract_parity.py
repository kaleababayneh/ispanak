"""The same contract, asserted against both lake implementations.

Why this file exists. The agent is built against `FakeLake` and is meant to run
against the real `lake/` layer by swapping one object -- which only works if the
two are actually interchangeable. Nothing else checks that. Each side has its
own tests, and each passes; they can still disagree about which functions exist,
what they are called with, what they hand back, and what values come out of
them. That disagreement then surfaces on the day the two lanes are merged
rather than on the day it is introduced.

Three things this file learned the hard way, all from real divergences:

* The surface is checked **without data**. Gating everything behind
  `data/gold/*.parquet` meant the whole file skipped on CI and on any fresh
  clone -- green and useless. Existence, signatures and return shapes need no
  data, so they always run; only the value checks skip.
* The real lane is **not one module**. `get`/`wide` live in `lake.gold`,
  `to_freq` in `analysis.transforms`, `profile_table` in `lake.quality`, and
  three of the seven exist nowhere. `tools/lake_adapter.LakeAdapter` is what
  joins them into the surface the agent is written against, so that is what
  this file puts opposite `FakeLake` -- testing `lake.gold` directly reported
  functions as missing that simply live elsewhere.
* **Return shapes are part of the signature.** `to_freq` handed back a bare
  frame on one side and `(frame, formula)` on the other; checking parameter
  names alone saw nothing, while `frame, note = lake.to_freq(...)` worked
  against one lane and raised against the other.

A failure here means the two implementations have drifted. The fix is a change
to one of them plus a line in `docs/contracts.md` -- never a relaxed assertion
here.
"""

from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest

from lakehouse_agent.contracts import QualityReport, SeriesNotFound
from lakehouse_agent.testing import FakeLake

#: The lake surface `docs/contracts.md` promises, with the parameter names it
#: promises for each. Names, not just presence: a `to_freq` taking
#: `(df, freq, kind)` satisfies "the function exists" while being uncallable by
#: anything written against `(df, kind, freq)`, and the caller finds out at the
#: call site rather than here.
REQUIRED_SIGNATURES = {
    "get": ["series_ids", "start", "end"],
    "wide": ["series_ids", "start", "end", "freq"],
    "to_freq": ["df", "kind", "freq"],
    "search": ["query", "k"],
    "catalog_lookup": ["series_ids"],
    "prepare_table": ["df", "hints"],
    "profile_table": ["df"],
}

REQUIRED_FUNCTIONS = tuple(REQUIRED_SIGNATURES)

#: Present in both implementations, so it can anchor a value assertion.
SHARED_SERIES = "evds:TP.KTF12"

OBSERVATION_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
    "obs_date",
]

#: `freq` is a code, not a label. `contracts.Freq` allows exactly these.
ALLOWED_FREQ = {"D", "W", "M", "Q", "A"}


def _real_surface():
    """The real lane behind its adapter, without requiring any data. Skips only
    when the data lane is not on this branch at all."""
    try:
        module = importlib.import_module("lakehouse_agent.tools.lake_adapter")
    except ImportError:  # pragma: no cover - depends on branch
        pytest.skip("veri katmani (lakehouse_agent.lake) bu dalda yok")
    return module.LakeAdapter()


@pytest.fixture(params=["fake", "real"])
def surface(request):
    """For checks that need no data: existence, signatures, return shapes."""
    return FakeLake() if request.param == "fake" else _real_surface()


@pytest.fixture(params=["fake", "real"])
def lake(request):
    """For checks that read actual observations."""
    if request.param == "fake":
        return FakeLake()
    real = _real_surface()
    gold = importlib.import_module("lakehouse_agent.lake.gold")
    if not gold.OBSERVATIONS_PATH.exists():
        pytest.skip(f"gold verisi yok: {gold.OBSERVATIONS_PATH}")
    return real


@pytest.fixture
def monthly_frame() -> pd.DataFrame:
    """A small "period"/"value" frame, so the pure transforms can be exercised
    without anyone having built a gold layer first."""
    return pd.DataFrame(
        {
            "period": pd.date_range("2021-01-01", periods=12, freq="MS"),
            "value": [float(i) for i in range(1, 13)],
        }
    )


# --- surface: no data needed -----------------------------------------------


def test_every_contracted_function_exists(surface):
    """The agent calls all seven. A missing one is not a runtime surprise to
    discover during a demo; it is a contract breach visible without any data."""
    missing = [name for name in REQUIRED_FUNCTIONS if not hasattr(surface, name)]
    assert missing == [], (
        f"docs/contracts.md'deki lake imzalarindan eksik olanlar: {missing}"
    )


@pytest.mark.parametrize("name", REQUIRED_FUNCTIONS)
def test_each_function_takes_the_contracted_parameters(surface, name):
    """Checked per function, so a failure names the one that drifted."""
    if not hasattr(surface, name):
        pytest.skip(f"{name} yok; eksigi test_every_contracted_* raporluyor")
    actual = list(inspect.signature(getattr(surface, name)).parameters)
    expected = REQUIRED_SIGNATURES[name]
    assert actual[: len(expected)] == expected, (
        f"{name}: sozlesme {expected} diyor, gelen {actual}"
    )


# --- return shapes: still no data needed -----------------------------------


def test_to_freq_returns_the_frame_and_the_formula(surface, monthly_frame):
    """Plan 4.1: every transform hands back what it did, so the column's
    lineage can say it. This is the divergence a parameter-name check misses --
    `frame, note = lake.to_freq(...)` either works on both sides or on neither.
    """
    if not hasattr(surface, "to_freq"):
        pytest.skip("to_freq yok")
    returned = surface.to_freq(monthly_frame, "rate", "M")

    assert isinstance(returned, tuple) and len(returned) == 2, (
        f"to_freq (DataFrame, formül) döndürmeli, gelen: {type(returned).__name__}"
    )
    frame, formula = returned
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(formula, str) and formula


def test_profile_table_returns_the_contract_type(surface, monthly_frame):
    """A plain DataFrame would carry the same numbers, but the quality gate
    reads named fields off this object and the approval card renders them."""
    if not hasattr(surface, "profile_table"):
        pytest.skip("profile_table yok")
    assert isinstance(surface.profile_table(monthly_frame), QualityReport)


# --- values: gold data required --------------------------------------------


def test_get_accepts_the_contracted_year_month_format(lake):
    """`docs/contracts.md`: "start ve end bicimi YYYY-MM". The agent builds
    these from a user's question, so a lake that only takes YYYY-MM-DD breaks
    the first call of every session."""
    assert isinstance(lake.get([SHARED_SERIES], "2021-01", "2021-06"), pd.DataFrame)


def test_get_returns_the_gold_columns_in_order(lake):
    frame = lake.get([SHARED_SERIES], "2021-01", "2021-06")
    assert list(frame.columns) == OBSERVATION_COLUMNS


def test_period_is_a_datetime_column(lake):
    """Not `datetime.date` in an object column. Under pandas 3 the two do not
    merge: joining them either raises or silently matches nothing, and both
    `get_series(into="active")` and `deflate` do exactly that join."""
    frame = lake.get([SHARED_SERIES], "2021-01", "2021-06")
    assert pd.api.types.is_datetime64_any_dtype(frame["period"])


def test_freq_uses_contract_codes_not_source_labels(lake):
    """`M`, not `AYLIK`. `contracts.Freq` is a Literal, so a raw source label
    fails validation the moment the frame reaches the tool layer."""
    frame = lake.get([SHARED_SERIES], "2021-01", "2021-06")
    unexpected = set(frame["freq"].unique()) - ALLOWED_FREQ
    assert unexpected == set(), f"sozlesme disi freq degerleri: {unexpected}"


def test_source_is_the_contract_spelling(lake):
    """`EVDS`, not `evds`. The trust panel groups by this string and the answer
    prompt prints it, so two spellings become two sources to a reader."""
    sources = set(lake.get([SHARED_SERIES], "2021-01", "2021-06")["source"].unique())
    assert sources == {"EVDS"}, f"beklenen {{'EVDS'}}, gelen {sources}"


def test_every_row_belongs_to_a_requested_series(lake):
    frame = lake.get([SHARED_SERIES], "2021-01", "2021-06")
    assert set(frame["series_id"].unique()) <= {SHARED_SERIES}


def test_wide_is_one_column_per_series_keyed_on_period(lake):
    wide = lake.wide([SHARED_SERIES], "2021-01", "2021-06")
    assert "period" in wide.columns
    assert SHARED_SERIES in wide.columns, (
        "wide() sutun adlari series_id'nin birebir kendisi olmali"
    )


def test_wide_period_is_also_a_datetime_column(lake):
    wide = lake.wide([SHARED_SERIES], "2021-01", "2021-06")
    assert pd.api.types.is_datetime64_any_dtype(wide["period"])


def test_an_unknown_series_raises_series_not_found(lake):
    """`contracts.SeriesNotFound` exists for exactly this, and the loop turns it
    into a message the model can act on ("search the catalog first"). A lake
    that answers with an empty frame instead gives the model nothing to react
    to, so it reports "no data" for what is really a typo in a series id."""
    with pytest.raises(SeriesNotFound):
        lake.get(["evds:BOYLE.BIR.SERI.YOK"], "2021-01", "2021-06")


def test_an_empty_window_is_an_empty_frame_with_the_full_schema(lake):
    """A known series with no rows in the window is a normal answer, not an
    error -- but the shape must not change. Falling back to a different set of
    columns makes the caller's next line a KeyError precisely when there is
    nothing to see."""
    frame = lake.get([SHARED_SERIES], "2019-01", "2019-06")
    assert list(frame.columns) == OBSERVATION_COLUMNS
    assert frame.empty
