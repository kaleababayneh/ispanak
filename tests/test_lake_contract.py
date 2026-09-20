"""The same contract, asserted against both lake implementations.

Why this file exists. The agent is built against `FakeLake` and is meant to run
against the real `lake/` layer by swapping one object -- which only works if the
two are actually interchangeable. Nothing else checks that. Each side has its
own tests, and each passes; they can still disagree about column values, dtypes
and accepted argument formats, and that disagreement surfaces on the day the
two lanes are merged rather than on the day it is introduced.

So the assertions here are about the *contract*, never about the numbers: which
functions exist, what `get()` accepts, which columns come back in which order,
what dtype `period` has, which values `freq` and `source` are allowed to take.

Reading a failure. The real side is skipped until `lakehouse_agent.lake.gold`
exists and `data/gold/` has been built, so on a branch without the data lane
this file reports skips, not failures. Once the data lane is merged, a failure
here means the two implementations have drifted -- the fix is a change to one
of them plus a line in `docs/contracts.md`, never a relaxed assertion here.
"""

from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest

from lakehouse_agent.contracts import SeriesNotFound
from lakehouse_agent.testing import FakeLake

#: The lake surface `docs/contracts.md` promises the agent lane.
REQUIRED_FUNCTIONS = (
    "get",
    "wide",
    "to_freq",
    "search",
    "catalog_lookup",
    "prepare_table",
    "profile_table",
)

#: Present in both implementations, so it can anchor a shape assertion.
SHARED_SERIES = "evds:TP.KTF12"

OBSERVATION_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
]

#: `freq` is a code, not a label. `contracts.Freq` allows exactly these.
ALLOWED_FREQ = {"D", "W", "M", "Q", "A"}


def _real_lake():
    """The real data lane, wrapped so it is callable like `FakeLake`.

    Skips rather than fails when the lane or its data is absent: this file has
    to be mergeable before the data lane lands.
    """
    try:
        gold = importlib.import_module("lakehouse_agent.lake.gold")
    except ImportError:  # pragma: no cover - depends on which branch is checked out
        pytest.skip("veri katmani (lakehouse_agent.lake.gold) bu dalda yok")

    if not gold.OBSERVATIONS_PATH.exists():
        pytest.skip(f"gold verisi yok: {gold.OBSERVATIONS_PATH}")

    class _RealLake:
        """Only forwards. Any adaptation belongs in the adapter, not here --
        smoothing a difference over in the test is how the test stops seeing it.
        """

        def __getattr__(self, name):
            return getattr(gold, name)

    return _RealLake()


@pytest.fixture(params=["fake", "real"])
def lake(request):
    return FakeLake() if request.param == "fake" else _real_lake()


# --- surface ---------------------------------------------------------------


def test_every_contracted_function_exists(lake):
    """The agent calls all seven. A missing one is not a runtime surprise to
    discover during a demo; it is a contract breach visible without any data."""
    missing = [name for name in REQUIRED_FUNCTIONS if not hasattr(lake, name)]
    assert missing == [], (
        f"docs/contracts.md'deki lake imzalarindan eksik olanlar: {missing}"
    )


def test_get_takes_series_ids_start_and_end(lake):
    signature = inspect.signature(lake.get)
    assert list(signature.parameters)[:3] == ["series_ids", "start", "end"]


# --- what get() accepts ----------------------------------------------------


def test_get_accepts_the_contracted_year_month_format(lake):
    """`docs/contracts.md`: "start ve end bicimi YYYY-MM". The agent builds these
    from a user's question, so a lake that only takes YYYY-MM-DD breaks the
    first call of every session."""
    frame = lake.get([SHARED_SERIES], "2021-01", "2021-06")
    assert isinstance(frame, pd.DataFrame)


# --- what comes back -------------------------------------------------------


def test_get_returns_the_seven_gold_columns_in_order(lake):
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
    frame = lake.get([SHARED_SERIES], "2021-01", "2021-06")
    sources = set(frame["source"].unique())
    assert sources == {"EVDS"}, f"beklenen {{'EVDS'}}, gelen {sources}"


def test_every_row_belongs_to_a_requested_series(lake):
    frame = lake.get([SHARED_SERIES], "2021-01", "2021-06")
    assert set(frame["series_id"].unique()) <= {SHARED_SERIES}


# --- wide() ----------------------------------------------------------------


def test_wide_is_one_column_per_series_keyed_on_period(lake):
    wide = lake.wide([SHARED_SERIES], "2021-01", "2021-06")
    assert "period" in wide.columns
    assert SHARED_SERIES in wide.columns, (
        "wide() sutun adlari series_id'nin birebir kendisi olmali"
    )


def test_wide_period_is_also_a_datetime_column(lake):
    wide = lake.wide([SHARED_SERIES], "2021-01", "2021-06")
    assert pd.api.types.is_datetime64_any_dtype(wide["period"])


# --- how the two report "not there" -----------------------------------------


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
