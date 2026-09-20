"""Tests for tools/analysis.py.

track B's contracts.py / agent/context.py / tools/registry.py are not on
this branch (merge deliberately deferred -- config.py/pyproject.toml would
conflict). So instead of importing the real ToolResult/ToolContext/ToolSpec/
ColumnInfo, this file installs minimal duck-typed stubs into `sys.modules`
before importing `lakehouse_agent.tools.analysis`, then exercises the real
wrapper logic (and the real analysis/anomaly.py + analysis/change.py) against
those stubs. `lakehouse_agent.analysis.anomaly`/`.change` are real -- only the
B-owned plumbing is faked.
"""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pydantic
import pytest

_STUBBED_MODULES = [
    "lakehouse_agent.contracts",
    "lakehouse_agent.agent",
    "lakehouse_agent.agent.context",
    "lakehouse_agent.tools.registry",
    "lakehouse_agent.tools.analysis",
]


class StubColumnInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class StubToolResult:
    def __init__(
        self,
        *,
        ok,
        summary,
        table_ref=None,
        preview=None,
        columns=None,
        sources=None,
        warnings=None,
        artifacts=None,
        quality=None,
    ):
        self.ok = ok
        self.summary = summary
        self.table_ref = table_ref
        self.preview = preview or []
        self.columns = columns or []
        self.sources = sources or []
        self.warnings = warnings or []
        self.artifacts = artifacts or []
        self.quality = quality


class StubToolContext:
    """Duck-typed stand-in for agent/context.py's real ToolContext -- just
    enough of `table`/`put_table`/`new_table_ref`/`lineage`/`step_no` for
    tools/analysis.py's wrappers to run against."""

    def __init__(self, tables, *, active_table=None, lineage=None, step_no=1):
        self._tables = dict(tables)
        self.active_table = active_table
        self.lineage = lineage or {}
        self.step_no = step_no
        self.written = {}

    def table(self, ref=None):
        key = ref or self.active_table
        source = self.written.get(key, self._tables.get(key))
        if source is None:
            raise KeyError(key)
        return source.copy()

    def put_table(self, ref, frame):
        self.written[ref] = frame

    def new_table_ref(self, suffix):
        known = {**self._tables, **self.written}
        return f"t{len(known) + 1:02d}_{suffix}"


class StubToolSpec:
    def __init__(self, *, name, description_tr, args_model, fn, brief_name=""):
        self.name = name
        self.description_tr = description_tr
        self.args_model = args_model
        self.fn = fn
        self.brief_name = brief_name


def _install_stub_modules():
    # contracts.py itself has no heavy deps (only pydantic) -- reuse its real
    # TableRef so the hallucination-shield pattern is actually exercised,
    # not re-approximated here (SINGLE DEFINITION RULE, contracts.py).
    from lakehouse_agent.contracts import TableRef as _real_table_ref

    contracts_mod = types.ModuleType("lakehouse_agent.contracts")
    contracts_mod.ColumnInfo = StubColumnInfo
    contracts_mod.ToolResult = StubToolResult
    contracts_mod.TableRef = _real_table_ref

    agent_pkg = types.ModuleType("lakehouse_agent.agent")
    agent_context_mod = types.ModuleType("lakehouse_agent.agent.context")
    agent_context_mod.ToolContext = StubToolContext

    tools_registry_mod = types.ModuleType("lakehouse_agent.tools.registry")
    tools_registry_mod.ToolSpec = StubToolSpec

    sys.modules["lakehouse_agent.contracts"] = contracts_mod
    sys.modules["lakehouse_agent.agent"] = agent_pkg
    sys.modules["lakehouse_agent.agent.context"] = agent_context_mod
    sys.modules["lakehouse_agent.tools.registry"] = tools_registry_mod


@pytest.fixture
def analysis_tools():
    """Import tools/analysis.py fresh, against stubbed B dependencies."""
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    sys.modules.pop("lakehouse_agent.tools.analysis", None)
    _install_stub_modules()
    try:
        yield importlib.import_module("lakehouse_agent.tools.analysis")
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_import_without_b_dependencies_raises_clear_error():
    """Without B's branch merged (the real situation on this branch), import
    fails loudly with an actionable message -- not a bare ModuleNotFoundError.
    """
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    for name in _STUBBED_MODULES:
        sys.modules.pop(name, None)
    try:
        with pytest.raises(ImportError, match="contracts.py"):
            importlib.import_module("lakehouse_agent.tools.analysis")
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _monthly_periods(n: int):
    return pd.date_range("2020-01-01", periods=n, freq="MS")


def test_detect_anomalies_args_rejects_hallucinated_table_ref(analysis_tools):
    """Halüsinasyon kalkanı: table_ref contracts.py'nin TableRef'i olmalı
    (^t\\d{2}_[a-z][a-z0-9_]{0,24}$) -- model uydurma bir ad verirse Pydantic
    aracı hiç çalıştırmadan reddetmeli, düz str'de olduğu gibi sessizce
    geçmemeli."""
    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    with pytest.raises(pydantic.ValidationError):
        DetectAnomaliesArgs(table_ref="konut_tablosu", column="TP_KTF12")


def test_detect_changes_args_rejects_hallucinated_table_ref(analysis_tools):
    from lakehouse_agent.tools.analysis import DetectChangesArgs

    with pytest.raises(pydantic.ValidationError):
        DetectChangesArgs(table_ref="konut_tablosu", column="TP_KTF12")


def test_detect_anomalies_flags_point_anomaly(analysis_tools):
    n = 36
    values = 10.0 + np.sin(np.arange(n) * 2 * np.pi / 12)
    values[20] += 8.0  # obvious point anomaly
    frame = pd.DataFrame({"period": _monthly_periods(n), "TP_KTF12": values})

    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectAnomaliesArgs(column="TP_KTF12")

    spec = analysis_tools.make_detect_anomalies()
    result = spec.fn(args, ctx)

    assert result.ok is True
    assert "anomali" in result.summary
    assert result.table_ref in ctx.written
    assert not ctx.written[result.table_ref].empty
    assert {c.name for c in result.columns} == {
        "dönem",
        "gözlenen",
        "beklenen",
        "z",
        "yön",
    }
    assert all(c.tool == "detect_anomalies" for c in result.columns)


def test_detect_anomalies_short_series_warns_about_iqr_fallback(analysis_tools):
    n = 10
    frame = pd.DataFrame(
        {"period": _monthly_periods(n), "value_col": np.arange(n, dtype=float)}
    )

    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectAnomaliesArgs(column="value_col")

    result = analysis_tools.make_detect_anomalies().fn(args, ctx)

    assert result.ok is True
    assert any("IQR" in w for w in result.warnings)


def test_detect_anomalies_iqr_path_uses_tukey_formula_text(analysis_tools):
    """B'nin ölçtüğü: IQR yolunda formül metni hâlâ STL/MAD diyordu, z de
    (MAD sıfıra dejenere olduğu için) 0.00 gösteriyordu. Artık IQR yolunda
    formül metni Tukey çitini adlandırıyor."""
    n = 12
    values = [10.0] * 11 + [100.0]
    frame = pd.DataFrame({"period": _monthly_periods(n), "TP_KTF12": values})

    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectAnomaliesArgs(column="TP_KTF12")

    result = analysis_tools.make_detect_anomalies().fn(args, ctx)

    z_col = next(c for c in result.columns if c.name == "z")
    assert "çit" in z_col.formula.lower() or "tukey" in z_col.formula.lower()


def test_detect_anomalies_missing_column_returns_ok_false(analysis_tools):
    frame = pd.DataFrame({"period": _monthly_periods(5), "value_col": range(5)})

    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectAnomaliesArgs(column="does_not_exist")

    result = analysis_tools.make_detect_anomalies().fn(args, ctx)

    assert result.ok is False
    assert "does_not_exist" in result.summary
    assert "value_col" in result.summary


def test_detect_anomalies_reports_dropped_missing_values(analysis_tools):
    n = 30
    values = 10.0 + np.sin(np.arange(n) * 2 * np.pi / 12)
    values[5] = np.nan
    frame = pd.DataFrame({"period": _monthly_periods(n), "value_col": values})

    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectAnomaliesArgs(column="value_col")

    result = analysis_tools.make_detect_anomalies().fn(args, ctx)

    assert any("eksik" in w for w in result.warnings)


def test_detect_changes_finds_level_break(analysis_tools):
    n = 60
    break_idx = 30
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [
            10.0 + rng.normal(scale=0.3, size=break_idx),
            20.0 + rng.normal(scale=0.3, size=n - break_idx),
        ]
    )
    frame = pd.DataFrame({"period": _monthly_periods(n), "TP_KTF12": values})

    from lakehouse_agent.tools.analysis import DetectChangesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectChangesArgs(column="TP_KTF12")

    result = analysis_tools.make_detect_changes().fn(args, ctx)

    assert result.ok is True
    assert "kırılma" in result.summary
    assert not ctx.written[result.table_ref].empty
    assert {c.name for c in result.columns} == {
        "tarih",
        "ortalama_öncesi",
        "ortalama_sonrası",
        "eğim_öncesi",
        "eğim_sonrası",
        "seviye_degisim_yuzde",
        "sicrama_yuzde",
    }
    assert all(c.tool == "detect_changes" for c in result.columns)


def test_detect_anomalies_wraps_unexpected_exception_as_ok_false(
    analysis_tools, monkeypatch
):
    def _boom(series, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(analysis_tools, "detect_anomalies", _boom)

    frame = pd.DataFrame({"period": _monthly_periods(10), "value_col": range(10)})
    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectAnomaliesArgs(column="value_col")

    result = analysis_tools.make_detect_anomalies().fn(args, ctx)

    assert result.ok is False
    assert "boom" in result.summary


def test_detect_changes_wraps_unexpected_exception_as_ok_false(
    analysis_tools, monkeypatch
):
    def _boom(series):
        raise RuntimeError("boom")

    monkeypatch.setattr(analysis_tools, "detect_changes", _boom)

    frame = pd.DataFrame({"period": _monthly_periods(10), "value_col": range(10)})
    from lakehouse_agent.tools.analysis import DetectChangesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectChangesArgs(column="value_col")

    result = analysis_tools.make_detect_changes().fn(args, ctx)

    assert result.ok is False
    assert "boom" in result.summary


def test_detect_anomalies_derives_stl_period_from_column_freq(
    analysis_tools, monkeypatch
):
    captured = {}

    def spy(series, **kwargs):
        captured.update(kwargs)
        return (
            pd.DataFrame(columns=["dönem", "gözlenen", "beklenen", "z", "yön"]),
            "ok",
        )

    monkeypatch.setattr(analysis_tools, "detect_anomalies", spy)

    n = 30
    frame = pd.DataFrame(
        {"period": _monthly_periods(n), "TP_KTF12": np.arange(n, dtype=float)}
    )
    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    lineage = {
        "t01_get_series.TP_KTF12": StubColumnInfo(
            source_series=["evds:TP.KTF12"], freq="W"
        )
    }
    ctx = _new_ctx(
        analysis_tools,
        {"t01_get_series": frame},
        active_table="t01_get_series",
        lineage=lineage,
    )
    args = DetectAnomaliesArgs(column="TP_KTF12")

    analysis_tools.make_detect_anomalies().fn(args, ctx)

    assert captured.get("period") == 52


def test_detect_changes_missing_column_returns_ok_false(analysis_tools):
    frame = pd.DataFrame({"period": _monthly_periods(10), "value_col": range(10)})

    from lakehouse_agent.tools.analysis import DetectChangesArgs

    ctx = _new_ctx(
        analysis_tools, {"t01_get_series": frame}, active_table="t01_get_series"
    )
    args = DetectChangesArgs(column="does_not_exist")

    result = analysis_tools.make_detect_changes().fn(args, ctx)

    assert result.ok is False


def test_column_lineage_traces_upstream_source_series(analysis_tools):
    n = 30
    frame = pd.DataFrame(
        {"period": _monthly_periods(n), "TP_KTF12": 10.0 + np.arange(n) * 0.1}
    )
    from lakehouse_agent.tools.analysis import DetectAnomaliesArgs

    lineage = {
        "t01_get_series.TP_KTF12": StubColumnInfo(source_series=["evds:TP.KTF12"])
    }
    ctx = _new_ctx(
        analysis_tools,
        {"t01_get_series": frame},
        active_table="t01_get_series",
        lineage=lineage,
    )
    args = DetectAnomaliesArgs(column="TP_KTF12")

    result = analysis_tools.make_detect_anomalies().fn(args, ctx)

    assert all(c.source_series == ["evds:TP.KTF12"] for c in result.columns)


def _new_ctx(analysis_tools, tables, **kwargs):
    ToolContext = sys.modules["lakehouse_agent.agent.context"].ToolContext
    return ToolContext(tables, **kwargs)
