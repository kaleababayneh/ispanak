"""Agent-facing wrappers around analysis/anomaly.py and analysis/change.py
(plan 4.5).

A tool never sees a DataFrame in its arguments and never returns one: the
model passes a `table_ref` + a column name, this module pulls the frame out
of the session (`ToolContext.table`), calls the pure analysis function, and
puts the result table back into the session (`ToolContext.put_table`). Only
`table_ref`, `preview` (first/last rows) and `summary` travel back to the
model -- never the full table (CLAUDE.md #7, contracts.py's ToolResult
docstring).

Per docs/contracts.md section 5 ("Aktif tablo kuralı"): analysis-result
tables never change the session's active table (only get_series(into="new"),
run_sql, ingest, read_url do) -- so these tools deliberately do not call
`ctx.set_active()`.

ToolResult / ToolContext / ToolSpec are defined once, in track B's
contracts.py / agent/context.py / tools/registry.py -- imported here, never
redefined (CLAUDE.md, contracts.py's own "SINGLE DEFINITION RULE").
tools/registry.py itself is track B's file: not edited here, and the actual
tool registration (wiring `TOOLS` below into the live registry) is theirs.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from lakehouse_agent.analysis.anomaly import (
    _MIN_PERIODS_FOR_STL,
    IQR_FALLBACK_NOTE,
    detect_anomalies,
)
from lakehouse_agent.analysis.change import detect_changes

try:
    from lakehouse_agent.agent.context import ToolContext
    from lakehouse_agent.contracts import ColumnInfo, TableRef, ToolResult
    from lakehouse_agent.tools.registry import ToolSpec
except ImportError as exc:  # pragma: no cover - real absence, not exercised by tests
    raise ImportError(
        "tools/analysis.py, track B'nin contracts.py / agent/context.py / "
        "tools/registry.py dosyalarina ihtiyac duyar. Bu dalda henuz yoklar "
        "(B'nin dali kasitli olarak merge edilmedi -- config.py ve "
        "pyproject.toml degisiklikleri cakisiyor). Gercek registry icinde "
        "kullanmadan once B'nin dalini merge edin. Testlerde bu modulu import "
        "etmeden once ToolResult/ToolContext/ToolSpec/ColumnInfo icin "
        "sys.modules stub'lari kurun (bkz. tests/test_tools_analysis.py)."
    ) from exc

PREVIEW_ROWS = 5

_STL_FORMULAS = {
    "beklenen": "STL trend + mevsimsellik",
    "z": "z = (kalıntı - medyan) / (1.4826 × MAD)",
}
_IQR_FORMULAS = {
    "beklenen": "Medyan (Tukey 1.5×IQR yolunda)",
    "z": "Çit mesafesi: max(0, deger-üst_çit, alt_çit-deger), 1.5×IQR çiti",
}


def _preview_rows(frame: pd.DataFrame, n: int = PREVIEW_ROWS) -> list[dict]:
    """First and last `n` rows -- enough for the model to see the shape of
    the result without ever holding the whole table (contracts.py)."""
    if len(frame) <= 2 * n:
        return frame.to_dict("records")
    return pd.concat([frame.head(n), frame.tail(n)]).to_dict("records")


def _missing_warning(series: pd.DataFrame) -> list[str]:
    missing = int(series["value"].isna().sum())
    if not missing:
        return []
    return [f"{missing} eksik değerli dönem atlandı (doldurulmadı, CLAUDE.md #2)."]


def _series_for_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[["period", column]].rename(columns={column: "value"})


def _lineage_entry(ctx: ToolContext, table_ref: TableRef | None, column: str):
    """The session's recorded lineage for `column`
    (`ctx.lineage["{table_ref}.{column}"]`), or None if it has none (e.g. it
    was hand-built in a test fixture)."""
    ref = table_ref or ctx.active_table or ""
    return ctx.lineage.get(f"{ref}.{column}")


def _upstream_source_series(
    ctx: ToolContext, table_ref: TableRef | None, column: str
) -> list[str]:
    """The catalog series_id(s) behind `column`."""
    upstream = _lineage_entry(ctx, table_ref, column)
    return list(upstream.source_series) if upstream else []


# EVDS/BDDK frequency (contracts.py's Freq literal, D|W|M|Q|A) -> the STL
# seasonal period detect_anomalies needs. Weekly TP.KTF12 needs period=52,
# not the function's monthly-shaped default of 12 -- this tool layer never
# passed a period at all before, so every non-monthly column got the wrong
# seasonal cycle length.
_FREQ_TO_PERIOD = {"D": 7, "W": 52, "M": 12, "Q": 4, "A": 1}


def _upstream_period(
    ctx: ToolContext, table_ref: TableRef | None, column: str
) -> int | None:
    """STL seasonal period implied by `column`'s own frequency (via
    lineage), e.g. weekly -> 52, monthly -> 12. None when lineage/freq is
    unknown -- detect_anomalies then falls back to its own default."""
    upstream = _lineage_entry(ctx, table_ref, column)
    freq = getattr(upstream, "freq", None) if upstream else None
    return _FREQ_TO_PERIOD.get(freq)


class DetectAnomaliesArgs(BaseModel):
    table_ref: TableRef | None = Field(
        None,
        description="Analiz edilecek tablo; boş bırakılırsa aktif tablo kullanılır.",
        examples=["t01_get_series"],
    )
    column: str = Field(
        ...,
        description=(
            "Tabloda anomali aranacak sütun adı. Önceki araç çıktısından "
            "birebir kopyalayın, uydurmayın."
        ),
        examples=["TP_KTF12"],
    )


def make_detect_anomalies() -> ToolSpec:
    """STL+MAD point-anomaly detection (analysis/anomaly.py), exposed as a tool."""

    def run(args: DetectAnomaliesArgs, ctx: ToolContext) -> ToolResult:
        frame = ctx.table(args.table_ref)
        if args.column not in frame.columns:
            return ToolResult(
                ok=False,
                summary=(
                    f"'{args.column}' sütunu tabloda yok. "
                    f"Mevcut sütunlar: {', '.join(map(str, frame.columns))}."
                ),
            )

        series = _series_for_column(frame, args.column)
        warnings = _missing_warning(series)
        if len(series.dropna(subset=["value"])) < _MIN_PERIODS_FOR_STL:
            warnings.append(
                "Seri 24 dönemden kısa; STL yerine IQR (Tukey) kuralı kullanıldı."
            )

        period = _upstream_period(ctx, args.table_ref, args.column)
        try:
            if period is not None:
                result, summary = detect_anomalies(series, period=period)
            else:
                result, summary = detect_anomalies(series)
        except Exception as exc:
            return ToolResult(
                ok=False, summary=f"detect_anomalies çalıştırılamadı: {exc}"
            )

        ref = ctx.new_table_ref("detect_anomalies")
        ctx.put_table(ref, result)

        source_series = _upstream_source_series(ctx, args.table_ref, args.column)
        used_iqr = (
            len(series.dropna(subset=["value"])) < _MIN_PERIODS_FOR_STL
            or IQR_FALLBACK_NOTE in summary
        )
        formulas = _IQR_FORMULAS if used_iqr else _STL_FORMULAS
        columns = [
            ColumnInfo(
                name=name,
                op="detect_anomalies",
                formula=formulas.get(name),
                inputs=[args.column],
                source_series=source_series,
                tool="detect_anomalies",
                step_no=ctx.step_no,
            )
            for name in result.columns
        ]

        return ToolResult(
            ok=True,
            summary=summary,
            table_ref=ref,
            preview=_preview_rows(result),
            columns=columns,
            warnings=warnings,
        )

    return ToolSpec(
        name="detect_anomalies",
        description_tr=(
            "Bir tablodaki sütunda nokta anomalilerini tespit eder "
            "(STL + MAD; 24 dönemden kısa seride IQR)."
        ),
        args_model=DetectAnomaliesArgs,
        fn=run,
        brief_name="Analysis",
    )


class DetectChangesArgs(BaseModel):
    table_ref: TableRef | None = Field(
        None,
        description="Analiz edilecek tablo; boş bırakılırsa aktif tablo kullanılır.",
        examples=["t01_get_series"],
    )
    column: str = Field(
        ...,
        description=(
            "Tabloda kırılma (yapısal değişim) aranacak sütun adı. Önceki "
            "araç çıktısından birebir kopyalayın, uydurmayın."
        ),
        examples=["TP_KTF12"],
    )


def make_detect_changes() -> ToolSpec:
    """PELT structural-break detection (analysis/change.py), exposed as a tool."""

    def run(args: DetectChangesArgs, ctx: ToolContext) -> ToolResult:
        frame = ctx.table(args.table_ref)
        if args.column not in frame.columns:
            return ToolResult(
                ok=False,
                summary=(
                    f"'{args.column}' sütunu tabloda yok. "
                    f"Mevcut sütunlar: {', '.join(map(str, frame.columns))}."
                ),
            )

        series = _series_for_column(frame, args.column)
        warnings = _missing_warning(series)

        try:
            result, summary = detect_changes(series)
        except Exception as exc:
            return ToolResult(
                ok=False, summary=f"detect_changes çalıştırılamadı: {exc}"
            )

        ref = ctx.new_table_ref("detect_changes")
        ctx.put_table(ref, result)

        source_series = _upstream_source_series(ctx, args.table_ref, args.column)
        formulas = {
            "ortalama_öncesi": "kırılma öncesi bölümün ortalaması",
            "ortalama_sonrası": "kırılma sonrası bölümün ortalaması",
            "eğim_öncesi": "kırılma öncesi bölümün doğrusal regresyon eğimi",
            "eğim_sonrası": "kırılma sonrası bölümün doğrusal regresyon eğimi",
            "seviye_degisim_yuzde": (
                "(ortalama_sonrası - ortalama_öncesi) / |ortalama_öncesi| × 100"
            ),
            "sicrama_yuzde": (
                "kırılma anındaki değer ile bir önceki dönemin değeri arasındaki "
                "yüzde değişim (anlık sıçrama, ortalama farkı değil)"
            ),
        }
        columns = [
            ColumnInfo(
                name=name,
                op="detect_changes",
                formula=formulas.get(name),
                inputs=[args.column],
                source_series=source_series,
                tool="detect_changes",
                step_no=ctx.step_no,
            )
            for name in result.columns
        ]

        return ToolResult(
            ok=True,
            summary=summary,
            table_ref=ref,
            preview=_preview_rows(result),
            columns=columns,
            warnings=warnings,
        )

    return ToolSpec(
        name="detect_changes",
        description_tr=(
            "Bir tablodaki sütunda yapısal kırılma (seviye/eğim değişimi) "
            "tespit eder (PELT)."
        ),
        args_model=DetectChangesArgs,
        fn=run,
        brief_name="Analysis",
    )


#: Track B's tools/registry.py wires these into the live registry -- this
#: module only builds the specs, it never registers itself.
TOOLS = [make_detect_anomalies(), make_detect_changes()]
