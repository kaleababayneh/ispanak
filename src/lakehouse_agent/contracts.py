"""Shared, dependency-light data contracts for both lanes.

These types are the written interface between the data lane (ingest/lake/analysis)
and the agent lane (llm/agent/tools). Only pydantic is imported here on purpose:
`lake/quality.py` must be able to import `QualityReport` without pulling in the
agent package, and the agent must be able to import it without pulling in pandas.

SINGLE DEFINITION RULE: the classes in this module are never redefined anywhere
else in the codebase. A second definition is a review blocker, not a style nit —
two copies of `ToolResult` silently diverge and the mismatch only shows up when
the two lanes are merged.

Docstrings and identifiers are English (team rule). `Field(description=...)` text
is Turkish because those strings are rendered into the JSON schema the model sees.
"""

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

# A table reference is generated, never invented. `ToolContext.new_table_ref()`
# produces `t01_get_series`, `t02_deflate`, ... and this pattern is also sent to
# the model inside the tool schema, so a hallucinated name like "konut_tablosu"
# is rejected by validation before it can reach a tool.
TABLE_REF_PATTERN = r"^t\d{2,}_[a-z][a-z0-9_]{0,24}$"

TableRef = Annotated[str, StringConstraints(pattern=TABLE_REF_PATTERN)]

# A chart reference: c01, c02, ...
CHART_REF_PATTERN = r"^c\d{2,}$"

ChartRef = Annotated[str, StringConstraints(pattern=CHART_REF_PATTERN)]

Kind = Literal["stock", "flow", "rate", "index"]
"""What a series measures. Drives both the maths (a balance aggregates to the
period's last value, a flow sums, a rate averages) and the wording of the answer."""

Freq = Literal["D", "W", "M", "Q", "A"]

AggMethod = Literal["last", "sum", "avg", "first", "min", "max"]

CumType = Literal["none", "ytd", "since_start"]


class SeriesNotFound(LookupError):
    """Raised when a requested series id is not in the catalog.

    Carries the ids so the caller can put them in a message the model can act on.
    """

    def __init__(self, missing: list[str], available_hint: str | None = None) -> None:
        self.missing = list(missing)
        self.available_hint = available_hint
        detail = ", ".join(self.missing)
        message = f"Katalogda bulunamayan seri kimlikleri: {detail}."
        if available_hint:
            message = f"{message} {available_hint}"
        super().__init__(message)


class Usage(BaseModel):
    """Token and time counters. Summable on purpose: the same type is used for a
    single LLM call and for a whole session total.

    Deliberately four plain integers. Fields like `model` or `prompt_versions`
    are not here because they have no meaning under addition — that information
    lives per-step in `Step` and per-call in the usage log.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    duration_ms: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
            duration_ms=self.duration_ms + other.duration_ms,
        )

    def __sub__(self, other: "Usage") -> "Usage":
        """The cost between two readings of a running total.

        Clamped at zero: a negative token count is not a thing that can be
        reported, and the only way to get one is subtracting readings in the
        wrong order -- better a zero than a number nobody can interpret.
        """
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=max(0, self.prompt_tokens - other.prompt_tokens),
            completion_tokens=max(0, self.completion_tokens - other.completion_tokens),
            calls=max(0, self.calls - other.calls),
            duration_ms=max(0, self.duration_ms - other.duration_ms),
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Source(BaseModel):
    """Where a number came from. One per series or per fetched URL.

    Every figure in an answer must trace back to one of these; the trust layer
    the brief asks for is built by showing this list next to the answer.
    """

    series_id: str | None = Field(
        None,
        description="Katalog kimligi, ornek 'evds:TP.KTF12'. Web kaynaklarinda bos birakilir.",
        examples=["evds:TP.KTF12", "bddk:aylik:tuketici:konut:TL:sektor"],
    )
    url: str | None = Field(
        None,
        description="Okunan web adresi. Seri kaynaklarinda bos birakilir.",
    )
    name: str | None = Field(
        None, description="Kaynagin okunur adi: seri adi veya sayfa basligi."
    )
    fetched_at: datetime = Field(
        ..., description="Verinin kaynaktan cekildigi an (UTC)."
    )
    metadata_url: str | None = Field(
        None, description="Serinin metaveri sayfasi, varsa."
    )
    note: str | None = Field(
        None,
        description="Kisa not: birim cevrimi, frekans donusumu veya yapilan varsayim.",
    )

    @model_validator(mode="after")
    def _require_series_id_or_url(self) -> "Source":
        if not self.series_id and not self.url:
            raise ValueError("Source needs either series_id or url")
        return self

    def key(self) -> str:
        """Dedupe key. The session keeps one entry per underlying source."""
        return self.series_id or self.url or ""


class ColumnInfo(BaseModel):
    """Provenance for one column of a table: where it came from and what was done to it.

    `source_series` and `inputs` answer different questions and both are kept:
    `source_series` lists the external series ids behind the column (a deflated
    column genuinely has two), while `inputs` lists the column names in the same
    table that fed the operation.
    """

    name: str = Field(..., description="Tablodaki sutun adi.")
    label_tr: str | None = Field(
        None, description="Arayuzde gosterilecek Turkce baslik."
    )
    unit: str | None = Field(
        None,
        description="Birim: '%', 'milyon TL', 'endeks' gibi.",
        examples=["%", "milyon TL"],
    )
    kind: Kind | None = Field(
        None, description="Buyuklugun turu: stock, flow, rate veya index."
    )
    freq: Freq | None = Field(None, description="Sutunun frekansi.")
    source_series: list[str] = Field(
        default_factory=list,
        description=(
            "Sutunun dayandigi series_id listesi. Turetilmis sutunlarda girdilerin "
            "hepsi yazilir, ornek reel kredi icin hem nominal seri hem TUFE."
        ),
    )
    source_url: str | None = Field(
        None, description="Sutun bir web kaynagindan geldiyse adresi."
    )
    op: str | None = Field(
        None,
        description="Uygulanan islem adi: 'deflate', 'diff', 'to_freq', 'rebase' gibi.",
    )
    formula: str | None = Field(
        None,
        description="Insan okunur formul, ornek 'reel = nominal x TUFE_baz / TUFE_t'.",
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Islemin girdi olarak kullandigi SUTUN adlari (ayni tablodan).",
    )
    tool: str | None = Field(None, description="Sutunu ureten aracin adi.")
    step_no: int | None = Field(None, description="Sutunun uretildigi adim numarasi.")


class Artifact(BaseModel):
    """Something produced alongside a table: a chart, a file, a rendered report."""

    type: Literal["chart", "file", "report"] = Field(..., description="Ciktinin turu.")
    ref: str = Field(
        ..., description="Oturum deposundaki anahtar, ornek 'c01'.", examples=["c01"]
    )
    title: str | None = Field(None, description="Basligi, arayuzde gosterilir.")

    @model_validator(mode="after")
    def _chart_ref_is_generated(self) -> "Artifact":
        """Chart refs are generated, so they get the same shield as table refs.

        File and report refs are filenames chosen by the exporter, so they are
        left free; only charts are looked up by key in the session store.
        """
        if self.type == "chart" and not re.fullmatch(CHART_REF_PATTERN, self.ref):
            raise ValueError(
                f"Grafik anahtari uretilmis olmali ('c01' bicimi), verilen: {self.ref!r}"
            )
        return self


class QualityReport(BaseModel):
    """The quality card for one table.

    Produced by the data lane for pool series and by `profile_table` for anything
    that arrives at demo time. The brief calls data quality a differentiator, so
    this travels with the table instead of being computed on demand.
    """

    table_ref: str | None = Field(None, description="Ilgili tablonun anahtari.")
    rows: int = Field(..., description="Satir sayisi.")
    date_range: tuple[str, str] | None = Field(
        None, description="Ilk ve son donem, 'YYYY-AA' bicimde."
    )
    freq: Freq | None = Field(None, description="Cikarilan frekans.")
    missing_periods: int = Field(0, description="Eksik donem sayisi.")
    missing_pct: float = Field(0.0, description="Eksik deger yuzdesi.")
    duplicates: int = Field(0, description="Tekrarli donem sayisi.")
    outliers: int = Field(0, description="Basit IQR kuralina gore aykiri deger sayisi.")
    unit_guess: str | None = Field(
        None, description="Birim tahmini; bilinmiyorsa bos, bu bir kalite bayragidir."
    )
    issues: list[str] = Field(
        default_factory=list, description="Insan okunur sorun listesi."
    )


class CatalogHit(BaseModel):
    """One candidate from a catalog search.

    `score` is a ranking score whose scale depends on the backend. Hybrid search
    returns reciprocal-rank-fusion values that are not in 0..1 and are only
    comparable inside a single query, so ambiguity is measured as a ratio between
    the top two hits, never as an absolute gap.
    """

    series_id: str = Field(..., description="Katalog kimligi.")
    name_tr: str = Field(..., description="Serinin Turkce adi.")
    name_en: str | None = Field(None, description="Serinin Ingilizce adi.")
    source: str | None = Field(None, description="Kaynak: 'EVDS', 'BDDK-Aylik' gibi.")
    freq: Freq | None = Field(None, description="Yayim frekansi.")
    unit: str | None = Field(None, description="Birim.")
    kind: Kind | None = Field(None, description="Buyuklugun turu.")
    agg_method: AggMethod | None = Field(
        None, description="Varsayilan toplulastirma yontemi."
    )
    cum_type: CumType | None = Field(
        None, description="Ham veride kumulatiflik durumu."
    )
    start_date: str | None = Field(None, description="Serinin baslangic donemi.")
    end_date: str | None = Field(None, description="Serinin bitis donemi.")
    score: float = Field(
        ...,
        description=(
            "Siralama skoru. Olcegi kaynaga baglidir ve yalnizca ayni sorgu "
            "icindeki adaylar arasinda karsilastirilabilir."
        ),
    )


class ToolResult(BaseModel):
    """What every tool returns.

    The single most important rule in this file: the model never sees the table,
    it sees `summary` plus `preview`. The full frame stays in the session store
    and goes straight to the UI. That keeps the context small and stops the model
    from copying figures by hand and shifting a digit.
    """

    ok: bool = Field(..., description="Islem basarili mi.")
    summary: str = Field(
        ...,
        description=(
            "Modele giden 2-5 cumle: ne yapildi, kac satir, hangi aralik, hangi uyarilar."
        ),
    )
    table_ref: TableRef | None = Field(
        None, description="Oturum deposundaki tablo anahtari, ornek 't01_get_series'."
    )
    preview: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ilk ve son 5 satir; modelin ne var gormesine yeter.",
    )
    columns: list[ColumnInfo] = Field(
        default_factory=list, description="Uretilen veya eklenen sutunlarin soyagaci."
    )
    sources: list[Source] = Field(
        default_factory=list, description="Kullanilan kaynaklar."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Kullaniciya soylenmesi gereken uyarilar, ornek birim cevrimi yapildi.",
    )
    artifacts: list[Artifact] = Field(
        default_factory=list, description="Grafik veya dosya gibi yan ciktilar."
    )
    quality: QualityReport | None = Field(
        None, description="Yeni bir tablo uretildiyse kalite karti."
    )
