"""Kind, aggregation and unit for pool series, from what the source says.

The EVDS catalog has no `kind` (stock/flow/rate/index) and leaves the unit
empty for about a third of its series, while the pool holds tens of
thousands of them -- a hand-written mapping per series (the old
`SCENARIO_SERIES`) does not scale. What EVDS does publish per series:

* `DEFAULT_AGG_METHOD` (last/avg/sum/min/max): how the source itself turns
  the series into a coarser frequency. That becomes `agg_method` as is, and
  `to_freq` uses it (docs/decisions.md, 22 Sep).
* `BIRIMI`, the unit, set per datagroup.
* The series name, which often carries the unit in a trailing
  parenthesis ("(%)", "(2010=100)").

`kind` then only says what the quantity *means* -- whether a difference is in
percentage points, whether deflating makes sense. It is derived by the rules
in `evds_kind`, in order, and a series no rule covers gets None: an average
of counts or a TL/m2 price is neither a stock nor a rate, and a guessed kind
would steer the tools wrong without anyone seeing it. Every decision carries
its reason, so the review list (`docs/kind_review.md`) can show the rule
behind each series.

Pure functions: strings in, strings out.
"""

from __future__ import annotations

import re

#: EVDS `DEFAULT_AGG_METHOD` codes are already the contract's AggMethod
#: spellings (measured on the full catalog: last, avg, sum, min, max).
EVDS_AGG_METHODS = frozenset({"last", "avg", "sum", "min", "max", "first"})

#: Units that say the value is a percentage or a rate. "Ağırlıklı ortalama"
#: is how EVDS labels its weighted-average interest rates (TP.KTF12).
_RATE_UNITS = frozenset(
    {
        "%",
        "% degisim",
        "yuzde",
        "net yuzde degisim",
        "gsyih'nin yuzdesi",
        "oran",
        "agirlikli ortalama",
    }
)
_INDEX_TEXT = re.compile(r"endeks|\b\d{4}\s*=\s*100\b")
#: Rate cues in a name. Not the bare word "faiz": "Faiz Gelirleri" is an
#: amount of interest income in TL, a flow (measured: 39 budget lines).
_RATE_NAME = re.compile(r"\(%\)|%\s*degisim|faiz orani|\boran[i]?\s*\(%|yuzde")

#: Units read out of a series name's trailing parenthesis, and only these:
#: explicit unit words, never codes such as "24D2" or notes such as "Arşiv".
_NAME_UNIT = re.compile(
    r"^(%|% De[gğ]i[sş]im|TL|Tutar TL|ABD [Dd]olar[ıi]|Adet|Gram|"
    r"(?:[İI]TO )?\d{4}=100)$"
)
_TRAILING_PARENS = re.compile(r"\(([^()]*)\)\s*$")

_FOLD = str.maketrans("çğıöşüâîûİI", "cgiosuaiuii")


def _text(value: object) -> str:
    """A catalog cell as text: pandas hands a missing one over as NaN."""
    return value if isinstance(value, str) else ""


def _fold(value: object) -> str:
    """Lower-cased, Turkish letters folded to ASCII, spaces squeezed."""
    folded = _text(value).replace("İ", "i").replace("I", "ı").lower().translate(_FOLD)
    return " ".join(folded.split())


def evds_unit(birimi: object, name: object) -> tuple[str | None, str]:
    """The series' unit and where it came from.

    `BIRIMI` when EVDS gives one. Otherwise the name's trailing parenthesis,
    but only when it is an explicit unit ("(%)", "(TL)", "(2010=100)"); the
    text is kept as written (CLAUDE.md #3: units are never converted).
    Otherwise None -- no unit is guessed.

    One exception: a name ending "(%)" wins over BIRIMI. BIRIMI is set per
    datagroup, and a mixed group gives its percentages the levels' unit --
    "8.İşsizlik oranı (%)" reads "Bin kişi" (measured: 23 such series). A
    name ending "(TL)" does not win: there BIRIMI ("bin TL") carries the
    scale the name leaves out.
    """
    unit = _text(birimi).strip()
    match = _TRAILING_PARENS.search(_text(name))
    stated = match.group(1).strip() if match else ""
    if stated.startswith("%") and _NAME_UNIT.match(stated):
        return stated, "ad (grup birimi yerine)" if unit else "ad"
    if unit:
        return unit, "BIRIMI"
    if stated and _NAME_UNIT.match(stated):
        return stated, "ad"
    return None, "yok"


def evds_kind(unit: object, name: object, agg_method: object) -> tuple[str | None, str]:
    """(kind, reason) for an EVDS series. The first rule that applies wins:

    1. the unit names an index ("Endeks", "2010=100") -> index
    2. the unit is a percentage or rate ("Yüzde", "Ağırlıklı ortalama") -> rate
    3. EVDS sums it to coarser periods (KÜMÜLATİF) -> flow
    4. the name names an index -> index
    5. the name marks a percentage or rate ("(%)", "faiz oranı") -> rate
    6. EVDS takes the period's last value (BİTİŞ) -> stock
    7. none of these -> None ("belirsiz")

    The unit is trusted before the aggregation, and the aggregation before
    the name: a unit is the source's statement about the value, a name only
    a description of it.
    """
    folded_unit, folded_name = _fold(unit), _fold(name)
    if _INDEX_TEXT.search(folded_unit):
        return "index", "birim endeks"
    if folded_unit in _RATE_UNITS:
        return "rate", "birim oran/yuzde"
    if agg_method == "sum":
        return "flow", "EVDS toplami aliyor (KUMULATIF)"
    if _INDEX_TEXT.search(folded_name):
        return "index", "ad endeks"
    if _RATE_NAME.search(folded_name):
        return "rate", "ad oran/yuzde"
    if agg_method == "last":
        return "stock", "EVDS donem sonu degeri aliyor (BITIS)"
    return None, "belirsiz: oran/endeks isareti yok, EVDS ortalama/uc deger aliyor"


def render_kind_review(classification, catalog) -> str:  # noqa: ANN001 - DataFrames
    """docs/kind_review.md: what the rules decided, and what they could not.

    `classification` is gold's classification.parquet (EVDS series, with
    the reason for each kind); `catalog` is series_catalog.parquet (every
    source). Plan 2.2's review list: the series without a kind, grouped by
    datagroup so a person can read them in one pass.
    """
    lines = [
        "# Tür (kind) gözden geçirme listesi",
        "",
        "Otomatik üretildi: `python -m lakehouse_agent.lake.gold`. Kurallar:",
        "`lake/classify.py` (EVDS), `ingest/bddk_tables.yaml` + `lake/bddk.py` (BDDK).",
        "Tür yalnızca yorumu belirler; frekans dönüşümünü `agg_method` belirler.",
        "",
        "## Kaynak ve tür",
        "",
        "| Kaynak | stock | flow | rate | index | boş |",
        "|---|---|---|---|---|---|",
    ]
    kinds = catalog.assign(kind=catalog["kind"].fillna("boş"))
    table = kinds.groupby(["source", "kind"]).size().unstack(fill_value=0)
    for source, row in table.iterrows():
        cells = [
            str(int(row.get(kind, 0)))
            for kind in ("stock", "flow", "rate", "index", "boş")
        ]
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    lines += ["", "## EVDS: hangi kural", "", "| Gerekçe | Seri |", "|---|---|"]
    for reason, count in classification["kind_reason"].value_counts().items():
        lines.append(f"| {reason} | {count} |")

    empty = classification[classification["kind"].isna()]
    lines += [
        "",
        f"## EVDS: türü boş kalan {len(empty)} seri, veri grubuna göre",
        "",
        "Oran ya da endeks işareti yok ve EVDS ortalama/uç değer alıyor: bir",
        "ortalama sayı, bir TL/m² fiyatı ne bakiye ne oran. Tür uydurulmadı;",
        "`deflate` bu serileri reddeder, `diff` birimini değiştirmez.",
        "",
        "| Veri grubu | Seri | Birimler | Örnek |",
        "|---|---|---|---|",
    ]
    names = catalog.set_index("series_id")["name_tr"]
    for group, part in empty.groupby("datagroup"):
        units = ", ".join(sorted({str(u) for u in part["unit"].dropna()})) or "-"
        example = names.get(part["series_id"].iloc[0], "")
        lines.append(f"| {group} | {len(part)} | {units} | {example} |")

    bddk_empty = catalog[
        catalog["source"].str.startswith("BDDK") & catalog["kind"].isna()
    ]
    lines += [
        "",
        f"## BDDK: türü boş kalan {len(bddk_empty)} seri",
        "",
        "Rasyo tablolarında birimi yüzde olmayan oranlar (personel başına aktif,",
        "ortalama vade) ve FinTürk'te kişi başı / şube başına düşen değerler.",
        "",
    ]
    return "\n".join(lines) + "\n"
