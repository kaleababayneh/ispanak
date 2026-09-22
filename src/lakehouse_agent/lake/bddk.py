"""BDDK Aylık Bülten, bronze -> silver: every table, every row, every value
column (sector total, TL report).

Bronze is one JSON file per table x month (ingest/bddk_monthly.py
`fetch_table_month`), the response as it came. What a table measures is in
ingest/bddk_tables.yaml; everything else comes from the response itself:

* the unit from the table's caption ("Bilanço (milyon TL), Dönem:2026/6"),
  unless the row or the column states its own -- "... (%)", "(YÜZDE)",
  "(Bin TL)", "(Gün)", or a count ("Müşteri Sayısı", "Mudi Sayısı");
* a series id from the table, the row label and the column name, so a
  series stays the same series when BDDK reorders rows. The label is taken
  without its formula ("Risk Ağırlıklı Kalemler Toplamı (10+30+31)" was
  "(10+27+28)" in earlier months: one item, renumbered); the scenario's
  three ids are kept through the yaml's aliases;
* a row label that repeats inside one table (the deposit tables list
  "a) Gerçek Kişiler" under six parents) is qualified with the bold row
  above it.

Year-to-date tables (`cum_type: ytd`, the profit-and-loss lines) become
monthly flows here: `analysis.transforms.decumulate_ytd`. Gold keeps the
flow; `cum_type` records what the source published.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from lakehouse_agent.analysis import transforms
from lakehouse_agent.ingest import bddk_monthly
from lakehouse_agent.lake.silver import (
    OBSERVATION_COLUMNS,
    _dedupe_latest_fetch,
    _to_float_bddk,
)

logger = logging.getLogger(__name__)

TABLES_YAML = Path("ingest/bddk_tables.yaml")
MONTHLY_SOURCE = "BDDK-Aylik"
MONTHLY_METADATA_URL = "https://www.bddk.org.tr/BultenAylik/"

#: cell = [taraf, sira, ad, stil, value columns...]; colNames runs alongside.
_FIRST_VALUE = 4

_CAPTION = re.compile(r"^(?P<name>.*?)\s*(?:\((?P<unit>[^()]*)\))?\s*(?:,\s*Dönem.*)?$")
_ROW_UNIT = re.compile(r"\((%|YÜZDE|Bin TL|Kişi|Gün|Adet)\)[\s*]*$", re.IGNORECASE)
_FOLD = str.maketrans("çğıöşüâîûÇĞÖŞÜ", "cgiosuaiucgosu")
_HTML_TAG = re.compile(r"<[^>]+>")
#: A formula's row references: "(3+4)", "(16+17+18)".
_ROW_REFS = re.compile(r"\(([\d+\s]+)\)")
#: A bracket holding only row numbers and arithmetic: "(2+3+4)", "(1+...+14)",
#: "[(26+34+50)-45]", "((6/7)*100)". Removed innermost-first.
_FORMULA = re.compile(r"[(\[][\d\s+\-*/.…,]*[)\]]")


@dataclass(frozen=True)
class TableDef:
    tablo_no: int
    key: str
    kind: str
    agg_method: str
    cum_type: str


def load_tables(
    path: Path = TABLES_YAML,
) -> tuple[dict[int, TableDef], dict[tuple[int, str, str], str]]:
    """The table definitions, and the aliases keyed by (table, row, column)
    in their folded spelling."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tables = {entry["tablo_no"]: TableDef(**entry) for entry in raw["tables"]}
    aliases = {
        (spec["tablo_no"], _fold(spec["satir"]), _fold(spec["sutun"])): series_id
        for series_id, spec in (raw.get("aliases") or {}).items()
    }
    return tables, aliases


def _fold(text: str) -> str:
    folded = text.replace("İ", "i").replace("I", "ı").lower().translate(_FOLD)
    return " ".join(folded.split())


def label_key(label: str) -> str:
    """A row label without its formula references and footnote marks: what
    stays the same when BDDK renumbers the rows a total adds up."""
    text = label
    while True:
        stripped = _FORMULA.sub(" ", text)
        if stripped == text:
            break
        text = stripped
    return " ".join(text.replace("*", " ").split()).strip(" -")


def slug(text: str) -> str:
    """Lower-case ASCII words joined by "_": "Tüketici Kredileri - Konut" ->
    "tuketici_kredileri_konut"."""
    return re.sub(r"[^a-z0-9]+", "_", _fold(text)).strip("_")


def caption_parts(caption: str) -> tuple[str, str | None]:
    """("Bilanço", "milyon TL") from "Bilanço (milyon TL), Dönem:2026/6"; a
    caption without a unit ("Rasyolar") gives None."""
    match = _CAPTION.match(caption.strip())
    if not match:
        return caption.strip(), None
    name = " ".join(match.group("name").split())
    unit = match.group("unit")
    return name, unit.strip() if unit else None


def row_labels(rows: list[dict]) -> list[str]:
    """Each row's label, qualified where it would be ambiguous: see
    `qualified_labels`. The monthly bulletin marks section rows "bold"."""
    return qualified_labels(
        [
            (
                int(row["cell"][1]) if str(row["cell"][1]).isdigit() else 0,
                str(row["cell"][2]).strip(),
                str(row["cell"][3]).strip() == "bold",
            )
            for row in rows
        ]
    )


#: A list item's marker: "a) Konut".
_LIST_ITEM = re.compile(r"^[a-zçğıöşü]\)\s")


def qualified_labels(rows: list[tuple[int, str, bool]]) -> list[str]:
    """Labels for (sira, label, is_section) rows, each unambiguous alone.

    A label is qualified with its parent when it is a list item ("a)
    Konut", which says nothing without "Tüketici Kredileri") or when its
    formula-free form repeats in the table ("Bankalar (16+17+18)" and
    "Bankalar (38+39+40)" are two different sections' banks). The parent is
    the earlier row whose formula lists this one ("Tüketici Kredileri
    (4+5+6)"); else, for a list item, the row its list hangs from (the last
    row above that is not a list item: "Kredilerden Alınan Ücret ve
    Komisyonlar" has no formula, and the bold row above it is another
    section); else the nearest section row above. A parent that was itself
    qualified lends its qualified label, so "a) Kamu Borçlanma Senetleri"
    under the banks of two different sections stays two series.
    """
    keys = Counter(label_key(label) for _, label, _ in rows)
    listed_by: dict[int, int] = {}
    for index, (_sira, label, _section) in enumerate(rows):
        for group in _ROW_REFS.findall(label):
            for number in re.findall(r"\d+", group):
                listed_by.setdefault(int(number), index)
    out: list[str] = []
    section: int | None = None
    head: int | None = None
    for index, (sira, label, is_section) in enumerate(rows):
        is_item = bool(_LIST_ITEM.match(label))
        parent = listed_by.get(sira)
        if parent is None or parent >= index:
            parent = head if is_item and head is not None else section
        if (keys[label_key(label)] > 1 or is_item) and parent is not None:
            out.append(f"{label_key(out[parent])} - {label}")
        else:
            out.append(label)
        if not is_item:
            head = index
        if is_section:
            section = index
    return out


def row_unit(label: str, column: str, table_unit: str | None) -> str | None:
    """The unit a value is in: the row's or the column's own when stated,
    else the table's. None when nothing states one -- not guessed."""
    match = _ROW_UNIT.search(label)
    if match:
        unit = match.group(1)
        return "%" if unit in ("%", "YÜZDE") or _fold(unit) == "yuzde" else unit
    if "sayisi" in _fold(label) or "sayisi" in _fold(column):
        return "Adet"
    if table_unit:
        return table_unit
    if _fold(column) == "adet":
        return "Adet"
    return None


def row_kind(table: TableDef, unit: str | None) -> tuple[str | None, str]:
    """(kind, agg_method) for one value. A percentage is a rate whatever the
    table; a ratio table's value in "Bin TL" or "Gün" (assets per employee,
    average maturity) is not a percentage, so it gets no kind."""
    if unit == "%":
        return "rate", "last"
    if table.kind == "rate" and unit is not None:
        return None, table.agg_method
    if unit == "Adet":
        return "stock", "last"
    return table.kind, table.agg_method


def _notes(payload_json: dict) -> str | None:
    text = _HTML_TAG.sub(" ", str(payload_json.get("uyari") or ""))
    text = " ".join(text.split())
    return text[:600] or None


def monthly_records(root: Path | None = None) -> pd.DataFrame:
    """Every value of every bronze table file, one row each: the raw value,
    its period, unit and the metadata its series needs."""
    root = root or bddk_monthly.TABLES_ROOT
    tables, aliases = load_tables()
    out: list[dict] = []
    for path in sorted(root.glob("t*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        params = record["params"]
        table = tables.get(int(params["tabloNo"]))
        if table is None:
            continue
        body = record["payload"].get("Json", {})
        rows = body.get("data", {}).get("rows", []) or []
        columns = [str(c).strip() for c in body.get("colNames", [])[_FIRST_VALUE:]]
        table_name, table_unit = caption_parts(str(body.get("caption") or ""))
        notes = _notes(body)
        period = date(int(params["yil"]), int(params["ay"]), 1)
        seen: set[str] = set()
        for row, label in zip(rows, row_labels(rows), strict=True):
            cell = row["cell"]
            for offset, column in enumerate(columns):
                position = _FIRST_VALUE + offset
                if position >= len(cell):
                    continue
                series_id = aliases.get(
                    (table.tablo_no, _fold(label), _fold(column)),
                    f"bddk:aylik:{table.key}:{slug(label_key(label))}:"
                    f"{slug(column)}:sektor",
                )
                if series_id in seen:
                    # Two rows of one table-month on one id: keep both, apart,
                    # rather than let deduplication drop one without a word.
                    logger.warning(
                        "BDDK %s %s: '%s' ayni kimlige dustu, satir no eklendi",
                        path.parent.name,
                        path.stem,
                        label,
                    )
                    series_id = f"{series_id}:r{cell[1]}"
                seen.add(series_id)
                unit = row_unit(label, column, table_unit)
                kind, agg_method = row_kind(table, unit)
                out.append(
                    {
                        "series_id": series_id,
                        "period": period,
                        "raw": cell[position],
                        "unit": unit,
                        "kind": kind,
                        "agg_method": agg_method,
                        "cum_type": table.cum_type,
                        "name_tr": f"{label} ({column})",
                        "topic": table_name,
                        "tablo_no": table.tablo_no,
                        "notes": notes,
                        "fetched_at": record["fetched_at"],
                    }
                )
    return pd.DataFrame(out)


def build_monthly(
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(observations, catalog rows, dropped duplicates) for the whole bulletin.

    Observations follow the gold contract; `obs_date` is the period (the
    bulletin names a month, not a day). Year-to-date series are turned into
    monthly flows. A series' catalog row describes its latest month.
    """
    records = monthly_records(root)
    if records.empty:
        empty = pd.DataFrame(columns=OBSERVATION_COLUMNS)
        return empty, pd.DataFrame(), 0

    records["value"] = records["raw"].map(_to_float_bddk)
    observations = pd.DataFrame(
        {
            "series_id": records["series_id"],
            "period": records["period"],
            "freq": "M",
            "value": records["value"],
            "unit": records["unit"],
            "source": MONTHLY_SOURCE,
            "fetched_at": records["fetched_at"],
            "obs_date": records["period"],
        }
    )
    observations, dropped = _dedupe_latest_fetch(observations)

    ytd = set(records.loc[records["cum_type"] == "ytd", "series_id"])
    if ytd:
        frames = []
        for series_id, part in observations.groupby("series_id", sort=False):
            if series_id in ytd:
                flows, _ = transforms.decumulate_ytd(part[["period", "value"]])
                part = part.sort_values("period").copy()
                part["value"] = flows["value"].to_numpy()
            frames.append(part)
        observations = pd.concat(frames, ignore_index=True)

    latest = records.sort_values(["period", "fetched_at"]).groupby("series_id").tail(1)
    catalog = pd.DataFrame(
        {
            "series_id": latest["series_id"],
            "name_tr": latest["name_tr"],
            "name_en": None,
            "source": MONTHLY_SOURCE,
            "freq": "M",
            "unit": latest["unit"],
            "kind": latest["kind"],
            "agg_method": latest["agg_method"],
            "cum_type": latest["cum_type"],
            "start_date": None,
            "end_date": None,
            "topic": latest["topic"],
            "tags": "BDDK Aylık Bülten, sektör",
            "metadata_url": MONTHLY_METADATA_URL,
            "notes": latest["notes"],
        }
    ).reset_index(drop=True)

    units = records.groupby("series_id")["unit"].nunique(dropna=False)
    if (units > 1).any():
        logger.warning(
            "BDDK: %d seride birim donem icinde degisiyor: %s",
            int((units > 1).sum()),
            list(units[units > 1].index[:5]),
        )
    return observations[OBSERVATION_COLUMNS], catalog, dropped


# --- the weekly bulletin ------------------------------------------------

WEEKLY_SOURCE = "BDDK-Haftalik"
WEEKLY_METADATA_URL = "https://www.bddk.org.tr/BultenHaftalik/"

_WEEKLY_CAPTION = re.compile(
    r"/\s*(?P<table>[^()]+?)\s*\(\s*(?P<day>\d{1,2})\s+(?P<month>\S+)\s+(?P<year>\d{4})"
    r"\s+\S+\s*\)\s*\((?P<unit>[^()]+)\)"
)
_TR_MONTHS = {
    "ocak": 1,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "eylul": 9,
    "ekim": 10,
    "kasim": 11,
    "aralik": 12,
}


def _cells(fragment: str) -> list[tuple[bool, list[str]]]:
    """The fragment's table rows: (is a bold section row, cell texts)."""
    import html as html_lib

    rows = []
    for attrs, row in re.findall(r"<tr([^>]*)>(.*?)</tr>", fragment, re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        texts = [
            " ".join(
                html_lib.unescape(_HTML_TAG.sub("", c)).replace("\xa0", " ").split()
            )
            for c in cells
        ]
        if any(texts):
            rows.append(("bold" in attrs, texts))
    return rows


def weekly_records(root: Path | None = None) -> pd.DataFrame:
    """Every value of every stored weekly table, one row each."""
    from lakehouse_agent.ingest import bddk_weekly

    root = root or bddk_weekly.BRONZE_ROOT
    out: list[dict] = []
    for path in sorted(root.glob("t*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        marked = _cells(record["html"])
        rows = [cells for _bold, cells in marked]
        # The header row: an empty cell, the caption ("Sektör / Krediler ( 8
        # Ocak 2021 Cuma ) (Milyon TL)"), then the value columns' names.
        header = next(
            (row for row in rows if any(_WEEKLY_CAPTION.search(cell) for cell in row)),
            None,
        )
        if header is None:
            logger.warning("BDDK haftalik %s: baslik satiri okunamadi", path)
            continue
        caption_at = next(
            i for i, cell in enumerate(header) if _WEEKLY_CAPTION.search(cell)
        )
        match = _WEEKLY_CAPTION.search(header[caption_at])
        table_name = match.group("table").strip()
        observed = date(
            int(match.group("year")),
            _TR_MONTHS[_fold(match.group("month"))],
            int(match.group("day")),
        )
        table_unit = match.group("unit").strip()
        columns = header[caption_at + 1 :]
        body = [(bold, r) for bold, r in marked if r and r[0].isdigit() and len(r) >= 3]
        if not body:
            continue
        labels = qualified_labels([(int(r[0]), r[1], bold) for bold, r in body])
        key = slug(table_name)
        period = observed - pd.Timedelta(days=observed.weekday())
        for (_bold, row), label in zip(body, labels, strict=True):
            for column, raw in zip(columns, row[2:], strict=False):
                unit = row_unit(label, column, table_unit)
                kind = "rate" if unit == "%" else "stock"
                out.append(
                    {
                        "series_id": (
                            f"bddk:haftalik:{key}:{slug(label_key(label))}:"
                            f"{slug(column)}:sektor"
                        ),
                        "period": period,
                        "obs_date": observed,
                        "raw": raw,
                        "unit": unit,
                        "kind": kind,
                        "agg_method": "last",
                        "cum_type": "none",
                        "name_tr": f"{label} ({column})",
                        "topic": f"Haftalık Bülten: {table_name}",
                        "fetched_at": record["fetched_at"],
                    }
                )
    return pd.DataFrame(out)


def build_weekly(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(observations, catalog rows, dropped duplicates) for the weekly
    bulletin: balances at each Friday, stored under the week's Monday with
    the Friday as `obs_date`."""
    records = weekly_records(root)
    return _assemble(
        records,
        source=WEEKLY_SOURCE,
        freq="W",
        tags="BDDK Haftalık Bülten, sektör",
        metadata_url=WEEKLY_METADATA_URL,
    )


# --- FinTürk (provinces) ------------------------------------------------

FINTURK_SOURCE = "BDDK-FinTurk"
FINTURK_METADATA_URL = "https://www.bddk.org.tr/BultenFinTurk"

#: Short keys for the seven FinTürk tables, by tabloNo.
_FINTURK_KEYS = {
    1: "krediler",
    2: "mevduat",
    3: "bireysel",
    4: "sektorel",
    5: "oranlar",
    6: "subeler",
    7: "altin",
}
_FINTURK_FIRST_VALUE = 5  # cell = [taraf, yil, ay, sehir, grup, values...]


def finturk_unit(column: str, table_unit: str | None) -> tuple[str | None, str | None]:
    """(unit, kind) for one FinTürk column. The "Şubeler" table mixes a count
    (branches), people per branch and TL per head: only the count is a stock."""
    folded = _fold(column)
    if "sayisi" in folded:
        return "Adet", "stock"
    if "nufus" in folded:
        return "Kişi", None
    if "kisi basi" in folded:
        return "TL", None
    if table_unit == "%":
        return "%", "rate"
    return table_unit, "stock"


def finturk_records(root: Path | None = None) -> pd.DataFrame:
    from lakehouse_agent.ingest import bddk_finturk

    root = root or bddk_finturk.BRONZE_ROOT
    out: list[dict] = []
    for path in sorted(root.glob("t*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        tablo_no = int(record["params"]["tabloNo"])
        table_name, table_unit = caption_parts(str(record.get("table") or ""))
        body = record["payload"].get("Json", {})
        columns = [
            str(c).strip() for c in body.get("colNames", [])[_FINTURK_FIRST_VALUE:]
        ]
        for row in body.get("data", {}).get("rows", []) or []:
            cell = row["cell"]
            year, month, province = int(cell[1]), int(cell[2]), str(cell[3]).strip()
            period = date(year, 3 * ((month - 1) // 3) + 1, 1)
            for offset, column in enumerate(columns):
                position = _FINTURK_FIRST_VALUE + offset
                if position >= len(cell):
                    continue
                unit, kind = finturk_unit(column, table_unit)
                out.append(
                    {
                        "series_id": (
                            f"bddk:finturk:{_FINTURK_KEYS.get(tablo_no, tablo_no)}:"
                            f"{slug(province)}:{slug(column)}:sektor"
                        ),
                        "period": period,
                        "obs_date": period,
                        "raw": cell[position],
                        "unit": unit,
                        "kind": kind,
                        "agg_method": "last",
                        "cum_type": "none",
                        "name_tr": f"{province} - {column}",
                        "topic": f"FinTürk: {table_name}",
                        "fetched_at": record["fetched_at"],
                    }
                )
    return pd.DataFrame(out)


def build_finturk(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(observations, catalog rows, dropped duplicates) for FinTürk: one
    series per province x table column, quarterly."""
    return _assemble(
        finturk_records(root),
        source=FINTURK_SOURCE,
        freq="Q",
        tags="BDDK FinTürk, il, sektör",
        metadata_url=FINTURK_METADATA_URL,
    )


def _assemble(
    records: pd.DataFrame, *, source: str, freq: str, tags: str, metadata_url: str
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if records.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS), pd.DataFrame(), 0
    observations = pd.DataFrame(
        {
            "series_id": records["series_id"],
            "period": records["period"],
            "freq": freq,
            "value": records["raw"].map(_to_float_bddk),
            "unit": records["unit"],
            "source": source,
            "fetched_at": records["fetched_at"],
            "obs_date": records["obs_date"],
        }
    )
    observations, dropped = _dedupe_latest_fetch(observations)
    latest = records.sort_values(["period", "fetched_at"]).groupby("series_id").tail(1)
    catalog = pd.DataFrame(
        {
            "series_id": latest["series_id"],
            "name_tr": latest["name_tr"],
            "name_en": None,
            "source": source,
            "freq": freq,
            "unit": latest["unit"],
            "kind": latest["kind"],
            "agg_method": latest["agg_method"],
            "cum_type": latest["cum_type"],
            "start_date": None,
            "end_date": None,
            "topic": latest["topic"],
            "tags": tags,
            "metadata_url": metadata_url,
            "notes": None,
        }
    ).reset_index(drop=True)
    return observations[OBSERVATION_COLUMNS], catalog, dropped
