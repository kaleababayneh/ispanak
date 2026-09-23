"""Human labels for table columns, derived from lineage and never invented.

A column is named for the machine: `TP_KTF12`, `reel_TP_KTF12_diff`. The name
is what the model copies from one tool call into the next (planner rule 3),
so it cannot change. A person looking at a chart legend or a table header
wants something else, and the session already knows it: `ColumnInfo`
(`docs/contracts.md` section 5, key `"{table_ref}.{column}"`) carries the
catalog's Turkish name and unit for a fetched column, and the operation and
its input columns for a derived one.

So the label of `TP_KTF12` is the catalog name, and the label of
`TP_KTF12_diff` is that name followed by what was done to it. Nothing here
shortens or rewrites a catalog name; when the lineage has nothing to say, the
raw column name is returned, which is at least true.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

#: Columns every table shares, named by the tools rather than the catalog.
FIXED = {"period": "Dönem", "obs_date": "Gözlem tarihi"}

#: What each `transform_table` operation did, as a person would say it. The
#: keys are `tools/lakehouse.py`'s `Op` literal; `to_freq` is `get_series`'s
#: own resampling note and the catalog name already covers that column.
OP_PHRASES = {
    "diff": "fark",
    "pct_change": "% değişim",
    "yoy": "yıllık % değişim",
    "change_since": "% değişim",
    "rebase": "endeks",
    "ratio": "payı (%)",
    "deflate": "reel",
    "ytd": "yılbaşından bu yana",
}

#: A derived column carries its base period in its own name
#: (`_endeks_202101`, `_degisim_202101`, `reel_..._202512`). Read back from
#: there rather than from the formula text, which is written for the model.
_BASE_IN_NAME = re.compile(r"_(\d{4})(\d{2})$")

#: How far a chain of derived columns is followed. Lineage is a DAG the tools
#: build, so a cycle cannot happen; the guard is against a hand-built fixture.
MAX_DEPTH = 6


def column_label(
    table_ref: str,
    column: str,
    lineage: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
    *,
    with_unit: bool = True,
    depth: int = 0,
) -> str:
    """The label of one column, or the column's own name when nothing is known.

    `overrides` are the model's own words for some columns (what it wrote on
    the chart legend). They are consulted first, and -- the point of taking
    them here rather than pasting them over the result -- they are what a
    derived column is built from: once the model has called `TP_KTF12`
    "Konut kredisi faiz oranı", `TP_KTF12_diff` reads "Konut kredisi faiz
    oranı — fark" instead of starting again from the catalog's technical
    name. A model label is taken as written, unit and all.

    `with_unit=False` is for the input of a derived column: a rebased index
    has no "milyon TL" and a ratio of two counts is not "(Adet) / (Adet)".
    The derived column appends its own unit, which the tool recorded.
    """
    if overrides and column in overrides and overrides[column].strip():
        return overrides[column].strip()
    if column in FIXED:
        return FIXED[column]
    info = lineage.get(f"{table_ref}.{column}")
    if info is None or depth >= MAX_DEPTH:
        return column

    op = _field(info, "op")
    label = _tidy(_field(info, "label_tr"))
    if not label:
        inputs = [str(name) for name in (_field(info, "inputs") or [])]
        if op == "mark":
            # The model named this column itself (`faiz_dustu_kredi_artmadi`);
            # its inputs are the conditions, not what it measures.
            label = f"{_tidy(column)} (koşul)"
        elif op in OP_PHRASES and inputs:
            label = _derived(table_ref, column, op, inputs, lineage, overrides, depth)
        else:
            # A tool's own column (`ortalama_öncesi`, `p_degeri`): already a
            # word, just written for a machine.
            label = _tidy(column)
    if not label:
        return column

    unit = _field(info, "unit")
    if with_unit and unit and unit.casefold() not in label.casefold():
        label = f"{label} ({unit})"
    return label


def _tidy(text: Any) -> str:
    """Underscores to spaces, nothing else. `Türkiye_Konut_İpotekli Satışlar`
    is the catalog's own spelling; only the machine glue is removed."""
    return " ".join(str(text or "").replace("_", " ").split())


def _derived(
    table_ref: str,
    column: str,
    op: str,
    inputs: list[str],
    lineage: Mapping[str, Any],
    overrides: Mapping[str, str] | None,
    depth: int,
) -> str:
    """`<input label> — <what was done>`, with the base period when there is one."""
    first = column_label(
        table_ref, inputs[0], lineage, overrides, with_unit=False, depth=depth + 1
    )
    phrase = OP_PHRASES[op]
    if op == "ratio" and len(inputs) > 1:
        second = column_label(
            table_ref, inputs[1], lineage, overrides, with_unit=False, depth=depth + 1
        )
        return f"{first} / {second} {phrase}"
    base = _BASE_IN_NAME.search(column)
    # `deflate` is left out on purpose: the tool writes the price base into
    # the column's unit ("milyon TL, 2021-01 fiyatlariyla"), and naming it
    # here too printed "(2021-01 fiyatlarıyla)" twice in one header.
    if base and op in {"rebase", "change_since"}:
        when = f"{base.group(1)}-{base.group(2)}"
        phrase = (
            f"{phrase} ({when} = 100)" if op == "rebase" else f"{phrase} ({when} bazlı)"
        )
    return f"{first} — {phrase}"


def table_labels(
    table_ref: str,
    columns: Iterable[Any],
    lineage: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Every column's label, keyed by the column's name. Identity when unknown."""
    return {
        str(column): column_label(table_ref, str(column), lineage, overrides)
        for column in columns
    }


def column_units(
    table_ref: str, columns: Iterable[Any], lineage: Mapping[str, Any]
) -> dict[str, str]:
    """Each column's unit as the lineage recorded it; "" when unknown."""
    units = {}
    for column in columns:
        info = lineage.get(f"{table_ref}.{column}")
        units[str(column)] = str(_field(info, "unit") or "").strip() if info else ""
    return units


def _field(info: Any, name: str) -> Any:
    """Read a lineage field off a model or a plain dict. Tests build both."""
    if isinstance(info, Mapping):
        return info.get(name)
    return getattr(info, name, None)
