"""Checks a finished answer against what the tools actually returned (plan 5.3).

Rule-based and cheap: no model call, so it adds nothing to a turn's time
budget. Four checks:

* **Numbers.** Every number the answer states must be found -- within the
  rounding its own precision implies -- in the session's tables or in what the
  tools told the model. Measured 22 Sep 2026: a live answer turned a monthly
  rate change of -0.2785 points into "%27,85 puan"; nothing in the tables is
  within 0.005 of 27.85, so this check catches it. A bound ("200'ü aştı") is
  checked as a bound and a range ("165 ile 205 arasında") as a range.
* **Sources.** Every series id, series code or URL the answer cites must be in
  the session's source list -- and, the other way round, a turn that read the
  web must name what it read: an answer that quietly uses a page reads as if
  the lake had known it.
* **Dates.** Every month the answer names must lie inside the range the tables
  cover, unless the question, the user or a tool named it first -- a tool
  saying "havuzdaki veri 2026-06 ayinda bitiyor" and the answer repeating it
  is the answer being right.

A false flag is printed under a correct answer in front of whoever is
watching, so the checks lean towards passing what could be a faithful
restatement. What they therefore cannot see, stated rather than hidden:

* a wrong number that happens to equal some other value in the tables (the
  window is narrow, the tables are wide);
* the sign or direction: "arttı" against a value that fell passes, because
  "0,28 puan düştü" has to pass against -0.2785;
* a correct number attached to the wrong month;
* anything written out in words.

Unit mixing is flagged where it arises, by `get_series`, not here.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from lakehouse_agent.agent.formatting import without_fenced
from lakehouse_agent.agent.state import (
    SessionState,
    VerificationFlag,
    VerificationResult,
)

#: More flags than this and the list stops being read.
MAX_FLAGS = 8

#: How far a bound reaches: "200'ü aştı" is confirmed by a value between 200
#: and 250. Left open, a table holding credit in millions would confirm every
#: "above 40" anyone writes.
_BOUND_REACH = 1.25
_APPROX_SHARE = 0.05

_APPROX_BEFORE = re.compile(
    r"(?:~|≈|yaklaşık(?:\s+olarak)?|yaklasik(?:\s+olarak)?|kabaca|takriben|"
    r"aşağı\s+yukarı)\s*%?\s*$",
    re.I,
)

#: What may stand between a number and the word that qualifies it: a percent
#: sign, a suffix after either apostrophe, a scale word, a unit. "800 milyar
#: TL'nin üzerine" and "%40’ın üstüne" are one bound each.
#:
#: `rest` keeps the next few words rather than only the next one, because
#: Turkish puts a noun between the number and the verb that bounds it:
#: "%40 seviyesinin üzerine çıktı", "%40 eşiğini aştı", "%40'tan daha fazla".
#: With one word these read as exact claims and a correct sentence is
#: flagged (measured). The scan stops at punctuation on its own, so a comma
#: or a full stop ends the phrase before an unrelated verb can be read as a
#: bound.
_TAIL = re.compile(
    r"^\s*%?(?:['’]\w*)?"
    r"(?:\s*(?P<scale>bin|milyon|milyar|trilyon)(?:['’]?\w*)?)?"
    # "20 bin milyon TL": a multiplier and then the table's own scale.
    r"(?:\s*(?:milyon|milyar)(?:['’]?\w*)?)?"
    r"(?:\s*(?P<unit>baz\s+puan\w*|puan\w*|TL\w*|lira\w*|adet\w*)(?:['’]\w*)?)?"
    # `\s*` rather than `\s+`: the pattern's own leading `^\s*` has already
    # eaten the space in "200 seviyesinin", and a word cannot run straight
    # into the number (the number pattern's lookahead forbids it).
    r"(?P<rest>(?:\s*\w+){0,3})",
    re.I,
)

#: A negated verb anywhere in the qualifying phrase. "%50'yi geçmedi" and
#: "%40'ın altına inmedi" are true of a series that never reached 50 or never
#: fell below 40, and this checker has no way to test that -- read as bounds
#: they say the opposite. A claim it cannot express is left alone rather than
#: flagged under a correct answer.
_NEGATED = re.compile(r"(?:m[ae]d[ıi]|m[ae]m[ıi]ş|m[ae]z|m[ıiuü]yor)\b", re.I)
_BELOW_WORDS = re.compile(r"^(?:alt[ıi]|aşağı|asagi|düşük|dusuk|kadar|geride)", re.I)
_ABOVE_WORDS = re.compile(
    r"^(?:üzer|uzer|üst|ust|aşt|aşk|aşar|aşan|aşm|geçt|geçe|geçm|geçer|fazla|"
    r"yüksek|yuksek|ileri)",
    re.I,
)
_APPROX_WORDS = re.compile(r"^(?:civar|dolay)", re.I)

#: A scale word read three ways, because the table it is compared with can be
#: in TL, bin TL or milyon TL: "804 milyar TL" is 804e3 in milyon TL, 804e6 in
#: bin TL and 804e9 in TL, and "20 bin milyon TL" is 20e3 on top of all that.
_SCALE_FACTORS = {
    "bin": (1.0, 1e3),
    "milyon": (1.0, 1e3, 1e6),
    "milyar": (1e3, 1e6, 1e9),
    "trilyon": (1e6, 1e9, 1e12),
}

_MONTH_ISO = re.compile(
    r"(?<![\d.])((?:19|20)\d{2})[-./](0[1-9]|1[0-2])(?:-\d{2})?(?![\d])"
)
_MONTH_TR = re.compile(r"(?<![\d.,])(0[1-9]|1[0-2])[./]((?:19|20)\d{2})(?![\d])")
_YEAR_RANGE = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}(?!\d)")
_INDEX_BASE = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*=\s*100(?!\d)")
_SERIES_ID = re.compile(r"\b(?:evds|bddk):[A-Za-z0-9_.:]*[A-Za-z0-9_]")
_SERIES_CODE = re.compile(r"(?<![\w:])TP\.[A-Za-z0-9_.]*[A-Za-z0-9_]")
_URL = re.compile(r"https?://[^\s)\]>'\"]+[^\s)\]>'\".,;:]")
_CODE_SPAN = re.compile(r"`[^`]*`")
_IDENTIFIER = re.compile(r"\b[A-Za-z_]+[A-Za-z0-9_.:]*\d[A-Za-z0-9_.:]*\b")

_NUM = (
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"  # Turkish thousands: 1.234.567,89
    r"|\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?"  # English thousands: 1,234,567.89
    r"|\d+,\d+|\d+\.\d+|\d+"
)
_NUMBER = re.compile(rf"(?<![\w.,])({_NUM})(?![\w]|[.,]\d)")
_RANGE = re.compile(
    rf"(?<![\w.,])%?\s*(?P<a>{_NUM})\s*%?\s*"
    rf"(?P<sep>[-–—]|\s+ile\s+)\s*%?\s*(?P<b>{_NUM})(?![\w]|[.,]\d)",
    re.I,
)
_RANGE_WORDS = re.compile(r"arası|aralığ|araliğ|arasi|band", re.I)
_MACHINE_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w]|\.\d)")


@dataclass(frozen=True)
class _Claim:
    """One number, or one range, as written in the answer, with every
    interval of values it could mean."""

    raw: str
    readings: tuple[tuple[float, float], ...]


class AnswerVerifier:
    """The verifier the agent runs with; satisfies `loop.Verifier`.

    `today` is the day the answer is written, read at verify time unless one
    is given (a test needs a fixed one). The planner prompt puts the date in
    front of the model as a *system* message, which `_stated_text` does not
    read, so "bugün 2026-09-23 itibarıyla" was flagged as a month outside the
    tables -- a flag under a sentence the prompt itself asked for.
    """

    def __init__(self, today: date | None = None) -> None:
        self._today = today

    def verify(self, answer: str, state: SessionState) -> VerificationResult:
        stated = _stated_text(state)
        flags: list[VerificationFlag] = []
        flags += _check_sources(answer, state, stated)
        flags += _check_web_cited(answer, state)
        text, months = _without_dates_and_ids(answer)
        flags += _check_months(months, state, stated, self._today or date.today())
        flags += _check_numbers(text, state, stated)

        unique: dict[str, VerificationFlag] = {}
        for flag in flags:
            unique.setdefault(flag.text, flag)
        kept = list(unique.values())[:MAX_FLAGS]
        return VerificationResult(passed=not kept, flags=kept)


# --------------------------------------------------------------------------
# what was said before the answer
# --------------------------------------------------------------------------


def _stated_text(state: SessionState) -> str:
    """Everything the answer may repeat without it being its own claim: every
    user message of the session (the questions, and any note typed at an
    approval card) and every tool message.

    Every turn's, not just this one's. `agent/history.py` keeps earlier
    turns' questions in front of the model and drops their tool messages, so
    restricting this to the current turn flagged the model for repeating a
    number it can still see while waving through numbers it can no longer
    read. Whichever way the asymmetry falls it is wrong; the cheap side is
    the one that does not print a flag under a faithful answer.

    Fenced regions are cut out (`agent/formatting.without_fenced`). A web
    page's text and a search result's snippet arrive inside a tool message,
    and treating them as grounded meant any number a stranger wrote passed
    the answer's number check as if a tool had computed it. Measured 23 Sep:
    an answer reading "dolar kuru 99,99 oldu" passed with no flags against a
    table that held nothing of the sort, because the figure sat in a search
    snippet. `web_search`'s own text and `answer.md` rule 5 both promise that
    cannot happen; this is what makes the promise true rather than advisory.
    """
    parts = [
        without_fenced(str(message.get("content") or ""))
        for message in state.messages
        if message.get("role") in ("tool", "user")
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def _check_sources(
    answer: str, state: SessionState, stated: str
) -> list[VerificationFlag]:
    """Series, codes and addresses the answer names that nothing produced.

    Only `get_series` and `deflate` register a `Source`; `search_catalog`
    registers none, although naming what it found is the honest answer to
    "which series are there?". So a candidate a tool printed counts as
    citable too, exactly as `_check_months` already treats a month a tool
    named -- otherwise listing four search hits earns four flags.
    """
    ids = {source.series_id for source in state.sources if source.series_id}
    codes = {sid.partition(":")[2] for sid in ids}
    urls = {source.url for source in state.sources if source.url}
    urls |= {source.metadata_url for source in state.sources if source.metadata_url}

    ids |= set(_SERIES_ID.findall(stated))
    codes |= set(_SERIES_CODE.findall(stated))
    urls |= set(_URL.findall(stated))

    cited = [c for c in _SERIES_ID.findall(answer) if c not in ids]
    cited += [c for c in _SERIES_CODE.findall(answer) if c not in codes]
    cited += [u for u in _URL.findall(answer) if u not in urls]
    return [
        VerificationFlag(kind="source_not_listed", text=f"'{c}' kaynak listesinde yok.")
        for c in cited
    ]


#: The tools that bring something in from outside the pool. What they read is
#: the one kind of source a reader cannot check for themselves unless the
#: answer says where it came from.
_WEB_TOOLS = frozenset({"read_url", "web_search"})


def _check_web_cited(answer: str, state: SessionState) -> list[VerificationFlag]:
    """Did this turn read the web without saying so?

    `_check_sources` catches the opposite mistake -- a citation nobody
    fetched. This one catches the silent version: the run opened a PDF or
    searched the web, the person watched it happen, and the answer mentions
    neither. Measured 22 Sep: the first break-plus-search run cited nothing
    at all, and the answer read as if the lake had known the reason.

    Only this turn's fetches count, and any of them being named is enough: an
    answer is not obliged to list five search hits.
    """
    used = any(
        step.turn_no == state.turn_no and step.tool in _WEB_TOOLS and step.ok
        for step in state.steps
    )
    if not used:
        return []
    urls = [source.url for source in state.sources if source.url]
    if not urls:
        return []
    if any(_mentions(answer, url) for url in urls):
        return []
    return [
        VerificationFlag(
            kind="source_not_cited",
            text=(
                "Bu turda web'den okundu ama cevapta kaynak adresi anilmadi; "
                "okunan adresi cevaba yaz."
            ),
        )
    ]


def _mentions(answer: str, url: str) -> bool:
    """Is this address in the answer?

    Counted generously, because the flag is for silence, not for style: the
    full URL, the same without its scheme (models write
    `borsaistanbul.com/...`), the file it ends in (`kmp_au.pdf`), or its
    host. The host matters -- a long address is routinely quoted elided,
    `tcmb.gov.tr/.../duy2023-25`, and that is a citation a reader can follow.
    """
    bare = url.split("://", 1)[-1]
    host = bare.split("/", 1)[0].removeprefix("www.")
    tail = bare.rstrip("/").rsplit("/", 1)[-1]
    candidates = [url, bare, host]
    if "." in tail and len(tail) > 4:
        candidates.append(tail)
    return any(candidate and candidate in answer for candidate in candidates)


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------


def _months_in(text: str) -> list[str]:
    months = [f"{year}-{month}" for year, month in _MONTH_ISO.findall(text)]
    months += [f"{year}-{month}" for month, year in _MONTH_TR.findall(text)]
    return months


def _without_dates_and_ids(answer: str) -> tuple[str, list[str]]:
    """The answer with everything that is not a stated value blanked out --
    code spans, ids, URLs, index bases like 2025=100, year ranges, months in
    either order -- and the months it named."""
    months = _months_in(answer)
    text = answer
    for pattern in (
        _CODE_SPAN,
        _URL,
        _SERIES_ID,
        _SERIES_CODE,
        _INDEX_BASE,
        _YEAR_RANGE,
        _MONTH_ISO,
        _MONTH_TR,
    ):
        text = pattern.sub(" ", text)
    return text, months


def _check_months(
    months: list[str], state: SessionState, stated: str, today: date
) -> list[VerificationFlag]:
    covered = _covered_months(state)
    if covered is None:
        return []
    first, last = covered
    said = set(_months_in(stated))
    said.add(today.strftime("%Y-%m"))
    return [
        VerificationFlag(
            kind="date_out_of_range",
            text=f"'{month}' tablolarin kapsadigi donemin disinda ({first} - {last}).",
        )
        for month in dict.fromkeys(months)
        if not first <= month <= last and month not in said
    ]


def _covered_months(state: SessionState) -> tuple[str, str] | None:
    firsts, lasts = [], []
    for frame in state.tables.values():
        if "period" not in frame.columns or frame.empty:
            continue
        periods = pd.to_datetime(frame["period"], errors="coerce").dropna()
        if periods.empty:
            continue
        firsts.append(periods.min().strftime("%Y-%m"))
        lasts.append(periods.max().strftime("%Y-%m"))
    if not firsts:
        return None
    return min(firsts), max(lasts)


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


def _check_numbers(
    text: str, state: SessionState, stated: str
) -> list[VerificationFlag]:
    claims = list(_claims(text))
    if not claims:
        return []
    known = _known_values(state, stated)
    return [
        VerificationFlag(
            kind="number_not_found",
            text=(
                f"'{claim.raw}' tablolarda ve arac ciktilarinda bulunamadi; "
                "arac ciktisindan gelmemis olabilir."
            ),
        )
        for claim in claims
        if not _found(claim, known)
    ]


@dataclass(frozen=True)
class _Context:
    percent: bool
    scale: str | None
    unit: str | None
    approx: bool
    above: bool
    below: bool
    negated: bool


def _context(before: str, after: str) -> _Context:
    tail = _TAIL.match(after)
    scale = tail.group("scale").lower() if tail and tail.group("scale") else None
    unit = tail.group("unit").lower() if tail and tail.group("unit") else None
    words = (tail.group("rest") or "").split() if tail else []
    above = next((w for w in words if _ABOVE_WORDS.match(w)), None)
    below = next((w for w in words if _BELOW_WORDS.match(w)), None)
    approx = next((w for w in words if _APPROX_WORDS.match(w)), None)
    return _Context(
        percent=before.rstrip().endswith("%") or after.lstrip().startswith("%"),
        scale=scale,
        unit=unit,
        approx=bool(_APPROX_BEFORE.search(before)) or bool(approx),
        above=bool(above) and not below,
        below=bool(below),
        negated=any(_NEGATED.search(word) for word in words),
    )


def _claims(text: str) -> Iterator[_Claim]:
    """Every number or range in `text` worth checking, with its readings.

    Skipped: plain whole numbers up to 31 (counts, days, list markers -- "4
    ay", "3 tur") and plain four-digit years, unless a percent sign, a scale
    word or a unit makes them an amount: "5 trilyon TL" and "9 puan" are
    claims. Ranges are found first and checked as one interval, so "165 ile
    205 arasında" is not two made-up numbers.
    """
    spans: list[tuple[int, int]] = []
    for match in _RANGE.finditer(text):
        after = text[match.end() : match.end() + 60]
        if "ile" in match.group("sep").lower() and not _RANGE_WORDS.search(after[:25]):
            continue  # "A ile B" is a range only when it says "arasında"
        spans.append(match.span())
        a_raw, b_raw = match.group("a"), match.group("b")
        context = _context(text[max(0, match.start() - 25) : match.start()], after)
        if context.negated:
            continue
        percent = context.percent or "%" in match.group(0)
        amount = percent or context.scale or context.unit
        if not amount and _plain_small_or_year(a_raw) and _plain_small_or_year(b_raw):
            continue
        yield _Claim(
            raw=f"{a_raw}-{b_raw}",
            readings=tuple(_range_readings(a_raw, b_raw, context)),
        )
    for begin, finish in spans:
        text = text[:begin] + " " * (finish - begin) + text[finish:]

    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        start, end = match.span(1)
        before, after = text[max(0, start - 25) : start], text[end : end + 40]
        context = _context(before, after)
        if context.negated:
            continue
        if (
            not context.percent
            and not context.scale
            and not context.unit
            and _plain_small_or_year(raw)
        ):
            continue
        readings = []
        for value, decimals in _readings(raw):
            low, high = _window(value, decimals, context)
            for factor in _factors(context):
                readings.append((low * factor, high * factor))
        yield _Claim(
            raw=f"%{raw}" if context.percent and before.rstrip().endswith("%") else raw,
            readings=tuple(readings),
        )


def _plain_small_or_year(raw: str) -> bool:
    if not raw.isdigit():
        return False
    value = int(raw)
    return value <= 31 or 1900 <= value <= 2100


def _window(value: float, decimals: int, context: _Context) -> tuple[float, float]:
    """The values a stated number can faithfully stand for.

    Rounded, the true value is within half a unit of the last digit either
    way. Cut short instead, it can be up to a whole unit above: 22 Sep, a live
    answer wrote 1716202.69 as "1.716.202" and was flagged under a correct
    answer. Both are restatements, so both pass; a slip like "27,85" for
    0.2785 is a factor of a hundred away and is still caught.
    """
    unit = 10.0 ** (-decimals)
    low, high = value - 0.5 * unit, value + unit
    if context.approx:
        low = min(low, value - value * _APPROX_SHARE)
        high = max(high, value + value * _APPROX_SHARE)
    if context.above:
        high = value * _BOUND_REACH
    elif context.below:
        low = value / _BOUND_REACH
    return low, high


def _factors(context: _Context) -> tuple[float, ...]:
    factors = _SCALE_FACTORS.get(context.scale or "", (1.0,))
    if context.unit and context.unit.startswith("baz"):
        # "27,85 baz puan" is 0.2785 points: one basis point is 0.01.
        return tuple(f / 100 for f in factors)
    return factors


def _range_readings(
    a_raw: str, b_raw: str, context: _Context
) -> Iterator[tuple[float, float]]:
    for a, a_dec in _readings(a_raw):
        for b, b_dec in _readings(b_raw):
            low, high = min(a, b), max(a, b)
            unit = 10.0 ** (-max(a_dec, b_dec))
            below, above = 0.5 * unit, unit  # rounded or cut short, as `_window`
            if context.approx:
                below = max(below, high * _APPROX_SHARE)
                above = max(above, high * _APPROX_SHARE)
            for factor in _factors(context):
                yield ((low - below) * factor, (high + above) * factor)


def _readings(raw: str) -> list[tuple[float, int]]:
    """(value, decimals shown) for each way `raw` can be read.

    Turkish uses '.' for thousands and ',' for decimals; the model sometimes
    slips into the English style, where the two swap. "1.229" is both 1229
    and 1.229, "678,912" both 678.912 and 678912; either reading can match,
    so a faithful number is not flagged over its separators.
    """
    if re.fullmatch(r"\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?", raw):
        whole, _, fraction = raw.partition(".")
        return [
            (float(whole.replace(",", "") + "." + (fraction or "0")), len(fraction))
        ]
    if "," in raw:
        whole, _, fraction = raw.partition(",")
        readings = [(float(whole.replace(".", "") + "." + fraction), len(fraction))]
        if "." not in whole and len(fraction) == 3:
            readings.append((float(whole + fraction), 0))
        return readings
    if "." not in raw:
        return [(float(raw), 0)]
    groups = raw.split(".")
    readings = []
    if all(len(group) == 3 for group in groups[1:]):
        readings.append((float("".join(groups)), 0))
    if len(groups) == 2:
        readings.append((float(raw), len(groups[1])))
    return readings


def _known_values(state: SessionState, stated: str) -> np.ndarray:
    """Absolute values of everything the answer may quote: every numeric cell
    of every session table, and every number the tools or the user stated.

    Digits inside dates and identifiers in that text are not numbers:
    'TP_KTF12' must not make 12 a known value, nor '2025-09-01' make 9 one.
    """
    chunks: list[np.ndarray] = []
    for frame in state.tables.values():
        numeric = frame.select_dtypes("number")
        if not numeric.empty:
            chunks.append(numeric.to_numpy(dtype=float, na_value=np.nan).ravel())
    cleaned = stated
    for pattern in (_URL, _SERIES_ID, _SERIES_CODE, _IDENTIFIER, _MONTH_ISO, _MONTH_TR):
        cleaned = pattern.sub(" ", cleaned)
    said = [float(token) for token in _MACHINE_NUMBER.findall(cleaned)]
    said += [
        (low + high) / 2 for claim in _claims(cleaned) for low, high in claim.readings
    ]
    chunks.append(np.array(said, dtype=float))
    values = np.abs(np.concatenate(chunks))
    return values[np.isfinite(values)]


def _found(claim: _Claim, known: np.ndarray) -> bool:
    if known.size == 0:
        return False
    return any(
        bool(np.any((known >= low - 1e-9) & (known <= high + 1e-9)))
        for low, high in claim.readings
    )
