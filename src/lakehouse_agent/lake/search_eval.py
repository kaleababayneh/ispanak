"""Measure the pool search against eval/search_queries.yaml (plan 2.5).

Real embeddings and the real pool index: this is a measurement, not a unit
test, and it calls MIA once per query. What it reports:

* for each labelled query, the rank and cosine score of the first expected
  series among the candidates (the plan's bar: in the top 5 for 90% of the
  queries), and the same for the held-out ones (`ayrik`) alone -- queries
  the ranking rules were not designed from;
* for each unrelated query, the best score any pool series reaches -- the
  floor (`MIN_SIMILARITY`) has to sit above these to keep "we do not have
  that" honest, and below the expected hits' scores to keep them.

Scores are read with the floor switched off, so the numbers show where a
floor could go rather than what the current one lets through.

    uv run python -m lakehouse_agent.lake.search_eval
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

from lakehouse_agent.tools import lake_adapter

QUERIES_PATH = Path("eval/search_queries.yaml")
RESULTS_ROOT = Path("eval/results")
CANDIDATES = 20

logger = logging.getLogger(__name__)


def _expected(entry: dict, ids: list[str]) -> set[str]:
    if "beklenen" in entry:
        return set(entry["beklenen"])
    pattern = re.compile(entry["beklenen_desen"])
    return {series_id for series_id in ids if pattern.search(series_id)}


def run(adapter: lake_adapter.LakeAdapter | None = None) -> dict:
    spec = yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))
    adapter = adapter or lake_adapter.LakeAdapter()
    all_ids = list(adapter._series_catalog()["series_id"])
    floor = lake_adapter.MIN_SIMILARITY
    lake_adapter.MIN_SIMILARITY = -1.0  # see every score
    try:
        labelled = []
        for entry in spec["sorgular"]:
            expected = _expected(entry, all_ids)
            missing = sorted(expected - set(all_ids))
            hits, notes = adapter.search_detailed(entry["soru"], CANDIDATES)
            ranked = [(hit.series_id, hit.score) for hit in hits]
            found = next(
                (
                    (rank, score)
                    for rank, (sid, score) in enumerate(ranked, start=1)
                    if sid in expected
                ),
                None,
            )
            labelled.append(
                {
                    "soru": entry["soru"],
                    "ayrik": bool(entry.get("ayrik")),
                    "sira": found[0] if found else None,
                    "skor": round(found[1], 4) if found else None,
                    "ilk": [(sid, round(score, 4)) for sid, score in ranked[:5]],
                    "havuzda_olmayan_beklenen": missing,
                    "notlar": notes,
                }
            )
        unrelated = []
        for question in spec.get("ilgisiz", []):
            hits, notes = adapter.search_detailed(question, 1)
            unrelated.append(
                {
                    "soru": question,
                    "en_yuksek": round(hits[0].score, 4) if hits else None,
                    "ilk": hits[0].series_id if hits else None,
                }
            )
    finally:
        lake_adapter.MIN_SIMILARITY = floor

    def top5(rows: list[dict]) -> str:
        hits = sum(1 for row in rows if row["sira"] is not None and row["sira"] <= 5)
        return f"{hits}/{len(rows)}"

    return {
        "zaman": datetime.now(UTC).isoformat(),
        "havuz": len(all_ids),
        "esik": floor,
        "ilk5_isabet": top5(labelled),
        "ilk5_ayrik": top5([row for row in labelled if row["ayrik"]]),
        "sorgular": labelled,
        "ilgisiz": unrelated,
    }


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    report = run()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESULTS_ROOT / f"search_eval_{report['zaman'][:10]}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"ilk 5 isabet: {report['ilk5_isabet']}, ayrik sorgularda "
        f"{report['ilk5_ayrik']}  (havuz {report['havuz']} seri)"
    )
    for row in report["sorgular"]:
        mark = "*" if row["ayrik"] else " "
        print(f" {mark}{row['sira']!s:>4}  {row['skor']!s:>7}  {row['soru']}")
    for row in report["ilgisiz"]:
        print(f"  ilgisiz {row['en_yuksek']!s:>7}  {row['soru']}  -> {row['ilk']}")
    print(f"yazildi: {out}")
