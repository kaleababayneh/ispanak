"""CLI entrypoint for ingest.

The one place `.env` is loaded into the process environment -- `config.py`
never reads it directly (see its docstring); this module loads it, then
`Settings()` is constructed from whatever ends up in `os.environ`.

Usage:
    uv run python -m lakehouse_agent.ingest.run --source evds \
        --datagroups bie_kt100h,bie_kfe --since 2021-01 --until 2026-06
    uv run python -m lakehouse_agent.ingest.run --source evds --all \
        --since 2021-01 --until 2026-06
"""

import argparse
import logging
import sys

import pandas as pd
from dotenv import load_dotenv

from lakehouse_agent.config import get_settings
from lakehouse_agent.ingest import evds

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _year_range(since: str, until: str) -> list[int]:
    since_year = int(since.split("-")[0])
    until_year = int(until.split("-")[0])
    return list(range(since_year, until_year + 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lakehouse_agent.ingest.run")
    parser.add_argument("--source", required=True, choices=["evds"])
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--datagroups", help="virgüllü DATAGROUP_CODE listesi")
    scope.add_argument("--all", action="store_true", help="tam katalog + tam çekim")
    parser.add_argument("--since", required=True, help="YYYY-MM")
    parser.add_argument("--until", required=True, help="YYYY-MM")
    parser.add_argument(
        "--force", action="store_true", help="var olan bronze dosyalarını yeniden çek"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    settings = get_settings()
    years = _year_range(args.since, args.until)

    if args.all:
        evds.build_catalog(settings)
        catalog = pd.read_parquet(evds.CATALOG_PATH)
        codes = sorted(catalog["DATAGROUP_CODE"].dropna().unique())
    else:
        codes = [c.strip() for c in args.datagroups.split(",") if c.strip()]

    logger.info(
        "EVDS toplu çekim başlıyor: %d veri grubu x %d yıl", len(codes), len(years)
    )
    results = evds.fetch_datagroups_years(codes, years, settings, force=args.force)

    ok = [r for r in results if r.outcome == "ok"]
    cached = [r for r in results if r.outcome == "cached"]
    empty = [r for r in results if r.outcome == "empty"]
    failed = [r for r in results if r.outcome in ("failed", "truncated")]

    logger.info(
        "Bitti: ok=%d cached=%d empty=%d failed/truncated=%d",
        len(ok),
        len(cached),
        len(empty),
        len(failed),
    )
    if failed:
        logger.warning("Başarısız/kesik gruplar:")
        for r in failed:
            logger.warning("  %s/%s -> %s", r.datagroup_code, r.year, r.outcome)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
