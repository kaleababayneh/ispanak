"""LanceDB katalog arama indeksini kurar (plan 2.5).

Kullanim:
    uv run python build_index.py

Embedding cache'i data/catalog_emb_cache/ altinda tutulur; ikinci calistirmada
yalnizca yeni metinler embed edilir.
"""

import logging

import pandas as pd
from dotenv import load_dotenv

from lakehouse_agent.lake import catalog


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_dotenv(".env")
    df = pd.read_parquet("data/bronze/evds/catalog.parquet")
    filtered = catalog.filter_catalog(df)
    print("filtre sonrasi:", len(filtered))
    catalog.build_catalog_index(filtered, "data/catalog.lance")
    print("indeks hazir")


if __name__ == "__main__":
    main()
