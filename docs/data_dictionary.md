# Veri Sözlüğü — EVDS bronze katalog

Bu dosya `src/lakehouse_agent/ingest/evds.py`'nin ürettiği bronze dosyalarını
belgeler. Gold katmanı sözleşmeleri için `CLAUDE.md`'ye bakın (bu dosya
onları tekrar etmez).

## `data/bronze/evds/catalog.parquet`

`build_catalog()` tarafından üretilir. Her satır bir EVDS serisi; `serieList`
+ `datagroups` + `categories` yanıtlarının birleşimidir (DATAGROUP_CODE ve
CATEGORY_ID üzerinden left-join). Ham EVDS alan adları korunur (dönüştürme
yok) — sütun adı çakışmalarında datagroup sütunları `_dg`, kategori sütunları
`_cat` soneki alır.

Kaynak: `serieList` yanıtı (`SERIE_CODE, DATAGROUP_CODE, SERIE_NAME(_ENG),
FREQUENCY_STR, DEFAULT_AGG_METHOD(_STR), TAG, DATASOURCE, METADATA_LINK,
START_DATE, END_DATE, AVGABLE/FIRSTABLE/LASTABLE/MAXABLE/MINABLE/SUMABLE,
SEVIYE, UST_SERIE_CODE`), `datagroups` yanıtı (`DATAGROUP_CODE, CATEGORY_ID,
DATAGROUP_NAME(_ENG), FREQUENCY, FREQUENCY_STR, DATASOURCE, METADATA_LINK,
START_DATE, END_DATE, NOTE, BIRIMI, LAST_UPDATED`), `categories` yanıtı
(`CATEGORY_ID, TOPIC_TITLE_TR/ENG, SEVIYE, UST_CATEGORY_ID`).

Not: `SERIE_CODE` noktalı (`TP.ISTIRAKBS.A1`); bu, EVDS veri yanıtındaki
alt çizgili sütun adlarının (`TP_ISTIRAKBS_A1`) katalogla eşleştiği alandır
— `evds.denormalize_response_columns()` bu çevirimi tek bir yerde yapar.

## Ara bronze dosyaları

- `data/bronze/evds/categories.parquet`, `datagroups.parquet` — ham
  `categories`/`datagroups` yanıt anlık görüntüleri; `build_catalog()`
  tarafından, varsa tekrar çekilmeden okunur (idempotency).
- `data/bronze/evds/_catalog_cache/<DATAGROUP_CODE>.parquet` — her veri
  grubunun ham `serieList` yanıtı; ~680 çağrının ortasında kesinti olursa
  tamamlanmışlar yeniden çekilmez.

## `data/bronze/evds/fetch_log.jsonl` / `fetch_log.parquet`

Her EVDS HTTP çağrısı (başarılı/başarısız, her retry denemesi dahil) satır
satır `fetch_log.jsonl`'a append edilir (parquet'in append desteklemediği,
binlerce çağrıda O(n²) rewrite'a yol açacağı için). `compact_fetch_log()`
bunu tek seferde `fetch_log.parquet`'e derler.

Sütunlar: `url` (anahtar `key=***` olarak redakte edilmiş), `params`
(çağrılan path, örn. `datagroups/mode=0&type=json`), `http_status`,
`row_count` (yanıt bir liste ise eleman sayısı, değilse `null`),
`fetched_at` (ISO 8601, UTC), `outcome` (`ok | redirect | auth_error |
retry | failed`).

Anahtar bu dosyalarda **hiçbir zaman** ham biçimde bulunmaz.
