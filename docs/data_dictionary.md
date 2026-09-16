# Veri Sözlüğü — EVDS bronze katalog

Bu dosya `src/lakehouse_agent/ingest/evds.py`'nin ürettiği bronze dosyalarını
belgeler. Gold katmanı sözleşmeleri için `CLAUDE.md`'ye bakın (bu dosya
onları tekrar etmez).

Pipeline: `ingest/evds.py` (bronze) → `lake/silver.py` (long, NaN korunur)
→ `lake/gold.py` (observations.parquet + series_catalog.parquet + DuckDB) →
`analysis/transforms.py` (to_freq/deflate/rebase/pct_change/yoy/diff/align_join,
saf DataFrame→DataFrame).

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
`row_count` (yanıt bir liste ise eleman sayısı; bulk veri yanıtı
`{"totalCount":N,"items":[...]}` biçimindeyse `items`'ın eleman sayısı;
ikisi de değilse `null`), `fetched_at` (ISO 8601, UTC), `outcome`
(`ok | redirect | auth_error | retry | failed`).

Anahtar bu dosyalarda **hiçbir zaman** ham biçimde bulunmaz.

## `data/bronze/evds/datagroups/<DATAGROUP_CODE>/<year>.parquet`

`evds.fetch_datagroup_year()` / `fetch_datagroups_years()` tarafından
üretilir — bir veri grubunun bir takvim yılına ait ham (wide) yanıtı, hiç
dönüştürülmeden. Her satır bir `Tarih`; her sütun o gruptaki bir seri
(nokta yerine alt çizgi, örn. `TP_KTF12`). Ek sütunlar da (`YEARWEEK`,
`UNIXTIME`) kaynaktan geldiği gibi tutulur. `_fetched_at` (ISO 8601, UTC)
ingest'in eklediği tek alan — dönüşüm değil, kaynak zaman damgası.

**16 Eyl 2026 doğrulandı — CLAUDE.md'nin metaveri gözlemlerini GEÇERSİZ
KILMAZ, tamamlar:** bulk veri endpoint'i (`datagroup=<code>&startDate=...`)
metaveri endpoint'leri (`categories`, `datagroups`, `serieList`) gibi çıplak
liste DÖNDÜRMÜYOR — `{"totalCount": N, "items": [...]}` zarfı içinde
dönüyor. Resmi kılavuzda belgelenmemiş. `evds._extract_rows()` bu zarfı
açar; `_row_count()` her iki biçimi de `fetch_log`'a doğru sayar.

**Seri henüz başlamamış dönemler:** bulk yanıt, istenen yılın TÜM
dönemlerini döndürür; seri o dönemde henüz yoksa hücre gerçek JSON `null`
(pandas'ta `None`/`NaN`) olarak gelir — `"-"` string DEĞİL (ölçüldü:
`TP.HPBITABLO6.3`, 28-06-2024'ten önceki her hafta `None`). `detect_truncation`
bunu kesme SAYMAZ (satır zaten dönemin tamamını kapsıyor); silver bunu
zaten NaN olarak taşır, ikisi de doldurma yapmaz.

## Senaryo serileri (16 Eylül entegrasyon)

`lake/gold.py`'deki `SCENARIO_SERIES` sabitiyle birebir. `data/gold/` gerçek
EVDS verisiyle üretildi (16 Eyl 2026, `--datagroups bie_kt100h,bie_kfe,
bie_tukfiy2025,bie_akonutsat1,bie_akonutsat2,bie_hpbitablo6 --since 2021-01
--until 2026-06`).

| series_id | datagroup | Frekans | Birim | kind | agg | cum_type | Kapsam |
|---|---|---|---|---|---|---|---|
| evds:TP.KTF12 | bie_kt100h | HAFTALIK(CUMA) | Ağırlıklı ortalama (%) | rate | avg | none | 2002-01 → 2026-09 |
| evds:TP.KFE.TR | bie_kfe | AYLIK | (endeks, birim boş) | index | avg | none | 2010-01 → 2026-07 |
| evds:TP.TUKFIY2025.GENEL | bie_tukfiy2025 | AYLIK | (endeks 2025=100, birim boş) | index | avg | none | 2021-01 → 2026-08 (2005-01'den itibaren dolu) |
| evds:TP.AKONUTSAT1.KTRTOPLAM | bie_akonutsat1 | AYLIK | Adet | flow | sum | none | 2013-01 → 2026-07 |
| evds:TP.AKONUTSAT2.KTRTOPLAM | bie_akonutsat2 | AYLIK | Adet | flow | sum | none | 2013-01 → 2026-07 |
| evds:TP.HPBITABLO6.3 | bie_hpbitablo6 | HAFTALIK(CUMA) | bin TL | stock | last | none | 2024-06 → 2026-09 (bilinçli kısa seri) |

**Faiz serisi seçimi:** TP.KTF12 (akım) seçildi, TP.BKR.TRY.18 (stok) değil.
Akım = o hafta açılan yeni kredilerin faizi; stok = mevcut portföyün ortalaması.
Senaryo yeni kredi davranışını sorguluyor; stok serisi faiz değişimini aylar
sonra yansıtır ve aranan ilişkiyi gizler.

**TÜFE:** zincirleme GEREKMİYOR. TP.TUKFIY2025.GENEL 2021-01'den itibaren dolu
(TÜİK geriye doğru hesaplamış). Eski TP.FG.J0 (2003=100) arşiv, 2026-01'de bitiyor,
kullanılmayacak. Doğrulama: iki serinin oranı sabit (2025-12: 31.832, 2026-01:
31.831) ve aylık değişimleri aynı (%4.84) — yeni seri eskinin yeniden bazlanmış hali.
deflate aracı doğrudan bu seriyi kullanır, baz 2025=100.

**AKONUTSAT1/2 `DEFAULT_AGG_METHOD_STR="KÜMÜLATİF"` tuzağı:** katalogdaki bu
alan "seri değeri yıl başından kümülatif" DEMEK DEĞİL — EVDS'in bu seriyi
daha düşük frekansa (örn. yıllık) çevirirken önerdiği yöntem (aylıkları
topla). Gerçek 2022 aylık değerleri kontrol edildi (93K, 103K, 146K, ...,
226K) — birbirine yakın büyüklükte, yıl içinde tekrar tekrar katlanmıyor;
yani seri zaten aylık akım (flow), BDDK kâr-zarar kalemleri gibi ytd değil.
`kind=flow, agg_method=sum, cum_type=none` bu yüzden doğru.
