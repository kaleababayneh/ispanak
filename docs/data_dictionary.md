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

## Senaryo doğrulaması (16 Eylül, gerçek veriyle)

Faiz (TP.KTF12, aylık ortalama) düşerken konut kredisi bakiyesi
(TP.HPBITABLO6.3) davranışı, 2024-06 – 2026-08 arası 27 ay:

- NOMİNAL bakiyede daralma: **0 ay**
- REEL bakiyede (TÜFE ile deflate, baz 2026-08) daralma: **9 ay**
  2024-07, 2024-08, 2024-09, 2025-01, 2025-02, 2025-07, 2025-09,
  2026-01, 2026-02

Nominal bakışta hiçbir örüntü görünmüyor çünkü bakiye enflasyonla zaten
artıyor. deflate aracının gerekçesi budur ve demoda bu karşılaştırma
gösterilecek. En belirgin daralma 2026-01 (-19.5M reel).

## Analiz araçları — gerçek veriyle göz kontrolü (16 Eylül)

detect_changes, TP.KTF12 (konut kredisi faizi) serisinde kırılmayı
**2023-07** olarak buldu: ortalama %18.7 → %40.4, eğim +0.089 → -0.080.
TCMB'nin Haziran 2023'te sıkılaşmaya dönüşüyle örtüşüyor.

detect_anomalies aynı seride 2023-03..06 aylarını "beklenenin altında"
işaretledi — sıkılaşma öncesi baskılanmış faiz dönemi.

TP.KFE.TR: kırılma 2023-03, anomali 2023-09/10 (enflasyonun endekse yansıması).

TP.AKONUTSAT1.KTRTOPLAM: anomali ve kırılma YOK. Seri güçlü mevsimsel
(Aralık zirve, Ocak dip) ama trend kırılması içermiyor; STL mevsimselliği
ayıklamış, yanlış alarm üretmemiş.

**Düzeltildi (18 Eylül):** `yüzde_değişim` tek sütun kırılma anındaki
sıçramayı değil, iki dönem ortalaması arasındaki farkı veriyordu (KFE'de
%230.3 — jüri "endeks bir ayda %230 mü arttı?" diye sorardı). İkiye
ayrıldı: `seviye_degisim_yuzde` (eski hesap, ortalamalar arası fark) ve
`sicrama_yuzde` (kırılma tarihindeki değer ile bir önceki dönemin değeri
arasındaki gerçek tek dönem sıçraması). Özet metni ikisini ayrı ayrı adlandırıyor.

## Katalog arama indeksi (`lake/catalog.py`, plan 2.5)

`data/bronze/evds/catalog.parquet` 53792 satır — agent'ın arayacağı ham
kapsam bu kadar büyük olunca embedding maliyeti kabul edilemez hale geliyor
(tek worker'la ~242 dk, ölçüldü). `filter_catalog()` bu yüzden önce kapsamı
daraltıyor: arşiv serileri, 2025-06'dan önce bitmiş (ölü) seriler, senaryo
dışı frekanslar, IMF ülke serileri (`bie_imfcpind`, `bie_imfgdpusdn`) ve
"konutsat" gruplarının il kırılımları (yalnız TOPLAM kalıyor) çıkarılıyor.
Sonuç: **53792 → 19646 seri**.

`embed_with_cache()` metni sha256 hash'iyle `data/catalog_emb_cache/`
altında diske önbelleğe alıyor — aynı metin iki kez embed edilmiyor,
8 worker paralel çalışıyor. Vektör hiç yoksa veya embedding tamamen
başarısız olursa `search_catalog(use_vector=False)` yalnızca LanceDB
full-text index'i (FTS) ile çalışıyor; eş anlamlı yakalama zayıflıyor
ama arama tamamen durmuyor.

## Senaryo doğrulaması — tam BDDK serisiyle (17 Eylül)

Faiz (TP.KTF12, aylık ort.) düşerken konut kredisi bakiyesi
(bddk:aylik:tuketici:konut:TL:sektor) davranışı, 2021-01 – 2026-06, 66 ay:

- NOMİNAL bakiyede daralma: **4 ay**
- REEL bakiyede (TÜFE ile deflate) daralma: **23 ay**

Gözlemlerin üçte birinden fazlası nominal veriyle görünmez. deflate aracının
gerekçesi budur; demoda bu karşılaştırma gösterilecek.
En belirgin daralmalar: 2024-01 (-64k reel), 2023-12 (-34k), 2026-01 (-23k).

Not: 16 Eylül'de aynı analiz kısa EVDS serisiyle (TP.HPBITABLO6.3, 2024-06'dan)
27 ayda 9 sonucu vermişti. BDDK serisi aralığı 2021'e uzatınca örüntü güçlendi.

## Çapraz kaynak kontrolü (plan 5.4) — 19 Eylül

Konut kredisi bakiyesi iki bağımsız kaynaktan hesaplandı ve karşılaştırıldı:

| Kaynak | series_id | Frekans | Birim |
|---|---|---|---|
| BDDK Aylık Bülten, tablo 4 | bddk:aylik:tuketici:konut:TL:sektor | AYLIK | milyon TL |
| TCMB EVDS, haftalık kredi büyüklükleri | evds:TP.HPBITABLO6.3 | HAFTALIK(CUMA) | bin TL |

EVDS serisi ay sonuna alınıp (last) 1000'e bölünerek milyon TL'ye çevrildi.
Örtüşen 25 ay (2024-06 – 2026-06) için sapma:

- Ortalama mutlak fark: **%0.1**
- Maksimum fark: **%0.6** (2026-03)
- 25 ayın 19'unda fark %0.2'nin altında

Sapma tanım farkından kaynaklanıyor: BDDK ayın son iş gününü raporluyor,
EVDS haftalık serisinin ay sonuna en yakın Cuma değeri alınıyor. Ay sonunun
hafta ortasına denk geldiği aylarda fark büyüyor. Veri hatası değil.

Sonuç: iki kaynak birbirini doğruluyor. Senaryoda BDDK serisi kullanılıyor
(2021-01'den itibaren tam kapsam; EVDS serisi 2024-06'da başlıyor).
