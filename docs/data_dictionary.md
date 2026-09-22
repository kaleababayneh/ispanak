# Veri Sözlüğü

Bu dosya bronze dosyalarını (EVDS, BDDK Aylık, BDDK Haftalık, BDDK FinTürk)
ve havuzun kurallarını belgeler. Gold katmanı sözleşmeleri için `CLAUDE.md` ve
`docs/contracts.md` §1'e bakın (bu dosya onları tekrar etmez). Havuzun
tamamı: aşağıda "Havuz (adım 2)".

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
dönüştürülmeden.

Yıl zaman aşımına uğrarsa (büyük günlük gruplar, örn. `bie_pydibs`) aynı
klasöre daha küçük pencereler yazılır: `<year>-Q<n>.parquet` (çeyrek), o da
aşılırsa `<year>-<MM>.parquet` (ay). Silver bir grubun bütün parçalarını
okur. Veri döndürmeyen pencere `<pencere>.empty.json` ile işaretlenir
(istek, durum, zaman); ikinci koşu ona istek atmaz. Her satır bir `Tarih`; her sütun o gruptaki bir seri
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
EVDS ve BDDK verisiyle üretildi ve repoya eklendi (22 Eyl 2026, `--datagroups
bie_kt100h,bie_kfe,bie_tukfiy2025,bie_akonutsat1,bie_akonutsat2,bie_hpbitablo6
--since 2021-01 --until 2026-06`). EVDS çekimi takvim yılını bütün olarak aldığı
için `build_gold` gözlemleri `GOLD_UNTIL = "2026-06"` sonrasında keser (bronze
dokunulmadan kalır); haftalık serilerde son hafta 2026-06-29 Pazartesi haftasıdır.

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

Tarihsel kayıt: bu ölçüm gold 2026-06'da kesilmeden önce yapıldı (baz 2026-08,
aylar 2026-08'e kadar). Bugünkü gold ile aynen tekrarlanamaz; `deflate` 2026-08
baz ayını havuzda olmadığı için reddeder. Güncel sonuç aşağıdaki 17 Eylül
bölümünde ve `tests/test_scenario.py`'de.

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

> **Agent bu indeksi kullanmıyor.** Bu indeks ham EVDS kataloğu üzerinde ve
> yalnızca havuz **dışı** keşif için duruyor (`build_index.py`,
> `data/catalog.lance`). Agent'ın araması 22 Eyl'den beri havuzun kendi
> kataloğunda: `data/pool_index`, aşağıda "Havuz arama indeksi".
> `embed_with_cache` ikisinde ortak.

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
EVDS haftalık serisinde ise Pazartesi'si o ayda olan son haftanın (Cuma) değeri
alınıyor. Gold haftayı Pazartesi'sine yazdığı için ayın ilk günlerine düşen bir
Cuma önceki aya sayılabiliyor (2025'te 12 ayın 6'sında "ay sonu" değeri
sonraki ayın ilk Cuma'sı). Veri hatası değil.

**22 Eyl yeniden ölçüm (`obs_date` ile, adım 2):** hafta artık gözlem gününe
göre aya düşüyor; ayın değeri ayın içindeki son Cuma. Aynı 25 ay:

| Yöntem | Ort. mutlak fark | En büyük | %0,2 altı | Cuma'nın ay sonuna ort. uzaklığı |
|---|---|---|---|---|
| Pazartesi'ye göre (eski) | %0,14 | %0,59 (2026-03) | 20 ay | 1,9 gün (bazen sonraki ay) |
| `obs_date`'e göre (yeni) | %0,20 | %0,70 (2026-04) | 15 ay | 2,8 gün (hep ay içinde) |

Beklenenin tersine fark biraz büyüdü. Neden: bakiye zamanla büyüyor ve fark
seçilen Cuma'nın ay sonuna uzaklığını izliyor. Ayın içindeki son Cuma ay
sonundan ortalama 2,8 gün önce (yeni farkların hepsi eksi yönde); eski yöntem
bazen sonraki ayın 1–4 gününe ait Cuma'yı alıyordu ve ay sonuna ortalamada
daha yakındı. `obs_date`'in kazancı ay sonu değerine yakınlık değil,
doğruluk: bir ayın değeri artık o aydan sonra gözlenmiş bir değer olamıyor.
Hiçbir kural haftalık seriden BDDK'nın ay sonu değerini çıkaramaz; iki kaynak
yine birbirini doğruluyor (fark her ay %1'in altında).

Sonuç: iki kaynak birbirini doğruluyor. Senaryoda BDDK serisi kullanılıyor
(2021-01'den itibaren tam kapsam; EVDS serisi 2024-06'da başlıyor).

## Havuz (adım 2, 22 Eylül)

`python -m lakehouse_agent.lake.gold` bütün havuzu kurar: EVDS'nin sözleşme
frekansındaki her serisi ve BDDK'nın üç bülteninin her serisi. Pencere
`2021-01-01 <= obs_date <= 2026-06-30`; penceresinde hiç değeri olmayan seri
gold'a (ve kataloğa) girmez. Aynı komut `data/gold/classification.parquet`
(EVDS serilerinin tür gerekçesi) ve `docs/kind_review.md`'yi de yazar.
Kararların gerekçesi: `docs/decisions.md`, "Havuz genişletme (adım 2)".

### Havuz ölçümü (22 Eyl 2026)

| | Seri | Gözlem satırı |
|---|---|---|
| EVDS | 36.642 (A 8.458, D 5.496, M 10.693, Q 5.838, W 6.157) | 11.655.240 |
| BDDK Aylık | 2.045 | 133.485 |
| BDDK Haftalık | 535 | 147.154 |
| BDDK FinTürk | 6.200 | 136.400 |
| **Toplam** | **45.422** | **12.072.279** |

- `observations.parquet` 33,0 MB (zstd), `series_catalog.parquet` 1,3 MB,
  `classification.parquet` 0,3 MB, `quality_by_series.parquet` 0,9 MB.
  `catalog.duckdb` (~120 MB) ve `data/pool_index` (~0,5 GB) türetilir,
  gitignore'da. Gold kurulumu ~65 sn.
- Türler: stock 28.032, rate 8.048, flow 4.972, index 1.663, boş 2.707
  (`docs/kind_review.md`).
- Satırların 5.571.837'sinin değeri boş (doldurulmaz). %86'sı tek grup:
  `bie_pydibs` (Devlet İç Borçlanma Senetlerinin günlük gösterge değerleri,
  tahvil başına bir seri). Kaynak her tahvile pencerenin bütün günleri için
  hücre veriyor; ihraç öncesi ve vade sonrası günler boş. Bu grup gold'un
  8,2 milyon satırı (%68). Kalite raporu eksikliği serinin ilk ve son dolu
  günü arasında saydığı için bu boşlukları eksik saymaz.
- Silver'da birleştirilen tekrar satır 13.668: hepsi yıl sınırındaki hafta
  (iki yıl isteğinde de geliyor), gözlem günü ve değer aynı. Farklı günlü ya da
  farklı değerli tekrar 0.
- EVDS çekimi (`--all`, 2021–2026 yılları, 74 dk, bronze 254 MB): 2.674
  pencere geldi, 1.352 pencere boş (seri o yıl yayımlanmamış), 1000 satır
  sınırına takılan 0. Zaman aşımına uğrayan büyük günlük gruplar (`bie_pydibs`,
  `bie_pydibsarsiv`, `bie_kt210a` vb.) çeyrek ya da aya bölünerek geldi.
  **Eksik:** `bie_sekbily01a` ("Devir Hızları - Borç Devir Hızı", yıllık, 91
  seri): her yıl için 400 dönüyor, iki denemede de; katalogda başlangıç ve
  bitiş tarihi de boş. Havuzda yok.
- BDDK: Aylık 17 tablo × 66 ay, Haftalık 9 tablo × 286 hafta, FinTürk 7
  tablo × 22 çeyrek; hepsi tam, düşen satır 0.

### EVDS

Bronze kataloğu 53.985 seri, 678 veri grubu. Frekanslar:

| FREQUENCY_STR | Seri | Gold `freq` |
|---|---|---|
| AYLIK | 16.557 | M |
| GÜNLÜK | 11.446 | D |
| YILLIK | 9.540 | A |
| HAFTALIK(CUMA) | 8.108 | W |
| ÜÇ AYLIK | 7.409 | Q |
| İŞ GÜNÜ | 631 | D |
| HAFTALIK(ÇARŞAMBA) | 5 | W |
| AYDA İKİ KEZ | 224 | **dışarıda** |
| ALTI AYLIK | 65 | **dışarıda** |

Ayda iki kez ve altı aylık seriler sözleşmenin `freq` kümesinde karşılığı
olmadığı için dışarıda (`gold.EXCLUDED_FREQUENCY_STRS`).

Her seri için (`lake/classify.py`, kaynağın söylediğinden, tahmin yok):

- `unit`: adın sonu "(%)" ise o (veri grubunun birimi karışık gruplarda
  yüzdelere de düzey birimini veriyor: "İşsizlik oranı (%)" → "Bin kişi",
  23 seri). Değilse veri grubunun `BIRIMI`. O da boşsa adın sonundaki
  parantez, yalnızca açık bir birimse ("(TL)", "(2010=100)"). Hiçbiri yoksa
  boş. Çevrilmez.
- `agg_method`: EVDS'nin `DEFAULT_AGG_METHOD`'u olduğu gibi (last, avg,
  sum, min, max). "KÜMÜLATİF" = sum: EVDS'nin seyrek frekansa çevirirken
  toplamayı önermesi, verinin yıl başından kümülatif olması değil (bkz.
  "AKONUTSAT1/2 tuzağı" yukarıda). EVDS serilerinde `cum_type` hep `none`.
- `kind`, ilk tutan kural: birim endeks → index; birim yüzde/oran/"Ağırlıklı
  ortalama" → rate; EVDS topluyor → flow; ad endeks → index; ad yüzde/"faiz
  oranı" → rate; EVDS dönem sonunu alıyor → stock; hiçbiri → boş. Boş kalan
  seriler ve gerekçeleri `docs/kind_review.md`'de.
- Senaryonun altı EVDS serisi `gold.SCENARIO_SERIES` ile elle eşli; kuralların
  bu seriler için aynı sonucu verdiği test ediliyor (`tests/test_classify.py`).

### BDDK Aylık Bülten

Bronze: `data/bronze/bddk_aylik/tables/tNN/YYYY-MM.json`, tablo × ay başına
bir dosya: `{url, params, http_status, fetched_at, payload}`, payload
yanıtın kendisi. Uç nokta `POST /BultenAylik/tr/Home/BasitRaporGetir`,
`tabloNo` 1–17, sektör (taraf 10001), TL raporu.

Tablolar ve anlamları: `ingest/bddk_tables.yaml`. Her satır ve her değer
sütunu (`colNames[4:]`, örn. TP / YP / Toplam) bir seri:

    bddk:aylik:<tablo anahtarı>:<formülsüz etiket>:<sütun>:sektor
    bddk:aylik:bilanco:nakit_degerler:toplam:sektor

- Etiketin formülü atılır ("Tüketici Kredileri (2+3+4)" → "Tüketici
  Kredileri"): BDDK satırları yeniden numaralandığında formül değişiyor, kalem
  aynı kalıyor.
- Liste maddesi ("a) Konut") ve tabloda tekrar eden etiket üst satırıyla
  nitelenir: onu toplayan formül satırı, yoksa listenin asıldığı satır, yoksa
  en yakın kalın satır. `name_tr` = nitelenmiş etiket + " (sütun)".
- Senaryonun üç kimliği (`bddk:aylik:tuketici:konut:TL:sektor` vb.)
  yaml'daki `aliases` ile korunur.
- Birim: tablo başlığından ("Bilanço (milyon TL)"); satır ya da sütun kendi
  birimini söylüyorsa o ("(%)", "(YÜZDE)", "(Bin TL)", "(Gün)"; "... Sayısı"
  → Adet).
- `kind`/`agg_method`: tablo tanımından; yüzde birimli değer her tabloda
  rate/last. Rasyo tablolarında yüzde olmayan değerlerin (çalışan başına
  aktif "Bin TL", ortalama vade "Gün") türü boş.

#### BDDK kümülatif kalemler

Kâr-zarar tablosunun (tablo 2) değerleri yıl başından kümülatif. Ölçüm (22 Eyl,
bronze, 2021-12 → 2026-01 arası beş yıl geçişi, "Toplam" sütunları):

| Tablo | Aralık→Ocak geçişi | Ocak < Aralık'ın yarısı | Medyan Ocak/Aralık |
|---|---|---|---|
| 2 Kâr-zarar | 248 | %98 | 0,11 |
| 1 Bilanço | 301 | %3 | 1,02 |

Örnek, "Kredilerden Alınan Faizler (Toplam)", milyon TL: 2021-10 349.063,
2021-11 397.131, 2021-12 443.032, **2022-01 47.522**, 2022-02 92.998,
2022-03 147.642.

Bu yüzden `cum_type: ytd` ve silver'da aylık akıma çevrilir
(`transforms.decumulate_ytd`: akım_t = küm_t − küm_t−1, Ocak'ta akım = küm;
önceki ay yoksa boş). Gold'daki değer akımdır. Yıl başından toplam
`transform_table` `ytd` işlemiyle geri kurulur.

### BDDK Haftalık Bülten

Bronze: `data/bronze/bddk_haftalik/tNNN/YYYY-MM-DD.json`, tablo × hafta,
tarih bültenin günü. Kayıt: istek, hafta, tablo, durum, zaman ve
`<table id="TabloExcel2">` öğesi olduğu gibi. Hafta ve tablo listesi
`listing.json`'da. Tablolar 289–297: Krediler, Takipteki Alacaklar, Menkul
Değerler, Mevduat, Diğer Bilanço Kalemleri, Bilanço Dışı İşlemler, Bankalarda
Saklanan Menkul Değerler - 1 ve - 2, Yabancı Para Pozisyonu.

    bddk:haftalik:<tablo adı>:<formülsüz etiket>:<sütun>:sektor
    bddk:haftalik:krediler:tuketici_kredileri:toplam:sektor

- `period` haftanın Pazartesi'si, `obs_date` tablo başlığındaki gün (Cuma;
  tatil haftasında Perşembe ya da başka gün: 2021-01 – 2026-06'da 139.958
  değer Cuma, 4.629 Perşembe, 2.567 diğer).
- Birim başlıktan ("Milyon TL"); `kind` stock, yüzde birimli satırda rate;
  `agg_method` last.
- 286 hafta × 9 tablo, eksik hafta yok. 535 serinin 490'ı her hafta dolu.
  Kalan 45 seri BDDK'nın kendi tanım değişiklikleri: Menkul değerler
  tablosunda "a) Kamu borçlanma senetleri / b) Diğer" kırılımı 2022-09-09'da
  bitiyor (88 hafta), yerine "Devlet tahvilleri / Kira sertifikaları (bilgi
  için)" 2022-09-16'da başlıyor (198 hafta); kur korumalı mevduat
  2022-02-18'de, kredili mevduat hesabı kırılımı 2022-09-23'te başlıyor. Ana
  toplamlar 286 haftanın hepsinde dolu. Eski ve yeni kırılım ayrı seri:
  tanımları farklı.

### Havuz arama indeksi (`lake/pool_index.py`)

`python -m lakehouse_agent.lake.pool_index` gold kataloğunun her serisi için bir
arama metni ve vektörü yazar: `data/pool_index` (LanceDB, 719 MB, gitignore'da).
45.422 metin 3,5 dakikada embed edildi (Qwen3-Embedding-8B, 4096 boyut, 8
worker, hata yok); metinler `data/catalog_emb_cache`'te hash'le önbellekte, aynı
metin ikinci kez embed edilmiyor. Arama yalnızca sorguyu embed eder: uçtan uca
0,25 sn (yakın komşu 0,05 sn).

Arama metni: ad + (EVDS'te) veri grubu adı + konu + etiketler + birim + frekans +
tür + kaynak. `ids.parquet` sıralama için seri başına şunları saklar: veri
grubu, EVDS'in ilk listelediği seri mi (ana seri), hangi ili anlatıyor, arşiv mi,
hangi kırılım ailesinden. Sıralama kuralları ve ölçüm: `docs/decisions.md`,
"Havuz araması: sıralama kuralları". Ölçüm dosyası:
`eval/results/search_eval_2026-09-22.json` (ilk 5 isabet 25/27, ayrık sorgularda
6/6).

### BDDK FinTürk (illere göre)

Bronze: `data/bronze/bddk_finturk/tNN/YYYY-M.json` (M = 3/6/9/12), tablo ×
çeyrek, tek çağrıda bütün iller. Uç nokta `POST
/BultenFinturk/tr/Home/VeriGetir`. 7 tablo: Krediler, Mevduat, Bireysel
Bankacılık, Seçilmiş Sektörel Krediler, Oranlar, Şubeler, Altın.

    bddk:finturk:<tablo anahtarı>:<il>:<sütun>:sektor
    bddk:finturk:krediler:istanbul:nakdi_krediler:sektor

- 82 il ("YURT DIŞI" dahil). `freq` Q; `period` çeyreğin ilk günü,
  `obs_date = period` (kaynak gün vermiyor).
- Sütun adında TP/YP ayrımı yok, sütun kendisi kalem ("Nakdi Krediler",
  "Tasarruf Mevduatı DTH"). 6.210 seri.
- Birim tablo adından ("Krediler (Bin TL)"); "Oranlar (%)" rate. Şubeler
  tablosunda şube sayısı Adet/stock; "Şubeye Düşen Nüfus" (Kişi) ve kişi başı
  sütunlarının (TL) türü boş: 405 seri.
