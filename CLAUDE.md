# CLAUDE.md — lakehouse-agent

Bu dosya Claude Code'un her oturumda okuduğu proje hafızasıdır. Kısa tut (<200 satır).
Uzun açıklamalar `docs/` altına, buraya sadece **kural** yaz.

## Proje

KKB Hackathon 2026. EVDS + BDDK verisini (2021-01 – 2026-06) bronze→silver→gold
katmanlarına indiren, DuckDB üzerinden sorgulanabilir hale getiren ve bir
kendi agent döngüsüne (LangGraph değil, `docs/decisions.md`) araç olarak sunan
Python projesi. Teslim tarihi 20 Eylül'den ertelendi.

Agent'ın **arama** aracı havuzun kendi kataloğunda (gold `series_catalog`)
çalışıyor, LanceDB indeksinde değil (22 Eyl, `docs/decisions.md`). LanceDB
indeksi (`build_index.py`, `data/catalog.lance`) havuz dışı EVDS keşfi için
duruyor; agent yolunda kullanılmıyor.

## Geliştirici ve sahiplik (ÖNEMLİ)

22 Eylül'den beri projenin **tek geliştiricisi Derya**; eski A (veri) ve B (agent)
şeritlerinin ikisi de onun. Repo'nun her yoluna yazılabilir; `docs/contracts.md`
§9'daki sahiplik tablosu tarihsel. Aşağıdaki sözleşmeler yine bağlayıcı: bir
şema değişikliği önce `docs/contracts.md`'ye yazılır ve kullanıcıya sorulur.

## Sözleşmeler (değiştirilemez — agent ve araçlar bunlara göre yazıldı)

`data/gold/observations.parquet` sütunları, tam olarak:
`series_id, period, freq, value, unit, source, fetched_at`

`data/gold/series_catalog.parquet` sütunları:
`series_id, name_tr, name_en, source, freq, unit, kind, agg_method, cum_type,
start_date, end_date, topic, tags, metadata_url, notes`

- `series_id` önekli: `evds:TP.KTF12`, `bddk:aylik:tuketici:konut:TL:sektor`
- `period` = dönemin İLK günü (aylık → ayın 1'i, haftalık → Pazartesi). DATE, saat yok.
- `kind` ∈ stock | flow | rate | index → frekans dönüşümünü ve yorumu belirler
- `cum_type` ∈ none | ytd | since_start
- Şema değişikliği gerekiyorsa önce `docs/contracts.md` güncellenir, bana sor.

## Komutlar

```bash
uv sync                      # ortam
uv run pytest -q             # testler (ağ ERİŞİMİ OLMADAN geçmeli)
uv run ruff check --fix .
uv run ruff format .         # CI `--check` ile koşuyor, format da kırar
# --source evds için --datagroups ya da --all ZORUNLU; ikisi de yoksa çıkar.
uv run python -m lakehouse_agent.ingest.run --source evds \
  --datagroups bie_kt100h,bie_kfe,bie_tukfiy2025,bie_akonutsat1,bie_akonutsat2,bie_hpbitablo6 \
  --since 2021-01 --until 2026-06
uv run python -m lakehouse_agent.ingest.run --source bddk --since 2021-01 --until 2026-06
uv run python -m lakehouse_agent.lake.gold        # bronze+silver -> data/gold
uv run python -m lakehouse_agent.lake.quality     # docs/data_quality.md
uv run python build_index.py                      # havuz DIŞI EVDS keşif indeksi
# Agent: üç mod. --real demoda koşan moddur.
uv run python -m lakehouse_agent.agent.loop --fake --question "..."
uv run python -m lakehouse_agent.agent.loop --real --question "..." --question "..."
```

## Kesin kurallar

1. **Bronze immutable.** Kaynaktan ne geldiyse o + url, parametreler, HTTP status, timestamp.
   Hiçbir dönüşüm bronze'da yapılmaz. Sayılar Türkçe biçimdeyse metin olarak saklanır.
2. **NaN doldurulmaz.** Eksik dönem eksik kalır; `fillna(0)`, `ffill`, `interpolate` yasak
   (açıkça istemediysem). Eksikler kalite raporuna yazılır.
3. **Birim çevrilmez.** silver'da `unit` sütununda orijinal birim durur; çevrim yalnızca
   araç katmanında yapılır ve `warnings`'e yazılır.
4. **Yüksek frekanstan düşüğe** gidilir. Aylıktan haftalığa upsampling = veri icat etmek, yasak.
5. **Testler ağsız.** Birim testlerde gerçek HTTP çağrısı yok; kaydedilmiş küçük fixture
   JSON/parquet kullan (`tests/fixtures/`). Sentetik veri üret, elle hesaplanmış beklenen değer yaz.
6. **Yeniden çalıştırılabilirlik.** Her ingest betiği var olan dosyayı atlar; ikinci koşuda
   sıfır HTTP çağrısı yapar.
7. Fonksiyonlar saf: `analysis/` DataFrame alır, DataFrame döndürür; model, oturum, araç bilmez.
8. Kod ve docstring İngilizce; prompt ve doküman Türkçe.

## Güvenlik (pazarlık yok)

- `.env` dosyasını **okuma, yazma, cat'leme, log'lama**. İçeriğini bana gösterme.
  Anahtarlar `config.py` içinde `pydantic-settings` ile ortamdan okunur.
- Hiçbir anahtarı koda, teste, fixture'a, docstring'e, örnek çıktıya, commit mesajına yazma.
- EVDS anahtarı **HTTP header'da `key` olarak** gider, URL query string'inde asla
  (query string proxy log'una, browser history'sine, Referer'a düşer — CWE-598).
- `fetch_log` ve kalite raporu URL saklıyor: URL'den anahtarı **redact et** (`key=***`).
- DuckDB bağlantısı `read_only=True` **ve** `enable_external_access=false`.
  Gerekçe: `run_sql` yalnızca SELECT kabul etse bile DuckDB'de SELECT içinden
  `read_csv('/etc/passwd')`, `ATTACH`, `INSTALL/LOAD httpfs` ile local file read ve
  data exfiltration mümkündür. Anahtar kelime filtresi (blacklist) tek başına yetmez;
  asıl kontrol motor seviyesindeki kısıt + zorunlu LIMIT'tir.
- `pickle.load` ile dışarıdan gelen dosya açma; parquet/json kullan.
- Playwright ya da requests ile çekilen HTML'i `eval`/`exec`'e sokma.

## "Bitti" tanımı

kod + test + docstring + (gerekiyorsa) `docs/data_dictionary.md` güncellemesi +
`ruff check` temiz + `uv run pytest` yeşil + main'de çalışıyor.
PR boyutu için sınır yok; iş mantıksal bütün olarak tek PR'da gider (tek geliştirici).

## Sık düşülen tuzaklar (bunlara karşı kod yaz)

- EVDS tek istekte **en fazla 1000 gözlem** döndürür ve fazlasını **hata vermeden keser**.
  Bu yüzden veri grubu × yıl parçalanır.
- EVDS `startDate` dönemin ilk günü olmalı (`01-01-YYYY`); değilse ilk dönem sessizce düşer.
- EVDS yanıtında seri kodundaki nokta alt çizgiye dönebilir (`TP.KTF12` → `TP_KTF12`),
  alan adlarında büyük/küçük harf sürprizi olur. İlk yanıtı gözle doğrula, varsayma.
- BDDK sayıları `"1.234.567,89"` biçiminde; binlik ayıracı nokta, ondalık virgül.
- BDDK kâr-zarar kalemleri yıl başından kümülatif (`ytd`); Ocak değeri akımla eşitse doğrulanır.
- TÜFE: `evds:TP.TUKFIY2025.GENEL`, 2025=100, **zincirleme yok** (TÜİK yeni bazı
  geriye doğru hesaplamış; `docs/decisions.md`). `evds:TUFE_ZINCIR` yok.
- STL 24 dönemden kısa seride çalışmaz → IQR fallback.

## EVDS v3 — doğrulanmış gerçekler (12 Eyl 2026, smoke test ile ölçüldü)

Aşağıdakiler bu dosyanın önceki "tuzaklar" bölümündeki EVDS satırlarını GEÇERSİZ KILAR.

- Base: https://evds3.tcmb.gov.tr/igmevdsms-dis/   (v2 ve /service/evds ÖLÜ)
- Parametreler path-style: /datagroups/mode=0&type=json — requests'in params=
  argümanı `?mode=0` üretir ve ÇALIŞMAZ. URL'i elle kur.
- Anahtar HTTP header'da "key". Eksik/yanlış = 403 Forbidden.
- 1000 gözlem sınırı ÖLÇÜLDÜ: aşılırsa endDate'ten GERİYE 1000 gözlem gelir,
  EN ESKİ veri sessizce düşer. (Test: 2021-01..2026-12 istendi, 2023-12-20'den
  başlayan 1000 satır geldi.) Bu yüzden veri grubu x YIL bölünür.
  Dönen satır 1000 ise WARNING logla VE ilk tarihi kontrol et.
- startDate o frekansın ilk günü olmalı. (Aylıkta fark ölçülmedi; kılavuz yıllık
  örnek veriyor. Yine de her zaman ayın 1'i yazılır — ucuz sigorta.)
- decimalSeperator'a DOKUNMA — varsayılan nokta.
- aggregationTypes / frequency / formulas parametrelerini BRONZE'da KULLANMA.
  Dönüşüm silver'da, kind'a göre yapılır; aksi halde lineage kaybolur.
- Yanıttaki sütun adlarında seri kodunun noktaları ALT ÇİZGİ olur:
  TP.ISTIRAKBS.A8 -> TP_ISTIRAKBS_A8. Katalogla eşlerken geri çevir.
- datagroups alanları: DATAGROUP_CODE, CATEGORY_ID, DATAGROUP_NAME(_ENG),
  FREQUENCY, FREQUENCY_STR, DATASOURCE, METADATA_LINK, START_DATE, END_DATE,
  NOTE, BIRIMI, LAST_UPDATED
- serieList alanları: SERIE_CODE, DATAGROUP_CODE, SERIE_NAME(_ENG),
  FREQUENCY_STR, DEFAULT_AGG_METHOD(_STR), TAG, DATASOURCE, METADATA_LINK,
  START_DATE, END_DATE, AVGABLE/FIRSTABLE/LASTABLE/MAXABLE/MINABLE/SUMABLE,
  SEVIYE, UST_SERIE_CODE
- 1000 sınırı SATIR bazlı, seri sayısından bağımsız (ölçüldü: 185 seri x 262 gün
  tek çağrıda geldi). Yıllık parçalama her frekans için güvenli.
- datagroup yanıtı WIDE gelir: her seri bir sütun, ilk sütun "Tarih".
  silver'da melt ile long'a çevrilir; aynı adımda sütun adındaki alt çizgi
