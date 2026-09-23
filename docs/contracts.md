# Sözleşmeler — v0.5

İki şerit aynı anda geliştirip sonra birleştiriyor. Bu belge, "benim modülüm
seninkine nasıl bağlanır" sorusunun yazılı cevabı. Amacı tam bir spesifikasyon
üretmek değil, birleştirme gününde imza uyuşmazlığı çıkmasını engellemek.

**Değişiklik kuralı:** şema veya imza değişikliği PR ile gelir, karşı şerit
onaylar, sürüm numarası artar.

| Şerit | Sahibi | Kapsam |
|---|---|---|
| A — veri | Burak | ingest, lake, analysis (nedensellik hariç) |
| B — agent | Derya | llm, agent, tools, arayüz, eval |

---

## 1. Veri sözleşmesi

A'nın `CLAUDE.md`'sindeki kararlar geçerli, burada tekrar edilmiyor. Özet:

- `data/gold/observations.parquet` sütunları, tam olarak ve bu sırada:
  `series_id, period, freq, value, unit, source, fetched_at, obs_date`
  (`obs_date` 22 Eyl 2026, v0.5: kaynağın değere verdiği tarih. Haftalık seride
  gözlem günü (Cuma, tatil haftasında başka gün), günlükte günün kendisi;
  aylık/üç aylık/yıllıkta kaynak gün vermediği için `period`.)
- `data/gold/series_catalog.parquet` sütunları (15):
  `series_id, name_tr, name_en, source, freq, unit, kind, agg_method, cum_type,
  start_date, end_date, topic, tags, metadata_url, notes`
- `series_id` önekli: `evds:TP.KTF12`, `bddk:aylik:tuketici:konut:TL:sektor`
- `period` = dönemin ilk günü. Aylık ayın 1'i, haftalık Pazartesi.
- `kind ∈ stock | flow | rate | index` — cevabın dilini belirler (fark yüzde
  puan mı, enflasyondan arındırılabilir mi). Kurallar hiçbir türe yerleştiremediği
  seride boş (`lake/classify.py`, `docs/kind_review.md`). v0.5'ten beri hesabı
  belirlemiyor: bkz. `agg_method`.
- `agg_method ∈ last | sum | avg | first | min | max` — daha seyrek frekansa
  çevirirken ne alınacağı. EVDS'de kaynağın kendi önerisi (`DEFAULT_AGG_METHOD`),
  BDDK'da tablo tanımı (`ingest/bddk_tables.yaml`). `to_freq` bunu kullanır.
- `cum_type ∈ none | ytd | since_start` — **ham verinin** kümülatifliği. `ytd`
  seri (BDDK kâr-zarar) silver'da aylık akıma çevrilir; gold'daki değer akımdır.
- Pencere gözlem tarihine göre: `2021-01-01 <= obs_date <= 2026-06-30`
  (`lake/gold.py` `GOLD_SINCE`/`GOLD_UNTIL`). `get(start, end)` de `obs_date`'e
  göre süzer.
- TÜFE = `evds:TP.TUKFIY2025.GENEL`, 2025=100. **Zincirleme yok**,
  `evds:TUFE_ZINCIR` hiçbir yerde geçmiyor.

### 1.1 `period` dtype kuralı (bağlayıcı)

Gold/Parquet tarafında `DATE`. **Python tarafında, araç katmanına giren her
tabloda `datetime64`.**

Gerekçe: pandas 3'te `datetime.date` nesnelerinden oluşan bir sütunu datetime
sütunuyla birleştirmek ya hata veriyor ya da **sessizce sıfır satır eşleştiriyor**.
`get_series(into="active")` ve `deflate` tam olarak bu birleştirmeyi yapıyor, yani
hata sessiz kalırsa boş tabloyla demo yapmış oluruz.

Normalizasyon noktaları: veri katmanının çıkışı ve agent tarafındaki adaptör.
İkisinde de yapılıyorsa biri kaldırılır; **A'ya sorulacak.**

---

## 2. Lake imza teklifi

B teklif ediyor, A onaylıyor. `FakeLake` bu imzaların birebir aynısını uyguluyor,
yani teklif çalışan kodla birlikte geliyor. A itiraz ederse değişen tek dosya
sahte katman olur, agent kodu değil.

```python
def get(series_ids: list[str], start: str, end: str) -> pd.DataFrame
    # long biçim; sütunlar TAM OLARAK: series_id, period, freq, value, unit, source,
    # fetched_at, obs_date. Pencere obs_date'e göre.

def wide(series_ids: list[str], start: str, end: str, freq: str = "M") -> pd.DataFrame
    # 'period' + her series_id için bir sütun; her seri kendi agg_method'uyla

def to_freq(df, kind: str | None, freq: str, *, agg_method: str | None = None)
        -> tuple[pd.DataFrame, str]
    # yöntem: agg_method verilirse o, yoksa kind'ın varsayılanı (stock -> son,
    # flow -> toplam, rate/index -> ortalama). v0.4'te kaldırılan agg_method
    # v0.5'te geri geldi: bitiş değeri yayımlanan bir oran kind'a göre
    # ortalanıyordu.
    # değer, obs_date'i (yoksa period'u) hangi döneme düşüyorsa oraya sayılır:
    # 4 Nisan Cuma'lı hafta Nisan'ındır.
    # yalnızca yüksek frekanstan düşüğe; tersi veri icat etmektir
    # ikinci dönüş değeri uygulanan formül (plan 4.1)

def search(query: str, k: int = 20) -> list[CatalogHit]
    # havuzun kendi kataloğunda (gold series_catalog) arar; havuzda olmayanı döndürmez
    # LakeAdapter ayrıca search_detailed(query, k) -> (hits, notlar) sunar:
    # embedding yapılamadığında, büyük havuzda indeks kurulmamışsa ya da indeks
    # eski katalogdan kurulmuşsa bunu not düşer. Büyük havuz (> 2000 seri)
    # data/pool_index (LanceDB) üzerinden aranır; kurulum:
    # python -m lakehouse_agent.lake.pool_index

def catalog_lookup(series_ids: list[str]) -> pd.DataFrame     # 15 katalog sütunu

def prepare_table(df, hints: dict | None = None) -> tuple[pd.DataFrame, QualityReport]

def profile_table(df) -> QualityReport
```

`start` ve `end` biçimi `YYYY-MM`. Ayrı bir `Protocol` sınıfı **yok**: uyum bu
tablo ve gerçek koşumla sağlanıyor, gereksiz soyutlama eklenmiyor.

### 2.1 Katalog skoru

`CatalogHit.score` (22 Eylül'den beri): sorgu ile serinin arama metni arasındaki
**kosinüs benzerliği** (Qwen3-Embedding-8B). Seri kimliği, kodu ya da araçların
verdiği sütun adı sorguda geçiyorsa (`TP.KTF12'nin`, `TP_KTF12` da sayılır) 1,0.
Embedding yapılamazsa kelime eşleşme oranı (0–1; Türkçe harfler ASCII'ye
katlanır, alt çizgi kelime ayırır). 0,40'ın altındaki aday hiç önerilmez;
eşiğin ölçümü `docs/decisions.md`'de.

**Sıra skorla aynı şey değil** (22 Eyl, 45 bin serilik havuz): skor kosinüs
benzerliği olarak kalır, ama adaylar `lake/pool_index.rank` sırasına göre
sunulur. Kaynağın seri hakkında söylediğine bakar: sorgu yer adı söylemiyorsa
il/bölge serileri ulusal serilerden sonra gelir ve aynı kalemin il kırılımları
tek adayda toplanır (aday kaç seriyi temsil ettiğini nota yazar); EVDS'in
"(Arşiv)" işaretlediği seri güncelden sonra gelir; sorgu bir veri grubunun adını
söylüyorsa o grubun ana serisi öne gelir. Ölçüm: `docs/decisions.md`, "Havuz
araması: sıralama kuralları".

`CatalogHit.table` (23 Eylül'den beri, isteğe bağlı): serinin oturduğu kaynak
tablo, EVDS veri grubunun adı. `search_catalog` bunu modele `tablo` olarak
gösterir ve onay kartı ayrıntı satırının başına koyar. Eklenme sebebi: "Konut
Kredisi (TL, Akım, %)" hem "Kredi Faiz Oranları" hem "Kredi Kâr Oranları"
tablosunda var; biri faiz, öteki katılım bankası kâr payı, ve ad, birim,
frekans, tür dördü de aynı. Ayıran tek alan tablo. İndeksi olmayan küçük
havuzda boş kalır.

`CatalogHit.match` adayın nasıl bulunduğunu söyler: `code` | `semantic` |
`keyword`. `search_catalog` bunu modele `eslesme: kod | anlam | kelime` olarak
gösterir. Kelime eşleşmesiyle hiç aday çıkmazsa araç "havuzda yok" demez:
kaba arama bir şey bulamadı diye veri yok sayılmaz. Embedding hatası 180 sn
hatırlanır; o sürede aramalar doğrudan kelime eşleşmesine geçer.

`start_date`/`end_date` havuzun **gerçek kapsamı** (değer içeren ilk ve son ay,
`YYYY-MM`, `obs_date`'e göre), katalogdaki kaynak yayın aralığı değil. Kapsam
`lake/gold.py::coverage()` ile okunur; modele açık `run_sql`'in zorunlu
`LIMIT 1000`'i havuzun tamamını kesiyordu.

"Belirsiz seri" onayı **yalnızca oranla** ölçülüyor (`top2 >= top1 × 0.9`).
Eski EVDS indeksinde skor sıradan türetildiği için (1/(61+sıra)) bu oran her
zaman ~0,98'di. Kapı ayrıca ilk iki adayın birimi ya da türü farklıysa da
açılıyordu; havuzdaki 36 çiftin 31'i böyle olduğundan neredeyse her arama
durdu. 22 Eylül'den beri birim/tür farkı kapıyı tek başına açmaz, sorguda
koduyla geçen seriler (`eslesme: kod`) seçim sayılmaz.

---

## 3. Ortak tipler

Alan listesi burada tekrar edilmiyor, tek kaynak `src/lakehouse_agent/contracts.py`.

**Tek tanım kuralı (bağlayıcı):** şu sekiz tip başka hiçbir dosyada yeniden
tanımlanmaz. İkinci bir tanım inceleme engelidir, biçim uyarısı değil: iki kopya
sessizce ayrışır ve fark ancak birleştirme gününde ortaya çıkar.

`Usage`, `Source`, `ColumnInfo`, `Artifact`, `QualityReport`, `CatalogHit`,
`ToolResult`, `SeriesNotFound`

Modülün bağımlılığı yalnızca pydantic. Böylece A'nın `lake/quality.py`'si
`QualityReport`'u agent paketini çekmeden, agent da pandas'ı çekmeden import
edebiliyor.

**A'ya sorulacak:** `lake/quality.py` `QualityReport`'u buradan import edecek mi?
Etmezse katman tersine döner.

---

## 4. Araç sonucu

Her araç `ToolResult` döndürür. En önemli kural: **model tabloyu görmez, özetini
görür.** Tam tablo oturum deposunda kalır ve doğrudan arayüze gider. Böylece
bağlam şişmez ve model rakamları kopyalarken basamak kaydırmaz.

`table_ref` deseni: `^t\d{2,}_[a-z][a-z0-9_]{0,24}$`, örnek `t01_get_series`.
`chart_ref` deseni: `^c\d{2,}$`, örnek `c01`. İkisi de en az iki basamak;
yüzüncü tabloda numara genişler, desen bunu kabul eder.

Üretimi yalnızca `ToolContext.new_table_ref()` yapar. Desen araç şemasında
modele de görünür, yani model `"konut_tablosu"` uydurursa doğrulama anında
reddeder. MIA testinde modelin açıklamasız bir alana SQL yazdığı görüldüğü için
bu ucuz bir sigorta.

`table_ref=None` aktif tablo demektir. `"active"` diye özel bir değer yok.

Modele giden araç mesajı **2000 karakterle sınırlı**; aşarsa kesilir ve sonuna
`... (kisaltildi)` yazılır. Konuşkan bir aracın konuşmanın geri kalanını
bağlamdan itmemesi için. Uzun çıktı, aracın daha iyi özetlemesi gerektiğinin
işareti.

Araç kaydında **ne ürettiğini beyan eden alan yok.** Yeni tablo ve yeni sütun
farkını döngü, aracın gerçekte yazdıklarından hesaplıyor; beyan ile gerçeğin
ayrışması sessiz bir yanlışa yol açardı.

---

## 5. Oturum durumu

`SessionState` düz bir `dataclass`. Tek süreç, tek thread olduğu için reducer ve
checkpointer gerekmiyor.

Alanlar: `session_id, messages, tables, active_table, lineage, steps, sources,
charts, usage`
artı tur defteri (`turn_no, step_no, steps_this_turn, active_s, approval_wait_s,
pending_calls, arg_retries, call_signatures, warnings, truncated_reason,
draft_answer, verification, approvals, retired_tables`) ve modelin gördüğü
geçmiş için iki dizin: `turn_starts` (her turun ilk mesajının sırası) ve
`answer_indices` (son cevap mesajlarının sırası). Önceki turlar modele yalnızca
soru + son cevap olarak gider (`agent/history.py`); `messages` hiç kırpılmaz.

`lineage` anahtarı: `"{table_ref}.{column}"`.

`truncated_reason` ∈ `step_limit | time_budget | repeat_loop | llm_error |
cancelled`. `cancelled` tur durdurulduğunda yazılır: `done` olayı bunu taşır ve
yanında bir `error` olayı gider, yoksa durdurulan tur bitmiş turdan ayırt
edilemiyordu. `draft_answer` her tur başında temizlenir.

**Doğrulayıcı hata veremez.** `loop._verify` doğrulayıcıyı try/except içinde
çağırır; çöken bir doğrulayıcı "işaretlenecek bir şey yok" olarak okunur ve
loglanır. Cevap o noktada zaten yazılmıştır, bir şüpheyi bildirmek için bitmiş
işi çöpe atmak turun en kötü takasıdır.

**Zaman muhasebesi.** Bütçe `active_s` üzerinden ölçülüyor ve insan beklemesi ona
girmiyor: `approval_wait_s` yalnız onay kartının açık kaldığı süreyi tutuyor,
adımın çalışma süresinden düşülüyor. Jüri kartı okurken tur düşmesin diye.

Aktif tablo kuralı: `get_series(into="new")`, `run_sql`, `ingest` ve `read_url`
aktif tabloyu değiştirir; analiz sonucu tabloları değiştirmez. Üretim kaydı
(`tools/toolset.py`, sekiz araç): search_catalog, get_series,
transform_table, detect_anomalies, detect_changes, **read_url**,
**test_causality**, **web_search**. `run_sql` ve `ingest` henüz yalnızca
sahte araçlarda; kural sıraları geldiğinde uyacakları kural.

`read_url` aktif tabloyu devraldığında bunu **uyarıda söyler** ve önceki
tablonun adını verir: analizin ortasında okunan bir adres, sonraki her
`into='active'` çağrısını sessizce başka tabloya yönlendiriyordu.

---

## 6. Olay listesi

Agent her adımda bir olay yayınlar, arayüz olay türüne göre çizer. Agent
çekirdeği arayüzü **import etmez**.

| Olay | Payload | Arayüzde |
|---|---|---|
| `plan` | `{text}` | plan mesajı |
| `step_start` | `{step_no, node, tool, args}` | adım kutusu açılır |
| `step_end` | `{step_no, node, ok, summary, error, duration_ms, tokens}` | adım kutusu kapanır |
| `table_update` | `{table_ref, columns, rows, new_columns}` | tablo, yeni sütun vurgulu |
| `chart` | `{chart_ref, plotly_json}` | grafik |
| `answer_delta` | `{text}` | cevap akışı |
| `verification` | `{passed, flags}` | doğrulanamayan ifadeler |
| `sources` | `{sources: [{series_id \| url, name, fetched_at}]}` | kaynak listesi |
| `quality_report` | `{table_ref, rows, freq, missing_pct, ...}` | kalite kartı |
| `approval_request` | `{request_id, kind, question, options, default}` | onay kartı |
| `approval_decision` | `{request_id, kind, choice, note, automatic}` | kart kapanır; `automatic` kimse cevaplamadı demek |
| `done` / `error` | `{...}` | kapanış |

**Payload kuralı:** JSON serileştirilebilir olmalı. DataFrame asla konmaz,
yalnızca `preview` satırları.

**Payload sözlüktür.** `Event.payload` tipi `dict[str, Any]`; düz liste
taşıyamaz. Tabloda liste görünen her payload bir anahtarın altındadır.

**Sapma koruması:** `agent/events.py` yazıldı, bu tablodaki adlar koddaki
`EventType` ile `tests/test_events.py` içinde **iki yönlü** karşılaştırılıyor:
kodda olup belgede olmayan da, belgede olup kodda olmayan da testi kırar.
`EVENT_TYPES` ayrıca `EventType`'tan türetiliyor, elle yazılmıyor.

---

## 7. LLM erişimi

`openai` paketini yalnızca `llm.py` import eder. Statik bir test bunu koruyor.

```python
chat(messages, tools=None, thinking=True, schema=None, *, max_tokens=None, tag=None,
     session_id=None, timeout_s=None) -> ChatResult
embed(texts, *, tag=None) -> list[list[float]]
ocr(images, *, prompt=None, tag=None) -> OcrResult
```

Ölçülen davranış (15 Eylül, canlı MIA):

| Konu | Sonuç |
|---|---|
| openai sürümü | 3.14.0, bağımlılığı `httpx2` |
| Araç çağırma | çalışıyor, `tool_call` id biçimi `chatcmpl-tool-<hex>` |
| JSON şema | çalışıyor, `strict` bayrağı da hata vermiyor |
| Embedding boyutu | 4096 |
| Hatalı model kimliği | 403 döner, 404 değil |
| thinking açık | 2044 ms / 188 token |
| thinking kapalı | 523 ms / 48 token |
| thinking açıkken | mesaja `reasoning` alanı ekleniyor, geri gönderilmiyor |

Thinking politikası: planlama ve cevapta açık, gözlemde kapalı.

Retry bizde: 429, 5xx ve kopan bağlantı için üç deneme, üstel bekleme. SDK'nın
kendi retry'ı kapalı, çünkü onunki sessiz, bizimki loglanıyor. **Zaman aşımı
tekrar denenmez** (22 Eylül): aynı istek aynı yük altında yine aynı sürer.

`timeout_s` çağrının toplam süre bütçesidir, bekleme süreleri dahil; her
denemenin HTTP zaman aşımı kalan sürenin tamamıdır (`LLM_TIMEOUT_S` ile
kırpılmaz, sınırı çağıran seçer). Sonraki deneme bekleme sonrasında 5 saniyeden
az süre bulacaksa hiç beklenmez ve başlanmaz; o zaman hata `LlmTimeoutError`
olur (son sunucu hatası `__cause__`, logda `budget_after_<Sınıf>`): denemeyi
sunucu değil süre bitirmiştir. Her denemesi 5xx dönen çağrı hâlâ
`LlmServerError`.

Döngü tarafı (`loop._chat`): çağrının payı başta bir kez hesaplanır, turun
kalan bütçesi; cevapta en az 30 sn. Denemelerin hepsi bu tek paydan yer. Plan ve
gözlem denemesi ayrıca `LLM_TIMEOUT_S` ile sınırlanır, ardından gelen adımlara
süre kalsın diye. Düşünme açık deneme `pay - 10 sn` ile sınırlanır; zaman
aşımına uğrarsa, **geriye en az 5 sn (`MIN_ATTEMPT_S`) kaldıysa** bir kez
düşünme kapalı ve `max_tokens=1024` ile denenir. Eşik 5 sn, çünkü zaman aşımı
her zaman verilen süreyi bir miktar aşar ve 10 sn'lik bir eşik yedeği pratikte
hiç çalıştırmıyordu (23 Eyl, `docs/decisions.md`). Cevabın düşünme kapalı
denemesi son şanstır,
kalan payın tamamını alır. Oturumun son model çağrılarında ölçülen hız
(token/sn) düşünme açık bir cevabı (800 token) pencereye sığdıramıyorsa cevap
baştan düşünme kapalı başlar.

Kullanım logu `data/logs/llm_usage.jsonl`, her mantıksal çağrı için tek satır,
**yalnızca sayaçlar**. Mesaj içeriği ve anahtar asla yazılmaz.

---

## 7.1 Adres okuma (`read_url`, 22 Eyl)

Demo günü prompt'ta URL verilecek (gereksinim: PDF, Excel, resim, metin).
`tools/web_url.py` bunu okur; `web/fetch.py` indirir, `web/extract.py` ayrıştırır
(ağ yalnızca ilkinde, bu yüzden ayrıştırma tamamen ağsız test edilir).

- **Sınırlar:** yalnız `http(s)`; alan adı çözülür ve localhost/özel/link-local
  adresler reddedilir (her yönlendirmede yeniden); 10 MB ve 50 sn; en çok 5
  yönlendirme. Kimlik bilgisi gönderilmez.
- **Önbellek:** indirilen dosya `data/adhoc/url/<hash>` ve yanında istek kaydı
  (`url, final_url, http_status, content_type, fetched_at, bytes`). İkinci okuma
  çağrı yapmaz. Havuza (gold) hiçbir şey yazılmaz.
- **Çıkan her veri tablosu** oturuma ayrı tablo olarak girer (`t..._read_url`),
  en büyüğü aktif tablo olur ve kalite kartı onun içindir. En çok 10 tablo;
  fazlası sayılır ve söylenir. Yerleşim tabloları ve boş kabuklar elenir
  (en az 2 sütun; sayfada bulunmuşsa en az 2 satır).
- **Tablo çıkmazsa** sayfanın bağlandığı veri dosyaları (PDF/Excel/CSV)
  adlarıyla listelenir — gereksinimin ilk hedefi tam olarak böyle bir sayfa.
- `prepare_table` (lake adaptörü) her tabloyu temizler: adı ne olursa olsun
  tarih sütunu `period` olur, Türkçe biçimli sayılar sayıya çevrilir, ne
  yapıldığı kalite kartına yazılır.

`transform_table`'a `top` işlemi eklendi: bir sütuna göre en yüksek (ya da
`ascending` ile en düşük) satırları sıralar ve değerleriyle listeler. Tabloya
sütun eklemez ve `period` sütunu olmayan tablolarda da çalışır — basılı bir
tablonun satırları tarihli değil adlıdır ("Ocak / January"). "Hangi ay en
yüksek" sorusunun cevabı böylece araçtan gelir, modelin saymasından değil.

---

## 7.2 Öncüllük (`test_causality`, 22 Eyl)

`analysis/causality.py` saf, `tools/analysis.py` sarmalar. Aynı tablodan iki
sütun alır, iki tablo yazar (hiçbiri aktif tabloyu değiştirmez):

- **Granger tablosu**, yön başına bir satır: `yon, en_iyi_gecikme, p_degeri,
  anlamli, gozlem, donusum`.
- **Gecikmeli korelasyon tablosu**, −n…+n gecikme için `korelasyon` ve
  `gozlem`. Pozitif gecikme = ilk sütun önde (x[t−k] ile y[t]).

Kurallar:

- Durağanlık şart. ADF durağan demiyorsa fark alınır; sütunun türü stok ya da
  akımsa log-fark (değişim = büyüme oranı), oran ve endekste düz fark. Tür
  sütunun soyağacından okunur; bilinmiyorsa düz fark. Uygulanan dönüşüm
  tabloda yazar.
- Eşik `0,05 / denenen gecikme` (Bonferroni): altı gecikmenin en iyisini
  almak yirmide bir yanlış alarmı dörtte bire çıkarır.
- Gecikme başına en az 3 gözlem + 10 taban; seri kısaysa gecikme sayısı
  düşürülür, tek gecikmeye bile yetmiyorsa araç reddeder.
- Özet metni her zaman "bu nedensellik kanıtı değildir, üçüncü etkenler
  dışlanmadı" cümlesini taşır; `prompts/answer.md` kural 8 modeli buna bağlar.

---

## 7.3 Web araması (`web_search`, 22 Eyl)

Havuzun cevaplayamadığı soru için: bir tarihte ne olduğu, hangi kararın
alındığı. `web/search.py` arar, `tools/web_search.py` sunar.

- **Sağlayıcı değiştirilebilir.** Bugün `ddgs` (DuckDuckGo, anahtarsız,
  ücretsiz, hız sınırlı). Ücretli bir servise geçmek tek fonksiyon değiştirir;
  araç, prompt ve testler aynı kalır. Sağlayıcı **çağrı anında** çözülür;
  varsayılan argüman olarak bağlamak, testlerin sessizce ağa çıkmasına yol
  açıyordu (ölçüldü, üç test).
- **Sonuçlar veri, talimat değil.** Araç çıktısının başında bunu söyleyen bir
  etiket var: içindeki yönergelere uyulmaz. Hiçbir şey `eval`/`exec`'e
  girmez, tabloya yazılmaz.
- **Sayı üretmez.** Tablo yazmaz, `preview` döndürmez. Cevaptaki sayılar
  yalnızca havuz araçlarından gelir; `answer.md` kural 5 modele arama
  özetinden sayı aktarmayı yasaklar (gerekçe: ölçümde model Temmuz 2023
  kararını Ağustos'un oranıyla yazdı).
- **Onay kapısı:** `approval="fetch"`, `read_url` ile aynı. Kart sorguyu
  gönderilmeden önce gösterir.
- Sonuçlar sorgu başına `data/adhoc/search/` altında önbelleklenir (zaman
  damgasıyla); aynı soru ikinci kez çağrı yapmaz.

**Kırılma → açıklama zinciri:** `detect_changes` bir tarih döndürdüğünde
planner (kural 19) kullanıcı istemeden o ay için arama yapar, `answer.md`
kural 5 de cevabın bunu kaynağıyla anlatmasını zorunlu kılar.

---

### 7.4 Doğrulayıcı bayrakları

`VerificationFlag.kind` ∈ `number_not_found | source_not_listed |
source_not_cited | date_out_of_range | unit_mix | model_contradiction`.
`source_not_cited` 22 Eyl'de web araçlarıyla eklendi: cevapta uydurma kaynak
olmasın diye `source_not_listed` vardı, ama sessiz hali eksikti — tur web'den
okuduysa cevabın okuduğu adresi anması gerekir, yoksa bilgi havuzdan gelmiş
gibi görünür. Adresin tamamı, şemasız hali, dosya adı ya da alan adı geçerse
anılmış sayılır (uzun adresler "tcmb.gov.tr/…/duy2023-25" diye kısaltılıyor).

---

## 8. Onay noktaları

Üç tane, fazlası değil. Kararı döngü verir, araçlar değil.

| Tür | Ne zaman | Seçenekler |
|---|---|---|
| `series_choice` | ilk iki adayın skor oranı eşiği aşarsa (sorguda koduyla geçen seriler hariç) | aday `series_id`'leri |
| `fetch` | havuzda olmayan veri çekilmeden önce (`read_url` ve `web_search`) | `approve`, `cancel` |
| `quality` | eksik oranı eşiği aşarsa, tekrar varsa, birim bilinmiyorsa | `continue`, `change_assumption`, `cancel` |

Kimlikler İngilizce çünkü kodda karşılaştırılıyor; etiketler Türkçe çünkü
kullanıcı onları okuyor.

Araç kaydında iki bayrak kapıyı tetikler: `approval="fetch"` ve
`returns_candidates=True`. Araç adına bakılmıyor — araç yeniden adlandırıldığında
bir insan onayının sessizce devre dışı kalması mümkün olmamalı.

Onay geri çağrısı **async ve engelleyici**. Onay beklerken tur süresi bütçesi
işlemez, yoksa jüri düşünürken tur düşer. Geri çağrı **hata verirse** (tarayıcı
kapandı, kart zaman aşımına uğradı) ya da tur kartta durdurulursa araç
çalıştırılmamış sayılır ve çağrıya yine de bir araç mesajı yazılır: cevapsız bir
`tool_call_id` sonraki isteği protokol hatasına düşürür ve oturumun kalanını da
götürür. Bir tur yine de cevapsız çağrı bırakırsa sonraki tur başlarken o turun
içinde onarılır.

Kimsenin cevaplamadığı karar (`auto_default`, ya da geçersiz/başka karta ait
cevap yüzünden varsayılana düşülmesi) `automatic=True` taşır. Otomatik seri
seçimi modele hiç söylenmez; otomatik kalite kararı "[Otomatik] ..." diye
söylenir, "kart görüldü" denmez.

---

## 9. Dosya sahipliği

**22 Eylül'den beri tek geliştirici var (Derya); aşağıdaki tablo tarihsel.**
Şerit kuralları artık yazma izni belirlemiyor, yalnızca kodun kimin
gerekçesiyle yazıldığını gösteriyor.

| Yol | Sahip | İnceleyen |
|---|---|---|
| `ingest/**`, `lake/**`, `analysis/transforms.py`, `analysis/anomaly.py`, `analysis/change.py` | A | B |
| `docs/data_dictionary.md`, `docs/data_quality.md` | A | B |
| `llm.py`, `contracts.py`, `agent/**`, `testing/**` | B | A |
| `tools/registry.py`, `tools/lakehouse.py`, `tools/web_*.py`, `tools/chart.py` | B | A |
| `analysis/causality.py`, `ui/**`, `eval/**` | B | A |
| `tools/analysis.py` | A (anomali, kırılma) + B (nedensellik) | karşı taraf |
| `config.py` | A (`Settings`) + B (`AgentSettings`) | karşı taraf |
| `pyproject.toml`, `.gitignore`, `.env.example`, `CLAUDE.md`, `docs/decisions.md`, `docs/contracts.md` | ortak | karşı taraf |
| `.github/**`, `Dockerfile`, `deploy/**` | açık | — |

Test dosyaları kendi modüllerinin sahibine aittir.

---

## 10. A'dan B'ye teslim sırası — tamamlandı

Yedi fonksiyonun tamamı geldi. `FakeLake` artık yedek değil, sözleşme paritesi
testinin diğer yarısı (`tests/test_lake_contract_parity.py`).

| # | Yüzey | Nerede | Kim yazdı |
|---|---|---|---|
| 1 | `get()`, `wide(freq="M")` | `lake/gold.py` | A |
| 2 | `to_freq()` | `analysis/transforms.py`, adapter sarıyor | A |
| 3 | `search()` | `tools/lake_adapter.py::search_detailed`, havuzun kendi kataloğu üzerinde (`lake/pool_index.py`, `data/pool_index`; kelime eşleşmesi yedeğiyle). `lake/catalog.py::search_catalog` + `data/catalog.lance` yalnızca havuz dışı keşifte. | B |
| 4 | `profile_table()` | `lake/quality.py` (`QualityReport` buradan import ediliyor) | A |
| 5 | `catalog_lookup()`, `prepare_table()` | `tools/lake_adapter.py` | B |
| 6 | anomali ve kırılma sarmalayıcıları | `tools/analysis.py` | A |
| 7 | `ingest()` canlı çekim | `ingest/run.py` | A |

**`lake/prepare.py` yazılmayacak.** `catalog_lookup` ve `prepare_table`
adapter'da; dört yüzeyi tek lake arayüzünde toplayan yer orası.

**Kapandı (22 Eylül):** `gold.wide(freq=...)` dönüşüm formülünü hâlâ atıyor;
`tools/lakehouse.py::get_series` onu `catalog_lookup`'ın frekansı ve
`agg_method`'undan türetiyor: sütunun soyağacına `op="to_freq"` ve Türkçe
formül, `warnings`'e "haftalık yayımlanıyor; aylık değere ... alınarak
çevrildi" giriyor. `get_series(freq=D|W|M|Q|A)`, varsayılan M; serinin kendi
frekansından sık bir tablo istenirse reddeder, aktif tabloya başka frekansta
sütun eklemeyi de reddeder.

B'den A'ya tek teslim: `MiaLLM.embed()`, katalog gömme için. **Hazır**
(`llm.py`, ölçülen boyut 4096).
