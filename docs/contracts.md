# Sözleşmeler — v0.2

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
  `series_id, period, freq, value, unit, source, fetched_at`
- `data/gold/series_catalog.parquet` sütunları (15):
  `series_id, name_tr, name_en, source, freq, unit, kind, agg_method, cum_type,
  start_date, end_date, topic, tags, metadata_url, notes`
- `series_id` önekli: `evds:TP.KTF12`, `bddk:aylik:tuketici:konut:TL:sektor`
- `period` = dönemin ilk günü. Aylık ayın 1'i, haftalık Pazartesi.
- `kind ∈ stock | flow | rate | index` — hem hesabı hem cevabın dilini belirler
- `cum_type ∈ none | ytd | since_start`
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
    # long biçim; sütunlar TAM OLARAK: series_id, period, freq, value, unit, source, fetched_at

def wide(series_ids: list[str], start: str, end: str, freq: str = "M") -> pd.DataFrame
    # 'period' + her series_id için bir sütun

def to_freq(df, kind: str, freq: str) -> tuple[pd.DataFrame, str]
    # stock -> son değer, flow -> toplam, rate/index -> ortalama
    # yalnızca yüksek frekanstan düşüğe; tersi veri icat etmektir
    # ikinci dönüş değeri uygulanan formül (plan 4.1: her dönüşüm ne yaptığını
    # söyler, sütunun soyağacına o metin girer)
    # agg_method parametresi KALDIRILDI: kind zaten yöntemi belirliyor ve
    # gerçek analysis/transforms.to_freq onu almıyor

def search(query: str, k: int = 20) -> list[CatalogHit]

def catalog_lookup(series_ids: list[str]) -> pd.DataFrame     # 15 katalog sütunu

def prepare_table(df, hints: dict | None = None) -> tuple[pd.DataFrame, QualityReport]

def profile_table(df) -> QualityReport
```

`start` ve `end` biçimi `YYYY-MM`. Ayrı bir `Protocol` sınıfı **yok**: uyum bu
tablo ve gerçek koşumla sağlanıyor, gereksiz soyutlama eklenmiyor.

### 2.1 Katalog skoru

`CatalogHit.score` bir sıralama skoru, benzerlik değil. Hibrit arama RRF skoru
döndürüyorsa değerler 0–1 aralığında olmaz ve yalnızca aynı sorgu içinde
karşılaştırılabilir.

Bu yüzden "belirsiz seri" onayı **oranla** ölçülüyor (`top2 >= top1 × 0.9`),
mutlak farkla değil. Mutlak eşik her aramada onay tetiklerdi.

**A'ya sorulacak:** ham skor mu dönecek, normalize mi edilecek? Eşiği gerçek
katalogda kim kalibre edecek?

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

`table_ref` deseni: `^t\d{2}_[a-z][a-z0-9_]{0,24}$`, örnek `t01_get_series`.

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

Alanlar: `messages, tables, active_table, lineage, steps, sources, charts, usage`
artı tur defteri (`turn_no, step_no, steps_this_turn, active_s, approval_wait_s,
pending_calls, arg_retries, call_signatures, warnings, truncated_reason,
draft_answer, verification, approvals`).

`lineage` anahtarı: `"{table_ref}.{column}"`.

Aktif tablo kuralı: `get_series(into="new")`, `run_sql`, `ingest` ve `read_url`
aktif tabloyu değiştirir; analiz sonucu tabloları değiştirmez.

---

## 6. Olay listesi

Agent her adımda bir olay yayınlar, arayüz olay türüne göre çizer. Agent
çekirdeği arayüzü **import etmez**.

| Olay | Payload | Arayüzde |
|---|---|---|
| `plan` | `{text}` | plan mesajı |
| `step_start` | `{step_no, tool, args}` | adım kutusu açılır |
| `step_end` | `{step_no, ok, summary, duration_ms, tokens}` | adım kutusu kapanır |
| `table_update` | `{table_ref, columns, rows, new_columns}` | tablo, yeni sütun vurgulu |
| `chart` | `{chart_ref, plotly_json}` | grafik |
| `answer_delta` | `{text}` | cevap akışı |
| `verification` | `{passed, flags}` | doğrulanamayan ifadeler |
| `sources` | `[{series_id \| url, name, fetched_at}]` | kaynak listesi |
| `quality_report` | `{table_ref, rows, freq, missing_pct, ...}` | kalite kartı |
| `approval_request` | `{request_id, kind, question, options, default}` | onay kartı |
| `done` / `error` | `{...}` | kapanış |

**Payload kuralı:** JSON serileştirilebilir olmalı. DataFrame asla konmaz,
yalnızca `preview` satırları.

**Sapma koruması (aşama 3'te gelecek):** `agent/events.py` yazıldığında bu
tablodaki adlar koddaki `EventType` ile bir testle karşılaştırılacak. Şu an
`EventType` henüz yok, yani bu tablo tek taraflı bir taahhüt; testi yazılana
kadar elle güncel tutulmalı.

---

## 7. LLM erişimi

`openai` paketini yalnızca `llm.py` import eder. Statik bir test bunu koruyor.

```python
chat(messages, tools=None, thinking=True, schema=None, *, tag=None, session_id=None) -> ChatResult
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

Retry bizde: 429 ve 5xx için üç deneme, üstel bekleme. SDK'nın kendi retry'ı
kapalı, çünkü onunki sessiz, bizimki loglanıyor.

Kullanım logu `data/logs/llm_usage.jsonl`, her mantıksal çağrı için tek satır,
**yalnızca sayaçlar**. Mesaj içeriği ve anahtar asla yazılmaz.

---

## 8. Onay noktaları

Üç tane, fazlası değil. Kararı döngü verir, araçlar değil.

| Tür | Ne zaman | Seçenekler |
|---|---|---|
| `series_choice` | ilk iki adayın skor oranı eşiği aşarsa, ya da birim/kind farklıysa | aday `series_id`'leri |
| `fetch` | havuzda olmayan veri çekilmeden önce | `approve`, `cancel` |
| `quality` | eksik oranı eşiği aşarsa, tekrar varsa, birim bilinmiyorsa | `continue`, `change_assumption`, `cancel` |

Kimlikler İngilizce çünkü kodda karşılaştırılıyor; etiketler Türkçe çünkü
kullanıcı onları okuyor.

Onay geri çağrısı **async ve engelleyici**. Onay beklerken tur süresi bütçesi
işlemez, yoksa jüri düşünürken tur düşer.

---

## 9. Dosya sahipliği

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

## 10. A'dan B'ye teslim sırası

B'nin ihtiyaç sırası. Her biri gelene kadar `FakeLake` yerini tutuyor.

| # | A'nın parçası | B'nin ihtiyaç günü | Gelene kadar |
|---|---|---|---|
| 1 | `lake/gold.py::get()` ve `wide()`, senaryo serileri gold'da | 17 Eylül | `FakeLake` 66 aylık sentetik veri |
| 2 | `analysis/transforms.py` | 17–18 Eylül | `FakeLake.to_freq`; deflate taklit edilmiyor |
| 3 | `lake/catalog.py::search()` | 17 Eylül | anahtar kelime araması |
| 4 | `lake/prepare.py`, `lake/quality.py` | 18 Eylül | ince stub'lar |
| 5 | anomali ve kırılma sarmalayıcıları | 18 Eylül | — |
| 6 | `ingest()` canlı çekim | 19 Eylül | — |

**Kırmızı çizgi:** 17 Eylül akşamı gold yoksa en az beş senaryo serisi için
minimal bir gold üretilir. Demo sentetik veriyle yapılmayacak.

B'den A'ya tek teslim: `MiaLLM.embed()`, katalog gömme için. **Hazır**
(`llm.py`, ölçülen boyut 4096).
