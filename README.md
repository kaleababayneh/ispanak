# lakehouse-agent

KKB Hackathon 2026 için geliştirilen bu proje, EVDS (TCMB) ve BDDK verisini
2021-01 – 2026-06 penceresinde ham halinden (bronze) geçerek temizlenmiş,
sınıflandırılmış tek bir gold havuzuna (`observations.parquet` +
`series_catalog.parquet`) indirir; bu havuzu DuckDB ve LanceDB üzerinden
sorgulanabilir hale getirir; ve havuzu, kendi yazdığımız plan/act/observe/
answer/verify döngüsüne sahip, sekiz araçlı bir agent'a sunar — sorulan bir
finansal/ekonomik soruyu havuzdaki serilerle yanıtlar, gerekirse web'e çıkar,
her adımda kaynağını gösterir ve kendi cevabını doğrular.

## Kurulum ve demo

```bash
uv sync
```

`.env` dosyasına `EVDS_API_KEY` ve `MIA_API_KEY` konur (bkz. `.env.example`);
anahtarlar yalnızca `config.py` içinde `pydantic-settings` ile ortamdan okunur,
hiçbir örnek komuta anahtar yazılmaz.

Demoyu çalıştıran komut (gerçek model + gerçek araç + gerçek gold veri):

```bash
uv run python -m lakehouse_agent.agent.loop --real --question "Konut kredisi faizleri son bir yılda nasıl değişti?"
```

Agent'ın ağsız/çevrimdışı smoke testi için `--fake`, modeli veri havuzundan
izole test etmek için sentetik veriyle `--live` modu da var (bkz.
`docs/architecture.md`).

## Mimari özeti

Veri; `ingest` (yeniden çalıştırılabilir HTTP çekimi) → `bronze` (değişmez ham
kayıt) → `silver` (temizleme, birleştirme) → `gold` (tek şemalı gözlem +
katalog tabloları) katmanlarından geçer. Gold üzerine iki ayrı erişim yüzeyi
kurulur: agent'ın arama yaptığı `data/pool_index` (LanceDB) ve salt-okunur
`catalog.duckdb`. Agent döngüsü bu yüzeyleri sekiz araç üzerinden kullanır.
Detaylı katman anlatımı ve mermaid diyagramı için
[docs/architecture.md](docs/architecture.md).

## Veri kapsamı

- **45.422 seri, 12.072.279 gözlem satırı**
- Dağılım: EVDS 36.642 · BDDK aylık 2.045 · BDDK haftalık 535 · FinTürk 6.200
- Pencere: **2021-01-01 – 2026-06-30**, `obs_date` üzerinden kesilir
- Boyut: `observations.parquet` 33 MB (zstd); gold kurulumu ~65 sn

## Sekiz araç

| Araç | Ne yapar |
|---|---|
| `search_catalog` | Havuzdaki serileri doğal dille arar, aday `series_id`'leri skorlarıyla döner. |
| `get_series` | Bir veya birden çok seriyi seçilen tarih aralığında tek tabloya getirir, gerekirse frekans dönüştürür. |
| `transform_table` | Var olan bir tabloya türetilmiş sütun ekler (fark, yüzde değişim, endeks, deflate, top-n, vb.). |
| `detect_anomalies` | STL + MAD ile nokta anomalisi bulur; 24 dönemden kısa seride IQR kuralına düşer. |
| `detect_changes` | PELT algoritmasıyla yapısal kırılma (seviye/eğim değişimi) noktası bulur. |
| `test_causality` | İki seri arasında öncüllüğü Granger testi + gecikmeli korelasyonla ölçer; nedensellik iddia etmez. |
| `read_url` | Bir web adresini (PDF/Excel/CSV/JSON/sayfa) okur, tabloyu havuzla aynı temizlikten geçirip oturuma ekler. |
| `web_search` | Açık webde arama yapar; sayı üretmez, yalnızca bağlam ve kaynak döner. |

`read_url` ve `web_search` havuz dışına çıktığı için onay kapısından geçer.

## Güven katmanı

- **Üç onay noktası**: `fetch` (havuz dışına çıkmadan önce), `series_choice`
  (en iyi iki aday skoru birbirine yakınsa), `quality` (yeni tabloda eksik
  oran, tekrarlı dönem veya bilinmeyen birim varsa).
- **Kural tabanlı doğrulayıcı**: her cevaptan sonra sayıların oturum
  tablolarında karşılığı var mı, kaynaklar listelenmiş mi, kullanılan
  `read_url`/`web_search` adresi cevapta geçiyor mu, tarihler kapsam
  içinde mi kontrol edilir; bulgular cevabın altına görünür şekilde eklenir.
- **Kaynak gösterimi**: cevap metninde model kendi kaynağını alıntılar; tur
  sonunda oturumun bütün kaynakları tam adresleriyle ayrıca özetlenir.

## Bilinen sınırlar

- `run_sql` aracı test edilmiş ve hazır ama henüz canlı sekiz araç kaydına
  bağlanmadı (yalnızca sahte araç setinde) — bkz. `docs/contracts.md` §7.
- STL, 24 dönemden kısa seride çalışmaz; bu durumda IQR'a düşülür.
- `docs/contracts.md` §9'daki dosya sahipliği tablosu tarihsel; projenin tek
  geliştiricisi var, repo'nun her yoluna yazılabilir (bkz. `CLAUDE.md`).

Daha fazla detay için:
- [docs/architecture.md](docs/architecture.md) — katman katman mimari + diyagram
- [docs/data_dictionary.md](docs/data_dictionary.md) — bronze/silver/gold dosya
  tanımları, havuz ölçümü, EVDS/BDDK kaynak detayları
- [docs/contracts.md](docs/contracts.md) — şemalar, ortak tipler, `ToolResult`,
  oturum durumu, olay listesi, onay noktaları
- [docs/decisions.md](docs/decisions.md) — her kararın gerekçesi ve ölçümü

## Test ve kalite

```bash
uv run pytest -q          # 960 test, ağ erişimi olmadan yeşil
uv run ruff check --fix .
uv run ruff format .
```
