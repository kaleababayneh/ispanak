# Veri Kalite Raporu

Otomatik üretildi: `python -m lakehouse_agent.lake.quality`. Kaynak: `data/gold/observations.parquet` + `data/gold/series_catalog.parquet`.

Silver aşamasında birleştirilen mükerrer (aynı seri+dönem) satır sayısı: 8.

## Kaynak bazında özet

| Kaynak | Seri sayısı | Toplam gözlem | Toplam eksik | Toplam aykırı |
|---|---|---|---|---|
| BDDK-Aylik | 3 | 198 | 0 | 2 |
| EVDS | 6 | 658 | 0 | 7 |

## Seri detayları

| series_id | ad | frekans | ilk | son | n | eksik | tekrar | aykırı |
|---|---|---|---|---|---|---|---|---|
| bddk:aylik:tuketici:konut:TL:sektor | Tüketici Kredileri - Konut | M | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 0 |
| bddk:aylik:tuketici:konut_takip:TL:sektor | Takipteki Konut Kredileri | M | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 2 |
| bddk:aylik:tuketici:toplam:TL:sektor | Tüketici Kredileri (2+3+4) | M | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 0 |
| evds:TP.AKONUTSAT1.KTRTOPLAM | Türkiye_Konut_Toplam Satışlar | M | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 4 |
| evds:TP.AKONUTSAT2.KTRTOPLAM | Türkiye_Konut_İpotekli Satışlar | M | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 3 |
| evds:TP.HPBITABLO6.3 | 1.1.1.Konut (TL + YP) | W | 2024-06-24 | 2026-06-29 | 106 | 0 | 0 | 0 |
| evds:TP.KFE.TR | Konut Fiyat Endeksi (KFE) | M | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 0 |
| evds:TP.KTF12 | Konut Kredisi (TL, Akım, %) | W | 2020-12-28 | 2026-06-29 | 288 | 0 | 0 | 0 |
| evds:TP.TUKFIY2025.GENEL | Genel Endeks | M | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 0 |
