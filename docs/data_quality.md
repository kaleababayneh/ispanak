# Veri Kalite Raporu

Otomatik üretildi: `python -m lakehouse_agent.lake.quality`. Kaynak: `data/gold/observations.parquet` + `data/gold/series_catalog.parquet`.

## Kaynak bazında özet

| Kaynak | Seri sayısı | Toplam gözlem | Toplam eksik | Toplam aykırı |
|---|---|---|---|---|
| BDDK-Aylik | 3 | 198 | 0 | 2 |
| evds | 6 | 682 | 0 | 7 |

## Seri detayları

| series_id | ad | frekans | ilk | son | n | eksik | tekrar | aykırı |
|---|---|---|---|---|---|---|---|---|
| bddk:aylik:tuketici:konut:TL:sektor | Tüketici Kredileri - Konut | AYLIK | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 0 |
| bddk:aylik:tuketici:konut_takip:TL:sektor | Takipteki Konut Kredileri | AYLIK | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 2 |
| bddk:aylik:tuketici:toplam:TL:sektor | Tüketici Kredileri (2+3+4) | AYLIK | 2021-01-01 | 2026-06-01 | 66 | 0 | 0 | 0 |
| evds:TP.AKONUTSAT1.KTRTOPLAM | Türkiye_Konut_Toplam Satışlar | AYLIK | 2021-01-01 | 2026-07-01 | 67 | 0 | 0 | 4 |
| evds:TP.AKONUTSAT2.KTRTOPLAM | Türkiye_Konut_İpotekli Satışlar | AYLIK | 2021-01-01 | 2026-07-01 | 67 | 0 | 0 | 3 |
| evds:TP.HPBITABLO6.3 | 1.1.1.Konut (TL + YP) | HAFTALIK(CUMA) | 2024-06-24 | 2026-08-31 | 115 | 0 | 0 | 0 |
| evds:TP.KFE.TR | Konut Fiyat Endeksi (KFE) | AYLIK | 2021-01-01 | 2026-08-01 | 68 | 0 | 0 | 0 |
| evds:TP.KTF12 | Konut Kredisi (TL, Akım, %) | HAFTALIK(CUMA) | 2020-12-28 | 2026-08-31 | 297 | 0 | 0 | 0 |
| evds:TP.TUKFIY2025.GENEL | Genel Endeks | AYLIK | 2021-01-01 | 2026-08-01 | 68 | 0 | 0 | 0 |
