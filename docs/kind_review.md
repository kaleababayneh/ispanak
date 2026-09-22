# Tür (kind) gözden geçirme listesi

Otomatik üretildi: `python -m lakehouse_agent.lake.gold`. Kurallar:
`lake/classify.py` (EVDS), `ingest/bddk_tables.yaml` + `lake/bddk.py` (BDDK).
Tür yalnızca yorumu belirler; frekans dönüşümünü `agg_method` belirler.

## Kaynak ve tür

| Kaynak | stock | flow | rate | index | boş |
|---|---|---|---|---|---|
| BDDK-Aylik | 1848 | 159 | 30 | 0 | 8 |
| BDDK-FinTurk | 4509 | 0 | 1286 | 0 | 405 |
| BDDK-Haftalik | 535 | 0 | 0 | 0 | 0 |
| EVDS | 21140 | 4813 | 6732 | 1663 | 2294 |

## EVDS: hangi kural

| Gerekçe | Seri |
|---|---|
| EVDS donem sonu degeri aliyor (BITIS) | 21139 |
| birim oran/yuzde | 5063 |
| EVDS toplami aliyor (KUMULATIF) | 4811 |
| belirsiz: oran/endeks isareti yok, EVDS ortalama/uc deger aliyor | 2294 |
| ad oran/yuzde | 1668 |
| birim endeks | 1443 |
| ad endeks | 218 |
| elle eslendi (SCENARIO_SERIES) | 6 |

## EVDS: türü boş kalan 2294 seri, veri grubuna göre

Oran ya da endeks işareti yok ve EVDS ortalama/uç değer alıyor: bir
ortalama sayı, bir TL/m² fiyatı ne bakiye ne oran. Tür uydurulmadı;
`deflate` bu serileri reddeder, `diff` birimini değiştirmez.

| Veri grubu | Seri | Birimler | Örnek |
|---|---|---|---|
| bie_ackap2 | 18 | Adet | 1.Açılan Anonim Şirket (Adet) |
| bie_birimfiyat | 77 | TL/m2 | Türkiye Konut Birim Fiyatları |
| bie_bk | 75 | TL/m2 | Türkiye Konut Birim Kiraları |
| bie_cli2 | 2 | - | MBONCU_SUE (Trend Kapsayan) |
| bie_dbfy | 79 | TL/m2 | Türkiye Dükkan Birim Fiyatları |
| bie_dfe | 3 | - | TR 10 (İstanbul) |
| bie_dkdovizgn | 74 | - | (USD) ABD doları (Döviz Alış) (Arşiv) |
| bie_dkdovytl | 70 | Türk lirası | (USD)  ABD  Doları  (Döviz  Alış) |
| bie_dkefektif | 46 | - | (USD) ABD doları (Efektif Alış) (Arşiv) |
| bie_dkefkytl | 44 | Türk lirası | (USD) ABD Doları (Efektif Alış) |
| bie_dkkurbil | 52 | - | (AED) Birleşik Arap Emirlikleri Dirhemi |
| bie_elektuketim | 8 | MWh | Üretim |
| bie_feoktg | 27 | - | A.Mevsimlik ürünler hariç TÜFE (Arşiv) |
| bie_fhge | 12 | - | 1.1. İş Durumu (Son 3 ay)-Genel |
| bie_goldimprt | 1 | - | Kıymetli Madenler Piyasası Altın İthalat Verileri (BIST) (Kg) |
| bie_hanebek | 25 | Yüzde, TL | SORU_15. Son bir yıl içinde fiyatlarının en çok arttığını düşündüğünüz iki grubu seçiniz. (GIDA) |
| bie_hanebekdemog | 312 | Yüzde, TL | Katılımcı Sayısı |
| bie_itouge85 | 8 | - | 2.GIDA |
| bie_kfe | 19 | - | TR10 (İstanbul) |
| bie_mkaltytl | 3 | - | Cumhuriyet Altını Satış Fiyatı (TL/Adet) |
| bie_obfy | 31 | TL/m2 | Türkiye Ofis Birim Fiyatları |
| bie_ofe | 3 | - | TR 10 (İstanbul) |
| bie_oktug2025 | 27 | - | A. Mevsimlik ürünler hariç TÜFE |
| bie_onrepofbap | 1 | - | İşlem Hacmi |
| bie_pydovef | 8 | - | DÖVİZ/EFEKTİF Adet (İşlem hacimleri ve adetleri çift taraflıdır) |
| bie_pyrepo | 10 | - | (1 GÜN)Ağırlıklı Ortalama Faiz (REPO) |
| bie_rkdrk | 1 | milyon ABD doları | Reeskont Kredilerinin TCMB Döviz Rezervlerine Katkısı (Milyon $) |
| bie_rkgema | 8 | - | 1.1. Toplam Sipariş-MA (Mevcut Durum) |
| bie_rkgey2 | 8 | - | 1.1. Toplam Sipariş (Mevcut Durum) |
| bie_sanayrev11 | 41 | - | Toplam Sanayi (Arşiv) |
| bie_sanayrev4 | 41 | - | Toplam Sanayi (Arşiv) |
| bie_sgegi | 6 | milyon ABD doları | 1.Toplam Seyahat Geliri |
| bie_tldov | 16 | milyon ABD doları | 1. Yurt İçi Interbank (Vadeli) |
| bie_tukfiy2025 | 348 | - | 01. Gıda Ve Alkolsüz İçecekler |
| bie_tukfiy4 | 56 | - | GENEL (Tüketici) (Arşiv) (Arşiv) |
| bie_tuksehir2 | 27 | - | TÜRKİYE (Arşiv) |
| bie_turizmgelgit | 2 | Bin ABD doları | Turizm Geliri |
| bie_turizmyzs | 2 | Kişi | Çıkış Yapan Ziyaretçi Sayısı |
| bie_turizmyzsy | 32 | Kişi | Almanya |
| bie_turizmzyvs | 2 | Kişi | Yurtdışını Ziyaret Eden Vatandaş Sayısı |
| bie_tuufey | 4 | - | Tarım, Ormancılık ve Balıkçılık (Arşiv) |
| bie_undnakliyeroro | 12 | Adet | 1.Türk Araçlar İhracat ( Kara ) |
| bie_undnakliyetbg | 66 | Adet | 1.Afganistan (TBG) |
| bie_undnakliyetih | 70 | Adet | 1.Afganistan (TIH) |
| bie_undnakliyetit | 72 | Adet | 1.Afganistan (TIT) |
| bie_undnakliyeybg | 66 | Adet | 1.Afganistan (YBG) |
| bie_undnakliyeyih | 68 | Adet | 1.Afganistan (YIH) |
| bie_undnakliyeyit | 66 | Adet | 1.Afganistan (YIT) |
| bie_uraltin | 1 | - | Maden Firmalarının Altın Üretim Bildirimleri (BIST) (Kg) |
| bie_uristsayil | 82 | Kişi | Toplam (Arşiv) |
| bie_urkisi15 | 37 | - | TS-Toplam Sanayi (Arşiv) |
| bie_urkisi21 | 37 | - | TS. Toplam Sanayi |
| bie_uroto | 8 | Adet | 1. Binek |
| bie_ursaat15 | 37 | - | TS-Toplam Sanayi (Arşiv) |
| bie_ursaat21 | 37 | - | TS. Toplam Sanayi |
| bie_yssk | 6 | ABD doları | 1-2 Yıl (Arşiv) |

## BDDK: türü boş kalan 413 seri

Rasyo tablolarında birimi yüzde olmayan oranlar (personel başına aktif,
ortalama vade) ve FinTürk'te kişi başı / şube başına düşen değerler.

