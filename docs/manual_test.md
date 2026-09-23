# Elle test: örnek sorular ve beklenen cevaplar

Bu dosya demoyu elle denemek için. Sorular yapıştırılmaya hazır; beklenen
değerlerin **hepsi 23 Eylül 2026'da gold veriden, gerçek araçlarla ve model
olmadan hesaplandı** (`get_series`, `transform_table`, `detect_*`,
`test_causality`). Model cümleyi kendi kurar; cümle değişir, sayı değişmez.
Bir sayı tutmuyorsa önce adım kutusundaki araç özetine bak: araç doğru,
cevap yanlışsa sorun modelde; araç da farklıysa sorun veride ya da pencerede.

Nasıl koşulur:

```bash
uv run chainlit run app.py -w          # tarayıcı
uv run lakehouse-agent --real --chat   # terminal (soru + boş satır gönderir)
```

Sayı biçimi: aşağıda Türkçe yazıldı (`40,335`); model bazen `40.335` yazar,
ikisi de aynı sayıdır. Tablo ve grafikteki ham sütun adları (`TP_KTF12`)
modelin çalıştığı adlardır, değişmez; başlık ve lejant okunur etikettir.

## Her soruda bakılacaklar

- **Adımlar:** her araç çağrısı bir kutu; açınca girdi ve özet görünür.
- **Tablo:** tur bitince bir kez gelir, tam satır sayısıyla, sayfalanmış.
  Başlıklar okunur etiket, tarih `2021-01-01` (saat yok), sayılar en çok
  dört ondalık. Yalnızca tablo **ilk kurulduğunda** ve **sütun eklendiğinde**
  gelir; tabloyu yalnız okuyan bir takip sorusundan (`top`, "en yüksek üç
  ay") sonra tablo yeniden gelirse hata.
- **Grafik:** zaman serisi cevaplarında gelir. Lejant ve eksen başlığı
  okunur olmalı (`TP_KTF12` değil). Değer ekseninde `300k` **olmamalı**,
  `300.000` olmalı; birim eksen başlığında (`milyon TL`). Model grafik
  istemediyse arayüz **bu turda kurulan** tablodan kendisi çizer ve
  başlığa "(arayuz cizdi)" yazar; tablo üretmeyen bir turun altında
  (adres okuma, reddetme, yalnız web) grafik **olmamalı** — önceki sorunun
  tablosu çizilirse hata. Arayüzün çizdiği grafikte farklı birimler ayrı
  eksene gider; üçüncü bir birim çizilmez.
- **Kaynaklar:** seri kimlikleri (ve varsa adresler) listelenir.
- **Doğrulanamayan ifadeler:** çıkarsa cevaptaki o sayı araç çıktısında yok
  demektir. Bir-iki bayrak normal (yuvarlama, "%4,3" gibi web'den gelen
  sayı); her sayının bayraklanması değil.
- **Onay kartı:** çıkarsa düğmeye bas. Hangi soruda çıkacağı arama
  sonucuna bağlıdır, önceden söylenemez; kart yalnızca adı ve türü aynı iki
  seri arasında (`series_choice`), havuz dışı veri çekilirken (`fetch`) ve
  düşük kaliteli tabloda (`quality`) çıkar. Bastığın seçim
  `data/adhoc/sessions/<oturum>.jsonl` içinde `approval_decision` olayında
  `"automatic": false` olarak görünmeli.

---

## 1. Son 12 ay faiz

> Konut kredisi faiz oranı son 12 ayda nasıl seyretti?

Seri `evds:TP.KTF12` (Konut Kredisi (TL, Akım, %), haftalık). Araçlar:
`search_catalog`, `get_series`, `make_chart`.

Aylık ortalama (haftalık seri aya ortalamayla çevrilir; araç bunu uyarıda
söyler, cevap da söylemeli):

| Ay | % | Ay | % |
|---|---|---|---|
| 2025-07 | 42,5625 | 2026-01 | 36,238 |
| 2025-08 | 40,36 | 2026-02 | 35,155 |
| 2025-09 | 38,80 | **2026-03** | **34,655 (dip)** |
| 2025-10 | 37,94 | 2026-04 | 36,615 |
| 2025-11 | 37,7625 | 2026-05 | 38,426 |
| 2025-12 | 37,29 | **2026-06** | **40,335 (son)** |

Beklenen cümle: 2026-03'e kadar düşüş, sonra üç ayda 40,3'e yükseliş.
Model "son 12 ay"ı bugünden (2026-09) sayarsa 2026-07/08 boş gelir; bu
hata değildir, veri 2026-06'da bitiyor. Yanlış olan: "ağırlıklı ortalama"
kavramının hiç geçmemesi ya da haftalık değerlerin toplanması.

## 2. Endeksleme

> Konut fiyat endeksini 2021-01 = 100 olacak şekilde endeksle.

Seri `evds:TP.KFE.TR`. Araçlar: `get_series`, `transform_table (rebase)`.
Yeni sütun `TP_KFE_TR_endeks_202101`; etiketi "Konut Fiyat Endeksi (KFE) —
endeks (2021-01 = 100)".

| Dönem | Ham KFE | Endeks |
|---|---|---|
| 2021-01 | 16,49 | **100** |
| 2021-02 | 16,98 | 102,97 |
| 2026-05 | 226,99 | 1.376,53 |
| 2026-06 | 231,34 | **1.402,91** |

## 3. İpotekli satış payı

> İpotekli konut satışlarının toplam konut satışları içindeki payı nedir?

Seriler `evds:TP.AKONUTSAT2.KTRTOPLAM` (ipotekli) / `evds:TP.AKONUTSAT1.KTRTOPLAM`
(toplam). Araçlar: `get_series`, `transform_table (ratio)`. Sütun
`TP_AKONUTSAT2_KTRTOPLAM_pay`.

- 2026-06: **%20,0** (19,998)
- 2021-01: %15,3
- En düşük **2023-12: %4,4**; en yüksek **2022-06: %26,7**

Yanlış olan: payın oranı yerine sayının kendisini vermek, ya da yüzdeyi
zihinden hesaplamak.

## 4. Reel bakiye

> Konut kredisi bakiyesini enflasyondan arındır, reel değeri göster.

Bakiye `bddk:aylik:tuketici:tuketici_kredileri_konut:toplam:sektor`
(milyon TL), deflatör `evds:TP.TUKFIY2025.GENEL` (araç kendisi çeker).
Araçlar: `get_series`, `transform_table (deflate)`.

| Dönem | Nominal (milyon TL) | Reel, 2025 fiyatlarıyla |
|---|---|---|
| 2021-01 | 276.785 | **1.716.482** |
| 2026-06 | 801.438 | **616.538** |

Cevapta "reel" ve baz ("2025 yılı ortalama fiyatlarıyla", "TÜFE 2025=100")
geçmeli. Nominal artarken reelin **düşmesi** doğru sonuçtur. Model bakiye
için başka bir seri (`evds:TP.KM.B11`, bin TL) seçerse sayılar farklı olur
ama yön aynıdır; araç faiz ya da endeksi reelleştirmeyi reddeder.

## 5. Haftalıktan aylığa

> Konut kredisi faizi haftalık yayınlanıyor, aylık ortalamasını ver.

Seri `evds:TP.KTF12`. 2026-06'nın dört haftası: 39,33 / 39,72 / 40,73 /
41,56 → **ortalama 40,335**. Toplam 161,34 çıkarsa yanlış toplulaştırma
(oran toplanmaz). 66 aylık tablo (2021-01 – 2026-06).

## 6. TÜFE artışı

> Tüketici fiyat endeksi 2021 başından bugüne yüzde kaç arttı?

Seri `evds:TP.TUKFIY2025.GENEL` (2025=100, zincirleme yok). Araçlar:
`get_series`, `transform_table (change_since 2021-01)`.

- 2021-01: 16,125 → 2026-06: 129,99 → **%706,1**

`TP.GENENDEKS.T1` gibi başka bir TÜFE üzerinden benzer bir yüzde gelirse
seri yanlıştır; kaynaklarda `TP.TUKFIY2025.GENEL` olmalı.

## 7. Veri var mı (reddetme)

> Konut kredisi bakiyesinin 2021 verisi var mı?

Doğru cevap **"evet"**: BDDK aylık toplam konut bakiyesi 2021-01'den
(276.785 milyon TL) itibaren dolu. EVDS haftalık `TP.HPBITABLO6.3` ise
**2024-06-24**'ten önce boştur. Yanlış olan: HPBITABLO6.3 için 2021'e
değer uydurmak. Yalnızca `search_catalog` çağrılması yeterlidir.

## 8. Kırılma

> Konut kredisi faizinde yapısal bir kırılma var mı, varsa ne zaman?

Seri `evds:TP.KTF12`. Araçlar: `get_series`, `detect_changes`; kural 19
gereği model **kendiliğinden** `web_search` de yapmalı.

- Kırılma **2023-07** (aylıkta 2023-07-01; haftalık çekildiyse 2023-07-10
  haftası — ikisi de doğru, tarih Temmuz 2023)
- Aylık: öncesi ortalama 18,72 → sonrası 40,25 (**%115**), sıçrama %51,4
- Web bağlamı: TCMB 20 Temmuz 2023, politika faizi %15 → %17,5 (sayı
  web'den değil, olay web'den; cevap kaynak adresi vermeli)

## 9. Aykırı aylar

> Konut fiyat endeksinde aykırı aylar var mı?

Seri `evds:TP.KFE.TR`. Araçlar: `get_series`, `detect_anomalies`.
Tam iki ay, ikisi de beklenenin **üstünde**:

| Ay | Gözlenen | Beklenen | z |
|---|---|---|---|
| 2023-09 | 116,18 | 111,15 | 3,71 |
| 2023-10 | 119,18 | 114,18 | 3,68 |

## 10. Öncüllük (Granger)

> Konut kredisi faizi konut fiyat endeksine öncül mü? Granger ile test et.

Seriler `TP.KTF12` ve `TP.KFE.TR`. Araçlar: `get_series`, `test_causality`.

- Faiz → KFE: en iyi gecikme 3, p = 0,0495, **anlamlı değil** (Bonferroni
  eşiği 0,0042; 6 gecikme × 2 yön)
- KFE → Faiz: gecikme 4, p = 0,168, anlamlı değil
- En güçlü gecikmeli korelasyon r = 0,25 (zayıf)

Doğru cevap **"hayır, bu veriyle gösterilemedi"**. p = 0,0495'i "0,05'in
altında, anlamlı" diye okumak yanlıştır; araç `anlamli = false` diyor.

## 11. Şartnamenin demo sorusu

> 2021-2025 arası konut kredisi ve faiz oranlarını göster. Faiz düştüğü halde
> kredinin artmadığı dönem oldu mu?

Seriler `TP.KTF12` + BDDK toplam konut bakiyesi. Araçlar: `get_series`, iki
`transform_table (diff)`, `transform_table (mark)`, `make_chart`.

Araç özeti birebir: *"4 ay koşulların hepsini sağlıyor: 2021-02, 2021-05,
2023-12, 2024-01."*

| Ay | Faiz (%) | Faiz farkı | Bakiye (milyon TL) | Bakiye farkı |
|---|---|---|---|---|
| 2021-02 | 17,985 | −0,403 | 276.263 | −522 |
| 2021-05 | 17,835 | −0,299 | 275.130 | −1.196 |
| 2023-12 | 42,088 | −0,062 | 438.222 | −2.190 |
| 2024-01 | 41,578 | −0,511 | 436.688 | −1.534 |

Bağlam: 60 ayda faiz 32 ay düştü, bakiye 8 ay artmadı; ikisinin kesişimi
bu dört ay. Faiz 18,39 → 37,29; bakiye 276.785 → 678.970 milyon TL. Model
ayları kendisi sayarsa (araç çağırmadan) yanlıştır; `mark` çağrısı
adımlarda görünmeli. Bu soruda onay kartı çıkabilir (bakiye için ikiz
seriler); "toplam" olanı seç.

## 12. Grafik

> Konut kredisi faizinin son 24 ayını grafikle göster.

Araçlar: `search_catalog`, `get_series`, `make_chart`. Bakılacak: lejant
"Konut kredisi faiz oranı" gibi okunur bir ad (modelin yazdığı; boş
bırakırsa katalog adı "Konut Kredisi (TL, Akım, %)"), eksen sayıları tam,
x ekseni "Dönem". 2024-07 – 2026-06; dip 2026-03 (34,655), son 40,335.

## 13. Adres okuma

> https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/main+menu/duyurular/basin/2023/duy2023-25 adresini oku ve o gün alınan kararı özetle.

Araç: `read_url`. Cevap: 20 Temmuz 2023, PPK, politika faizi **%15 → %17,5**.
Sayfada tablo yok, bu yüzden **tablo ve grafik gelmemeli**; önceki bir
sorudan kalan tablonun grafiği gelirse hata (23 Eylül'de görüldü ve
kapatıldı). Dış siteye bağlı; sayfa açılmazsa model aramaya düşer ve
sayıyı yazmaz — bu doğru davranıştır (aramadan sayı alınmaz), önce adresi
kontrol et.

## 14. Web bağlamı

> Konut kredisi faizindeki kırılmanın olduğu tarihte ne olduğuna web'den bak.

Araçlar: `get_series`, `detect_changes`, `web_search`. Kaynaklarda en az
bir adres olmalı. Cevaptaki sayılar (kırılma tarihi, %115) havuzdan; web
yalnızca olayı verir (TCMB faiz artırımı). Web'den alınmış bir sayı
doğrulayıcıda bayraklanır; bu beklenen bir bayraktır.

---

## Takip sorusu denemesi

Soru 11'den sonra aynı oturumda:

> Aynı tabloda faizin en yüksek olduğu üç ayı ver.

`transform_table (top, n=3)` çağrılmalı; yeni tablo kurulmaz, sütun
eklenmez, araç dönemleri listeler. **Tablo ve grafik yeniden gelmemeli**
(23 Eylül'de geldi, kapatıldı); cevap yalnız metin. Araç özeti birebir: *"'TP_KTF12'
sutununda en yuksek 3 satir: 2024-04: 44.72, 2024-05: 44.694, 2024-06:
44.4825."* Model `get_series`'i yeniden çağırırsa takip sorusu aktif
tablo üzerinde çalışmamış demektir (kural 6).

## Nelerin değişmesi normal, nelerin değil

Değişebilir: cümle kurgusu, hangi ayların örnek verildiği, arama sayısı,
pencere ("son 12 ay"ın nereden sayıldığı), lejant etiketinin sözcükleri.

Değişmemeli: yukarıdaki sayılar ve tarihler, seçilen seri kimlikleri,
araçların çağrılmış olması (özellikle `mark`, `detect_changes`,
`test_causality`, `make_chart`), "anlamlı değil" hükmü, reel serinin yönü.
