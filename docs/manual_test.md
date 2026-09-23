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

- **Adımlar:** turun bütün araç çağrıları "Ajan adımları (N)" adlı tek
  katlı kartta; açınca her adım araç adı ve süresiyle (`✓ get_series ·
  164 ms`), içinde girdi ve özet. Soru ile cevap arasında adım kartı
  yığını görünüyorsa hata.
- **Tablo:** tur bitince bir kez gelir, tam satır sayısıyla, sayfalanmış.
  Başlığı ne olduğunu söyler: "Seri tablosu — …", "Kırılma analizi — Konut
  kredisi faiz oranı", "Granger öncüllük testi — X ve Y"; altında `t02_…`
  kimliği. Yalnız `t02_detect_changes` yazan bir tablo hata.
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
- **Kaynaklar:** turun **en altında**, "Kaynaklar (N)" adlı katlı kartta;
  açınca seri kimlikleri ve varsa adresler, her biri bir kez. Cevabın
  üstünde ya da birden çok kaynak bloğu görünürse hata.
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

## 15. JavaScript ile kurulan sayfa (tarayıcı katmanı)

> https://quotes.toscrape.com/js/ adresini oku ve sayfadaki ilk üç alıntıyı yazarlarıyla listele.

Finans sayfası değil, bilerek: `read_url`'ün üçüncü katmanını kanıtlamak için
kullanılan sayfa (Burak'ın canlı testiyle aynı). Ölçüldü (23 Eyl): statik
HTML'de tablo yok, metin 96 karakter → `needs_browser_render = True`; alıntılar
sayfanın kendi JavaScript'inde. Havuzda karşılığı olmadığı için `fetch` onay
kartı çıkabilir; onayla.

İki aşamada test edilir:

1. **Tarayıcı katmanı kurulu değilken** (varsayılan kurulum): `read_url` adım
   özetinde *"Sayfa JavaScript gerektiriyor, tarayici katmani kurulu degil
   (uv sync --extra browser && uv run playwright install chromium)"* notu
   olmalı; cevap sayfayı okuyamadığını söylemeli, alıntı **uydurmamalı**.
2. **Kurduktan sonra** (`uv sync --extra browser && uv run playwright install
   chromium`): adım özetinde *"Sayfa JavaScript ile render edildi (headless
   tarayici)"* notu; cevapta ilk üç alıntı sırayla:

| # | Yazar | Alıntının başı |
|---|---|---|
| 1 | Albert Einstein | "The world as we have created it is a process of our thinking…" |
| 2 | J.K. Rowling | "It is our choices, Harry, that show what we truly are…" |
| 3 | Albert Einstein | "There are only two ways to live your life…" |

Render 30 saniyeyi (sert sınır) aşarsa araç *"saniyede bitmedi, surec
sonlandirildi"* der ve statik sonuca düşer; bu çökme değil, sınırın çalışması.
Karşı kontrol: 13. soru (TCMB duyurusu) statik metin taşıdığı için tarayıcıyı
**çağırmamalı** — adımlarda render notu görünürse hata.

---

## Şartnamenin adres soruları (Borsa İstanbul)

Şartname demo günü prompt'ta adres verileceğini söylüyor: kıymetli madenler
piyasası sayfasındaki **Altın İşlemleri PDF'i** ve **`endeks/xtumy`**
sayfası (BIST 100 günlük/aylık değişim). Aşağıdaki değerler 23 Eylül'de
`read_url` ve havuz araçlarıyla ölçüldü. **PDF her ay güncellenir**: 2026
tablosunda Ocak–Ağustos dolu, Eylül ve sonrası boş; Ekim'de Eylül satırı
dolar ve "en yüksek ay" değişebilir. Değerler tutmuyorsa önce PDF'i açıp
bakın. Her adres okuma bir **`fetch` onay kartı** çıkarır ("… üzerinden
veri çekilsin mi?"); onaylayın.

### A1. Sayfadan PDF'e (iki adımlı okuma)

> https://www.borsaistanbul.com/veriler/kiymetli-madenler-ve-kiymetli-taslar-piyasasi/piyasa-verileri
> adresindeki Altın İşlemleri PDF'ini oku. 2026'da TL cinsinden altın işlem
> hacminin en yüksek olduğu ayı ve yılbaşından bu yana toplam işlem
> miktarını (kg) söyle.

Beklenen akış: ilk `read_url` sayfada tablo bulamaz, **15 veri dosyası**
listeler ("Altın İşlemleri: …/kmp_au.pdf"); ikinci `read_url` PDF'i açar
(planner kural 17). Tablo `t01_read_url`: 13 satır (12 ay + TOPLAM) × 11
sütun; başlıklar kelime kelime onarılmış ("Hacim/Volume (TL)").

- TL hacmi en yüksek ay **Mart 2026: 151.349.741.325 TL** (≈151,3 milyar)
- Yılbaşından bu yana toplam miktar **258.809 kg** (TOPLAM satırı; TL
  91.977 + USD 165.807 + EUR 1.025)
- Kaynak: PDF adresi ve "KIYMETLİ MADENLER PİYASASI ALTIN İŞLEM VERİLERİ
  (2026)" başlığı listelenmeli

Yanlış olan: TOPLAM satırını bir ay sanmak; boş Eylül–Aralık satırlarını
sıfır saymak; **"yalnızca Ocak–Mart dolu" demek** (23 Eylül'de dedi: `top
n=3` TOPLAM, Mart, Ocak döndürmüştü). Artık `top` toplam satırını
sıralamaya almaz ve "8 dolu satır arasından" der; cevapta dolu ay sayısı
geçerse 8 olmalı. Grafik çizilirse **TOPLAM çubuğu olmamalı**. Tablo
başlığı "Adresten okunan tablo — kmp_au.pdf" olmalı. Tabloda **"Sayısı/
Trans" sütunları dolu** olmalı (Ocak 1.017, Mart 2.922 işlem); 23 Eylül'de
adında nokta olan üç sütun boş göründü, başlıktaki nokta gösterimde
değiştirildi — boş kalırsa bana söyleyin, mekanizma doğrulanmadı.

### A2. Doğrudan PDF, karşılaştırma

> https://www.borsaistanbul.com/dosyalar/kmtp/veriler/kmp_au.pdf adresini
> oku. 2026'nın ilk üç ayında TL cinsinden işlem hacmi nasıl değişti ve
> altın kilogram bazında TL ile mi USD ile mi daha çok işlem gördü?

| Ay | Hacim (TL) | TL miktar (kg) | USD miktar (kg) |
|---|---|---|---|
| Ocak | 93.824.682.381 | 13.837 | 19.747 |
| Şubat | 90.534.914.196 | 12.649 | 20.070 |
| Mart | 151.349.741.325 | 22.265 | 32.742 |

Beklenen: Şubat'ta hafif düşüş, Mart'ta yaklaşık %67 sıçrama (yüzdeyi
model hesaplamamalı; `transform_table` ya da "yaklaşık 1,7 kat" gibi
tablodan okunur bir ifade). Kilogram bazında **her ay USD** işlemleri
önde (2026'nın sekiz dolu ayının hepsinde). Toplam işlem sayısı Ocak 1.017,
Mart 2.922.

### A3. XTUMY sayfası + havuz (tuzak)

> https://www.borsaistanbul.com/endeks/xtumy adresini oku ve BIST 100
> endeksinin 2026'daki aylık değişimini ver.

Sayfa **XTUMY = BIST TÜM-100** endeksinin künyesidir (Endeks Kodu, ISIN
`TRAIMKB01380`, Endeks Tipi…); fiyat tablosu JavaScript ile dolduğu için
`read_url` yalnız 17 satırlık künye tablosunu getirir, fiyat gelmez.
Doğru davranış: bunu söylemek ve BIST 100'ü **havuzdan** çekmek —
`evds:TP.MK.F.BILESIK` (XU100 kapanış, günlük; aylığa ayın son değeriyle
çevrilir, araç bunu uyarıda yazar).

| Ay | Kapanış | Aylık değişim |
|---|---|---|
| 2026-01 | 13.838,29 | — |
| 2026-02 | 13.717,81 | −0,87 % |
| 2026-03 | 12.790,98 | −6,76 % |
| 2026-04 | 14.442,56 | **+12,91 %** |
| 2026-05 | 13.662,75 | −5,40 % |
| 2026-06 | 14.121,83 | +3,36 % |

Yanlış olan: XTUMY künyesini BIST 100 sanmak; sayfadan fiyat "okumak"
(orada yok); değişimi zihinden hesaplamak.

### A4. Günlük değişim (frekans)

> BIST 100 endeksi Haziran 2026'da günlük bazda en çok hangi gün yükseldi,
> hangi gün düştü?

`get_series` **`freq="D"`** ile çekilmeli (varsayılan aylık bunu
cevaplayamaz), sonra `transform_table (pct_change)` ve `top`.

- En büyük yükseliş **2026-06-02: +3,62 %** (14.200,20)
- En büyük düşüş **2026-06-03: −1,65 %** (13.965,65)
- Ay sonu 2026-06-30: 14.121,83 (−0,43 %)

Hafta sonları boş satırdır; "eksik veri" diye bayraklanmamalı.

---

## Uzun sorular (2-3 cümle, birden çok araç)

Demo günü sorular kısa olmayabilir. Aşağıdakiler birkaç aracı sırayla
zorlar; beklenen değerler gerçek araçlarla ölçüldü.

### U1. Kırılma ve sonrası

> 2021'den bu yana konut kredisi faiz oranı ile konut fiyat endeksini aynı
> tabloya getir. Faizde yapısal kırılmanın olduğu tarihi bul ve o tarihten sonra
> konut fiyat endeksinin nasıl seyrettiğini anlat. Sonucu grafikle göster.

Araçlar: `get_series` (iki seri, tek tablo), `detect_changes`,
`transform_table` (opsiyonel), `make_chart`; kural 19 gereği `web_search` de
gelebilir.

- Kırılma **2023-07**
- KFE 2023-07: **101,87** → 2026-06: **231,34** (**%127,1** artış)
- Kırılma öncesi KFE ortalaması 44,72; sonrası 165,38
- Grafikte faiz ve endeks **ayrı eksende** olmalı (biri %, biri endeks)

### U2. Reel bakiye, endeksli

> Konut kredisi bakiyesini 2021 başından itibaren getir, TÜFE ile reel hale
> getir ve reel seriyi 2021-01 = 100 olacak şekilde endeksle. Reel bakiye 2026
> ortasında 2021 başına göre yüzde kaç değişmiş?

Araçlar: `get_series`, `transform_table (deflate)`, `transform_table
(rebase)`, `transform_table (change_since)`; yüzdeyi **araç** vermeli.

- Reel düzey **baza bağlı**, ikisi de doğru: model `base_period` vermezse
  2025 ortalama fiyatlarıyla 2021-01 **1.716.482** → 2026-06 **616.538**;
  "2021-01 = 100" dendiği için 2021-01 fiyatlarını seçerse (23 Eylül'de
  seçti) 2021-01 **276.785** (nominale eşit) → 2026-06 **99.417,6**
- Endeks ve yüzde bazdan bağımsız: endeks 2026-06 **35,9**; değişim
  **−%64,1** (ölçülen: 35,9187 / −64,0813)
- Reel serinin dibi **2024-09**
- Cevapta "reel" ve hangi baz olduğu geçmeli; nominal artarken reelin
  düştüğü söylenmeli

### U3. Pay ve o aylardaki faiz

> İpotekli konut satışlarının toplam satışlar içindeki payını 2021'den bu yana
> hesapla. Payın en düşük ve en yüksek olduğu ayları ver ve o aylarda konut
> kredisi faizinin ne düzeyde olduğunu söyle.

Araçlar: `get_series` (üç seri tek tabloda), `transform_table (ratio)`,
`transform_table (top, ascending=true ve false)` — ayları model saymamalı.

**İkiz seri tuzağı (23 Eylül'de ölçüldü):** "konut kredisi faiz oranı"
araması ilk sıraya `evds:TP.KBK.TRY.18`'i getiriyor — adı `TP.KTF12` ile
virgül farkıyla aynı ("Konut Kredisi, (TL, Akım, %)") ama veri grubu **Kredi
Kâr Oranları**, yani katılım bankalarının kâr payı, faiz değil. Artık bu
soruda **onay kartı çıkmalı**: iki seçenek, altlarında "Kredi Kâr Oranları
(Akım)" ve "Kredi Faiz Oranları (Akım)"; **varsayılan işareti faiz olanda**
(soruda "faiz" geçtiği için), onu seç. Kart çıkmazsa, varsayılan kâr
payındaysa ya da model kâr payıyla devam ederse hata. Aynı kart 1. soruda
(son 12 ay faiz) da çıkar.

| | Ay | Pay | Faiz (KTF12) o ay | Kâr payı (KBK) seçilirse |
|---|---|---|---|---|
| En düşük | **2023-12** | **%4,4** | **42,088** | 42,204 |
| En yüksek | **2022-06** | **%26,7** | **18,178** | 21,938 |

Beklenen yorum: pay, faiz düşükken yüksek, faiz yüksekken düşük. Cevap
**metin** olmalı; yalnız "…tablodan çekiyorum." gibi bir niyet cümlesi
gelirse hata (23 Eylül'de geldi, kapatıldı). `top` seçilen ayın **bütün
satırını** listelediği için o aydaki faiz aynı adımdan okunur; model faizi
ay ay yeniden `get_series` ile çekiyor ya da "adım sınırı" diyorsa hata
(23 Eylül'de ikisi de oldu, kapatıldı). Beklenen adım sayısı 6-8.

### U4. Kırılma, karar ve sonraki 12 ay

> Konut kredisi faizindeki kırılmanın tarihini bul, o ay TCMB'nin ne karar
> aldığını web'den araştır ve kırılmadan sonraki 12 ayda faizin ortalamasını
> kırılmadan önceki 12 ayla karşılaştır.

Araçlar: `get_series`, `detect_changes`, `web_search`, ortalama için
`get_series` (pencereli) ya da `transform_table`.

- Kırılma **2023-07**; web: TCMB 20 Temmuz 2023 politika faizi artırımı
  (kaynak adresi listelenmeli; sayı web'den değil)
- Sonraki 12 ay (2023-08 – 2024-07) ortalaması **42,09**; en düşük 2023-08
  (36,22), en yüksek 2024-04 (44,72)
- Önceki 12 ay (2022-07 – 2023-06) ortalaması **19,92**
- Yani ortalama yaklaşık iki katına çıktı; cevap "2,1 kat" ya da "%111" gibi
  bir oran veriyorsa bunu **araç** hesaplamış olmalı, model değil

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
