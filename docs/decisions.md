# Karar Kayıtları

## EVDS erişim yöntemi
Bağlam: EVDS Ocak 2026'da v3'e geçti. evds2 ve /service/evds yolu ölü
        (302 -> evds3 ana sayfa, 12 Eyl 2026 doğrulandı). İnternetteki üçüncü
        taraf örneklerin tamamı v2 dönemine ait, güvenilmez.
Karar:  Resmi kılavuzdaki v3 web servisi kullanılacak:
        https://evds3.tcmb.gov.tr/igmevdsms-dis/  — anahtar HTTP header'da "key".
        Kaynak: EVDS Web Servis ve API Kullanımı kılavuzu (docs/refs/).
Sonuç:  ingest/evds.py bu base'i kullanır. 678 veri grubu, 154 kategori.

## TÜFE serisi ve zincirleme
Bağlam: Plan, TÜİK'in Ocak 2026'da 2025=100 bazına geçmesi nedeniyle eski ve
        yeni TÜFE serilerinin zincirlenmesini öngörüyordu (görev 2.3).
Karar:  Zincirleme yapılmayacak. TP.TUKFIY2025.GENEL 2021-01 – 2026-08 aralığının
        tamamında dolu; TÜİK yeni bazı geriye doğru hesaplamış. Doğrulandı:
        iki serinin oranı sabit (31.83) ve aylık yüzde değişimleri birebir aynı.
Sonuç:  evds:TUFE_ZINCIR üretilmiyor. deflate doğrudan TP.TUKFIY2025.GENEL,
        baz 2025=100. Görev 2.3'ün zincirleme kısmı kapsam dışı.

## MIA istemcisi: openai sürümü ve ölçülen API yüzeyi
Bağlam: Kurulu openai sürümü 3.14.0 ve bağımlılığı httpx değil httpx2. İnternetteki
        her örnek 1.x dönemine ait, yani istemci kurulumu ve parametre adları
        varsayımla yazılamazdı.
Karar:  Kod yazmadan önce API yüzeyi ölçüldü (15 Eyl 2026, canlı MIA).
        Ölçülenler:
          - OpenAI(api_key, base_url, timeout=float, max_retries=int) çalışıyor.
          - chat.completions.create: tools, tool_choice, response_format,
            extra_body, stream, temperature, max_tokens hepsi yerinde.
          - İstisna sınıfları hâlâ openai.* altında; status_code __init__'te set
            ediliyor; alt sınıflarda sınıf düzeyi varsayılan var (RateLimitError=429).
          - Araç çağırma çalışıyor; tool_call id biçimi "chatcmpl-tool-<hex>".
          - response_format=json_schema çalışıyor, strict bayrağıyla da hata vermiyor.
          - Embedding boyutu 4096.
          - models.list üç modeli döndürüyor ama max_model_len alanını vermiyor.
          - Hatalı model kimliğine 403 dönüyor (404 değil).
          - thinking=True: 2044 ms / 188 completion token.
            thinking=False: 523 ms / 48 completion token. Yaklaşık 4 kat fark.
          - thinking=True iken mesajda ek bir "reasoning" alanı geliyor.
Sonuç:  pyproject'te openai>=3,<4 pinlendi. llm.py bu ölçüme göre yazıldı:
        max_retries=0 (retry bizde, loglanıyor), as_message() yalnızca
        {role, content, tool_calls} süzüyor (reasoning geri gönderilmiyor),
        403 mesajı hem anahtarı hem model kimliğini işaret ediyor.

## tools ve response_format aynı istekte
Bağlam: MIA ikisini birlikte kabul ediyor (ölçüldü, hata vermiyor).
Karar:  Buna rağmen llm.py ikisini birlikte gönderirse LlmInputError fırlatıyor.
        Gerekçe: aynı çağrıda hem araç seçtirmek hem şemaya zorlamak anlamsız;
        modelin hangisine uyacağı belirsiz ve bizim hiçbir akışımızda gerekmiyor.
        Sunucu izin veriyor diye kullanmak, sessiz bir davranış farkını
        keşfedilmemiş bırakmak demek.
Sonuç:  Yasak bizim tarafımızda, sunucuda değil. İhtiyaç doğarsa tek satır kaldırılır.

## Agent motoru: LangGraph yerine kendi döngümüz
Bağlam: Mentor "agent motoru kendi kodunuz olmalı" dedi; LangGraph gibi açık kaynak
        kütüphaneler serbest. Kurulu langgraph 1.2.11 ve internetteki örneklerin
        tamamı 0.2.x dönemine ait.
Karar:  Kendi async while döngümüz yazılacak. LangGraph, checkpointer ve interrupt()
        kullanılmayacak, bağımlılık eklenmeyecek.
Sonuç:  Tek süreçli Chainlit'te insan onayı engelleyici bir await ile çözülüyor;
        checkpointer'ın çözdüğü problem (süreç dışı duraklatma) bizde yok.
        Döngü async, LLM ve araçlar senkron; döngü onları asyncio.to_thread ile
        çağırıyor, böylece arayüz donmuyor ve testler pytest-asyncio istemiyor.

## Sahte katmanın yeri: tests/ değil src/lakehouse_agent/testing/
Bağlam: FakeLLM ve FakeLake'i hem testler hem --fake CLI modu hem de ileride eval
        koşturucusu kullanacak.
Karar:  src/lakehouse_agent/testing/ altında, kurulu paketin parçası olarak.
Sonuç:  src düzeninde tests/ kurulu pakete girmiyor; oradan import etmek pythonpath
        hilesi gerektirir ve Docker imajında kırılır. Üretim modülleri bu paketi
        import etmez; tek istisna CLI içindeki --fake bayrağına bağlı tembel import.

## AgentSettings ayrı sınıf
Bağlam: Settings sınıfı EVDS anahtarını zorunlu tutuyor ve bir testi anahtar yokken
        hata bekliyor. Agent testlerinin EVDS anahtarıyla işi yok.
Karar:  config.py'ye dokunulmadan, dosyanın altına ayrı bir AgentSettings eklendi.
Sonuç:  Veri şeridinin sözleşmesi değişmedi, merge çakışması riski yok, agent
        testleri anahtarsız ortamda yeşil.

## Embedding partileme ve OCR doğrulaması
Bağlam: llm.embed() önce tek istekte tüm listeyi gönderiyordu. A'nın kataloğu
        678 veri grubu, on binlerce seri; gömme işini o yapacak.
Ölçüm:  MIA'da 1 metin 0,4 sn; 256 metin 5,6 sn; 1024 metin 22,4 sn.
        İstemci zaman aşımı 45 sn. Yani ~2000 metnin üstünde tek istek düşerdi.
Karar:  EMBED_BATCH_SIZE = 256. Sıra partiler arasında korunuyor ve dönen
        vektör sayısı girdiyle karşılaştırılıyor.
Sonuç:  A'nın katalog gömme işi güvenli; llm.embed() istediği kadar metin alır.

Ayrıca OCR canlı doğrulandı (15 Eyl): geçerli bir PNG ile 314 ms, çıktıda
rehberin bahsettiği <|det|> yerleşim etiketleri var. Yani Unlimited-OCR yolu
uçtan uca çalışıyor; web araçları aşamasında yeniden keşfedilmesi gerekmiyor.
## EVDS tam çekim kapsamı
Bağlam: 678 veri grubu x 6 yıl = 4068 çağrı hedeflendi. 34 çağrı kalıcı olarak
        timeout verdi (read timeout=30); ikinci tur da aynı grupları getiremedi.
        Başarısız gruplar ağırlıklı olarak bie_pydibs / bie_sekbily01a gibi
        günlük ve çok serili DİBS/bilanço grupları — yanıt boyutu 30 sn'de dönmüyor.
Karar:  %99.2 kapsam (3384/4068 parquet) yeterli kabul edildi. Timeout'u artırıp
        tekrar denemek yerine kapsam kalite raporunda açıkça belgelenecek.
        Senaryo ve eval serilerinin hiçbiri bu gruplarda değil.
Sonuç:  ingest ağ hatasında durmuyor, logla-atla-devam ediyor; başarısız liste
        fetch_log'da ve docs/data_quality.md'de görünür.

## Tablo okuma ve tablo numaralandırma: iki sessiz hata
İkisi de sahte katmanla yeşil geçen, gerçek araçlarla patlayacak hatalardı.

1. `ToolContext.table()` içinde `written.get(k) or _tables.get(k)` vardı.
   DataFrame'in doğruluk değeri yok; `or` ilk dal bir şey bulduğu anda
   `ValueError` fırlatıyor. Yani bir aracın aynı adımda yazdığı tabloyu geri
   okuması imkânsızdı. Hiçbir sahte araç bunu yapmadığı için testler görmedi.
   Üyelik kontrolüne çevrildi ve testi yazıldı.
2. `new_table_ref` tablo *sayısına* göre numara veriyordu. Kalite kapısı bir
   tabloyu sildiğinde -- ki tam olarak bunu yapacak -- sonraki tablo
   kullanımdaki bir numarayı alıp üstüne yazıyordu. Artık en yüksek numaradan
   devam ediyor; silme numarayı serbest bırakmıyor.

## Araç kaydından iki ölü alan çıkarıldı
`ToolSpec` üzerinde `creates_table` ve `sets_active` bayrakları vardı ve hiçbir
yer okumuyordu. Döngü bu iki bilgiyi zaten aracın gerçekte yazdıklarından
hesaplıyor (`ctx.written`, `ctx.new_active`) -- beyana değil gerçeğe bakmak
bilinçli bir tercihti. Okunmayan bir beyan bırakmak, birinin onu doldurup bir
etki beklemesine davetiye; kaldırıldı. `brief_name` duruyor çünkü sunumun altı
aracıyla bizim fonksiyonlarımızı eşleyecek `docs/tools.md` ona dayanacak.

## Model uydurduğu argüman adıyla yanlış tabloyu analiz edebiliyordu
Pydantic'in varsayılanı `extra="ignore"`. Şema modele "bu aracın alanları
şunlar" diyordu ama doğrulama fazlasını sessizce atıyordu. Şemanın tamamı
modele Türkçe açıklamalarla gidiyor, yani `table_ref` yerine `tablo` yazmak
beklenen kayma; MIA testinde modelin açıklamasız bir alana SQL yazdığı zaten
görülmüştü.

Ölçüldü: `{"tablo": "t02_deflate", "column": "TP_KTF12"}` çağrısı `table_ref`i
`None` bırakıyor, `None` aktif tablo demek, araç `t01`i analiz edip `ok=True`
ve beş anomali bildiriyordu. Doğru anahtarla aynı çağrı üç anomali veriyor.
Özet, tarihler ve kaynaklar tutarlı görünüyor ve hepsi yanlış tabloya ait.

Karar: kural araç kaydında, argüman modellerinde değil. Şemaya
`additionalProperties: false`, doğrulamaya bilinmeyen anahtar reddi eklendi.
Her araç modeline `extra="forbid"` yazmak da olurdu ama o modeller aracı
yazanın; "şema ile doğrulama ayrışmaz" sözü kaydın sözü ve yarın bu dosyayı
hiç okumamış birinin kaydettiği araç için de geçerli olmalı.

## preview_rows "JSON-safe" diyordu, gold'un period sütununu serileştiremiyordu
Yalnızca `datetime64` sütunlar çevriliyordu. Gold sözleşmesi `period`i
`datetime.date` olarak saklıyor -- object dtype, hiçbir dtype kontrolü görmez.
Yani gerçek bir gold tablosunun önizlemesi `json.dumps`'ta patlıyordu; hata da
burada değil, tur ortasında olay kanalında çıkacaktı. `Timedelta`, `Period`,
`Decimal` aynı durumda. `NaN` ayrı bir sorun: `json.dumps` onu çıplak `NaN`
yazıyor, bu geçerli JSON değil ve arayüzdeki `JSON.parse` reddediyor.

Çevrim artık dtype başına değil değer başına, çünkü kararı veren dtype değil.
Eksik eksik kalıyor: `null` oluyor, doldurulmuyor (CLAUDE.md #2).

## Üç sessiz sınır: yüzüncü tablo, yüzüncü grafik, sayısal period sütunu
- `t{n:02d}` genişlemeye devam ediyor ama desen tam iki basamak kabul ediyordu;
  yüzüncü tablo `ValueError` fırlatıyordu -- hem de `pragma: no cover -
  defensive` ile işaretli, yani ulaşılamaz sanılan bir satırda. Desen `\d{2,}`
  oldu, mevcut her referans hâlâ geçerli.
- `new_chart_ref` hâlâ *sayarak* numaralandırıyordu (tablolarda düzeltilen
  hatanın aynısı) ve desen kontrolü yoktu: `c100` sessizce üretilip çok sonra
  `ChartSpec` doğrulamasında patlardı.
- `describe_table` sayısal bir `period` sütununu `to_datetime`'a veriyordu; o da
  tamsayıyı epoch'tan nanosaniye okuyor. Modelin tablo hakkında okuduğu tek
  satır "1970-01 - 1970-01" diyordu. Artık okumayı reddediyor.

## ToolContext'in salt-okunur yarısı salt-okunur değildi
`_tables` ve `lineage` `Mapping` diye yazılmış, `dict` olarak veriliyordu --
üstelik oturumun kendi sözlükleri. Bir araç yazma tamponunu tamamen atlayıp
doğrudan oturuma yazabilirdi; iptal edilen adımın iz bırakmaması sözü de
böylece kalkardı ve hiçbir test kırılmazdı. Artık `MappingProxyType`, üstelik
anlık görüntü üzerinden: zaman aşımına uğramış bir araç oturumun ilerlemesini
de görmemeli.

## Argüman kapısı sınıfa değil, modele gösterilen şemaya bakıyor
İlk düzeltme kabul edilen anahtar kümesini `model_fields`'tan kuruyordu ve
takma adlarda aynı anda iki yönde birden yanılıyordu: Pydantic'in sessizce
atacağı bir anahtarı geçiriyor, şemanın kendi ilan ettiği anahtarı ise
reddediyordu. Ayrıca `additionalProperties: false` şemada her iç nesneye
yazılırken doğrulama yalnızca en üst seviyeye bakıyordu -- yani aynı söz bir
kat aşağıda yine tutulmuyordu.

Kontrol artık `tool_schema(model)` üzerinden yürüyor ve değerin içine iniyor.
Modülün ilk paragrafındaki "şema ile doğrulama aynı yerden gelir" cümlesi böyle
kod oluyor: iki yarı aynı nesne. `extra="allow"` da bedavaya doğru çalışıyor,
çünkü Pydantic o durumda şemaya zaten `additionalProperties: true` yazıyor.

Birleşim (`Optional[Model]`) dallarında bir tuzak çıktı: `{"type": "null"}`
dalının denetleyecek `properties`'i yok, yani uydurma anahtarlı bir nesneyi
"kabul ediyor" ve tek dalın kabulü yetiyordu. Artık yalnızca değerin şeklini
alabilecek dallar soruluyor.

## $ref açmada sayaç değil, takip edilen referans zinciri
Derinlik sayacı üç kat iç içe geçmiş sıradan bir modeli döngüden ayırt
edemiyordu: acyclic modellerde ateşleniyor, modeli kesiyor ve -- `Optional[X]`
işaretçiyi `anyOf`'un bir kat altına koyduğu için -- sarkan `$ref`'i yine de
dışarı bırakıyordu. Yani fonksiyon tam olarak önlemek için var olduğu şeyi
üretiyordu. Artık takip edilen referans zinciri tutuluyor; kendine dönen bir
model açık nesne olarak bitiyor: kısıtsız ama çözülebilir.

## Grafik referansı iki ayrı adla taşınıyor
`ChartSpec` alana `chart_ref`, `contracts.Artifact` ise `ref` diyor -- ve
`add_chart` ikincisini de kabul ediyor. Yalnızca birini okumak, saymaktan
kötüydü: tanınmayan taşıyıcı hiç sayı katmıyor, tarama boş liste görüyor ve
oturumdaki her grafiğe yine `c01` veriyordu. Çakışma kontrolü düşünüldü ve
yazılmadı: aday her zaman okunan en yüksek referansın bir fazlası, o sayıya
ayrışan bir referans zaten en yükseği o yapardı; ateşlenemezdi.

## describe_table'ın 1970 koruması da değer bazlı
İlk düzeltme `is_numeric_dtype`'a bakıyordu; o da `int64` görüyor ama aynı
tamsayıları tutan object sütunu görmüyor -- yani düzeltme tam da dtype
kontrolünden saklanan sütun şeklini kaçırıyordu. Aynı commit'teki `json_safe`
zaten "kararı veren dtype değil" diyordu; ikisi artık aynı ilkeyi izliyor.

## Önizleme modele JSON olarak gidiyor
`format_tool_message` Python repr'ı basıyordu: model `datetime.date(2020, 9, 1)`
ve `nan` okuyordu. Burası önizlemenin modele ulaştığı **tek** yer ve her araç
buradan geçiyor -- önizlemesini `preview_rows` ile kurmayanlar da. Dönüşüm bu
kapıya kondu, tek tek araçlara değil.

## Araç mesajı kesilirken ortadan eleniyor, sondan değil
`preview_rows` iki ucu da bilerek tutuyor ("son dönem ilki kadar bilgilendirici").
Ama önizleme mesaja en son ekleniyor ve bitmiş metin 2000 karakterden kesiliyordu,
yani kesme her zaman önce son satırları yiyordu -- üstelik `(kisaltildi)` neyin
gittiğini söylemeden. Ölçüldü: FakeLake'in dört serilik demo penceresi, uyarılar
ve kalite kartıyla birlikte 1961/2000 karakter -- tetiklenmiyor ama 39 karakter
kalıyor. Önizleme artık kalan bütçeye göre ortadan eleniyor, etiket kaç satırın
atlandığını yazıyor. Hiç sığmıyorsa yarım JSON yerine "sığmadı" deniyor.

## Argüman kapısının izlemediği iki kenar
`dict[str, Model]` hiç `properties` üretmiyor, değer modelini
`additionalProperties` altına koyuyor; `tuple[A, B]` ise `items` olmadan
`prefixItems` üretiyor. Şema ikisinde de iç modeli kapatıyordu, doğrulama ise
yanından geçiyordu -- yani şema anahtarı yasaklarken doğrulama kabul edip
sessizce atıyordu. Tam olarak bu modülün var olma sebebi olan sapma.

Ayrıca ayırt edicili (discriminated) birleşimde her dalı sormak yanlış: etiket
her dalın alanı olduğu için `{op: "deflate", anchor: ...}` yanlış dal tarafından
bütünüyle kabul ediliyor ve kapı susuyordu; Pydantic ise katı biçimde etikete
göre yönlendirip `anchor`'ı varsayılanına düşürüyordu. Artık etiket okunuyor.

`discriminator.mapping` ise `#/$defs/...` işaretçilerini düz string **değer**
olarak tutuyor; tarama yalnızca `$ref` **anahtarı** olan sözlükleri yeniden
yazdığı için `$defs` siliniyor ve mapping boşluğa işaret ediyordu. Testimiz de
göremiyordu (`"$ref" not in ...` — belge işaretçiyi içeriyor, sadece o adla
değil). Mapping düşürüldü, test `#/$defs/` arıyor.

## MappingProxyType anahtarı korur, arkasındaki nesneyi korumaz
`lineage` değerleri oturumla kimlik paylaşan sıradan değiştirilebilir modeller.
Bir araç bir sütunun nereden geldiğini yerinde değiştirebiliyor, yazma tamponu
boş kalıyor ve izlenebilirlik paneli düzenlemeyi hep oradaymış gibi gösteriyordu.
Değerler artık anlık görüntüye kopyalanıyor. Tablolar için aynı şey geçerli
değil: `table()` zaten kopya veriyor -- ama sığ, yani hücre içinde duran bir
nesne oturumunkiyle aynı nesne. Gold sözleşmesinde iç içe tablo yok; docstring
artık bunu söylüyor.
## Onay beklemesi tur bütçesinden düşülüyor
Bağlam: Bütçe (180 sn) `active_s` üzerinden ölçülüyor ama ilk yazımda fetch onayı
        adımın içinde kalıyordu; `_act` adımın başında saati alıp sonunda farkı
        `active_s`'e ekliyordu. Jüri kartı 60 sn okusa o 60 sn çalışma sayılırdı.
Karar:  `SessionState.approval_wait_s` eklendi. `_ask` yalnız insanı bekleme
        süresini oraya yazıyor, `_act` içindeki `working_s()` bu farkı hem
        `active_s`'ten hem adımın `duration_ms`'inden düşüyor.
Sonuç:  Onay kapısı ile zaman sınırı aynı anda var olabiliyor. Sahte saatle
        ölçülen test: kart 500 sn açık kalıyor, tur düşmüyor, `active_s < 180`.

## Argüman yeniden deneme sayacı araç adına göre, çağrı kimliğine göre değil
Bağlam: `arg_retries` sözlüğü `call.id` ile anahtarlanıyordu.
Sorun:  Model her araç çağrısına yeni bir kimlik veriyor. Sayaç tam da saymak
        istediği yeniden denemede sıfır okuyor, yani "hatayı bir kez geri ver"
        kuralı hiç devreye girmiyor, model aynı argümanları sonsuza dek
        deneyebiliyordu. Tekrar-çağrı koruması ancak üçüncü denemede yakalıyordu.
Karar:  Anahtar `call.name`.
Sonuç:  İkinci başarısız denemede modele "bu argümanlarla tekrar deneme" gidiyor.

## Araç uyarıları cevap prompt'una taşınıyor
Bağlam: `answer.md`'deki `{{warnings}}` yer tutucusunu `_collected_warnings`
        dolduruyordu ama fonksiyon adım hata kodlarını topluyordu ("timeout").
        `ToolResult.warnings` hiçbir yerde biriktirilmiyordu.
Sonuç:  "Haftalık seri aylık ortalamaya çevrildi" gibi cümlenin anlamını
        değiştiren bir uyarı cevaba hiç ulaşmıyordu; planlayıcı prompt'unun
        8. kuralı karşılıksız kalıyordu.
Karar:  `SessionState.warnings` eklendi; `_merge_context` hem aracın kendi
        uyarılarını hem döngünün ürettiği "mevcut sütunlar değişti" uyarısını
        oraya yazıyor. Uyarılar ve başarısız adımlar ayrı ayrı sınırlanıyor,
        yoksa dört başarısız adım uyarıları listeden itiyor.

## Sahte araç katmanında sütun adı kuralı
Bağlam: Sahte `get_series`, sütun adını `series_id.split(":")[-1]` ile üretiyordu.
Sorun:  EVDS kimliğinde doğru (`evds:TP.KTF12` -> `TP_KTF12`) ama BDDK kimliğinde
        iki nokta hiyerarşi ayracı: `bddk:aylik:tuketici:konut:TL:sektor` -> `sektor`.
        Bütün tüketici kredisi serileri tek ada çökerdi.
Karar:  Yalnız kaynak öneki atılıyor, kalan ayraçlar alt çizgiye dönüyor.
Açık:   Gerçek `wide()` sütun adlarını A'nın belirlemesi gerekiyor (istek listesi
        madde 5). Cevap gelince değişecek tek yer `_column_name`.

## Cevapsız araç çağrısı: oturumu bitiren iki hata
OpenAI protokolünde bir asistan mesajı araç çağırıyorsa o `tool_call_id` için bir
araç mesajı gelmek zorunda. Gelmezse **bir sonraki** istek 400 döner. Mesaj
geçmişi turlar arası taşındığı için bu, turu değil oturumun tamamını bitiriyor.
Döngünün iki yerinde oluyordu:

1. Adım sınırı veya süre bütçesi dolduğunda döngü duruyordu ama modelin son araç
   çağrısı cevapsız kalıyordu. Tur yarım kalmakla kalmıyor, cevabı yazma çağrısı
   da patlıyordu.
2. Onay geri çağrısı hata verirse (tarayıcı kapandı, kart zaman aşımına uğradı)
   aynı yetim çağrı oluşuyordu.

Karar:  `_answer_abandoned_calls` cevabı yazmadan önce geçmişi tarayıp cevapsız
        her çağrıya "bu araç çalıştırılmadı" mesajı yazıyor; onay hatası ise
        başarısız bir adım olarak kaydediliyor. İkisi de test edildi, hem de
        genel bir `orphan_call_ids` yardımcısıyla erken biten her testte.

## Döngüde diğer düzeltmeler (aynı inceleme)
- Başarısız dönen bir aracın (`ok=False`) yazdıkları artık oturuma karışmıyor;
  zaman aşımına uğrayan adımla aynı kural.
- Seri seçimi kapısı `spec.name == "search_catalog"` yerine `returns_candidates`
  bayrağına bakıyor. Bir aracı yeniden adlandırmak bir insan onayını sessizce
  devre dışı bırakmamalı.
- Doğrulayıcı `asyncio.to_thread` içinde çalışıyor: bugün `NullVerifier` olduğu
  için zararsız ama 5. aşamada model çağıracak ve arayüzü dondururdu. (22 Eyl:
  varsayılan artık kural tabanlı `AnswerVerifier`, model çağırmıyor.)
- `describe_table` okunamayan bir dönem sütunu yüzünden başarılı bir adımı
  düşüremiyor.
- Başarısız adımlar da `step_start`/`step_end` yayınlıyor. Hatalı argüman ve
  iptal edilen çekim jürinin en çok göreceği iki şey; olay yayınlanmayınca
  arayüzde kapanmayan bir adım kartı, yani donmuş bir ekran kalıyordu.
- `llm.chat` artık isimli argümanlarla çağrılıyor.
## CLI'da iki mod: --fake ve --live
Bağlam: Çevrimdışı test paketi bir şeyi söyleyemiyor -- model gerçekten doğru
        aracı seçiyor mu, şemadan argümanları doğru dolduruyor mu, Türkçesi
        tutarlı mı?
Karar:  `--fake` (sahte model + sahte araç, ağ yok, deterministik) ve `--live`
        (gerçek MIA + sahte araç). Araçlar iki modda da sahte; böylece canlı
        modda bir hata çıktığında sebebi tek: model ya da prompt, veri katmanı
        değil. Varsayılan mod yok, biri açıkça seçilmek zorunda: canlı
        varsayılan olsa masum görünen bir komut kota harcardı, sahte varsayılan
        olsa insan modeli test ettiğini sanırdı.

Canlı ölçüm (15 Eyl, planın 18 Eylül'e koyduğu duman testi): "2021-2025 arası
konut kredisi faiz oranını getir ve aylık farkını hesapla" sorusu tek turda
bitti. Model search_catalog -> get_series -> transform_table sırasını kendi
kurdu, `TP_KTF12` sütun adını araç çıktısından birebir kopyaladı, 12,6 saniye,
dokuz adım, sıfır onay, kesinti yok. Cevap kaynağı belirtti ve haftalık seriyi
aylık okuduğunu kendisi yazdı.

Ölçülen bağlam maliyeti: tek turda dört model çağrısı toplam 11.556 prompt
token. Sahte araçların şemaları küçük; gerçek araçlarla ve üç turlu bir
oturumda bu rakam ciddi biçimde büyüyecek. Mesaj geçmişi kırpma kararı bu yüzden
arayüz aşamasına bırakıldı, ama unutulmamalı.

## Reddedilen tablo neyi yanında götürür
Kalite kapısında "İptal" yalnızca iki şey yapıyordu: tabloyu sözlükten çıkarmak
ve aktif tabloyu boşaltmak. Geride kalanlar daha önemliydi.

`state.sources`'ta duran seri `answer.md`'nin 4. kuralına göre cevapta kaynak
olarak anılabiliyordu -- yani jürinin az önce kalite gerekçesiyle reddettiği
veri, cevabın provenance panelinde. Soyağacındaki satırlar da kalıyordu.
Numara ise serbest bırakılıyordu: kapı her zaman **en yeni** tabloyu reddediyor,
o da `new_table_ref`'in saydığı en yüksek numara, yani aynı ref hemen yeniden
dağıtılıyordu. Ölçüldü: reddedilen `t02_quality`'nin ardından model aynı aracı
çağırınca yeni tablo yine `t02_quality` oluyor ve eski soyağacı satırı yeni
tabloya yapışıyor. `decisions.md`'nin "silme numarayı serbest bırakmıyor"
kaydı, yazıldığı vakayı kapsamıyormuş.

Geri alma artık tam: soyağacı, kaynaklar (yalnızca o adımın eklediği), grafik,
aktif tablo (öncekine döner, boşa değil) ve `retired_tables`. Arayüze de
`table_update {removed: true}` gidiyor.

## Uyarılar tura ait, oturuma değil
`_collected_warnings` iki listeyi aynı etiket altında basıyordu: `state.warnings`
her tur sıfırlanıyor, `state.steps` sıfırlanmıyor. `answer.md` 8. kural modele
"eksik kaldıysan ilk cümlende söyle" dediği için 3. tur, 1. turda patlamış bir
aracı canlı arıza gibi bildiriyordu. Tek satır: `step.turn_no == state.turn_no`.

## Başarısız adım hiçbir şey yayınlamaz
`_merge_context` "hata bildiren araç yarım tablo, sarkan soyağacı ya da kimsenin
göremeyeceği bir kaynak bırakamaz" diyor ve oturum için bunu uyguluyordu. Olay
kanalı aynı kuralı uygulamıyordu: `ok=False` dönen adımın kaynakları, kalite
kartı ve grafiği arayüze gidiyordu. Dahası kalite kapısı da açılıyor, "iptal"
hiç birleştirilmemiş tabloyu değil **önceki sağlam** tabloyu siliyordu.

## Grafik üç taşıyıcıyla geliyor, döngü birini okuyordu
`add_chart(chart: Any)`, `test_context.py` düz sözlüğü sabitliyor ve
`state.chart_ref_of` özellikle `ref`'i de okumak için var. Döngü ise yalnızca
`.chart_ref` okuyordu: kod tabanının kabul ettiğini söylediği taşıyıcılar turu
`done` yayınlandıktan **sonra** ham traceback ile öldürüyordu. Ayrıca
`ToolResult.artifacts` -- iki şeridin de bağlı olduğu sekiz ortak tipten biri --
hiç okunmuyordu; o yoldan dönen grafik izsiz kayboluyordu. `plotly_json` ise
adını aldığı kütüphanenin ürettiği şeyi serileştiremiyordu: sıradan bir
`px.line` numpy dizisi taşıyor ve istisna emitter'dan her yakalayıcıyı aşarak
çıkıyordu. Grafik yükü artık `json_safe`'den geçiyor.

## Onay cevabı da kart kadar denetleniyor
Karar motorun dışından geliyor: arayüz, eval koşucusu, test. Kartta olmayan bir
seçenek insanın yapabileceği bir seçim değil ve ona göre davranmak kimsenin
onaylamadığı şeyi yapmak demek -- `cancel` özelinde bir tabloyu silmek. Yanlış
`request_id` de aynı sınıf. İkisi de kartın kendi varsayılanına düşüyor ve not
alanına yazılıyor. Kartın `context`'i artık `json_safe`'den geçiyor: bir katalog
isabetindeki numpy skoru `emit`'te patlıyordu ve döngü başarısız kapıyı
"kapıyı atla" diye yorumladığı için bu, hata olarak değil **insanı karardan
sessizce çıkarmak** olarak görünüyordu.

## Onay kartı da bir adımdır: açılışı kadar kapanışı da olay
`approval_decision` eklendi (sözleşme v0.4). `JsonlSink`'in varlık sebebi bir
oturumun replay edilebilmesi; iki kapının açılıp hiç kapanmadığı bir kayıt o
oturum değil. Aynı gerekçeyle model adımları da artık `step_start`/`step_end`
yayınlıyor: sözleşme `step_end` payload'ında `tokens` vaat ediyordu ve token
taşıyan tek adım türü hiç olay yayınlamıyordu.

## Kesme ve boş cevap
Araç mesajı kesilirken `_preview_line` zaten ortadan eliyordu; burada kalan iki
şey vardı. `asyncio.CancelledError` `BaseException` türevi olduğu için hiçbir
`except` yakalamıyordu: iptal edilen tur modelin araç çağrısını cevapsız
bırakıyor, oturumun geri kalanı protokol hatası alıyordu. Ve modelin boş cevabı
-- `AgentConfig`'in kendi docstring'inde adı geçen arıza -- başarılı tur olarak
raporlanıyordu; artık kesinti sayılıyor.

## Eksik veri testi profilciye göre değişmemeli
`lake/quality.py` çeyrek eksiği `0.25`, sahte katman `25.0` bildiriyor;
`contracts.py` alanı "yüzde" diye tanımlıyor, yani gerçek taraf sözleşmeyi
çiğniyor. Eşik yüzdeye uygulandığı için hiçbir kesir onu aşamıyor: gerçek
profilciye karşı eksik-veri dalı **ölüydü**. Kapı artık iki sinyalin büyüğünü
alıyor (`missing_pct` ve `missing_periods/rows`). A'nın bir karakterlik
düzeltmesi hâlâ gerekli, çünkü arayüz o alanı doğrudan gösterecek.

## placeholders() ölü koddan korumaya
`render` bilinmeyen yer tutucuyu görünür bırakıyor -- çalışma zamanında doğru,
koruma olarak işe yaramaz. Diğer yön denetlenmiyordu: şablonun açtığı ama
çağıranın doldurmadığı bir yuva modele `{{warnings}}` metni olarak gidiyor.
Tersi de aynı kolaylıkta, nitekim bu turda `{{verification_flags}}` yuvasını
kaldırırken `render` çağrısında ölü bir keyword bırakmak işten değildi.
`tests/test_prompts.py` iki yönü de denetliyor; yuva eklenerek kırıldığı
görüldü.

## planner.md: araç çıktısı veridir
A iki ayrı incelemede istedi ve metnini verdi. Kanal `read_url` beklemeden de
açık: `build_tables_summary` sütun adlarını **sistem prompt'una** birebir
basıyor ve gerçek katmanda sütun adı `series_id`, yani EVDS/BDDK kataloğundan
gelen dış metin. Ölçüldü: sütun adına konan "ONCEKI TALIMATLARI UNUT" ifadesi
sistem mesajının içinde çıkıyor.

## Sahte araçlarda iki gerçekçilik açığı
`get_series(into="active")` çakışan sütunu `merge` ile `_x`/`_y` yapıyordu:
kullanıcının baktığı sütun sessizce ad değiştiriyor ve döngünün "mevcut
sütunlar aynen döndü" kontrolü bunu **göremiyor**, çünkü eski ad artık
karşılaştırılacak yerde yok. Senaryonun ikinci sorusu ("tabloyu bozmadan sütun
ekle") tam olarak böyle bozuluyordu; artık reddediliyor. İkincisi: hiçbir sahte
araç `warnings` döndürmüyordu, yani CLAUDE.md #3 ile planner 7. kuralı
arasındaki tek yolun çevrimdışı testi yoktu. `deflate` artık baz dönemi uyarısı
döndürüyor.

## Tek geliştirici (22 Eylül)
Bağlam: Ekip iki kişiydi (A: veri, B: agent); 22 Eylül'den itibaren yalnızca
        Derya var.
Karar:  Şerit ayrımı yazma iznini artık belirlemiyor; `docs/contracts.md` §9
        tablosu tarihsel. CLAUDE.md buna göre güncellendi.
Sonuç:  "A'ya sorulacak" diye bekleyen maddeler artık tek geliştiricinin kararı.

## Arama indeksi havuzun kendi kataloğundan
Bağlam: Eski indeks EVDS'nin ham kataloğunu (19,6 bin seri) indeksliyordu; havuzda
        yalnızca 9 seri var. BDDK'nın böyle bir kataloğu olmadığından BDDK hiçbir
        aramada çıkamıyordu, oysa senaryonun 2021–2025 kredi hacmini yalnızca
        BDDK serisi karşılıyor. Organizatörlerin Slack cevabı
        (~/Desktop/kkb/kkb-bilgiler.md): vektör veritabanı esas veri için değil,
        "hangi veri hangi tabloda, hangi kolon neyi temsil ediyor" soruları için.
Karar:  `LakeAdapter.search` gold `series_catalog` üzerinde arar. Arama metni:
        ad + konu + etiket + birim + frekans + tür + kaynak (plan 2.5'in alanları
        ve türün Türkçe adı). Skor kosinüs benzerliği; kimlik/kod birebir geçerse
        yalnızca o seri. Embedding yapılamazsa kelime eşleşmesi + uyarı. Adayın
        tarihleri havuzun gerçek kapsamı. `data/catalog.lance` ve `build_index.py`
        havuz dışı keşif (ingest) için duruyor, aramada kullanılmıyor.
Sonuç:  Ölçüm (22 Eyl, Qwen3-Embedding-8B, 9 seri): ilgili sorgularda en iyi
        adayın en düşük skoru 0,470 ("enflasyon" → TÜFE). İlgisiz yedi sorgunun
        dördü 0,40 altında (dolar 0,329, altın 0,262, cari açık 0,230, işsizlik
        0,386), üçü üstünde (BIST 100 0,495, mevduat faizi 0,462, otomobil
        kredisi 0,457). Eşik 0,40: bariz ilgisizi eler; kalanını ayırt etmek
        modelin işi. Bilinen zaaf: "konut kredisi bakiyesi" sorgusunda
        "Takipteki Konut Kredileri" (0,650) asıl bakiyenin (0,615) üstünde
        çıkıyor. Kaynağın kendi tablo hiyerarşisini metne eklemek sıralamayı
        değiştirmedi (ölçüldü). BDDK satırlarının konu/etiketi boş; 2. adımda
        havuz genişlerken katalog metinleri de zenginleştirilecek.

## Kimsenin yapmadığı seri seçimi modele "insan seçti" diye söylenmez
Bağlam: Onay kapısı etkileşimsiz modda (CLI, eval) kartın varsayılanını, yani
        aramanın en üst adayını uyguluyor ve modele "[Onay] Seri secimi: X. Bu
        seriyle devam et, tekrar arama yapma." yazıyordu. Gerçek havuzda en üst
        aday bazen yanlış (takipteki krediler); bu mesaj modelin adayları
        okuyup doğru seriyi seçmesini engelliyordu ve olmayan bir insan kararını
        ima ediyordu.
Karar:  `ApprovalDecision.automatic`. Otomatik seri seçimi kayda ve olaya girer
        ama modele mesaj olarak gitmez; model adaylardan kendisi seçer. Bir insan
        seçtiğinde davranış aynı.
Sonuç:  `approval_decision` olayına `automatic` alanı eklendi.

## Gerçek Lakehouse araçları (tools/lakehouse.py)
Karar:  - Sütun adı kuralı sahte araçlarla ortak tek fonksiyon (`column_name`):
          `evds:TP.KTF12` → `TP_KTF12`. Testler, prompt örnekleri ve model aynı
          adları görür.
        - `get_series` her zaman aylık tablo döndürür; frekans parametresi yok.
          Haftalık seri türüne göre aya çevrilir (bakiye: ay sonu, oran/endeks:
          ortalama, akım: toplam), bu uyarıda ve sütun soyağacında yazar. Üç
          aylık/yıllık seri reddedilir (aylığa çevirmek veri icat etmek olur).
        - Havuzdan okunan tabloya kalite kartı eklenmez: `profile_table` birimi
          hep boş döndüğü için kapı her seferinde açılırdı; kalite kartı demo
          anında gelen tablolar içindir (plan 2.6). Eksik aylar uyarıda sayılır.
        - `transform_table` işlemleri: diff, pct_change, yoy, change_since,
          rebase, ratio, deflate, mark. Hepsi yeni sütun ekler, var olan adı
          reddeder. `mark` koşulu sağlayan ayları özet metninde listeler: "hangi
          aylar" sorusunu model saymaz (planner kural 1).
        - `deflate` yalnızca stok/akım serilerine uygulanır; oran ve endeks
          reddedilir. TÜFE sabit (TP.TUKFIY2025.GENEL); baz 2025 ortalaması = 100,
          ya da `base_period` verilirse o ayın fiyatları. TÜFE kaynaklara eklenir.
        - Önizlemede ondalıklar 4 haneye yuvarlanır (tablo tam kalır):
          0.4499999999999993 gibi kayan nokta artıkları modele gitmesin.
Sonuç:  Senaryonun üç sorusu gerçek araçlar ve gerçek gold ile tek oturumda
        çalışıyor (tests/test_scenario.py). Gerçek veride "faiz düşerken kredi
        artmayan aylar": 2021-02, 2021-05, 2023-12, 2024-01.

## Model çağrıları tur bütçesine bağlı; zaman aşımı tekrar denenmez
Bağlam: 22 Eyl `--live` koşusu: araç adımları 57 sn sürdü, sonra düşünme açık
        cevap çağrısı üç kez 45 sn'lik zaman aşımına uğradı (138 sn); tur 195 sn
        sürdü ve hiçbir cevap üretmedi. Bütçe yalnızca araç adımlarından önce
        kontrol ediliyordu. O gün MIA 15 Eylül'e göre 6–10 kat yavaştı
        (5,6–23 token/sn'ye karşı 61–136).
Karar:  Zaman aşımı `llm.py`'de tekrar denenmez. Döngü her model çağrısına turun
        kalan bütçesini `timeout_s` olarak verir; plan/gözlem için 10 sn'den az
        kaldıysa çağrı yapılmaz ve cevaba geçilir; cevap en az 30 sn alır.
        Düşünme açık bir çağrı zaman aşımına uğrarsa bir kez düşünme kapalı ve
        `max_tokens=1024` ile denenir. Düşünme açık çağrıya token sınırı
        konmadı: akıl yürütme token'ları sınırı tüketirse cevap boş kalır.
Sonuç:  CLI başarısız turda da adımları ve tabloları gösteriyor.

Ek (aynı gün, ilk `--real` koşusu): senaryonun 1. sorusunda model doğru serileri
seçti, iki farkı aldı ve `mark` dört ayı listeledi; ardından gelen gözlem
çağrısı (düşünme kapalı) 45 sn'yi aştı ve bitmiş analiz yine kayboldu. MIA
8–22 token/sn çalışıyordu, istem 7,8 bin token'a çıkmıştı. İki değişiklik:
- Gözlem çağrısı zaman aşımına uğrarsa araç döngüsü biter ve cevaba geçilir
  (`truncated_reason="llm_error"`; cevap kesintiyi söyler). Gözlem yalnızca
  sıradaki adıma karar verir, araç sonuçları zaten oturumda. 5xx hâlâ turu
  düşürür: sunucu çökükse cevap çağrısı da aynı sunucuya gider.
- Hata veren model çağrısı artık `step_end(ok=false)` yayınlıyor; açılan adım
  kartı kapanmadan kalıyordu.
- planner 14. kural: gözlem adımında cevap yazılmaz, "Veriler hazır." denir.
  Gözlemde yazılan uzun özetler (ölçülen 257–320 token) cevap adımında yeniden
  yazılıyordu.

İkinci `--real` koşusu (22 Eyl, değişikliklerden sonra, üç soru tek oturum):
üç tur da kesintisiz bitti (118 / 121 / 136 sn, bütçe 180). Model iki aramada da
otomatik varsayılanı değil doğru seriyi seçti (en üstteki "Takipteki Konut
Kredileri" idi). Cevaplardaki sayılar gold'dan yeniden hesaplanarak kontrol
edildi: 2. ve 3. cevap tam doğru; 1. cevapta dört ay doğru, fakat "2021-05'te
faiz %27,85 puan düştü" dedi (gerçek −0,2785 puan: 100 kat kayma) ve 2025 son
çeyrek kredi artışını "14-20 bin" verdi (gerçek 16,6–20,2 bin). Sistematik
çözüm cevaptaki sayıları araç çıktısıyla karşılaştıran doğrulayıcı (plan 5.3);
o gelene kadar `transform_table` özetinde yeni sütunun birimi yazılıyor.
Bağlam büyüyor: 3. turun beş çağrısı 59,5 bin prompt token'ı; mesaj geçmişi
kırpma kararı artık ertelenemez. (İkisi de aynı gün geldi: aşağıda "Cevap
doğrulayıcı" ve "Modelin gördüğü geçmiş".)

## CLI'da üçüncü mod: --real
Karar:  `--real` gerçek MIA + gerçek araçlar + gerçek veri; hiçbir sahte modül
        import edilmez. `--question` birden çok kez verilebilir; sorular aynı
        oturumun turlarıdır, böylece "tabloyu bozmadan" takip soruları önceki
        tabloyu görür. `--fake` ve `--live` aynen duruyor.

## Cevap doğrulayıcı (agent/verifier.py)
Bağlam: 22 Eyl `--real` koşusunda cevap "%27,85 puan" dedi (gerçek −0,2785).
        Model sayıyı araç çıktısından yanlış aktardı; bunu istem düzeltemez.
Karar:  Kural tabanlı `AnswerVerifier` varsayılan doğrulayıcı; model çağırmaz,
        tur bütçesinden süre yemez. Üç kontrol: (1) cevaptaki her sayı, kendi
        yuvarlama hassasiyeti içinde tablolarda ya da araç mesajlarında
        bulunmalı; "aştı/altında" sınır, "A ile B arasında" aralık,
        "yaklaşık" %5 pay olarak okunur; bin/milyon/milyar ölçek kelimeleri ve
        baz puan hesaba katılır; (2) anılan seri kimliği/kodu/URL kaynaklarda
        olmalı; (3) anılan ay tabloların aralığında olmalı, soru, kullanıcı ya
        da bir araç o ayı zaten anmadıysa. Bayraklar cevabın altına
        "Doğrulanamayan ifadeler" olarak eklenir; cevap yeniden yazdırılmaz
        (bir düşünme açık çağrı daha ve iyi bir cevabı bozma riski).
        Göremedikleri modül belgesinde yazılı: tesadüfen tabloda olan yanlış
        sayı, yön/işaret, yanlış aya bağlanmış doğru sayı, yazıyla sayı.
Sonuç:  Üç canlı cevap aynı tablolarla tekrar oynatıldığında yalnızca "27,85"
        işaretleniyor (tests/test_verifier.py); "14-20 bin" gerçek artışları
        (16,6–20,2 bin) kapsayan bir aralık olduğu için geçiyor.

## Modelin gördüğü geçmiş (agent/history.py)
Bağlam: 3. turun beş çağrısı 59,5 bin prompt token'ı okudu, çoğu önceki turların
        aday listeleri ve tablo önizlemeleri. Yavaş MIA gününde her token
        gecikme.
Karar:  `state.messages` hiç kırpılmaz (adım günlüğü ve izlenebilirlik paneli
        onu okur). Modele giden görünümde önceki turlar yalnızca soru + son
        cevap; içinde bulunulan tur tam. Tablolar kaybolmaz: sistem istemi her
        tabloyu ve sütunlarını listeliyor. Görünümde çağrısı olmayan araç
        mesajı gönderilmez (protokol hatası). Cevapsız çağrı bırakan bir tur
        (kartta durdurma, çökme) sonraki tur başlarken kendi bölgesinde
        onarılır; onarım adımı o turun numarasını taşır.

## 22 Eyl incelemesinin düzeltmeleri
Bağlam: Adım 1'in değişiklikleri çok ajanlı bir incelemeden geçti; 49 bulgu
        doğrulandı. Doğrulayıcıyla ilgili olanlar yukarıda. Üçüncü `--real`
        koşusu da bir bulgu ekledi: cevap düşünme açık 45 sn'de zaman aşımına
        uğradı, düşünme kapalı yedeği de yine 45 sn ile sınırlandığı için
        bütçede yaklaşık bir dakika dururken düştü (MIA 7–10 token/sn).
Karar:  Zaman bütçesi:
        - MIA istemcisi verilen bütçeye tam uyar (`LLM_TIMEOUT_S` ile
          kırpmaz); sınırı döngü seçer. Bütçe bir yeniden denemeyi durdurursa
          hata `LlmTimeoutError` olur, 503'ten sonra bile: gözlemdeki
          "elindekiyle cevapla" kurtarması o zaman da çalışır. Yeniden deneme
          başlayamayacaksa beklenmez.
        - Çağrının payı başta bir kez hesaplanır; cevabın 30 sn tabanı artık
          deneme başına değil toplam (eskiden bütçeyi 60 sn aşabiliyordu).
        - Cevabın düşünme kapalı denemesi kalan payın tamamını alır. Oturumun
          son model çağrılarında ölçülen hız düşünme açık cevabı (800 token;
          15 Eyl ölçümü 750) pencereye sığdıramıyorsa cevap baştan düşünme
          kapalı başlar.
        - Başarısız model adımı gerçek süresini ve son denemenin düşünme
          bayrağını raporlar; her türlü hata ya da iptalde adım kartı kapanır.
        - CLI bütün soruları tek olay döngüsünde çalıştırır.
        - Cevap istemine turun ilk 10 uyarısı gider (sonuncular değil), fazlası
          sayılır: birim dönüşümü uyarısı artık sonraki uyarılarla düşmüyor.
        Arama ve onay:
        - Seri seçimi kapısı yalnızca skorlar yakınsa açılır; birim/tür farkı
          tek başına açmaz (havuzda 36 çiftin 31'i farklıydı). Sorguda koduyla
          geçen seriler seçim sayılmaz.
        - Kod eşleşmesi noktalama, kesme işareti ve sütun adı biçimini tanır;
          kelime yedeği Türkçe harfleri katlar, alt çizgiyi böler; kelime
          yedeği boş dönerse araç "havuzda yok" demez.
        - Embedding hatası 180 sn hatırlanır; kapsam okuması yalnızca başarıda
          önbelleğe alınır.
        - Geçersiz/başka karta ait cevap yüzünden varsayılana düşülen karar
          otomatiktir; otomatik kalite kararı "[Otomatik] ..." diye söylenir.
        Araçlar:
        - Oran serisinin farkı birimi "yuzde puan" (türe bakarak; havuzdaki
          faizin birimi "Ağırlıklı ortalama", "%" değil, bu yüzden eski kontrol
          gerçek veride hiç çalışmıyordu).
        - `deflate` yalnızca TL cinsinden, henüz reelleştirilmemiş aylık
          stok/akım tutarına uygulanır (adet ve reel sütun reddedilir); kural
          sahte araçlarla ortak (`deflate_refusal`). Baz ay tablonun dışında
          olabilir; baz ay sütun adına girer (`reel_X_202512`).
        - Aynı adı başka bir hesap tutuyorsa (başka payda, başka koşul) sütun
          `_2` ekiyle eklenir; "aynı işlem yapılmış" yalnızca gerçekten aynı
          işlemse (op, girdiler, formül) söylenir.
        - İşlemin kullanmadığı argüman reddedilir; `period` ya da sayısal
          olmayan sütun anlaşılır bir hatayla reddedilir.
        - `into='active'` boş ayları birleşmiş tabloda sayar, özet tablonun
          aylarını söyler.
        - Haftalık seriler için "ay sonu bakiyesi" iddiası kaldırıldı: hafta
          Pazartesi gününün ayına sayılıyor, uyarı bunu söylüyor; eksik
          haftayla hesaplanan ay (2020-12: 1/4 hafta) ayrıca uyarılıyor.
          Haftayı gözlem tarihine göre aya koymak adım 2'de.
        - `--live` artık gerçek araçları FakeLake üzerinde çalıştırıyor; sahte
          `transform_table` istemin istediği `mark`'ı bilmiyordu.
        - `catalog.duckdb` parquet dosyalarından eskiyse yeniden kuruluyor.
        - planner v3: örnekteki "toplam kredi serisi" ifadesi yanlış seriye
          (tüketici kredileri toplamı) işaret ediyordu; adet sayıları
          reelleştirilmez.


Dördüncü `--real` koşusu (22 Eyl, düzeltmelerden sonra, üç soru tek oturum): üç
tur da kesintisiz bitti (137 / 62 / 26 sn). 1. turda son model çağrılarının
hızı ~26 token/sn ölçüldü, düşünme açık cevap denendi ve 45 sn'de düştü;
düşünme kapalı yedek kalan payın tamamını aldı (~72 sn) ve 31 sn'de bitti.
22 Eylül'deki altı düşünme açık cevap denemesinin altısı da zaman aşımına
uğradı; başarılı tek örnek 15 Eylül'deki. Cevaplar gold'dan kontrol edildi:
- 1. cevap: dört ay doğru; 2025-08..12 aylık artış "14-20 milyar" doğru
  (14,1–20,2). Fakat "2021-2023 boyunca faiz %17-20, bakiye 275-300 milyar"
  2023'ün ikinci yarısı için yanlış (faiz 32,8–42,4; bakiye 447 milyara
  çıkıyor). Doğrulayıcı bunu göremez: aralık dönemin bir kısmı için doğru,
  yanlış olan bağlandığı dönem (modül belgesindeki kör nokta).
- 2. cevap: sayılar doğru, ama "1.716.202" (gerçek 1.716.202,69, yuvarlanmamış
  kırpılmış) işaretlendi. Yanlış alarm; doğrulayıcı artık kırpılmış sayıyı da
  kabul ediyor (son hanede bir birim yukarısı).
- 3. cevap: doğru.
Gözlem çağrısı yine uzun bir cevap yazdı (734 token, 30 sn; planner 14. kurala
rağmen).
