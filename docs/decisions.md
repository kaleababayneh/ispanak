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
  için zararsız ama 5. aşamada model çağıracak ve arayüzü dondururdu.
- `describe_table` okunamayan bir dönem sütunu yüzünden başarılı bir adımı
  düşüremiyor.
- Başarısız adımlar da `step_start`/`step_end` yayınlıyor. Hatalı argüman ve
  iptal edilen çekim jürinin en çok göreceği iki şey; olay yayınlanmayınca
  arayüzde kapanmayan bir adım kartı, yani donmuş bir ekran kalıyordu.
- `llm.chat` artık isimli argümanlarla çağrılıyor.
