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
her tur sıfırlanıyor, `state.steps` sıfırlanmıyor. `answer.md`'nin o günkü 8.
kuralı (bugün 9.) modele
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
          rebase, ratio, deflate, mark. Hepsi yeni sütun ekler; var olan adı
          **aynı hesap** tutuyorsa reddeder, başka bir hesap tutuyorsa sütun
          `_2` ekiyle eklenir (aynı günün ilerleyen maddesi). `mark` koşulu
          sağlayan ayları özet metninde listeler: "hangi aylar" sorusunu model
          saymaz (planner kural 1).
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

## DeepSeek-V4.1-Flash ölçümü (22 Eyl, karar bekliyor)
Bağlam: Slack: Kloudeks MIA artık `deepseek-ai/DeepSeek-V4.1-Flash` de sunuyor ve
        demoda kullanılabilir. Varsayılan sohbet modeli hâlâ Qwen3.8-27B.
Ölçüm:  - `models.list` modeli döndürüyor. Hız 200–310 token/sn (Qwen 22 Eyl'de
          7–25). `enable_thinking` anahtarı çalışıyor; açıkken yanıtta `reasoning`
          alanı geliyor (Qwen'deki gibi, geri gönderilmiyor). Araç çağırma doğru.
        - İlk `--real` koşusu: araç adımları doğru ve hızlı (1. tur 8,6 sn), ama
          cevaplar bozuk (16 token anlamsız metin, bir turda ham JSON). Neden:
          cevap isteği gözlem notuyla ("Veriler hazır.") yani bir asistan
          mesajıyla bitiyordu; DeepSeek şablonu bunu yeni tur değil devam
          edilecek mesaj sayıyor ve düşünmeden devam ediyor.
        - Düzeltme modelden bağımsız: cevap isteği asistanın metin mesajıyla
          bitmez (`loop._answer_messages`). Qwen'de davranış değişmedi (aynı
          istekle ölçüldü).
        - Düzeltmeden sonra üç tur 10 / 3 / 6,6 sn (Qwen: 137 / 62 / 26 sn). Dört
          ay doğru; cevaptaki bütün sayılar gold'dan doğrulandı (0,35 / 0,28 /
          2,03 puan düşüş, %18,3 → %37,1, 276.740 → 678.912 milyon TL, KFE
          16,49 → 204,36). Nedensellik dili temkinli; doğrulayıcı bayrak yok.
Karar:  Kullanıcıya bırakıldı; 22 Eyl akşamı karar verildi, aşağıya bakın.

## Havuz genişletme (adım 2): EVDS'nin tamamı ve BDDK'nın üç bülteni
Bağlam: Gold 9 seriydi; gereksinim (kkb-bilgiler.md) EVDS ve BDDK'nın geniş
        kapsamını istiyor, demo soruları senaryonun dışına çıkabilir.
Karar:  - EVDS: bronze kataloğundaki her seri, frekansı sözleşmedeki D/W/M/Q/A
          ise. "AYDA İKİ KEZ" ve "ALTI AYLIK" dışarıda (`gold.EXCLUDED_FREQUENCY_STRS`):
          sözleşmenin `freq` kümesinde karşılıkları yok, dönem kuralı tanımsız.
        - Çekim veri grubu × takvim yılı. Zaman aşımına uğrayan yıl çeyreklere,
          çeyrek aylara bölünür (`evds._Window.split`); boş yıl `.empty.json` ile
          hatırlanır, ikinci koşu ona istek atmaz. Ay da zaman aşımına uğrarsa
          başarısız sayılır, raporlanır.
        - BDDK Aylık: 17 tablonun bütün satırları ve bütün değer sütunları,
          sektör toplamı, TL raporu (`ingest/bddk_tables.yaml`). Senaryonun üç
          kimliği `aliases` ile korunur.
        - BDDK Haftalık: 9 tablo, her hafta. JSON uç noktası yok; sayfa oturumla
          çalışan HTML. Hafta ve tablo form token'larıyla seçilir, tablo öğesi
          olduğu gibi saklanır. Yazmadan önce başlıktaki tarih ve tablo adı
          istenenle karşılaştırılır: başka haftayı döndüren oturum yanlış dosyaya
          yazılmasın. Zaman serisi uç noktası (KiyaslamaJsonGetir) her çağrıda
          13 hafta döndürdüğü için kullanılmadı (~23 çağrı/seri).
        - BDDK FinTürk: 7 tablo, çeyreklik, 82 il ("YURT DIŞI" dahil), tek
          çağrıda bütün iller.
Sonuç:  Seri ve boyut sayıları "Havuz ölçümü" girişinde.

## Frekans çevrimini agg_method belirler; kind yalnızca anlam
Bağlam: `to_freq` yöntemi kind'dan türetiyordu (stok → son, akım → toplam,
        oran/endeks → ortalama). Havuzda bu eşleme tutmayan seriler çok: dönem
        sonu yayımlanan kurlar, ay sonu faizler, EVDS'nin "KÜMÜLATİF" önerdiği
        akımlar. EVDS her seri için kendi önerisini veriyor (`DEFAULT_AGG_METHOD`).
Karar:  Sözleşme v0.5: `agg_method` çevrimi belirler (EVDS'de kaynağın önerisi,
        BDDK'da tablo tanımı). `kind` cevabın dilini belirler: fark yüzde puan mı,
        enflasyondan arındırılabilir mi. `to_freq(df, kind, freq, *, agg_method)`:
        agg_method verilmezse kind'ın varsayılanı.
Sonuç:  Senaryonun dört serisinde yöntem değişmedi (KTF12 ort., HPBITABLO6.3 son,
        AKONUTSAT toplam, TÜFE/KFE ort.); sonuç ayları aynı.

## Belirsiz seride kind boş (onaylandı 22 Eyl)
Bağlam: Kurallar (`lake/classify.py`: birim → EVDS toplama önerisi → ad) bazı
        serileri hiçbir türe koyamıyor: birimi yazmayan, adı tür söylemeyen,
        ortalama önerilen seriler. BDDK oran tablolarında da "Bin TL" ya da "Gün"
        birimli değerler (çalışan başına aktif, ortalama vade) var.
Karar:  Tahmin yok: `kind` boş, gerekçe `classification.parquet`'te
        (`kind_reason`), liste `docs/kind_review.md`'de. Çevrim bundan
        etkilenmez (agg_method dolu). Etkilenen: `deflate` boş türü reddeder,
        cevap metni türe göre konuşmaz.
Sonuç:  Sözleşmede "boş olabilir" notu var (contracts.md §1). Kullanıcı onayladı
        (22 Eyl): tür uydurmak yerine boş kalır.

## obs_date: haftalık değer gözlem gününe göre aya düşer
Bağlam: Gold haftayı Pazartesi'sine yazıyordu. Pazartesi'si ayın son günlerine,
        Cuma'sı sonraki aya düşen hafta önceki ayın "ay sonu" oluyordu: bir ayın
        değeri o aydan sonra gözlenmiş bir değerdi (19 Eyl çapraz kaynak
        kontrolünde 2025'in 12 ayının 6'sı).
Karar:  Sözleşme v0.5: observations'a `obs_date` (kaynağın değere verdiği gün).
        `period` sözleşmedeki anlamıyla kalıyor. `to_freq` ve pencere kesimi
        `obs_date`'e göre. Aylık ve daha seyrek seride `obs_date = period`.
Sonuç:  Senaryonun sonuç ayları değişmedi (2021-02, 2021-05, 2023-12, 2024-01).
        Beklentinin tersine BDDK ay sonu değerine uyum biraz azaldı (ortalama
        mutlak fark %0,14 → %0,20): ayın son Cuma'sı ay sonundan ortalama 2,8
        gün önce, eski yöntemin seçtiği Cuma 1,9 gün uzakta ama bazen sonraki
        ayda. Kazanç yakınlık değil doğruluk: ay değeri artık ay sonrasından
        gelmiyor. Ayrıntı: data_dictionary.md, "Çapraz kaynak kontrolü".

## BDDK kümülatif kalemler silver'da akıma çevriliyor
Bağlam: BDDK kâr-zarar tablosunun (tablo 2) değerleri yıl başından kümülatif.
        Veriyle doğrulandı: bütün kalemler her Ocak'ta sıfırdan başlıyor,
        bilanço kalemleri başlamıyor.
Karar:  Silver'da aylık akım: akım_t = küm_t − küm_t−1, Ocak'ta akım = küm
        (`transforms.decumulate_ytd`). Önceki ay yoksa o ayın akımı boş kalır,
        doldurulmaz. `cum_type = ytd` kaynağın yayımladığı biçimi kaydeder.
        Yıl başından toplam gerekince `transform_table` `ytd` işlemi yeniden
        biriktirir (`transforms.year_to_date`, yalnızca aylık akım).
Sonuç:  Toplamalar (çeyrek, yıl) akım üzerinden doğru; ham kümülatif değerle
        toplamak yıl içi değerleri defalarca saymak olurdu.

## BDDK satır kimlikleri formülsüz etiketten
Bağlam: BDDK satır etiketlerinde formül var ("Tüketici Kredileri (2+3+4)") ve
        satırlar yeniden numaralandığında formül değişiyor
        ("(10+27+28)" → "(10+30+31)"): aynı kalem iki seri olurdu. Mevduat
        tablolarında "a) Gerçek Kişiler" altı farklı başlık altında tekrarlanıyor.
Karar:  Kimlik = tablo anahtarı + formülü atılmış etiket + sütun
        (`bddk.label_key`, `slug`). Liste maddeleri ("a) ...") ve tabloda tekrar
        eden etiketler üstlerindeki formül satırıyla, yoksa en yakın kalın
        satırla nitelenir, gerekirse yinelemeli. Aylık bültende buna rağmen aynı
        kimliğe düşen iki satır sessizce silinmez, satır numarası eklenerek ikisi
        de tutulur; haftalık ve FinTürk'te çakışan satır sayılır ve kurulum
        çıktısında raporlanır.
Sonuç:  Aylık 2045 seri, haftalık 535 seri, çakışmadan düşen satır 0.
        BDDK'nın kendi tanım değişiklikleri ayrı seri kalıyor: haftalık menkul
        değerler tablosunda Eylül 2022'de "a) Kamu borçlanma senetleri /
        b) Diğer" kırılımı "Devlet tahvilleri / Kira sertifikaları (bilgi için)"
        oldu. Eski kırılım 88 hafta, yenisi 198 hafta; ana toplamlar 286 haftanın
        hepsinde dolu. Kur korumalı mevduat Şubat 2022'de, kredili mevduat
        hesabı kırılımı Eylül 2022'de başlıyor.

## Havuz araması: sıralama kuralları (22 Eyl)
Bağlam: 9 serilik havuzda kosinüs benzerliği tek başına yetiyordu. 45 bin seride
        yetmiyor: sorgu setinin (eval/search_queries.yaml) 21 sorgusunun yalnızca
        10'unda doğru seri ilk 5 adayda çıkıyordu. Üç tekrarlayan hata:
        81 ilin aynı kalemi ulusal serinin önüne geçiyor; EVDS'in "(Arşiv)" diye
        işaretlediği, yerine yenisi gelmiş seri güncelin önüne geçiyor; bir
        tablonun ana serisi kendi alt kalemlerine yeniliyor ("tüketici fiyat
        endeksi" sorgusunda TÜFE Genel Endeks kosinüste 232. sırada).
Karar:  Skor kosinüs benzerliği olarak KALIYOR (eşik ve onay kapısı onu okuyor);
        değişen yalnızca sıra ve gruplama (`lake/pool_index.rank`). Kurallar
        sorgunun konusuna değil, kaynağın seri hakkında söylediğine bakar:
        - Arama metnine veri grubunun adı eklendi ("Genel Endeks" →
          "Tüketici Fiyat Endeksi (2025=100) | Genel Endeks").
        - Sorgu yer adı söylemiyorsa (ya da "illere göre" demiyorsa) il/bölge
          serileri ulusal serilerden sonra gelir; sorgu bir il söylüyorsa o ilin
          serisi öne gelir.
        - Arşiv serisi, sorgu "arşiv" demedikçe güncellerden sonra gelir.
        - Aynı kalemin il kırılımları tek adayda toplanır; aday, kaç seriyi
          temsil ettiğini uyarı olarak söyler.
        - Sorgu bir veri grubunun adının tamamını söylüyorsa o grubun ana serisi
          (EVDS'in ilk listelediği seri: "Genel Endeks", "Toplam Sanayi",
          "Türkiye") öne gelir, alt kalemleri tek adayda toplanır.
        - Aday havuzu 500 en yakın seriye çıkarıldı (`SEARCH_DEPTH`); ana seri
          kendi alt kalemlerinin yüzlercesinin altında kalabiliyor.
        - Aynı kurallar kelime yedeğine de uygulanır; orada sıra artık tam metin
          indeksinin (BM25) sırası, "eşleşen kelime oranı" değil (45 binde yüzlerce
          seri aynı oranı alıyor ve sıra katalog sırasına kalıyordu).
Sonuç:  İlk 5 isabet 10/21 → 25/27. Kurallar tasarlandıktan SONRA eklenen ve
        tasarımda kullanılmayan 6 ayrık sorguda 6/6 (beklenenler arama sonucuna
        bakılmadan katalogdan seçildi) — kurallar sorgu setine özel değil.
        Kalan iki hata kural değil anlam sorunu, uydurulmadı: "enflasyon"
        (TÜFE'nin metninde bu kelime geçmiyor; "TÜFE genel endeks" çalışıyor) ve
        "dolar kuru" (13. sıra; "ABD doları kuru" 1. sıra). Arama 0,25 sn.

## Eşik (MIN_SIMILARITY) 0,40'ta kalıyor
Bağlam: 45 binlik havuzda yeniden ölçüldü. İlgili sorgularda doğru serinin
        skoru 0,48–0,78. İlgisiz beş sorgunun en iyi adayı: futbol 0,359,
        hava durumu 0,373, üniversite sınavı 0,382, otomobil yakıt tüketimi
        0,486, sinema gişe hasılatı 0,579.
Karar:  Eşik 0,40. Ayıran temiz bir değer yok: 0,50'ye çıkarmak "dolar kuru"nun
        doğru serisini (0,484) elerdi ama sinemayı elemezdi.
Sonuç:  Eşik bariz ilgisizi eler; kalanı adayların adına, türüne ve kapsamına
        bakarak modelin ayırması gerekir (planner kural 12). `SEARCH_CANDIDATES`
        5'te kaldı: ölçüm ilk 5 üzerinden yapıldı ve 25/27 onunla elde edildi.

## Kapsam sorgusu modelin kapısından geçmiyor
Bağlam: Aday listesindeki "kapsam" (serinin havuzdaki ilk–son dönemi) 45 binlik
        havuzda boş geliyordu. Neden: adaptör kapsamı `gold.run_sql` ile
        okuyordu ve oradaki zorunlu `LIMIT 1000` (güvenlik kuralı) sonucu
        kesiyordu; ilk bin serinin dışındaki her aday kapsamsız görünüyordu.
        Model o zaman serinin havuzda hangi dönemleri kapsadığını göremiyor
        (planner kural 12 tam da buna bakmasını söylüyor).
Karar:  `lake/gold.py`'ye `coverage()` eklendi: aynı salt-okunur bağlantı,
        LIMIT yok. `run_sql`'in sınırı olduğu gibi duruyor — o sınır modelin
        çekebileceği satır için, lake'in kendi okumaları için değil.
Sonuç:  45.422 serinin hepsi kapsamıyla geliyor. Kapsam artık `obs_date`'e göre
        (pencere ve `get_series` de öyle).

## Büyük havuzda canlı koşu: planner v5 ve adım bütçesi (22 Eyl)
Bağlam: Senaryonun üç sorusu 45 bin serilik havuzda `--real` koşuldu (Qwen).
        1. tur cevapsız bitti: model "60 aylık"ı kredi VADESİ sanıp havuzda vade
        kırılımı aradı, 8 arama hakkının 6'sını buna harcadı, doğru serileri
        7. ve 8. aramada 1. sırada buldu ama veri çekecek adım kalmadı
        (`kesinti step_limit`). Havuz 9 seriyken ilk arama zaten tek doğruyu
        veriyordu; bu, genişlemenin getirdiği yeni bir hata biçimi.
Karar:  planner.md v5, iki kural: (4) sorudaki süre ifadesi ("60 aylık") kaç
        dönem istendiğini söyler, kredinin vadesi değildir; (12) aynı veri için
        ikiden fazla arama yapma — her arama bir adımdır, arayıp durursan veri
        çekecek adım kalmaz. Kural 12'ye ayrıca havuz ölçeği ve il/ulusal ayrımı
        eklendi.
Sonuç:  Aynı koşu: 1. tur 2 arama + get_series + 3 dönüşüm ile tabloyu kurdu
        (işaretli aylar: 2021-02, 2021-05, 2023-12, 2024-01 — senaryonun bilinen
        sonucu). `max_steps = 8` değiştirilmedi; bu bir döngü politikası kararı,
        kullanıcıya soruluyor.

## DeepSeek-V4.1-Flash: büyük havuzda ikinci ölçüm (22 Eyl, karar hâlâ kullanıcıda)
Ölçüm:  Aynı üç soru, aynı havuz, tek fark model.
        - Qwen3.8-27B (varsayılan): 1. turda araçlar doğru çalıştı ama cevap
          adımı 47 sn sonra BOŞ içerik döndürdü (`kesinti llm_error`) — modelin
          bilinen "düşünüyor, metin üretmiyor" hatası. Tur 78 sn. 2. turda
          cevap birimi yanlış aktardı: seri milyon TL iken "1,7 milyon TL" dedi.
        - DeepSeek-V4.1-Flash: üç tur da tamamlandı, 1. tur 15 sn, doğru dört ay
          listelendi, birimler doğru ("276 bin milyon TL"). 2. ve 3. turda
          cevapta kendi hesapladığı iki oran ("12,4 kat", "1 milyon 716 bin")
          doğrulayıcı tarafından yakalandı (`number_not_found`) — koruma
          çalışıyor ama model planner kural 1'i çiğniyor.
Karar:  Varsayılan model **DeepSeek-V4.1-Flash** oldu (kullanıcı kararı, 22 Eyl).
        Qwen bir ortam değişkeni uzakta: `MIA_CHAT_MODEL=kkbhackathon2026/Qwen3.8-27B`.
        Embedding modeli değişmedi (Qwen3-Embedding-8B); havuz indeksi onunla kurulu.

## Adım 1'den kalan iki soru, DeepSeek'le yeniden ölçüldü (22 Eyl)
Bağlam: İkisi de Qwen'in yavaşlığı yüzünden açık kalmıştı: cevap adımında
        düşünme açık mı kalsın, observe çıktısına token sınırı konsun mu?
Ölçüm:  - Cevap adımı, düşünme AÇIK: DeepSeek'te üç tur 3,2 / 2,2 / 2,6 sn
          (Qwen'de 47 / 62 / 6,5 sn ve bir turda boş cevap). Sorun modeldi,
          ayar değil.
        - Observe adımı: DeepSeek iki kez cevabın tamamını observe adımında
          yazdı (696 ve 352 completion token) ve aynı metni cevapta tekrarladı.
          Qwen'de bu adımlar 49 token ortalamaydı; yani sınır Qwen'le değil,
          DeepSeek'le gerekli hale geldi.
Karar:  - Cevapta düşünme AÇIK kalıyor (`thinking = {plan: True, observe: False,
          answer: True}`), değişiklik yok.
        - `AgentConfig.observe_max_tokens = 300`. Sınır yalnızca observe
          çağrısına uygulanır. Çağrı sınıra takılır VE içinde araç çağrısı
          varsa çağrı sınırsız tekrarlanır: yarım kalan bir araç çağrısı,
          kazanılan metinden pahalıdır.
Sonuç:  Plan ve cevap çağrıları sınırsız kalıyor; iki test bunu koruyor
        (tests/test_loop.py).

## Adım sınırı 8 → 12 (22 Eyl, kullanıcı onayı)
Bağlam: 45 bin serilik havuzda arama adım harcıyor; bir tur 8 adımın 6'sını
        aramaya harcayıp veri çekemeden bitti.
Karar:  `AgentConfig.max_steps = 12`. Bu bir tavan, hedef değil: senaryonun
        turları hâlâ 2–4 adımda bitiyor, dolayısıyla iyi giden tur uzamıyor.
        Tur süresi zaten `total_budget_s` (180 sn) ile sınırlı; adım sayısı
        onu aşamaz.

## Web URL aracı (`read_url`, 22 Eyl)
Bağlam: Gereksinim demo günü prompt'ta URL verileceğini açıkça söylüyor ve iki
        örnek adres paylaşmış. Havuzda bu yolun hiçbir kodu yoktu.
Ölçüm:  Adresler önce elle incelendi (ikisi de kamuya açık):
        - `borsaistanbul.com/endeks/xtumy` sunucuda render ediliyor; metaveri
          tablosu okunuyor (XTUMY'nin adı **BIST TUM-100**, BIST 100 değil),
          ama "Güncel Endeks Değerleri" tablosu HTML'de boş — onu JavaScript
          dolduruyor. Sayfanın JS paketlerinde veri uç noktası görünmüyor.
          Endeksin tarihsel serisi zaten havuzda (`evds:TP.MK.F.BILESIK`,
          `...BILESIK.TUM`).
        - Kıymetli madenler sayfasında tablo yok, 21 PDF bağlantısı var; veri
          `kmp_au.pdf`'te. PDF'in metin katmanı sağlam ama tablosunda çizgi yok:
          pdfplumber'ın varsayılan (çizgi tabanlı) çıkarımı tek sütun döndürdü,
          metin hizalı çıkarım 11 sütunlu doğru tabloyu verdi.
Karar:  - Playwright yok. İki hedef de sunucuda render ediliyor; JavaScript'li
          sayfalarda önce sayfaya gömülü JSON denenir, o da yoksa araç durumu
          dürüstçe söyler. Tarayıcı kararı Docker imajı kurulurken verilir.
        - PDF'te çizgi tabanlı çıkarım bozuksa metin hizalı çıkarıma düşülür;
          başlık, ilk sayı satırına kadarki satırların sütun sütun birleşimidir
          (basılı tabloların başlığı üç satır olabiliyor).
        - Bir belgeden çıkan her veri tablosu ayrı oturum tablosu olur (en çok
          10). Tek tablo alıp gerisini özetlemek bilgi kaybı; hepsini almak da
          yerleşim tablolarıyla oturumu şişirir — eleme + üst sınır ikisini
          birden çözüyor.
        - `prepare_table` genişletildi (plan 3.9'un istediği gibi): tarih sütunu
          adı ne olursa olsun bulunur ve `period` olur, Türkçe sayı biçimi
          çevrilir, ikisi de kalite kartına yazılır.
        - `transform_table`'a `top` işlemi eklendi. "Hangi ay en yüksek"
          sorusunu bir araç cevaplamalı (planner kural 1) ve bu işlem `period`
          sütunu olmayan tablolarda da çalışmalı.
Sonuç:  Canlı koşu: soru bir sayfa adresiyle geldi → araç sayfayı okudu, PDF
        bağlantısını listeledi → PDF'i okudu (13 satır: 12 ay + toplam) → `top`
        ile sıraladı → cevap: "Mart / March: 151.349.741.325 TL", toplam satırı
        ayrı belirtilmiş, kaynak adresle verilmiş, doğrulayıcı bayrağı yok.
        Bilinen eksik: "Ocak / January" gibi ay adları tarihe çevrilmiyor (yıl
        tablonun içinde değil, başlığında), bu yüzden böyle bir tablo zaman
        serisi araçlarına (anomali, kırılma) girmiyor; `top` ve `mark` çalışır.

## Testler makinedeki havuz indeksini kullanamaz (22 Eyl, CI hatası)
Bağlam: PR #11'in ilk CI koşusunda iki test kırmızı döndü, oysa ikisi de yerelde
        yeşildi. Sebep: `data/pool_index` türetilmiş ve gitignore'da — indeksi
        kuran makinede var, CI'da yok. Kelime yedeği indeks varsa tam metin
        (BM25) sırasını kullanıyor, yoksa kataloğu baştan tarıyor; iki sıralama
        farklı olduğu için "doğru seri ilk beşte" diyen iki iddia ortamına göre
        değişiyordu.
Karar:  `tests/conftest.py`'de oturum kapsamlı autouse fikstür
        `lake_adapter.POOL_INDEX_PATH`'i var olmayan bir yola çeviriyor. İndeksi
        sınayan testler kendi indekslerini kurup `index_path` veriyor.
        Kapsam oturum: test kapsamlı bir yama, modül kapsamlı fikstürler (senaryo
        oturumu) adapter'ını zaten kurduktan sonra uygulanıyordu — senaryo bu
        yüzden gerçek indeksi okumaya devam ediyordu.
        İki testin iddiaları da indeksten bağımsız hale getirildi: kelime
        yedeğinde hangi serilerin çıktığı değil, aday döndüğü, adayların
        "kelime" diye işaretlendiği ve "havuzda yok" denmediği sınanıyor. Belirli
        bir seriye ulaşmanın ortamdan bağımsız yolu kimliğiyle aramaktır; test
        artık onu gösteriyor (`eslesme: kod`).
Sonuç:  Havuz indeksi, indirilen dosyalar ve DuckDB dosyası silinerek koşuldu:
        867 test geçti. Kural: repoda olmayan bir dosyaya bağlı test yazılmaz;
        gold repoda olduğu için gerçek veriyle test yazılabilir, indeks değil.

## Öncüllük aracı (`test_causality`, 22 Eyl)
Bağlam: Altı aracın beşincisi ve örnek senaryonun üçüncü sorusu ("faiz düştüğü
        halde kredilerin artmamasının sebebi fiyat artışı olabilir mi?") tam
        olarak bunu soruyor. Araç yokken agent nedensellikten söz etmiyordu —
        doğru davranış, ama ölçemediği için.
Karar:  İki ölçüm birlikte veriliyor, çünkü farklı sorulara cevap veriyorlar:
        Granger "bu tesadüf mü?" (iki yönde, 1..n gecikme, ssr F testi),
        gecikmeli korelasyon "ne kadar güçlü ve hangi gecikmede?". p-değeri
        büyüklük hakkında bir şey söylemez; korelasyon tablosu okunur olanı.
        Durağanlaştırma zorunlu (trendli iki seri konusu ne olursa olsun
        ilişkili çıkar); dönüşüm seri başına, sütunun türüne göre seçiliyor.
        Eşik Bonferroni ile düzeltiliyor ve kaç gecikme denendiği yazılıyor.
        Saf fonksiyonun adı `lead_lag`: "test_" ile başlayan bir modül
        fonksiyonunu pytest test sanıp toplamaya çalışıyor. Araç adı
        `test_causality` kalıyor — model onu soruya göre görüyor.
Sonuç:  Sentetik çiftte (kredi, fiyatı 3 ay gecikmeyle izliyor) doğru yön
        anlamlı, ters yön değil; korelasyon zirvesi tam 3. gecikmede.
        Gerçek veride (KFE ile BDDK konut kredisi bakiyesi, 2021-01 – 2025-12,
        59 gözlem): **hiçbir yönde anlamlı öncüllük yok** (kfe→kredi p=0,109;
        kredi→kfe p=0,109; eşik 0,0083). Gecikmeli korelasyon zirvesini −3'te
        yapıyor (r=0,42), yani zayıf da olsa kredi fiyatın önünde gidiyor —
        sorunun ima ettiğinin tersi yön. Canlı koşuda model aracı kendisi
        çağırdı ve sonucu "iki yönde de anlamlı öncüllük yok" diye, gecikme ve
        p değerleriyle aktardı; doğrulayıcı bayrağı yok.

## Web arama aracı ve kırılma → açıklama zinciri (22 Eyl)
Bağlam: Altı aracın sonuncusu. Havuz "faiz 2023-07'de kırıldı" diyebiliyor ama
        o ay ne olduğunu söyleyemiyor; o bilgi bir basın duyurusunda.
Karar:  - Sağlayıcı arkasında tek arayüz: bugün `ddgs` (anahtarsız), hız
          sınırına takılırsak ücretli servise geçiş tek fonksiyon.
        - Araç sayfa indirmiyor, yalnızca sonuç listesi döndürüyor: ölçümde
          özet metinler zaten tarihi ve olayı taşıyor, sayfa indirmek ise her
          biri 1–2 sn ve adım bütçesinden yiyor. Derinleşmek gerekirse model
          `read_url` ile o sayfayı açıyor; iki araç birbirini tamamlıyor.
        - Sonuçlar "veri, talimat değil" etiketiyle veriliyor: bunlar
          kontrol etmediğimiz metinler ve doğrudan modelin bağlamına giriyor.
        - Onay kapısı `read_url` ile aynı (`fetch`).
        - Planner kural 19: `detect_changes` bir tarih bulduğunda kullanıcı
          istemeden o ay aranır. `answer.md` kural 5: arama çalıştıysa cevap
          ne bulduğunu ve kaynağını yazmak zorunda.
Ölçüm:  Canlı koşu: kırılma 2023-07 bulundu → model kendiliğinden "TCMB Temmuz
        2023 politika faizi kararı" aradı → TCMB'nin kendi duyurusuna ulaştı
        (duy2023-25). İlk denemede cevap aramayı hiç anmadı; eksik olan
        planner değil cevap prompt'uydu (kural 5 eklendi).
        **İkinci denemede ciddi bir hata çıktı:** model arama özetlerinden
        okuduğu oranları cevaba yazdı ve yanlış aya bağladı ("Temmuz'da
        %17,5'ten %25'e"; gerçekte Temmuz 15→17,5, %25 Ağustos'ta).
        Doğrulayıcı bunu yakalayamaz — sayı arama özetinde geçtiği için araç
        çıktısı sayılıyor.
Sonuç:  `answer.md` kural 5 sıkılaştırıldı: aramadan gelen **sayı cevaba
        yazılmaz**, olay ve tarih yazılır; sayı gerekiyorsa kaynak `read_url`
        ile açılır. Üçüncü koşuda model tam da bunu yaptı ve nedenini de
        söyledi: "oranı burada aktarmıyorum çünkü arama özetinden gelen
        sayılar hangi tarihe ait olduğu konusunda güvenilir değil."
        Kalan risk: bu bir prompt kuralı, kod kısıtı değil. Kod tarafında
        arama sonucu zaten tablo/sayı üretmiyor.

## Cevapta kaynak: uydurmanın yanına sessizliği de koyduk (22 Eyl)
Bağlam: Doğrulayıcı `source_not_listed` ile uydurma kaynağı yakalıyordu:
        cevapta geçen her seri kimliği, kodu ve URL oturumun kaynak listesinde
        olmalı. Ters yön açıktı — cevap hiç kaynak yazmasa bayrak kalkmıyordu.
        Ölçüldü: kırılma + arama koşusunda model TCMB duyurusunu buldu ama
        cevapta ne bulduğunu ne de adresi andı; cevap, bilgi havuzdan gelmiş
        gibi okunuyordu.
Karar:  `verifier._check_web_cited`: tur içinde `read_url` ya da `web_search`
        başarıyla çalıştıysa, cevapta okunan adreslerden en az biri anılmalı.
        Yeni bayrak türü `source_not_cited` (sözleşme §7.4).
        Eşleşme cömert: adresin tamamı, şemasız hali, dosya adı ya da alan adı.
        Gerekçe: bayrak sessizlik içindir, biçim için değil — modeller uzun
        adresleri "tcmb.gov.tr/…/duy2023-25" diye kısaltıyor ve bu da okuyucunun
        izleyebileceği bir kaynaktır. Yalnız bu turun çekimleri sayılır; bir
        önceki turda yapılan arama bu turu bağlamaz.
        `answer.md` kural 4 de sıkılaştırıldı: adres olduğu gibi yazılır,
        yalnız dosya adı yetmez.
Sonuç:  Altı test (anmama, şemasız adres, dosya adı, kısaltılmış adres, web
        okumayan tur, başarısız çekim). Canlı koşuda cevap TCMB duyurusunu ve
        PDF'i adlarıyla andı, bayrak yok.

## Kaynaklar cevabın içinde değil, yanında (23 Eyl)
Bağlam: Cevaptaki adres modelin yazdığıdır ve model uzun adresi kısaltıyor
        ("tcmb.gov.tr/…/duy2023-25"): okunur ama tıklanamaz. Tam liste zaten
        `sources` olayında var, ama ekrana basan bir yer yoktu.
Karar:  Tur özetinde (`_summarise`, stderr) "KAYNAKLAR (bakilan)" bloğu:
        oturumun bütün kaynakları, seri kimliği ya da tam URL ve adıyla.
        Cevabın **metnine eklenmiyor**: cevap modelin yazdığı şeydir, ayrıca
        kendi bastığımız adresleri doğrulayıcıya kaynak diye geri vermiş
        olurduk (bayrak sırası: doğrula, sonra bas).
        Başlık "bakilan": arama beş sonucun hepsini kaynak olarak kaydeder,
        cevap ikisine dayanabilir; ikisini aynı adla anmak yanıltıcı olurdu.
        Arayüz geldiğinde aynı olayı panelde tıklanabilir gösterecek; bu blok
        o zamana kadar tek görünür yer ve atılacak bir şey değil (CLI demo
        yüzeyi).
Sonuç:  Cevapsız biten turda da basılıyor — nitekim ölçüm koşularından birinde
        cevap boş döndü ve kaynaklar yine göründü. İki test (tam adres basılır,
        kaynak yoksa başlık basılmaz).

## 23 Eylül incelemesinin düzeltmeleri
Bağlam: Adım 1 (`feat/real-tools`) dört ayrı incelemeden geçti. Bulguların
        üçü ölçülerek doğrulandı ve testle sabitlendi; kalanlar aşağıda
        "bilerek değiştirilmedi" başlığında.
Karar:  Süre bütçesi:
        - Düşünme kapalı yedek cevap artık **`MIN_ATTEMPT_S` (5 sn)** kalmışsa
          çalışıyor, `min_call_s` (10 sn) değil. Düşünme açık deneme
          `pay - min_call_s` ile sınırlı olduğu için zaman aşımını her zaman
          bir miktar aşıyor ve `left()` tam da 10 sn'nin altına düşüyordu:
          gerçek saatle yedek **hiç** çalışmıyordu. Yalnızca payı 55 sn'nin
          altına düşen turlarda, yani turun 180 sn'nin 125'ini harcadığı
          durumda görünüyordu, ki yedeğin var olma sebebi tam o tur. Test
          sahte saati zaman aşımı kadar ilerlettiği için bunu göremiyordu;
          yeni test bir miktar aşım ekliyor.
        - Tur durdurulduğunda (`CancelledError`) `done` olayı artık
          `truncated_reason="cancelled"` taşıyor ve bir `error` olayı
          yayımlanıyor. Cancellation bir `Exception` olmadığı için bitmiş bir
          turdan ayırt edilemiyordu. `TruncatedReason`'a `cancelled` eklendi.
        - `draft_answer` tur başında temizleniyor; yarıda kalan tur öncekinin
          cevabını taşıyordu.
        - Doğrulayıcı `_verify` içinde try/except ile sarıldı. Cevap zaten
          yazılmışken bir kural motorunun çökmesi bitmiş işi çöpe atıyordu.
        Araçlar:
        - `into='active'` eklenen sütunun önündeki boşluk için artık "seri
          havuzda X itibariyla basliyor" demiyor; o boşluk istenen pencerenin
          sonucu. Havuzun 2021'den tuttuğu bir seri için "2024-01'de başlıyor"
          diyordu, ki planner 5. kuralın yasakladığı yanlış "bizde yok".
        - `mark` ile üretilen işaret sütunu aritmetik işlemlerde reddediliyor
          (`pct_change` yakalanmayan `ZeroDivisionError` veriyordu, `diff`
          anlamsız bir sütun ekliyordu); `mark` koşullarında hâlâ kullanılıyor.
          İşlem hataları `TypeError` ve `ZeroDivisionError`'ı da yakalıyor.
        - `deflate` tablonun tamamı TÜFE kapsamı dışındaysa `ok=False`
          dönüyor; tamamen boş bir sütunu başarı saymak modeli olmayan reel
          tutarları anlatmaya davet ediyordu.
        - `gold.connect()` yenileme başarısız olursa eski kopyayla devam
          ediyor: DuckDB aynı dosyayı aynı süreçte farklı yapılandırmayla
          açmayı reddediyor (ölçüldü, 1.5.5) ve adım zaman aşımına uğramış bir
          araç hâlâ okurken bu mümkün.
        - `llm.parse_tool_calls` kimliksiz gelen çağrıyı kendi verdiği
          kimlikle (`call_0`) geri yazıyor; aksi halde araç cevabı hiçbir
          çağrıya ait olmuyor ve geçmiş görünümünden düşüyordu.
        Doğrulayıcı (yanlış alarmlar, hepsi ölçüldü):
        - Sayı ile sınır kelimesi arasına giren kelimeler okunuyor: "200
          seviyesinin üzerine çıktı", "200 eşiğini aştı", "200'den daha fazla"
          artık sınır sayılıyor. Tek kelimeye bakmak bunları kesin iddia
          yapıp doğru cümleyi işaretliyordu.
        - Olumsuz sınırlar ("geçmedi", "aşmadı", "altına inmedi") hiç
          değerlendirilmiyor. Sınır gibi okunduklarında tersini söylüyorlardı;
          ifade edilemeyen iddia yargılanmıyor.
        - Bugünün ayı kapsam dışı sayılmıyor. Tarih modele **sistem** mesajıyla
          gidiyor, doğrulayıcı sistem mesajını okumuyor; "bugün ... itibarıyla"
          istemin kendi istediği cümleydi ve bayrak yiyordu.
        - Bir aracın yazdığı seri kimliği/kodu/adresi cevapta anılabiliyor.
          `search_catalog` kaynak kaydetmiyor, dolayısıyla "hangi seriler var?"
          sorusuna adayları sayan cevap her aday için ayrı bayrak alıyordu.
        - `_stated_text` artık bütün turların kullanıcı mesajlarını okuyor.
          `agent/history.py` eski turların sorularını modelin önünde tutup
          araç mesajlarını attığı için asimetri tersineydi: model gördüğü
          sayıyı tekrarladığında bayrak yiyor, göremediği sayı serbest
          geçiyordu.
        Testler:
        - Testler artık `data/gold/catalog.duckdb` dosyasını repo ağacına
          yazmıyor; `tests/conftest.py` oturum boyunca geçici bir dizine
          yönlendiriyor. Dosya gitignore'da olduğu için `git status` bunu hiç
          göstermiyordu. `tests/test_gold.py` kendi küçük parquet'iyle
          çalıştığı için kendi dosyasını alıyor (paylaşılanı doldurunca
          senaryo turları tablosuz kalıyordu, ölçüldü).
Bilerek değiştirilmedi (gerekçesiyle):
        - **Seri seçim kapısının 0,9 oranı.** Kosinüs skorları 0,40-0,65
          arasına sıkıştığı için kapı sık açılıyor; ama belgelenen tek gerçek
          örnek (takipteki 0,650 / bakiye 0,615) tam da insanın karar vermesi
          gereken durum ve organizatörler insan onayını puanlanan bir ayrım
          olarak sayıyor. Havuz 45 bin seriye çıkınca yeniden ölçülmeli.
        - **`_recent_rate` ölçümü.** Token/saniye çağrının tamamına bölünüyor,
          yani istem okuma süresi de içinde; bu yüzden düşünme açık cevap
          pratikte hep kapalıya düşüyor. Doğrusu ilk token'a kadar geçen
          süreyi ayırmak, o da streaming istiyor. Ölçüm düzelene kadar
          davranışı değiştirmek körlemesine ayar olurdu.
        - **Doğrulayıcı bayraklarının modele gösterilmemesi.** Bayraklar
          kullanıcıya gidiyor, `state.messages`'a yazılan cevap bayraksız.
          Göstermek modelin kendi hatasını görmesini sağlar ama sonraki turun
          bağlamını da kirletir; tasarım kararı, hata değil.
        - **`contracts.md` sürüm numarası.** Düzeltmeler için artırılmadı;
          dosya havuz genişletmesiyle zaten v0.5'e geçti.
        - **Kullanılmayan `VerificationFlag` türleri** (`unit_mix`,
          `model_contradiction`) ve `NullVerifier`. İleriye dönük; kaldırmak
          sonraki dallarla çakışma riski taşıyor.

## 23 Eylül incelemesi, ikinci tur (havuz genişletme dalı)
Bağlam: `feat/pool-expansion` dört ayrı incelemeden geçti. Aşağıdakilerin
        hepsi ölçülerek doğrulandı ve testle sabitlendi.
Karar:  Web katmanı (demo günü prompt'tan adres geliyor, tehdit modeli bu):
        - **Türkçe ondalık.** `pd.read_html` Anglo varsayılanlarıyla okuyordu:
          "30,15" değeri `int64` 3015 olarak geliyordu, yani yüz kat hata, ve
          sütun tamsayı geldiği için `prepare_table`'ın Türkçe onarımı da
          göremiyordu. `thousands="."`, `decimal=","` verildi.
        - **URL redaksiyonu.** CLAUDE.md kuralı yalnızca `fetch_log` için
          yazılmıştı ama asıl mesele diske ve insanın önüne ne düştüğü.
          `web/fetch.py::redact_url` eklendi; önbellek kaydı, log, hata
          mesajları ve `Source` redakte ediliyor. İstek hâlâ ham adrese
          gidiyor, göreli bağlantılar ham adresten çözülüyor.
        - **CGNAT.** Beş ayrı yüklem (private/loopback/link-local/reserved/
          multicast) 100.64.0.0/10 için hepsi False dönüyor, ölçüldü. Tek
          yüklem `not is_global` oldu: "internette yönlendirilebilir mi"
          sorusu zaten bu aracın sorması gereken tek soru.
        - **DNS yeniden bağlama.** Kontrol adı kendi çözüp adresi HTTP
          istemcisine *isim* olarak veriyordu; istemci yeniden çözüyordu. İki
          çözüm arasında değişen kayıt korumayı tamamen atlatıyordu. Gövdeden
          tek bayt okunmadan soketin gerçek karşı ucu yeniden kontrol ediliyor.
          Elden geldiğince: saplama oturumda okunacak soket yok, o zaman ön
          kontrol tek başına kalır.
        - **Karesel regex.** `<script[^>]*>.*?</script>` bir kapanışsız açılışta
          belgenin sonuna kadar tarayıp geri izliyordu; ölçülen 16 sn / 1,3 MB,
          10 MB sınırında ~17 dakika, hem de adım zaman aşımının bırakıp
          öldüremediği bir iş parçacığında. `str.find` ile tek geçişe çevrildi.
          `<a>` etiketinin metni de 300 karakterle sınırlandı.
        - **İki sütunlu PDF tablosu.** (etiket, değer) en yaygın basılı biçim
          ve iki hücrede tam bir sayı içeriyor; kesin çoğunluk kuralı bunu
          başlık satırı sayıyordu, dolayısıyla veri satırı hiç bulunmuyor ve
          tablo başlıksız dönüyordu. Eşik "yarısı" oldu.
        - **Bağlantı listesi.** Yalnızca http(s). Sayfa `file:///etc/passwd.csv`
          listelediğinde modele denenecek adres menüsü veriyorduk; `check_url`
          zaten reddediyor ama liste modelin önüne hiç gelmemeli.
        - **Güvenilmez metin sınırı.** Sayfa metni ve bağlantı etiketleri artık
          `>>> sayfa metni (guvenilmez) >>>` ... `<<<` arasında. Sınır olmadan
          bir sayfa harness'ın kendi `Uyarilar:` / `Kalite:` satırlarını taklit
          edebiliyordu, ve doğrulayıcı araç mesajlarını kanıtlanmış saydığı
          için uydurma bir sayı cevapta bayrak yemeden geçebiliyordu.
        - **Onay kartı.** Araç `adres`/`amac` gönderiyor, kart `source`/`detail`
          okuyor: kart "dis kaynak uzerinden veri cekilsin mi?" diyor ve insan
          hiç görmediği adresi onaylıyordu. İki anahtar da veriliyor.
        - **Aktif tablo.** `read_url` en büyük tabloyu aktif yapmaya devam
          ediyor ama artık uyarıda söylüyor ve önceki tablonun adını veriyor.
        Havuz ve araçlar:
        - **Yükseltme (CLAUDE.md #4 ihlali).** `wide()`'ın docstring'i hep
          "daha seyrek seri kendi dönemlerinde bırakılır" diyordu; koruma
          yalnızca `LakeAdapter.to_freq`'te vardı ve üretim yolu `gold.wide`
          üzerinden onu atlıyordu. Ölçüldü: 4 satırlık çeyreklik FinTürk serisi
          10 satırlık aylık tabloya dönüşüyor ve her değer yanlış aya
          etiketleniyordu. `wide()` artık kendi kontrolünü yapıyor.
        - **Dönem etiketleri.** `top` ve `mark` her satırı ay olarak
          adlandırıyordu: günlük tabloda dört ayrı gün aynı etiketi alıyor,
          çeyreklikte Q2 "2024-04", yıllıkta 2024 "2024-01" oluyordu. Doğru
          biçimlendirici (`_label`) zaten vardı; tablonun frekansı soyağacından
          okunup ona veriliyor. `mark` dönem kelimesini de (`ay`/`ceyrek`/`yil`)
          frekansa göre yazıyor.
        - **`top` ve bayrak sütunu.** Rebase'te `arithmetic=False` seçilmişti;
          bir `mark` sütununu sıralamak hiçbir şey cevaplamıyor (işaretli
          dönemleri `mark` zaten listeliyor) ve araç çıktısına çıplak 1/0
          sokuyor. `top` artık reddediyor; `_row_labels` için seçim doğruydu,
          orada soru "bu metin mi".
        - **`period` olmayan tablo.** `transform_table` çıplak `KeyError`
          veriyordu. Bu dal dönemsiz tabloları birinci sınıf yaptığı için
          model oraya ulaşabiliyor; anlaşılır bir ret dönüyor.
        - **Arama metni hizası.** İndeks veri grubunun adını metne katarak
          kuruluyordu, yedek arama katmadan puanlıyordu; `search_text`'in kendi
          belgesi "Genel Endeks" ancak grup adıyla TÜFE olarak bulunabilir
          diyor. Yedek artık aynı metinle puanlıyor.
        - **Yer ailesi uyarısı.** Kelime yedeği `rank`'ın `standing` bilgisini
          atıyordu; 81 ili tek adayın arkasına katlayıp bunu söylememek, havuz
          tek il tutuyormuş gibi görünmesine yol açıyordu. Yedek de uyarıyor.
        - **İndeks sağlamlığı.** `manifest.json` okuması `try` dışındaydı;
          dosya atomik yazılmadığı için yarıda kesilen bir kurulum her aramayı
          düşürüyordu. İçeri alındı. Manifest artık embedding modelinin adını
          da tutuyor: aynı boyutta başka bir modelle kurulmuş indeks anlamsız
          kosinüs döndürüyor ve hiçbir şey uyarmıyordu.
        Belgeler: `contracts.md` §5 `read_url`'ü üretim kaydında gösteriyor;
        CI'ın atlama sayacı 21'den 39'a, alt sınır 500'den 850'ye çıktı
        (885 testin %43'ünü kaybetsek koruma uyarmıyordu); `docs/licenses.md`
        dört yeni bağımlılık ve bağımlılıkları için yeniden üretildi (66 paket,
        hepsi permissive).
Uygulanmadı (gerekçesiyle):
        - **FinTürk `obs_date` çeyreğin ilk ayı.** Sözleşme "kaynağın verdiği
          tarih" diyor, bronze "2021-12" diyor, gold 2021-10 yazıyor. Düzeltmek
          gold'u yeniden üretmeyi gerektirir (33 MB parquet, yeniden ingest);
          veri üretimi ayrı bir iş ve pencere sınırı bugünkü aralıkta güvenli.
        - **BDDK etiket yeniden numaralandırması** ("(2 den 24'e)" →
          "(2 den 26'ya)") bir kalemi iki seriye bölüyor; gold'da şu an 6 kalem
          böyle. Çözümü `label_key`'in düzyazı numaralandırmayı da soğurması ve
          ardından yeniden ingest; aynı sebeple ertelendi.
        - **`to_freq`'in `last`'i NaN atlıyor** (11.287 ay kovası, %1,86): ay
          sonu gözlemi eksikse bir önceki değer "ay sonu bakiyesi" diye
          sunuluyor. Düzeltmek havuzdan türeyen her sayıyı değiştirir; ölçüp
          ayrı bir adımda ele almak doğru.
        - **Seri seçim kapısının 0,9 oranı** 45 binde yeniden ölçülmedi.
        - **`_recent_rate`** hâlâ istem okuma süresini token hızına katıyor;
          doğrusu akış (streaming) istiyor.

## Sorgu bir tablonun adını söylediğinde o tablonun toplamı öne alınır
Bağlam: 23 Eyl canlı koşusu, senaryonun 3. sorusu. "konut fiyat endeksi"
        araması Konut Fiyat Endeksi'ni hiç döndürmedi; beş aday TÜFE'nin konut
        alt kalemleriydi ("Konut Sigortası", "Konutun Bakımı ...").
Ölçüm:  `evds:TP.KFE.TR` havuzda, adı/konusu/etiketleri sorguyla birebir
        örtüşüyor, en yakın 500'de **40. sırada, kosinüs 0,639** — eşiğin
        (0,40) çok üstünde. `tables_asked` sorguyu doğru okuyor (`bie_kfe`,
        `bie_hkfe`) ve KFE.TR o tablonun başlık serisi. Yani ne arama ne de
        eşik hatalıydı: rakipler biraz daha yüksek puan aldı (0,645 / 0,660 /
        0,679 / 0,684) ve beş aday sınırı KFE.TR'yi tam dışarıda bıraktı.
Karar:  `rank` bir tablonun adı geçtiğinde yalnızca o tablonun alt kalemlerini
        geriye atıyordu; başlık serisini hiç öne almıyordu, oysa fonksiyonun
        kendi belgesi "böyle bir sorgu tablonun kendi toplamını ister" diyor.
        `tier` artık tek sayı değil ikili anahtar döndürüyor: (öne alındı mı,
        ne kadar geriye atıldı). Öne alma geriye atmayı iptal edemiyor, bu
        yüzden arşiv bir tablonun arşiv başlığı (`TP.HKFE01`, 0,726) güncel
        endeksin önüne geçmiyor.
Sonuç:  Aynı skorlarla yeniden hesaplandı: ilk sıra `evds:TP.KFE.TR`, arşiv
        serileri en sonda. Tablo adı geçmeyen sorgularda sıralama hiç
        değişmiyor (testle sabitlendi). "tüketici fiyat endeksi" sorgusunda
        da aynı kural TÜFE Genel Endeks'i öne alıyor.

## Boş cevap bir kez daha sorulur (23 Eyl)
Bağlam: `--real` koşusu, araçların hepsi başarılı: arama doğru seriyi buldu,
        `get_series` 66 ayı getirdi, ve cevap adımı **374 tamamlama token'ı
        harcayıp tek karakter içerik üretmedi**. Kullanıcı bitmiş bir analizin
        altında "Model bu tur icin bir cevap uretmedi" gördü. Aynı arıza 22
        Eylül'de 31 turun 6'sında ölçülmüştü.
Karar:  Cevap boş dönerse ve düşünme açıksa, bir kez daha düşünme kapalı
        sorulur. Farklı bir istek: akıl yürütme kanalı yok, dolayısıyla cevap
        oraya kaybolamıyor. Ucuz, çünkü iş zaten bitmiş, eksik olan yalnızca
        yazma adımı. Bir kez: iki kez cevap vermeyen model üçüncüde de
        vermez, ve turun kişiye hâlâ bir yanıt borcu var. Yeniden denemenin
        süresi aynı paydan yeniyor (`_Spent` paylaşılıyor), yani tur bütçesi
        aşılmıyor; `LlmError` gelirse eski davranışa düşülüyor.
Sonuç:  `_answer_messages`'ın 22 Eylül'deki düzeltmesi (istek asistan
        mesajıyla bitmesin) yeterli değildi: bu koşuda istek araç sonucuyla
        bitiyordu ve cevap yine boş geldi. İkisi farklı arızalar.

## Cevap adımı sistemin yetenekleri hakkında konuşmaz (23 Eyl)
Bağlam: `--real`, "Konut fiyat endeksini aylik getir." Model soruyu "aylık
        getiri" diye okudu (şapkasız yazım) ve cevabında şunu yazdı: "istediğin
        aylık getiri serisini üretecek bir hesaplama aracı elimde yok".
        `transform_table` bunu `pct_change` ile yapıyor; araç tanımında da
        "yuzde degisim" yazılı.
Neden:  `answer.md` araçlardan hiç bahsetmiyor. Cevap adımında modelin elinde
        araç listesi yok, yalnızca tablolar, kaynaklar ve uyarılar var. Model
        1. kuralı ("kendi hesabını yapma") doğru uyguladı ama bunu "böyle bir
        araç yok" diye genelleştirdi. Doğrulayıcı bunu göremez: ortada yanlış
        bir sayı değil, sistem hakkında yanlış bir ifade var.
Karar:  answer.md v2, 9. kural: cevap adımı sistemin neyi yapıp yapamadığı
        hakkında iddiada bulunmaz. Bir hesap yapılmadıysa "yapılmadı" denir,
        "aracım yok" denmez.
Sonuç:  Kural prompt'ta; etkisi ancak canlı koşumlarda ölçülebilir. Asıl
        çözüm planlama adımının dönüşümü baştan yapması olurdu, ama planner
        10. kural kullanıcının istemediği dönüşümü eklemeyi yasaklıyor ve bu
        soruda ne istendiği gerçekten belirsizdi ("getir" mi "getiri" mi).

## PDF başlıkları hücreden değil kelimeden kurulur (23 Eyl)
Bağlam: Demo adresinin altın PDF'i okundu, veriler doğru geldi ama 11 sütun
        başlığının 7'si parçalanmıştı: "Hacim/Volume (TL)" bir sütunda "H",
        diğerinde "acim/Volume (TL)"; "TOPLAM" → "TO" + "PLAM"; "of" → "o" +
        "f". Model bununla başa çıktı ama ekranda görünen buydu.
Neden:  Çizgisiz tabloda pdfplumber sütun sınırlarını **sayıların** durduğu
        yerden türetiyor; basılı başlık o sınırlara oturmuyor ve sınır
        kelimenin ortasından geçiyor. `text_x_tolerance` ve
        `min_words_vertical` ayarlarının hiçbiri değiştirmedi (ölçüldü).
Karar:  Başlık bandı hücre hücre değil kelime kelime okunuyor: her kelime
        orta noktasının düştüğü sütuna yazılıyor. Kelime bölünemez olduğu
        için sınır artık kelimeyi kesemiyor. Band tablonun üstü ile ilk veri
        satırı arasıyla sınırlı; bu da belgenin kendi başlığını (tablonun
        üstünde, tam genişlikte basılı) sütun adlarından uzak tutuyor.
        Kurulamadığı her durumda (tablo bulunamadı, sütun sayısı tutmadı,
        pdfplumber hata verdi) eski okuma kullanılıyor.
Sonuç:  Aynı PDF'te 11 başlığın 11'i doğru: `AY/MONTH`, `Hacim/Volume (TL)`,
        `TOPLAM / TOTAL Miktar/Amount (KG)` ... Yinelenen `Sayısı/ Trans.`
        adları `_unique` ile ayrılıyor. Veri satırları değişmedi.

## 23 Eylül incelemesi, üçüncü tur (nedensellik ve arama dalı)
Bağlam: `feat/causality` üç incelemeden geçti; aşağıdakilerin hepsi ölçülerek
        doğrulandı ve testle sabitlendi.
Karar:  Güven hikâyesi (en önemlisi):
        - **Arama sonucundaki sayı artık cevaba kanıt olamıyor.** Doğrulayıcı
          her araç mesajını kanıtlanmış metin sayıyordu; `web_search` sonucu
          da bir araç mesajı olduğu için yalnızca bir haber parçacığında geçen
          "dolar kuru 99,99" bayraksız geçiyordu (ölçüldü). Oysa aracın kendi
          metni, cevap isteminin 5. kuralı ve sözleşme §7.3 bunun
          olamayacağını söylüyordu. Çit (`agent/formatting.fence`) artık tek
          yerde tanımlı: `read_url` ve `web_search` yabancının yazdığı her
          şeyi ona sarıyor, `_stated_text` çitli bölgeleri kesiyor. Ölçüm:
          aynı cümle artık bayrak alıyor, tablodan gelen sayı ve kaynaklı
          olay iddiası geçiyor.
        - **Arama metni satır uyduramıyor.** `ddgs` adresleri `unquote`
          ediyor ve sağlayıcı yalnızca başlık ile özeti sıkıştırıyordu; tek
          bir `%0A` üç satıra dönüşüp `Uyarilar:`, `Kalite:` satırlarını
          sıfırıncı sütunda taklit edebiliyordu (ölçüldü). Üç alan da
          `one_line`'dan geçiyor ve çit içine giriyor.
        - **Onay kartı sorguyu gösteriyor.** `web_search` önizlemesi
          `source`/`detail` vermiyordu, kart "dis kaynak uzerinden veri
          cekilsin mi?" diyordu. `read_url`'de düzeltilen regresyonun aynısı.
        - **Arama motoru sabitlendi.** `ddgs` varsayılanı `auto`, yani google,
          bing, brave, startpage de taranıyordu; her doküman DuckDuckGo
          diyor. `BACKEND = "duckduckgo"`.
        Süre bütçesi:
        - **Boş cevap yeniden denemesi geç turlarda çalışmıyordu.** `_chat`
          payı `active_s`'ten yeniden hesaplıyor, `left()` ise aynı
          `_Spent`'ten ilk denemeyi bir kez daha düşüyordu. Ölçüm: tur 125 sn
          harcamışken yeniden denemeye −15 sn veriliyor, bu `LlmTimeoutError`
          oluyor ve `_answer` onu yutuyordu. Artık her çağrı yalnızca kendi
          harcadığını düşüyor: 135 / 75 / 30 / 30 sn.
        Nedensellik:
        - **Durağanlık iddia edilmek yerine sağlanıyor.** Tek fark alınıp bir
          daha bakılmıyordu. Ölçüm: 120 bağımsız iki-kez-entegre çiftin
          117'si tek farktan sonra hâlâ durağan değildi ve %27,5'i "öncül"
          çıkıyordu. Fark en çok iki kez yineleniyor; hâlâ durağan değilse
          özet "DIKKAT ... sonucu kullanma" diyor. Yeni oran **%5,0** ve
          kalan altı vakanın altısı da uyarı taşıyor.
        - **Bonferroni ailesi iki yönü de sayıyor.** Eşik gecikme sayısına
          bölünüyordu ama iki yön test edilip çifte tek karar veriliyor; 22
          gözlemde gerçek oran %9 iken metin %5 diyordu. Artık `2 x gecikme`.
        - **`donusum` iki serinin de dönüşümünü yazıyor.** Yalnızca neden
          serisininkini yazıyordu, dolayısıyla bir satır "yok" derken diğer
          taraf farklanmış olabiliyordu: seviyeye karşı fark regresyonu,
          hiçbir şey olmamış gibi etiketlenmiş.
        - Yan etki, ölçüldü: gerçek havuz çiftinde sonuç artık `kind` alanının
          dolu olup olmamasına göre ters dönmüyor (p=0,6237 ve p=0,0312, ikisi
          de anlamsız). Önceki halde biri "öncül" diğeri "öncüllük yok"
          diyordu.
        Belgeler: `answer.md` v3 (iki tane 9 numaralı kural vardı, rebase
        artığı; ikincisi 10 oldu), `contracts.md` sekiz aracı ve `fetch`
        kapısının iki aracını sayıyor, `docs/licenses.md` 69 paketle yeniden
        üretildi (ddgs, primp, click dahil, hepsi permissive).
Uygulanmadı (gerekçesiyle):
        - **22 gözlemde testin gücü düşük** (güçlü etkide 0,42). Bu verinin
          sınırı, kodun değil; "öncüllük yok" cevabını "ilişki yok" diye
          okutmamak için ayrı bir güç cümlesi gerekir, o da ölçüm ister.
        - **Tekil tasarım** (bir seri diğerinin katı) hâlâ "p=1,0, öncüllük
          yok" diyor. Dürüst bir ret yazmak `_granger`'ın statsmodels
          uyarısını ayırt etmesini gerektiriyor.
        - **Tekrarlanan dönem, takvim boşluğu, karışık frekans** kontrol
          edilmiyor. Üçü de `lead_lag`'in elindeki `period` sütunuyla
          bakılabilir; ayrı bir adım.
        - **Önbellekte süre sınırı yok**, boş sonuç kalıcı olarak saklanıyor
          ve `force` parametresinin çağıranı yok.
        - **Yeni araç testleri** hâlâ `sys.modules` saplaması kuran dosyanın
          içinde; gerçek sözleşme nesneleriyle denenmiyor.

## Alıntılanan adresin içindeki rakam iddia değildir (23 Eyl, canlı koşumdan)
Bağlam: Çit düzeltmesinden sonraki ilk `--real` koşusunda doğru bir cevap iki
        bayrakla çıktı: '906699' ve '900606'. İkisi de modelin kısaltarak
        yazdığı haber adreslerinin içindeki numaralar
        (`ekonomim.com/…-906699`).
Neden:  Çit, arama sonucundaki adresleri `_stated_text`'ten çıkarıyor, ki asıl
        istenen buydu. Ama cevaptaki kısaltılmış adres `https://` taşımadığı
        için `_URL` ile de eşleşmiyordu, dolayısıyla içindeki rakamlar çıplak
        sayı iddiası sayılıyordu. Kendi düzeltmemin yan etkisi.
Karar:  Şemasız adresler de (`_BARE_URL`) diğer kimliklerle birlikte, sayı
        okunmadan önce boşaltılıyor. `answer.md` 4. kuralı adresin yazılmasını
        istiyor ve `_mentions` kısa biçimi kabul ediyor; bir alıntının gerçek
        biçimi bu.
Sonuç:  Ölçüm: alıntılı cevap geçiyor, arama parçacığındaki sayı ve uydurma
        sayı hâlâ bayrak alıyor. Doğrulayıcının iki işi de yerinde.

## `nearest` skor sütununu adıyla istiyor (23 Eyl, canlı koşumdan)
Bağlam: `--real` koşusunda her `search_catalog` çağrısında LanceDB uyarı
        basıyordu: arama çıktı sütunlarını belirtmiş ama `_distance`'ı
        listeye koymamış; şimdilik sütun eklenmeye devam edecek, ileride
        eklenmeyecek.
Neden:  `pool_index.PoolIndex.nearest` yalnızca `series_id` seçiyor, sonra
        `hits["_distance"]` okuyordu. Otomatik ekleme kalktığı gün satır
        `KeyError` veriyor ve anlamsal arama komple düşüyor; bunu ilk gören
        yer demo olurdu. Ölçüm (23 Eyl, gerçek havuz indeksi): sütunu adıyla
        istediğimde sonuçlar birebir aynı.
Karar:  `nearest` artık `["series_id", "_distance"]` seçiyor. `keyword` aynı
        şeyi YAPMIYOR: onun skor sütunu `_score` ve tam metin tarayıcısı
        seçilen alan olarak kabul etmiyor ("No field named _score"), yani
        aynı düzeltme orada aramayı anında kırar. Zaten skoru hiç okumuyor.
Sonuç:  `nearest` doğrudan test edildi (`test_nearest_scores_the_hits_it_
        returns`): sıra benzerliğe göre ve en yakın kayıt 1,0 veriyor. Skoru
        sabitleyince test kırmızıya düşüyor, yani sıralamanın sessizce ekleme
        sırasına dönmesini de yakalıyor.

## Agent katmanı ölçülüyor (23 Eyl)
Bağlam: Doğrulama katmanı vardı (onay noktaları, kural tabanlı doğrulayıcı,
        kaynak gösterimi) ama uçtan uca hiçbir ölçüm yoktu.
        `eval/questions_A.yaml` 16 Eylül'de yazılmış, beklenen seri ve elle
        hesaplanmış değerlerle 7 soru taşıyordu, fakat repoda bu dosyayı
        okuyan tek satır kod yoktu. Arama katmanının ölçümü vardı
        (`lake/search_eval.py`), agent katmanının yoktu.
Neden:  Demo günü bilinmeyen girdiler gelecek. Neyin kırıldığını jüriden önce
        görmenin tek yolu bir soru takımı; ayrıca bundan sonraki her değişiklik
        için gerileme ağı oluyor.
Karar:  `agent/answer_eval.py`, `search_eval` deseninin aynısı: enjekte
        edilebilir `deps`, `run()` içinde yazma yok, Türkçe anahtarlı düz
        rapor, `eval/results/answer_eval_<tarih>.json`. Her soru **temiz bir
        oturumda** koşuyor; CLI takip sorusu için oturumu paylaşıyor, eval
        maddeleri bağımsız, paylaşılsa bir sorunun tablosu diğerinin
        kontrolünü sahte geçirirdi. Onay noktaları demo ile aynı davranıyor
        (otomatik varsayılan, seri sabitlenmiyor), yani ölçülen şey gerçek
        uçtan uca davranış. Kapsama kuralı üst küme serbest: listede olmayan
        bir araç çağırmak düşürmüyor, eksik bırakmak düşürüyor.
        Doğrulayıcı bayrağı satırı DÜŞÜRMÜYOR, ayrı raporlanıyor; doğrulayıcı
        da ölçüm altında ve bayrağı geçme koşulu yapmak onu sessizce
        gevşetmeye teşvik ederdi.
        `questions_A.yaml` → `questions.yaml`: "A şeridi" tarihsel.
Sonuç:  12 soru, sekiz kayıtlı aracın hepsi kapsanıyor ve bunu bir test
        bekçilik ediyor. Taşırken üç hata çıktı, üçü de ölçümle bulundu:
        A2'nin beklediği `kfe_endeks` sütunu hiç üretilmiyor (`rebase`
        çıktısı `{sutun}_endeks_{bazYYYYAA}`), A5 ve A6'nın makine kontrolü
        yoktu, A7 hiçbir şey ölçmüyordu. A5 artık asıl tuzağı ölçüyor:
        2026-06'da ortalama 40,335, toplam 161,34; oran serisini toplamla
        toplulaştırmak buradan yakalanıyor.
        Reddetme kontrolü (`tabloda_donem_yok`) sütun bazlı ve yalnızca DOLU
        değere bakıyor. İlk yazdığım hali "tabloda o dönem olmasın" diyordu ve
        yanlıştı: EVDS pencerenin tamamına hücre verip seri yayımlanmadan
        önceki dönemleri boş bırakıyor, yani 2021'den çekmek doğru davranışken
        kontrolü düşürüyordu.
        `deger` dönem verilmediğinde bütün satırlara bakıyor; öncüllük tablosu
        iki satırlı ve hangi yönün önce geldiği modelin hangi sütunu `column`
        diye geçtiğine bağlı, satır indeksine bakmak kırılgan olurdu.
        27 test, hepsi ağsız. Dört kural tek tek bozulup testin kırmızıya
        düştüğü görüldü. Bu sırada kendi testimin bir tanesi yakalanmadı:
        `_same`'in bool dalını kaldırdığımda test yeşil kalıyordu, çünkü
        seçtiğim durum ayırt edici değildi. Gerçek risk tolerans verildiğinde:
        `beklenen: false` + `tolerans_yuzde: 1` sayısal okunursa "sıfıra 1
        yakın" olur ve `True` bunu sağlar, yani soru ters hükümle geçerdi.
        Test o duruma çevrildi ve artık ısırıyor.

## Eval'in ilk bulgusu: TÜFE serisi sözleşmeye aykırı seçiliyor (23 Eyl)
Bağlam: `answer_eval` ilk gerçek koşusunda 11/12 verdi. Düşen satır A6:
        "TÜFE 2021 başından bugüne yüzde kaç arttı?"
Neden:  Agent yüzdeyi DOĞRU buluyor (%706,1) ama seriyi `TP.GENENDEKS.T1`
        (2003=100) üzerinden hesaplıyor. `CLAUDE.md` TÜFE'yi
        `evds:TP.TUKFIY2025.GENEL` olarak şart koşuyor. Sonuç aynı çıkıyor,
        çünkü ikisi de aynı enflasyonu ölçüyor; seri seçimi yine de
        sözleşmeye aykırı ve tutarlılığı bozuyor. `deflate` işlemi kendi
        deflatörünü içeriden doğru seçtiği için A4'te sorun yok; açık uçlu
        soruda arama ne bulursa o geliyor.
Karar:  Soru KIRMIZI BIRAKILDI. Beklentiyi gevşetmek bulguyu gizlemek olurdu
        ve eval'in varlık sebebi tam olarak bu. İki koşumda da aynı şekilde
        düştü, yani tesadüf değil.
Sonuç:  Açık kalan iş: agent'ın havuzdaki onlarca TÜFE serisinden sözleşmenin
        şart koştuğunu seçmesi gerekiyor. Muhtemel yollar, ölçülmeden karar
        verilmemeli: katalog arama metnine sözleşme işareti koymak, ya da
        planlayıcı promptunda seriyi adıyla anmak. İkincisi tek seriyi
        ayrıcalıklı kılar ve başka sorularda yan etki yapabilir.
        Aynı koşuda iki eskimiş beklenti de düzeldi: A4'ün beklediği
        `TP.HPBITABLO6.3`'ün değeri dolu ilk haftası 2024-06-24, yani soru
        2021'den reel seri isterken o seriyle cevaplanamaz; havuz
        genişleyince BDDK'nın aylık konut kredisi bakiyesi doğru cevap oldu.
        A4'ün `cevapta_gecmeli: "baz"` kontrolü de kelime arıyordu: agent
        bazı "TÜFE 2025=100" diye yazdı, kavramı verip kelimeyi vermedi;
        kontrol bazın kendisini arar hale getirildi.

## Havuz sıralaması: başka ülkenin kopyası ve eskimiş baz yılı (23 Eyl)
Bağlam: `answer_eval`'in A6 bulgusunu kovalarken sıralamayı ölçtüm.
        "tüketici fiyat endeksi" sorgusunda ilk sıra **Almanya'nın TÜFE'si**
        (0,6680), ikinci 2003=100 Türk endeksi (0,6665), sözleşmenin şart
        koştuğu seri üçüncü (0,6484). Onay kartı çıkıyor ama varsayılanı
        Almanya.
Neden:  İki ayrı boşluk. Havuzda bir serinin başka bir ülkeyi anlattığını
        gösteren hiçbir işaret yoktu; il için `places` vardı, ülke için
        yoktu. İkincisi, TÜFE havuzda iki canlı sürümle duruyor (2003=100 ve
        2025=100) ve EVDS ikisini de "(Arşiv)" işaretlemiyor, yani baz yılı
        dışında onları ayıran hiçbir şey yok. İki aday metin olarak
        ayırt edilemez: aynı konu, aynı etiketler, aynı not.
Karar:  İki kural, ikisi de mevcut kalıba oturuyor ve ikisi de kaynağın
        serisi hakkında söylediğini okuyor, sorgunun konusunu değil.
        1) `country`: seri adı "Ülke (ISO3) ..." ile başlıyorsa o ülkeye
        aittir; sorgu ülke adı anmıyorsa geri çekilir. Türkiye'nin kendi
        satırları (TUR) yabancı sayılmaz. Kalıp tek başına yetmiyor, çünkü
        "Konut Fiyat Endeksi (KFE)" de üç kelime artı üç büyük harf; bu
        yüzden işaret veri grubuna bağlandı. Ölçüldü: gerçek ülke tabloları
        39–185 farklı kod taşıyor, yanlış pozitifler tam olarak 1, arada
        hiçbir şey yok. Eşik 10.
        2) `_superseded`: bir tablonun daha yeni bazlı sürümü varsa eski
        sürüm geri çekilir, sorgu o baz yılını anmadıkça. Karşılaştırmadan
        önce bütün parantezler atılıyor, çünkü sürümler adın geri kalanında
        anlaşmıyor ("Tüketici Fiyat Endeksi (Genel) (2003=100)" ile
        "Tüketici Fiyat Endeksi (2025=100)" aynı tablonun iki sürümü).
        Ölçüldü: baz yılı taşıyan 25 veri grubundan yalnızca iki ailenin
        rakibi var, diğer 20'si etkilenmiyor.
Sonuç:  Arama ölçümü: 25/27 aynı kaldı, iki sorgu iyileşti. "tüketici fiyat
        endeksi" 3. sıradan 1.'ye, "cari işlemler dengesi" 5.'ten 4.'e.
        Gerileme yok. İndeks yeniden kuruldu, gömme önbellekten geldi
        (45.422/45.422 cache, sıfır MIA çağrısı).
        **A6 yine de kapanmadı** ve nedeni üçüncü bir şey çıktı, aşağıda.

## `tables_asked` sorguya kelime eklenince çöküyor — denendi, geri alındı (23 Eyl)
Bağlam: İki sıralama kuralından sonra A6 hâlâ 2003=100 serisini seçiyordu.
        Agent'ın gerçek sorgusunu izledim: "tüketici fiyat endeksi TÜFE
        genel endeks". Bu sorguda ilk beş aday TÜFE'nin **alt kalemleri**:
        "08. Bilgi Ve İletişim", "133. Sosyal Koruma", "011. Gıda". Ana
        endeks listede yok.
Neden:  `tables_asked` "sorgunun HER kelimesi tablo adında geçmeli" diyor.
        Sorguya kendi eş anlamlısını ekleyen bir kullanıcı ("tufe") kuralı
        devre dışı bırakıyor: hiçbir tablo istenmemiş sayılıyor, dolayısıyla
        alt kalem indirimi de çalışmıyor ve alt kalemler kosinüste ana
        endeksi geçiyor. Kural yalnızca sorgu tam olarak tablo adı olduğunda
        tutuyor.
Karar:  Ters okuma denendi (tablonun adı sorgunun içinde olsun, fazladan
        kelime serbest). Agent'ın sorgusunu düzeltiyor ama **arama ölçümünü
        bozuyor: 25/27 -> 23/27**, ayrık sorgularda 6/6 -> 5/6. "Ankara
        konut fiyat endeksi" 3. sıradan listeden düştü, çünkü sorgu "Konut
        Fiyat Endeksi" adını heceliyor ve tablonun ulusal toplamı, sorgunun
        asıl istediği Ankara serisinin önüne geçiyor. "sanayi üretim
        endeksi" 1.'den 20.'ye düştü. GERİ ALINDI.
Sonuç:  Açık iş. Bir tablonun toplamını öne almakla, adı anılan ili
        onurlandırmak birbirini çekiştiriyor; bu çözülmeden gevşek okuma
        gelemez. A6 kırmızı kalıyor ve nedeni artık kayıtlı: sorun modelde
        ya da gömmede değil, `tables_asked`'ın hep-ya-hiç olmasında.
        Ölçüm: 11/12, `eval/results/answer_eval_2026-09-23.json`.

## İnceleme bulguları: ölçüm kendi kendini vuruyordu (23 Eyl)
Bağlam: `feat/answer-eval` dalının baştan incelemesi. Her bulgu bir kanıt
        betiğine yazıldı, düzeltmeden önce 17 kontrolün 15'i kırmızıydı,
        sonra 17'si de yeşil.
Neden ve karar, en ağırdan başlayarak:
  1. **Puanlama çökmesi bütün koşuyu siliyordu.** Kontroller `_one`'ın koruma
     bloğunun dışındaydı ve rapor yalnızca `run()` bittikten sonra yazılıyor.
     Yarım yazılmış tek bir `deger` kontrolü (`sutun` yok) `KeyError` ile
     12 sorunun MIA maliyetini birden götürüyordu. Puanlama da korumaya
     alındı: bozuk kontrol kendi satırını düşürüyor, koşu devam ediyor.
  2. **Politika faizi sorusu yanlış ülkeyi döndürebiliyordu.** Ülke kalıbı üç
     harfli kod arıyordu, Euro Bölgesi'nin kodu `(XM)` iki harf. 40 satırlık
     `bie_bispolfaiz` ailesinden yalnızca Türkiye ve Euro Bölgesi geri
     çekilmeden kalıyordu; aile tek adaya daraldığı için hangisinin
     görüneceğine yalnızca kosinüs karar veriyordu. Kalıp bilerek gevşetildi
     (2-3 harf, büyük/küçük farketmez, tire serbest, altı kelimeye kadar) ve
     asıl koruma eşiğe bırakıldı. Ölçüm: 48 gerçek yabancı seri daha
     yakalanıyor, yanlış pozitif sıfır, `Bosna-Hersek`, `Gürcistan(Geo)`,
     `Güney Kıbrıs Rum Yönetimi` artık işaretli.
  3. **Bronze olmadan kurulan indekste koruma çöküyordu.** Grubu olmayan
     seriler `str(None)` ile tek kovaya düşüyor, o kova eşiği geçiyor ve
     "Konut Fiyat Endeksi (KFE)" ülke sayılıyordu. `evds=None` belgelenmiş
     bir mod ve bronze gitignore'da olduğu için taze klonun aldığı mod.
     Grubu olmayan satır artık atlanıyor.
  4. **Bayat indeks kimseyi uyarmıyordu.** Manifest digest'i kimlik ve arama
     metnini hash'liyor, meta sütunu eklemek ikisini de değiştirmiyor.
     Ölçüldü: `country` sütunu düşürülünce "tüketici fiyat endeksi" yine
     Almanya'yı birinci getiriyor ve hiçbir not bunu söylemiyor. `meta_notes`
     eklendi, eksik sütunu adıyla söyleyip yeniden kurmayı istiyor.
  5. **Ülke adları alt dize olarak aranıyordu.** "içindeki" ve "cinsinden"
     `cin`, "karşılık" `sili` eşliyordu. Eval'in kendi A3 sorusu Çin
     serilerini kuraldan muaf tutuyordu, yani kural o soruda kapalıydı.
     `_country_named` il kuralındaki kelime sınırını kullanıyor.
  6. **Kontroller en eski tabloyu okuyordu.** `sorted()` ilk eşleşmeyi
     alıyordu; agent seriyi yeniden çekince (A5'te haftalıktan aylığa) doğru
     cevap `t02`'de, kontrol `t01`'e bakıyordu. Artık en yenisi okunuyor.
  7. **Eksik değer hüküm sayılıyordu.** Boş bir `anlamli` sütunu
     `beklenen: false` kontrolünü geçiyordu, yani "test hiç koşmadı" hali
     "öncüllük yok" diye okunuyordu; `pd.NA` ise koşuyu öldürüyordu.
  8. **Hayali araç çağrılmış sayılıyordu.** Registry'nin tanımadığı bir ad
     `expected_tools`'u karşılıyordu. `unknown_tool` hatası artık sayılmıyor;
     kapıda iptal edilen çağrı sayılmaya devam ediyor, o kasıtlı.
  9. **Çöken satır maliyetini sıfır bildiriyor ve iki kez sayılıyordu.**
     Gerçek maliyet `state.usage`'da duruyor; `kesintili` artık yalnızca
     hatasız kesilen turları sayıyor.
  10. Tek bir liste maddesine iki kontrol yazılırsa ikincisi sessizce
      düşüyordu, artık biçim hatası olarak kırmızı veriyor. Testler çalışma
      dizinine bağlıydı, `Path(__file__)` ile bağlandı. `dogrulandi` alanı
      yazılıp okunmuyordu, artık verify adımı yoksa özet onu basıyor.
Sonuç:  1021 test yeşil (23 yeni), ruff temiz. Arama ölçümü 25/27 ve ayrık
        6/6 ile sabit, gerileme yok. Agent ölçümü 11/12, A6 hâlâ ve bilerek
        kırmızı. Dört düzeltme tek tek bozulup testin kırmızıya düştüğü
        görüldü. Doğrulanan ama DÜZELTİLMEYEN bir madde var: `_superseded`
        bütün parantezleri attığı için ileride "Fiyat Endeksi (Üretici)"
        gibi bir tablo tüketici endeksiyle aynı gövdeye düşebilir. Bugün
        havuzda böyle bir çakışma yok, spekülatif olduğu için elle
        değiştirilmedi.

## B5 sahte yeşildi, adım kaydı yoktu, bağlantılar kaçak (23 Eyl, koşumdan)
Bağlam: Manuel doğrulama koşusunda B5 geçti ama cevabında bayrak vardı ve
        "sayfa 404 döndü" yazıyordu. Önbellekte o adres için 200 duruyordu.
Neden:  Üç ayrı şey çıktı, biri benim hatamdı.
        1. **`read_url` PDF'i tam okuyor**, ölçüldü: doğrudan verilince
           `ok: True`, `t01_read_url` tablosu kuruluyor, metinde "yüzde
           15'ten yüzde 17,5" geçiyor. Yetenek eksik değil. Araç sayfadaki
           veri dosyalarını gösteriyor ama takip etmiyor; bu kasıtlı, her
           çekim ayrı onaydan geçiyor.
        2. **B5 sahte yeşil veriyordu.** Tek kontrolü `kaynak_url_var` idi ve
           web araması da adres döndürüyor. HTML kabuğunda durup cevabı
           kendi belleğinden yazan koşu da geçiyordu.
        3. **Koşucu adım kaydı tutmuyordu**, bu yüzden hangi adresin
           çağrıldığını söyleyemedim ve modeli haksız yere 404 uydurmakla
           suçladım. Suçlamayı geri aldım.
Karar:  B5'in kontrolü sonuca bağlandı: `tablo_var: read_url` artı
        `cevapta_gecmeli: "17,5"`. O tabloyu yalnızca gerçekten okunan bir
        belge kurar, web araması kuramaz.
        Rapora `adimlar` eklendi: tur içindeki her araç çağrısı, argümanları
        (200 karaktere kırpılı), `ok` ve hata. Düşen satırlarda özete de
        basılıyor.
        `_data_links` artık `href`'i `html.unescape` ile çözüyor. TCMB
        bağlantılarını `?MOD=AJPERES&amp;CACHEID=...` diye yazıyor ve modele
        olduğu gibi verilince ikinci parametre `amp;CACHEID` oluyor. Bu
        sunucuda zararsız (ikisi de 200), parametreyi okuyan bir sunucuda
        modelin açması istenen adres sayfanın kastettiği adres olmuyor.
Sonuç:  Ölçüm 11/12'den 10/12'ye düştü ve bu bir gerileme DEĞİL: B5 artık
        dürüstçe kırmızı. Adım kaydı ilk koşuda işe yaradı, A6'nın her
        seferinde başka bir TÜFE serisi seçtiği görüldü (bu koşuda
        `evds:TP.FE25.OKTG01`).
        AÇIK KALAN: aynı koşuda agent tek bir `read_url` yapıp kabuk sayfayı
        okudu, PDF bağlantısını takip etmedi ve "duyuruyu okudum" diyerek
        kendi belleğinden sayı yazdı. Doğrulayıcı '250'yi yakaladı, '17,5'i
        yakalamadı; boş oturumda ikisini de yakalıyor, yani gerçek koşuda
        '17,5' araç çıktısında bir yerde bulunmuş. Hangisi olduğu
        incelenmedi. Modelin veri dosyası bağlantısını takip etmemesi de
        ayrı bir iş.

## İki çit, iki anlam: arama kanıt değil, okunan belge kanıt (23 Eyl)
Bağlam: B5'in `cevapta_gecmeli: "17,5"` kontrolü düştü. Kullanıcı "daha önce
        URL'den sayı yazmasın diye bir şey yapmamış mıydık" diye sordu.
Neden:  İki ayrı şey bulundu.
        1. **Kural var ve aramayı hedefliyor.** `answer.md` 5. madde:
           "Aramadan gelen sayıyı cevaba yazma... Sayı gerekiyorsa kaynağı
           `read_url` ile aç ve oradan al." B5'in o koşumunda ilk `read_url`
           başarısız oldu, model aramaya düştü ve kurala uyarak oranı
           yazmadı. Kontrol doğru davranışı cezalandırdı; satırın notuna
           yazıldı, kontrol değiştirilmedi çünkü kaynağı okumayan bir koşunun
           kırmızı olması doğrudur.
        2. **Çit ayrımı tesadüfe dayanıyordu.** `without_fenced` yalnızca
           `formatting.UNTRUSTED_OPEN`'ı tanıyor. `web_search` onu kullanıyor,
           dolayısıyla parçacıklardaki sayılar kanıt sayılmıyor. `read_url`
           kendi metnini kullanıyor ve tanınmadığı için sayfa metnindeki
           sayılar kanıt sayılıyor. Ölçüldü: çit içine konan 98.765,4 cevapta
           yazılınca bayrak almıyor, hiç geçmeyen sayı alıyor. Doğrulayıcının
           docstring'i ise ikisini de kestiğini söylüyordu, yani davranış ile
           yazılı niyet ayrışmıştı.
Karar:  Mevcut davranış DOĞRU kabul edildi ve kasıtlı hale getirildi.
        Gerekçe: arama yabancı bir derlemedir ve bir sayı başka aya ait
        olabilir (ölçüldü: model Temmuz kararını Ağustos oranıyla yazmıştı);
        `read_url` ise kullanıcının kendi sorusunda verdiği belgeyi okur ve
        şartname "pdf, excel, resim veya ilgili yazı bölümlerinden anlam
        çıkartılması" diyor. Çiti de kesmek, cevabı tablo değil düzyazı olan
        her sayfada 5. kuralı uyulamaz hale getirirdi.
        `web_url` sabitleri `SOURCE_OPEN`/`SOURCE_CLOSE` diye adlandırıldı,
        eski adlar takma ad olarak duruyor. İki dosyada da ayrımın neden
        ayrım olduğu yazıldı. Doğrulayıcının docstring'i gerçeğe çekildi.
Sonuç:  Üç test ayrımı sabitliyor: arama parçacığındaki sayı bayrak alıyor,
        okunan sayfadaki sayı almıyor, ve iki çit metninin eşit olmaması
        ayrıca iddia ediliyor. Çitleri birleştirince üçünden ikisi kırmızıya
        düşüyor, denendi. Güvenlik tarafı değişmedi: iki çitin içi de talimat
        olarak okunmuyor.

## Üç oynak satır kapatıldı: adres, TÜFE tablosu, faiz ikizi (23 Eyl)
Bağlam: Üç ardışık ölçüm koşusu 10/12 verdi ama düşen satırlar değişiyordu.
        Adım kaydı üç kök neden gösterdi: model uzun adresi yeniden yazarken
        bozuyor (`main+menu/duyurular` -> `main+menu+duyurular`, çekim 404);
        `tables_asked` sorguya kelime eklenince kapanıyor (A6 üç koşuda üç
        farklı yanlış TÜFE serisi); ve "Konut Kredisi (TL, Akım, %)" ile
        "Konut Kredisi, (TL, Akım, %)" ad, birim, frekans ve türde birebir
        aynı, biri faiz öteki katılım bankası kâr payı (A1).
Karar:  1. **Adres sorudan sabitleniyor.** `read_url`, verilen adres soruda
           birebir yoksa ama aynı host'ta üç düzenleme uzağında bir adres
           varsa onu okuyor ve uyarı yazıyor. Host birebir karşılaştırılıyor,
           bir harflik host farkı başka site demektir. Sorudaki metin, ne
           okunacağının tek doğru kaynağı.
        2. **`tables_asked` iki okumanın birleşimi.** Düz okuma (sorgunun her
           kelimesi tablo adında) tablo adı sorgudan uzun olduğunda gerekli:
           "sanayi üretim endeksi" mevsimsellikten arındırılmış tabloyu da
           istiyor ve arama onun başını bekliyor. Gevşek okuma (tablonun
           parantezsiz adı sorgunun içinde) sorgu tablo adından uzun olduğunda
           gerekli: A6'nın "TÜFE" ekli sorgusu. 22 Eylül'de yalnız gevşek
           okumaya geçiş 25/27'den 23/27'ye düşürmüştü; birleşim 25/27'de
           tutuyor, üç sorgu da (SÜE 1, TÜFE 1, Ankara KFE 3) yerinde.
        3. **İl ya da ülke anan sorgu tablo istemiyor.** "Ankara konut fiyat
           endeksi" KFE tablosunu heceliyor ama Ankara satırını istiyor; o
           satır bir alt kalem ve tablonun toplamını öne almak onu beş adayın
           dışına atıyordu. Ülke için aynı şekil bir üst katta: "Almanya
           tüketici fiyat endeksi" TÜİK toplamını öne alıp Almanya'yı
           gömüyordu. Yer anılıyorsa `asked` boş.
        4. **`CatalogHit.table` eklendi** (isteğe bağlı, sözleşme belgesine
           yazıldı). `search_catalog` modele `tablo` olarak gösteriyor, onay
           kartı ayrıntının başına koyuyor. İkizleri ayıran tek alan bu.
Sonuç:  Agent ölçümü iki koşumda da **12/12**, sıfır bayrak, sıfır kesinti;
        ilk kez. A6 kendi sorgusuyla ("tüketici fiyat endeksi TÜFE")
        sözleşmedeki `TP.TUKFIY2025.GENEL`'i seçti, A1 `TP.KTF12`'yi. Arama
        ölçümü 25/27 ve ayrık 6/6, gerileme yok. Yeni testler dört sabotajda
        da kırmızıya düştü. A6'nın "bile bile kırmızı" notu artık geçersiz,
        soru dosyasından kaldırıldı.

## Etkileşimli CLI: `--chat` (23 Eyl)
Bağlam: Demo CLI'dan yapılacak (22 Eyl Slack: deploy zorunlu değil), ama
        bugünkü CLI demoyu taşımıyordu. Üç ölçülmüş boşluk: `deps.approve`
        üretim kodunda hiç atanmıyordu, yani her onay kartı `auto_default`'a
        düşüyor ve kararın notu "otomatik onay (etkilesimsiz mod)" oluyordu;
        üç mod da `JsonlSink` ile stdout'a yazdığı için soru başına kırk satır
        JSON akıyordu; sorular `--question` ile baştan verildiği için takip
        sorusu sorulamıyordu. Dördüncüsü kozmetik: kurulu `lakehouse-agent`
        komutu uv taslağına bağlıydı ve "Hello from lakehouse-agent!"
        yazdırıyordu.
Karar:  Yeni `--chat` bayrağı. Mevcut çağrılar hiç değişmiyor: `--question`
        ile başsız koşum, eval koşucusu ve bütün CLI testleri aynen duruyor.
        **Sunum ayrı modülde.** Bütün `rich` kullanımı `agent/console.py`
        içinde ve bunu bir test bekçilik ediyor, `chainlit` kuralının aynısı:
        motor arayüzü import etmez, `loop.py` `--chat` verildiğinde tembel
        import eder. Başsız koşum sunum kütüphanesini hiç yüklemez.
        `Terminal` üç dikiş sunuyor, üçü de enjekte edilebilir olduğu için
        ağsız test ediliyor: `emit` (`EventSink`), `approve` (`ApproveFn`),
        `ask` (sıradaki soru). Klavye okuması `asyncio.to_thread` ile, çünkü
        döngü async ve bu repoda engelleyici çağrının kalıbı bu.
        **Olaylar kaybolmuyor.** Sohbet modunda JSONL dosyaya yazılıyor
        (`--events`, varsayılanı `data/adhoc/sessions/<zaman>.jsonl`) ve ekran
        insan görünümüne kalıyor. JSONL replay ve eval sözleşmesi olduğu için
        kapatılmadı, yalnızca yeri değişti.
        **Model metni `markup=False` ile basılıyor.** `rich` `[...]`'i biçim
        kodu okuyor; `TP.KTF12 [aylik]` diye bir alıntı köşeli parantezi
        kaybederdi ya da daha kötüsü, bizim yazmadığımız metin ekranı
        renklendirirdi.
        `--chat` TTY istiyor; boru bağlıysa cümleyle reddediyor, çünkü boruyu
        insan yazısı sanıp okurdu.
Sonuç:  `--fake --chat` ağsız çalışıyor ve testleniyor. Bunun için `FakeLLM`'e
        `extend()` eklendi: kurucu listeyi kopyaladığı için REPL'de ikinci
        soru `ScriptExhausted` veriyordu, artık her soru geldiğinde bir turluk
        betik ekleniyor. Asıl kazanç sohbet döngüsünün kendisinin ağsız test
        edilebilmesi.
        18 yeni test. İki düzeltme tek tek bozulup kırmızıya düştüğü görüldü;
        bu sırada onay testimin boş liste üzerinde geçtiği ortaya çıktı ve
        belirsiz sahte aramayla kartı gerçekten tetikleyecek şekilde yeniden
        yazıldı.
        `rich` 15.0.0 eklendi (MIT), `docs/licenses.md` 72 paketle yeniden
        üretildi, lisans iddiası bozulmadı. `lakehouse-agent` komutu gerçek
        CLI'a bağlandı ve `pyproject` açıklaması taslak metinden çıkarıldı.
        Grafik kapsam dışı: hiçbir araç grafik üretmiyor, gösterilecek bir şey
        yok.

## Yapıştırılan soru tek sorudur (23 Eyl, ilk sohbet koşumundan)
Bağlam: `--chat`'in ilk gerçek koşusunda şartnamenin üç satırlık senaryosu
        yapıştırıldı ve **üç ayrı tur** koştu. İkinci tur kimsenin sormadığı
        bir parçaya gitti: "gösteriniz. Buna ek olarak konut kredisi faiz
        oranlarını da". Adım kaydındaki `search_catalog` sorguları bunu
        birebir gösterdi.
Neden:  Okuyucuyu satır bazlı yazmıştım. Bir soru satır değil paragraf, ve
        jüri demoda yapıştıracak; şartnamedeki soruların kendisi çok satırlı
        basılmış.
Karar:  İlk denemem "terminalde bekleyen satır var mı" diye bakmaktı
        (`select`, sıfır beklemeyle). ÇALIŞMADI ve nedeni ölçüldü: yapıştırma
        terminale parça parça geliyor, sıfır beklemeli kontrol sonraki parça
        gelmeden "bekleyen yok" diyor, kalan satırlar tamponda kalıp bir
        sonraki okumaya, yani onay kartının cevabına gidiyor. pty ile yeniden
        üretildi: 120 ms arayla gelen üç satırlık yapıştırmada soru "ilk
        satir" olarak kesildi ve onay kartı iki kez "Gecerli bir numara yaz."
        dedi. Kullanıcının gerçek koşumundaki belirti buydu.
        Bekleme penceresi ölçüldü: 150 ms 120 ms'lik boşluğu kapatıyor ve
        insanın 400 ms sonra yazdığı satırı yutmuyor. Ama pencere ne olursa
        olsun yarışı yalnızca daraltıyor, kapatmıyor; yavaş bir bağlantıda
        demo sırasında tekrar edebilirdi.
        Bu yüzden KESİN ÇÖZÜM seçildi: **boş satır soruyu gönderir**,
        satırlar o ana kadar biriktirilir, oturum Ctrl-D ile biter. Terminal
        hızından bağımsız, yarış yok. Bedeli tek satırlık soruda da ikinci
        bir Enter. Yarım yazılmış soruda Ctrl-D soruyu kaybetmiyor, gönderiyor.
Sonuç:  300 ms arayla gelen parçalı yapıştırmada bile soru bütün geliyor ve
        onay kartı uyarı vermiyor (pty ile doğrulandı). Beş test; paragraf
        birleştirmesi kaldırılınca ikisi kırmızıya düşüyor.

## Ekran tabloyu tam gösterir, önizlemeyi değil (23 Eyl, ilk `--real --chat` koşumundan)
Bağlam: Jüri sorusu 60 aylık veri istiyordu, ekranda 10 satır göründü. Veri
        eksik değildi: adım satırı "2 seri, 60 ay" diyordu ama tablo yalnızca
        önizlemeyi çiziyordu ve bunu hiç belli etmiyordu. Demoda bu "veri
        eksik" diye okunur.
Neden:  `table_update` olayı bilerek yalnızca önizleme taşıyor; yükte toplam
        satır sayısı diye bir alan bile yok. Ama sözleşme bu durumu zaten
        çözmüş: §4 "model tabloyu görmez, özetini görür. **Tam tablo oturum
        deposunda kalır ve doğrudan arayüze gider.**" Yani ekranın tabloyu
        depodan okuması tasarlanmış yol, sözleşme değişikliği değil.
Karar:  `Terminal` bir çözücü alıyor (`tables=state.tables.get`) ve tabloyu
        tam çiziyor. Üst sınır 200 satır, yalnızca kırk bin satırlık bir
        tabloya karşı koruma; kesildiğinde gerçek toplamı söylüyor. Depo
        bağlanmamışsa olayın önizlemesine düşüyor, yani eski çağıran bozulmuyor.
        Yanında iki ekran hatası daha düzeldi: kayan nokta gürültüsü
        (`-0.40299999999999997` ekranda `-0.403`) ve kesilen sütun adları
        (`overflow="fold"`; ad bir sonraki sorunun kullanacağı şey, okunur
        kalmalı). Onay isteminde "bos=varsayilan" yerine "bos=1" yazıyor,
        çünkü kullanıcı hangi numaranın varsayılan olduğunu aramak zorunda
        kalmıştı.
Sonuç:  Gerçek gold verisiyle çizdirildi: 60 aylık tablo 60 satır çıkıyor,
        sütun adı sarılıyor, sayı gürültüsü yok. Altı yeni test. Üç düzeltme
        tek tek bozulup kırmızıya düştü; bu sırada katlama testinin geniş
        ekranda ayırt etmediği görüldü ve dar ekrana çekildi.

## Onay kapısı daraltıldı: kart seçim içindir, aramanın kararsızlığı için değil (23 Eyl)
Bağlam: İlk `--real --chat` koşumunda iki kart çıktı. İkincisi gerçek bir
        seçimdi (faiz mi kâr payı mı, akım mı stok mu) ve eklenen tablo adı
        sayesinde insan ayırt edebildi. Birincisi değildi: takipteki konut
        kredileri, tüketici konut kredileri, toplam krediler ve kısa vadeli
        toplam krediler, yani **dört ayrı büyüklük**. Kullanıcı "bunlar
        arasında toplam konut kredileri diye bir seçenek yok ki" dedi ve
        haklıydı; varsayılan da yanlıştı (takipteki krediler).
Neden:  Kapı yalnızca skor yakınlığına bakıyordu. 45 bin serilik havuzda her
        şey her şeye yakın skor alıyor. **Ölçüldü: 29 gerçek sorgunun 29'unda
        da kart çıkıyordu**, yani her arama tıklama için duruyordu. Skor
        yakınlığı iki ayrı şeyi karıştırıyor: aynı büyüklüğün birkaç yazılışı
        (gerçek seçim) ile aramanın farklı büyüklükler arasında kararsız
        kalması (seçim değil, ve orada adayların adını ve kapsamını okuyan
        model insandan daha iyi karar verir).
Karar:  Kart yalnızca adaylar **aynı büyüklüğün varyantları** görünüyorsa
        çıkıyor: gösterilen bütün adayların ortak kelime oranı 0,40'ın
        üstündeyse. Ölçüm, 29 sorgu: dört kart kalıyor, 25'i kayboluyor.
        Kalanlar — mevduat faizi vadeye göre (1,00), bireysel kredi kartı
        borçları kart grubuna göre (0,80), konut kredisi faizi (0,60),
        ihtiyaç kredisi faizi (0,44) — dördü de "bir büyüklük, birkaç tanım".
        Kaybolanlar: "konut fiyat endeksi" KFE ile "Konut Sigortası"nı yan
        yana koyuyordu (0,00), "dolar kuru" Kuveyt dinarını (0,00), "toplam
        konut kredisi bakiyesi" o dört ayrı büyüklüğü (0,14).
        Oran **gösterilen bütün adaylar** üzerinden ölçülüyor, ilk ikili
        üzerinden değil: kötü kartın ilk ikilisi de üç kelime paylaşıyor
        ("Takipteki Konut Kredileri" ile "Tüketici Kredileri - Konut"), yani
        ilk ikiliye bakmak o kartı tutardı. Dördü birden yalnızca "toplam"
        paylaşıyor.
        Kelime katlama `approvals.py` içinde yerel yazıldı; agent katmanı
        `lake`'ten hiçbir şey import etmiyor ve bir yardımcı için ilk kenarı
        açmaya değmez.
Sonuç:  Uçtan uca ölçüm 12/12'de kaldı, gerileme yok. Dört yeni test; daraltma
        kaldırılınca ve oran yalnızca ilk ikiliye bakacak şekilde bozulunca
        kırmızıya düşüyor.
        AÇIK: insanın seçtiği serinin bağlayıcı olup olmayacağı hâlâ karara
        bağlı. Kapı daraldığı için artık güvenli, çünkü kart yalnızca iki
        seçeneğin de soruyu cevapladığı durumlarda çıkıyor.

## Cevap soruyu görmüyordu, ekran tabloyu beş kez çiziyordu (23 Eyl, demo provasından)
Bağlam: Şartnamenin üç parçalı sorusu koşuldu. Araç doğru cevabı buldu
        (`'faiz_dustu_kredi_artmadi' sutunu eklendi. 3 ay kosullarin hepsini
        sagliyor: 2021-05, 2023-11, 2024-01`) ama modelin yazdığı son cevap
        şuydu: "Faizin düştüğü ayların sayısını ve dağılımını görmek için ek
        bir işaretleme yapıyorum." Yani iş bittiğini fark etmemiş, niyet
        cümlesi yazmış. Tur kesilmemişti, doğrulama da geçmişti; cevap
        boştu ama "geçerli" görünüyordu.
Neden:  **Soru cevap promptuna hiç verilmiyordu.** `_answer` `question`
        parametresini alıyor ama `render`'a geçirmiyordu; model, aralarında on
        beş adımlık araç çıktısı olan ilk kullanıcı mesajından soruyu
        hatırlamak zorundaydı. `_answer_messages`'ın docstring'i aynı modelde
        çok yakın bir bozulmayı zaten kaydetmişti (22 Eyl, "16 token saçmalık").
Karar:  `answer.md` v4: soru promptun içinde, "Sorunun her parçasını cevapla",
        "bir araç sana dönem listesi verdiyse onu yaz", ve "ne yapacağını
        anlatma, yaptığının sonucunu söyle; bu adım bitti, yeni araç
        çağıramazsın".
Sonuç:  Aynı soru tekrar koşuldu. Cevap: "Faizin düştüğü ama kredinin
        artmadığı aylar sorusunun cevabı net: tabloda işaretlenen dört ay var
        — 2021-02, 2021-05, 2023-12 ve 2024-01." Üç alt sorunun üçü de
        cevaplanıyor, karşı örnekler veriliyor, doğrulama 0 bayrak.

        Aynı provada ekran da düzeldi: tablo her dönüşümden sonra yeniden
        çiziliyordu, dört sütun eklenen bir tur aynı 60 satırlık tabloyu beş
        kez bastı ve cevap ekranın üstünden kaydı. Artık tur sürerken yalnızca
        "tablo t01_get_series: X eklendi" satırı çıkıyor, tablo `done` ile bir
        kez çiziliyor. Tablo deposu bağlanmamış bir çağıran için eski davranış
        (olayın kendi önizlemesi) korundu, yoksa hiç tablo görmezdi.

## Tarayıcı arayüzü: `app.py` kökte, mantık `webui.py`'de (23 Eyl)
Bağlam: Şartname çıktıyı "rapor, grafik veya tablo" diye tanımlıyor; jüriye
        gösterilecek yüzey tarayıcı. Motor hazırdı: on üç olay, `approve`
        async, sözleşme §6 her olayın arayüz karşılığını yazıyor.
Karar:  Chainlit uygulaması `src/` dışında, repo kökünde `app.py`.
        `test_agent_and_tools_do_not_import_chainlit` `src/lakehouse_agent`
        altını muafiyetsiz tarıyor; kök dışarıda kaldığı için kural tek harfi
        değişmeden yürürlükte. Bedeli `app.py`'nin pytest'ten import
        edilememesi; bu yüzden karar veren her şey `agent/webui.py`'de
        (`action_specs`, `decision`, `default_chart`, metin yardımcıları),
        `chainlit` ve `plotly` import etmez ve test edilir. `app.py` yalnızca
        `cl.*` çağrıları. `plotly` de yalnızca orada:
        `test_no_module_imports_plotly`, `rich` ve `chainlit` kuralının
        üçüncü sürümü.
        Onay kartı `cl.AskActionMessage` ile düğme; cevap gelmezse (zaman
        aşımı, kapanan sekme, karttaki olmayan seçenek) `default` +
        `automatic=True`. Bayrak kozmetik değil: `decision_message` kişi
        seçtiğinde modele başka cümle söylüyor.
        Tablo tur sonunda, `done`'da bir kez ve **tam** (`cl.Dataframe` kendi
        sayfalıyor; §4 "tam tablo doğrudan arayüze gider"). Terminalin 200
        satır sınırı burada yok.
        Oturum kimliği `cl.user_session["id"]`; `"cli"` sabiti terminalde
        kaldı. Olaylar `data/adhoc/sessions/<zaman>-<id>.jsonl`'e de yazılır.
        `.chainlit/config.toml` commit edilir (ad, açıklama), `translations/`
        edilmez: silinince açılışta yeniden üretildiği ölçüldü.

## Grafik: model tarif eder, `tools/charts.py` sözlük kurar, `app.py` çizer (23 Eyl)
Bağlam: Boru hattının tamamı vardı — `ChartSpec.plotly_json`, `add_chart`,
        `chart` olayı, `answer.md`'de `Grafikler: {{charts}}` — ama
        `add_chart`'ı çağıran üretim kodu yoktu; yalnızca testler çağırıyordu.
Karar:  `make_chart` aracı (`tools/charts.py`). `ChartSpec` docstring'inin
        seçtiği tasarım uygulandı: "The model describes it; it never draws
        it." `figure_json` plotly'nin tel biçimini (`{"data", "layout"}`) düz
        sözlük olarak kurar, kütüphane yok; `Event.payload` JSON kalır.
        `cl.Plotly` gerçek `Figure` istediği için sarmalama `app.py`'de
        (`go.Figure(sözlük)`). Eksik değer `None` olarak gider, satır
        düşürülmez: plotly null'da çizgiyi keser, düşürülen satır iki komşuyu
        düz çizgiyle birleştirip ölçülmemiş ayı icat ederdi (CLAUDE.md #2).
        Grafik `ctx.add_chart` ile döner, `artifacts` ile DEĞİL: döngü
        ikisini de `state.charts`'a ekliyor (`loop.py` 1208-1215), ikisini
        birden vermek çift sayardı; sabotajla kanıtlandı.
        `ToolContext.new_chart_ref` eklendi: oturumdaki + bu adımda yazılan
        grafikleri birlikte sayar; `new_table_ref`'in aynısı.
        `planner.md` v9, kural 20: zaman serisi cevabında tablo hazır olunca
        `make_chart`. `max_steps = 12` ve kural 12 adım cimriliği istediği için
        model bazı turlarda grafiği atlayacak; çare motorda değil arayüzde:
        tur grafiksiz bitmişse `app.py` aktif tablodan varsayılan çizgi grafik
        çizer ve başlığına "(arayuz cizdi)" yazar. Eval'de bu yedek yok, yani
        `grafik_var` kontrolü modelin aracı çağırdığını ölçer. C1 sorusu
        eklendi; `test_the_shipped_questions_cover_every_registered_tool`
        yeni aracı sorusuz bırakmayı zaten reddediyordu.

## `LakeAdapter` yarışı: bayrak indeksten önce set ediliyordu (23 Eyl)
Bağlam: Chainlit çok oturumlu; tek `LakeAdapter` bütün oturumlarca paylaşılır
        (indeks ~0,5 GB, oturum başına açmak anlamsız) ve araçlar worker
        thread'de koşar.
Neden:  `_pool_index` `self._index_checked = True` yazıp SONRA
        `self._index = PoolIndex.open(...)` yapıyordu (`lake_adapter.py`
        365-367). Açılış sürerken gelen ikinci çağıran "bakıldı, indeks yok"
        okuyup sessizce kelime aramasına düşüyordu; uyarı bile yoktu.
Karar:  `threading.Lock` (asyncio değil: thread'deyiz), çift kontrollü;
        notlar dâhil her şey hazırlanıp bayrak **en son** set edilir.
Kanıt:  `test_two_threads_opening_the_index_at_once_both_get_it`: açılışa
        50 ms gecikme, iki thread. Kilit kaldırılıp eski sıra geri konunca
        test kırmızı (sabotaj koşumu, 7/7 yakalandı).

## Grafik ve tablo başlıkları insan için, sütun adları model için (23 Eyl, ekran görüntüsünden)
Bağlam: İlk gerçek grafikte lejant `aylik_tuketici_tuketici_kredileri_konut_
        toplam_sektor` ve `TP_KTF12` yazıyordu; tabloda `TP_KTF12_diff`. Tablo
        adlarını bilmeyen biri için anahtar, başlık değil.
Neden:  Ad değiştirilemez: planner kural 3 modele sütun adını bir sonraki
        çağrıya **birebir** kopyalamasını söylüyor, eval kontrolleri
        (`deger: {sutun: "TP_KTF12"}`) o adı okuyor. Ama insan için gereken
        bilgi zaten oturumda: `ColumnInfo` (sözleşme §5, anahtar
        `"{table_ref}.{column}"`) çekilen sütunda kataloğun `name_tr` ve
        `unit` değerini, türetilmişte `op` ve `inputs`'u taşıyor.
Karar:  `agent/labels.py`, saf: `column_label` etiketi soyağacından türetir,
        uydurmaz, kısaltmaz. Çekilen sütun = katalog adı + birim (birim adın
        içinde geçiyorsa tekrar yok; `TP.KTF12`'nin adı `%` taşıyor).
        Türetilmiş sütun = girdinin etiketi (**birimsiz**) + işlem sözü
        (`OP_PHRASES`: fark, % değişim, endeks, payı…) + sütunun kendi
        birimi; baz dönem sütun adının sonundan okunur (`_endeks_202101`).
        Soyağacının bilmediği sütun kendi adıyla kalır; `period` → Dönem.
        Etiketler üç yere gider: `make_chart` lejant ve eksenlere
        (`figure_json(labels=)`; `ChartSpec.series[].column` ad olarak
        kalır), tarayıcı tablosu `webui.display_frame` ile **kopyanın**
        başlıklarına (oturumun çerçevesi dokunulmaz; aynı etiketli iki
        sütun kendi adında kalır, tek başlığa çökmez), terminal tablosu
        başlığa etiket + altına soluk ham ad (sonraki soru o adı kullanacak).
        Modelin gördüğü `tables_summary` ham adla kalır.
        Sahte `get_series` de `label_tr`'yi sahte katalogdan yazıyor artık;
        gerçek araç bunu zaten yapıyordu, prova demo ekranını göstermeliydi.
Ölçüm:  Gold veri, gerçek araçlar, model yok. İlk sürüm endeks sütununda
        girdinin birimini sürükledi: "Tüketici Kredileri - Konut (Toplam)
        (milyon TL) — endeks (2021-01 = 100)". Endeksin birimi milyon TL
        değil; türetilmişte girdi birimsiz alınır oldu. Şimdi:
        `TP_KTF12` → "Konut Kredisi (TL, Akım, %) (Ağırlıklı ortalama)"
        (katalogda `unit` sütununda "Ağırlıklı ortalama" yazıyor; kaynağın
        verisi, düzeltilmedi), `TP_KTF12_diff` → "… — fark (yuzde puan)",
        `..._pay` → "Türkiye_Konut_İpotekli Satışlar / Türkiye_Konut_Toplam
        Satışlar payı (%)". Sekiz sabotaj, sekizi yakalandı. `answer_eval`
        yeniden koşulmadı: modelin girdisi (araç özeti, sütun adları)
        değişmedi, yalnızca insan tarafı.

## "300k" 300 milyon değil 300 milyar; tarihte saat yok (23 Eyl, ikinci ekran görüntüsünden)
Bağlam: Grafikte sol eksen "300k", başlık "(milyon TL)"; okuyan "300 milyon"
        anladı. Tabloda dönem `2021-01-01T00:00:00.000`.
Ölçüm:  `bddk:aylik:tuketici:tuketici_kredileri_konut:toplam:sektor` 2021-01
        değeri **276.785**, birimi `milyon TL`. Yani 300k = 300 bin milyon TL
        = 300 milyar TL. Birim doğru; plotly'nin SI kısaltması (`k`) Türkçe
        birim başlığının yanında yanlış okunuyor.
Karar:  Birim çevrilmedi (CLAUDE.md #3; grafik tabloyla aynı sayıyı
        göstermeli). Değer eksenlerinde `exponentformat: "none"` (sayı tam
        yazılır), `separators: ",."` (Türkçe: ondalık virgül, binlik nokta).
        "300k" → "300.000", başlık "(milyon TL)" olduğu gibi. Tarih ekseni
        sayı değil, dokunulmadı.
        Tarih: `webui.display_frame` gösterim kopyasında datetime64
        sütunları `YYYY-MM-DD` yazar; `period` sözleşmede tarihtir, saat
        veri değil serileştirme gürültüsüydü. Oturumun çerçevesi datetime64
        kalır. Terminal zaten `json_safe` ile tarih yazıyordu.
        Üç sabotaj (kısaltma geri, İngilizce ayraç, saat geri), üçü yakalandı.

        Açık soru: `TP.KTF12` etiketi "Konut Kredisi (TL, Akım, %)" — bu
        kataloğun `name_tr`'si, "faiz oranı" kelimesi adında yok. Kelime
        `topic`'te ("FAİZ VE KÂR PAYI İSTATİSTİKLERİ (AKIM)"), `tags`'te
        ("kredi, faiz"), `notes`'ta ve bronze veri grubu adında ("Kredi Faiz
        Oranları (Akım)") var; veri grubu adı gold kataloğa taşınmıyor
        (sütun listesi sözleşme). Nasıl gösterileceği kullanıcıya soruldu.

## Lejant etiketini model yazar, katalog yedektir (23 Eyl, kullanıcı kararı)
Bağlam: "Konut Kredisi (TL, Akım, %)" kataloğun kendi adı; "faiz oranı"
        kelimesi adında yok. Kullanıcı okuyanın anlayacağı etiket istedi.
        Üç yol soruldu: model yazsın (katalog yedek), EVDS veri grubu adını
        kataloğa taşı (şema değişikliği, gold yeniden kurulur, BDDK'da
        karşılığı yok), olduğu gibi kalsın. İlki seçildi.
Karar:  `make_chart` seri başına isteğe bağlı `label` alır (≤ 80 karakter;
        şemada sınır ve örnek var). `ChartSeries.label` kaydeder. Araç
        önce kataloğun etiketini kurar (`agent/labels.py`), sonra modelin
        yazdığını üstüne yazar; boş ya da yalnız boşluk etiket üstüne
        yazmaz. `title` zaten modelindi, `label` aynı sınıf: model tarif
        eder, çizmez. Planner kural 20 etiketi ister ve sınırı koyar:
        "sorudaki dille yaz ama anlamı değiştirme — stoku akım, oranı tutar
        diye adlandırma". Doğrulayıcı etiketi denetlemez; bilinen risk.
        Tablo başlığı lejantla aynı şeyi söyler: `webui.chart_labels`
        oturumun grafiklerinden o tablonun sütun etiketlerini okur (son
        grafik kazanır, etiketsiz seri katkı vermez), `display_frame` ve
        terminal tablosu kataloğun etiketinin üstüne yazar. Sütun adı
        yine ad; modelin gördüğü özet değişmez.
Kanıt:  Beş sabotaj (etiket kullanılmıyor, boş etiket kataloğu eziyor,
        tablo etiketi almıyor, başka tablonun etiketi sızıyor, terminal
        almıyor), beşi yakalandı.
Ölçüm:  Gerçek modelle tek soru ("Konut kredisi faizi ile konut kredisi
        bakiyesini 2021-2025 için grafikle göster"). Model `make_chart`'ı
        şöyle çağırdı: `TP_KTF12` → label "Konut kredisi faiz oranı (%)",
        bakiye sütunu → "Konut kredisi bakiyesi (milyon TL)", ikincisi sağ
        eksende, başlık "Konut kredisi faiz oranı ve konut kredisi bakiyesi
        (2021-2025)". Birimi kimse istemedi, araç özetinden kendisi taşıdı.
        Lejant ve eksen başlıkları bu etiketlerle çıktı; sütun adları ve
        modelin gördüğü özet değişmedi.

## Yedek grafik önceki sorunun tablosunu çizdi (23 Eyl, üçüncü ekran görüntüsünden)
Bağlam: 13. soru (TCMB duyurusunu oku) doğru cevaplandı, altına konut satış
        serilerinin grafiği geldi: "t03_get_series (arayuz cizdi)".
Neden:  Yedek grafik "bu turda grafik eklendi mi" ve "aktif tablo var mı"
        diye bakıyordu. Aktif tablo turdan tura yaşar; `read_url`'ün
        sayfasında tablo yoktu, aktif tablo 3. sorudan kalan `t03` idi.
        `TurnResult.table_refs` de yardımcı değil: oturumun bütün
        tablolarını taşıyor, bu turunkileri değil (`loop.py` 337).
        İkinci kusur aynı grafikte: üç sütun tek eksende, "payı (%)" adet
        ekseninde sıfıra yapışık düz çizgi.
Karar:  `webui.fallback_table(active, touched, charts_added)`, saf: yalnız
        bu turda `table_update` ile dokunulan tablo, yalnız tur kendisi
        grafik çizmediyse. `ChainlitSink` turun dokunduğu tabloları `done`
        anında `last_touched` olarak saklar; `app.py` kararı vermez, sorar.
        `default_chart` sütunları soyağacındaki birime göre eksene dağıtır:
        ilk birim sol, ikinci sağ, üçüncü çizilmez — yanlış ölçekte çizmek
        yerine eksik çizmek. Birimi olmayan sütunlar kendi grubudur.
Kanıt:  Dört sabotaj (bayat tablo, grafik çizilmişken yedek, ikinci birim
        solda, üçüncü birim çiziliyor), dördü yakalandı. Tablosuz bir
        turda artık grafik yok; `docs/manual_test.md` 13. soru bunu
        bekliyor.

## Modelin sözcükleri türetilmiş sütunlara da akar; tablo dört ondalık; ad "🥬 Ispanak" (23 Eyl)
Bağlam: Tablo başlıkları karışık görünüyordu: modelin lejant için verdiği
        etiket yalnız grafiğe koyduğu sütunlara gidiyordu, `_diff`, `_pay`
        ve koşul sütunları kataloğun uzun teknik adından üretiliyordu, koşul
        sütunu ise ilk girdisinden ("…faiz oranı — koşul işareti") yanlış
        adlanıyordu. Hücrelerde 1716481.7634920634 gibi değerler vardı.
Karar:  `column_label` modelin etiketlerini (`overrides`) **sonuçtan sonra
        yapıştırmak yerine** türetimin başında okur: `TP_KTF12` "Konut
        kredisi faiz oranı (%)" ise `TP_KTF12_diff` "Konut kredisi faiz
        oranı (%) — fark (yuzde puan)" olur. Model etiketi olduğu gibi
        alınır, birim eklenmez. `mark` sütunu modelin verdiği adı taşır
        (`faiz dustu kredi artmadi (koşul)`), girdilerini değil. Katalog
        adlarındaki ve araç sütunlarındaki alt çizgiler boşluk olur
        (`Türkiye Konut İpotekli Satışlar`, `ortalama öncesi`); başka
        hiçbir harf değişmez. `make_chart`, `display_frame` ve terminal
        aynı yolu kullanır.
        Tablo gösterim kopyasında float sütunlar 4 ondalığa yuvarlanır
        (terminalle aynı `DECIMALS`); oturumun çerçevesi her basamağı
        tutar, eval ve sonraki araç oradan okur.
        Chainlit `[UI] name = "🥬 Ispanak"`; ıspanak emojisi Unicode'da yok,
        en yakını "leafy green" (U+1F96C). Karşılama sayfası başlığı aynı.
Kanıt:  Beş sabotaj (etiket türetilmişe akmıyor, koşul girdisinden adlanıyor,
        alt çizgi kalıyor, yuvarlama yok, grafik aracı etiketi türetilmişe
        vermiyor), beşi yakalandı. Gerçek gold tablosunda ölçülen
        başlıklar bu girdinin altındaki koşum çıktısında.

## Takip sorusu tabloyu ve grafiği yeniden getirdi; yedek grafik katalog adıyla çıktı (23 Eyl)
Bağlam: 11. sorudan sonra "faizin en yüksek üç ayı" soruldu. Cevap doğru
        (`top`: 2024-04, 2024-05, 2024-06), ama altına 60 satırlık tablo
        yeniden geldi ve "t03_get_series (arayuz cizdi)" grafiği ikinci kez
        çizildi; lejantı da modelin sözcükleri değil katalog adlarıydı.
Neden:  Üç ayrı kusur. (1) `top` sütun eklemez ama `table_update` fırlatır
        (`new_columns` boş); arayüz her `table_update`'i "dokunuldu" sayıp
        tabloyu çiziyordu. (2) Yedek grafik "bu turda dokunulan tablo" diye
        bakıyordu; takip sorusu tabloya dokunur. (3) `default_chart`'a
        modelin etiketleri (`chart_labels`) verilmiyordu; tablo başlığı
        modelin sözcüklerini alırken grafik kataloğu gösteriyordu. Bir de
        iki seri aynı eksende olunca eksen başlığı iki uzun etiketin virgülle
        birleşimi oluyor, grafiğin üstünden taşıyordu.
Karar:  `webui.note_table(known, ref, fresh) -> (yeni mi, çizilsin mi)`:
        tablo ilk görüldüğünde ve sütun eklendiğinde çizilir, yalnız
        okunduğunda çizilmez. Tarayıcı ve terminal aynı kuralı kullanır.
        `fallback_table` yalnız **bu turda kurulan** tablo için: yedek grafik
        "model adımlarını bitirdi, tablo grafiksiz kaldı" durumu içindir,
        takip sorusu için değil. `default_chart` `overrides` alır, arayüz
        `chart_labels`'ı geçer. `figure_json(units=)`: bir eksende birden
        çok seri varsa ve birimleri aynıysa eksen başlığı birimdir
        ("milyon TL"), farklıysa etiketler; tek seride etiket.
        Soruya cevap: her soruda tablo/grafik zorunlu değil. Model 14. soruda
        grafik istemedi; ikisini de arayüz kuralları getirdi, kapatıldı.
Kanıt:  Beş sabotaj (okunan tablo yine çiziliyor, terminal de basıyor, yedek
        grafik modelin etiketini almıyor, paylaşılan eksen etiketlerle
        başlıklanıyor, farklı birimler tek birimle), beşi yakalandı. Canlı
        kontrol `docs/manual_test.md` takip sorusuyla elle yapılır.

        Logo ve favicon: `public/logo_light.png`, `public/logo_dark.png`,
        `public/favicon.png`. Chainlit `public/` altındaki bu adları
        kendi varsayılanının yerine servis ediyor (`server.py` `/logo`,
        `/favicon`); ölçüldü, üçü de bizimki geldi. Dosyalar Pillow ile
        Apple Color Emoji (🥬, 160 px tek bitmap boyu) + Arial Rounded
        Bold'dan üretildi; üretici betik commit edilmedi, macOS fontuna
        bağlı, PNG'ler kaynaktır.
