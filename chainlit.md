# 🥬 Ispanak

EVDS ve BDDK verisi (2021-01 – 2026-06) üzerinde çalışan finansal analiz
asistanı. Türkçe sor; agent önce hangi veriye ihtiyacı olduğunu düşünür, aracı
çağırır, sonucu görür, sonra cevaplar. Cevaptaki her sayı bir araç çıktısından
gelir, hiçbiri zihinden hesaplanmaz.

**Ekranda ne göreceksin**

- Her araç çağrısı bir adım kutusu; açıp girdi ve özetini görebilirsin.
- Tur bitince kurulan tablo tam haliyle, sayfalanmış olarak gelir.
- Zaman serisi cevaplarında grafik çizilir; grafiği agent istememişse arayüz
  aktif tablodan kendisi çizer ve başlığında bunu söyler.
- Agent emin olamadığında durur ve sana sorar: hangi seri, havuz dışı veri
  çekilsin mi, kalitesi düşük tabloyla devam edilsin mi. Düğmeye basmazsan
  varsayılan uygulanır ve bu cevaba "otomatik" diye işlenir.
- Doğrulayıcı cevabı araç çıktılarına karşı kontrol eder; doğrulayamadığı
  ifadeleri ayrıca listeler.

**Örnek sorular**

- Konut kredisi faiz oranı son 12 ayda nasıl seyretti?
- Konut fiyat endeksinde aykırı aylar var mı?
- Konut kredisi faizinde yapısal bir kırılma var mı, varsa ne zaman?
- 2021-2025 arası konut kredisi ve faiz oranlarını göster. Faiz düştüğü halde
  kredinin artmadığı dönem oldu mu?
