# Karar Kayıtları

## EVDS erişim yöntemi
Bağlam: EVDS Ocak 2026'da v3'e geçti. evds2 ve /service/evds yolu ölü
        (302 -> evds3 ana sayfa, 12 Eyl 2026 doğrulandı). İnternetteki üçüncü
        taraf örneklerin tamamı v2 dönemine ait, güvenilmez.
Karar:  Resmi kılavuzdaki v3 web servisi kullanılacak:
        https://evds3.tcmb.gov.tr/igmevdsms-dis/  — anahtar HTTP header'da "key".
        Kaynak: EVDS Web Servis ve API Kullanımı kılavuzu (docs/refs/).
Sonuç:  ingest/evds.py bu base'i kullanır. 678 veri grubu, 154 kategori.
