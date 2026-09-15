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
