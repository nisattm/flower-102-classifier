# CPU hedefi

23 Ekim 2026'da kullanıcı, hocanın teslim edilen modeli CPU'da çalıştıracağını bildirdi. 25 Ekim'de yarışma yönergesi PDF dosyası geldi. Eğitim GPU'da devam edebilir; son model seçimi ve hız karşılaştırması CPU çıkarımına göre yapılacaktır. Kullanıcı 25 Ekim'de çalışmaya yeniden devam edilmesini istedi. Teslim tarihi 30 Ekim 2026.

Bilinenler:

- Belgede final puanı `%60 doğruluk + %25 hız puanı + %15 model boyutu puanı` olarak tanımlanıyor. Aynı Intel Core i7 / 16 GB RAM bilgisayar kullanılacak; GPU yok.
- Hız puanı `min(100, 50/T × 100)`, boyut puanı `min(100, 35/S × 100)` biçiminde. **50 ms ve 35 MB yalnız önerilen eşiklerdir; kesinleşmemiştir.** Bu formüllere göre eşiğin altındaki ek hız veya boyut kazancı puan getirmez.
- Hızın ön işleme, çözünürlük, batch, ısınma ve zamanlama ayrıntıları henüz açıklanmadı. Belgedeki örnek hız hesabında `80/25` yazıyor; metin `50 ms` dediği için öğretmenden teyit istenmeli. Örnek her iki eşikte de 100'e doyduğu için örnek final puanı değişmiyor.
- FP16 olarak kaydedilen ağırlıklar `predict.load_model(..., device='cpu')` ile FP32 modele yüklenir. Bu dosyayı küçültür, CPU hesaplamasını tek başına hızlandırmaz.
- İlk CPU ölçümleri `EXPERIMENT_REPORT.md` içindedir; bunlar kullanıcının bilgisayarında yapılmıştır ve hocanın CPU'sunu temsil ettiği varsayılmayacaktır.
- Finalistler için protokoldeki 50 ısınma ve 5×1000 tekrar CPU'da çalıştırılacak; tensör, açılmış fotoğraf ve disk dahil süreler ayrı kaydedilecektir. Teslim edilecek gerçek dosya ve arayüz ayrıca CPU'da doğrulanmalıdır.
- Mevcut FP32 paket ve yerel disk-dahil ilk CPU ortalamaları, **yalnız geçici 50 ms / 35 MB eşikleriyle**, EfficientNet-B0 için yaklaşık `99.32`, bilgi aktarılmış MobileNetV4 için `98.96` puan verir. İkisi de iki eşiğin altında kalır; bu varsayımda doğruluk farkı belirleyicidir. Bu sayılar yarışma skoru veya hocanın bilgisayarındaki ölçüm değildir.
- Öğretmen ConvNeXtV2 Tiny dosyası FP16'da bile `55.958 MB` ile önerilen 35 MB eşiğini aşar; yerel CPU disk-dahil ilk ortalaması da yaklaşık 91 ms'dir. Bu nedenle bağımsız teslim adayı olarak ek fold eğitimine öncelik verilmemeli. Önceden hazırlanmış `flower_colab_fold1.ipynb` öğretmen fold1 eğitimi başlatılmamalı; ileride gerekirse ayrı iş için yeniden hazırlanmalı.
- Önceki plandaki "doğruluk liderinin 1 puan gerisindeki en hızlı model" seçim kuralı artık ana seçim kuralı değildir. Onun yerine açıklanan puan formülü, kesin eşikler ve finalistlerin üç fold doğruluğu kullanılacaktır.
- Verilen arşivdeki 819 görüntülük `test` bölümü bizim **iç kilitli testimizdir**. Belgede sözü edilen, öğrencilerin erişemediği **yarışma gizli testi** ayrı bir değerlendirmedir.
- Belge izin verilen veri ve kaynakları tek tek saymıyor. Hazır ön eğitim ağırlıkları ve dış fotoğraf kullanımının serbest olduğu öğretmenden teyit edilmeli; bu netleşmeden dış veri eklenmemeli.
- Teslim arayüzü ve zorunlu dosyalar henüz açıklanmadı. Mevcut `predict.py` CPU'da çalışıyor, fakat hocanın isteyeceği yükleme arayüzüne sonradan uyarlanması gerekebilir.

Kilitli test, model ayarları CPU sonuçlarıyla dondurulduktan sonra bir kez kullanılacaktır.

## 30 Ekim teslimine göre öncelik

1. EfficientNet-B0 temel ayarının fold 1 kaydedilmiş en iyi ağırlığını CPU/FP32 doğrula; fold 2'yi aynı dondurulmuş bölünmeyle eğit.
2. MobileNetV4 doğrudan eğitiminin fold 1 ve 2 sonuçlarını al. Bilgi aktarımının fold 0 kazancı yalnız iki görüntü olduğundan yeni öğretmen fold eğitimlerine öncelik verme.
3. EfficientNet ve MobileNet'in gerçek teslim dosyalarını CPU'da 50 ısınma + 5×1000 tekrar protokolüyle ölç. 50 ms eşiğine güvenli mesafeyi ve 35 MB dosya sınırını değerlendir.
4. Üç fold doğruluğu ve açıklanan puan formülüyle tek modeli seç; arayüz, çevrimdışı yükleme ve 102 sınıf indekslerini doğrula.
5. Ayarlar sabitlenince verilen arşivin 819 görüntülük iç kilitli testini bir kez aç. Yarışma gizli testinin sonucunu yalnız öğretmen görecektir.
