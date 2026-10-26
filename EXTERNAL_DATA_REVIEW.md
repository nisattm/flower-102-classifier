# Hazır dış veri incelemesi

22 Ekim 2026 itibarıyla dış görüntü eklenmedi. [Pl@ntNet-300K resmî deposu](https://github.com/plantnet/PlantNet-300K) 306.146 görüntü, 1.081 bilimsel tür, yüksek etiket belirsizliği ve güçlü sınıf dengesizliği bildiriyor. [Zenodo kaydı](https://zenodo.org/records/5645731) görüntü arşivini 31,7 GB olarak listeliyor. Depo, görüntü kimliği, tür kimliği, bölünme, yazar ve lisans içeren meta veri dosyalarını ayrıca sunuyor.

Oxford sınıf adları çoğunlukla ortak ad. Bu adları Pl@ntNet türlerine yalnız metin eşitliğiyle bağlamak güvenilir değil; örneğin Oxford'daki bazı adlar birden fazla biyolojik türü kapsayabilir. Önce bilimsel takson eşleştirmesi ve çiçek organı filtresi gerekir. Bu doğrulama tamamlanmadan büyük görüntü arşivi indirilmedi ve dış veri eğitime katılmadı.

Bu karar ilk temiz temel model deneylerini geciktirmemek için verilmiştir. Dış veri eklenirse ayrı eğitim/geliştirme/dış test bölünmeleri, gözlem grupları, Oxford kopya taraması ve kapsanan sınıf listesi ayrıca kaydedilmelidir.
