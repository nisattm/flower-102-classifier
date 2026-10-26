# Deney raporu

Oluşturulma: 2026-10-26T14:08:41+03:00

## Veri denetimi

- Görüntü: 8189; sınıf: 102.
- Train/valid/test: 6552/818/819.
- Birebir kopya grubu: 4; görsel olarak doğrulanan yakın kopya çifti: 98.
- Kilitli testte yakın kopyası olduğu için geliştirmeden çıkarılan görüntü: 25.
- Farklı etiketli aynı fotoğraf nedeniyle çıkarılan görüntü: 2.
- Geliştirme havuzundan çıkarılan birebir dosya kopyası: 4.
- Üç doğrulama parçası: {'0': 2478, '1': 2447, '2': 2414}.
- Oxford sınıf adı doğrulama hatası: 0.
- Oxford özgün görüntü etiketi karşılaştırma hatası: 0; tüm 8189 görüntü kimliği mevcut.

## Eğitim sonuçları

| Deney | Durum | Epoch | En iyi top-1 | En iyi macro | Son epoch süresi | Tepe CUDA | En iyi model boyutu |
|---|---|---:|---:|---:|---:|---:|---:|
| efficient_f0 | Geçersiz: NaN kayıp | 1 | — | — | — | — | — |
| efficient_f0_v2 | Eski veri bölünmesi, model seçimi için kullanılmaz | 2/15 | 89.3130% | 87.7998% | 184.2 s | 90.9 MB | 16.83 MB |
| efficient_f0_v3 | Eski veri bölünmesi, model seçimi için kullanılmaz | 3/15 | 92.4779% | 91.3562% | 151.7 s | 88.5 MB | 16.83 MB |
| efficient_f0_v4 | Veri bölünmesi güncellendi; epoch tamamlanmadı | 0 | — | — | — | — | — |
| efficient_f0_v5 | Tamamlandı | 15/15 | 98.8701% | 98.7669% | 148.4 s | 458.2 MB | 16.83 MB |
| efficient_f1 | Tamamlandı | 15/15 | 98.2019% | 97.8167% | 122.4 s | 458.2 MB | 16.83 MB |
| efficient_f2 | Tamamlandı | 15/15 | 98.3016% | 97.9230% | 257.2 s | 458.1 MB | 16.83 MB |
| efficient_final_all_dev | Tamamlandı; tüm geliştirme verisi | 14/14 | — | — | 148.7 s | 458.1 MB | 16.83 MB |
| efficient_lr3e4_f0 | Erken durduruldu | 23/30 | 98.7086% | 98.5425% | 120.2 s | 458.2 MB | 16.83 MB |
| mobile_distill_b8_f0 | Tamamlandı | 12/12 | 98.2647% | 98.0913% | 239.4 s | 311.9 MB | 10.68 MB |
| mobile_f0 | Tamamlandı | 15/15 | 96.8119% | 96.4909% | 108.3 s | 140.7 MB | 10.68 MB |
| mobile_lr3e4_f0 | Tamamlandı | 30/30 | 98.1840% | 97.8836% | 71.6 s | 140.7 MB | 10.68 MB |
| mobile_lr3e4_f1 | Tamamlandı | 30/30 | 97.6297% | 97.1557% | 76.8 s | 141.8 MB | 10.68 MB |
| mobile_lr3e4_f2 | Tamamlandı | 30/30 | 97.8459% | 97.4989% | 77.1 s | 140.7 MB | 10.68 MB |
| teacher_f0 | Tamamlandı | 15/15 | 99.6772% | 99.6753% | 858.2 s | 773.2 MB | 111.84 MB |

Tablodaki doğruluklar yalnız geliştirme parçalarına aittir. İç kilitli test sonucu aşağıda ayrı verilmiştir; yarışma gizli testi ayrıca öğretmendedir.

## Tek seferlik iç kilitli test

- Model: `submission/model.pth`.
- Görüntü: 819; top-1: 98.7790%; macro: 98.8489%; kayıp: 0.2002.
- Bu 819 görüntü geliştirme veya son eğitime katılmadı. Bu sonuç yarışmanın gizli test puanı değildir.

## Kaydedilmiş ağırlık doğrulaması

| Deney | Doğrulama görüntüsü | Top-1 | Macro |
|---|---:|---:|---:|
| efficient_f0_v5 | 2478 | 98.8701% | 98.7669% |
| efficient_f1 | 2447 | 98.2019% | 97.8167% |
| efficient_f2 | 2414 | 98.3016% | 97.9230% |
| efficient_lr3e4_f0 | 2478 | 98.7086% | 98.5425% |
| mobile_distill_b8_f0 | 2478 | 98.2647% | 98.0913% |
| mobile_f0 | 2478 | 96.8523% | 96.5286% |
| mobile_lr3e4_f0 | 2478 | 98.1840% | 97.8836% |
| mobile_lr3e4_f1 | 2447 | 97.6297% | 97.1557% |
| mobile_lr3e4_f2 | 2414 | 97.8459% | 97.4989% |
| teacher_f0 | 2478 | 99.6772% | 99.6753% |

## Dışa aktarılan dosyalar

| Dosya | Gerçek boyut |
|---|---:|
| efficient_f0_v5_fp16.pth | 8.477 MB |
| efficient_f0_v5_fp32.pth | 16.839 MB |
| efficient_lr3e4_f0_fp16.pth | 8.478 MB |
| efficient_lr3e4_f0_fp32.pth | 16.840 MB |
| mobile_distill_b8_f0_fp16.pth | 5.393 MB |
| mobile_distill_b8_f0_fp32.pth | 10.691 MB |
| mobile_f0_fp16.pth | 5.390 MB |
| mobile_f0_fp32.pth | 10.687 MB |
| mobile_lr3e4_f0_fp16.pth | 5.391 MB |
| mobile_lr3e4_f0_fp32.pth | 10.689 MB |
| teacher_f0_fp16.pth | 55.958 MB |
| teacher_f0_fp32.pth | 111.848 MB |

## Teslim modeli

- Dosya: `submission/model.pth`; gerçek boyut: 8.471 MB.
- FP16 ağırlıklar CPU modeline FP32 yüklenir; bu dosya boyutunu küçültür, hesaplamayı kendiliğinden hızlandırmaz.

## Dışa aktarılan model doğruluğu

| Paket | Görüntü | Top-1 | Macro |
|---|---:|---:|---:|
| efficient_f0_v5_fp16_verified_cpu | 2478 | 98.8701% | 98.7669% |
| mobile_f0_fp16_verified_cpu | 2478 | 96.8523% | 96.5286% |
| mobile_f0_fp32_verified_cpu | 2478 | 96.8119% | 96.4909% |
| mobile_lr3e4_f0_fp16_verified_cpu | 2478 | 98.1840% | 97.8836% |

## Tahmin süresi

| Ölçüm | Cihaz | Tensör ort./medyan/p95 | Açılmış fotoğraf ort. | Disk dahil ort. | Isınma + tekrar |
|---|---|---:|---:|---:|---:|
| efficient_f0_v5_fp16_cpu_full | cpu | 26.85/26.87/31.49 ms | 28.48 ms | 30.21 ms | 50 + 5×1000 |
| efficient_f0_v5_fp32_cpu | cpu | 26.07/26.19/27.67 ms | 27.34 ms | 29.99 ms | 10 + 1×100 |
| efficient_f0_v5_fp32_cuda | cuda | 13.59/13.53/16.71 ms | 16.73 ms | 20.15 ms | 10 + 1×100 |
| efficient_final_fp16_cpu_full | cpu | 33.89/32.36/45.96 ms | 36.08 ms | 43.20 ms | 50 + 5×1000 |
| efficient_lr3e4_f0_fp32_cpu | cpu | 34.85/31.71/49.33 ms | 47.40 ms | 37.94 ms | 10 + 1×100 |
| efficient_lr3e4_f0_fp32_cuda | cuda | 13.58/12.55/18.91 ms | 19.39 ms | 21.40 ms | 10 + 1×100 |
| mobile_distill_b8_f0_fp32_cpu | cpu | 10.46/10.70/11.36 ms | 13.04 ms | 15.23 ms | 10 + 1×100 |
| mobile_distill_b8_f0_fp32_cuda | cuda | 6.37/6.32/6.54 ms | 10.72 ms | 13.33 ms | 10 + 1×100 |
| mobile_f0_fp32_cpu | cpu | 13.33/13.66/14.64 ms | 16.81 ms | 18.81 ms | 10 + 1×100 |
| mobile_f0_fp32_cuda | cuda | 6.70/6.65/7.15 ms | 12.65 ms | 14.11 ms | 10 + 1×100 |
| mobile_lr3e4_f0_fp16_cpu_full | cpu | 11.93/12.06/14.32 ms | 15.37 ms | 18.07 ms | 50 + 5×1000 |
| mobile_lr3e4_f0_fp32_cpu | cpu | 11.40/11.36/12.72 ms | 14.80 ms | 18.24 ms | 10 + 1×100 |
| mobile_lr3e4_f0_fp32_cuda | cuda | 6.54/6.48/6.93 ms | 11.76 ms | 13.68 ms | 10 + 1×100 |
| teacher_f0_fp32_cpu | cpu | 83.56/83.68/87.40 ms | 88.85 ms | 90.89 ms | 10 + 1×100 |
| teacher_f0_fp32_cuda | cuda | 12.78/12.46/14.25 ms | 17.37 ms | 23.89 ms | 10 + 1×100 |

Her satır tek bir görüntü ile ölçülmüştür. Tensör süresi görüntü okuma ve ön işlemeyi içermez.

## Yarışma kılavuzu v1

- Final puanı: %60 doğruluk + %25 CPU tahmin süresi puanı + %15 kayıtlı model boyutu puanı.
- CPU ortamı: aynı Intel Core i7 bilgisayar, 16 GB RAM, GPU yok. İşlemci modeli ve kesin zamanlama yöntemi henüz belirtilmedi.
- Önerilen hız ve boyut eşikleri: 50 ms/görüntü ve 35 MB. Bu değerler kesinleşmiş değildir.
- Bu rapordaki hızlar yerel bilgisayarda ölçülmüştür; yarışma CPU süresi veya yarışma puanı değildir.

## Ortam

- GPU: Quadro T2000 with Max-Q Design, 4096 MiB, 595.95; CUDA compute capability [7, 5].
- Python 3.12.14; PyTorch 2.13.0+cu132; torchvision 0.28.0+cu132; timm 1.0.25.
- CPU: Intel64 Family 6 Model 165 Stepping 2, GenuineIntel; PyTorch thread sayısı 6.
- GPU CUDA işlemi doğrulandı. Yarışma değerlendirmesi CPU üzerinde yapılacaktır.

## Kaynaklar

- [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Oxford kategorileri](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html)
- [PyTorch kurulum sürümleri](https://pytorch.org/get-started/previous-versions/)
- [Sınıf ID-ad eşleştirmesi](https://github.com/udacity/aipnd-project/blob/master/cat_to_name.json)
