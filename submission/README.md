# Flower 102: kullanıma hazır CPU tahmin paketi

Bu klasör, bir fotoğraftaki çiçeği **102 kategoriden biri** olarak sınıflandırmak için gereken dosyaları içerir. Model EfficientNet-B0'dır. Eğitim görüntüleri veya GPU gerekmez; bağımlılıklar kurulduktan sonra tahminler internet bağlantısı olmadan CPU'da çalışır.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `model.pth` | FP16 biçiminde kaydedilmiş son model ağırlıkları ve ön işleme ayarları |
| `predict.py` | Tek fotoğraf, klasör ve Python API'si için tahmin kodu |
| `class_names.json` | 1–102 kategori numaralarının okunabilir adları |
| `requirements.txt` | Python bağımlılıklarının sabitlenmiş sürümleri |

Bu dosyaları birlikte tutun. `model.pth` yaklaşık **8,47 MB** yer kaplar. Model dosyasının içinde de sınıf adları vardır; dıştaki `class_names.json` ayrıca inceleme ve eşleme için sunulur.

## Kurulum

**Python 3.12** kullanın. Aşağıdaki komutları bu `submission` klasöründe çalıştırın. Windows PowerShell:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

macOS veya Linux'ta uyumlu PyTorch paketleri varsa:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Kurulumda paketler indirilir. **Tahmin aşaması** ağırlıkları indirmez ve GPU istemez. Bu proje Windows CPU ortamında doğrulandı; başka işletim sistemlerinde paketlerin bulunabilirliği değişebilir.

## Tek fotoğrafı sınıflandırma

Windows:

```powershell
./.venv/Scripts/python.exe predict.py --weights model.pth --image "C:/Photos/my flower.jpg" --device cpu
```

macOS veya Linux'ta aynı komutta `./.venv/bin/python` kullanın. Dosya yolunu kendi fotoğrafınızla değiştirin; boşluk içeren yolları tırnak içine alın. Örnek çıktı biçimi:

```json
{
  "class_id": "1",
  "class_name": "pink primrose",
  "confidence": 0.9916
}
```

`class_id` **metin olarak dönen 1–102 kategori numarasıdır**. `class_name` kategori adıdır. `confidence` 0–1 arası softmax skorudur; gerçek dünyadaki kesinlik garantisi değildir. Model, veri kümesi dışındaki görüntüler için de 102 sınıftan birini tahmin eder.

## Bir klasörü tarama

```powershell
./.venv/Scripts/python.exe predict.py --weights model.pth --folder "C:/Photos/flowers" --output predictions.csv --device cpu
```

Kod, klasörün **alt klasörlerini de** tarar ve `.jpg`, `.jpeg`, `.png`, `.webp` uzantılarını işler. CSV sütunları `path`, `class_id`, `class_name`, `confidence` olur. `--output` seçeneğini çıkarırsanız sonuçlar ekrana JSON listesi olarak yazılır. Çıktı klasörü önceden var olmalıdır.

## Python kodundan kullanım

```python
from predict import load_model

classifier = load_model("model.pth", device="cpu")
result = classifier.predict("my_flower.jpg")
print(result["class_id"], result["class_name"], result["confidence"])
```

`classifier.predict(path)` resmi açar, EXIF yönünü düzeltir, RGB'ye dönüştürür ve modelin kayıtlı ayarlarıyla yeniden boyutlandırıp normalize eder. Daha düşük düzeyli `classifier.logits(images)`, **önceden normalize edilmiş** `[batch, 3, height, width]` tensöründen `[batch, 102]` ham skor üretir. Özel bir ad listesi kullanmak isterseniz komut satırında `--names path/to/class_names.json` verebilirsiniz.

## Model ve sonuçların kapsamı

`model.pth` diskte FP16 ağırlık kullanır; `predict.py` bunları CPU'ya yüklerken FP32 ile hesap yapar. Model `pretrained=False` kurulduğu için tahmin sırasında dışarıdan ağırlık indirilmez.

Final model, temizlenmiş **7.339 geliştirme görüntüsünde 14 epok** eğitildi. Ayarlar dondurulduktan sonra bize verilen 819 görüntülük iç testte **809 doğru tahmin (%98,7790 top-1)** yaptı. Bu, yarışmanın gizli test sonucu değildir. Bu dizüstünde, dosya okuma dahil ortalama CPU süresi **43,196 ms/görüntü** olarak ölçüldü (50 ısınma, 5 × 1.000 tahmin). Başka CPU'larda süre değişebilir.

Veri denetimi, diğer modeller, üç doğrulama parçası ve ölçüm ayrıntıları üst klasördeki [deney raporunda](../EXPERIMENT_REPORT.md) bulunur. Hata alırsanız komutu bu klasörde çalıştırdığınızı, `model.pth` yolunu ve `requirements.txt` kurulumunu kontrol edin.
