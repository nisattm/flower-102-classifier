# Flower 102: CPU üzerinde çiçek sınıflandırma

Bu proje, Oxford Flowers 102 veri kümesindeki **102 çiçek türünü** fotoğraftan tahmin eder. Teslime seçilen model EfficientNet-B0'dır. `submission/` klasöründeki model ve tahmin kodu **GPU olmadan, CPU üzerinde ve çevrimdışı** çalışır. Modeli denemek için eğitim verisini indirmek gerekmez.

## Depoda neler var?

| Konum | İçerik |
|---|---|
| [`submission/`](submission/) | Kullanıma hazır model, tahmin kodu, 102 sınıf adı ve bağımlılıklar |
| [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) | Deneylerin gerçek doğruluk, hız ve boyut sonuçları |
| [`FINAL_MODEL_DECISION.md`](FINAL_MODEL_DECISION.md) | Final modelin hangi ölçütlerle seçildiği |
| [`CPU_TARGET.md`](CPU_TARGET.md) | Yarışmanın CPU ve puanlama koşulları |
| `train.py`, `train_final.py` | Geliştirme ve son eğitim kodu |
| `artifacts/data/folds.jsonl` | Model seçimi için dondurulmuş üç doğrulama parçası |

Orijinal çiçek fotoğrafları, büyük eğitim checkpoint'leri ve önceden indirilmiş ağırlıklar GitHub'a eklenmedi. **Çalıştırılabilir son model** [`submission/model.pth`](submission/model.pth) içindedir.

## Kendi fotoğrafınızda deneyin

Bu komutlar Windows PowerShell'de, deponun ana klasöründen çalışır. Python 3.12 kurulu olmalıdır. Önce depoyu indirin ve bağımlılıkları yükleyin:

```powershell
git clone https://github.com/nisattm/flower-102-classifier.git
cd flower-102-classifier
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r submission/requirements.txt
```

Ardından fotoğrafın yolunu kendi dosyanızla değiştirin. Yol boşluk içeriyorsa tırnak işaretlerini koruyun:

```powershell
./.venv/Scripts/python.exe submission/predict.py --weights submission/model.pth --image "C:/Photos/my flower.jpg" --device cpu
```

Çıktı JSON biçimindedir; örnek çıktı:

```json
{
  "class_id": "1",
  "class_name": "pink primrose",
  "confidence": 0.9916
}
```

`class_id`, veri kümesindeki **1–102 arasındaki özgün kategori numarasıdır**; sıfırdan başlamaz. `class_name` okunabilir tür adıdır. `confidence`, seçilen sınıfın softmax skorudur; doğruluk garantisi olarak yorumlanmamalıdır. Fotoğrafta veri kümesi dışı bir çiçek olsa bile model bu 102 sınıftan birini seçer.

Bir klasördeki JPG, JPEG, PNG ve WebP dosyalarını, alt klasörleriyle birlikte tarayıp CSV almak için:

```powershell
./.venv/Scripts/python.exe submission/predict.py --weights submission/model.pth --folder "C:/Photos/flowers" --output predictions.csv --device cpu
```

CSV sütunları `path`, `class_id`, `class_name` ve `confidence` olur. `--output` verilmezse sonuçlar ekrana JSON olarak yazılır. İşletim sisteminize göre ortam kurma ve Python içinden kullanım ayrıntıları [teslim paketi README dosyasında](submission/README.md) bulunur.

## Ölçülmüş sonuçlar

| Ölçüm | Sonuç |
|---|---:|
| Üç geliştirme parçası toplam top-1 doğruluk | **7.226 / 7.339 = %98,4603** |
| Bir kez açılan iç testte top-1 doğruluk | **809 / 819 = %98,7790** |
| FP16 model dosyası | **8,471 MB** |
| Son paketin bu dizüstünde ortalama CPU süresi; dosya okuma dahil | **43,196 ms/görüntü** |

CPU ölçümünde 50 ısınma ve 5 × 1.000 tahmin kullanıldı. Seçim sırasında aynı bilgisayarda ayrı bir koşuda ölçülen fold modelinin ortalaması 30,210 ms idi; süreler koşullara göre değişebiliyor. Yarışma bilgisayarındaki gerçek süreyi ve **yarışmanın gizli test doğruluğunu henüz bilmiyoruz**. Yukarıdaki 819 görüntü, bize verilen ayrı iç testtir. Deneylerin tamamı ve ölçüm yöntemi [deney raporunda](EXPERIMENT_REPORT.md) kayıtlıdır.

## Veri ve eğitimin özeti

Hocanın verdiği arşivde 8.189 görüntü vardı: 6.552 `train`, 818 `valid`, 819 `test`. Geliştirme sırasında `train + valid` verisinden üç sabit doğrulama parçası oluşturuldu; doğrulanmış kopyalar ve iç testle örtüşen yakın görüntüler temizlendi. Son EfficientNet-B0, kalan **7.339 geliştirme görüntüsünün tamamında 14 epok** eğitildi. 819 iç test görüntüsü eğitime katılmadı.

Eğitimleri tekrar yapmak için hocanın özgün `flower_data` arşivi ve ön eğitimli ağırlıklar gerekir; bunlar depoda yoktur. Son eğitim komutu:

```powershell
./.venv/Scripts/python.exe train_final.py --data "C:/path/to/flower_data" --run artifacts/runs/efficient_final_all_dev --model efficient --batch 8 --workers 2 --epochs 14 --backbone-lr 0.0001
```

Eğitim her tamamlanan epoktan sonra aynı koşu klasöründeki `last.pt` ile devam edebilir. Yardımcı betikler `FLOWER_DATA_ROOT` ortam değişkenini okuyabilir; tanımlı değilse proje içindeki `flower_data` klasörünü bekler. Başlangıç planı [`FLOWER_PLAN.txt`](FLOWER_PLAN.txt) içindedir; yarışma belgesi sonradan netleştiği için nihai seçim [karar dosyasında](FINAL_MODEL_DECISION.md) açıklanır.

## Sonraki sürümler

Modelin doğruluğu veya CPU hızı daha sonra geliştirilebilir. Yeni fikirleri aynı geliştirme parçalarında veya gerçekten yeni bir doğrulama kümesinde karşılaştırıp ayrı bir model sürümü yayımlamak gerekir. Bu sürüm seçildikten sonra 819 görüntülük iç test **bir kez** kullanıldı; sonraki ayarları o test sonucuna göre tekrar tekrar seçmemek gerekir.
