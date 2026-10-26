"""Build a report using only measurements saved by completed runs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    audit = json.loads(Path('artifacts/data/audit.json').read_text(encoding='utf8'))
    names = json.loads(Path('artifacts/data/name_verification.json').read_text(encoding='utf8'))
    source_labels = json.loads(Path('artifacts/data/source_label_verification.json').read_text(encoding='utf8'))
    environment = json.loads(Path('artifacts/environment.json').read_text(encoding='utf8'))
    lines = [
        '# Deney raporu', '',
        f'Oluşturulma: {datetime.now().astimezone().isoformat(timespec="seconds")}', '',
        '## Veri denetimi', '',
        f'- Görüntü: {audit["total"]}; sınıf: {audit["classes"]}.',
        f'- Train/valid/test: {audit["splits"]["train"]}/{audit["splits"]["valid"]}/{audit["splits"]["test"]}.',
        f'- Birebir kopya grubu: {audit["exact_duplicate_groups"]}; görsel olarak doğrulanan yakın kopya çifti: {audit["confirmed_near_pairs"]}.',
        f'- Kilitli testte yakın kopyası olduğu için geliştirmeden çıkarılan görüntü: {sum(len(g) - sum(r.startswith("test/") for r in g) for g in audit["test_dev_duplicate_groups"])}.',
        f'- Farklı etiketli aynı fotoğraf nedeniyle çıkarılan görüntü: {sum(map(len, audit["cross_label_groups"]))}.',
        f'- Geliştirme havuzundan çıkarılan birebir dosya kopyası: {len(audit["excluded_exact_duplicate_paths"])}.',
        f'- Üç doğrulama parçası: {audit["fold_counts"]}.',
        f'- Oxford sınıf adı doğrulama hatası: {len(names["mismatches"])}.',
        f'- Oxford özgün görüntü etiketi karşılaştırma hatası: {len(source_labels["mismatches"])}; tüm {source_labels["unique_image_ids"]} görüntü kimliği mevcut.', '',
        '## Eğitim sonuçları', '',
        '| Deney | Durum | Epoch | En iyi top-1 | En iyi macro | Son epoch süresi | Tepe CUDA | En iyi model boyutu |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for run in sorted(Path('artifacts/runs').iterdir()):
        if not run.is_dir():
            continue
        if (run / 'failure.json').exists():
            lines.append(f'| {run.name} | Geçersiz: NaN kayıp | 1 | — | — | — | — | — |')
            continue
        if (run / 'aborted.json').exists():
            lines.append(f'| {run.name} | Veri bölünmesi güncellendi; epoch tamamlanmadı | 0 | — | — | — | — | — |')
            continue
        history_path = run / 'history.jsonl'
        if not history_path.exists():
            continue
        history = [json.loads(line) for line in history_path.read_text(encoding='utf8').splitlines()]
        if not history:
            continue
        recent = history[-1]
        config = json.loads((run / 'config.json').read_text(encoding='utf8'))
        planned_epochs = config.get('distillation', {}).get('epochs', config['epochs'])
        status = ('Eski veri bölünmesi, model seçimi için kullanılmaz' if (run / 'superseded.json').exists()
                  else 'Erken durduruldu' if (run / 'early_stop.json').exists()
                  else 'Tamamlandı' if recent['epoch'] >= planned_epochs else 'Sürüyor / devam edilebilir')
        peak = recent.get('peak_cuda_mb')
        peak_text = f'{peak:.1f} MB' if peak is not None else '—'
        if config.get('fold') == 'all_development':
            final = run / 'final.pt'
            size_text = f'{final.stat().st_size / 1e6:.2f} MB' if final.exists() else '—'
            lines.append(f'| {run.name} | {status}; tüm geliştirme verisi | {recent["epoch"]}/{planned_epochs} | — | — | {recent["seconds"]:.1f} s | {peak_text} | {size_text} |')
            continue
        best = max(history, key=lambda x: x['top1'])
        size = (run / 'best.pt').stat().st_size / 1e6 if (run / 'best.pt').exists() else float('nan')
        lines.append(f'| {run.name} | {status} | {recent["epoch"]}/{planned_epochs} | {best["top1"]:.4%} | {best["macro"]:.4%} | {recent["seconds"]:.1f} s | {peak_text} | {size:.2f} MB |')
    locked_test = Path('artifacts/locked_test_final.json')
    if locked_test.exists():
        lines += ['', 'Tablodaki doğruluklar yalnız geliştirme parçalarına aittir. İç kilitli test sonucu aşağıda ayrı verilmiştir; yarışma gizli testi ayrıca öğretmendedir.', '']
    else:
        lines += ['', 'Bu doğruluklar yalnız geliştirme parçasına aittir. Verilen arşivdeki 819 görüntülük iç kilitli test kullanılmadı; yarışma gizli testi ayrıca öğretmendedir.', '']
    if locked_test.exists():
        result = json.loads(locked_test.read_text(encoding='utf8'))
        lines += ['## Tek seferlik iç kilitli test', '',
                  f'- Model: `{result["checkpoint"]}`.',
                  f'- Görüntü: {result["count"]}; top-1: {result["top1"]:.4%}; macro: {result["macro"]:.4%}; kayıp: {result["loss"]:.4f}.',
                  '- Bu 819 görüntü geliştirme veya son eğitime katılmadı. Bu sonuç yarışmanın gizli test puanı değildir.', '']
    verified = []
    for run in sorted(Path('artifacts/runs').iterdir()):
        path = run / 'verified_best.json'
        if path.exists():
            result = json.loads(path.read_text(encoding='utf8'))
            verified.append((run.name, result))
    if verified:
        lines += ['## Kaydedilmiş ağırlık doğrulaması', '',
                  '| Deney | Doğrulama görüntüsü | Top-1 | Macro |', '|---|---:|---:|---:|']
        for name, result in verified:
            lines.append(f'| {name} | {result["count"]} | {result["top1"]:.4%} | {result["macro"]:.4%} |')
        lines.append('')
    exports = sorted(Path('artifacts/exports').glob('*.pth')) if Path('artifacts/exports').exists() else []
    if exports:
        lines += ['## Dışa aktarılan dosyalar', '', '| Dosya | Gerçek boyut |', '|---|---:|']
        for export in exports:
            lines.append(f'| {export.name} | {export.stat().st_size / 1e6:.3f} MB |')
        lines.append('')
    submission_model = Path('submission/model.pth')
    if submission_model.exists():
        lines += ['## Teslim modeli', '',
                  f'- Dosya: `{submission_model.as_posix()}`; gerçek boyut: {submission_model.stat().st_size / 1e6:.3f} MB.',
                  '- FP16 ağırlıklar CPU modeline FP32 yüklenir; bu dosya boyutunu küçültür, hesaplamayı kendiliğinden hızlandırmaz.', '']
    package_checks = sorted(Path('artifacts/exports').glob('*_verified_*.json')) if Path('artifacts/exports').exists() else []
    if package_checks:
        lines += ['## Dışa aktarılan model doğruluğu', '',
                  '| Paket | Görüntü | Top-1 | Macro |', '|---|---:|---:|---:|']
        for path in package_checks:
            result = json.loads(path.read_text(encoding='utf8'))
            lines.append(f'| {path.stem} | {result["count"]} | {result["top1"]:.4%} | {result["macro"]:.4%} |')
        lines.append('')
    benchmarks = sorted(Path('artifacts/benchmarks').glob('*.json')) if Path('artifacts/benchmarks').exists() else []
    if benchmarks:
        lines += ['## Tahmin süresi', '',
                  '| Ölçüm | Cihaz | Tensör ort./medyan/p95 | Açılmış fotoğraf ort. | Disk dahil ort. | Isınma + tekrar |',
                  '|---|---|---:|---:|---:|---:|']
        for path in benchmarks:
            result = json.loads(path.read_text(encoding='utf8'))
            lat = result['latency']
            tensor = lat['tensor']
            lines.append(f'| {path.stem} | {result["device"]} | {tensor["mean_ms"]:.2f}/{tensor["median_ms"]:.2f}/{tensor["p95_ms"]:.2f} ms | '
                         f'{lat["opened"]["mean_ms"]:.2f} ms | {lat["disk"]["mean_ms"]:.2f} ms | '
                         f'{result["warmup"]} + {result["repeats"]}×{result["per_repeat"]} |')
        lines += ['', 'Her satır tek bir görüntü ile ölçülmüştür. Tensör süresi görüntü okuma ve ön işlemeyi içermez.', '']
    else:
        lines += ['Hız ölçümleri henüz tamamlanmadı.', '']
    lines += ['## Yarışma kılavuzu v1', '',
              '- Final puanı: %60 doğruluk + %25 CPU tahmin süresi puanı + %15 kayıtlı model boyutu puanı.',
              '- CPU ortamı: aynı Intel Core i7 bilgisayar, 16 GB RAM, GPU yok. İşlemci modeli ve kesin zamanlama yöntemi henüz belirtilmedi.',
              '- Önerilen hız ve boyut eşikleri: 50 ms/görüntü ve 35 MB. Bu değerler kesinleşmiş değildir.',
              '- Bu rapordaki hızlar yerel bilgisayarda ölçülmüştür; yarışma CPU süresi veya yarışma puanı değildir.', '',
              '## Ortam', '',
              f'- GPU: {environment["gpu_nvidia_smi"]}; CUDA compute capability {environment["gpu_capability"]}.',
              f'- Python {environment["python"]}; PyTorch {environment["torch"]}; torchvision {environment["torchvision"]}; timm {environment["timm"]}.',
              f'- CPU: {environment["processor"]}; PyTorch thread sayısı {environment["cpu_threads"]}.',
              '- GPU CUDA işlemi doğrulandı. Yarışma değerlendirmesi CPU üzerinde yapılacaktır.', '',
              '## Kaynaklar', '',
              '- [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)',
              '- [Oxford kategorileri](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html)',
              '- [PyTorch kurulum sürümleri](https://pytorch.org/get-started/previous-versions/)',
              '- [Sınıf ID-ad eşleştirmesi](https://github.com/udacity/aipnd-project/blob/master/cat_to_name.json)', '']
    Path('EXPERIMENT_REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    print('Wrote EXPERIMENT_REPORT.md')


if __name__ == '__main__':
    main()
