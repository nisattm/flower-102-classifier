param(
    [string]$DataRoot = $(if ($env:FLOWER_DATA_ROOT) { $env:FLOWER_DATA_ROOT } else { 'flower_data' }),
    [switch]$FullBenchmark
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot 'artifacts\hf_cache'
$env:HF_HUB_OFFLINE = '1'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$image = Join-Path $DataRoot 'train\1\image_06734.jpg'
$exportDir = Join-Path $PSScriptRoot 'artifacts\exports'
$benchmarkDir = Join-Path $PSScriptRoot 'artifacts\benchmarks'
New-Item -ItemType Directory -Force -Path $exportDir, $benchmarkDir | Out-Null

if ($FullBenchmark) {
    $warmup = 50
    $repeats = 5
    $perRepeat = 1000
} else {
    $warmup = 10
    $repeats = 1
    $perRepeat = 100
}

foreach ($name in @('efficient_f0_v5', 'mobile_f0', 'teacher_f0')) {
    $run = Join-Path $PSScriptRoot "artifacts\runs\$name"
    $best = Join-Path $run 'best.pt'
    $history = Join-Path $run 'history.jsonl'
    $config = Join-Path $run 'config.json'
    if (!(Test-Path $best) -or !(Test-Path $history) -or !(Test-Path $config)) {
        throw "Missing baseline files for $name"
    }
    $lastEpoch = (Get-Content -LiteralPath $history | Select-Object -Last 1 | ConvertFrom-Json).epoch
    $totalEpochs = (Get-Content -LiteralPath $config -Raw | ConvertFrom-Json).epochs
    if ($lastEpoch -lt $totalEpochs) {
        throw "$name completed $lastEpoch/$totalEpochs epochs. Resume training first."
    }
    & $python evaluate_checkpoint.py --weights $best --data $DataRoot --split fold --device cuda --batch 16 --output (Join-Path $run 'verified_best.json')
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $name" }
    foreach ($precision in @('fp32', 'fp16')) {
        $output = Join-Path $exportDir "${name}_${precision}.pth"
        & $python export_model.py --weights $best --output $output --precision $precision
        if ($LASTEXITCODE -ne 0) { throw "Export failed: $name $precision" }
    }
    foreach ($device in @('cpu', 'cuda')) {
        $output = Join-Path $benchmarkDir "${name}_fp32_${device}.json"
        & $python benchmark.py --weights (Join-Path $exportDir "${name}_fp32.pth") --image $image --device $device --warmup $warmup --repeats $repeats --per-repeat $perRepeat --output $output
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $name $device" }
    }
    & $python make_report.py
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
}
