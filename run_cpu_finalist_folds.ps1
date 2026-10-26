param(
    [int]$WaitForEfficientF2Pid = 8444,
    [string]$DataRoot = $(if ($env:FLOWER_DATA_ROOT) { $env:FLOWER_DATA_ROOT } else { 'flower_data' })
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot 'artifacts\hf_cache'
$env:HF_HUB_OFFLINE = '1'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

while (Get-Process -Id $WaitForEfficientF2Pid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}

$efficientHistory = Join-Path $PSScriptRoot 'artifacts\runs\efficient_f2\history.jsonl'
if (!(Test-Path -LiteralPath $efficientHistory)) {
    throw 'efficient_f2 has no completed epoch; inspect its process before continuing.'
}
$efficientLast = (Get-Content -LiteralPath $efficientHistory | Select-Object -Last 1 | ConvertFrom-Json).epoch
if ($efficientLast -lt 15) {
    throw "efficient_f2 stopped at epoch $efficientLast/15; resume it before starting the next GPU run."
}

$runs = @(
    @{ Name = 'efficient_f2'; Model = 'efficient'; Fold = 2; Epochs = 15; BackboneLR = '0.0001'; Patience = 0 },
    @{ Name = 'mobile_lr3e4_f1'; Model = 'mobile'; Fold = 1; Epochs = 30; BackboneLR = '0.0003'; Patience = 8 },
    @{ Name = 'mobile_lr3e4_f2'; Model = 'mobile'; Fold = 2; Epochs = 30; BackboneLR = '0.0003'; Patience = 8 }
)

foreach ($item in $runs) {
    $run = Join-Path $PSScriptRoot ('artifacts\runs\' + $item.Name)
    if ($item.Name -ne 'efficient_f2') {
        $history = Join-Path $run 'history.jsonl'
        $lastEpoch = if (Test-Path -LiteralPath $history) {
            (Get-Content -LiteralPath $history | Select-Object -Last 1 | ConvertFrom-Json).epoch
        } else { 0 }
        $earlyStop = Join-Path $run 'early_stop.json'
        if ($lastEpoch -lt $item.Epochs -and !(Test-Path -LiteralPath $earlyStop)) {
            $trainArgs = @('train.py', '--data', $DataRoot, '--run', $run, '--model', $item.Model,
                           '--fold', [string]$item.Fold, '--batch', '8', '--workers', '2',
                           '--epochs', [string]$item.Epochs, '--backbone-lr', $item.BackboneLR,
                           '--patience', [string]$item.Patience)
            & $python @trainArgs
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $($item.Name)" }
        }
    }
    $best = Join-Path $run 'best.pt'
    if (!(Test-Path -LiteralPath $best)) { throw "Missing best.pt: $($item.Name)" }
    $verified = Join-Path $run 'verified_best.json'
    if (!(Test-Path -LiteralPath $verified) -or (Get-Item -LiteralPath $verified).LastWriteTime -lt (Get-Item -LiteralPath $best).LastWriteTime) {
        & $python evaluate_checkpoint.py --weights $best --data $DataRoot --split fold --device cpu --batch 16 --output $verified
        if ($LASTEXITCODE -ne 0) { throw "CPU verification failed: $($item.Name)" }
        & $python make_report.py
        if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
    }
}

$image = Join-Path $DataRoot 'train\1\image_06734.jpg'
foreach ($name in @('efficient_f0_v5', 'mobile_lr3e4_f0')) {
    $package = Join-Path $PSScriptRoot ('artifacts\exports\' + $name + '_fp16.pth')
    if (!(Test-Path -LiteralPath $package)) { throw "Missing FP16 package: $name" }
    $verified = Join-Path $PSScriptRoot ('artifacts\exports\' + $name + '_fp16_verified_cpu.json')
    if (!(Test-Path -LiteralPath $verified)) {
        & $python evaluate_checkpoint.py --weights $package --data $DataRoot --split fold --device cpu --batch 16 --output $verified
        if ($LASTEXITCODE -ne 0) { throw "FP16 package verification failed: $name" }
    }
    $benchmark = Join-Path $PSScriptRoot ('artifacts\benchmarks\' + $name + '_fp16_cpu_full.json')
    if (!(Test-Path -LiteralPath $benchmark)) {
        & $python benchmark.py --weights $package --image $image --device cpu --warmup 50 --repeats 5 --per-repeat 1000 --output $benchmark
        if ($LASTEXITCODE -ne 0) { throw "Full CPU benchmark failed: $name" }
        & $python make_report.py
        if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
    }
}
