param(
    [string]$DataRoot = $(if ($env:FLOWER_DATA_ROOT) { $env:FLOWER_DATA_ROOT } else { 'flower_data' })
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot 'artifacts\hf_cache'
$env:HF_HUB_OFFLINE = '1'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

foreach ($item in @(@{ Name = 'mobile'; Batch = 8 }, @{ Name = 'teacher'; Batch = 2 })) {
    $run = Join-Path $PSScriptRoot ('artifacts\runs\' + $item.Name + '_f0')
    & $python train.py --data $DataRoot --run $run --model $item.Name --fold 0 --batch $item.Batch --workers 2
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $($item.Name); inspect $run and rerun this script to resume."
    }
    & $python make_report.py
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
}
