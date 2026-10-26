param(
    [string]$DataRoot = $(if ($env:FLOWER_DATA_ROOT) { $env:FLOWER_DATA_ROOT } else { 'flower_data' })
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$run = Join-Path $PSScriptRoot 'artifacts\runs\efficient_final_all_dev'
$final = Join-Path $run 'final.pt'
$history = Join-Path $run 'history.jsonl'
$submission = Join-Path $PSScriptRoot 'submission'
$model = Join-Path $submission 'model.pth'
$testResult = Join-Path $PSScriptRoot 'artifacts\locked_test_final.json'
$archive = Join-Path $PSScriptRoot 'Flower102_submission.zip'
$image = Join-Path $DataRoot 'train\1\image_06734.jpg'
$env:HF_HUB_OFFLINE = '1'

if (!(Test-Path -LiteralPath $final)) { throw 'Final training has not produced final.pt.' }
$epoch = (Get-Content -LiteralPath $history | Select-Object -Last 1 | ConvertFrom-Json).epoch
if ($epoch -ne 14) { throw "Final training is only at epoch $epoch/14." }
if (!(Test-Path -LiteralPath $image)) { throw "Smoke-test image not found: $image" }

& $python export_model.py --weights $final --output $model --precision fp16
if ($LASTEXITCODE -ne 0) { throw 'FP16 export failed.' }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'predict.py') -Destination $submission
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'requirements.txt') -Destination $submission
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'class_names.json') -Destination $submission

Push-Location -LiteralPath $submission
try {
    $prediction = (& $python predict.py --weights model.pth --image $image --device cpu | Out-String | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0) { throw 'Offline CPU prediction failed.' }
    if ([int]$prediction.class_id -lt 1 -or [int]$prediction.class_id -gt 102 -or
        $prediction.confidence -lt 0 -or $prediction.confidence -gt 1) {
        throw 'Invalid class ID or confidence from CPU prediction.'
    }
    Write-Output "Offline CPU smoke prediction: class $($prediction.class_id), confidence $($prediction.confidence)"
} finally {
    Pop-Location
}

if (!(Test-Path -LiteralPath $testResult)) {
    & $python evaluate_checkpoint.py --weights $model --data $DataRoot --split test --unlock-test --device cpu --batch 16 --output $testResult
    if ($LASTEXITCODE -ne 0) { throw 'One-time internal locked test failed.' }
} else {
    Write-Output 'Internal locked test result already exists; skipping repeat.'
}

& $python make_report.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }

Compress-Archive -Path (Join-Path $submission '*') -DestinationPath $archive -Force
$entries = [System.IO.Compression.ZipFile]::OpenRead($archive)
try {
    $names = @($entries.Entries | ForEach-Object FullName)
    foreach ($required in @('model.pth', 'predict.py', 'requirements.txt', 'class_names.json', 'README.md')) {
        if ($required -notin $names) { throw "Archive is missing $required" }
    }
} finally {
    $entries.Dispose()
}
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
Write-Output "Package: $archive"
Write-Output "Archive size: $([math]::Round((Get-Item -LiteralPath $archive).Length / 1MB, 2)) MiB"
Write-Output "SHA256: $hash"
