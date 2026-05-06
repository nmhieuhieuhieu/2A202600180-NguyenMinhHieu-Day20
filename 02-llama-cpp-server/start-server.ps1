# Launch llama-server (via llama-cpp-python) reading models/active.json.
# Windows PowerShell 7+.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$active = Get-Content -Raw -Path "models/active.json" | ConvertFrom-Json
$model = $active.primary_model

$hw = Get-Content -Raw -Path "hardware.json" | ConvertFrom-Json
$threads = if ($hw.cpu.cores_physical) { $hw.cpu.cores_physical } else { 4 }

$gpu     = if ($env:LAB_N_GPU_LAYERS) { $env:LAB_N_GPU_LAYERS } else { '99' }
$ctx     = if ($env:LAB_N_CTX) { $env:LAB_N_CTX } else { '2048' }

$pythonPath = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

Write-Host "==> Starting llama-server" -ForegroundColor Cyan
Write-Host "    model     : $model"
Write-Host "    threads   : $threads"
Write-Host "    gpu_layers: $gpu"
Write-Host "    ctx       : $ctx"
Write-Host "    listening : http://0.0.0.0:8080"
Write-Host ""

& $pythonPath -m llama_cpp.server `
    --model "$model" `
    --host 0.0.0.0 --port 8080 `
    --n_threads $threads `
    --n_gpu_layers $gpu `
    --n_ctx $ctx
