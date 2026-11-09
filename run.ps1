param(
    [ValidateSet('verify','plot','tests')][string]$Action = 'verify',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$simuRoot = $PSScriptRoot
if (-not $Python) { $Python = Join-Path $simuRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'No Python selected. Supply -Python with an existing interpreter, or prepare simu/.venv separately. Nothing is installed automatically.'
}
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$overrides = @{
    PYTHONNOUSERSITE='1'; PYTHONDONTWRITEBYTECODE='1'; PYTHONUTF8='1';
    OMP_NUM_THREADS='1'; MKL_NUM_THREADS='1'; OPENBLAS_NUM_THREADS='1';
    MPLCONFIGDIR=(Join-Path $simuRoot '.cache/matplotlib');
    TORCH_HOME=(Join-Path $simuRoot '.cache/torch');
    XDG_CACHE_HOME=(Join-Path $simuRoot '.cache'); PYTHONPATH=$null
}
$previous = @{}
foreach ($key in $overrides.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    [Environment]::SetEnvironmentVariable($key, $overrides[$key], 'Process')
}
Push-Location (Join-Path $simuRoot 'reproduction/xl_mimo_sim')
try {
    if ($Action -eq 'verify') {
        & $pythonPath (Join-Path $simuRoot 'tools/verify_bundle.py')
    } elseif ($Action -eq 'plot') {
        & $pythonPath (Join-Path $simuRoot 'tools/replot.py')
    } else {
        & $pythonPath -m unittest discover -s joint_power -t . -p 'test*.py'
    }
    if ($LASTEXITCODE -ne 0) { throw "Action $Action failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
