# Volumetric Texture Generator - dependency check/install + launcher (Windows)
#
# Usage:
#   .\run.ps1                     # launch GUI
#   .\run.ps1 gui                 # launch GUI
#   .\run.ps1 cli --size 64 ...   # launch CLI (args passed through)
#   .\run.ps1 --gpu ...           # also check/install optional GPU deps (pyopencl, numpy)
#
# Zero-friction: use run.cmd (double-click or `run.cmd ...`), which invokes this
# with -ExecutionPolicy Bypass.
#
# Checks:
#   - a Python >= 3.10 (GUI additionally requires working tkinter; the python.org
#     installer bundles it under the optional "tcl/tk and IDLE" feature)
#   - optional GPU stack: pyopencl + numpy (only with --gpu)

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Info([string]$m) { Write-Host "[run.ps1] $m" }
function Warn([string]$m) { Write-Host "[run.ps1] WARNING: $m" -ForegroundColor Yellow }

# --- Parse (same semantics as run.sh: --gpu anywhere; gui/cli only while no
#     passthrough args collected yet) ---
$wantGpu = $false
$mode = 'gui'
$cliArgs = @()
foreach ($a in $Rest) {
    if ($a -eq '--gpu') { $wantGpu = $true; continue }
    if ($cliArgs.Count -eq 0 -and ($a -eq 'gui' -or $a -eq 'cli')) { $mode = $a; continue }
    $cliArgs += [string]$a
}

# Candidate = @{ e = executable; a = py-launcher version flag or '' }
$CANDIDATES = @(
    @{ e = 'py'; a = '-3.13' }, @{ e = 'py'; a = '-3.12' }, @{ e = 'py'; a = '-3.11' },
    @{ e = 'py'; a = '-3.10' }, @{ e = 'python'; a = '' }, @{ e = 'python3'; a = '' }
)

function Invoke-Py($cand, [string[]]$pyArgs) {
    $a = @()
    if ($cand.a) { $a += $cand.a }
    $a += $pyArgs
    & $cand.e @a
}

function Try-Py($cand, [string]$snippet) {
    if (-not (Get-Command $cand.e -ErrorAction SilentlyContinue)) { return $false }
    try { Invoke-Py $cand @('-c', $snippet) 2>$null | Out-Null } catch { return $false }
    return ($LASTEXITCODE -eq 0)
}

# --- 1) Pick interpreter ------------------------------------------------------
$versionSnippet = 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'
$tkSnippet = 'import tkinter, tkinter.ttk'

$py = $null
foreach ($c in $CANDIDATES) {
    if (-not (Try-Py $c $versionSnippet)) { continue }
    if ($mode -eq 'gui' -and -not (Try-Py $c $tkSnippet)) { continue }
    $py = $c
    break
}
if (-not $py) {
    if ($mode -eq 'gui') {
        Warn 'No Python >= 3.10 with tkinter found.'
        Warn 'Install Python from https://python.org and keep the optional "tcl/tk and IDLE" feature checked.'
    } else {
        Warn 'No Python >= 3.10 found. Install Python 3.10 or later from https://python.org.'
    }
    exit 1
}

$ver = ((Invoke-Py $py @('-c', 'import sys; print(sys.version.split()[0])')) -join '')
$label = if ($py.a) { "$($py.e) $($py.a)" } else { $py.e }
Info "Interpreter: $label ($ver)"

# --- 2) Optional GPU stack (pyopencl + numpy) ---------------------------------
if ($wantGpu) {
    if (Try-Py $py 'import pyopencl, numpy') {
        Info 'GPU deps already available'
    } else {
        Info 'Installing optional GPU deps (pyopencl, numpy)...'
        $ok = $true
        try {
            Invoke-Py $py @('-m', 'pip', 'install', '--user', '-q', 'pyopencl', 'numpy')
            $ok = ($LASTEXITCODE -eq 0)
        } catch { $ok = $false }
        if (-not $ok) {
            Info 'pip --user failed; retrying without --user...'
            try {
                Invoke-Py $py @('-m', 'pip', 'install', '-q', 'pyopencl', 'numpy')
                $ok = ($LASTEXITCODE -eq 0)
            } catch { $ok = $false }
        }
        if ($ok) { Info 'Installed via pip' }
        else { Warn 'Could not install pyopencl/numpy - continuing with CPU only.' }
    }
    if (Try-Py $py 'import pyopencl as cl; cl.get_platforms()') {
        Info 'OpenCL platform detected - GPU acceleration available'
    } else {
        Warn 'No OpenCL platform found - CPU only (install/update GPU drivers for OpenCL).'
    }
}

# --- 3) Launch (Windows GUI needs no display environment) ---------------------
if ($mode -eq 'gui') {
    Info 'Launching GUI...'
    Invoke-Py $py @('generate_volumetric_gui.py')
} else {
    Invoke-Py $py (@('generate_volumetric.py') + $cliArgs)
}
exit $LASTEXITCODE
