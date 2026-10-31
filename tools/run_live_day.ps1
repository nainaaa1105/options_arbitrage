# Launcher used by the scheduled task. Runs one live paper day (read-only market data, simulated fills, no orders).
# Logs to data\paper\logs\<date>.log. Exit codes: 0 ok, 2 preflight failed (token/network/master), 3 ran but open exposure or reconciliation issue.
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [ValidateSet("forward", "replay")] [string]$Mode = "forward"
)
$ErrorActionPreference = "Stop"
Set-Location $Root
$logs = Join-Path $Root "data\paper\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs ((Get-Date -Format "yyyy-MM-dd") + ".log")
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$python = Join-Path $Root ".venv\Scripts\python.exe"
"=== start $(Get-Date -Format o) mode=$Mode ===" | Out-File -Append -Encoding utf8 $log
& $python -W ignore (Join-Path $Root "src\run_paper_day.py") live --mode $Mode --base (Join-Path $Root "data\paper\live") *>> $log
$code = $LASTEXITCODE
"=== end $(Get-Date -Format o) exit=$code ===" | Out-File -Append -Encoding utf8 $log
exit $code
