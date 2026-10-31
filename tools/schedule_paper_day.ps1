# Registers (or removes) the weekday scheduled task for the live paper day.
#   Create DISABLED (default, safe):   powershell -File tools\schedule_paper_day.ps1
#   Enable after you have reviewed the dry run:   powershell -File tools\schedule_paper_day.ps1 -Enable
#   Remove completely:                 powershell -File tools\schedule_paper_day.ps1 -Remove
# The task starts at 09:00 local time Monday-Friday, may wake the machine, runs at most 7 hours and never places orders.
# It runs as the current user, so it can read .env. It will NOT run while the machine is shut down or the user is logged out
# (the task is registered for interactive logon); check data\paper\logs each evening.
param([switch]$Enable, [switch]$Remove, [string]$TaskName = "NiftyPaperDay", [string]$At = "09:00")
$root = Split-Path -Parent $PSScriptRoot
if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed $TaskName"; exit 0
}
$launcher = Join-Path $root "tools\run_live_day.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 7) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
if ($Enable) { Enable-ScheduledTask -TaskName $TaskName | Out-Null; Write-Output "$TaskName registered and ENABLED ($At, Mon-Fri)" }
else { Disable-ScheduledTask -TaskName $TaskName | Out-Null; Write-Output "$TaskName registered DISABLED. Re-run with -Enable to activate." }
