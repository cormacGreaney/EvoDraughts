# Quick script to check if evolution is still running
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, @{Name="CPU_Seconds";Expression={[math]::Round($_.CPU,2)}}, @{Name="CPU_Minutes";Expression={[math]::Round($_.CPU/60,2)}}, @{Name="Memory_MB";Expression={[math]::Round($_.WorkingSet/1MB,2)}}, StartTime | Format-Table -AutoSize
