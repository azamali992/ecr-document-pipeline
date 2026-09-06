param([int]$ParentPid)
while ($true) {
    $count = (Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -like '*multiprocessing-fork*' -and $_.ParentProcessId -eq $ParentPid } |
        Measure-Object).Count
    if ($count -eq 0) { break }
    Start-Sleep -Seconds 20
}
Write-Output "IDLE"
