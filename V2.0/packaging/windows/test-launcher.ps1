[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TestRoot,
    [string]$CandidateRoot,
    [switch]$ExperienceOnly
)
$ErrorActionPreference = 'Stop'
if ($CandidateRoot) {
    if ((Test-Path -LiteralPath $TestRoot) -and @(Get-ChildItem -LiteralPath $TestRoot -Force).Count) {
        throw 'TestRoot must be empty when preparing a fixture.'
    }
    $candidate = (Resolve-Path -LiteralPath $CandidateRoot).Path
    New-Item -ItemType Directory -Force -Path (Join-Path $TestRoot 'bin') | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'bin') -File | Copy-Item -Destination (Join-Path $TestRoot 'bin')
    foreach ($version in @('2.2.0','2.2.1')) {
        $versionRoot = Join-Path $TestRoot "versions\$version"
        New-Item -ItemType Directory -Force -Path (Join-Path $versionRoot 'bin') | Out-Null
        foreach ($directory in @('runtime','V2.0')) {
            New-Item -ItemType Junction -Path (Join-Path $versionRoot $directory) -Target (Join-Path $candidate $directory) | Out-Null
        }
        foreach ($script in @('start-labviz-portable.ps1','process-job.ps1')) {
            Copy-Item -LiteralPath (Join-Path $PSScriptRoot "bin\$script") -Destination (Join-Path $versionRoot 'bin')
        }
    }
}
$root = (Resolve-Path -LiteralPath $TestRoot).Path
$oldLocal = $env:LOCALAPPDATA
$env:LOCALAPPDATA = Join-Path $root '测试 用户'
$launcher = Join-Path $root 'bin\start-labviz-installed.ps1'
$shell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$logRoot = Join-Path $env:LOCALAPPDATA 'LabViz\logs'
$process = $null
$blocker = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 3355)
function Run-Check {
    & $shell -NoProfile -ExecutionPolicy Bypass -File $launcher -WebPort 3355 -ApiPort 8355 -SkipOpenBrowser -HealthCheckOnly
    if ($LASTEXITCODE -ne 0) { throw 'Health probe failed' }
}
try {
    $blocker.Start()
    [IO.File]::WriteAllText((Join-Path $root 'pending-version.txt'), '2.2.0')
    Run-Check
    if ((Get-Content -LiteralPath (Join-Path $root 'current-version.txt') -Raw).Trim() -ne '2.2.0') { throw 'Activation failed' }
    $startArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $launcher), '-WebPort', '3355', '-ApiPort', '8355', '-SkipOpenBrowser')
    $process = Start-Process -FilePath $shell -ArgumentList $startArgs -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Path -LiteralPath (Join-Path $logRoot 'running.json'))) {
        if ($process.HasExited -or (Get-Date) -gt $deadline) { throw 'No ready state' }
        Start-Sleep -Milliseconds 300
    }
    $state = Get-Content -LiteralPath (Join-Path $logRoot 'running.json') -Raw | ConvertFrom-Json
    if ($state.webUrl -eq 'http://127.0.0.1:3355') { throw 'Occupied port was reused' }
    if ((Invoke-WebRequest -UseBasicParsing $state.webUrl).StatusCode -ne 200) { throw 'Web unavailable' }
    Run-Check # A second process must reuse the existing instance without starting another group.
    if ((Get-Content -LiteralPath (Join-Path $logRoot 'running.json') -Raw | ConvertFrom-Json).launcherPid -ne $state.launcherPid) { throw 'Duplicate launch changed owner' }
    [IO.File]::WriteAllText((Join-Path $logRoot 'stop.request'), 'stop')
    if (-not $process.WaitForExit(15000)) { throw 'Graceful stop timed out' }
    if (Test-Path -LiteralPath (Join-Path $logRoot 'running.json')) { throw 'Ready state leaked' }
    Write-Host 'PASS occupied port, duplicate start, graceful stop'
    if ($ExperienceOnly) {
        $process = Start-Process -FilePath $shell -ArgumentList $startArgs -WindowStyle Hidden -PassThru
        $deadline = (Get-Date).AddSeconds(120)
        while (-not (Test-Path -LiteralPath (Join-Path $logRoot 'running.json'))) {
            if ($process.HasExited -or (Get-Date) -gt $deadline) { throw 'Restart failed' }
            Start-Sleep -Milliseconds 300
        }
        $state = Get-Content -LiteralPath (Join-Path $logRoot 'running.json') -Raw | ConvertFrom-Json
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit(10000) | Out-Null
        Start-Sleep -Seconds 2
        if (Get-NetTCPConnection -State Listen -LocalPort $state.apiPort -ErrorAction SilentlyContinue) { throw 'API leaked after launcher crash' }
        Write-Host 'PASS forced launcher termination cleans children'
        Run-Check
        Write-Host 'PASS restart after abandoned mutex'
        return
    }

    $data = Join-Path $env:LOCALAPPDATA 'LabViz\data'
    [IO.File]::WriteAllText((Join-Path $data 'synthetic-marker.txt'), 'before-upgrade')
    [IO.File]::WriteAllText((Join-Path $root 'pending-version.txt'), '2.2.1')
    Run-Check
    if ((Get-Content -LiteralPath (Join-Path $root 'current-version.txt') -Raw).Trim() -ne '2.2.1') { throw 'Upgrade failed' }
    [IO.File]::WriteAllText((Join-Path $data 'synthetic-marker.txt'), 'after-upgrade')
    [IO.File]::WriteAllText((Join-Path $root 'pending-version.txt'), '2.2.0')
    Run-Check
    if ((Get-Content -LiteralPath (Join-Path $data 'synthetic-marker.txt') -Raw) -ne 'before-upgrade') { throw 'Rollback did not restore data' }
    Write-Host 'PASS upgrade and data rollback'

    # Fault injection only changes the disposable candidate launcher, never source or user data.
    $faulty = Join-Path $root 'versions\2.2.1\bin\start-labviz-portable.ps1'
    $original = Get-Content -LiteralPath $faulty -Raw
    try {
        [IO.File]::WriteAllText($faulty, "param([int]`$WebPort,[int]`$ApiPort,[switch]`$SkipOpenBrowser,[switch]`$HealthCheckOnly)`n[IO.File]::WriteAllText((Join-Path `$env:LOCALAPPDATA 'LabViz\data\synthetic-marker.txt'),'failed-upgrade')`nthrow 'Injected startup failure'", [Text.Encoding]::UTF8)
        [IO.File]::WriteAllText((Join-Path $root 'pending-version.txt'), '2.2.1')
        Run-Check
        if ((Get-Content -LiteralPath (Join-Path $root 'current-version.txt') -Raw).Trim() -ne '2.2.0') { throw 'Failed upgrade changed active version' }
        if ((Get-Content -LiteralPath (Join-Path $data 'synthetic-marker.txt') -Raw) -ne 'before-upgrade') { throw 'Failed upgrade damaged data' }
    } finally { [IO.File]::WriteAllText($faulty, $original, [Text.Encoding]::UTF8) }
    Write-Host 'PASS failed-upgrade recovery'

    # Kill the disposable launcher after its transaction journal is written.
    # The next start must restore the snapshot before serving the old version.
    $interrupted = Join-Path $root 'versions\2.2.1\bin\start-labviz-portable.ps1'
    $original = Get-Content -LiteralPath $interrupted -Raw
    try {
        [IO.File]::WriteAllText($interrupted, "param([int]`$WebPort,[int]`$ApiPort,[switch]`$SkipOpenBrowser,[switch]`$HealthCheckOnly)`nStop-Process -Id `$PID -Force", [Text.Encoding]::UTF8)
        [IO.File]::WriteAllText((Join-Path $root 'pending-version.txt'), '2.2.1')
        $crashArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $launcher), '-WebPort', '3355', '-ApiPort', '8355', '-SkipOpenBrowser', '-HealthCheckOnly')
        $crashed = Start-Process -FilePath $shell -ArgumentList $crashArgs -WindowStyle Hidden -PassThru
        if (-not $crashed.WaitForExit(15000)) { Stop-Process -Id $crashed.Id -Force }
        Start-Sleep -Seconds 1
        if (-not (Test-Path -LiteralPath (Join-Path $root 'upgrade.json'))) { throw 'Interrupted upgrade did not leave a journal' }
        if ((Get-Content -LiteralPath (Join-Path $root 'current-version.txt') -Raw).Trim() -ne '2.2.0') { throw 'Interrupted upgrade changed active version' }
        Run-Check
        if (Test-Path -LiteralPath (Join-Path $root 'upgrade.json')) { throw 'Interrupted upgrade journal was not recovered' }
        if ((Get-Content -LiteralPath (Join-Path $root 'current-version.txt') -Raw).Trim() -ne '2.2.0') { throw 'Interrupted recovery changed active version' }
        if ((Get-Content -LiteralPath (Join-Path $data 'synthetic-marker.txt') -Raw) -ne 'before-upgrade') { throw 'Interrupted recovery damaged data' }
    } finally { [IO.File]::WriteAllText($interrupted, $original, [Text.Encoding]::UTF8) }
    Write-Host 'PASS interrupted-upgrade recovery'

    # End-to-end compatibility fixture: a stopped V2.1.1-style .labviz tree is
    # snapshotted, imported through the installed launcher, and then opened by
    # the real API. The fixture is disposable and never leaves the test root.
    $python = Join-Path $root 'versions\2.2.0\runtime\python\python.exe'
    $seedCode = 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute("CREATE TABLE IF NOT EXISTS migration_sentinel(value TEXT NOT NULL)"); c.execute("DELETE FROM migration_sentinel"); c.execute("INSERT INTO migration_sentinel(value) VALUES (?)", ("v2.1.1-guest-data",)); c.execute("INSERT OR REPLACE INTO projects(id,title,source_json,storage_mode,guest_token_digest,updated_at) VALUES (?,?,?,?,?,?)", ("legacy-guest-project","V2.1.1 guest project","{}","inline","guest-token-digest","2026-01-01T00:00:00Z")); c.execute("INSERT OR REPLACE INTO auth_sessions(token_digest,user_id,email,expires_at,created_at) VALUES (?,?,?,?,?)", ("old-session","old-user","old@example.test","2099-01-01T00:00:00Z","2026-01-01T00:00:00Z")); c.commit(); c.execute("PRAGMA wal_checkpoint(TRUNCATE)"); c.close()'
    & $python -c $seedCode (Join-Path $data 'labviz-v2.db')
    if ($LASTEXITCODE -ne 0) { throw 'Could not seed the V2.1.1 compatibility fixture' }
    $legacyObject = Join-Path $data 'objects\legacy-object.bin'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $legacyObject) | Out-Null
    [IO.File]::WriteAllBytes($legacyObject, [byte[]](0, 17, 34, 255))
    $source = Join-Path $root '旧版 V2.1.1 数据'
    & $python (Join-Path $root 'bin\local-data.py') snapshot $data $source
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the V2.1.1 compatibility snapshot' }
    $sourceDb = Join-Path $source 'labviz-v2.db'
    $sourceObject = Join-Path $source 'objects\legacy-object.bin'
    $sourceDbHash = (Get-FileHash -LiteralPath $sourceDb -Algorithm SHA256).Hash
    $sourceObjectHash = (Get-FileHash -LiteralPath $sourceObject -Algorithm SHA256).Hash
    $heldData = Join-Path $env:LOCALAPPDATA 'LabViz\data.before-import'
    Move-Item -LiteralPath $data -Destination $heldData
    New-Item -ItemType Directory -Force -Path $data | Out-Null
    & $shell -NoProfile -ExecutionPolicy Bypass -File $launcher -ImportData $source
    if ($LASTEXITCODE -ne 0) { throw 'Installed launcher import failed' }
    $readCode = 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print("|".join((c.execute("SELECT value FROM migration_sentinel").fetchone()[0], c.execute("SELECT guest_token_digest FROM projects WHERE id=?", ("legacy-guest-project",)).fetchone()[0], str(c.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0])))); c.close()'
    $sentinel = (& $python -c $readCode (Join-Path $data 'labviz-v2.db') | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $sentinel -ne 'v2.1.1-guest-data|guest-token-digest|0') { throw 'Imported database content or session policy was not preserved' }
    $sourceReadCode = 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0]); c.close()'
    $sourceSessions = (& $python -c $sourceReadCode $sourceDb | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceSessions -ne '1') { throw 'Import changed the source browser session data' }
    if ((Get-FileHash -LiteralPath $sourceDb -Algorithm SHA256).Hash -ne $sourceDbHash) { throw 'Import changed the source database' }
    if ((Get-FileHash -LiteralPath $sourceObject -Algorithm SHA256).Hash -ne $sourceObjectHash) { throw 'Import changed the source object' }
    if ((Get-FileHash -LiteralPath (Join-Path $data 'objects\legacy-object.bin') -Algorithm SHA256).Hash -ne $sourceObjectHash) { throw 'Imported object content was not preserved' }
    Run-Check
    Write-Host 'PASS V2.1.1 compatibility import, source retention, and API reopen'
} finally {
    if ($process -and -not $process.HasExited) {
        [IO.File]::WriteAllText((Join-Path $logRoot 'stop.request'), 'stop')
        $process.WaitForExit(15000) | Out-Null
    }
    $blocker.Stop()
    $env:LOCALAPPDATA = $oldLocal
}
