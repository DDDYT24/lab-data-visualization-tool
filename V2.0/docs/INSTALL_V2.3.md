# Install and manage the V2.3 Windows RC

V2.3.0-rc.1 is an unsigned preview for Windows 10 or later on compatible 64-bit systems. Use synthetic data or copies while evaluating it. Actual clean-machine, physical offline, screen-reader, phone, and researcher-pilot acceptance remain open.

## Install and start

Download the installer and `SHA256SUMS.txt` from the V2.3 RC release. Compare the installer hash before running it:

```powershell
Get-FileHash .\LabViz-V23-2.3.0-rc.1-Setup.exe -Algorithm SHA256
```

The RC has no code signature. Keep Windows security controls enabled. If Windows or security software blocks it, retain the message and report it; do not disable protection to finish a test.

The installer includes Python, Node.js, the website, scientific dependencies, fonts, and synthetic examples. It does not install dependencies when the application starts. Start **LabViz V2.3** from the Start menu. Use **Stop LabViz V2.3** to close its services before backup or maintenance.

V2.3 uses a separate installation identity and profile from V2.2. It does not automatically adopt V2.2 data.

| Item | Default location |
| --- | --- |
| Program | `%LOCALAPPDATA%\Programs\LabVizV23` |
| Default profile | `%LOCALAPPDATA%\LabVizV23` |
| Independent restored/imported profiles | `%LOCALAPPDATA%\LabVizV23Profiles` |
| Database and object files | The profile's `data` directory |
| Logs and backups | The profile's `logs` and `backups` directories |

Web and API services listen on loopback. Preferred ports are 3301 and 8301; an occupied port is replaced without stopping its existing owner. Each installed profile has its own browser cookie namespace.

## Back up and restore

Stop the app and wait for its services to exit. These commands address the default profile:

```powershell
$maintenance = "$env:LOCALAPPDATA\Programs\LabVizV23\bin\maintain-v23.ps1"
& $maintenance -Action Info
& $maintenance -Action Backup
```

Backup reports its directory. Keep the complete directory together; it contains the database, object files, and integrity manifest. Verify it before restoring:

```powershell
& $maintenance -Action Verify -BackupDirectory 'D:\LabViz-backups\my-backup'
& $maintenance -Action Restore -BackupDirectory 'D:\LabViz-backups\my-backup'
```

Restore creates an independent profile and prints its start command. Project identifiers, ownership records, revisions, and frozen files are retained. The restored profile receives a new local browser identity. Add `-ProfileRoot` with that profile's printed path for subsequent maintenance.

Keep original instrument files separately. A LabViz profile backup preserves the application's stored data; it does not replace an external raw-data archive.

## Copy V2.2 data into V2.3

Stop V2.2 first. To copy its default installed data into a new V2.3 profile:

```powershell
& $maintenance -Action Import -SourceDirectory "$env:LOCALAPPDATA\LabViz\data"
```

Import creates an independent destination and prints its start command. It copies the closed source, retains scientific identifiers and frozen files, and leaves the source unchanged. It does not combine data with the active V2.3 profile. Account-owned records retain their owners; import does not grant access to another account's projects.

For a source checkout, explicitly select its stopped database directory. Both `labviz.db` and `labviz-v2.db` are recognized. Imports into an existing nonempty destination are refused.

## Upgrade, rollback, and uninstall

Stop every installed V2.3 profile before installing another candidate. Program versions use immutable candidate directories. Activation backs up existing data, checks the candidate's API and website, and changes the active candidate only after those checks pass. A failed activation restores the prior candidate and its verified data snapshot when one exists.

Use rollback only when a compatible previous candidate and matching snapshot exist:

```powershell
& $maintenance -Action Rollback -SkipOpenBrowser
```

Newer data is retained separately when a compatible snapshot is restored. Automatic rollback refuses a newer unsupported data schema. Retained data and backup copies still need disk space.

Uninstalling removes the program and keeps profiles and backups. Reinstalling can reopen that retained data. To remove one profile deliberately, stop it, read its exact ID with `Info`, and supply both the profile path and that ID:

```powershell
& $maintenance -Action RemoveData -ProfileRoot 'C:\path\to\the\printed\profile' -ConfirmProfileId 'exact-profile-id-from-Info'
```

This removes that profile, including its backups. Copy any backup you want to retain outside it first.
