#ifndef AppName
  #define AppName "LabViz"
#endif
#ifndef AppPublisher
  #define AppPublisher "LabViz"
#endif
#ifndef AppId
  #define AppId "{{A7C9E5A0-2B18-4CE7-9A80-3B6C7F5B3D22}"
#endif

#ifndef AppVersion
  #define AppVersion "2.2.0"
#endif
#ifndef CandidateRoot
  #define CandidateRoot "..\\..\\..\\outputs\\v22-package-candidate-production-20260913"
#endif
#ifndef OutputDir
  #define OutputDir "..\\..\\..\\outputs\\v22-installer"
#endif

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\LabViz
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=LabViz-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#CandidateRoot}\V2.0\assets\labviz-logo.ico
UninstallDisplayIcon={app}\versions\{#AppVersion}\V2.0\assets\labviz-logo.ico
Uninstallable=yes
CloseApplications=yes
RestartApplications=no
VersionInfoVersion={#AppVersion}.0
VersionInfoDescription=LabViz local-first scientific data visualization
VersionInfoCopyright=LabViz contributors

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "languages\ChineseSimplified.isl"

[Files]
Source: "{#CandidateRoot}\*"; DestDir: "{app}\versions\{#AppVersion}"; Excludes: "*.pyc,*.pyo"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "bin\start-labviz-installed.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\start-labviz-installed.cmd"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\set-labviz-version.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion

Source: "bin\local-data.py"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\maintain-labviz.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion

[Icons]
Name: "{group}\Import old data - 迁移旧数据"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bin\maintain-labviz.ps1"" -Action Import"
Name: "{group}\Stop LabViz - 退出"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bin\maintain-labviz.ps1"" -Action Stop"
Name: "{group}\LabViz logs - 日志"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bin\maintain-labviz.ps1"" -Action Logs"
Name: "{group}\Rollback LabViz - 回滚"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bin\maintain-labviz.ps1"" -Action Rollback"
Name: "{group}\LabViz"; Filename: "{app}\bin\start-labviz-installed.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\versions\{#AppVersion}\V2.0\assets\labviz-logo.ico"
Name: "{userdesktop}\LabViz"; Filename: "{app}\bin\start-labviz-installed.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\versions\{#AppVersion}\V2.0\assets\labviz-logo.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut / 创建桌面快捷方式"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\bin\start-labviz-installed.cmd"; Description: "Launch LabViz / 启动 LabViz"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\versions"
Type: filesandordirs; Name: "{app}\bin"
Type: files; Name: "{app}\current-version.txt"
Type: files; Name: "{app}\pending-version.txt"
Type: files; Name: "{app}\upgrade.json"
Type: files; Name: "{app}\last-upgrade.json"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    SaveStringToFile(ExpandConstant('{app}\pending-version.txt'), '{#AppVersion}' + #13#10, False);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Answer: Integer;
  UserLabVizRoot: String;
begin
  if CurUninstallStep = usUninstall then begin
    #ifdef TestDeleteData
      Answer := IDNO;
    #else
    if UninstallSilent then
      Answer := IDYES
    else
      Answer := MsgBox(
        'Keep local data, logs, backups? / 保留本地数据、日志和备份？ No = permanently delete / 否 = 永久删除。',
        mbConfirmation, MB_YESNO);
    #endif
    if Answer = IDNO then begin
      UserLabVizRoot := GetEnv('LOCALAPPDATA');
      if UserLabVizRoot = '' then
        UserLabVizRoot := ExpandConstant('{localappdata}');
      DelTree(AddBackslash(UserLabVizRoot) + 'LabViz\data', True, True, True);
      DelTree(AddBackslash(UserLabVizRoot) + 'LabViz\logs', True, True, True);
      DelTree(AddBackslash(UserLabVizRoot) + 'LabViz\backups', True, True, True);
      DelTree(AddBackslash(UserLabVizRoot) + 'LabViz\data.retained-*', False, True, True);
      DelTree(AddBackslash(UserLabVizRoot) + 'LabViz\data.failed-*', False, True, True);
    end;
  end;
end;
