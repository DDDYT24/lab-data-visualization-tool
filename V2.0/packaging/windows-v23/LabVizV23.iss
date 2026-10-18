#ifndef CandidateRoot
  #error CandidateRoot is required; inherited V2.2 installers cannot be used.
#endif
#ifndef CandidateId
  #error CandidateId is required.
#endif
#ifndef OutputDir
  #error OutputDir is required.
#endif
#ifndef CandidateDir
  #error CandidateDir is required and must match the first 24 characters of CandidateId.
#endif
#if !FileExists(AddBackslash(CandidateRoot) + "portable-lifecycle-verified.json")
  #error Verify the independent portable lifecycle before compiling an installer.
#endif

[Setup]
AppId={{55B46BB7-101E-46DC-951A-3DE7E402EDD0}
AppName=LabViz V2.3
AppVersion=2.3.0
AppVerName=LabViz V2.3 2.3.0-rc.1
AppPublisher=LabViz
DefaultDirName={localappdata}\Programs\LabVizV23
DefaultGroupName=LabViz V2.3
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=LabViz-V23-2.3.0-rc.1-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#CandidateRoot}\V2.0\assets\labviz-logo.ico
UninstallDisplayIcon={app}\versions\{#CandidateDir}\V2.0\assets\labviz-logo.ico
Uninstallable=yes
CloseApplications=no
RestartApplications=no
AppMutex=Local\LabVizV23-App
SetupMutex=Local\LabVizV23-Setup
VersionInfoVersion=2.3.0.0
VersionInfoDescription=LabViz V2.3 release candidate
VersionInfoCopyright=LabViz contributors

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "..\windows\languages\ChineseSimplified.isl"

[Files]
Source: "{#CandidateRoot}\*"; DestDir: "{app}\versions\{#CandidateDir}"; Excludes: "portable-lifecycle-verified.json"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "bin\common.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\preflight-v23.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\start-v23-installed.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\start-v23-installed.cmd"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\maintain-v23.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "identity.json"; DestDir: "{app}"; DestName: "installation.json"; Flags: ignoreversion

[Icons]
Name: "{group}\LabViz V2.3"; Filename: "{app}\bin\start-v23-installed.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\versions\{#CandidateDir}\V2.0\assets\labviz-logo.ico"
Name: "{group}\Stop LabViz V2.3 - 退出"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bin\maintain-v23.ps1"" -Action Stop"
Name: "{userdesktop}\LabViz V2.3"; Filename: "{app}\bin\start-v23-installed.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\versions\{#CandidateDir}\V2.0\assets\labviz-logo.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut / 创建桌面快捷方式"; Flags: unchecked

[Run]
Filename: "{app}\bin\start-v23-installed.cmd"; Description: "Launch LabViz V2.3 / 启动 LabViz V2.3"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\current-candidate.json"
Type: files; Name: "{app}\pending-candidate.json"
Type: files; Name: "{app}\upgrade.json"
Type: files; Name: "{app}\last-upgrade.json"
Type: files; Name: "{app}\failed-upgrade.json"
Type: dirifempty; Name: "{app}\versions"
Type: dirifempty; Name: "{app}\bin"

[Code]
function MoveFileEx(ExistingFile, NewFile: String; Flags: Cardinal): Boolean;
  external 'MoveFileExW@kernel32.dll stdcall';

function Preflight(Installed: Boolean): Boolean;
var
  Params: String;
  Code: Integer;
begin
  if Installed then
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\bin\preflight-v23.ps1') + '" -InstallRoot "' + ExpandConstant('{app}') + '" -RequireIdentity -ScanTree'
  else begin
    ExtractTemporaryFile('common.ps1');
    ExtractTemporaryFile('preflight-v23.ps1');
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\preflight-v23.ps1') + '" -InstallRoot "' + ExpandConstant('{app}') + '" -CandidateId "{#CandidateId}"';
  end;
  Result := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params, '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not Preflight(False) then
    Result := 'V2.3 installation identity/path check failed. Existing installations and data were retained. / 安装身份或路径校验失败，已有安装与数据保留。';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Temporary: String;
begin
  if CurStep = ssPostInstall then begin
    Temporary := ExpandConstant('{app}\pending-candidate.new.json');
    if not SaveStringToFile(Temporary, '{"appId":"55B46BB7-101E-46DC-951A-3DE7E402EDD0","candidateId":"{#CandidateId}"}', False) then
      RaiseException('Cannot prepare the V2.3 activation marker.');
    if not MoveFileEx(Temporary, ExpandConstant('{app}\pending-candidate.json'), 9) then
      RaiseException('Cannot atomically activate the V2.3 candidate marker.');
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := Preflight(True);
end;
