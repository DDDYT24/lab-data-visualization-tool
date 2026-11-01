#ifndef BundleRoot
  #error BundleRoot is required.
#endif
#ifndef PackageId
  #error PackageId is required.
#endif
#ifndef PackageDir
  #error PackageDir is required.
#endif
#ifndef SupportZip
  #error SupportZip is required.
#endif
#ifndef OutputDir
  #error OutputDir is required.
#endif
#if !FileExists(AddBackslash(BundleRoot) + "desktop-package.json")
  #error An exact desktop package manifest is required.
#endif

[Setup]
AppId={{CA2B3AF4-BFB0-4541-8EA5-2E978F105D63}
AppName=LabViz Desktop Preview
AppVersion=2.3.0
AppVerName=LabViz Desktop Preview 2.3.0-desktop.1
AppPublisher=LabViz
DefaultDirName={localappdata}\Programs\LabVizDesktopPreview
DefaultGroupName=LabViz Desktop Preview
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=LabViz-Desktop-2.3.0-desktop.1-Private-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#BundleRoot}\app\V2.0\assets\labviz-logo.ico
UninstallDisplayIcon={app}\LabViz.exe
CloseApplications=no
RestartApplications=no
SetupMutex=Local\LabVizDesktopPreview-Setup
VersionInfoVersion=2.3.0.1
VersionInfoDescription=LabViz private desktop preview

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "..\..\V2.0\packaging\windows\languages\ChineseSimplified.isl"

[Files]
Source: "{#BundleRoot}\*"; DestDir: "{app}\versions\{#PackageDir}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BundleRoot}\LabViz.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "identity.json"; DestDir: "{app}"; DestName: "desktop-installation.json"; Flags: ignoreversion
Source: "prepare-install.ps1"; DestDir: "{app}\maintenance"; Flags: ignoreversion
Source: "webview2.ps1"; DestDir: "{app}\maintenance"; Flags: ignoreversion
Source: "webview2.json"; DestDir: "{app}\maintenance"; Flags: ignoreversion
Source: "{#SupportZip}"; DestDir: "{app}\maintenance"; DestName: "installer-support.zip"; Flags: ignoreversion

[Icons]
Name: "{group}\LabViz Desktop Preview"; Filename: "{app}\LabViz.exe"; WorkingDir: "{app}"; Check: CreateUserShortcuts
Name: "{userdesktop}\LabViz Desktop Preview"; Filename: "{app}\LabViz.exe"; WorkingDir: "{app}"; Tasks: desktopicon; Check: CreateUserShortcuts

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut / 创建桌面快捷方式"; Flags: unchecked

[Run]
Filename: "{app}\LabViz.exe"; Description: "Open LabViz Desktop Preview / 打开桌面预览"; Flags: nowait postinstall skipifsilent; Check: CanLaunchDesktop

[UninstallDelete]
Type: files; Name: "{app}\current-candidate.json"
Type: files; Name: "{app}\pending-candidate.json"
Type: files; Name: "{app}\upgrade.json"
Type: files; Name: "{app}\last-upgrade.json"
Type: files; Name: "{app}\failed-upgrade.json"
Type: dirifempty; Name: "{app}\versions"
Type: dirifempty; Name: "{app}\maintenance"

[Code]
var
  Prepared, Completed, ActivationFailed: Boolean;

function CreateUserShortcuts(): Boolean;
begin
  Result := ExpandConstant('{param:ACCEPTANCEROOT|}') = '';
end;

function CanLaunchDesktop(): Boolean;
begin
  Result := Completed and not ActivationFailed;
end;

function GetCustomSetupExitCode(): Integer;
begin
  if ActivationFailed then Result := 20 else Result := 0;
end;

function MoveFileEx(ExistingFile, NewFile: String; Flags: Cardinal): Boolean;
  external 'MoveFileExW@kernel32.dll stdcall';

function ProfileBase(): String;
var
  Acceptance: String;
begin
  Result := ExpandConstant('{localappdata}\LabVizDesktopPreview');
  Acceptance := ExpandConstant('{param:ACCEPTANCEROOT|}');
  if Acceptance <> '' then begin
    if ((Pos(GetEnv('USERPROFILE') + '\LabVizD6\', Acceptance) <> 1) and
        (Pos(GetEnv('USERPROFILE') + '\LabVizD2\', Acceptance) <> 1)) or
       (Pos('..', Acceptance) > 0) or not FileExists(AddBackslash(Acceptance) + 'acceptance.json') then
      RaiseException('The private installer acceptance path is invalid.');
    Result := AddBackslash(Acceptance) + 'profile';
  end;
end;

function Maintain(Action: String; Temporary: Boolean): Boolean;
var
  Folder, Params: String;
  Code: Integer;
begin
  if Temporary then begin
    ExtractTemporaryFile('prepare-install.ps1');
    ExtractTemporaryFile('webview2.ps1');
    ExtractTemporaryFile('webview2.json');
    ExtractTemporaryFile('installer-support.zip');
    Folder := ExpandConstant('{tmp}');
  end else
    Folder := ExpandConstant('{app}\maintenance');
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + AddBackslash(Folder) + 'prepare-install.ps1" -Action ' + Action +
    ' -InstallRoot "' + ExpandConstant('{app}') + '" -ProfileBase "' + ProfileBase() +
    '" -SupportZip "' + AddBackslash(Folder) + 'installer-support.zip" -PackageId "{#PackageId}"';
  Result := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params, '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not Maintain('Prepare', True) then
    Result := 'Desktop preparation or WebView2 installation failed. Finish active work, check the connection and select Retry. Earlier programs and data are retained. / 准备或组件补装失败，请完成任务、检查网络后重试，原程序和资料保留。'
  else Prepared := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Temporary: String;
begin
  if CurStep = ssPostInstall then begin
    ActivationFailed := True;
    Temporary := ExpandConstant('{app}\pending-candidate.new.json');
    if not SaveStringToFile(Temporary, '{"appId":"org.labviz.desktop.preview","packageId":"{#PackageId}"}', False) or
       not MoveFileEx(Temporary, ExpandConstant('{app}\pending-candidate.json'), 9) then
      RaiseException('Cannot atomically prepare desktop activation. Data is retained.');
    if not Maintain('Activate', False) then
      RaiseException('The new desktop candidate could not activate. Previous files and data were retained. Run setup again to retry. / 新候选激活失败，已有资料保留，请重新运行安装器。');
    Completed := True;
    ActivationFailed := False;
  end;
end;

procedure DeinitializeSetup();
begin
  if Prepared and not Completed then Maintain('Abort', True);
end;

function InitializeUninstall(): Boolean;
begin
  Result := Maintain('Uninstall', False);
  if not Result then SuppressibleMsgBox('Finish current work and exit LabViz, then retry uninstall. Your data is retained. / 请完成任务并退出后重试，资料保留。', mbError, MB_OK, IDOK);
end;
