#define AppName "LabViz"
#define AppPublisher "LabViz"
#define AppId "{{A7C9E5A0-2B18-4CE7-9A80-3B6C7F5B3D22}"

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

[Files]
Source: "{#CandidateRoot}\*"; DestDir: "{app}\versions\{#AppVersion}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "bin\start-labviz-installed.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\start-labviz-installed.cmd"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "bin\set-labviz-version.ps1"; DestDir: "{app}\bin"; Flags: ignoreversion

[Icons]
Name: "{group}\LabViz"; Filename: "{app}\bin\start-labviz-installed.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\versions\{#AppVersion}\V2.0\assets\labviz-logo.ico"
Name: "{commondesktop}\LabViz"; Filename: "{app}\bin\start-labviz-installed.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\versions\{#AppVersion}\V2.0\assets\labviz-logo.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\bin\start-labviz-installed.cmd"; Description: "Launch LabViz"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\versions"
Type: filesandordirs; Name: "{app}\bin"
Type: files; Name: "{app}\current-version.txt"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    SaveStringToFile(ExpandConstant('{app}\current-version.txt'), '{#AppVersion}' + #13#10, False);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Answer: Integer;
begin
  if CurUninstallStep = usUninstall then begin
    if UninstallSilent then
      Answer := IDYES
    else
      Answer := MsgBox(
        'Keep LabViz local data, logs, and backups? Choosing No permanently deletes them.',
        mbConfirmation, MB_YESNO);
    if Answer = IDNO then begin
      DelTree(ExpandConstant('{localappdata}\LabViz\data'), True, True, True);
      DelTree(ExpandConstant('{localappdata}\LabViz\logs'), True, True, True);
      DelTree(ExpandConstant('{localappdata}\LabViz\backups'), True, True, True);
    end;
  end;
end;
