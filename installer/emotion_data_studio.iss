; ============================================================
; Emotion Data Studio — Inno Setup Installer Script
; ============================================================
; For PySide6 desktop app (NOT Electron)
; Packages the PyInstaller output + FFmpeg + data directories
;
; Build: ISCC.exe /DMyAppVersion=1.0.0 emotion_data_studio.iss
; ============================================================

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#define MyAppName "Emotion Data Studio"
#define MyAppPublisher "BCDA Team"
#define MyAppURL "https://github.com/bcda-team/emotion-data-studio"
#define MyAppExeName "EmotionDataStudio.exe"
#define MyAppId "{{B8C5D9E2-4F1A-4B7D-9E3C-8F2A1D5B6E7C}"

[Setup]
AppId={#MyAppId}
AppMutex={{B8C5D9E2-4F1A-4B7D-9E3C-8F2A1D5B6E7C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output settings
OutputDir=output
OutputBaseFilename=EmotionDataStudio-{#MyAppVersion}-Setup
; Compression
Compression=lzma2/ultra64
SolidCompression=yes
; Visual
SetupIconFile=..\assets\icon.ico
WizardStyle=modern
WizardSizePercent=120
; Requirements
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Uninstaller
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
; Architecture
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; === Main Application (PyInstaller output) ===
Source: "..\dist\EmotionDataStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; === FFmpeg binaries (nếu có) ===
Source: "..\bin\ffmpeg.exe"; DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\bin\ffprobe.exe"; DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist

; === External downloader / runtime helpers (nếu có) ===
Source: "..\bin\aria2c.exe"; DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\bin\deno.exe"; DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\bin\denort.exe"; DestDir: "{app}\bin"; Flags: ignoreversion skipifsourcedoesntexist

; === Data directories (create empty if not exist) ===
; These will be created by the app on first run, but we ensure the structure exists
; Source: "..\data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs onlyifdoesntexist skipifsourcedoesntexist

[Dirs]
; Ensure data directories exist with proper permissions
Name: "{app}\data"; Permissions: users-modify
Name: "{app}\data\videos"; Permissions: users-modify
Name: "{app}\data\clips"; Permissions: users-modify  
Name: "{app}\data\frames"; Permissions: users-modify
Name: "{app}\data\audio"; Permissions: users-modify
Name: "{app}\data\transcripts"; Permissions: users-modify
Name: "{app}\data\exports"; Permissions: users-modify
Name: "{app}\data\models_cache"; Permissions: users-modify
Name: "{app}\data\logs"; Permissions: users-modify

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Registry]
; Register file associations (optional)
Root: HKCU; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletekey

; Set environment variable for data directory (so backend/config.py can find it)
; Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "EDS_DATA_DIR"; ValueData: "{app}\data"; Flags: uninsdeletevalue

; Set FFmpeg path
; Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "EDS_FFMPEG_PATH"; ValueData: "{app}\bin\ffmpeg.exe"; Flags: uninsdeletevalue
; Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "EDS_FFPROBE_PATH"; ValueData: "{app}\bin\ffprobe.exe"; Flags: uninsdeletevalue

[UninstallDelete]
; Clean up generated files on uninstall (but keep user data)
Type: files; Name: "{app}\data\logs\*.log"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\build"

[Code]
// === Custom code for advanced installer behavior ===

function InitializeSetup(): Boolean;
begin
  Result := True;
  
  // Check if already running
  if CheckForMutexes('{#MyAppId}') then
  begin
    if MsgBox('{#MyAppName} is currently running.' + #13#10 + 
              'Please close it before installing.', 
              mbError, MB_RETRYCANCEL) = IDRETRY then
    begin
      // Give user a chance to close
      Sleep(2000);
      if CheckForMutexes('{#MyAppId}') then
      begin
        MsgBox('Please close {#MyAppName} and try again.', mbError, MB_OK);
        Result := False;
      end;
    end
    else
      Result := False;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Notify Windows of environment variable changes
    // This ensures EDS_DATA_DIR is available immediately
    // RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'EDS_DATA_DIR', 
    //                     ExpandConstant('{app}\data'));
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
end;
