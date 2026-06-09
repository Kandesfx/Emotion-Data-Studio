<#
.SYNOPSIS
    Build script for Emotion Data Studio desktop application.
    
.DESCRIPTION
    Automates the full build pipeline:
    1. Clean previous build
    2. Run PyInstaller to create app bundle
    3. Copy external dependencies (FFmpeg, models)
    4. Run Inno Setup to create Windows installer
    
.EXAMPLE
    .\build\build.ps1
    .\build\build.ps1 -SkipInstaller
    .\build\build.ps1 -Version "1.2.0"
#>

param(
    [string]$Version = "1.0.0",
    [switch]$SkipInstaller,
    [switch]$SkipClean,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

# === Paths ===
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"
$DistDir = Join-Path $ProjectRoot "dist"
$AppDir = Join-Path $DistDir "EmotionDataStudio"
$InstallerDir = Join-Path $ProjectRoot "installer"
$SpecFile = Join-Path $BuildDir "emotion_studio.spec"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Emotion Data Studio - Build Script v$Version" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# === Step 1: Clean ===
if (-not $SkipClean) {
    Write-Host "[1/5] Cleaning previous build..." -ForegroundColor Yellow
    if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
    if (Test-Path (Join-Path $ProjectRoot "build/__pycache__")) {
        Remove-Item -Recurse -Force (Join-Path $ProjectRoot "build/__pycache__")
    }
    Write-Host "  [OK] Clean complete" -ForegroundColor Green
} else {
    Write-Host "[1/5] Skipping clean" -ForegroundColor DarkGray
}

# === Step 1.5: Stamp version into source ===
Write-Host "[1.5/5] Stamping version into source code..." -ForegroundColor Yellow
$configPath = Join-Path $ProjectRoot "backend\config.py"
$configContent = Get-Content $configPath -Raw
$originalVersionLine = ($configContent | Select-String 'VERSION:\s*str\s*=\s*"[^"]*"').Matches[0].Value
$configContent = $configContent -replace 'VERSION:\s*str\s*=\s*"[^"]*"', "VERSION: str = `"$Version`""
[System.IO.File]::WriteAllText($configPath, $configContent, [System.Text.UTF8Encoding]::new($false))
Write-Host "  [OK] Version stamped: $Version" -ForegroundColor Green

# === Step 2: PyInstaller ===
Write-Host "[2/5] Running PyInstaller..." -ForegroundColor Yellow
$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--distpath", $DistDir,
    "--workpath", (Join-Path $BuildDir "pyinstaller_work"),
    $SpecFile
)

if ($Debug) {
    $pyinstallerArgs += "--log-level", "DEBUG"
}

Push-Location $ProjectRoot
try {
    & python -m PyInstaller @pyinstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
Write-Host "  [OK] PyInstaller complete" -ForegroundColor Green

# === Step 3: Copy external dependencies ===
Write-Host "[3/5] Copying external dependencies..." -ForegroundColor Yellow

# Create required directories in app bundle
$dataDirs = @("data", "data/videos", "data/clips", "data/frames", "data/audio",
              "data/transcripts", "data/exports", "data/models_cache", "data/logs")
foreach ($dir in $dataDirs) {
    $fullPath = Join-Path $AppDir $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

# Copy FFmpeg if available locally
$ffmpegPath = Join-Path $ProjectRoot "bin\ffmpeg.exe"
if (Test-Path $ffmpegPath) {
    $binDir = Join-Path $AppDir "bin"
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    Copy-Item (Join-Path $ProjectRoot "bin\ffmpeg.exe") $binDir
    Copy-Item (Join-Path $ProjectRoot "bin\ffprobe.exe") $binDir -ErrorAction SilentlyContinue
    Write-Host "  [OK] FFmpeg copied" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] FFmpeg not found in bin/ - user will need system FFmpeg" -ForegroundColor Yellow
}

# Copy QSS styles (should be in datas already, but ensure it)
$stylesDir = Join-Path $AppDir "ui\styles"
if (-not (Test-Path $stylesDir)) {
    New-Item -ItemType Directory -Path $stylesDir -Force | Out-Null
}
Copy-Item (Join-Path $ProjectRoot "ui\styles\dark_theme.qss") $stylesDir -Force
Write-Host "  [OK] Styles copied" -ForegroundColor Green

# === Step 4: Write version info ===
Write-Host "[4/5] Writing version info..." -ForegroundColor Yellow
$versionInfo = @{
    version = $Version
    build_date = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    python_version = (python --version 2>&1).ToString().Replace("Python ", "")
} | ConvertTo-Json -Depth 2

[System.IO.File]::WriteAllText((Join-Path $AppDir "version.json"), $versionInfo, [System.Text.UTF8Encoding]::new($false))
Write-Host "  [OK] Version info written" -ForegroundColor Green

# === Step 5: Inno Setup (optional) ===
if (-not $SkipInstaller) {
    Write-Host "[5/5] Running Inno Setup..." -ForegroundColor Yellow
    
    $issFile = Join-Path $InstallerDir "emotion_data_studio.iss"
    if (-not (Test-Path $issFile)) {
        Write-Host "  ⚠️  Inno Setup script not found: $issFile" -ForegroundColor Yellow
        Write-Host "  Skipping installer creation" -ForegroundColor Yellow
    } else {
        $iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
        if (-not (Test-Path $iscc)) {
            $iscc = "iscc"  # Try system PATH
        }
        
        & $iscc "/DMyAppVersion=$Version" $issFile
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE"
        }
        Write-Host "  [OK] Installer created" -ForegroundColor Green
    }
} else {
    Write-Host "[5/5] Skipping installer" -ForegroundColor DarkGray
}

# === Summary ===
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  BUILD COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  App bundle: $AppDir" -ForegroundColor White
Write-Host "  Run: $AppDir\EmotionDataStudio.exe" -ForegroundColor White

if (-not $SkipInstaller -and (Test-Path (Join-Path $InstallerDir "output"))) {
    $installerFile = Get-ChildItem (Join-Path $InstallerDir "output") -Filter "*.exe" | Select-Object -First 1
    if ($installerFile) {
        Write-Host "  Installer: $($installerFile.FullName)" -ForegroundColor White
    }
}
Write-Host ""
