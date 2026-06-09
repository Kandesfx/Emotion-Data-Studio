<#
.SYNOPSIS
    Publish a new release of Emotion Data Studio to Cloudflare R2.

.DESCRIPTION
    Automates the full release pipeline:
    1. Build the app (calls build.ps1)
    2. Calculate SHA256 hash of the installer
    3. Generate latest.json metadata
    4. Upload installer + latest.json to Cloudflare R2 via wrangler, AWS CLI, or Python boto3

.EXAMPLE
    .\build\publish_release.ps1 -Version "1.1.0"
    .\build\publish_release.ps1 -Version "1.1.0" -SkipBuild
    .\build\publish_release.ps1 -Version "1.1.0" -ReleaseNotes "- Fix crash on export`n- Improve AI pipeline speed"
    .\build\publish_release.ps1 -Version "1.1.0" -DryRun

.NOTES
    Requirements:
    - R2 credentials configured in .env or environment variables
    - PyInstaller + Inno Setup for full build
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [string]$ReleaseNotes = "",

    [string]$R2BucketName = "emotion-data-studio",

    # Public base URL where R2 files are accessible
    [string]$R2PublicUrl = $env:EDS_UPDATE_URL,

    # Upload method: "python", "wrangler" or "aws" (S3-compatible)
    [ValidateSet("python", "wrangler", "aws")]
    [string]$UploadMethod = "python",

    [switch]$SkipBuild,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# === Paths ===
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"
$InstallerOutputDir = Join-Path $ProjectRoot "installer\output"
$InstallerFilename = "EmotionDataStudio-$Version-Setup.exe"
$InstallerPath = Join-Path $InstallerOutputDir $InstallerFilename
$LatestJsonPath = Join-Path $BuildDir "latest.json"

# === Load environment variables from .env ===
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | Foreach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $key, $value = $line -split '=', 2
            $key = $key.Trim()
            $value = $value.Trim()
            if ($value -match "^['`"\""](.*)['`"\""]$") {
                $value = $matches[1]
            }
            if (-not [string]::IsNullOrEmpty($key)) {
                [Environment]::SetEnvironmentVariable($key, $value)
            }
        }
    }
}

if (-not $R2PublicUrl -and $env:EDS_UPDATE_URL) {
    $R2PublicUrl = $env:EDS_UPDATE_URL
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  Emotion Data Studio - Publish Release v$Version" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

if ($DryRun) {
    Write-Host "  *** DRY RUN MODE - No files will be uploaded ***" -ForegroundColor Yellow
    Write-Host ""
}

# ================================================================
# Step 1: Build (optional)
# ================================================================

if (-not $SkipBuild) {
    Write-Host "[1/5] Building application..." -ForegroundColor Yellow
    $buildScript = Join-Path $BuildDir "build.ps1"

    if (-not (Test-Path $buildScript)) {
        throw "Build script not found: $buildScript"
    }

    & $buildScript -Version $Version

    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
    Write-Host "  [OK] Build complete" -ForegroundColor Green
} else {
    Write-Host "[1/5] Skipping build (using existing installer)" -ForegroundColor DarkGray
}

# ================================================================
# Step 2: Verify installer exists
# ================================================================

Write-Host "[2/5] Verifying installer..." -ForegroundColor Yellow

if (-not (Test-Path $InstallerPath)) {
    # Try to find any installer in the output directory
    $foundInstaller = Get-ChildItem $InstallerOutputDir -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($foundInstaller) {
        $InstallerPath = $foundInstaller.FullName
        $InstallerFilename = $foundInstaller.Name
        Write-Host "  [INFO] Found installer: $InstallerFilename" -ForegroundColor Cyan
    } else {
        throw "Installer not found at: $InstallerPath`nRun without -SkipBuild or build first."
    }
}

$fileSize = (Get-Item $InstallerPath).Length
$fileSizeMB = [math]::Round($fileSize / 1MB, 2)
Write-Host "  [OK] Installer found: $InstallerFilename ($fileSizeMB MB)" -ForegroundColor Green

# ================================================================
# Step 3: Calculate SHA256 hash
# ================================================================

Write-Host "[3/5] Calculating SHA256 hash..." -ForegroundColor Yellow

$sha256 = (Get-FileHash -Path $InstallerPath -Algorithm SHA256).Hash.ToLower()
Write-Host "  [OK] SHA256: $sha256" -ForegroundColor Green

# ================================================================
# Step 4: Generate latest.json
# ================================================================

Write-Host "[4/5] Generating latest.json..." -ForegroundColor Yellow

if (-not $R2PublicUrl) {
    $R2PublicUrl = "https://releases.your-domain.com"
    Write-Host "  [WARNING] R2PublicUrl not set. Using placeholder: $R2PublicUrl" -ForegroundColor Yellow
    Write-Host "  Set EDS_UPDATE_URL env var or pass -R2PublicUrl parameter" -ForegroundColor Yellow
}

# Clean trailing slash
$R2PublicUrl = $R2PublicUrl.TrimEnd("/")

$downloadUrl = "$R2PublicUrl/$InstallerFilename"

# Build release notes
if (-not $ReleaseNotes) {
    $ReleaseNotes = "Emotion Data Studio v$Version"
}

$latestJson = @{
    version       = $Version
    download_url  = $downloadUrl
    release_notes = $ReleaseNotes
    file_size     = $fileSize
    sha256        = $sha256
    published_at  = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json -Depth 4

# Write to file
[System.IO.File]::WriteAllText($LatestJsonPath, $latestJson, [System.Text.UTF8Encoding]::new($false))
Write-Host "  [OK] latest.json generated:" -ForegroundColor Green
Write-Host ""
Write-Host $latestJson -ForegroundColor DarkCyan
Write-Host ""

# ================================================================
# Step 5: Upload to R2
# ================================================================

Write-Host "[5/5] Uploading to Cloudflare R2..." -ForegroundColor Yellow

if ($DryRun) {
    Write-Host "  [DRY RUN] Would upload:" -ForegroundColor Yellow
    Write-Host "    $InstallerPath -> $R2BucketName/$InstallerFilename" -ForegroundColor DarkGray
    Write-Host "    $LatestJsonPath -> $R2BucketName/latest.json" -ForegroundColor DarkGray
} else {
    if ($UploadMethod -eq "python") {
        # === Upload via Python Script (boto3) ===
        Write-Host "  Uploading installer ($fileSizeMB MB) via Python..." -ForegroundColor Cyan
        & python "$ProjectRoot\build\upload_r2.py" "$InstallerPath" "$R2BucketName" "$InstallerFilename" "application/octet-stream"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload installer to R2 via Python script" }

        Write-Host "  Uploading latest.json via Python..." -ForegroundColor Cyan
        & python "$ProjectRoot\build\upload_r2.py" "$LatestJsonPath" "$R2BucketName" "latest.json" "application/json"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload latest.json to R2 via Python script" }

        Write-Host "  [OK] Upload complete via Python" -ForegroundColor Green

    } elseif ($UploadMethod -eq "wrangler") {
        # === Upload via Wrangler (Cloudflare CLI) ===

        # Check wrangler is installed
        $wranglerPath = Get-Command "wrangler" -ErrorAction SilentlyContinue
        if (-not $wranglerPath) {
            throw "wrangler CLI not found. Install with: npm install -g wrangler`nThen run: wrangler login"
        }

        Write-Host "  Uploading installer ($fileSizeMB MB)..." -ForegroundColor Cyan
        & wrangler r2 object put "$R2BucketName/$InstallerFilename" --file="$InstallerPath" --content-type="application/octet-stream"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload installer to R2" }

        Write-Host "  Uploading latest.json..." -ForegroundColor Cyan
        & wrangler r2 object put "$R2BucketName/latest.json" --file="$LatestJsonPath" --content-type="application/json"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload latest.json to R2" }

        Write-Host "  [OK] Upload complete via wrangler" -ForegroundColor Green

    } elseif ($UploadMethod -eq "aws") {
        # === Upload via AWS CLI (S3-compatible) ===

        # Check aws CLI is installed
        $awsPath = Get-Command "aws" -ErrorAction SilentlyContinue
        if (-not $awsPath) {
            throw "AWS CLI not found. Install from: https://aws.amazon.com/cli/"
        }

        # R2 endpoint from environment
        $r2Endpoint = $env:R2_ENDPOINT
        if (-not $r2Endpoint) {
            throw "R2_ENDPOINT environment variable not set. Set it to your R2 S3 API endpoint."
        }

        Write-Host "  Uploading installer ($fileSizeMB MB)..." -ForegroundColor Cyan
        & aws s3 cp $InstallerPath "s3://$R2BucketName/$InstallerFilename" `
            --endpoint-url $r2Endpoint `
            --content-type "application/octet-stream"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload installer to R2 via AWS CLI" }

        Write-Host "  Uploading latest.json..." -ForegroundColor Cyan
        & aws s3 cp $LatestJsonPath "s3://$R2BucketName/latest.json" `
            --endpoint-url $r2Endpoint `
            --content-type "application/json" `
            --cache-control "no-cache, max-age=0"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload latest.json to R2 via AWS CLI" }

        Write-Host "  [OK] Upload complete via AWS CLI" -ForegroundColor Green
    }
}

# ================================================================
# Summary
# ================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  RELEASE PUBLISHED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Version:      $Version" -ForegroundColor White
Write-Host "  Installer:    $InstallerFilename ($fileSizeMB MB)" -ForegroundColor White
Write-Host "  SHA256:       $sha256" -ForegroundColor White
Write-Host "  Download URL: $downloadUrl" -ForegroundColor White
Write-Host "  Metadata:     $R2PublicUrl/latest.json" -ForegroundColor White
Write-Host ""

if ($DryRun) {
    Write-Host "  *** This was a DRY RUN - no files were uploaded ***" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "  Users running the app will be notified of this update" -ForegroundColor DarkGray
Write-Host "  on their next launch (auto-updater checks latest.json)." -ForegroundColor DarkGray
Write-Host ""
