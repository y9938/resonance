# Idempotent downloader for FFmpeg 7.1 shared libraries on Windows (TorchCodec runtime requirement)
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DepsDir = Join-Path $ProjectRoot ".deps\ffmpeg7"
$BinDir = Join-Path $DepsDir "bin"
$TestFile = Join-Path $BinDir "avcodec-61.dll"

function Ensure-UserPath([string]$TargetDir) {
    $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $Paths = ($CurrentPath -split ";") | Where-Object { $_ -ne "" }
    if ($Paths -notcontains $TargetDir) {
        $NewPath = "$TargetDir;$CurrentPath"
        [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
        Write-Host "Added $TargetDir to User PATH."
    }
    if (($env:PATH -split ";") -notcontains $TargetDir) {
        $env:PATH = "$TargetDir;$env:PATH"
    }
}

if (Test-Path $TestFile) {
    Write-Host "FFmpeg 7.1 shared libraries already installed in $DepsDir"
    Ensure-UserPath $BinDir
    exit 0
}

Write-Host "Installing FFmpeg 7.1 shared libraries into $DepsDir..."
if (-not (Test-Path $DepsDir)) {
    New-Item -ItemType Directory -Path $DepsDir -Force | Out-Null
}

$ZipUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-win64-lgpl-shared-7.1.zip"
$ExpectedSha256 = "0f376f96fb38554ccefb1b2ae9c7c6a7b351f0e60a372b38262c320e8392c5d0"
$ZipPath = Join-Path $env:TEMP "ffmpeg-7.1-win64.zip"

Write-Host "Downloading $ZipUrl..."
Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath

Write-Host "Verifying SHA-256 checksum..."
$ActualSha256 = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash.ToLower()
if ($ActualSha256 -ne $ExpectedSha256.ToLower()) {
    Remove-Item -Force $ZipPath
    throw "ERROR: Checksum mismatch for downloaded archive! Expected $ExpectedSha256, got $ActualSha256"
}

$TempExtract = Join-Path $env:TEMP "ffmpeg-7.1-extract"
if (Test-Path $TempExtract) {
    Remove-Item -Recurse -Force $TempExtract
}
New-Item -ItemType Directory -Path $TempExtract -Force | Out-Null

Write-Host "Extracting archive..."
Expand-Archive -Path $ZipPath -DestinationPath $TempExtract -Force

$SubDir = Get-ChildItem -Directory -Path $TempExtract | Select-Object -First 1
Get-ChildItem -Path $SubDir.FullName | Copy-Item -Destination $DepsDir -Recurse -Force

Remove-Item -Recurse -Force $TempExtract
Remove-Item -Force $ZipPath

Ensure-UserPath $BinDir

Write-Host "Verification: avcodec-61.dll present: $(Test-Path $TestFile)"
