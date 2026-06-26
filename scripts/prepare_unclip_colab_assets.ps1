param(
    [string]$OutputDir = "colab_export",
    [string]$ArchiveName = "biai_unclip_assets.zip"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$relativeSources = @(
    "images",
    "reconstruction_manifests",
    "scripts/train_eeg_image_retrieval.py",
    "scripts/extract_unclip_image_embeddings.py",
    "scripts/generate_unclip_from_eeg.py"
)

$files = @()
foreach ($relativeSource in $relativeSources) {
    $source = Join-Path $repoRoot $relativeSource
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Brak wymaganego pliku lub katalogu: $source"
    }
    if (Test-Path -LiteralPath $source -PathType Container) {
        $files += Get-ChildItem -LiteralPath $source -Recurse -File
    }
    else {
        $files += Get-Item -LiteralPath $source
    }
}

$outputPath = Join-Path $repoRoot $OutputDir
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$archivePath = Join-Path $outputPath $ArchiveName
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

$archiveStream = [System.IO.File]::Open(
    $archivePath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
$archive = [System.IO.Compression.ZipArchive]::new(
    $archiveStream,
    [System.IO.Compression.ZipArchiveMode]::Create,
    $false
)
try {
    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($repoRoot.Length + 1).Replace("\", "/")
        $entry = $archive.CreateEntry($relativePath, [System.IO.Compression.CompressionLevel]::Optimal)
        $entryStream = $entry.Open()
        $fileStream = [System.IO.File]::OpenRead($file.FullName)
        try {
            $fileStream.CopyTo($entryStream)
        }
        finally {
            $fileStream.Dispose()
            $entryStream.Dispose()
        }
    }
}
finally {
    $archive.Dispose()
    $archiveStream.Dispose()
}

$hash = Get-FileHash -LiteralPath $archivePath -Algorithm SHA256
$manifest = [ordered]@{
    archive_name = $ArchiveName
    files = $files.Count
    archive_size_bytes = (Get-Item -LiteralPath $archivePath).Length
    sha256 = $hash.Hash.ToLowerInvariant()
    contents = $relativeSources
}
$manifestPath = Join-Path $outputPath "biai_unclip_assets.manifest.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Gotowe: $archivePath"
Write-Host "Manifest: $manifestPath"
Write-Host "SHA256: $($manifest.sha256)"
