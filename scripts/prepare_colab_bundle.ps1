param(
    [string]$DatasetDir = "event_epoch_multisession_image_on_0_0p8_qc",
    [string]$OutputDir = "colab_export",
    [string]$ArchiveName = "biai_eeg_qc_0_0p8.zip"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$datasetPath = (Resolve-Path (Join-Path $repoRoot $DatasetDir)).Path
$metadataPath = Join-Path $datasetPath "metadata.csv"
if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
    throw "Brak metadata.csv w $datasetPath"
}

$outputPath = Join-Path $repoRoot $OutputDir
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$archivePath = Join-Path $outputPath $ArchiveName
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

Write-Host "Pakowanie $datasetPath"
$files = Get-ChildItem -LiteralPath $datasetPath -Recurse -File
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
        $entry = $archive.CreateEntry(
            $relativePath,
            [System.IO.Compression.CompressionLevel]::Optimal
        )
        $entry.LastWriteTime = $file.LastWriteTime
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

$size = ($files | Measure-Object -Property Length -Sum).Sum
$metadata = Import-Csv -LiteralPath $metadataPath
$hash = Get-FileHash -LiteralPath $archivePath -Algorithm SHA256

$manifest = [ordered]@{
    archive_name = $ArchiveName
    dataset_dir = Split-Path -Leaf $datasetPath
    dataset_files = $files.Count
    dataset_size_bytes = $size
    metadata_rows = $metadata.Count
    participants = @($metadata.participant | Sort-Object -Unique)
    categories = @($metadata.image_category | Sort-Object -Unique)
    archive_size_bytes = (Get-Item -LiteralPath $archivePath).Length
    sha256 = $hash.Hash.ToLowerInvariant()
}

$manifestPath = Join-Path $outputPath "biai_eeg_qc_0_0p8.manifest.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Gotowe:"
Write-Host "  Archiwum: $archivePath"
Write-Host "  Manifest: $manifestPath"
Write-Host "  SHA256:   $($manifest.sha256)"
