param(
    [string]$PythonPath = ".\\.venv313\\Scripts\\python.exe",
    [string]$ResultsRoot = "vae_participant_sweep_no_abc_20260626",
    [string[]]$Participants = @("Bear", "fghx", "Reshi"),
    [string]$MoleResnetGenerationDir = "eeg_to_vae_resnet_participant_image_mole_no_abc_20260620",
    [string]$MoleDinov2GenerationDir = "eeg_to_vae_dinov2_participant_image_mole_no_abc_20260620",
    [string]$MoleEnsembleGenerationDir = "eeg_to_vae_ensemble_participant_image_mole_no_abc_20260620",
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"

function Invoke-Python {
    param([string[]]$Arguments)
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Nie znaleziono interpretera: $PythonPath"
}
if (-not (Test-Path -LiteralPath $ResultsRoot)) {
    New-Item -ItemType Directory -Path $ResultsRoot | Out-Null
}

function Test-Completed {
    param([string]$SummaryPath)
    return Test-Path -LiteralPath $SummaryPath -PathType Leaf
}

function Assert-CanCreateOutput {
    param([string]$OutputDirectory, [string]$SummaryName)
    if ((Test-Path -LiteralPath $OutputDirectory) -and -not (Test-Completed (Join-Path $OutputDirectory $SummaryName))) {
        throw "Niekompletny katalog wynikow: $OutputDirectory. Sprawdz go recznie przed wznowieniem."
    }
}

foreach ($participant in $Participants) {
    $key = $participant.ToLowerInvariant()
    $participantRoot = Join-Path $ResultsRoot $key
    New-Item -ItemType Directory -Path $participantRoot -Force | Out-Null

    $retrievalResnet = "retrieval_participant_image_${key}_no_abc_resnet18_25ep_20260620"
    $retrievalDinov2 = "retrieval_participant_image_${key}_no_abc_dinov2_25ep_20260620"
    $manifest = "reconstruction_manifests/participant_image_${key}_no_abc"
    $validationEnsemble = "no_abc_participant_sweep_20260620/$key/ensemble/ensemble_summary.json"
    foreach ($required in @($retrievalResnet, $retrievalDinov2, $manifest, $validationEnsemble)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Brak wymaganego artefaktu: $required"
        }
    }

    $weights = Get-Content -LiteralPath $validationEnsemble -Raw | ConvertFrom-Json
    $resnetWeight = [string]$weights.image_averaged_weight_first
    $resnetOutput = Join-Path $participantRoot "generation_resnet"
    $dinov2Output = Join-Path $participantRoot "generation_dinov2"
    $ensembleOutput = Join-Path $participantRoot "generation_ensemble"

    if (Test-Completed (Join-Path $resnetOutput "vae_generation_summary.json")) {
        Write-Host "[$participant] ResNet -> VAE: juz gotowe"
    }
    else {
        Assert-CanCreateOutput $resnetOutput "vae_generation_summary.json"
        Write-Host "[$participant] ResNet -> VAE"
        Invoke-Python @(
            "scripts/reconstruct_vae_from_eeg.py",
            "--retrieval-result-dir", $retrievalResnet,
            "--ridge-result-dir", "ridge_embedding_to_vae_latent_20260620",
            "--output-dir", $resnetOutput
        )
    }

    if (Test-Completed (Join-Path $dinov2Output "vae_generation_summary.json")) {
        Write-Host "[$participant] DINOv2 -> VAE: juz gotowe"
    }
    else {
        Assert-CanCreateOutput $dinov2Output "vae_generation_summary.json"
        Write-Host "[$participant] DINOv2 -> VAE"
        Invoke-Python @(
            "scripts/reconstruct_vae_from_eeg.py",
            "--retrieval-result-dir", $retrievalDinov2,
            "--ridge-result-dir", "ridge_dinov2_to_vae_latent_20260620",
            "--output-dir", $dinov2Output
        )
    }

    if (Test-Completed (Join-Path $ensembleOutput "ensemble_generation_summary.json")) {
        Write-Host "[$participant] Ensemble: juz gotowe"
    }
    else {
        Assert-CanCreateOutput $ensembleOutput "ensemble_generation_summary.json"
        Write-Host "[$participant] Ensemble"
        Invoke-Python @(
            "scripts/ensemble_generated_images.py",
            "--first-generation-dir", $resnetOutput,
            "--second-generation-dir", $dinov2Output,
            "--manifest-dir", $manifest,
            "--weight-first", $resnetWeight,
            "--output-dir", $ensembleOutput
        )
    }
}

$aggregateArgs = @(
    "scripts/aggregate_vae_generation_sweep.py",
    "--results-root", $ResultsRoot,
    "--participants"
)
$aggregateArgs += $Participants
$aggregateArgs += @(
    "--external-source", ("mole|{0}|{1}|{2}" -f $MoleResnetGenerationDir, $MoleDinov2GenerationDir, $MoleEnsembleGenerationDir),
    "--output", (Join-Path $ResultsRoot "participant_vae_generation_all_summary.csv")
)
if ($SkipAggregate) {
    Write-Host "Pomijam agregacje na zadanie."
}
elseif ($Participants | ForEach-Object {
    Test-Completed (Join-Path (Join-Path $ResultsRoot $_.ToLowerInvariant()) "generation_ensemble\\ensemble_generation_summary.json")
} | Where-Object { -not $_ }) {
    Write-Host "Nie wszystkie ensemble sa gotowe; pomijam agregacje. Uruchom skrypt ponownie po dokonczeniu brakujacych etapow."
}
elseif (Test-Path -LiteralPath (Join-Path $ResultsRoot "participant_vae_generation_all_summary.csv")) {
    Write-Host "Agregacja juz istnieje."
}
else {
    Invoke-Python $aggregateArgs
}
