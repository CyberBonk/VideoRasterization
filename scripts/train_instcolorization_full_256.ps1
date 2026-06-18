param(
    [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    [string]$InstRepo = "C:\Users\bebo\Documents\GitHub\InstColorization2025",
    [int]$Epochs = 3,
    [int]$BatchSize = 16,
    [int]$Workers = 4,
    [double]$LearningRate = 0.0005,
    [string]$ExperimentName = "coco_full_256_train2017",
    [string]$TrainDirOverride = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RepoRoot = $RepoRoot.Path

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}
if (-not (Test-Path $InstRepo)) {
    throw "InstColorization2025 repo not found: $InstRepo"
}

if ($TrainDirOverride) {
    $TrainDir = (Resolve-Path $TrainDirOverride).Path
} else {
    $RootTrain2017 = Join-Path $RepoRoot "train2017"
    $ChromaTrain2017 = Join-Path $RepoRoot "ChromaNet_v3_complete\chromanet_v3\data\train2017"
    $TrainZip = Join-Path $RepoRoot "train2017.zip"

    if (Test-Path $RootTrain2017) {
        $TrainDir = (Resolve-Path $RootTrain2017).Path
    } elseif (Test-Path $ChromaTrain2017) {
        $TrainDir = (Resolve-Path $ChromaTrain2017).Path
    } elseif (Test-Path $TrainZip) {
        Write-Host "[extract] train2017.zip -> $RepoRoot"
        Expand-Archive -LiteralPath $TrainZip -DestinationPath $RepoRoot -Force
        $TrainDir = (Resolve-Path $RootTrain2017).Path
    } else {
        throw "No train2017 folder or train2017.zip found."
    }
}

$CheckpointRoot = Join-Path $RepoRoot "tools\AImodels\instcolorization2025\checkpoints"
$CheckpointDir = Join-Path $CheckpointRoot $ExperimentName
$SeedCheckpoint = Join-Path $RepoRoot "tools\AImodels\DataSets\siggraph17-df00044c.pth"
$LatestCheckpoint = Join-Path $CheckpointDir "latest_net_G.pth"

if (-not (Test-Path $SeedCheckpoint)) {
    throw "Seed checkpoint not found: $SeedCheckpoint"
}

New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null
if (-not (Test-Path $LatestCheckpoint)) {
    Copy-Item -LiteralPath $SeedCheckpoint -Destination $LatestCheckpoint
}

$EndEpoch = $Epochs + 1

Write-Host "[start] VideoRasterization InstColorization full-branch training"
Write-Host "[data] $TrainDir"
Write-Host "[checkpoint] $LatestCheckpoint"
Write-Host "[config] epochs=$Epochs batch=$BatchSize workers=$Workers lr=$LearningRate size=256"

Push-Location $InstRepo
try {
    & $Python -u train.py `
        --stage full `
        --name $ExperimentName `
        --checkpoints_dir $CheckpointRoot `
        --sample_p 1.0 `
        --epoch_count 1 `
        --niter $EndEpoch `
        --niter_decay 0 `
        --load_model `
        --lr $LearningRate `
        --model train `
        --fineSize 256 `
        --loadSize 256 `
        --batch_size $BatchSize `
        --nThreads $Workers `
        --display_id -1 `
        --no_html `
        --display_freq 100000000 `
        --print_freq 400 `
        --save_epoch_freq 1 `
        --train_img_dir $TrainDir `
        2>&1 | Tee-Object -FilePath (Join-Path $RepoRoot "training_instcolorization_full_256.log")
} finally {
    Pop-Location
}
