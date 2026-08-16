[CmdletBinding()]
param(
    [string]$PythonExecutable = "python",
    [switch]$KeepWorkFiles
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$specPath = Join-Path $projectRoot "YOLODatasetAugmenter.spec"
$workPath = Join-Path $projectRoot ".pyinstaller-build"
$legacyWorkPath = Join-Path $projectRoot "build"
$distPath = Join-Path $projectRoot "dist"
$releasePath = Join-Path $distPath "YOLODatasetAugmenter"
$exePath = Join-Path $releasePath "YOLODatasetAugmenter.exe"
$pythonDllPath = Join-Path $releasePath "_internal\python312.dll"
$releaseNotesSource = Join-Path $PSScriptRoot "README_运行说明.txt"
$releaseNotesTarget = Join-Path $releasePath "README_运行说明.txt"

function Remove-WorkspaceDirectory {
    param([Parameter(Mandatory)][string]$TargetPath)

    $rootFull = [IO.Path]::GetFullPath($projectRoot).TrimEnd('\', '/')
    $targetFull = [IO.Path]::GetFullPath($TargetPath).TrimEnd('\', '/')
    $prefix = $rootFull + [IO.Path]::DirectorySeparatorChar
    if (-not $targetFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a directory outside the project: $targetFull"
    }
    if (Test-Path -LiteralPath $targetFull) {
        Remove-Item -LiteralPath $targetFull -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $specPath -PathType Leaf)) {
    throw "PyInstaller spec not found: $specPath"
}

Push-Location $projectRoot
try {
    & $PythonExecutable -m PyInstaller `
        --noconfirm `
        --clean `
        --workpath $workPath `
        --distpath $distPath `
        $specPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }

    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
        throw "Release executable is missing: $exePath"
    }
    if (-not (Test-Path -LiteralPath $pythonDllPath -PathType Leaf)) {
        throw "Bundled Python DLL is missing: $pythonDllPath"
    }

    Copy-Item -LiteralPath $releaseNotesSource -Destination $releaseNotesTarget -Force

    if (-not $KeepWorkFiles) {
        Remove-WorkspaceDirectory -TargetPath $workPath
        Remove-WorkspaceDirectory -TargetPath $legacyWorkPath
    }

    Write-Host "Build succeeded. Run only this executable:"
    Write-Host $exePath
    Write-Host "Keep the _internal directory beside the executable."
}
finally {
    Pop-Location
}
