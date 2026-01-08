# IronSight Command Center - PowerShell Launcher
# Simple wrapper for the Python launcher script

param(
    [switch]$Dev,
    [switch]$Cpu,
    [int]$Port = 8501,
    [string]$Host = "0.0.0.0",
    [switch]$NoBrowser,
    [switch]$Debug,
    [switch]$Help
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Change to script directory
Set-Location $PSScriptRoot

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

try {
    # Check if uv is available first
    try {
        $uvVersion = uv --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Found uv package manager - using for better performance" "Green"
            
            # Build arguments array
            $args = @()
            
            if ($Dev) { $args += "--dev" }
            if ($Cpu) { $args += "--cpu" }
            if ($Port -ne 8501) { $args += "--port", $Port }
            if ($Host -ne "0.0.0.0") { $args += "--host", $Host }
            if ($NoBrowser) { $args += "--no-browser" }
            if ($Debug) { $args += "--debug" }
            if ($Help) { $args += "--help" }

            # Run with uv
            Write-ColorOutput "Starting IronSight Command Center..." "Cyan"
            
            if ($args.Count -gt 0) {
                & uv run python run_ironsight.py @args
            } else {
                & uv run python run_ironsight.py
            }
            
            if ($LASTEXITCODE -ne 0) {
                Write-ColorOutput "An error occurred during execution" "Red"
                exit $LASTEXITCODE
            }
            return
        }
    } catch {
        # uv not available, continue to Python fallback
    }

    # Fallback to regular Python
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Error: Neither uv nor Python is installed or not in PATH" "Red"
        Write-ColorOutput "Please install uv (recommended) or Python 3.10-3.12 and try again" "Yellow"
        Write-ColorOutput "To install uv: https://docs.astral.sh/uv/getting-started/installation/" "Cyan"
        exit 1
    }

    Write-ColorOutput "Found: $pythonVersion" "Green"
    Write-ColorOutput "Using regular Python (consider installing uv for better performance)" "Yellow"

    # Build arguments array
    $args = @()
    
    if ($Dev) { $args += "--dev" }
    if ($Cpu) { $args += "--cpu" }
    if ($Port -ne 8501) { $args += "--port", $Port }
    if ($Host -ne "0.0.0.0") { $args += "--host", $Host }
    if ($NoBrowser) { $args += "--no-browser" }
    if ($Debug) { $args += "--debug" }
    if ($Help) { $args += "--help" }

    # Run the Python launcher
    Write-ColorOutput "Starting IronSight Command Center..." "Cyan"
    
    if ($args.Count -gt 0) {
        & python run_ironsight.py @args
    } else {
        & python run_ironsight.py
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "An error occurred during execution" "Red"
        exit $LASTEXITCODE
    }

} catch {
    Write-ColorOutput "Error: $_" "Red"
    Write-ColorOutput "Press any key to exit..." "Yellow"
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}