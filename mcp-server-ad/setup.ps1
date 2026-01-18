# PowerShell setup script for Windows
# This script helps set up the MCP server

Write-Host "Setting up Anomaly Detection MCP Server..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Test the server: python test_server.py" -ForegroundColor White
Write-Host "2. Configure in Claude Desktop config:" -ForegroundColor White
Write-Host "   %APPDATA%\Claude\claude_desktop_config.json" -ForegroundColor Gray
Write-Host "3. Use this path in config:" -ForegroundColor White
$scriptPath = (Resolve-Path "server.py").Path
Write-Host "   $scriptPath" -ForegroundColor Gray
Write-Host "4. Or use venv Python:" -ForegroundColor White
$venvPython = (Resolve-Path "venv\Scripts\python.exe").Path
Write-Host "   $venvPython" -ForegroundColor Gray

