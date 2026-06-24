<# ==============================================================
   Digital Shield Rail Defense — Windows Deploy Script
   ==============================================================
   Usage:
     .\deploy\deploy.ps1                  # Full deploy
     .\deploy\deploy.ps1 -Monitoring      # With Prometheus + Grafana
     .\deploy\deploy.ps1 -Rebuild         # Force rebuild
   ============================================================== #>

param(
    [switch]$Monitoring,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path (Split-Path $PSCommandPath)
Set-Location $ProjectDir

function Write-Header($msg) {
    Write-Host "`n$('=' * 50)" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "$('=' * 50)`n" -ForegroundColor Cyan
}
function Write-Step($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[!!] $msg" -ForegroundColor Yellow }

Write-Header "DIGITAL SHIELD RAIL DEFENSE — DEPLOY"

# Pre-flight
try { docker --version | Out-Null; Write-Step "Docker available" }
catch { Write-Host "[ERR] Docker not found" -ForegroundColor Red; exit 1 }

# Env file
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Warn ".env created from template — EDIT WITH PRODUCTION CREDENTIALS"
}
Write-Step "Environment configured"

# Directories
@("logs\nginx", "logs\audit", "output\uploads") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}
Write-Step "Directories created"

# Build frontend
Write-Header "BUILDING FRONTEND"
if (Test-Path frontend\node_modules) {
    Push-Location frontend
    npm run build
    Pop-Location
    Write-Step "Frontend built"
} else {
    Write-Warn "Frontend node_modules missing — Docker will build it"
}

# Build Docker
Write-Header "BUILDING CONTAINERS"
$buildArgs = if ($Rebuild) { "--no-cache" } else { "" }
docker compose build $buildArgs
Write-Step "Docker images built"

# Stop existing
docker compose down --remove-orphans 2>$null

# Start
Write-Header "STARTING SERVICES"
if ($Monitoring) {
    docker compose --profile monitoring up -d
    Write-Step "All services started (with monitoring)"
} else {
    docker compose up -d
    Write-Step "Core services started"
}

# Health check
Write-Host "`nWaiting for API health..."
for ($i = 1; $i -le 30; $i++) {
    try {
        $r = Invoke-RestMethod -Uri http://localhost:8000/health -TimeoutSec 2
        if ($r.status -eq "healthy") {
            Write-Step "API is healthy!"
            break
        }
    } catch {}
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline
}

# Status
Write-Header "DEPLOYMENT COMPLETE"
Write-Host "Services:" -ForegroundColor Green
Write-Host "  Dashboard:   http://localhost:80"
Write-Host "  API:         http://localhost:8000"
Write-Host "  API Docs:    http://localhost:8000/docs"
Write-Host "  Health:      http://localhost:8000/health"

if ($Monitoring) {
    Write-Host "  Prometheus:  http://localhost:9090"
    Write-Host "  Grafana:     http://localhost:3001 (admin / digital_shield_2026)"
}

Write-Host "`nCommands:" -ForegroundColor Green
Write-Host "  Logs:    docker compose logs -f api"
Write-Host "  Status:  docker compose ps"
Write-Host "  Stop:    docker compose down"
Write-Host ""

docker compose ps
