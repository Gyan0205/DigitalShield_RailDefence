#!/bin/bash
# ==============================================================
# Digital Shield Rail Defense — Production Deploy Script
# ==============================================================
# Usage:
#   chmod +x deploy/deploy.sh
#   ./deploy/deploy.sh                  # Full deploy
#   ./deploy/deploy.sh --monitoring     # With Prometheus + Grafana
#   ./deploy/deploy.sh --rebuild        # Force rebuild
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "\n${CYAN}═══════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════${NC}\n"
}

print_step() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Parse args
MONITORING=false
REBUILD=false
for arg in "$@"; do
    case $arg in
        --monitoring) MONITORING=true ;;
        --rebuild) REBUILD=true ;;
    esac
done

print_header "DIGITAL SHIELD RAIL DEFENSE — DEPLOY"

# ── Pre-flight checks ──────────────────────────────────────
echo "Pre-flight checks..."
command -v docker >/dev/null 2>&1 || { print_error "Docker not found"; exit 1; }
command -v docker compose >/dev/null 2>&1 || { print_error "Docker Compose not found"; exit 1; }
print_step "Docker available"

# ── Environment file ───────────────────────────────────────
if [ ! -f .env ]; then
    print_warn ".env not found, creating from template..."
    cp .env.example .env
    print_warn "EDIT .env WITH PRODUCTION CREDENTIALS BEFORE GOING LIVE"
fi
print_step "Environment configured"

# ── Create directories ─────────────────────────────────────
mkdir -p logs/nginx logs/audit output/uploads
print_step "Directories created"

# ── Build frontend ─────────────────────────────────────────
print_header "BUILDING FRONTEND"
if [ -d frontend/node_modules ]; then
    cd frontend
    npm run build
    cd ..
    print_step "Frontend built"
else
    print_warn "Frontend node_modules missing — Docker will build it"
fi

# ── Build Docker images ────────────────────────────────────
print_header "BUILDING CONTAINERS"
BUILD_ARGS=""
if [ "$REBUILD" = true ]; then
    BUILD_ARGS="--no-cache"
fi
docker compose build $BUILD_ARGS
print_step "Docker images built"

# ── Stop existing containers ───────────────────────────────
echo "Stopping existing containers..."
docker compose down --remove-orphans 2>/dev/null || true

# ── Start core services ────────────────────────────────────
print_header "STARTING SERVICES"

if [ "$MONITORING" = true ]; then
    docker compose --profile monitoring up -d
    print_step "All services started (with monitoring)"
else
    docker compose up -d
    print_step "Core services started"
fi

# ── Wait for health ────────────────────────────────────────
echo ""
echo "Waiting for services to be healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        print_step "API is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "API health check failed after 30s"
        docker compose logs api --tail 50
        exit 1
    fi
    sleep 1
    echo -n "."
done

# ── Verify database ────────────────────────────────────────
echo ""
echo "Verifying database..."
docker compose exec -T db psql -U "${DB_USER:-ds_admin}" -d "${DB_NAME:-digital_shield}" \
    -c "SELECT COUNT(*) as tables FROM information_schema.tables WHERE table_schema='public';" \
    2>/dev/null && print_step "Database verified" || print_warn "DB check skipped"

# ── Print status ────────────────────────────────────────────
print_header "DEPLOYMENT COMPLETE"

echo -e "${GREEN}Services:${NC}"
echo "  Dashboard:   http://localhost:80"
echo "  API:         http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  Health:      http://localhost:8000/health"

if [ "$MONITORING" = true ]; then
    echo "  Prometheus:  http://localhost:9090"
    echo "  Grafana:     http://localhost:3001 (admin / digital_shield_2026)"
fi

echo ""
echo -e "${GREEN}Commands:${NC}"
echo "  Logs:    docker compose logs -f api"
echo "  Status:  docker compose ps"
echo "  Stop:    docker compose down"
echo "  Rebuild: ./deploy/deploy.sh --rebuild"
echo ""

# Show container status
docker compose ps
