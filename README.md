# 🛡️ Digital Shield Rail Defense

**AI-powered Indian railway anti-trafficking surveillance platform**

An intelligent surveillance system that detects potential human trafficking at railway stations by fusing CCTV anomaly detection, railway metadata, and train schedules to generate actionable intelligence alerts for Railway Protection Force (RPF) officers.

[![Production Readiness](https://img.shields.io/badge/Audit-151%2F151%20passed-brightgreen)](#audit-results)
[![API Endpoints](https://img.shields.io/badge/API-40%20endpoints-blue)](#api-reference)
[![Docker](https://img.shields.io/badge/Docker-6%20services-blue)](#docker-deployment)

---

## 🚀 Quick Start

### Option 1: Development (2 terminals)

```bash
# Backend
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # Linux/Mac
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

**Dashboard**: http://localhost:5173 | **API Docs**: http://localhost:8000/docs

### Option 2: Docker (full stack)

```bash
copy .env.example .env             # Windows
# cp .env.example .env             # Linux/Mac
docker compose up -d
```

**Dashboard**: http://localhost | **API**: http://localhost:8000/docs | **Grafana**: http://localhost:3001

---

## 🏗️ Architecture

```
┌──────────────────── NGINX :80 ────────────────────┐
│         React Dashboard (Vite/Tailwind)            │
├────────────────────────────────────────────────────┤
│  FastAPI Backend :8000                             │
│  ├─ 6 Core API Endpoints                          │
│  ├─ Fusion Engine (4-source Dempster-Shafer)       │
│  ├─ XAI Pipeline (10-stage explainable AI)         │
│  ├─ Train Intelligence (300 trains, 49 stations)   │
│  └─ WebSocket /ws/alerts (real-time push)          │
├────────────────────────────────────────────────────┤
│  PostgreSQL 15  │  Redis 7  │  Prometheus/Grafana  │
└────────────────────────────────────────────────────┘
```

### Intelligence Workflow

```
CCTV Camera Event → 4-Source Fusion → XAI Explanation → Intelligence Alert
  │
  ├── Source 1: CCTV Anomaly Score (confidence)
  ├── Source 2: Camera Metadata (station/platform/zone)
  ├── Source 3: Train Intelligence (schedule correlation)
  └── Source 4: Coach Estimation (zone → coach mapping)
  │
  ▼
  Dempster-Shafer Evidence Fusion → Fused Confidence → Alert
```

---

## 📡 API Reference

### Core 6 Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/upload-video` | officer | Upload CCTV video |
| `POST` | `/api/detect-anomaly` | officer | Trigger fusion + alert |
| `GET` | `/api/metadata` | viewer | Railway metadata |
| `GET` | `/api/train-lookup` | viewer | Train schedules |
| `GET` | `/api/coach-estimation` | viewer | Coach/bogie mapping |
| `GET` | `/api/alerts` | officer | Intelligence alerts |

### System Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health + DB status |
| `GET /metrics` | Prometheus metrics |
| `GET /api/system/info` | All 40 routes |
| `WS /ws/alerts` | Real-time alert stream |

### Extended Routers (34 routes)

- **Schedule** (10): Station search, train lookup, conflict detection, platform occupancy
- **Coach** (9): Zone estimation, layout, fare tiers, OCR parsing
- **XAI** (5): Alert generation, scenario explanation, audit log
- **Fusion** (6): Event fusion, batch processing, architecture info

### Example: Trigger an Alert

```bash
curl -X POST http://localhost:8000/api/detect-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "CAM_SC_P03_B",
    "timestamp": "2026-05-12T14:30:00",
    "anomaly_type": "suspicious_escort",
    "anomaly_confidence": 0.82,
    "person_count": 2,
    "platform": 3
  }'
```

**Response**: `{ "alert_id": "FUSED_xxxx", "severity": "CRITICAL", "fused_confidence": 0.73, ... }`

---

## 🎯 Demo Mode

The system generates synthetic data automatically — no real infrastructure needed.

| Component | Demo Behavior |
|-----------|--------------|
| Trains | 300 algorithmically generated Indian trains |
| CCTV | Simulated bounding boxes + real API metadata |
| Alerts | Generated on-demand via `/api/detect-anomaly` |
| ML Models | Rule-based fallback when torch/ultralytics missing |

### Demo Walkthrough

1. Start backend: `uvicorn backend.main:app --reload --port 8000`
2. Open Swagger: http://localhost:8000/docs
3. POST `/api/detect-anomaly` with suspicious_escort payload
4. GET `/api/alerts` → see CRITICAL fused alert
6. Start frontend: `cd frontend && npm run dev`
7. Open dashboard: http://localhost:5173 → all panels show LIVE data

---

## 🔐 Security

- **RBAC**: 3-tier role hierarchy (admin > officer > viewer)
- **API Keys**: Loaded from environment variables, not source code
- **Middleware**: Security headers (HSTS, XSS, CSP), rate limiting
- **Dev mode**: Unauthenticated access allowed; blocked in production

### Environment Variables

```bash
# .env
ENVIRONMENT=production
DS_API_KEY_ADMIN=your-secure-admin-key
DS_API_KEY_OFFICER=your-secure-officer-key
DS_API_KEY_VIEWER=your-secure-viewer-key
DB_PASSWORD=your-secure-db-password
SECRET_KEY=your-random-secret
```

---

## 🐳 Docker Deployment

```bash
copy .env.example .env    # Edit with production values
docker compose up -d      # Start 6 services
docker compose ps         # Check status
docker compose logs -f api  # View logs
```

| Service | Port | Image |
|---------|------|-------|
| API | 8000 | Custom (FastAPI + React) |
| PostgreSQL | 5432 | postgres:15 |
| Redis | 6379 | redis:7-alpine |
| Nginx | 80 | nginx:alpine |
| Prometheus | 9090 | prom/prometheus |
| Grafana | 3001 | grafana/grafana |

---

## 🤖 ML Pipeline

| Model | Architecture | Purpose |
|-------|-------------|---------|
| Person Detector | YOLOv8n | Detect people in CCTV frames |
| Person Tracker | DeepSORT | Track individuals across frames |
| Pose Estimator | YOLOv8n-pose | 17-keypoint skeleton estimation |
| Behavior Analyzer | Rule-based | Pairwise interaction analysis |
| Anomaly Classifier | Bi-LSTM + Attention | 8-class temporal classification |

### Run ML Pipeline

```bash
# Full pipeline (works without GPU in simulation mode)
python ml/training/run_pipeline.py --epochs 5

# Inference on video (requires ultralytics)
python -m ml.inference.pipeline --input video.mp4 --output output/
```

---

## 🧪 Verification

```bash
python scripts/final_audit.py           # 151 comprehensive checks
python scripts/test_deployment.py       # 64 infrastructure checks
python scripts/test_unified_backend.py  # 53 route verification
```

### Audit Results

```
151/151 checks passed
0 failures
1 warning (ultralytics not installed — expected)
Production Readiness: 100%
```

---

## 📊 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, SQLAlchemy, Uvicorn, Pydantic |
| Frontend | React 18, Vite, Tailwind CSS v4, Framer Motion |
| ML | YOLOv8, DeepSORT, PyTorch (LSTM/GRU), OpenCV |
| Database | PostgreSQL 15, Redis 7 |
| Deploy | Docker, Nginx, Prometheus, Grafana |
| Auth | API Key RBAC, Security Headers, Rate Limiting |

---

## 📜 License

Built for the Digital Shield Rail Defense initiative — AI-powered railway surveillance for India.
