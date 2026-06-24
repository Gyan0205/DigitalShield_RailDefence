# ----------- Build stage -----------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ----------- Production stage -----------
FROM python:3.11-slim

# System deps (OpenCV + PostgreSQL client + curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY dataset/metadata/ ./dataset/metadata/
COPY scripts/ ./scripts/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create runtime dirs
RUN mkdir -p output/uploads logs/audit dataset/models/weights

# Non-root user
RUN useradd -m dsuser && chown -R dsuser:dsuser /app
USER dsuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--access-log"]
