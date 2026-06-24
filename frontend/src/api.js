/**
 * Digital Shield Rail Defense — API Client
 * All routes proxy through Vite to FastAPI at localhost:8000
 * (configured in vite.config.js)
 */

import axios from "axios";

const api = axios.create({
  baseURL: "",          // relative — Vite proxy handles /api → :8000
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

// ── Alerts ──────────────────────────────────────────────────────────────────
export const fetchAlerts = (params = {}) =>
  api.get("/api/alerts", { params: { limit: 50, ...params } }).then(r => r.data);

// ── Health ───────────────────────────────────────────────────────────────────
export const fetchHealth = () =>
  api.get("/api/health").then(r => r.data);

// ── WebSocket factory ────────────────────────────────────────────────────────
// Vite proxy: /ws → ws://localhost:8000/ws
export const createAlertSocket = (onMessage, onOpen, onClose, onError) => {
  const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/alerts`;
  const ws = new WebSocket(WS_URL);

  let keepAliveTimer = null;

  ws.onopen = () => {
    onOpen?.();
    // Send ping every 25s to keep connection alive
    keepAliveTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send("ping");
    }, 25000);
  };

  ws.onerror = (e) => onError?.(e);

  ws.onmessage = (e) => {
    if (e.data === "pong") return; // skip keep-alive replies
    try { onMessage?.(JSON.parse(e.data)); } catch {}
  };

  ws.onclose = () => {
    clearInterval(keepAliveTimer);
    onClose?.();
  };

  return ws;
};

export default api;

// ── Video Upload ─────────────────────────────────────────────────────────────
/**
 * Upload a CCTV video file and trigger the full ML pipeline.
 * Returns { job_id, status, poll_url, ... }
 */
export const uploadVideoForAnalysis = (file, { cameraId = "CAM_SC_P03_B", stationCode = "SC" } = {}) => {
  const form = new FormData();
  form.append("file", file);
  return axios.post(
    `/api/upload-video?camera_id=${encodeURIComponent(cameraId)}&station_code=${encodeURIComponent(stationCode)}`,
    form,
    {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 30000,   // 30s for upload; pipeline runs async
    }
  ).then(r => r.data);
};

/**
 * Poll job status. Returns:
 *   { job_id, status, stages, alert, ml_report, error }
 *   status: "queued" | "processing" | "complete" | "error"
 */
export const pollJobStatus = (jobId) =>
  api.get(`/api/jobs/${jobId}`).then(r => r.data);

