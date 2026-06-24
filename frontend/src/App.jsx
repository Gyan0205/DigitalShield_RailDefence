/**
 * Digital Shield Rail Defense — Main Application
 * ================================================
 * Three-column intelligence dashboard:
 *   Left  — CCTV feeds + System Health
 *   Center — Live Alerts list
 *   Right  — Analytics charts
 *
 * Data sources:
 *   GET /api/alerts  — polled every 30s
 *   GET /api/health  — polled every 15s
 *   WS  /ws/alerts   — real-time push
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { RefreshCw, Filter, ChevronDown, UploadCloud } from "lucide-react";

import { fetchAlerts, createAlertSocket, uploadVideoForAnalysis, pollJobStatus } from "./api";
import Header          from "./components/Header";
import StatsGrid       from "./components/StatsGrid";
import AlertCard       from "./components/AlertCard";
import AlertDetailPanel from "./components/AlertDetailPanel";
import TopRiskTrains   from "./components/TopRiskTrains";
import AnalyticsPanel  from "./components/AnalyticsPanel";
import CCTVPanel       from "./components/CCTVPanel";
import ToastNotification from "./components/ToastNotification";

const POLL_ALERTS = 30000;

export default function App() {
  const [alerts,      setAlerts]     = useState([]);
  const [wsStatus,    setWsStatus]   = useState("connecting");
  const [selected,    setSelected]   = useState(null);
  const [toasts,      setToasts]     = useState([]);
  const [loading,     setLoading]    = useState(true);

  const [filter,      setFilter]     = useState("ALL"); // ALL | CRITICAL | HIGH | MEDIUM | LOW
  const [newIds,      setNewIds]     = useState(new Set());

  // Upload modal state
  const [uploadOpen,  setUploadOpen] = useState(false);
  const [uploadFile,  setUploadFile] = useState(null);
  const [uploadBusy,  setUploadBusy] = useState(false);
  const [uploadJob,   setUploadJob]  = useState(null);   // { job_id, status, stages, alert, error }
  const [uploadErr,   setUploadErr]  = useState(null);

  const wsRef       = useRef(null);
  const toastTimers = useRef({});

  // ── Toast helpers ───────────────────────────────────────────────────────
  const addToast = useCallback((alert) => {
    const id = `toast_${Date.now()}_${Math.random()}`;
    const toast = { ...alert, _toastId: id };
    setToasts(prev => [toast, ...prev].slice(0, 5));
    toastTimers.current[id] = setTimeout(() => {
      setToasts(prev => prev.filter(t => t._toastId !== id));
      delete toastTimers.current[id];
    }, 6000);
  }, []);

  const dismissToast = useCallback((id) => {
    clearTimeout(toastTimers.current[id]);
    delete toastTimers.current[id];
    setToasts(prev => prev.filter(t => t._toastId !== id));
  }, []);

  // ── Fetch alerts ────────────────────────────────────────────────────────
  const loadAlerts = useCallback(async () => {
    try {
      const data = await fetchAlerts({ limit: 50 });
      setAlerts((data.alerts ?? []).sort((a, b) => {
        const ta = new Date(a.timestamp ?? a.created_at ?? 0).getTime();
        const tb = new Date(b.timestamp ?? b.created_at ?? 0).getTime();
        return tb - ta;
      }));
    } catch (err) {
      console.error("Failed to load alerts:", err);
    } finally {
      setLoading(false);
    }
  }, []);



  // ── WebSocket ────────────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    wsRef.current = createAlertSocket(
      (msg) => {
        if (!msg || !msg.alert_id) return;
        const newAlert = { ...msg, _isNew: true };
        setAlerts(prev => {
          if (prev.some(a => a.alert_id === msg.alert_id)) return prev;
          return [newAlert, ...prev].slice(0, 100);
        });
        setNewIds(prev => new Set([...prev, msg.alert_id]));
        addToast(newAlert);
        // Remove "new" badge after 8s
        setTimeout(() => setNewIds(prev => {
          const next = new Set(prev);
          next.delete(msg.alert_id);
          return next;
        }), 8000);
      },
      () => setWsStatus("connected"),
      () => {
        setWsStatus("disconnected");
        // Reconnect after 5s
        setTimeout(connectWS, 5000);
      },
      () => setWsStatus("error"),
    );
  }, [addToast]);

  // ── Lifecycle ────────────────────────────────────────────────────────────
  useEffect(() => {
    loadAlerts();
    connectWS();

    const t1 = setInterval(loadAlerts, POLL_ALERTS);

    return () => {
      clearInterval(t1);
      wsRef.current?.close();
      Object.values(toastTimers.current).forEach(clearTimeout);
    };
  }, [loadAlerts, connectWS]);

  // ── Job poller — watches active upload job ───────────────────────────────
  useEffect(() => {
    if (!uploadJob?.job_id || uploadJob?.status === "complete" || uploadJob?.status === "error") return;
    const t = setInterval(async () => {
      try {
        const data = await pollJobStatus(uploadJob.job_id);
        setUploadJob(data);
        if (data.status === "complete" && data.alert) {
          // Inject the alert into the live list
          const newAlert = { ...data.alert, _isNew: true };
          setAlerts(prev => {
            if (prev.some(a => a.alert_id === data.alert.alert_id)) return prev;
            return [newAlert, ...prev].slice(0, 100);
          });
          clearInterval(t);
        } else if (data.status === "error") {
          setUploadErr(data.error || "Pipeline failed");
          clearInterval(t);
        }
      } catch (e) {
        console.error("Job poll error:", e);
      }
    }, 2000);
    return () => clearInterval(t);
  }, [uploadJob?.job_id, uploadJob?.status, addToast]);

  // ── Upload handler ───────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploadBusy(true);
    setUploadErr(null);
    setUploadJob(null);
    try {
      const result = await uploadVideoForAnalysis(uploadFile, {
        cameraId: "CAM_SC_P03_B",
        stationCode: "SC",
      });
      setUploadJob({ job_id: result.job_id, status: result.status, stages: [], alert: null, error: null });
    } catch (e) {
      setUploadErr(e?.response?.data?.detail || e.message || "Upload failed");
    } finally {
      setUploadBusy(false);
    }
  };

  // ── Filtered alerts ──────────────────────────────────────────────────────
  const filtered = filter === "ALL"
    ? alerts
    : alerts.filter(a => (a.severity ?? "").toUpperCase() === filter);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden", background: "var(--bg-base)" }}>
      {/* Header */}
      <Header
        wsStatus={wsStatus}
        alertCount={alerts.length}
        onRefresh={() => loadAlerts()}
        onUploadClick={() => { setUploadOpen(true); setUploadJob(null); setUploadErr(null); setUploadFile(null); }}
      />

      {/* Main 3-column layout */}
      <div style={{
        flex: 1, display: "grid",
        gridTemplateColumns: "260px 1fr 260px",
        gap: 0, overflow: "hidden",
      }}>

        {/* ── LEFT COLUMN ── */}
        <aside style={{
          background: "var(--bg-sidebar)",
          borderRight: "1px solid var(--border)",
          display: "flex", flexDirection: "column",
          overflowY: "auto", padding: "18px 14px", gap: 20,
        }}>
          <SideSection title="Surveillance feeds">
            <CCTVPanel alerts={alerts} />
          </SideSection>
          <SideSection title="Top risk trains">
            <TopRiskTrains alerts={alerts} />
          </SideSection>
        </aside>

        {/* ── CENTER COLUMN ── */}
        <main style={{ display: "flex", flexDirection: "column", overflow: "hidden", background: "var(--bg-base)" }}>
          {/* Stats KPIs */}
          <div style={{ padding: "16px 20px 12px", borderBottom: "1px solid var(--border)", background: "var(--bg-secondary)" }}>
            <StatsGrid alerts={alerts} />
          </div>

          {/* Filter bar */}
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "10px 20px", borderBottom: "1px solid var(--border)",
            background: "var(--bg-secondary)",
          }}>
            <Filter size={13} color="var(--text-muted)" />
            <span style={{ fontSize: 12, color: "var(--text-muted)", marginRight: 4 }}>Filter</span>
            {["All", "Critical", "High", "Medium", "Low"].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f.toUpperCase() === "ALL" ? "ALL" : f.toUpperCase())}
                style={{
                  fontSize: 12, fontWeight: 500,
                  padding: "4px 12px", borderRadius: 6, cursor: "pointer",
                  border: `1px solid ${filter === (f === "All" ? "ALL" : f.toUpperCase()) ? "#2563EB" : "var(--border)"}`,
                  background: filter === (f === "All" ? "ALL" : f.toUpperCase()) ? "#EFF6FF" : "#FFFFFF",
                  color: filter === (f === "All" ? "ALL" : f.toUpperCase()) ? "#2563EB" : "var(--text-secondary)",
                  transition: "all 0.15s",
                }}
              >
                {f}
              </button>
            ))}
            <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--text-muted)" }}>
              {filtered.length} alert{filtered.length !== 1 ? "s" : ""}
            </span>
          </div>

          {/* Alerts list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 10 }}>
            {loading ? (
              <LoadingState />
            ) : filtered.length === 0 ? (
              <EmptyState filter={filter} />
            ) : (
              <AnimatePresence initial={false}>
                {filtered.map(alert => (
                  <AlertCard
                    key={alert.alert_id ?? `${alert.timestamp}_${Math.random()}`}
                    alert={alert}
                    isNew={newIds.has(alert.alert_id)}
                    onClick={setSelected}
                  />
                ))}
              </AnimatePresence>
            )}
          </div>
        </main>

        {/* ── RIGHT COLUMN — Analytics ── */}
        <aside style={{
          background: "var(--bg-sidebar)",
          borderLeft: "1px solid var(--border)",
          overflowY: "auto", padding: "18px 14px",
        }}>
          <div className="section-label" style={{ marginBottom: 12 }}>Analytics</div>
          <AnalyticsPanel alerts={alerts} />
        </aside>
      </div>

      {/* Alert detail panel */}
      <AlertDetailPanel alert={selected} onClose={() => setSelected(null)} />

      {/* Toast notifications */}
      <ToastNotification toasts={toasts} onDismiss={dismissToast} />

      {/* Upload Modal */}
      {uploadOpen && (
        <UploadModal
          onClose={() => !uploadBusy && setUploadOpen(false)}
          file={uploadFile}
          onFileChange={e => setUploadFile(e.target.files?.[0])}
          onUpload={handleUpload}
          busy={uploadBusy}
          job={uploadJob}
          error={uploadErr}
        />
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function SideSection({ title, children }) {
  return (
    <div>
      <div className="section-label">{title}</div>
      {children}
    </div>
  );
}

function QuickStat({ label, value }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "5px 0", borderBottom: "1px solid rgba(255,255,255,0.04)",
    }}>
      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: "#94a3b8", fontFamily: "monospace" }}>
        {String(value)}
      </span>
    </div>
  );
}

function LoadingState() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {[1, 2, 3, 4].map(i => (
        <div key={i} style={{
          height: 90, borderRadius: 12,
          background: "#FFFFFF",
          border: "1px solid var(--border)",
          animation: "none", opacity: 0.6,
        }} />
      ))}
    </div>
  );
}

function EmptyState({ filter }) {
  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      gap: 10, padding: 48, color: "var(--text-muted)",
    }}>
      <div style={{ fontSize: 36, opacity: 0.25 }}>🛡️</div>
      <div style={{ fontSize: 14, fontWeight: 500, color: "var(--text-secondary)", textAlign: "center" }}>
        {filter === "ALL"
          ? "No alerts yet. Waiting for intelligence events…"
          : `No ${filter.toLowerCase()} alerts.`}
      </div>
      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
        Updates automatically via WebSocket
      </div>
    </div>
  );
}

function UploadModal({ onClose, file, onFileChange, onUpload, busy, job, error }) {
  // Stepper definition
  const PIPELINE_STAGES = [
    { key: "upload", label: "Data Injection" },
    { key: "vision", label: "Computer Vision" },
    { key: "behavior", label: "Behavioral AI" },
    { key: "fusion", label: "Context Fusion" },
  ];

  // Determine current active step based on job state
  let currentStepIndex = 0;
  if (busy || job) currentStepIndex = 1;
  if (job?.stages?.some(s => s.stage.toLowerCase().includes("pose") || s.stage.toLowerCase().includes("behavior"))) {
    currentStepIndex = 2;
  }
  if (job?.stages?.some(s => s.stage.toLowerCase().includes("metadata") || s.stage.toLowerCase().includes("fusion"))) {
    currentStepIndex = 3;
  }
  if (job?.status === "complete") {
    currentStepIndex = 4;
  }

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
      background: "rgba(255, 255, 255, 0.7)", zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center",
      backdropFilter: "blur(4px)",
    }}>
      <div style={{
        background: "#FFFFFF", borderRadius: 8, padding: 32,
        width: 520, boxShadow: "0 20px 40px rgba(0,0,0,0.08)",
        display: "flex", flexDirection: "column", gap: 24,
        border: "1px solid #E2E8F0"
      }}>
        <div style={{ fontSize: 20, fontWeight: 600, color: "#0F172A", letterSpacing: "-0.01em" }}>
          Intelligence Pipeline Injection
        </div>
        
        {/* ML Pipeline Visualizer (Stepper) */}
        {(busy || job) ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 16, background: "#F8FAFC", padding: 24, borderRadius: 6, border: "1px solid #E2E8F0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", position: "relative" }}>
              {/* Connecting line */}
              <div style={{ position: "absolute", top: 12, left: 16, right: 16, height: 2, background: "#E2E8F0", zIndex: 0 }} />
              <div style={{ position: "absolute", top: 12, left: 16, right: `calc(100% - 16px - (100% - 32px) * ${Math.min(currentStepIndex, 3) / 3})`, height: 2, background: "#3B82F6", zIndex: 0, transition: "right 0.5s ease" }} />
              
              {PIPELINE_STAGES.map((step, idx) => {
                const isCompleted = currentStepIndex > idx;
                const isActive = currentStepIndex === idx;
                return (
                  <div key={step.key} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, zIndex: 1, width: 80 }}>
                    <div style={{
                      width: 26, height: 26, borderRadius: "50%",
                      background: isCompleted ? "#3B82F6" : isActive ? "#EFF6FF" : "#FFFFFF",
                      border: `2px solid ${isCompleted || isActive ? "#3B82F6" : "#CBD5E1"}`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: isCompleted ? "#FFFFFF" : isActive ? "#3B82F6" : "#94A3B8",
                      fontSize: 12, fontWeight: 600, transition: "all 0.3s"
                    }}>
                      {isCompleted ? "✓" : (idx + 1)}
                    </div>
                    <div style={{
                      fontSize: 11, fontWeight: isActive || isCompleted ? 600 : 500,
                      color: isActive ? "#1E293B" : isCompleted ? "#64748B" : "#94A3B8",
                      textAlign: "center", lineHeight: 1.2
                    }}>
                      {step.label}
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Live Status Log */}
            <div style={{ marginTop: 8, background: "#FFFFFF", border: "1px solid #E2E8F0", borderRadius: 4, padding: 12, maxHeight: 120, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
              {job?.stages?.map((s, i) => (
                <div key={i} style={{ fontSize: 12, color: "#475569", display: "flex", gap: 8, fontFamily: "monospace" }}>
                  <span style={{ color: "#10B981" }}>[{s.stage}]</span>
                  <span>{s.detail}</span>
                </div>
              ))}
              {job?.status === "processing" && (
                <div style={{ fontSize: 12, color: "#64748B", display: "flex", alignItems: "center", gap: 6, fontFamily: "monospace" }}>
                  <RefreshCw size={12} style={{ animation: "spin 2s linear infinite" }} /> Processing...
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Drag & Drop Upload Zone */
          <div style={{ position: "relative" }}>
            <input 
              type="file" 
              accept="video/mp4,video/x-m4v,video/*" 
              onChange={onFileChange} 
              disabled={busy || job} 
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", opacity: 0, cursor: "pointer", zIndex: 10 }}
            />
            <div style={{
              border: "2px dashed #CBD5E1", borderRadius: 6, padding: "40px 24px",
              display: "flex", flexDirection: "column", alignItems: "center", gap: 12,
              background: "#F8FAFC", transition: "all 0.2s"
            }}>
              <div style={{ width: 48, height: 48, borderRadius: "50%", background: "#EFF6FF", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <UploadCloud size={24} color="#3B82F6" />
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: "#1E293B" }}>Drag and drop video feed</div>
                <div style={{ fontSize: 12, color: "#64748B", marginTop: 4 }}>MP4, AVI, or MKV up to 500MB</div>
              </div>
              {file && (
                <div style={{ marginTop: 8, padding: "6px 12px", background: "#FFFFFF", border: "1px solid #E2E8F0", borderRadius: 4, fontSize: 12, color: "#3B82F6", fontWeight: 500 }}>
                  Selected: {file.name}
                </div>
              )}
            </div>
          </div>
        )}
        
        {error && (
          <div style={{ color: "#B42318", fontSize: 13, background: "#FEF2F2", padding: "10px 14px", borderRadius: 4, border: "1px solid #FBC9C4" }}>
            {error}
          </div>
        )}
        
        {/* Footer Actions */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 12, paddingTop: 8 }}>
          <button onClick={onClose} disabled={busy} style={{
            padding: "8px 20px", borderRadius: 4, border: "1px solid #CBD5E1",
            background: "#FFFFFF", color: "#475569", fontSize: 13, fontWeight: 600,
            cursor: busy ? "not-allowed" : "pointer"
          }}>
            Cancel
          </button>
          
          {!job && (
            <button onClick={onUpload} disabled={!file || busy} style={{
              padding: "8px 24px", borderRadius: 4, border: "none",
              background: (!file || busy) ? "#94A3B8" : "#0F172A", color: "#FFFFFF", fontSize: 13, fontWeight: 600,
              cursor: (!file || busy) ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", gap: 8
            }}>
              {busy && <RefreshCw size={14} style={{ animation: "spin 2s linear infinite" }} />}
              Initialize Pipeline
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
