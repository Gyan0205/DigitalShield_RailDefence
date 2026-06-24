/**
 * CCTVPanel — clean enterprise camera list
 */
import { useState, useEffect } from "react";
import { Camera, Radio } from "lucide-react";

const FEEDS = [
  { id: "CAM_SC_P01_A", label: "Platform 1 · Gate A", platform: 1, status: "live" },
  { id: "CAM_SC_P03_B", label: "Platform 3 · Gate B", platform: 3, status: "live" },
  { id: "CAM_SC_P07_A", label: "Platform 7 · Gate A", platform: 7, status: "live" },
  { id: "CAM_SC_P09_C", label: "Platform 9 · Gate C", platform: 9, status: "idle" },
];

export default function CCTVPanel({ alerts }) {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const topAlert = alerts.find(a => (a.severity ?? "").toUpperCase() === "CRITICAL") ?? alerts[0];
  const topPlatform = topAlert ? String(topAlert.platform ?? topAlert.platform_number ?? "") : "";

  const isActive = (feed) => topPlatform && String(feed.platform) === topPlatform;

  const ts = time.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 2 }}>
        <span className="section-label" style={{ marginBottom: 0 }}>Surveillance feeds</span>
        <span style={{
          fontSize: 10, color: "var(--text-muted)",
          fontFamily: "monospace", letterSpacing: "0.04em",
        }}>{ts}</span>
      </div>

      {/* Camera rows */}
      {FEEDS.map(feed => {
        const active = isActive(feed);
        const isLive = feed.status === "live";
        return (
          <div key={feed.id} style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "9px 11px", borderRadius: 9,
            background: active ? "#FEF2F0" : "#FFFFFF",
            border: `1px solid ${active ? "#FBC9C4" : "var(--border)"}`,
            transition: "background 0.2s",
          }}>
            {/* Camera icon */}
            <div style={{
              width: 32, height: 32, borderRadius: 7, flexShrink: 0,
              background: active ? "#FEF2F0" : "var(--bg-base)",
              border: `1px solid ${active ? "#FBC9C4" : "var(--border)"}`,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Camera size={13} color={active ? "#B42318" : "var(--text-muted)"} />
            </div>

            {/* Label */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                fontSize: 12, fontWeight: 500,
                color: active ? "#B42318" : "var(--text-primary)",
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
              }}>
                {feed.label}
              </div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1, fontFamily: "monospace" }}>
                {feed.id}
              </div>
            </div>

            {/* Status */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3 }}>
              {active && (
                <span style={{
                  fontSize: 9, fontWeight: 600, color: "#B42318",
                  background: "#FEF2F0", border: "1px solid #FBC9C4",
                  padding: "1px 6px", borderRadius: 4,
                }}>
                  Anomaly
                </span>
              )}
              <span style={{
                display: "flex", alignItems: "center", gap: 4,
                fontSize: 10, color: isLive ? "#22C55E" : "var(--text-muted)",
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: isLive ? "#22C55E" : "#D4D4D4",
                  display: "block",
                  animation: isLive ? "liveDot 2s ease-in-out infinite" : "none",
                }} />
                {isLive ? "Live" : "Idle"}
              </span>
            </div>
          </div>
        );
      })}

      {/* Last detection */}
      {topAlert && (
        <div style={{
          marginTop: 4, padding: "9px 11px", borderRadius: 9,
          background: "#FFFFFF", border: "1px solid var(--border)",
        }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-muted)", marginBottom: 4 }}>
            Last detection
          </div>
          <div style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 500 }}>
            Platform {topAlert.platform ?? topAlert.platform_number ?? "—"} · Train {topAlert.train_number ?? "—"}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
            {(topAlert.anomaly_type ?? topAlert.alert_type ?? "anomaly")
              .replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
          </div>
        </div>
      )}
    </div>
  );
}
