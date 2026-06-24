/**
 * Header — clean enterprise top navigation bar
 */
import { useState, useEffect } from "react";
import { RefreshCw, Wifi, WifiOff, Shield, Upload } from "lucide-react";

export default function Header({ wsStatus, alertCount, onRefresh, onUploadClick }) {
  const isConnected = wsStatus === "connected";

  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <header style={{
      background: "var(--bg-header)",
      borderBottom: "1px solid var(--border)",
      padding: "0 24px",
      height: 56,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      flexShrink: 0,
      zIndex: 50,
    }}>
      {/* Left — brand */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 30, height: 30, borderRadius: 8,
          background: "#1E3A5F",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Shield size={15} color="#FFFFFF" />
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1 }}>
            Digital Shield
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1, marginTop: 2 }}>
            Rail Defence Intelligence
          </div>
        </div>
      </div>

      {/* Center — station + live indicator */}
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        {/* Live status */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%",
            display: "block",
            background: "#22C55E",
            animation: "liveDot 2s ease-in-out infinite",
          }} />
          <span style={{ fontSize: 12, fontWeight: 500, color: "#22C55E" }}>Live</span>
        </div>

        <div style={{ width: 1, height: 16, background: "var(--border)" }} />

        {/* Station */}
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary)" }}>
          Secunderabad Jn · SC
        </span>

        <div style={{ width: 1, height: 16, background: "var(--border)" }} />

        {/* Alert count */}
        {alertCount > 0 && (
          <span style={{
            fontSize: 12, fontWeight: 600,
            background: "#EFF6FF",
            color: "#2563EB",
            border: "1px solid #BFD9FE",
            padding: "2px 10px", borderRadius: 20,
          }}>
            {alertCount} alerts
          </span>
        )}
      </div>

      {/* Right — Upload + WS status + clock + refresh */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>

        {/* Upload CCTV button */}
        <button
          id="upload-cctv-btn"
          onClick={onUploadClick}
          style={{
            display: "flex", alignItems: "center", gap: 6,
            fontSize: 12, fontWeight: 600,
            background: "#1E3A5F", color: "#FFFFFF",
            border: "none",
            borderRadius: 8, padding: "6px 14px", cursor: "pointer",
            transition: "background 0.15s, box-shadow 0.15s",
          }}
          onMouseEnter={e => e.currentTarget.style.background = "#2563EB"}
          onMouseLeave={e => e.currentTarget.style.background = "#1E3A5F"}
        >
          <Upload size={12} />
          Upload CCTV
        </button>

        <div style={{ width: 1, height: 16, background: "var(--border)" }} />

        {/* WebSocket */}
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          {isConnected
            ? <Wifi size={14} color="#22C55E" />
            : <WifiOff size={14} color="var(--text-muted)" />
          }
          <span style={{ fontSize: 12, color: isConnected ? "#22C55E" : "var(--text-muted)" }}>
            {isConnected ? "Live stream" : "Reconnecting…"}
          </span>
        </div>

        <div style={{ width: 1, height: 16, background: "var(--border)" }} />

        {/* Live Clock */}
        <span style={{ fontSize: 12, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
          {now.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}
        </span>

        {/* Refresh */}
        <button
          onClick={onRefresh}
          style={{
            display: "flex", alignItems: "center", gap: 6,
            fontSize: 12, fontWeight: 500,
            background: "var(--bg-base)", color: "var(--text-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 8, padding: "5px 12px", cursor: "pointer",
            transition: "background 0.15s, box-shadow 0.15s",
          }}
          onMouseEnter={e => e.currentTarget.style.background = "#EEEBE4"}
          onMouseLeave={e => e.currentTarget.style.background = "var(--bg-base)"}
        >
          <RefreshCw size={12} />
          Refresh
        </button>
      </div>
    </header>
  );
}
