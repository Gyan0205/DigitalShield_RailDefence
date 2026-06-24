/**
 * ToastNotification — clean enterprise alert toasts
 */
import { motion, AnimatePresence } from "framer-motion";
import { AlertCircle, X } from "lucide-react";
import { SEV_COLOR, SEV_BG, SEV_BORDER, getConfidence, getAnomalyType, getPlatform, getTrainNumber } from "../utils";

export default function ToastNotification({ toasts, onDismiss }) {
  return (
    <div style={{
      position: "fixed", top: 68, right: 16,
      zIndex: 200, display: "flex", flexDirection: "column", gap: 8,
      maxWidth: 320, pointerEvents: "none",
    }}>
      <AnimatePresence>
        {toasts.map(toast => {
          const sev   = (toast.severity ?? "LOW").toUpperCase();
          const color  = SEV_COLOR[sev]  ?? "#94A3B8";
          const bg     = SEV_BG[sev]    ?? "#F8FAFC";
          const border = SEV_BORDER[sev] ?? "#E2E8F0";
          const conf   = getConfidence(toast);
          const atype  = getAnomalyType(toast);

          return (
            <motion.div
              key={toast._toastId}
              initial={{ opacity: 0, y: -12, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.96 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              style={{
                background: "#FFFFFF",
                border: `1px solid ${border}`,
                borderLeft: `3px solid ${color}`,
                borderRadius: 10,
                padding: "12px 14px",
                boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
                pointerEvents: "all",
              }}
            >
              <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                <AlertCircle size={15} color={color} style={{ flexShrink: 0, marginTop: 1 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                    <span style={{
                      fontSize: 10, fontWeight: 700, letterSpacing: "0.05em",
                      color, background: bg, border: `1px solid ${border}`,
                      padding: "1px 6px", borderRadius: 4,
                    }}>
                      {sev}
                    </span>
                    <span style={{
                      fontSize: 10, color: "#22C55E", fontWeight: 500,
                      display: "flex", alignItems: "center", gap: 3,
                    }}>
                      <span style={{
                        width: 5, height: 5, borderRadius: "50%", background: "#22C55E",
                        display: "block", animation: "liveDot 2s ease-in-out infinite",
                      }} />
                      Live
                    </span>
                    <span style={{ fontSize: 12, fontWeight: 700, color, marginLeft: "auto" }}>
                      {conf.toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ fontSize: 13, color: "var(--text-primary)", fontWeight: 600, marginBottom: 2 }}>
                    {atype}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                    Platform {getPlatform(toast)} · Train {getTrainNumber(toast)}
                  </div>
                </div>
                <button
                  onClick={() => onDismiss(toast._toastId)}
                  style={{
                    background: "none", border: "none", cursor: "pointer",
                    color: "var(--text-muted)", padding: 2, pointerEvents: "all",
                  }}
                >
                  <X size={12} />
                </button>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
