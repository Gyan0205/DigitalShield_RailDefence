/**
 * AlertCard — enterprise incident card (SOC/SIEM style)
 */
import { motion } from "framer-motion";
import { Train, MapPin, Clock, Brain, ChevronRight, AlertCircle } from "lucide-react";
import {
  SEV_COLOR, SEV_BG, SEV_BORDER,
  getConfidence, getAnomalyType, getXaiSummary,
  getCoach, getPlatform, getTrainNumber, getTrainName,
  getStation, getSuspects, getTimestamp,
  fmtTime, fmtDate, shortId,
} from "../utils";

export default function AlertCard({ alert, onClick, isNew = false }) {
  const sev     = (alert.severity || "LOW").toUpperCase();
  const color   = SEV_COLOR[sev]   ?? "#94A3B8";
  const bg      = SEV_BG[sev]     ?? "#F8FAFC";
  const border  = SEV_BORDER[sev] ?? "#E2E8F0";
  const isCrit  = sev === "CRITICAL";
  const isHigh  = sev === "HIGH";

  const confPct   = getConfidence(alert);
  const atype     = getAnomalyType(alert);
  const xaiLine   = getXaiSummary(alert);
  const trainNum  = getTrainNumber(alert);
  const trainName = getTrainName(alert);
  const platform  = getPlatform(alert);
  const coach     = getCoach(alert);
  const station   = getStation(alert);
  const suspects  = getSuspects(alert);
  const ts        = getTimestamp(alert);

  return (
    <motion.div
      layout
      initial={isNew ? { opacity: 0, y: -10 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      onClick={() => onClick?.(alert)}
      className="hoverable"
      style={{
        background: isCrit ? bg : "#FFFFFF",
        border: `1px solid ${isCrit ? border : "var(--border)"}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 12,
        padding: "16px 18px",
        cursor: "pointer",
        position: "relative",
      }}
    >
      {/* ── Row 1: Severity + Type + New pill + Confidence ── */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 0 }}>
          {/* Severity badge */}
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: "0.06em",
            color, background: bg, border: `1px solid ${border}`,
            padding: "2px 8px", borderRadius: 6, flexShrink: 0,
            textTransform: "uppercase",
          }}>
            {sev}
          </span>
          {/* Type */}
          <span style={{
            fontSize: 14, fontWeight: 600, color: "var(--text-primary)",
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}>
            {atype}
          </span>
          {/* RPF pill for CRITICAL */}
          {isCrit && (
            <span style={{
              fontSize: 10, fontWeight: 600, color: "#B42318",
              background: "#FEF2F0", border: "1px solid #FBC9C4",
              padding: "1px 7px", borderRadius: 20, flexShrink: 0,
              display: "flex", alignItems: "center", gap: 4,
            }}>
              <AlertCircle size={9} /> RPF action required
            </span>
          )}
          {/* New pill */}
          {isNew && (
            <span style={{
              fontSize: 10, fontWeight: 600, color: "#2563EB",
              background: "#EFF6FF", border: "1px solid #BFD9FE",
              padding: "1px 7px", borderRadius: 20, flexShrink: 0,
            }}>
              New
            </span>
          )}
        </div>

        {/* Confidence + chevron */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 20, fontWeight: 700, color, lineHeight: 1, letterSpacing: "-0.02em" }}>
              {confPct.toFixed(1)}<span style={{ fontSize: 11, fontWeight: 500 }}>%</span>
            </div>
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>confidence</div>
          </div>
          <ChevronRight size={14} color="var(--text-muted)" />
        </div>
      </div>

      {/* ── Row 2: Metadata chips ── */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginBottom: xaiLine ? 10 : 8, alignItems: "center" }}>
        <Meta icon={<Train size={12} />} text={`Train ${trainNum}${trainName ? ` · ${trainName}` : ""}`} />
        <Meta icon={<MapPin size={12} />} text={`Platform ${platform} · Coach ${coach} · ${station}`} />
        <Meta icon={<Clock size={12} />} text={`${fmtDate(ts)} · ${fmtTime(ts)}`} />
        {suspects > 0 && (
          <Meta icon={<AlertCircle size={12} />} text={`${suspects} suspects`} color={color} />
        )}
      </div>

      {/* ── Row 3: XAI preview ── */}
      {xaiLine && (
        <div style={{
          display: "flex", alignItems: "flex-start", gap: 6,
          padding: "8px 10px",
          background: "#F7F4EE",
          border: "1px solid var(--border-subtle)",
          borderRadius: 8, marginBottom: 10,
        }}>
          <Brain size={12} color="#6366F1" style={{ flexShrink: 0, marginTop: 1 }} />
          <span style={{ fontSize: 12, color: "#4F46E5", lineHeight: 1.45, fontStyle: "italic" }}>
            {xaiLine}
          </span>
        </div>
      )}

      {/* ── Confidence bar ── */}
      <div className="conf-bar-track">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(confPct, 100)}%` }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          style={{ height: "100%", background: color, borderRadius: 2 }}
        />
      </div>

      {/* ID */}
      <div style={{ position: "absolute", bottom: 8, right: 12, fontSize: 10, color: "var(--text-dim)", fontFamily: "monospace" }}>
        {shortId(alert.alert_id ?? "")}
      </div>
    </motion.div>
  );
}

function Meta({ icon, text, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <span style={{ color: color ?? "var(--text-muted)", flexShrink: 0 }}>{icon}</span>
      <span style={{ fontSize: 12, color: color ?? "var(--text-secondary)" }}>{text}</span>
    </div>
  );
}
