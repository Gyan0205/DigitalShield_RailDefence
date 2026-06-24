/**
 * AlertDetailPanel — clean enterprise slide-panel
 */
import { motion, AnimatePresence } from "framer-motion";
import { X, Train, MapPin, Clock, Brain, Activity, ChevronRight } from "lucide-react";
import {
  SEV_COLOR, SEV_BG, SEV_BORDER,
  getConfidence, getCctvScore, getTicketScore,
  getAnomalyType, getReasoningChain,
  getCoach, getPlatform, getTrainNumber, getTrainName,
  getStation, getSuspects, getTimestamp,
  fmtTime, fmtDate,
} from "../utils";

export default function AlertDetailPanel({ alert, onClose }) {
  if (!alert) return null;

  const sev     = (alert.severity || "LOW").toUpperCase();
  const color   = SEV_COLOR[sev]   ?? "#94A3B8";
  const bg      = SEV_BG[sev]     ?? "#F8FAFC";
  const border  = SEV_BORDER[sev] ?? "#E2E8F0";

  const confPct  = getConfidence(alert);
  const cctvPct  = getCctvScore(alert);
  const tktPct   = getTicketScore(alert);
  const atype    = getAnomalyType(alert);
  const chain    = getReasoningChain(alert);
  const trainNum = getTrainNumber(alert);
  const trainName= getTrainName(alert);
  const platform = getPlatform(alert);
  const coach    = getCoach(alert);
  const station  = getStation(alert);
  const suspects = getSuspects(alert);
  const ts       = getTimestamp(alert);

  return (
    <AnimatePresence>
      <motion.div
        key="backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        style={{
          position: "fixed", inset: 0,
          background: "rgba(0,0,0,0.18)",
          zIndex: 100,
          backdropFilter: "blur(2px)",
        }}
      />

      <motion.div
        key="panel"
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", stiffness: 280, damping: 34 }}
        style={{
          position: "fixed", top: 0, right: 0, bottom: 0,
          width: "min(560px, 95vw)",
          background: "#FFFDF9",
          borderLeft: "1px solid var(--border)",
          zIndex: 101,
          display: "flex", flexDirection: "column",
          overflowY: "auto",
          boxShadow: "-8px 0 40px rgba(0,0,0,0.08)",
        }}
      >
        {/* Header */}
        <div style={{
          padding: "22px 24px 18px",
          borderBottom: "1px solid var(--border)",
          background: bg,
          position: "sticky", top: 0, zIndex: 5,
        }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{
                  fontSize: 11, fontWeight: 700, letterSpacing: "0.06em",
                  color, background: "#FFFFFF", border: `1px solid ${border}`,
                  padding: "3px 10px", borderRadius: 6,
                }}>
                  {sev}
                </span>
                <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "monospace" }}>
                  {alert.alert_id ?? "—"}
                </span>
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>
                {atype}
              </div>
              <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
                {fmtDate(ts)} · {fmtTime(ts)} · Platform {platform} · {station}
              </div>
            </div>
            <button
              onClick={onClose}
              style={{
                background: "#FFFFFF", border: "1px solid var(--border)",
                borderRadius: 8, padding: 8, cursor: "pointer",
                color: "var(--text-secondary)", marginLeft: 16,
                transition: "background 0.15s",
              }}
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Inference Frame */}
          <div style={{
            width: "100%", height: 240, background: "#1E293B",
            borderRadius: 8, overflow: "hidden", position: "relative",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "inset 0 2px 10px rgba(0,0,0,0.2)"
          }}>
            <div style={{ color: "#64748B", fontSize: 13, fontFamily: "monospace", letterSpacing: "0.05em" }}>
              CCTV INFERENCE FEED
            </div>
            
            {/* Mock Bounding Box */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, x: [0, 15, -5, 0] }}
              transition={{ delay: 0.5, duration: 5, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" }}
              style={{
                position: "absolute", top: "25%", left: "45%", width: 70, height: 140,
                border: "2px solid #EF4444", background: "rgba(239, 68, 68, 0.1)"
              }}
            >
              <div style={{ position: "absolute", top: -18, left: -2, background: "#EF4444", color: "#FFF", fontSize: 9, fontWeight: 700, padding: "2px 6px", fontFamily: "monospace" }}>
                SUSPECT (98%)
              </div>
            </motion.div>
          </div>

          {/* Fusion Scores */}
          <Section title="Fusion scores" icon={<Activity size={14} />}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 10 }}>
              <ScoreBox label="CCTV" pct={cctvPct} color="#6366F1" note="60% weight" />
              <ScoreBox label="Tickets" pct={tktPct} color="#059669" note="40% weight" />
              <ScoreBox label="Fused" pct={confPct} color={color} note="Final score" large />
            </div>
            <div style={{
              fontSize: 12, color: "var(--text-muted)", textAlign: "center",
              background: "#F7F4EE", borderRadius: 8, padding: "7px 12px",
              fontFamily: "monospace",
            }}>
              60% × {cctvPct.toFixed(1)}% + 40% × {tktPct.toFixed(1)}% ={" "}
              <span style={{ color, fontWeight: 700 }}>{confPct.toFixed(1)}%</span>
            </div>
          </Section>

          {/* Train + Location */}
          <Section title="Train resolution" icon={<Train size={14} />}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
              <Row label="Train number" value={trainNum} mono />
              <Row label="Platform"     value={platform} />
              <Row label="Train name"   value={trainName || "—"} />
              <Row label="Coach"        value={coach} mono />
              <Row label="Station"      value={station} />
              {suspects > 0 && <Row label="Suspects" value={suspects} highlight />}
            </div>
          </Section>

          {/* Reasoning Chain */}
          <Section title="AI reasoning chain" icon={<Brain size={14} />}>
            {chain.length > 0
              ? chain.map((line, i) => <ReasonLine key={i} index={i} text={String(line)} />)
              : <p style={{ fontSize: 13, color: "var(--text-muted)", fontStyle: "italic", padding: "4px 0" }}>
                  No reasoning chain available.
                </p>
            }
          </Section>

          {/* Extra metadata */}
          {alert.metadata && Object.keys(alert.metadata).length > 0 && (
            <Section title="Event metadata" icon={<MapPin size={14} />}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                {Object.entries(alert.metadata).map(([k, v]) => (
                  <Row key={k} label={k.replace(/_/g, " ")} value={String(v ?? "—")} />
                ))}
              </div>
            </Section>
          )}

          <div style={{ fontSize: 11, color: "var(--text-dim)", textAlign: "center", fontFamily: "monospace", paddingBottom: 8 }}>
            {alert.alert_id ?? "—"} · {fmtDate(ts)} {fmtTime(ts)}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

function Section({ title, icon, children }) {
  return (
    <div style={{
      background: "#FFFFFF",
      border: "1px solid var(--border)",
      borderRadius: 12,
      padding: "14px 16px",
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 6,
        marginBottom: 12, fontSize: 12, fontWeight: 600,
        color: "var(--text-secondary)", letterSpacing: "0.02em",
      }}>
        <span style={{ color: "var(--text-muted)" }}>{icon}</span>
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ label, value, highlight, mono }) {
  return (
    <div style={{
      padding: "7px 0",
      borderBottom: "1px solid var(--border-subtle)",
    }}>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 2, textTransform: "capitalize" }}>{label}</div>
      <div style={{
        fontSize: 13, fontWeight: highlight ? 700 : 500,
        color: highlight ? "#B42318" : "var(--text-primary)",
        fontFamily: mono ? "monospace" : "inherit",
      }}>
        {String(value)}
      </div>
    </div>
  );
}

function ScoreBox({ label, pct, color, note, large }) {
  const v = isNaN(parseFloat(pct)) ? 0 : parseFloat(pct);
  return (
    <div style={{
      background: "#F7F4EE",
      border: "1px solid var(--border)",
      borderRadius: 10, padding: "12px 10px", textAlign: "center",
    }}>
      <div style={{ fontSize: large ? 28 : 22, fontWeight: 700, color, lineHeight: 1 }}>
        {v.toFixed(1)}<span style={{ fontSize: 11 }}>%</span>
      </div>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginTop: 4 }}>{label}</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>{note}</div>
    </div>
  );
}

function ReasonLine({ text, index = 0 }) {
  const tagMatch = text.match(/^\[(\w+)\]/);
  const tag  = tagMatch?.[1] ?? "";
  const body = text.replace(/^\[.*?\]\s*/, "").trim();

  const TAG_COLOR = {
    CCTV: "#6366F1", META: "#3B82F6", TRAIN: "#059669",
    TICKETS: "#C97A10", FUSION: "#7C3AED",
  };
  const tagColor = TAG_COLOR[tag] ?? "var(--text-muted)";

  return (
    <motion.div 
      initial={{ opacity: 0, x: -5 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.4 + 0.2, duration: 0.3 }}
      style={{
        display: "flex", gap: 8, alignItems: "flex-start",
        padding: "6px 0", borderBottom: "1px solid var(--border-subtle)",
        fontFamily: "monospace", fontSize: 12
      }}
    >
      {tagMatch && (
        <span style={{
          fontWeight: 600, color: tagColor,
          background: `${tagColor}12`,
          padding: "1px 6px", borderRadius: 4, flexShrink: 0,
          marginTop: 1, border: `1px solid ${tagColor}25`,
        }}>
          {tag}
        </span>
      )}
      {!tagMatch && <ChevronRight size={12} color="var(--text-dim)" style={{ flexShrink: 0, marginTop: 2 }} />}
      <div style={{ color: "var(--text-secondary)", lineHeight: 1.5 }}>
        {body || text}
      </div>
    </motion.div>
  );
}
