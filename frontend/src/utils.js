/**
 * utils.js — Enterprise theme constants + field normalizers
 */

export const SEV_COLOR = {
  CRITICAL: "#B42318",
  HIGH:     "#C97A10",
  MEDIUM:   "#3B82F6",
  LOW:      "#94A3B8",
};

export const SEV_BG = {
  CRITICAL: "#FEF2F0",
  HIGH:     "#FFF9ED",
  MEDIUM:   "#EFF6FF",
  LOW:      "#F8FAFC",
};

export const SEV_BORDER = {
  CRITICAL: "#FBC9C4",
  HIGH:     "#F5D99A",
  MEDIUM:   "#BFD9FE",
  LOW:      "#E2E8F0",
};

export function severityColor(sev) {
  return SEV_COLOR[(sev ?? "").toUpperCase()] ?? "#94A3B8";
}

/** Normalize confidence to 0–100 %.
 *  In-memory:  fused_confidence  (0.0–1.0)
 *  DB table:   fusion_confidence (0.0–1.0)
 */
export function getConfidence(alert) {
  const raw =
    alert.fused_confidence  ??
    alert.fusion_confidence ??
    alert.confidence        ??
    alert.anomaly_confidence ??
    alert.final_score        ??
    0;
  const n = parseFloat(raw);
  if (isNaN(n)) return 0;
  return n <= 1.0 ? n * 100 : n;
}

export function getCctvScore(alert) {
  const sc = alert.source_contributions?.cctv_anomaly;
  if (sc != null) return parseFloat(sc) * 100 / 0.60;
  const raw = alert.cctv_score ?? alert.anomaly_confidence ?? 0;
  const n = parseFloat(raw);
  return n <= 1 ? n * 100 : n;
}

export function getTicketScore(alert) {
  const raw = alert.ticket_score ?? alert.tickets_score ?? 0;
  const n = parseFloat(raw);
  return n <= 1 ? n * 100 : n;
}

export function getAnomalyType(alert) {
  const raw = alert.anomaly_type ?? alert.alert_type ?? alert.type ?? "anomaly";
  return raw.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export function getXaiSummary(alert) {
  const chain = alert.fusion_reasoning ?? alert.reasoning_chain ?? [];
  if (Array.isArray(chain) && chain.length > 0) {
    const reasons = chain.filter(l =>
      l.includes("Controller") || l.includes("Evasive") ||
      l.includes("escort") || l.includes("Minor") ||
      l.includes("IP") || l.includes("pattern") ||
      l.includes("unrelated") || l.includes("family")
    );
    if (reasons.length > 0) return reasons[0].replace(/^\s*[-·•\[.*?\]]\s*/, "").slice(0, 90);
    const meaningful = chain.filter(l => l.trim().length > 10).pop();
    if (meaningful) return meaningful.replace(/^\[.*?\]\s*/, "").slice(0, 90);
  }
  if (typeof alert.xai_explanation === "string" && alert.xai_explanation.length > 0)
    return alert.xai_explanation.slice(0, 90);
  if (typeof alert.suspect_description === "string" && alert.suspect_description.length > 0)
    return alert.suspect_description.slice(0, 90);
  return null;
}

export function getReasoningChain(alert) {
  const chain = alert.fusion_reasoning ?? alert.reasoning_chain ?? [];
  if (Array.isArray(chain) && chain.length > 0) return chain;
  if (typeof alert.xai_explanation === "string" && alert.xai_explanation.length > 0)
    return alert.xai_explanation.split("\n").filter(l => l.trim());
  return [];
}

export function getCoach(alert)       { return alert.estimated_coach ?? alert.coach ?? "—"; }
export function getPlatform(alert)    { return alert.platform ?? alert.platform_number ?? alert.metadata?.platform ?? "—"; }
export function getTrainNumber(alert) { return alert.train_number ?? alert.metadata?.train_number ?? "—"; }
export function getTrainName(alert)   { return alert.train_name ?? alert.metadata?.train_name ?? ""; }
export function getStation(alert)     { return alert.station_code ?? alert.station ?? "SC"; }
export function getSuspects(alert)    { return alert.suspects ?? alert.suspect_count ?? 0; }
export function getTimestamp(alert)   { return alert.timestamp ?? alert.generated_at ?? alert.created_at ?? ""; }

export function fmtTime(ts) {
  if (!ts) return "—";
  try { return new Date(ts).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true }); }
  catch { return ts; }
}

export function fmtDate(ts) {
  if (!ts) return "—";
  try { return new Date(ts).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }); }
  catch { return ts; }
}

export function shortId(id = "") {
  return id.split("_").pop()?.slice(0, 8).toUpperCase() ?? id.slice(-8);
}
