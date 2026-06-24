/**
 * StatsGrid — clean enterprise KPI cards
 */
import { AlertCircle, Zap, Activity, Database, TrendingUp } from "lucide-react";
import { getConfidence } from "../utils";

const CARDS = [
  { key: "CRITICAL", label: "Critical", icon: AlertCircle, color: "#B42318", bg: "#FEF2F0", border: "#FBC9C4" },
  { key: "HIGH",     label: "High",     icon: Zap,          color: "#C97A10", bg: "#FFF9ED", border: "#F5D99A" },
  { key: "MEDIUM",   label: "Medium",   icon: Activity,     color: "#3B82F6", bg: "#EFF6FF", border: "#BFD9FE" },
  { key: "total",    label: "Total",    icon: Database,     color: "#6366F1", bg: "#F5F3FF", border: "#DDD6FE" },
];

export default function StatsGrid({ alerts }) {
  const counts = {
    CRITICAL: alerts.filter(a => (a.severity ?? "").toUpperCase() === "CRITICAL").length,
    HIGH:     alerts.filter(a => (a.severity ?? "").toUpperCase() === "HIGH").length,
    MEDIUM:   alerts.filter(a => (a.severity ?? "").toUpperCase() === "MEDIUM").length,
    total:    alerts.length,
  };

  const avgConf = alerts.length > 0
    ? alerts.reduce((s, a) => s + getConfidence(a), 0) / alerts.length
    : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        {CARDS.map(card => {
          const Icon = card.icon;
          const val  = counts[card.key];
          return (
            <div key={card.key} style={{
              background: "#FFFFFF",
              border: "1px solid var(--border)",
              borderRadius: 12,
              padding: "14px 16px",
              display: "flex", alignItems: "center", gap: 12,
              boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
            }}>
              <div style={{
                width: 36, height: 36, borderRadius: 10,
                background: card.bg, border: `1px solid ${card.border}`,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                <Icon size={16} color={card.color} />
              </div>
              <div>
                <div style={{ fontSize: 24, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1 }}>
                  {val}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{card.label}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Avg confidence strip */}
      {alerts.length > 0 && (
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "7px 14px", borderRadius: 10,
          background: "#FFFFFF", border: "1px solid var(--border)",
        }}>
          <TrendingUp size={13} color="#6366F1" />
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            Avg confidence across {alerts.length} alerts
          </span>
          <span style={{ fontSize: 13, fontWeight: 700, color: "#6366F1", marginLeft: "auto" }}>
            {avgConf.toFixed(1)}%
          </span>
          <div style={{ width: 80, height: 4, background: "var(--border)", borderRadius: 2 }}>
            <div style={{ width: `${Math.min(avgConf, 100)}%`, height: "100%", background: "#6366F1", borderRadius: 2 }} />
          </div>
        </div>
      )}
    </div>
  );
}
