/**
 * TopRiskTrains — clean ranked train list
 */
import { Train, AlertCircle } from "lucide-react";
import { SEV_COLOR, getConfidence } from "../utils";

export default function TopRiskTrains({ alerts }) {
  const trainMap = {};
  alerts.forEach(a => {
    const num  = a.train_number ?? a.metadata?.train_number ?? "Unknown";
    const name = a.train_name  ?? a.metadata?.train_name   ?? "";
    const sev  = (a.severity ?? "LOW").toUpperCase();
    if (!trainMap[num]) {
      trainMap[num] = { num, name, count: 0, topSev: "LOW", sevCounts: { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 } };
    }
    trainMap[num].count += 1;
    trainMap[num].sevCounts[sev] = (trainMap[num].sevCounts[sev] ?? 0) + 1;
    const order = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
    if ((order[sev] ?? 0) > (order[trainMap[num].topSev] ?? 0)) trainMap[num].topSev = sev;
  });

  const ranked = Object.values(trainMap).sort((a, b) => b.count - a.count).slice(0, 5);

  if (ranked.length === 0) {
    return (
      <div style={{ padding: "16px 0", textAlign: "center" }}>
        <Train size={20} color="var(--text-dim)" style={{ margin: "0 auto 8px" }} />
        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>No alerts yet</div>
      </div>
    );
  }

  const max = ranked[0].count;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {ranked.map((train, i) => {
        const color = SEV_COLOR[train.topSev] ?? "#94A3B8";
        const bar   = Math.round((train.count / max) * 100);

        return (
          <div key={train.num} style={{
            padding: "10px 12px", borderRadius: 9,
            background: "#FFFFFF", border: "1px solid var(--border)",
            borderLeft: `3px solid ${color}`,
          }}>
            {/* Top row */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7 }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-dim)", width: 16 }}>
                #{i + 1}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 13, fontWeight: 600, color: "var(--text-primary)",
                  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                }}>
                  Train {train.num}
                </div>
                {train.name && (
                  <div style={{
                    fontSize: 10, color: "var(--text-muted)",
                    whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                  }}>
                    {train.name}
                  </div>
                )}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
                <AlertCircle size={11} color={color} />
                <span style={{ fontSize: 14, fontWeight: 700, color }}>{train.count}</span>
              </div>
            </div>

            {/* Bar */}
            <div style={{ height: 3, background: "var(--border)", borderRadius: 2, marginBottom: 6 }}>
              <div style={{
                width: `${bar}%`, height: "100%", background: color,
                borderRadius: 2, opacity: 0.7,
              }} />
            </div>

            {/* Severity chips */}
            <div style={{ display: "flex", gap: 5 }}>
              {Object.entries(train.sevCounts)
                .filter(([, v]) => v > 0)
                .map(([sev, count]) => (
                  <span key={sev} style={{
                    fontSize: 10, fontWeight: 500,
                    color: SEV_COLOR[sev],
                    background: sev === "CRITICAL" ? "#FEF2F0" : sev === "HIGH" ? "#FFF9ED" : sev === "MEDIUM" ? "#EFF6FF" : "#F8FAFC",
                    border: `1px solid ${sev === "CRITICAL" ? "#FBC9C4" : sev === "HIGH" ? "#F5D99A" : sev === "MEDIUM" ? "#BFD9FE" : "#E2E8F0"}`,
                    padding: "0 6px", borderRadius: 4,
                  }}>
                    {sev[0]}{count}
                  </span>
                ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
