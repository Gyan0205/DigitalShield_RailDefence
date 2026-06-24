/**
 * AnalyticsPanel — executive-grade muted charts
 */
import {
  BarChart, Bar, PieChart, Pie, Cell,
  Tooltip, ResponsiveContainer, XAxis, YAxis,
} from "recharts";
import { SEV_COLOR, getConfidence, getPlatform, getAnomalyType, getTrainNumber } from "../utils";

const SEV_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];

// Muted enterprise chart palette
const CHART_COLORS = ["#6366F1", "#3B82F6", "#059669", "#C97A10", "#B42318", "#7C3AED"];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#FFFFFF", border: "1px solid var(--border)",
      borderRadius: 8, padding: "8px 12px", fontSize: 12,
      boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
    }}>
      {label && <div style={{ color: "var(--text-muted)", marginBottom: 3, fontWeight: 500 }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: "var(--text-primary)" }}>
          {p.name ?? "count"}: <b>{p.value}</b>
        </div>
      ))}
    </div>
  );
};

export default function AnalyticsPanel({ alerts }) {
  // Severity distribution
  const sevData = SEV_ORDER.map(s => ({
    name: s.charAt(0) + s.slice(1).toLowerCase(),
    value: alerts.filter(a => (a.severity ?? "").toUpperCase() === s).length,
    color: SEV_COLOR[s],
  })).filter(d => d.value > 0);

  // Anomaly types
  const typeCount = {};
  alerts.forEach(a => { const t = getAnomalyType(a); typeCount[t] = (typeCount[t] ?? 0) + 1; });
  const typeData = Object.entries(typeCount)
    .sort((a, b) => b[1] - a[1]).slice(0, 6)
    .map(([name, count]) => ({ name, count }));

  // Platform distribution
  const platCount = {};
  alerts.forEach(a => { const p = `P${getPlatform(a)}`; platCount[p] = (platCount[p] ?? 0) + 1; });
  const platData = Object.entries(platCount)
    .sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([name, count]) => ({ name, count }));

  // Top trains
  const trainCount = {};
  alerts.forEach(a => { const t = getTrainNumber(a); if (t && t !== "—") trainCount[t] = (trainCount[t] ?? 0) + 1; });
  const trainData = Object.entries(trainCount).sort((a, b) => b[1] - a[1]).slice(0, 4);

  // Avg conf by severity
  const avgConf = SEV_ORDER.map(s => {
    const group = alerts.filter(a => (a.severity ?? "").toUpperCase() === s);
    if (!group.length) return null;
    return {
      name: s.charAt(0) + s.slice(1).toLowerCase(),
      avg: Math.round(group.reduce((acc, a) => acc + getConfidence(a), 0) / group.length * 10) / 10,
      color: SEV_COLOR[s],
    };
  }).filter(Boolean);

  const totalConf = alerts.length
    ? alerts.reduce((s, a) => s + getConfidence(a), 0) / alerts.length
    : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

      {/* Mini stat chips */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <MiniStat label="Avg confidence"  value={`${totalConf.toFixed(1)}%`} color="#6366F1" />
        <MiniStat label="Active trains"   value={Object.keys(trainCount).length} color="#059669" />
        <MiniStat label="Platforms hit"   value={Object.keys(platCount).length}  color="#C97A10" />
        <MiniStat label="Total alerts"    value={alerts.length}                  color="#3B82F6" />
      </div>

      {/* Severity pie */}
      <ChartBox title="Severity breakdown">
        {sevData.length > 0 ? (
          <>
            <ResponsiveContainer width="100%" height={120}>
              <PieChart>
                <Pie data={sevData} dataKey="value" innerRadius={30} outerRadius={50}
                  strokeWidth={1} stroke="#FFFFFF">
                  {sevData.map((d, i) => <Cell key={i} fill={d.color} opacity={0.85} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
              {sevData.map(d => (
                <div key={d.name} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: d.color, display: "block" }} />
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{d.name} ({d.value})</span>
                </div>
              ))}
            </div>
          </>
        ) : <Empty />}
      </ChartBox>

      {/* Anomaly types */}
      <ChartBox title="Anomaly types">
        {typeData.length > 0 ? (
          <ResponsiveContainer width="100%" height={typeData.length * 22 + 8}>
            <BarChart data={typeData} layout="vertical" margin={{ left: 0, right: 28, top: 2, bottom: 2 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "#8C8C8C" }} width={90} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}
                label={{ position: "right", fontSize: 10, fill: "#8C8C8C" }}>
                {typeData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} opacity={0.8} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <Empty />}
      </ChartBox>

      {/* Platform distribution */}
      <ChartBox title="By platform">
        {platData.length > 0 ? (
          <ResponsiveContainer width="100%" height={90}>
            <BarChart data={platData} margin={{ left: -24, right: 4, top: 4 }}>
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#8C8C8C" }} />
              <YAxis tick={{ fontSize: 10, fill: "#8C8C8C" }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" fill="#6366F1" radius={[3, 3, 0, 0]} opacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        ) : <Empty />}
      </ChartBox>

      {/* Top trains */}
      {trainData.length > 0 && (
        <ChartBox title="Top risk trains">
          {trainData.map(([name, count], i) => (
            <div key={name} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <span style={{ fontSize: 10, color: "var(--text-dim)", width: 16 }}>#{i + 1}</span>
              <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1, fontFamily: "monospace" }}>{name}</span>
              <div style={{ width: 50, height: 4, background: "var(--border)", borderRadius: 2 }}>
                <div style={{ width: `${(count / trainData[0][1]) * 100}%`, height: "100%", background: "#059669", borderRadius: 2 }} />
              </div>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#059669", width: 18, textAlign: "right" }}>{count}</span>
            </div>
          ))}
        </ChartBox>
      )}

      {/* Avg confidence bars */}
      {avgConf.length > 0 && (
        <ChartBox title="Avg confidence">
          {avgConf.map(d => (
            <div key={d.name} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "var(--text-secondary)", width: 52 }}>{d.name}</span>
              <div style={{ flex: 1, height: 5, background: "var(--border)", borderRadius: 3 }}>
                <div style={{ width: `${d.avg}%`, height: "100%", background: d.color, borderRadius: 3, opacity: 0.8 }} />
              </div>
              <span style={{ fontSize: 11, fontWeight: 600, color: d.color, width: 36, textAlign: "right" }}>{d.avg}%</span>
            </div>
          ))}
        </ChartBox>
      )}
    </div>
  );
}

function ChartBox({ title, children }) {
  return (
    <div style={{
      background: "#FFFFFF", border: "1px solid var(--border)",
      borderRadius: 10, padding: "12px 14px",
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", marginBottom: 10, letterSpacing: "0.02em" }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function MiniStat({ label, value, color }) {
  return (
    <div style={{
      background: "#FFFFFF", border: "1px solid var(--border)",
      borderRadius: 10, padding: "10px 12px", textAlign: "center",
    }}>
      <div style={{ fontSize: 20, fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>{label}</div>
    </div>
  );
}

function Empty() {
  return (
    <div style={{ height: 48, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-dim)", fontSize: 12 }}>
      No data yet
    </div>
  );
}
