/**
 * HealthPanel — system health sidebar section
 */
import { motion } from "framer-motion";
import { CheckCircle, XCircle, AlertCircle, Database, Server, Cpu, Wifi } from "lucide-react";

function StatusDot({ status }) {
  const ok  = status === "healthy" || status === "online" || status === true;
  const err = status === "error" || status === "offline" || status === false;
  const color = ok ? "#22c55e" : err ? "#ef4444" : "#eab308";
  return (
    <span className={ok ? "pulse-dot" : ""} style={{
      width: 7, height: 7, borderRadius: "50%",
      background: color, display: "inline-block", flexShrink: 0,
    }} />
  );
}

function HealthRow({ icon, label, status, detail }) {
  const ok  = status === "healthy" || status === "online" || status === true;
  const color = ok ? "#22c55e" : "#ef4444";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "7px 0", borderBottom: "1px solid rgba(255,255,255,0.04)",
    }}>
      <span style={{ color: "var(--text-muted)" }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 12, color: "#cbd5e1", fontWeight: 500 }}>{label}</div>
        {detail && <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{detail}</div>}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <StatusDot status={status} />
        <span style={{ fontSize: 10, color, fontWeight: 600 }}>
          {typeof status === "boolean" ? (status ? "OK" : "ERR") : (status ?? "—").toUpperCase()}
        </span>
      </div>
    </div>
  );
}

export default function HealthPanel({ health, loading }) {
  if (loading) return (
    <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 12, textAlign: "center" }}>
      Checking system health…
    </div>
  );

  if (!health) return (
    <div style={{ padding: 16, color: "#ef4444", fontSize: 12, textAlign: "center" }}>
      Health check unavailable
    </div>
  );

  const comps = health.components ?? {};
  const db    = comps.database ?? {};
  const redis = comps.redis ?? {};
  const tix   = comps.tickets_intelligence ?? {};
  const fusion= comps.fusion_engine ?? {};
  const trains= comps.trains_table ?? {};

  const overallOk = (health.status ?? "").toLowerCase() !== "degraded";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{ display: "flex", flexDirection: "column", gap: 2 }}
    >
      {/* Overall status badge */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8, marginBottom: 10,
        padding: "8px 12px", borderRadius: 8,
        background: overallOk ? "rgba(34,197,94,0.08)" : "rgba(234,179,8,0.08)",
        border: `1px solid ${overallOk ? "rgba(34,197,94,0.25)" : "rgba(234,179,8,0.25)"}`,
      }}>
        {overallOk
          ? <CheckCircle size={14} color="#22c55e" />
          : <AlertCircle size={14} color="#eab308" />
        }
        <span style={{ fontSize: 12, fontWeight: 600, color: overallOk ? "#22c55e" : "#eab308" }}>
          {(health.status ?? "unknown").toUpperCase()}
        </span>
        <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto" }}>
          {health.service ? "Digital Shield RD" : ""}
        </span>
      </div>

      <HealthRow icon={<Database size={12} />} label="PostgreSQL"
        status={db.status}
        detail={`${db.tables?.length ?? 0} tables`}
      />
      <HealthRow icon={<Server size={12} />} label="Redis Cache"
        status={redis.status}
        detail={redis.status === "healthy" ? "24h TTL cache" : "In-memory fallback"}
      />
      <HealthRow icon={<Cpu size={12} />} label="Fusion Engine"
        status={fusion.status}
        detail={fusion.total_alerts_this_session != null ? `${fusion.total_alerts_this_session} alerts this session` : undefined}
      />
      <HealthRow icon={<Database size={12} />} label="Ticket ML"
        status={tix.initialized ? "healthy" : (tix.status ?? "pending")}
        detail={tix.records != null ? `${tix.records?.toLocaleString()} records` : tix.note ?? "Loads on first call"}
      />
      <HealthRow icon={<Wifi size={12} />} label="Trains Table"
        status={trains.status}
        detail={trains.train_count != null ? `${trains.train_count} trains` : undefined}
      />
    </motion.div>
  );
}
