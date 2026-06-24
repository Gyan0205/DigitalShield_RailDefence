"""
Digital Shield Rail Defense -- Explainable AI Intelligence Engine
===================================================================
Unified alert generation engine that orchestrates CCTV subsystems
into a single explainable intelligence pipeline.

Pipeline:
  CCTV anomaly -> camera resolved -> platform mapped -> train inferred
  -> coach estimated -> EXPLAINABLE ALERT with full evidence chain

Every alert contains:
  - WHY: which anomaly signals triggered, with human-readable reasoning
  - CONFIDENCE: composite score from CCTV + train + coach signals
  - CONTEXT: train, platform, coach, timestamp, station
  - EVIDENCE: CCTV frame ref, railway context, behavioral features
  - ACTION: severity-based RPF recommendation
"""

import uuid
import json
import logging
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("xai_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


# ============================================================================
# ALERT DATA STRUCTURES
# ============================================================================

@dataclass
class CCTVEvidence:
    """Evidence from the CCTV anomaly detection subsystem."""
    camera_id: str = ""
    frame_number: int = 0
    timestamp: str = ""
    anomaly_type: str = ""
    anomaly_confidence: float = 0.0
    person_count: int = 0
    track_ids: List[int] = field(default_factory=list)
    video_source: str = ""

    def to_dict(self): return {k: v for k, v in self.__dict__.items()}


@dataclass
class RailwayContext:
    """Railway operational context resolved from CCTV observation."""
    station_code: str = ""
    station_name: str = ""
    platform: int = 0
    zone: str = ""
    train_number: str = ""
    train_name: str = ""
    train_type: str = ""
    train_status: str = ""
    estimated_coach: str = ""
    coach_class: str = ""
    coach_confidence: float = 0.0
    train_confidence: float = 0.0
    is_night: bool = False
    is_peak_hour: bool = False

    def to_dict(self): return {k: v for k, v in self.__dict__.items()}


@dataclass
class ConfidenceBreakdown:
    """Multi-factor confidence score decomposition."""
    cctv_confidence: float = 0.0        # From YOLO/anomaly detector
    train_match_confidence: float = 0.0  # From schedule correlation
    coach_confidence: float = 0.0        # From OCR / zone estimation
    temporal_modifier: float = 1.0       # Night/peak multiplier
    composite_confidence: float = 0.0    # Weighted final score

    def to_dict(self): return {k: round(v, 4) for k, v in self.__dict__.items()}


@dataclass
class ExplainableAlert:
    """Complete explainable intelligence alert."""
    # Identity
    alert_id: str = ""
    generated_at: str = ""

    # Classification
    severity: str = "LOW"              # LOW / MEDIUM / HIGH / CRITICAL
    alert_type: str = ""               # trafficking_suspect, behavioral_anomaly, etc.

    # Confidence
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)

    # Context
    cctv_evidence: CCTVEvidence = field(default_factory=CCTVEvidence)
    railway_context: RailwayContext = field(default_factory=RailwayContext)

    # Passengers
    passengers_narrowed: int = 0
    suspects: List[Dict] = field(default_factory=list)

    # Explainability
    triggered_rules: List[Dict] = field(default_factory=list)
    mitigating_factors: List[Dict] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    explanation_summary: str = ""
    recommended_action: str = ""

    # Audit
    pipeline_stages: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "generated_at": self.generated_at,
            "severity": self.severity,
            "alert_type": self.alert_type,
            "confidence": self.confidence.to_dict(),
            "cctv_evidence": self.cctv_evidence.to_dict(),
            "railway_context": self.railway_context.to_dict(),
            "passengers_narrowed": self.passengers_narrowed,
            "suspects": self.suspects[:10],
            "triggered_rules": self.triggered_rules,
            "mitigating_factors": self.mitigating_factors,
            "reasoning_chain": self.reasoning_chain,
            "explanation_summary": self.explanation_summary,
            "recommended_action": self.recommended_action,
            "pipeline_stages": self.pipeline_stages,
        }


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """Persistent audit trail for all generated alerts."""

    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir) if log_dir else Path("logs/audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict] = []

    def log_alert(self, alert: ExplainableAlert):
        """Log an alert to the audit trail."""
        entry = {
            "alert_id": alert.alert_id,
            "timestamp": alert.generated_at,
            "severity": alert.severity,
            "alert_type": alert.alert_type,
            "station": alert.railway_context.station_code,
            "platform": alert.railway_context.platform,
            "train": alert.railway_context.train_number,
            "coach": alert.railway_context.estimated_coach,
            "confidence": alert.confidence.composite_confidence,
            "suspects": len(alert.suspects),
            "rules_triggered": len(alert.triggered_rules),
            "pipeline_stages": len(alert.pipeline_stages),
        }
        self._entries.append(entry)
        logger.info(f"AUDIT | Alert {alert.alert_id} | {alert.severity} | "
                     f"conf={alert.confidence.composite_confidence:.2%}")

    def get_log(self, limit: int = 50) -> List[Dict]:
        return self._entries[-limit:]

    def get_stats(self) -> Dict:
        if not self._entries:
            return {"total_alerts": 0}
        severities = {}
        for e in self._entries:
            severities[e["severity"]] = severities.get(e["severity"], 0) + 1
        return {
            "total_alerts": len(self._entries),
            "severity_distribution": severities,
            "avg_confidence": round(
                sum(e["confidence"] for e in self._entries) / len(self._entries), 4
            ),
        }

    def save(self):
        """Persist audit log to disk."""
        path = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)


# ============================================================================
# XAI ENGINE (CORE ORCHESTRATOR)
# ============================================================================

class ExplainableIntelligenceEngine:
    """
    Unified explainable AI engine that orchestrates every Digital Shield
    subsystem into a single intelligence pipeline.

    Subsystems integrated:
      1. Camera Registry     -> resolve camera to station/platform
      2. Train Intelligence  -> infer active train from timestamp
      3. Bogie Mapper        -> estimate coach from zone/OCR
      4. Audit Logger        -> persistent evidence trail

    Usage:
        engine = ExplainableIntelligenceEngine()
        engine.initialize()

        alert = engine.generate_alert(
            camera_id="CAM_SC_P03_B",
            timestamp="2026-05-12T14:30:00",
            anomaly_type="suspicious_escort",
            anomaly_confidence=0.82,
        )
        print(alert.explanation_summary)
        print(alert.recommended_action)
    """

    def __init__(self):
        self._train_engine = None
        self._bogie_mapper = None
        self.audit = AuditLogger()
        self._initialized = False

    def initialize(self):
        """Initialize all subsystems."""
        if self._initialized:
            return

        logger.info("Initializing Explainable Intelligence Engine...")

        # Train Intelligence
        from backend.services.train_intelligence import TrainIntelligence
        self._train_engine = TrainIntelligence()
        self._train_engine.initialize(train_count=300)

        # Bogie Mapper
        from backend.services.bogie_mapper import BogieMapper
        self._bogie_mapper = BogieMapper(ocr_engine="pattern")

        self._initialized = True
        logger.info("Explainable Intelligence Engine ready (CCTV pipeline online)")

    # ------------------------------------------------------------------
    # CORE: GENERATE EXPLAINABLE ALERT
    # ------------------------------------------------------------------

    def generate_alert(
        self,
        camera_id: str = "",
        timestamp: str = "",
        anomaly_type: str = "behavioral_anomaly",
        anomaly_confidence: float = 0.5,
        person_count: int = 1,
        track_ids: List[int] = None,
        frame_number: int = 0,
        video_source: str = "",
        station_code: str = None,
        platform: int = None,
        train_number: str = None,
        coach: str = None,
    ) -> ExplainableAlert:
        """
        Generate a fully explainable alert from a CCTV anomaly event.

        This is the master pipeline that chains every subsystem together.
        """
        if not self._initialized:
            self.initialize()

        alert = ExplainableAlert(
            alert_id=str(uuid.uuid4())[:12],
            generated_at=datetime.utcnow().isoformat(),
            alert_type=anomaly_type,
        )

        # ── STAGE 1: CCTV Evidence ──────────────────────────────
        alert.cctv_evidence = CCTVEvidence(
            camera_id=camera_id, frame_number=frame_number,
            timestamp=timestamp, anomaly_type=anomaly_type,
            anomaly_confidence=anomaly_confidence,
            person_count=person_count,
            track_ids=track_ids or [],
            video_source=video_source,
        )
        alert.confidence.cctv_confidence = anomaly_confidence
        alert.pipeline_stages.append({
            "stage": "cctv_detection", "status": "complete",
            "detail": f"{anomaly_type} detected (conf={anomaly_confidence:.0%})"
        })
        alert.reasoning_chain.append(
            f"CCTV anomaly detected: {anomaly_type} with {anomaly_confidence:.0%} confidence"
        )

        # ── STAGE 2: Camera -> Station/Platform Resolution ──────
        resolved_station = station_code or self._resolve_station(camera_id)
        resolved_platform = platform or self._resolve_platform(camera_id)
        resolved_zone = self._resolve_zone(camera_id)

        alert.railway_context.station_code = resolved_station or "SC"
        alert.railway_context.platform = resolved_platform or 1
        alert.railway_context.zone = resolved_zone or "mid"

        station_name = self._get_station_name(alert.railway_context.station_code)
        alert.railway_context.station_name = station_name

        alert.pipeline_stages.append({
            "stage": "camera_resolution", "status": "complete",
            "detail": f"Camera {camera_id} -> {station_name} Platform {alert.railway_context.platform} ({alert.railway_context.zone} zone)"
        })
        alert.reasoning_chain.append(
            f"Camera {camera_id} resolved to {station_name}, "
            f"Platform {alert.railway_context.platform} ({alert.railway_context.zone} zone)"
        )

        # ── STAGE 3: Train Inference ────────────────────────────
        if train_number:
            alert.railway_context.train_number = train_number
            alert.confidence.train_match_confidence = 0.95
        else:
            train_result = self._train_engine.infer_train(
                station_code=alert.railway_context.station_code,
                platform=alert.railway_context.platform,
                timestamp=timestamp,
            )
            if train_result.get("primary_train"):
                pt = train_result["primary_train"]
                alert.railway_context.train_number = pt.get("train_number", "")
                alert.railway_context.train_name = pt.get("train_name", "")
                alert.railway_context.train_type = pt.get("train_type", "")
                alert.railway_context.train_status = pt.get("inferred_status", "")
                alert.confidence.train_match_confidence = pt.get("match_score", 0.5)

        alert.pipeline_stages.append({
            "stage": "train_inference", "status": "complete",
            "detail": f"Train {alert.railway_context.train_number} "
                      f"({alert.railway_context.train_name}) "
                      f"conf={alert.confidence.train_match_confidence:.0%}"
        })
        if alert.railway_context.train_number:
            alert.reasoning_chain.append(
                f"Train {alert.railway_context.train_number} "
                f"({alert.railway_context.train_name or 'identified'}) "
                f"at platform, status: {alert.railway_context.train_status or 'active'}"
            )

        # ── STAGE 4: Coach Estimation ───────────────────────────
        if coach:
            alert.railway_context.estimated_coach = coach
            alert.confidence.coach_confidence = 0.90
        else:
            coach_result = self._bogie_mapper.estimate_from_zone(
                alert.railway_context.zone,
                train_type=alert.railway_context.train_type or "Express",
            )
            alert.railway_context.estimated_coach = coach_result.estimated_coach
            alert.railway_context.coach_class = coach_result.coach_class
            alert.confidence.coach_confidence = coach_result.confidence

        alert.pipeline_stages.append({
            "stage": "coach_estimation", "status": "complete",
            "detail": f"Coach {alert.railway_context.estimated_coach} "
                      f"({alert.railway_context.coach_class or 'estimated'}) "
                      f"conf={alert.confidence.coach_confidence:.0%}"
        })
        alert.reasoning_chain.append(
            f"Coach estimated: {alert.railway_context.estimated_coach} "
            f"({alert.railway_context.coach_class or 'via zone mapping'}) "
            f"visible from {alert.railway_context.zone} zone camera"
        )

        # ── STAGE 5: Temporal Context ───────────────────────────
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                hour = dt.hour
                alert.railway_context.is_night = hour < 6 or hour >= 22
                alert.railway_context.is_peak_hour = hour in (7, 8, 9, 17, 18, 19)
                if alert.railway_context.is_night:
                    alert.confidence.temporal_modifier = 1.3
                    alert.reasoning_chain.append(
                        "ELEVATED: Night-time operation (22:00-06:00) increases trafficking risk"
                    )
                elif alert.railway_context.is_peak_hour:
                    alert.confidence.temporal_modifier = 0.9
            except (ValueError, TypeError):
                pass

        # ── STAGE 8: Composite Confidence ───────────────────────
        c = alert.confidence
        c.composite_confidence = self._compute_composite_confidence(c)

        # ── STAGE 9: Severity Classification ────────────────────
        alert.severity = self._classify_severity(alert)

        # ── STAGE 10: Generate Explanation & Action ─────────────
        alert.explanation_summary = self._build_explanation(alert)
        alert.recommended_action = self._build_recommendation(alert)

        # ── AUDIT ───────────────────────────────────────────────
        self.audit.log_alert(alert)

        return alert

    # ------------------------------------------------------------------
    # CONFIDENCE COMPUTATION
    # ------------------------------------------------------------------

    def _compute_composite_confidence(self, c: ConfidenceBreakdown) -> float:
        """Weighted composite confidence from CCTV subsystems."""
        weights = {
            "cctv": 0.50,
            "train": 0.30,
            "coach": 0.20,
        }
        raw = (
            weights["cctv"] * c.cctv_confidence
            + weights["train"] * c.train_match_confidence
            + weights["coach"] * c.coach_confidence
        ) * c.temporal_modifier

        return round(min(max(raw, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    # SEVERITY CLASSIFICATION
    # ------------------------------------------------------------------

    def _classify_severity(self, alert: ExplainableAlert) -> str:
        cc = alert.confidence.composite_confidence
        suspects = len(alert.suspects)
        has_critical_rule = any(
            r["rule_id"] in ("R1_REPEATED_ESCORT", "R2_VULNERABILITY")
            for r in alert.triggered_rules
        )

        if cc >= 0.55 or (has_critical_rule and suspects >= 2):
            return "CRITICAL"
        elif cc >= 0.35 or suspects >= 2 or has_critical_rule:
            return "HIGH"
        elif cc >= 0.20 or suspects >= 1:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # EXPLAINABLE OUTPUT GENERATORS
    # ------------------------------------------------------------------

    def _build_explanation(self, alert: ExplainableAlert) -> str:
        lines = [
            f"=== DIGITAL SHIELD INTELLIGENCE ALERT ===",
            f"Alert ID: {alert.alert_id} | Severity: {alert.severity} | "
            f"Confidence: {alert.confidence.composite_confidence:.0%}",
            f"Generated: {alert.generated_at}",
            "",
            f"OBSERVATION:",
            f"  Camera {alert.cctv_evidence.camera_id} detected "
            f"'{alert.cctv_evidence.anomaly_type}' "
            f"(conf={alert.cctv_evidence.anomaly_confidence:.0%})",
            f"  Station: {alert.railway_context.station_name} "
            f"({alert.railway_context.station_code}), "
            f"Platform {alert.railway_context.platform}",
            f"  Time: {alert.cctv_evidence.timestamp}",
        ]

        if alert.railway_context.train_number:
            lines.append(
                f"  Train: {alert.railway_context.train_number} "
                f"({alert.railway_context.train_name}), "
                f"Status: {alert.railway_context.train_status}"
            )
        if alert.railway_context.estimated_coach:
            lines.append(
                f"  Coach: {alert.railway_context.estimated_coach} "
                f"({alert.railway_context.coach_class})"
            )

        lines.append(f"\nINTELLIGENCE:")
        lines.append(f"  Passengers narrowed: {alert.passengers_narrowed}")
        lines.append(f"  Suspects flagged: {len(alert.suspects)}")

        if alert.triggered_rules:
            lines.append(f"\nRED FLAGS ({len(alert.triggered_rules)} rules):")
            for r in alert.triggered_rules:
                lines.append(f"  [{r['score']:+.1f}] {r['rule']}: {r['explanation'][:90]}")

        if alert.mitigating_factors:
            lines.append(f"\nMITIGATING ({len(alert.mitigating_factors)}):")
            for m in alert.mitigating_factors:
                lines.append(f"  [{m['score']:+.1f}] {m['rule']}: {m['explanation'][:90]}")

        lines.append(f"\nCONFIDENCE BREAKDOWN:")
        c = alert.confidence
        lines.append(f"  CCTV anomaly:     {c.cctv_confidence:.0%} (weight 50%)")
        lines.append(f"  Train match:      {c.train_match_confidence:.0%} (weight 30%)")
        lines.append(f"  Coach estimation: {c.coach_confidence:.0%} (weight 20%)")
        lines.append(f"  Temporal mod:     x{c.temporal_modifier:.1f}")
        lines.append(f"  COMPOSITE:        {c.composite_confidence:.0%}")

        return "\n".join(lines)

    def _build_recommendation(self, alert: ExplainableAlert) -> str:
        actions = {
            "CRITICAL": (
                "IMMEDIATE INTERVENTION REQUIRED:\n"
                "1. Alert RPF Control Room immediately\n"
                "2. Dispatch team to Platform {platform}, Coach {coach}\n"
                "3. Detain and verify identities of {n} flagged passengers\n"
                "4. Check minors' documentation and parental consent\n"
                "5. Activate CCTV recording on all platform cameras\n"
                "6. Contact local Child Welfare Committee if minors involved"
            ),
            "HIGH": (
                "URGENT ATTENTION:\n"
                "1. Alert on-duty RPF at Platform {platform}\n"
                "2. Monitor Coach {coach} exits\n"
                "3. Prepare identity verification for {n} suspects\n"
                "4. Cross-reference with missing persons database\n"
                "5. Increase camera surveillance on flagged area"
            ),
            "MEDIUM": (
                "ENHANCED MONITORING:\n"
                "1. Flag for continuous CCTV surveillance\n"
                "2. RPF standby at Platform {platform}\n"
                "3. Log event for pattern analysis\n"
                "4. Monitor Coach {coach} during boarding/alighting"
            ),
            "LOW": (
                "ROUTINE LOGGING:\n"
                "1. Record event for statistical analysis\n"
                "2. No immediate intervention required\n"
                "3. Review if pattern recurs within 24 hours"
            ),
        }
        template = actions.get(alert.severity, actions["LOW"])
        return template.format(
            platform=alert.railway_context.platform,
            coach=alert.railway_context.estimated_coach or "unknown",
            n=len(alert.suspects),
        )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _resolve_station(self, camera_id: str) -> str:
        import re
        m = re.match(r'CAM_([A-Z]+)_', camera_id)
        return m.group(1) if m else "SC"

    def _resolve_platform(self, camera_id: str) -> int:
        import re
        m = re.search(r'P(\d+)', camera_id)
        return int(m.group(1)) if m else 1

    def _resolve_zone(self, camera_id: str) -> str:
        zone_map = {"A": "entry", "B": "mid", "C": "exit"}
        if camera_id and camera_id[-1] in zone_map:
            return zone_map[camera_id[-1]]
        return "mid"

    def _get_station_name(self, code: str) -> str:
        try:
            from backend.services.railway_stations import STATION_LOOKUP
            s = STATION_LOOKUP.get(code)
            return s.name if s else code
        except Exception:
            return code

    # ------------------------------------------------------------------
    # BATCH & QUERY
    # ------------------------------------------------------------------

    def get_audit_log(self, limit: int = 50) -> List[Dict]:
        return self.audit.get_log(limit)

    def get_audit_stats(self) -> Dict:
        return self.audit.get_stats()
