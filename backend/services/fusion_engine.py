"""
Digital Shield Rail Defense -- Multi-Source Fusion Intelligence Engine
========================================================================
Event-driven architecture that fuses signals from 4 CCTV sources into
high-confidence railway intelligence alerts.

Sources:
  1. CCTV anomaly detection (YOLOv8/DeepSORT)
  2. Railway metadata (station/camera context)
  3. Train intelligence (schedule correlation)
  4. Coach estimation (OCR/zone mapping)

Architecture:
  - EventBus: async publish/subscribe for decoupled processing
  - EvidenceAccumulator: Dempster-Shafer-inspired belief combination
  - TemporalCorrelator: sliding-window event clustering
  - FusionEngine: weighted multi-source signal merger
"""

import uuid
import time
import logging
import threading
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("fusion_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class SignalSource(str, Enum):
    CCTV = "cctv_anomaly"
    RAILWAY_META = "railway_metadata"
    TRAIN_INTEL = "train_intelligence"
    COACH_EST = "coach_estimation"


# Source reliability priors (calibrated from system testing)
# Renormalized across 4 CCTV-only sources
SOURCE_WEIGHTS = {
    SignalSource.CCTV: 0.46,
    SignalSource.RAILWAY_META: 0.15,
    SignalSource.TRAIN_INTEL: 0.23,
    SignalSource.COACH_EST: 0.15,
}


@dataclass
class Signal:
    """A single intelligence signal from one source."""
    signal_id: str = ""
    source: str = ""
    timestamp: str = ""
    confidence: float = 0.0
    payload: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "signal_id": self.signal_id, "source": self.source,
            "timestamp": self.timestamp, "confidence": round(self.confidence, 4),
            "payload": self.payload, "metadata": self.metadata,
        }


@dataclass
class FusedAlert:
    """High-confidence alert produced by multi-source fusion."""
    alert_id: str = ""
    generated_at: str = ""
    severity: str = "LOW"
    suspects: int = 0                  # Number of flagged anomalous passengers

    # Fusion scores
    fused_confidence: float = 0.0
    source_contributions: Dict = field(default_factory=dict)
    agreement_score: float = 0.0      # How many sources agree
    evidence_mass: float = 0.0        # Dempster-Shafer belief mass

    # Context (from best-source signals)
    station_code: str = ""
    station_name: str = ""
    platform: int = 0
    train_number: str = ""
    train_name: str = ""
    estimated_coach: str = ""
    anomaly_type: str = ""
    timestamp: str = ""

    # Provenance
    source_signals: List[Dict] = field(default_factory=list)
    fusion_reasoning: List[str] = field(default_factory=list)
    temporal_cluster_size: int = 0
    recommended_action: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# ============================================================================
# EVENT BUS (Async Pub/Sub)
# ============================================================================

class EventBus:
    """
    Asynchronous event bus for decoupled signal processing.

    Publishers emit signals; subscribers process them independently.
    Supports topic-based routing and wildcard subscriptions.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_log: List[Dict] = []
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable):
        with self._lock:
            self._subscribers[topic].append(callback)

    def publish(self, topic: str, signal: Signal):
        with self._lock:
            self._event_log.append({
                "topic": topic, "signal_id": signal.signal_id,
                "source": signal.source, "timestamp": signal.timestamp,
                "confidence": signal.confidence,
            })
        # Deliver to topic subscribers
        for cb in self._subscribers.get(topic, []):
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Subscriber error on {topic}: {e}")
        # Deliver to wildcard subscribers
        for cb in self._subscribers.get("*", []):
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Wildcard subscriber error: {e}")

    def get_event_log(self, limit: int = 100) -> List[Dict]:
        return self._event_log[-limit:]

    @property
    def total_events(self) -> int:
        return len(self._event_log)


# ============================================================================
# EVIDENCE ACCUMULATOR (Dempster-Shafer Inspired)
# ============================================================================

class EvidenceAccumulator:
    """
    Combines evidence from multiple independent sources using
    a simplified Dempster-Shafer belief combination.

    Each source provides a belief mass m(threat) in [0, 1].
    Combined mass increases when independent sources agree,
    and conflict between sources reduces overall confidence.
    """

    @staticmethod
    def combine_beliefs(beliefs: List[float]) -> float:
        """
        Combine independent belief masses.

        Uses the complementary product rule:
          combined = 1 - product(1 - b_i) for all beliefs b_i

        This gives higher weight when multiple sources agree.
        """
        if not beliefs:
            return 0.0
        complement_product = 1.0
        for b in beliefs:
            complement_product *= (1.0 - max(0.0, min(1.0, b)))
        return round(1.0 - complement_product, 4)

    @staticmethod
    def weighted_combine(belief_weights: List[tuple]) -> float:
        """
        Weighted belief combination.

        Args:
            belief_weights: List of (belief, weight) tuples
        Returns:
            Weighted combined belief mass
        """
        if not belief_weights:
            return 0.0
        total_weight = sum(w for _, w in belief_weights)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(b * w for b, w in belief_weights)
        linear = weighted_sum / total_weight
        # Also compute agreement bonus
        beliefs = [b for b, _ in belief_weights]
        agreement = EvidenceAccumulator.combine_beliefs(beliefs)
        # Blend: 60% weighted linear + 40% agreement bonus
        return round(0.6 * linear + 0.4 * agreement, 4)

    @staticmethod
    def compute_agreement(beliefs: List[float]) -> float:
        """Compute pairwise agreement between sources (0-1)."""
        if len(beliefs) < 2:
            return 0.0
        above_threshold = sum(1 for b in beliefs if b > 0.2)
        return round(above_threshold / len(beliefs), 4)

    @staticmethod
    def compute_conflict(beliefs: List[float]) -> float:
        """Compute conflict between sources (0-1, lower is better)."""
        if len(beliefs) < 2:
            return 0.0
        mean_b = sum(beliefs) / len(beliefs)
        variance = sum((b - mean_b) ** 2 for b in beliefs) / len(beliefs)
        return round(min(variance * 4, 1.0), 4)  # Scale to [0,1]


# ============================================================================
# TEMPORAL CORRELATOR
# ============================================================================

class TemporalCorrelator:
    """
    Clusters signals within a time window to identify
    correlated events from the same real-world incident.

    Signals from the same station+platform within the window
    are grouped into a single incident cluster.
    """

    def __init__(self, window_seconds: int = 300):
        self.window = window_seconds
        self._clusters: Dict[str, List[Signal]] = defaultdict(list)

    def add_signal(self, signal: Signal) -> str:
        """Add signal and return its cluster ID."""
        cluster_key = self._find_cluster(signal)
        if not cluster_key:
            cluster_key = f"INC_{uuid.uuid4().hex[:8]}"
        self._clusters[cluster_key].append(signal)
        return cluster_key

    def get_cluster(self, cluster_id: str) -> List[Signal]:
        return self._clusters.get(cluster_id, [])

    def get_active_clusters(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._clusters.items() if v}

    def _find_cluster(self, signal: Signal) -> Optional[str]:
        """Find existing cluster this signal belongs to."""
        sig_station = signal.payload.get("station_code", "")
        sig_platform = signal.payload.get("platform", 0)
        sig_time = self._parse_time(signal.timestamp)

        for cid, signals in self._clusters.items():
            if not signals:
                continue
            first = signals[0]
            cluster_station = first.payload.get("station_code", "")
            cluster_platform = first.payload.get("platform", 0)
            cluster_time = self._parse_time(first.timestamp)

            if (sig_station == cluster_station
                    and sig_platform == cluster_platform
                    and sig_time and cluster_time
                    and abs((sig_time - cluster_time).total_seconds()) < self.window):
                return cid
        return None

    @staticmethod
    def _parse_time(ts: str) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None


# ============================================================================
# FUSION ENGINE (CORE)
# ============================================================================

class FusionEngine:
    """
    Multi-source fusion intelligence engine.

    Integrates all Digital Shield subsystems via an event-driven
    architecture with Dempster-Shafer evidence combination.

    Usage:
        engine = FusionEngine()
        engine.initialize()

        alert = engine.fuse_event(
            camera_id="CAM_SC_P03_B",
            timestamp="2026-05-12T14:30:00",
            anomaly_type="suspicious_escort",
            anomaly_confidence=0.82,
        )
        print(alert.severity, alert.fused_confidence)
    """

    def __init__(self, temporal_window: int = 300):
        self.event_bus = EventBus()
        self.accumulator = EvidenceAccumulator()
        self.correlator = TemporalCorrelator(window_seconds=temporal_window)

        # Subsystem references (lazy-loaded)
        self._train_engine = None
        self._bogie_mapper = None
        self._initialized = False

        # Alert history
        self._alert_history: List[FusedAlert] = []

        # Wire up event bus
        self.event_bus.subscribe("cctv.anomaly", self._on_cctv_signal)
        self.event_bus.subscribe("railway.context", self._on_context_signal)

    def initialize(self):
        if self._initialized:
            return

        logger.info("Initializing Multi-Source Fusion Engine...")

        from backend.services.train_intelligence import TrainIntelligence
        self._train_engine = TrainIntelligence()
        self._train_engine.initialize(train_count=300)

        from backend.services.bogie_mapper import BogieMapper
        self._bogie_mapper = BogieMapper(ocr_engine="pattern")

        self._initialized = True
        logger.info("Fusion Engine ready (4 CCTV sources online)")

    # ------------------------------------------------------------------
    # CORE: FUSE EVENT
    # ------------------------------------------------------------------

    def fuse_event(
        self,
        camera_id: str = "",
        timestamp: str = "",
        anomaly_type: str = "behavioral_anomaly",
        anomaly_confidence: float = 0.5,
        person_count: int = 1,
        # ── CCTV Metadata (pre-attached to footage, NOT derived) ─────────────
        # These values come directly from VideoMetadata sidecar files.
        # The pipeline MUST NOT re-derive them from camera IDs or the DB.
        platform_number: int = 1,     # VideoMetadata.platform_number
        date: str = "",              # VideoMetadata.date  (YYYY-MM-DD)
        time: str = "",              # VideoMetadata.time  (HH:MM)
        day: str = "",               # VideoMetadata.day   (e.g. "Tuesday")
        # ── Optional overrides ──────────────────────────────────────────
        station_code: str = None,
        train_number: str = None,
        coach: str = None,
    ) -> FusedAlert:
        """
        Fuse a real-time CCTV event with passenger booking intelligence using
        the 60-40 multi-model weighting architecture.

        CORRECTED PIPELINE:
          Step 1. Receive CCTV anomaly score.
          Step 2. Read pre-attached CCTV metadata:
                  platform_number, date, time, day  ← from VideoMetadata sidecar
          Step 3. If CCTV score >= 60%: use metadata to query `trains` table.
                  Match: platform_number + day (short abbr) + closest departure_time.
          Step 4. Derive train_number and train_name from query result.
          Step 5. Query tickets_1–tickets_8 via ML engine for this train + date.
          Step 6. Fetch passenger risk scores.
          Step 7. Final Score = 60% CCTV + 40% Tickets.
        """
        from backend.database import db_service
        from backend.services.tickets_intelligence import tickets_intelligence
        from sqlalchemy import text

        # ──────────────────────────────────────────────────────────────
        # Step 1: CCTV Score
        # ──────────────────────────────────────────────────────────────
        cctv_score = min(max(anomaly_confidence, 0.0), 1.0)
        cctv_signal = Signal(
            signal_id=f"CCTV_{uuid.uuid4().hex[:6]}",
            source=SignalSource.CCTV.value,
            timestamp=timestamp,
            confidence=cctv_score,
            payload={"anomaly_type": anomaly_type, "camera_id": camera_id, "person_count": person_count},
        )
        self.event_bus.publish("cctv.anomaly", cctv_signal)

        # ──────────────────────────────────────────────────────────────
        # Step 2: Read pre-attached CCTV metadata (authoritative source of truth)
        # Do NOT derive these from camera_id, timestamp parsing, or the database.
        # ──────────────────────────────────────────────────────────────
        resolved_station = station_code or "SC"
        station_name = self._get_station_name(resolved_station)
        zone = self._resolve_zone(camera_id)   # zone used for coach estimation only

        # Normalise day to 3-letter abbreviation for trains table matching.
        # VideoMetadata stores full day names ("Tuesday"), trains table uses "Tue".
        _DAY_SHORT = {
            "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed",
            "thursday": "Thu", "friday": "Fri", "saturday": "Sat", "sunday": "Sun",
        }
        day_abbr = _DAY_SHORT.get(day.lower(), day[:3].capitalize()) if day else "Mon"

        # Use metadata values directly — no parsing, no derivation
        jrny_date = date        # YYYY-MM-DD from VideoMetadata
        event_time_str = time   # HH:MM from VideoMetadata

        fusion_reasoning = [
            f"[CCTV] Behavioral anomaly detected on camera {camera_id} (Platform {platform_number})",
            f"[CCTV] CCTV anomaly score: {cctv_score:.2%}",
            f"[META] CCTV metadata: Platform {platform_number}, {day} ({day_abbr}) on {jrny_date} at {event_time_str}.",
        ]

        train_number_str = ""
        train_name_str = ""

        # ──────────────────────────────────────────────────────────────
        # Step 3: CCTV score >= 60% triggers trains table lookup
        # Query uses platform_number + day_abbr + closest departure_time
        # ──────────────────────────────────────────────────────────────
        if cctv_score >= 0.60:
            fusion_reasoning.append(
                f"[META] Anomaly score exceeds 60% threshold. "
                f"Querying trains table for Platform {platform_number}, {day_abbr} at {event_time_str}."
            )
            # If train_number is explicitly provided, resolve name only
            if train_number:
                train_number_str = str(train_number)
                try:
                    query = text("SELECT train_name FROM trains WHERE train_number = :tn LIMIT 1")
                    with db_service.engine.connect() as conn:
                        res = conn.execute(query, {"tn": train_number_str}).fetchone()
                        if res:
                            train_name_str = res[0]
                except Exception:
                    pass
            else:
                # Query trains table using CCTV metadata (platform + day + time)
                try:
                    query = text("""
                        SELECT train_number, train_name, departure_days, departure_time
                        FROM trains
                        WHERE platform_number = :plat
                    """)
                    with db_service.engine.connect() as conn:
                        rows = conn.execute(query, {"plat": platform_number}).fetchall()

                    # Filter trains that run on this specific weekday
                    candidates = [
                        dict(row._mapping)
                        for row in rows
                        if day_abbr in [d.strip() for d in dict(row._mapping).get("departure_days", "").split(",")]
                    ]

                    if candidates:
                        # Pick the train whose departure_time is closest to the CCTV event time
                        def time_diff(t1_str, t2_str):
                            try:
                                def to_secs(s):
                                    parts = str(s).split(":")
                                    h, m = int(parts[0]), int(parts[1])
                                    sec_val = int(parts[2]) if len(parts) > 2 else 0
                                    return h * 3600 + m * 60 + sec_val
                                return abs(to_secs(t1_str) - to_secs(t2_str))
                            except Exception:
                                return 999999

                        candidates.sort(
                            key=lambda c: time_diff(c["departure_time"], event_time_str)
                        )
                        best_match = candidates[0]
                        train_number_str = str(best_match["train_number"])
                        train_name_str = best_match["train_name"]
                        fusion_reasoning.append(
                            f"[TRAIN] Resolved active train: Train {train_number_str} "
                            f"({train_name_str}) — closest departure to {event_time_str} "
                            f"on Platform {platform_number} ({day_abbr})."
                        )
                    else:
                        fusion_reasoning.append(
                            f"[TRAIN] No scheduled trains found on Platform {platform_number} "
                            f"for day {day_abbr}. Check trains table coverage."
                        )
                except Exception as ex:
                    logger.error(f"Error querying trains table: {ex}")
                    fusion_reasoning.append(f"[TRAIN] Failed to query trains table: {ex}")
        else:
            fusion_reasoning.append(
                "[META] CCTV anomaly score is below 60% threshold. Ticket database correlation skipped."
            )

        # Step 4, 5 & 6: Tickets DB Anomaly Score
        tickets_score = 0.0
        pax_list = []
        tickets_reason = ""

        if train_number_str:
            # Query Ticket Booking ML outlier models
            try:
                res = tickets_intelligence.get_train_risk_score(train_number_str, jrny_date)
                tickets_score = res.get("tickets_score", 0.0)
                pax_list = res.get("passengers", [])
                tickets_reason = res.get("reason", "")
                
                if tickets_score > 0.0:
                    fusion_reasoning.append(
                        f"[TICKETS] Triggered Tickets DB anomaly engine for Train {train_number_str} on {jrny_date}."
                    )
                    fusion_reasoning.append(
                        f"[TICKETS] Max passenger risk score: {tickets_score * 10:.1f}/10.0 — {tickets_reason}"
                    )
                else:
                    fusion_reasoning.append(
                        f"[TICKETS] Tickets DB engine run complete: No booking outliers flagged for Train {train_number_str} on {jrny_date}."
                    )
            except Exception as ex:
                logger.error(f"Error querying tickets intelligence: {ex}")
                fusion_reasoning.append(f"[TICKETS] Error during ticket correlation: {ex}")

        # Step 7: 60-40 Weighting Score Calculation
        final_score = 0.60 * cctv_score + 0.40 * tickets_score
        
        # Calculate coach estimation
        estimated_coach_str = coach
        if not estimated_coach_str:
            coach_result = self._bogie_mapper.estimate_from_zone(zone or "mid")
            estimated_coach_str = coach_result.estimated_coach or "GEN"

        # Determine severity and recommended actions
        if final_score >= 0.70:
            severity = "CRITICAL"
        elif final_score >= 0.50:
            severity = "HIGH"
        elif final_score >= 0.30:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Build clean reasons summary for DB and alert
        if len(pax_list) > 0:
            reasons_summary = f"CCTV Anomaly + Tickets Outlier: {tickets_reason}"
        else:
            reasons_summary = f"CCTV Anomaly: {anomaly_type}"

        # Initialize FusedAlert
        alert = FusedAlert(
            alert_id=f"FUSED_{uuid.uuid4().hex[:10].upper()}",
            generated_at=datetime.utcnow().isoformat() + "Z",
            severity=severity,
            fused_confidence=final_score,
            source_contributions={
                "cctv_anomaly": round(cctv_score * 0.60, 4),
                "tickets_anomaly": round(tickets_score * 0.40, 4)
            },
            agreement_score=0.85 if tickets_score > 0.0 else 0.50,
            evidence_mass=final_score,
            station_code=resolved_station,
            station_name=station_name,
            platform=platform_number,
            train_number=train_number_str,
            train_name=train_name_str,
            estimated_coach=estimated_coach_str,
            anomaly_type=anomaly_type,
            timestamp=timestamp,
            suspects=len(pax_list) if pax_list else 1,
            fusion_reasoning=fusion_reasoning + [
                f"[FUSION] Multi-Model Score = 60% CCTV ({cctv_score:.1%}) + 40% Tickets ({tickets_score:.1%}) = {final_score:.2%}"
            ]
        )

        alert.recommended_action = self._action(severity, alert)
        
        # Add high-risk passenger list details in signals payload
        meta_signals = []
        if resolved_station and platform_number:
            meta_signals.append({
                "signal_id": f"META_{uuid.uuid4().hex[:6]}",
                "source": "railway_metadata",
                "timestamp": timestamp,
                "confidence": 0.95,
                "payload": {"station_code": resolved_station, "platform": platform_number, "zone": zone, "station_name": station_name}
            })
        if train_number_str:
            meta_signals.append({
                "signal_id": f"TRAIN_{uuid.uuid4().hex[:6]}",
                "source": "train_intelligence",
                "timestamp": timestamp,
                "confidence": 0.95,
                "payload": {"train_number": train_number_str, "train_name": train_name_str}
            })
        if pax_list:
            meta_signals.append({
                "signal_id": f"TICKETS_{uuid.uuid4().hex[:6]}",
                "source": "tickets_intelligence",
                "timestamp": timestamp,
                "confidence": tickets_score,
                "payload": {"passengers": pax_list}
            })

        alert.source_signals = [cctv_signal.to_dict()] + meta_signals

        # Step 8: Insert live Alert into Supabase Alerts Database
        if final_score > 0.75:
            try:
                alert_dict = {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "alert_type": alert.anomaly_type,
                    "station_code": alert.station_code,
                    "platform": alert.platform,
                    "train_number": alert.train_number,
                    "coach": alert.estimated_coach,
                    "suspect_description": reasons_summary,
                    "fusion_confidence": alert.fused_confidence,
                    "source_scores": {
                        "cctv_score": cctv_score,
                        "tickets_score": tickets_score
                    },
                    "triggered_rules": [
                        "CCTV Anomaly Detection",
                        "Tickets Database Correlation"
                    ] if tickets_score > 0.0 else ["CCTV Anomaly Detection"],
                    "xai_explanation": "\n".join(alert.fusion_reasoning),
                    "intervention_protocol": alert.recommended_action,
                    "status": "active"
                }
                db_service.insert_alert(alert_dict)
                logger.info(f"  ✓ Live Fused Alert {alert.alert_id} successfully persisted in Supabase database")
            except Exception as e:
                logger.error(f"Failed to persist alert in database: {e}")
        else:
            logger.info(f"  Alert {alert.alert_id} not persisted (confidence {final_score:.2%} <= 75% threshold)")

        # Keep history
        self._alert_history.append(alert)
        self.event_bus.publish("fusion.alert", Signal(
            signal_id=alert.alert_id,
            source="fusion_engine",
            timestamp=alert.generated_at,
            confidence=alert.fused_confidence,
            payload={"severity": alert.severity},
        ))

        # ── WebSocket broadcast (real-time dashboard push) ────────────────
        # Fires if the app registered a broadcast callback on startup.
        # Only broadcasts HIGH/CRITICAL alerts to avoid dashboard noise.
        if final_score >= 0.50 and hasattr(self, "_ws_broadcast_callback") and callable(self._ws_broadcast_callback):
            try:
                self._ws_broadcast_callback(alert.to_dict())
                logger.info(f"  📡 Alert {alert.alert_id} broadcast via WebSocket")
            except Exception as ws_err:
                logger.warning(f"  WebSocket broadcast error: {ws_err}")

        logger.info(
            f"FUSED 60-40 | {alert.alert_id} | {alert.severity} | "
            f"CCTV={cctv_score:.0%} | Tickets={tickets_score:.0%} | Final={alert.fused_confidence:.0%}"
        )

        return alert

    # ------------------------------------------------------------------
    # BATCH FUSION
    # ------------------------------------------------------------------

    def fuse_batch(self, events: List[Dict]) -> List[FusedAlert]:
        """Fuse a batch of events. Returns alerts sorted by confidence."""
        alerts = [self.fuse_event(**e) for e in events]
        alerts.sort(key=lambda a: a.fused_confidence, reverse=True)
        return alerts

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        return [a.to_dict() for a in self._alert_history[-limit:]]

    def get_fusion_stats(self) -> Dict:
        if not self._alert_history:
            return {"total_alerts": 0}
        severities = defaultdict(int)
        for a in self._alert_history:
            severities[a.severity] += 1
        confs = [a.fused_confidence for a in self._alert_history]
        return {
            "total_alerts": len(self._alert_history),
            "severity_distribution": dict(severities),
            "avg_confidence": round(sum(confs) / len(confs), 4),
            "max_confidence": round(max(confs), 4),
            "total_events_processed": self.event_bus.total_events,
            "active_incident_clusters": len(self.correlator.get_active_clusters()),
        }

    def get_event_bus_stats(self) -> Dict:
        return {
            "total_events": self.event_bus.total_events,
            "recent_events": self.event_bus.get_event_log(10),
            "clusters": self.correlator.get_active_clusters(),
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _resolve_station(self, cam_id: str) -> str:
        import re
        m = re.match(r'CAM_([A-Z]+)_', cam_id)
        return m.group(1) if m else "SC"

    def _resolve_platform(self, cam_id: str) -> int:
        import re
        m = re.search(r'P(\d+)', cam_id)
        return int(m.group(1)) if m else 1

    def _resolve_zone(self, cam_id: str) -> str:
        return {"A": "entry", "B": "mid", "C": "exit"}.get(
            cam_id[-1] if cam_id else "", "mid"
        )

    def _get_station_name(self, code: str) -> str:
        try:
            from backend.services.railway_stations import STATION_LOOKUP
            s = STATION_LOOKUP.get(code)
            return s.name if s else code
        except Exception:
            return code

    def _on_cctv_signal(self, signal: Signal):
        pass  # Hook for async processing extensions

    def _on_context_signal(self, signal: Signal):
        pass  # Hook for async processing extensions

    def _action(self, severity: str, alert: FusedAlert) -> str:
        templates = {
            "CRITICAL": (
                "IMMEDIATE: Dispatch RPF to Platform {p}, Coach {c}. "
                "Verify {s} suspects. Activate all cameras."
            ),
            "HIGH": (
                "URGENT: Alert RPF at Platform {p}. "
                "Monitor Coach {c}. Prepare identity verification."
            ),
            "MEDIUM": (
                "MONITOR: Enhanced CCTV on Platform {p}, Coach {c}. "
                "RPF standby."
            ),
            "LOW": "ROUTINE: Log for pattern analysis.",
        }
        return templates.get(severity, templates["LOW"]).format(
            p=alert.platform, c=alert.estimated_coach or "?",
            s=alert.suspects,
        )
