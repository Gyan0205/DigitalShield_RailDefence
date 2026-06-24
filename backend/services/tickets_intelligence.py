"""
Digital Shield Rail Defense — Tickets Intelligence Service
===========================================================
Loads, caches, and queries ticket risk scores computed by
the ML Outlier Detection engine across ALL 8 ticket tables.

Caching strategy:
  1. Check Redis for cached scored DataFrame (TTL: 24h)
  2. If cache miss: run full ML pipeline → store to Redis
  3. If Redis unavailable: run in-memory only (no persistence)

Query is O(1) — instant filter on the in-memory DataFrame.
"""

import os
import io
import logging
import pandas as pd

logger = logging.getLogger("tickets_intelligence")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)

# Redis cache key and TTL
_CACHE_KEY = "ds:ticket_scores:v2"
_CACHE_TTL_SECONDS = 86400  # 24 hours


def _get_redis():
    """Get a Redis connection from environment, or None if unavailable."""
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        r.ping()  # Verify connection
        return r
    except Exception as e:
        logger.warning(f"  Redis unavailable ({e}). Running in pure in-memory mode.")
        return None


def _load_from_redis(r) -> pd.DataFrame:
    """Deserialize scored DataFrame from Redis. Returns None on miss/error."""
    try:
        data = r.get(_CACHE_KEY)
        if data is None:
            return None
        df = pd.read_parquet(io.BytesIO(data))
        logger.info(f"  ✓ Loaded {len(df):,} scored records from Redis cache (key={_CACHE_KEY})")
        return df
    except Exception as e:
        logger.warning(f"  Redis cache read failed: {e}")
        return None


def _save_to_redis(r, df: pd.DataFrame):
    """Serialize and store scored DataFrame to Redis with TTL."""
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        r.setex(_CACHE_KEY, _CACHE_TTL_SECONDS, buf.getvalue())
        size_kb = buf.tell() / 1024
        logger.info(f"  ✓ Scored DataFrame cached to Redis ({size_kb:.1f} KB, TTL={_CACHE_TTL_SECONDS}s)")
    except Exception as e:
        logger.warning(f"  Redis cache write failed (non-critical): {e}")


def _run_ml_pipeline() -> pd.DataFrame:
    """
    Execute the full ML pipeline across ALL 8 ticket tables.

    Steps:
      1. load_data()        — UNION ALL tickets_1..tickets_8
      2. clean_data()       — fix nulls, types
      3. engineer_features()— 10 risk features
      4. prepare_features() — RobustScaler
      5. compute_risk_scores()— IsolationForest + LOF
      6. apply_hybrid_scoring()— boost/reducer multipliers
      7. generate_reasons() — text explanations
    """
    import sys
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ticket_model_path = os.path.join(root_dir, "ticket_model")
    if ticket_model_path not in sys.path:
        sys.path.insert(0, ticket_model_path)

    from src.data_loader import load_data
    from src.cleaner import clean_data
    from src.features import engineer_features
    from src.model import prepare_features, compute_risk_scores
    from src.scoring import apply_hybrid_scoring
    from src.explainer import generate_reasons

    logger.info("  🚀 Running ML Outlier Passenger Engine across ALL 8 ticket tables...")

    df = load_data()
    logger.info(f"  [1/6] Loaded {len(df):,} records from all ticket tables")

    df = clean_data(df)
    logger.info(f"  [2/6] Cleaned data: {len(df):,} records remain")

    df = engineer_features(df)
    logger.info(f"  [3/6] Feature engineering complete ({len(df.columns)} features)")

    X_scaled = prepare_features(df)
    logger.info(f"  [4/6] Feature matrix prepared: {X_scaled.shape}")

    risk_scores, lof_flags = compute_risk_scores(X_scaled)
    logger.info(f"  [5/6] IsolationForest + LOF scoring complete")

    df = apply_hybrid_scoring(df, risk_scores, lof_flags)
    logger.info(f"  [6/6] Hybrid boost/reducer scoring applied")

    df = generate_reasons(df)
    logger.info("  XAI reasons generated")

    df.columns = [c.lower() for c in df.columns]
    return df


class TicketsIntelligenceService:
    """
    Infers ticket-booking anomalous behavior from the ML scoring pipeline.
    Caches results in Redis (24h TTL) for instant O(1) query performance.

    Flow:
      initialize() → check Redis cache → if miss: run ML pipeline → cache
      get_train_risk_score(train, date) → filter in-memory DataFrame → return
    """

    def __init__(self):
        self._df: pd.DataFrame = None
        self._initialized: bool = False

    def initialize(self, force: bool = False):
        """Load scored ticket data — from Redis cache if available, else run ML pipeline."""
        if self._initialized and not force:
            return

        logger.info("Initializing Tickets Intelligence Service...")

        try:
            # ── Try Redis cache first ─────────────────────────────────────
            redis_client = _get_redis()
            if redis_client and not force:
                cached_df = _load_from_redis(redis_client)
                if cached_df is not None:
                    self._df = cached_df
                    self._initialized = True
                    logger.info(f"  ✓ Initialized from Redis cache ({len(self._df):,} records)")
                    return

            # ── Cache miss or force refresh: run full ML pipeline ─────────
            df = _run_ml_pipeline()
            self._df = df

            # ── Store result to Redis for next startup ────────────────────
            if redis_client:
                _save_to_redis(redis_client, df)

            self._initialized = True
            logger.info(f"  ✓ In-Memory ML cache loaded: {len(self._df):,} scored records ready")

        except Exception as e:
            logger.error(f"  ❌ Ticket ML Pipeline failed: {e}", exc_info=True)
            self._df = None
            self._initialized = False

    def invalidate_cache(self):
        """Clear Redis cache and force re-run on next initialize()."""
        try:
            r = _get_redis()
            if r:
                r.delete(_CACHE_KEY)
                logger.info("Redis ticket cache invalidated")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
        self._df = None
        self._initialized = False

    def get_cache_status(self) -> dict:
        """Return current cache status for the health endpoint."""
        redis_client = _get_redis()
        redis_ok = redis_client is not None
        redis_ttl = None
        if redis_ok:
            try:
                redis_ttl = redis_client.ttl(_CACHE_KEY)
            except Exception:
                pass
        return {
            "initialized": self._initialized,
            "records_in_memory": len(self._df) if self._df is not None else 0,
            "redis_available": redis_ok,
            "redis_cache_ttl_seconds": redis_ttl,
        }

    def get_train_risk_score(self, train_number: str, jrny_date: str) -> dict:
        """
        Query risk scores for a specific train and travel date.

        Args:
            train_number: Target train number (e.g. "12727")
            jrny_date:    Travel date in YYYY-MM-DD format

        Returns:
            {
              "tickets_score": float [0.0, 1.0],
              "reason":        str   (aggregated XAI reasons),
              "passengers":    list  (up to 5 highest-risk passenger records),
              "table_sources": list  (which ticket tables contributed),
            }
        """
        self.initialize()

        if not self._initialized or self._df is None:
            return {
                "tickets_score": 0.0,
                "reason": "Ticket ML engine not initialized.",
                "passengers": [],
                "table_sources": [],
            }

        required = {"train_number", "jrny_date", "risk_score"}
        if not required.issubset(set(self._df.columns)):
            missing = required - set(self._df.columns)
            logger.error(f"  ⚠ Scored DataFrame missing columns: {missing}")
            return {
                "tickets_score": 0.0,
                "reason": f"Scored data missing required columns: {missing}",
                "passengers": [],
                "table_sources": [],
            }

        # ── Build filter mask (handle both int and str train numbers) ──────
        try:
            train_num_int = int(train_number)
        except (ValueError, TypeError):
            train_num_int = None

        if train_num_int is not None:
            mask = (
                (self._df["train_number"].astype(str) == str(train_number)) |
                (self._df["train_number"] == train_num_int)
            ) & (self._df["jrny_date"] == str(jrny_date))
        else:
            mask = (
                (self._df["train_number"].astype(str) == str(train_number)) &
                (self._df["jrny_date"] == str(jrny_date))
            )

        matched = self._df[mask]

        if matched.empty:
            logger.info(f"  No passenger bookings for Train {train_number} on {jrny_date}")
            return {
                "tickets_score": 0.0,
                "reason": f"No passenger bookings flagged for Train {train_number} on {jrny_date}.",
                "passengers": [],
                "table_sources": [],
            }

        # ── Normalize max risk score (0-10 → 0.0-1.0) ─────────────────────
        max_risk = float(matched["risk_score"].max())
        normalized_score = round(min(max(max_risk / 10.0, 0.0), 1.0), 4)

        # ── Top 5 highest-risk passengers ──────────────────────────────────
        top_pax = matched.sort_values("risk_score", ascending=False).head(5)
        pax_list = []
        for _, row in top_pax.iterrows():
            pax_list.append({
                "pnr":        str(row.get("pnrno", "")),
                "user_id":    str(row.get("user_id", "")),
                "name":       str(row.get("psgn_name", "")),
                "age":        int(row.get("age", 0)) if pd.notna(row.get("age")) else 0,
                "sex":        str(row.get("sex", "")),
                "risk_score": round(float(row.get("risk_score", 0.0)), 2),
                "reason":     str(row.get("reason", "")),
                "source_table": str(row.get("source_table", "")),
            })

        # ── Aggregate XAI reasons ──────────────────────────────────────────
        critical = matched[matched["risk_score"] >= 6.0]["reason"].dropna().unique()
        if len(critical) > 0:
            reasons_summary = " | ".join(critical[:3])
        else:
            reasons_summary = f"Normal booking patterns (Max risk: {max_risk:.1f}/10.0)"

        # ── Which tables contributed ───────────────────────────────────────
        table_sources = []
        if "source_table" in matched.columns:
            table_sources = sorted(matched["source_table"].dropna().unique().tolist())

        logger.info(
            f"  Train {train_number} on {jrny_date}: "
            f"tickets_score={normalized_score:.2%}, "
            f"matched={len(matched)} records, "
            f"tables={table_sources}"
        )

        return {
            "tickets_score": normalized_score,
            "reason": reasons_summary,
            "passengers": pax_list,
            "table_sources": table_sources,
        }


# Singleton — shared across the entire application
tickets_intelligence = TicketsIntelligenceService()
