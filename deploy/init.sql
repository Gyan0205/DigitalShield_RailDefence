-- ==============================================================
-- Digital Shield Rail Defense — PostgreSQL Init
-- ==============================================================
-- Runs automatically on first container start
-- ==============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";


-- ============================================================
-- CCTV Detections
-- ============================================================
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id VARCHAR(30) NOT NULL,
    station_code VARCHAR(10) NOT NULL,
    platform INTEGER,
    timestamp TIMESTAMP NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL,
    anomaly_confidence DECIMAL(5,4),
    bounding_box JSONB,
    track_id INTEGER,
    frame_number INTEGER,
    video_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Intelligence Alerts
-- ============================================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(30) UNIQUE NOT NULL,
    severity VARCHAR(15) NOT NULL CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
    alert_type VARCHAR(50) NOT NULL,
    station_code VARCHAR(10) NOT NULL,
    platform INTEGER,
    train_number VARCHAR(10),
    coach VARCHAR(10),
    suspect_description TEXT,
    fusion_confidence DECIMAL(5,4),
    source_scores JSONB,
    triggered_rules JSONB,
    xai_explanation TEXT,
    intervention_protocol TEXT,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'investigating', 'resolved', 'false_alarm')),
    assigned_to VARCHAR(100),
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Audit Log
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_role VARCHAR(20),
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(200),
    details JSONB,
    ip_address VARCHAR(45),
    request_id VARCHAR(50)
);

-- ============================================================
-- Train Schedule Cache
-- ============================================================
CREATE TABLE IF NOT EXISTS train_schedules (
    id SERIAL PRIMARY KEY,
    train_number VARCHAR(10) NOT NULL,
    train_name VARCHAR(100),
    station_code VARCHAR(10) NOT NULL,
    platform INTEGER,
    arrival_time TIME,
    departure_time TIME,
    days_of_week VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(train_number, station_code)
);

-- ============================================================
-- Model Metrics (Experiment Tracking)
-- ============================================================
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20),
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6),
    hyperparams JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_detections_station ON detections(station_code);
CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(anomaly_type);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_station ON alerts(station_code);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);

CREATE INDEX IF NOT EXISTS idx_schedules_station ON train_schedules(station_code);
CREATE INDEX IF NOT EXISTS idx_schedules_train ON train_schedules(train_number);

-- ============================================================
-- Seed: Secunderabad Station Trains
-- ============================================================
INSERT INTO train_schedules (train_number, train_name, station_code, platform, arrival_time, departure_time, days_of_week)
VALUES
    ('12727', 'AP Express', 'SC', 1, '06:25', '06:40', 'DAILY'),
    ('12728', 'AP Express', 'SC', 1, '18:50', NULL, 'DAILY'),
    ('12649', 'Karnataka Sampark Kranti', 'SC', 3, '07:00', '07:10', 'MTWTFS'),
    ('12650', 'Karnataka Sampark Kranti', 'SC', 3, '22:30', NULL, 'MTWTFS'),
    ('17018', 'Rayalaseema Express', 'SC', 5, '17:15', '17:30', 'DAILY'),
    ('17017', 'Rayalaseema Express', 'SC', 5, '06:00', NULL, 'DAILY'),
    ('12773', 'Shalimar Express', 'SC', 7, '09:40', '09:55', 'MTWF'),
    ('12774', 'Shalimar Express', 'SC', 7, '14:10', NULL, 'TTSS'),
    ('21357', 'Duronto Express', 'SC', 2, '22:10', '22:20', 'MWF'),
    ('21358', 'Duronto Express', 'SC', 2, '05:30', NULL, 'TTS'),
    ('12604', 'Hyderabad Exp', 'SC', 4, '16:05', '16:20', 'DAILY'),
    ('12603', 'Hyderabad Exp', 'SC', 4, '10:30', NULL, 'DAILY')
ON CONFLICT (train_number, station_code) DO NOTHING;

-- ============================================================
-- Confirmation
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE '==============================================';
    RAISE NOTICE 'Digital Shield Rail Defense DB initialized';
    RAISE NOTICE '  Tables: detections, alerts, audit_log,';
    RAISE NOTICE '          train_schedules, model_metrics';
    RAISE NOTICE '  Indexes: 12 performance indexes created';
    RAISE NOTICE '  Seed data: 12 SC trains loaded';
    RAISE NOTICE '==============================================';
END $$;
