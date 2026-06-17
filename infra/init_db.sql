-- SENTINEL TimescaleDB schema
-- Runs automatically on first container start.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Raw sensor readings ingested from the stream
CREATE TABLE IF NOT EXISTS sensor_readings (
    time        TIMESTAMPTZ     NOT NULL,
    unit_id     INTEGER         NOT NULL,
    sensor_name TEXT            NOT NULL,
    value       DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('sensor_readings', by_range('time'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_readings_unit ON sensor_readings (unit_id, time DESC);

-- Anomaly scores produced by each model version
CREATE TABLE IF NOT EXISTS anomaly_scores (
    time            TIMESTAMPTZ     NOT NULL,
    unit_id         INTEGER         NOT NULL,
    model_version   TEXT            NOT NULL,   -- e.g. 'champion', 'challenger'
    modality        TEXT            NOT NULL,   -- 'timeseries' | 'audio'
    score           DOUBLE PRECISION NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    is_anomalous    BOOLEAN         NOT NULL,
    is_shadow       BOOLEAN         NOT NULL DEFAULT FALSE  -- TRUE = challenger (not served)
);
SELECT create_hypertable('anomaly_scores', by_range('time'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_scores_unit ON anomaly_scores (unit_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_scores_version ON anomaly_scores (model_version, time DESC);

-- Drift events logged by the monitor
CREATE TABLE IF NOT EXISTS drift_events (
    time                TIMESTAMPTZ     NOT NULL,
    n_drifted_sensors   INTEGER         NOT NULL,
    total_sensors       INTEGER         NOT NULL,
    drifted_sensors     TEXT[]          NOT NULL,
    share_drifted       DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('drift_events', by_range('time'), if_not_exists => TRUE);
