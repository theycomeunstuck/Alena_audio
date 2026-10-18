CREATE TABLE teachcopilot.source_snapshots (
    id BIGSERIAL PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_key TEXT NOT NULL,
    source_version TEXT NOT NULL,
    source_file TEXT NOT NULL,
    payload JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(source_kind,source_key,source_version)
);
CREATE INDEX ON teachcopilot.source_snapshots(source_kind,source_key);
