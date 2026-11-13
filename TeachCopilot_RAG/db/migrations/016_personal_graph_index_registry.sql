BEGIN;
CREATE TABLE IF NOT EXISTS teachcopilot.personal_graph_indexes (
    run_id text PRIMARY KEY,
    database_identity text NOT NULL,
    learner_id text NOT NULL,
    snapshot_fingerprint text NOT NULL,
    lessons_hash text NOT NULL,
    config_fingerprint text NOT NULL,
    state text NOT NULL CHECK (state IN ('building','ready','uncertain','invalid')),
    owner_token text,
    job_id text NOT NULL,
    lease_until timestamptz,
    manifest jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS personal_graph_indexes_lookup ON teachcopilot.personal_graph_indexes
    (database_identity, learner_id, snapshot_fingerprint, lessons_hash, config_fingerprint, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS personal_graph_indexes_active ON teachcopilot.personal_graph_indexes
    (database_identity, learner_id, snapshot_fingerprint, lessons_hash, config_fingerprint)
    WHERE state IN ('building','ready');
COMMIT;
