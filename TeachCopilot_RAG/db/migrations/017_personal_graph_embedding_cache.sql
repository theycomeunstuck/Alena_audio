BEGIN;

CREATE TABLE IF NOT EXISTS teachcopilot.personal_graph_source_embedding_cache (
    learner_id TEXT NOT NULL,
    model_identity_digest TEXT NOT NULL,
    text_sha256 CHAR(64) NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (learner_id, model_identity_digest, text_sha256)
);

COMMIT;
