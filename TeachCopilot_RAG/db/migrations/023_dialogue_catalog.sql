-- One active classifier recipe, with replaceable actor-scoped derived labels.
-- Conversation sources and source embeddings are deliberately not modified.
CREATE TABLE teachcopilot.dialogue_catalog_classifier (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    generation BIGINT NOT NULL CHECK (generation > 0),
    recipe_id CHAR(64) NOT NULL,
    classifier JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE teachcopilot.dialogue_catalog_scopes (
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    actor_id TEXT NOT NULL,
    generation BIGINT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('ready','pending')),
    source_snapshot_fingerprint CHAR(64),
    source_watermark BIGINT,
    summary JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (learner_id,actor_id),
    UNIQUE (learner_id,actor_id,generation),
    CHECK (state <> 'ready' OR (source_snapshot_fingerprint IS NOT NULL AND summary IS NOT NULL))
);

CREATE TABLE teachcopilot.dialogue_catalog_chunks (
    learner_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    generation BIGINT NOT NULL,
    chunk_id CHAR(64) NOT NULL,
    text_sha256 CHAR(64) NOT NULL,
    source_ids TEXT[] NOT NULL,
    source_spans JSONB NOT NULL,
    activity_days DATE[] NOT NULL,
    labels TEXT[] NOT NULL,
    topic_scores JSONB NOT NULL,
    PRIMARY KEY (learner_id,actor_id,chunk_id),
    FOREIGN KEY (learner_id,actor_id,generation)
        REFERENCES teachcopilot.dialogue_catalog_scopes(learner_id,actor_id,generation) ON DELETE CASCADE
);
CREATE INDEX dialogue_catalog_chunks_labels_idx ON teachcopilot.dialogue_catalog_chunks USING GIN(labels);
CREATE INDEX dialogue_catalog_chunks_days_idx ON teachcopilot.dialogue_catalog_chunks USING GIN(activity_days);
