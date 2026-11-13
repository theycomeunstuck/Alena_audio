CREATE TABLE teachcopilot.conversation_summary_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    actor_id UUID NOT NULL REFERENCES teachcopilot.accounts(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('day', 'rolling7')),
    activity_day DATE,
    scope_fingerprint TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    source_watermark BIGINT NOT NULL,
    payload JSONB NOT NULL,
    supersedes_version_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, learner_id, actor_id),
    UNIQUE NULLS NOT DISTINCT (learner_id, actor_id, kind, activity_day, scope_fingerprint, input_fingerprint),
    FOREIGN KEY(supersedes_version_id, learner_id, actor_id)
        REFERENCES teachcopilot.conversation_summary_versions(id, learner_id, actor_id),
    CHECK ((kind='day' AND activity_day IS NOT NULL) OR (kind='rolling7' AND activity_day IS NULL))
);

CREATE INDEX conversation_summary_versions_latest_idx
    ON teachcopilot.conversation_summary_versions
       (learner_id, actor_id, kind, activity_day, scope_fingerprint, source_watermark DESC, created_at DESC, id DESC);

COMMENT ON TABLE teachcopilot.conversation_summary_versions IS
    'Append-only derived versions of an actor-scoped conversation. Originals remain in memory_sources.';
