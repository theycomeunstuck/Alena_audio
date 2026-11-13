-- Extractive summaries are immutable, scoped versions of completed sessions.
CREATE TABLE teachcopilot.session_summary_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('session', 'rolling7')),
    learning_session_id UUID,
    scope_fingerprint TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    payload JSONB NOT NULL,
    supersedes_version_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, learner_id),
    FOREIGN KEY(learning_session_id, learner_id)
        REFERENCES teachcopilot.learning_sessions(id, learner_id) ON DELETE CASCADE,
    FOREIGN KEY(supersedes_version_id, learner_id)
        REFERENCES teachcopilot.session_summary_versions(id, learner_id),
    CHECK ((kind='session' AND learning_session_id IS NOT NULL)
        OR (kind='rolling7' AND learning_session_id IS NULL))
);

CREATE UNIQUE INDEX session_summary_versions_session_input_idx
    ON teachcopilot.session_summary_versions
       (learner_id, learning_session_id, scope_fingerprint, input_fingerprint)
    WHERE kind='session';
CREATE UNIQUE INDEX session_summary_versions_rolling_input_idx
    ON teachcopilot.session_summary_versions (learner_id, scope_fingerprint, input_fingerprint)
    WHERE kind='rolling7';
CREATE INDEX session_summary_versions_latest_idx
    ON teachcopilot.session_summary_versions
       (learner_id, kind, scope_fingerprint, created_at DESC, id DESC);
CREATE INDEX learning_sessions_completed_summary_idx
    ON teachcopilot.learning_sessions(learner_id, (COALESCE(completed_at, created_at)) DESC, id DESC)
    WHERE status='completed';
