-- Round 2 unified memory backend (client A, zcode/r2-memory-20261117).
-- Additive only: durable pre-login lesson assignments, a deterministic task
-- catalog column on saved lesson versions, a durable identity claim for
-- concurrent session processing, and a stable source-order key for the
-- durable personal change replay. Existing tables keep their shape except for
-- the two documented additive columns.

-- 1. Stable replay order for durable personal change events. created_at is a
--    transaction timestamp and can collide for events written in the same
--    instant; the effective projection must replay in actual insertion order.
ALTER TABLE teachcopilot.unified_personal_changes
    ADD COLUMN IF NOT EXISTS event_seq BIGINT;

WITH ordered AS (
    SELECT id, (ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC))::bigint AS rn
    FROM teachcopilot.unified_personal_changes
)
UPDATE teachcopilot.unified_personal_changes c
    SET event_seq = ordered.rn
    FROM ordered
    WHERE c.id = ordered.id AND c.event_seq IS NULL;

ALTER TABLE teachcopilot.unified_personal_changes
    ALTER COLUMN event_seq SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS unified_personal_changes_event_seq_key
    ON teachcopilot.unified_personal_changes(event_seq);
CREATE SEQUENCE IF NOT EXISTS teachcopilot.unified_personal_changes_event_seq
    START WITH 1000000 INCREMENT BY 1;
ALTER SEQUENCE teachcopilot.unified_personal_changes_event_seq
    OWNED BY teachcopilot.unified_personal_changes.event_seq;
ALTER TABLE teachcopilot.unified_personal_changes
    ALTER COLUMN event_seq
    SET DEFAULT nextval('teachcopilot.unified_personal_changes_event_seq');

-- 2. Durable pre-login lesson assignments (P0.2). An assignment exists before
--    any student session; the approved lesson revision is frozen at assignment
--    time and lesson content is never mutated to represent an assignment.
CREATE TABLE teachcopilot.unified_lesson_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    lesson_id UUID NOT NULL,
    lesson_revision INTEGER NOT NULL CHECK (lesson_revision >= 1),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','started','completed')),
    learning_session_id UUID,
    assigned_by TEXT NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_by TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    request_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, learner_id),
    UNIQUE(learner_id, lesson_id),
    UNIQUE(learner_id, request_id),
    FOREIGN KEY (lesson_id, learner_id)
        REFERENCES teachcopilot.unified_lessons(id, learner_id) ON DELETE CASCADE,
    FOREIGN KEY (learning_session_id, learner_id)
        REFERENCES teachcopilot.learning_sessions(id, learner_id) ON DELETE CASCADE
);
CREATE INDEX unified_lesson_assignments_learner_idx
    ON teachcopilot.unified_lesson_assignments(learner_id, status, assigned_at DESC);

-- 3. Deterministic per-lesson-version task catalog (P1.1). NULL means the
--    lesson version was saved before the catalog existed; an explicit
--    worker/admin operation may backfill, a read never writes.
ALTER TABLE teachcopilot.unified_lessons
    ADD COLUMN IF NOT EXISTS task_catalog JSONB;

-- 4. Durable identity claim for session processing (P1.2). The pre-insert run
--    lookup alone is not concurrency control; the claim row serializes
--    identical-snapshot processing with a bounded lease and no long row
--    transaction across model calls.
CREATE TABLE teachcopilot.unified_processing_claims (
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    source_fingerprint TEXT NOT NULL,
    prompt_recipe TEXT NOT NULL,
    model_identity TEXT NOT NULL,
    claim_token UUID NOT NULL DEFAULT gen_random_uuid(),
    claimed_by TEXT NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (learner_id, session_id, source_fingerprint, prompt_recipe, model_identity)
);
CREATE INDEX unified_processing_claims_lease_idx
    ON teachcopilot.unified_processing_claims(lease_expires_at);
