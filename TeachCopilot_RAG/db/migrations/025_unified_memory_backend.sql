-- Unified memory backend (client A, zcode-memory-backend-20261117).
-- Additive only: lesson artifacts, lesson state, personal change history,
-- academic candidates, session processing runs and summaries, and persisted
-- own-conversation search representations. Existing tables are unchanged
-- except for the documented queue constraint widening.

-- 1. Queue: accept the two new job kinds and let session processing use the
--    local personal_memory method. The structured_v2 and personal_graph
--    restrictions keep their original scope.
ALTER TABLE teachcopilot.memory_jobs
    DROP CONSTRAINT IF EXISTS memory_jobs_kind_check;
ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_kind_check
    CHECK (kind IN ('memory_update','next_step','chat','lesson_generate','session_process'));

ALTER TABLE teachcopilot.memory_jobs
    DROP CONSTRAINT IF EXISTS memory_jobs_personal_memory_job_kind_check;
ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_personal_memory_job_kind_check
    CHECK (method <> 'personal_memory'
           OR (kind IN ('next_step','session_process') AND provider = 'local'));

-- 2. Durable lesson versions (generation output). Drafts exist only with a
--    passed validation record; confirmation is a compare-and-set revision.
CREATE TABLE teachcopilot.unified_lessons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    minutes INTEGER NOT NULL CHECK (minutes BETWEEN 1 AND 480),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','approved')),
    lesson_text TEXT NOT NULL CHECK (length(lesson_text) > 0),
    answers_text TEXT NOT NULL DEFAULT '',
    validation JSONB NOT NULL,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer_sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    memory_revision TEXT,
    request_id TEXT NOT NULL,
    created_by TEXT NOT NULL,
    confirmed_by TEXT,
    confirmed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, learner_id),
    UNIQUE(learner_id, request_id)
);
CREATE INDEX unified_lessons_learner_topic_idx
    ON teachcopilot.unified_lessons(learner_id, topic_id, created_at DESC);

-- 3. Per-session guided lesson state. One attached lesson per session.
CREATE TABLE teachcopilot.unified_lesson_states (
    learning_session_id UUID PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    lesson_id UUID NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
    stage TEXT NOT NULL DEFAULT 'not_started'
        CHECK (stage IN ('not_started','warmup','explanation','practice','check','completed')),
    pending_task_id TEXT,
    last_question_source_id TEXT,
    completed_task_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    request_id TEXT,
    updated_by TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY(lesson_id, learner_id)
        REFERENCES teachcopilot.unified_lessons(id, learner_id) ON DELETE CASCADE,
    FOREIGN KEY(learning_session_id)
        REFERENCES teachcopilot.learning_sessions(id) ON DELETE CASCADE,
    CHECK (learning_session_id IS NOT NULL)
);
CREATE INDEX unified_lesson_states_learner_idx
    ON teachcopilot.unified_lesson_states(learner_id, updated_at DESC);

-- 4. Append-only personal change events with a current projection rebuilt
--    from events; teacher revert appends a reverting event, never rewrites.
CREATE TABLE teachcopilot.unified_personal_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    session_id UUID,
    run_id UUID,
    category TEXT NOT NULL CHECK (category IN ('interest','avoided_interest','preference')),
    operation TEXT NOT NULL CHECK (operation IN ('add','update','remove')),
    value TEXT NOT NULL CHECK (length(value) > 0),
    scope TEXT NOT NULL CHECK (scope IN ('durable','session')),
    status TEXT NOT NULL CHECK (status IN ('proposed','applied','rejected','reverted')),
    explicit_durable BOOLEAN NOT NULL DEFAULT FALSE,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    decided_by TEXT,
    decided_at TIMESTAMPTZ,
    request_id TEXT,
    reverts_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, learner_id),
    UNIQUE(learner_id, request_id)
);
CREATE INDEX unified_personal_changes_learner_status_idx
    ON teachcopilot.unified_personal_changes(learner_id, status, created_at DESC);

-- 5. Academic assessment candidates. Never effective learning state; the
--    teacher decision applies the existing canonical card rules.
CREATE TABLE teachcopilot.unified_academic_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    session_id UUID,
    run_id UUID,
    category TEXT NOT NULL CHECK (category IN ('knowledge_observation','error_pattern','learning_goal')),
    proposed_value JSONB NOT NULL,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','accepted','rejected')),
    decided_by TEXT,
    decided_at TIMESTAMPTZ,
    decision_note TEXT,
    request_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, learner_id),
    UNIQUE(learner_id, request_id)
);
CREATE INDEX unified_academic_candidates_learner_status_idx
    ON teachcopilot.unified_academic_candidates(learner_id, status, created_at DESC);

-- 6. Session processing runs. One durable artifact per exact snapshot,
--    recipe and model identity; reruns of the same identity return it.
CREATE TABLE teachcopilot.unified_processing_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed','partial','failed')),
    provisional BOOLEAN NOT NULL DEFAULT FALSE,
    source_fingerprint TEXT NOT NULL,
    prompt_recipe TEXT NOT NULL,
    model_identity TEXT NOT NULL,
    summary JSONB,
    coverage JSONB NOT NULL DEFAULT '{}'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    request_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE(id, learner_id),
    UNIQUE(learner_id, session_id, source_fingerprint, prompt_recipe, model_identity),
    UNIQUE(learner_id, request_id)
);
CREATE INDEX unified_processing_runs_session_idx
    ON teachcopilot.unified_processing_runs(learner_id, session_id, created_at DESC);

-- 7. Immutable summary versions. Earlier versions stay retrievable; the run
--    links to the version produced for its exact snapshot.
CREATE TABLE teachcopilot.unified_summary_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    run_id UUID,
    version_index INTEGER NOT NULL DEFAULT 1 CHECK (version_index >= 1),
    input_fingerprint TEXT NOT NULL,
    text TEXT NOT NULL,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    coverage JSONB NOT NULL DEFAULT '{}'::jsonb,
    provisional BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, learner_id),
    UNIQUE(learner_id, session_id, version_index, input_fingerprint)
);
CREATE INDEX unified_summary_versions_session_idx
    ON teachcopilot.unified_summary_versions(learner_id, session_id, version_index DESC);

-- 8. Persisted own-conversation search representations. One row per chunk
--    per actor scope; full-text and vector channels are both learner- and
--    actor-scoped before ranking. Embeddings are optional until indexed.
CREATE TABLE teachcopilot.unified_conversation_chunks (
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    actor_id TEXT NOT NULL,
    chunk_key TEXT NOT NULL,
    chunk_recipe TEXT NOT NULL,
    session_id UUID,
    chunk_index INTEGER NOT NULL,
    chunk_start INTEGER NOT NULL,
    chunk_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    source_ids JSONB NOT NULL,
    spans JSONB NOT NULL,
    activity_days TEXT[] NOT NULL DEFAULT '{}',
    first_source_order BIGINT NOT NULL DEFAULT 0,
    last_source_order BIGINT NOT NULL DEFAULT 0,
    search_document tsvector NOT NULL,
    embedding vector(1024),
    embedding_identity TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(learner_id, actor_id, chunk_key, chunk_recipe)
);
CREATE INDEX unified_conversation_chunks_fts_idx
    ON teachcopilot.unified_conversation_chunks USING gin(search_document);
CREATE INDEX unified_conversation_chunks_vector_idx
    ON teachcopilot.unified_conversation_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX unified_conversation_chunks_days_idx
    ON teachcopilot.unified_conversation_chunks(learner_id, actor_id, activity_days);

-- 9. Coordinator revision for personal change review (compare-and-set).
ALTER TABLE teachcopilot.learners ADD COLUMN unified_changes_revision INTEGER NOT NULL DEFAULT 0;
