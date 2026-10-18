-- Learning history is deliberately separate from authentication sessions.
ALTER TABLE teachcopilot.learners ADD COLUMN memory_source_revision BIGINT NOT NULL DEFAULT 0;

CREATE TABLE teachcopilot.learning_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    actor_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    client_session_id TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','completed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE(id,learner_id),
    UNIQUE(learner_id,actor_id,client_session_id)
);
CREATE INDEX learning_sessions_learner_topic_time_idx ON teachcopilot.learning_sessions(learner_id,topic_id,created_at DESC);

CREATE TABLE teachcopilot.memory_sources (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    source_order BIGSERIAL NOT NULL UNIQUE,
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    learning_session_id UUID,
    actor_id TEXT NOT NULL,
    author_role TEXT NOT NULL CHECK (author_role IN ('student','teacher','assistant')),
    content TEXT NOT NULL CHECK (length(content)>0),
    observed_on TEXT,
    approved BOOLEAN NOT NULL DEFAULT FALSE,
    source_key TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id,learner_id),
    UNIQUE(learner_id,source_key),
    FOREIGN KEY(learning_session_id,learner_id) REFERENCES teachcopilot.learning_sessions(id,learner_id) ON DELETE CASCADE,
    CHECK (NOT approved OR author_role <> 'assistant')
);
CREATE INDEX memory_sources_learner_topic_time_idx ON teachcopilot.memory_sources(learner_id,topic_id,source_order);

CREATE TABLE teachcopilot.memory_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('memory_update','next_step','chat')),
    method TEXT NOT NULL DEFAULT 'tail' CHECK (method IN ('profile_only','tail','summary','structured')),
    provider TEXT NOT NULL DEFAULT 'local' CHECK (provider IN ('local','external')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','completed','failed','unknown_outcome')),
    priority SMALLINT NOT NULL DEFAULT 0,
    base_revision INTEGER NOT NULL,
    source_revision BIGINT NOT NULL,
    idempotency_key TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    worker_id TEXT,
    lease_until TIMESTAMPTZ,
    call_started_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE(id,learner_id),
    UNIQUE(learner_id,kind,idempotency_key)
);
CREATE INDEX memory_jobs_queue_idx ON teachcopilot.memory_jobs(priority DESC,created_at) WHERE status='queued';
CREATE INDEX memory_jobs_learner_status_idx ON teachcopilot.memory_jobs(learner_id,status);

CREATE TABLE teachcopilot.learning_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learning_session_id UUID NOT NULL,
    learner_id TEXT NOT NULL,
    client_message_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    assistant_source_id TEXT,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','completed','failed','unknown_outcome')),
    response JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE(learning_session_id,client_message_id),
    FOREIGN KEY(learning_session_id,learner_id) REFERENCES teachcopilot.learning_sessions(id,learner_id) ON DELETE CASCADE,
    FOREIGN KEY(source_id,learner_id) REFERENCES teachcopilot.memory_sources(id,learner_id),
    FOREIGN KEY(assistant_source_id,learner_id) REFERENCES teachcopilot.memory_sources(id,learner_id)
);

CREATE TABLE teachcopilot.memory_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    base_revision INTEGER NOT NULL,
    source_revision BIGINT NOT NULL,
    state JSONB NOT NULL,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    approved BOOLEAN NOT NULL DEFAULT FALSE,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    job_id UUID UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id,learner_id),
    FOREIGN KEY(job_id,learner_id) REFERENCES teachcopilot.memory_jobs(id,learner_id)
);
CREATE INDEX memory_versions_learner_topic_time_idx ON teachcopilot.memory_versions(learner_id,topic_id,created_at DESC);

CREATE TABLE teachcopilot.next_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    base_revision INTEGER NOT NULL,
    source_revision BIGINT NOT NULL,
    method TEXT NOT NULL,
    result JSONB NOT NULL,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    memory_version_id UUID,
    job_id UUID UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY(memory_version_id,learner_id) REFERENCES teachcopilot.memory_versions(id,learner_id),
    FOREIGN KEY(job_id,learner_id) REFERENCES teachcopilot.memory_jobs(id,learner_id)
);
CREATE INDEX next_steps_learner_topic_time_idx ON teachcopilot.next_steps(learner_id,topic_id,created_at DESC);
