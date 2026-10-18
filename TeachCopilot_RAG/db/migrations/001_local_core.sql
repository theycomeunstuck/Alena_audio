CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS teachcopilot;
CREATE TABLE IF NOT EXISTS teachcopilot.schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE teachcopilot.accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('teacher','student','admin')),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE teachcopilot.learners (
    learner_id TEXT PRIMARY KEY,
    pseudonym TEXT NOT NULL,
    grade SMALLINT NOT NULL CHECK (grade BETWEEN 1 AND 12),
    age_group TEXT,
    language TEXT NOT NULL DEFAULT 'russian',
    card JSONB NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE teachcopilot.account_learners (
    account_id UUID NOT NULL REFERENCES teachcopilot.accounts(id) ON DELETE CASCADE,
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL DEFAULT 'teacher',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, learner_id)
);
CREATE TABLE teachcopilot.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES teachcopilot.accounts(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON teachcopilot.sessions(expires_at);

CREATE TABLE teachcopilot.competencies (
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    mastery NUMERIC(5,4) CHECK (mastery BETWEEN 0 AND 1),
    independence NUMERIC(5,4) CHECK (independence BETWEEN 0 AND 1),
    status TEXT,
    last_assessed DATE,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(learner_id, topic_id)
);
CREATE TABLE teachcopilot.error_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    error_tag TEXT NOT NULL,
    subject TEXT,
    count INTEGER NOT NULL CHECK (count >= 0),
    last_seen DATE,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(learner_id, topic_id, error_tag)
);
CREATE TABLE teachcopilot.scaffolding (
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    strategies JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(learner_id, topic_id)
);
CREATE TABLE teachcopilot.learner_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    event_key TEXT,
    event_type TEXT NOT NULL,
    happened_at TIMESTAMPTZ,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(learner_id, event_key)
);
CREATE INDEX ON teachcopilot.learner_events(learner_id, happened_at DESC);
CREATE TABLE teachcopilot.points_ledger (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    entry_key TEXT NOT NULL,
    points INTEGER NOT NULL,
    reason TEXT,
    role TEXT,
    topic_id TEXT,
    happened_at DATE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(learner_id, entry_key)
);
CREATE TABLE teachcopilot.projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    project_key TEXT NOT NULL,
    title TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(learner_id, project_key)
);
CREATE TABLE teachcopilot.ai_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    observation_key TEXT NOT NULL,
    observed_at DATE,
    payload JSONB NOT NULL,
    UNIQUE(learner_id, observation_key)
);
CREATE TABLE teachcopilot.card_audit (
    id BIGSERIAL PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    actor_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot JSONB NOT NULL,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE teachcopilot.catalog_topics (
    topic_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    grade SMALLINT NOT NULL CHECK (grade BETWEEN 1 AND 12),
    title TEXT NOT NULL,
    parent_topic_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE teachcopilot.materials (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_file TEXT NOT NULL,
    topic_id TEXT,
    grade SMALLINT CHECK (grade BETWEEN 1 AND 12),
    subject_id TEXT,
    source_version TEXT NOT NULL DEFAULT '1',
    embedding vector(384),
    embedding_model TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON teachcopilot.materials(topic_id, grade);
CREATE INDEX ON teachcopilot.materials USING gin(metadata);
CREATE INDEX ON teachcopilot.materials USING ivfflat (embedding vector_cosine_ops) WITH (lists=50);
