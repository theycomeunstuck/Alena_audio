CREATE TABLE teachcopilot.save_operations (
    operation_id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE teachcopilot.card_proposals (
    proposal_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    actor_id TEXT NOT NULL,
    base_revision INTEGER NOT NULL CHECK (base_revision > 0),
    proposed_card JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','confirmed','rejected')),
    result_card JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    confirmed_at TIMESTAMPTZ
);
CREATE INDEX ON teachcopilot.card_proposals(learner_id,created_at DESC);
