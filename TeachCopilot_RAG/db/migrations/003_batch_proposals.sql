CREATE TABLE teachcopilot.proposal_batches (
    proposal_id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','confirmed','rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    confirmed_at TIMESTAMPTZ
);
CREATE TABLE teachcopilot.proposal_batch_cards (
    proposal_id TEXT NOT NULL REFERENCES teachcopilot.proposal_batches(proposal_id) ON DELETE CASCADE,
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    base_revision INTEGER NOT NULL CHECK (base_revision > 0),
    proposed_card JSONB NOT NULL,
    result_card JSONB,
    PRIMARY KEY(proposal_id,learner_id)
);
