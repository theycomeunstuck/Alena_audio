ALTER TABLE teachcopilot.learner_events
    ADD COLUMN IF NOT EXISTS source_key TEXT,
    ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS supersedes UUID,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
UPDATE teachcopilot.learner_events SET source_key=COALESCE(event_key,'legacy:'||id::text) WHERE source_key IS NULL;
ALTER TABLE teachcopilot.learner_events ALTER COLUMN source_key SET NOT NULL;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='learner_events_supersedes_fk' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.learner_events ADD CONSTRAINT learner_events_supersedes_fk
            FOREIGN KEY(supersedes) REFERENCES teachcopilot.learner_events(id) DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname='learner_events_learner_id_event_key_key' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.learner_events DROP CONSTRAINT learner_events_learner_id_event_key_key;
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS learner_events_active_source_idx
    ON teachcopilot.learner_events(learner_id,event_type,source_key) WHERE active;
CREATE UNIQUE INDEX IF NOT EXISTS learner_events_event_key_idx
    ON teachcopilot.learner_events(learner_id,event_key);

ALTER TABLE teachcopilot.points_ledger
    ADD COLUMN IF NOT EXISTS source_key TEXT,
    ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS supersedes UUID,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
UPDATE teachcopilot.points_ledger SET source_key=entry_key WHERE source_key IS NULL;
ALTER TABLE teachcopilot.points_ledger ALTER COLUMN source_key SET NOT NULL;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='points_ledger_supersedes_fk' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.points_ledger ADD CONSTRAINT points_ledger_supersedes_fk
            FOREIGN KEY(supersedes) REFERENCES teachcopilot.points_ledger(id) DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname='points_ledger_learner_id_entry_key_key' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.points_ledger DROP CONSTRAINT points_ledger_learner_id_entry_key_key;
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS points_ledger_active_source_idx
    ON teachcopilot.points_ledger(learner_id,source_key) WHERE active;
CREATE UNIQUE INDEX IF NOT EXISTS points_ledger_entry_key_idx
    ON teachcopilot.points_ledger(learner_id,entry_key);
