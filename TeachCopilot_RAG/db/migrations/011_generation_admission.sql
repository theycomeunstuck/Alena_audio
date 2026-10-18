-- Actual model calls share admission across API and memory-worker processes.
-- Session advisory locks own active slots; leases clean up abandoned waiters.
CREATE TABLE teachcopilot.generation_admission (
    id UUID PRIMARY KEY,
    lane TEXT NOT NULL CHECK (lane IN ('local','external')),
    priority SMALLINT NOT NULL CHECK (priority IN (0,10)),
    status TEXT NOT NULL DEFAULT 'waiting' CHECK (status IN ('waiting','active','unknown_outcome')),
    slot SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    lease_until TIMESTAMPTZ NOT NULL,
    CHECK ((status='waiting' AND slot IS NULL) OR
           (status IN ('active','unknown_outcome') AND slot IS NOT NULL AND slot>=0 AND slot<4)),
    CHECK (lane<>'local' OR slot IS NULL OR slot=0)
);
CREATE INDEX generation_admission_wait_idx ON teachcopilot.generation_admission(lane,priority DESC,created_at,id) WHERE status='waiting';
CREATE UNIQUE INDEX generation_admission_active_slot_idx ON teachcopilot.generation_admission(lane,slot) WHERE status IN ('active','unknown_outcome');
