ALTER TABLE teachcopilot.materials
    ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;
CREATE INDEX IF NOT EXISTS materials_active_subject_idx
    ON teachcopilot.materials(active,subject_id,grade);
