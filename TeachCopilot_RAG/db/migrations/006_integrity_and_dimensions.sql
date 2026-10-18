-- Version 1 used vector(384). The local runtime can select models with other
-- dimensions, so keep vectors untyped and record/validate each model dimension.
DROP INDEX IF EXISTS teachcopilot.materials_embedding_idx;
ALTER TABLE teachcopilot.materials ALTER COLUMN embedding TYPE vector USING embedding::vector;
ALTER TABLE teachcopilot.materials ADD COLUMN IF NOT EXISTS embedding_dimension SMALLINT CHECK (embedding_dimension BETWEEN 1 AND 2000);
UPDATE teachcopilot.materials SET embedding_dimension=vector_dims(embedding) WHERE embedding IS NOT NULL AND embedding_dimension IS NULL;
ALTER TABLE teachcopilot.catalog_topics ALTER COLUMN grade DROP NOT NULL;
CREATE TABLE IF NOT EXISTS teachcopilot.source_snapshots (
    id BIGSERIAL PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_key TEXT NOT NULL,
    source_version TEXT NOT NULL,
    source_file TEXT NOT NULL,
    payload JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(source_kind,source_key,source_version)
);
CREATE INDEX IF NOT EXISTS source_snapshots_key_idx ON teachcopilot.source_snapshots(source_kind,source_key);

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='catalog_topics_parent_fk' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.catalog_topics ADD CONSTRAINT catalog_topics_parent_fk
            FOREIGN KEY(parent_topic_id) REFERENCES teachcopilot.catalog_topics(topic_id)
            DEFERRABLE INITIALLY DEFERRED NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='competencies_topic_fk' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.competencies ADD CONSTRAINT competencies_topic_fk
            FOREIGN KEY(topic_id) REFERENCES teachcopilot.catalog_topics(topic_id)
            DEFERRABLE INITIALLY DEFERRED NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='errors_topic_fk' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.error_patterns ADD CONSTRAINT errors_topic_fk
            FOREIGN KEY(topic_id) REFERENCES teachcopilot.catalog_topics(topic_id)
            DEFERRABLE INITIALLY DEFERRED NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='scaffolding_topic_fk' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.scaffolding ADD CONSTRAINT scaffolding_topic_fk
            FOREIGN KEY(topic_id) REFERENCES teachcopilot.catalog_topics(topic_id)
            DEFERRABLE INITIALLY DEFERRED NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='materials_topic_fk' AND connamespace='teachcopilot'::regnamespace) THEN
        ALTER TABLE teachcopilot.materials ADD CONSTRAINT materials_topic_fk
            FOREIGN KEY(topic_id) REFERENCES teachcopilot.catalog_topics(topic_id)
            DEFERRABLE INITIALLY DEFERRED NOT VALID;
    END IF;
END $$;
ALTER TABLE teachcopilot.catalog_topics VALIDATE CONSTRAINT catalog_topics_parent_fk;
ALTER TABLE teachcopilot.competencies VALIDATE CONSTRAINT competencies_topic_fk;
ALTER TABLE teachcopilot.error_patterns VALIDATE CONSTRAINT errors_topic_fk;
ALTER TABLE teachcopilot.scaffolding VALIDATE CONSTRAINT scaffolding_topic_fk;
ALTER TABLE teachcopilot.materials VALIDATE CONSTRAINT materials_topic_fk;
