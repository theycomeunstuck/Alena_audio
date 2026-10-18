ALTER TABLE teachcopilot.materials ADD COLUMN search_document tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('russian', content), 'A') ||
        setweight(to_tsvector('simple', content), 'B')
    ) STORED;
CREATE INDEX materials_search_document_idx ON teachcopilot.materials USING gin(search_document);
CREATE INDEX materials_active_model_idx ON teachcopilot.materials(embedding_model, embedding_dimension) WHERE active;
CREATE INDEX card_audit_learner_revision_idx ON teachcopilot.card_audit(learner_id, revision DESC);
