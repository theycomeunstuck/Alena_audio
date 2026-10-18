CREATE TABLE teachcopilot.embedding_models (
    model_name TEXT PRIMARY KEY,
    dimension SMALLINT NOT NULL CHECK (dimension BETWEEN 1 AND 2000),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
