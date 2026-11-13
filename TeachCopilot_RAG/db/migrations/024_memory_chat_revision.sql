-- Track child chat changes separately so a reviewed draft can be confirmed
-- without treating teacher corrections or unexplained revisions as child chat.
-- Existing drafts have no captured baseline and remain conservatively stale.
ALTER TABLE teachcopilot.learners
    ADD COLUMN memory_chat_source_revision BIGINT NOT NULL DEFAULT 0
        CHECK (memory_chat_source_revision >= 0);

ALTER TABLE teachcopilot.memory_versions
    ADD COLUMN chat_source_revision BIGINT
        CHECK (chat_source_revision IS NULL OR chat_source_revision >= 0);
