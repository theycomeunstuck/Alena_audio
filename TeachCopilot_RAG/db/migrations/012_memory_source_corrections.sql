-- Correct an observation by adding a linked teacher source; preserve both texts.
ALTER TABLE teachcopilot.memory_sources ADD COLUMN corrects_source_id TEXT;
ALTER TABLE teachcopilot.memory_sources ADD CONSTRAINT memory_sources_id_learner_topic_key
    UNIQUE(id,learner_id,topic_id);
ALTER TABLE teachcopilot.memory_sources ADD CONSTRAINT memory_sources_correction_target_fk
    FOREIGN KEY(corrects_source_id,learner_id,topic_id)
    REFERENCES teachcopilot.memory_sources(id,learner_id,topic_id);
ALTER TABLE teachcopilot.memory_sources ADD CONSTRAINT memory_sources_correction_author_check
    CHECK (corrects_source_id IS NULL OR (author_role='teacher' AND approved AND corrects_source_id<>id));
CREATE INDEX memory_sources_correction_target_idx
    ON teachcopilot.memory_sources(learner_id,topic_id,corrects_source_id)
    WHERE corrects_source_id IS NOT NULL;

CREATE FUNCTION teachcopilot.check_memory_source_correction() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE target_order BIGINT;
BEGIN
    IF TG_OP='UPDATE' AND (
        NEW.id IS DISTINCT FROM OLD.id OR NEW.source_order IS DISTINCT FROM OLD.source_order
        OR NEW.learner_id IS DISTINCT FROM OLD.learner_id OR NEW.topic_id IS DISTINCT FROM OLD.topic_id
        OR NEW.learning_session_id IS DISTINCT FROM OLD.learning_session_id
        OR NEW.actor_id IS DISTINCT FROM OLD.actor_id OR NEW.author_role IS DISTINCT FROM OLD.author_role
        OR NEW.content IS DISTINCT FROM OLD.content OR NEW.observed_on IS DISTINCT FROM OLD.observed_on
        OR NEW.source_key IS DISTINCT FROM OLD.source_key
        OR NEW.approved IS DISTINCT FROM OLD.approved OR NEW.created_at IS DISTINCT FROM OLD.created_at
        OR NEW.corrects_source_id IS DISTINCT FROM OLD.corrects_source_id
    ) THEN
        RAISE EXCEPTION 'memory source text and provenance are immutable' USING ERRCODE='23514';
    END IF;
    IF NEW.corrects_source_id IS NOT NULL THEN
        SELECT s.source_order INTO target_order
        FROM teachcopilot.memory_sources s
        LEFT JOIN teachcopilot.learning_sessions ls
            ON ls.id=s.learning_session_id AND ls.learner_id=s.learner_id
        WHERE s.id=NEW.corrects_source_id AND s.learner_id=NEW.learner_id AND s.topic_id=NEW.topic_id
            AND ((s.author_role='teacher' AND s.approved
                  AND (s.learning_session_id IS NULL OR ls.status='completed'))
                 OR (s.author_role='student' AND ls.status='completed'));
        IF target_order IS NULL OR target_order>=NEW.source_order THEN
            RAISE EXCEPTION 'correction target must be an earlier eligible source of the same learner and topic'
                USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER memory_source_correction_guard BEFORE INSERT OR UPDATE
    ON teachcopilot.memory_sources FOR EACH ROW
    EXECUTE FUNCTION teachcopilot.check_memory_source_correction();
