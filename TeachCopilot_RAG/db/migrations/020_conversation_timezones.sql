-- Display preferences and learning-day boundaries have separate meanings.
ALTER TABLE teachcopilot.accounts
    ADD COLUMN display_timezone TEXT NOT NULL DEFAULT 'Asia/Tomsk'
        CHECK (length(display_timezone) BETWEEN 1 AND 128);
ALTER TABLE teachcopilot.learners
    ADD COLUMN learning_timezone TEXT NOT NULL DEFAULT 'Asia/Tomsk'
        CHECK (length(learning_timezone) BETWEEN 1 AND 128);

-- Existing instants stay intact. Their original event zone was not recorded.
ALTER TABLE teachcopilot.learning_sessions
    ADD COLUMN event_timezone TEXT CHECK (length(event_timezone) BETWEEN 1 AND 128);
ALTER TABLE teachcopilot.memory_sources
    ADD COLUMN event_timezone TEXT CHECK (length(event_timezone) BETWEEN 1 AND 128),
    ADD COLUMN activity_timezone TEXT NOT NULL DEFAULT 'Asia/Tomsk'
        CHECK (length(activity_timezone) BETWEEN 1 AND 128),
    ADD COLUMN activity_day DATE GENERATED ALWAYS AS
        ((created_at AT TIME ZONE activity_timezone)::date) STORED;

-- Legacy activity days use the initial school's Tomsk calendar. This is a
-- grouping convention and does not assert the missing original event zone.

CREATE FUNCTION teachcopilot.keep_event_timezone() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.event_timezone IS DISTINCT FROM OLD.event_timezone THEN
        RAISE EXCEPTION 'event timezone is immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER memory_source_timezone_guard BEFORE UPDATE OF event_timezone
    ON teachcopilot.memory_sources FOR EACH ROW
    EXECUTE FUNCTION teachcopilot.keep_event_timezone();
CREATE TRIGGER learning_session_timezone_guard BEFORE UPDATE OF event_timezone
    ON teachcopilot.learning_sessions FOR EACH ROW
    EXECUTE FUNCTION teachcopilot.keep_event_timezone();

CREATE FUNCTION teachcopilot.keep_activity_timezone() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.activity_timezone IS DISTINCT FROM OLD.activity_timezone THEN
        RAISE EXCEPTION 'activity timezone is immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER memory_source_activity_timezone_guard BEFORE UPDATE OF activity_timezone
    ON teachcopilot.memory_sources FOR EACH ROW
    EXECUTE FUNCTION teachcopilot.keep_activity_timezone();

CREATE INDEX learning_sessions_actor_learner_idx
    ON teachcopilot.learning_sessions(learner_id, actor_id, id);
CREATE INDEX memory_sources_learner_created_idx
    ON teachcopilot.memory_sources(learner_id, created_at, source_order);
CREATE INDEX memory_sources_learner_activity_day_idx
    ON teachcopilot.memory_sources(learner_id, activity_day DESC, source_order DESC);
