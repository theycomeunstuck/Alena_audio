-- One voluntary opening question per learner, account and local activity day.
CREATE TABLE teachcopilot.daily_checkins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id TEXT NOT NULL REFERENCES teachcopilot.learners(learner_id) ON DELETE CASCADE,
    actor_id TEXT NOT NULL,
    activity_date DATE NOT NULL,
    learning_timezone TEXT NOT NULL,
    origin_learning_session_id UUID,
    learning_session_id UUID,
    answer_learning_session_id UUID,
    question TEXT NOT NULL CHECK (length(question) BETWEEN 1 AND 1000),
    question_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'offered' CHECK (status IN ('offered','shown','skipped','answered')),
    offered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    shown_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    assistant_source_id TEXT,
    answered_source_id TEXT,
    UNIQUE(id,learner_id),
    UNIQUE(learner_id,actor_id,activity_date),
    FOREIGN KEY(origin_learning_session_id,learner_id)
        REFERENCES teachcopilot.learning_sessions(id,learner_id) ON DELETE CASCADE,
    FOREIGN KEY(learning_session_id,learner_id)
        REFERENCES teachcopilot.learning_sessions(id,learner_id) ON DELETE CASCADE,
    FOREIGN KEY(answer_learning_session_id,learner_id)
        REFERENCES teachcopilot.learning_sessions(id,learner_id) ON DELETE CASCADE,
    FOREIGN KEY(assistant_source_id,learner_id)
        REFERENCES teachcopilot.memory_sources(id,learner_id),
    FOREIGN KEY(answered_source_id,learner_id)
        REFERENCES teachcopilot.memory_sources(id,learner_id),
    CHECK ((shown_at IS NULL) = (assistant_source_id IS NULL)),
    CHECK (status <> 'offered' OR (shown_at IS NULL AND resolved_at IS NULL)),
    CHECK (status <> 'shown' OR (shown_at IS NOT NULL AND resolved_at IS NULL)),
    CHECK ((status IN ('skipped','answered')) = (resolved_at IS NOT NULL)),
    CHECK ((status = 'answered') = (answered_source_id IS NOT NULL)),
    CHECK ((status = 'answered') = (answer_learning_session_id IS NOT NULL)),
    CHECK (status <> 'answered' OR shown_at IS NOT NULL)
);

CREATE TABLE teachcopilot.daily_checkin_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkin_id UUID NOT NULL,
    learner_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('offered','shown','skipped','answered')),
    source_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(checkin_id,event_type),
    FOREIGN KEY(checkin_id,learner_id)
        REFERENCES teachcopilot.daily_checkins(id,learner_id) ON DELETE CASCADE,
    FOREIGN KEY(source_id,learner_id)
        REFERENCES teachcopilot.memory_sources(id,learner_id)
);

CREATE INDEX daily_checkins_actor_day_idx
    ON teachcopilot.daily_checkins(learner_id,actor_id,activity_date DESC);

CREATE FUNCTION teachcopilot.guard_daily_checkin_history() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.id IS DISTINCT FROM OLD.id
        OR NEW.learner_id IS DISTINCT FROM OLD.learner_id
        OR NEW.actor_id IS DISTINCT FROM OLD.actor_id
        OR NEW.activity_date IS DISTINCT FROM OLD.activity_date
        OR NEW.learning_timezone IS DISTINCT FROM OLD.learning_timezone
        OR NEW.origin_learning_session_id IS DISTINCT FROM OLD.origin_learning_session_id
        OR NEW.question IS DISTINCT FROM OLD.question
        OR NEW.question_version IS DISTINCT FROM OLD.question_version
        OR NEW.offered_at IS DISTINCT FROM OLD.offered_at
        OR (OLD.shown_at IS NOT NULL AND (
            NEW.shown_at IS DISTINCT FROM OLD.shown_at
            OR NEW.assistant_source_id IS DISTINCT FROM OLD.assistant_source_id
            OR NEW.learning_session_id IS DISTINCT FROM OLD.learning_session_id))
        OR (OLD.status IN ('answered','skipped') AND NEW IS DISTINCT FROM OLD)
        OR (OLD.status='shown' AND NEW.status='offered')
    THEN
        RAISE EXCEPTION 'daily check-in history is immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER daily_checkin_history_guard BEFORE UPDATE
    ON teachcopilot.daily_checkins FOR EACH ROW
    EXECUTE FUNCTION teachcopilot.guard_daily_checkin_history();

CREATE FUNCTION teachcopilot.guard_daily_checkin_event() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'daily check-in events are append-only' USING ERRCODE='23514';
END;
$$;

CREATE TRIGGER daily_checkin_event_guard BEFORE UPDATE
    ON teachcopilot.daily_checkin_events FOR EACH ROW
    EXECUTE FUNCTION teachcopilot.guard_daily_checkin_event();
