ALTER TABLE teachcopilot.learning_sessions
    ADD COLUMN session_mode TEXT NOT NULL DEFAULT 'standard'
    CHECK (session_mode IN ('standard', 'autonomous_division'));

ALTER TABLE teachcopilot.learning_sessions
    ADD CONSTRAINT learning_sessions_autonomous_topic_check
    CHECK (session_mode = 'standard' OR topic_id = 'math.g4.numbers.division_by_1_2_digit');

CREATE INDEX learning_sessions_autonomous_lookup_idx
    ON teachcopilot.learning_sessions(learner_id, actor_id, topic_id, created_at DESC)
    WHERE session_mode = 'autonomous_division';
