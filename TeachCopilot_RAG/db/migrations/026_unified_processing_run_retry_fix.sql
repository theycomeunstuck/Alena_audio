-- Integration correction for A-report.md finding F1 (client B, merged tree).
-- The UNIQUE(learner_id, session_id, source_fingerprint, prompt_recipe,
-- model_identity) tuple on unified_processing_runs makes the documented
-- failed-run retry (<request_id>:r2) crash with UniqueViolation, because the
-- identity stays identical while the request id changes. Request-id
-- uniqueness (UNIQUE(learner_id, request_id)) plus the pre-insert
-- find_unified_run check remain the replay guards; after this change a retry
-- inserts its own run row and the failed artifact stays retrievable, with
-- processing_result reporting the latest run.
-- Idempotent: the constraint is dropped only when it exists.

DO $$
DECLARE
    constraint_name text;
BEGIN
    SELECT conname INTO constraint_name
    FROM pg_constraint
    WHERE conrelid = 'teachcopilot.unified_processing_runs'::regclass
      AND contype = 'u'
      AND pg_get_constraintdef(oid) LIKE '%source_fingerprint%'
      AND pg_get_constraintdef(oid) LIKE '%prompt_recipe%'
      AND pg_get_constraintdef(oid) LIKE '%model_identity%'
      AND pg_get_constraintdef(oid) NOT LIKE '%request_id%';
    IF constraint_name IS NOT NULL THEN
        EXECUTE 'ALTER TABLE teachcopilot.unified_processing_runs DROP CONSTRAINT '
                || quote_ident(constraint_name);
    END IF;
END
$$;
