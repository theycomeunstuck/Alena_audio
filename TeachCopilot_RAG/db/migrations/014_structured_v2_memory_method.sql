BEGIN;

ALTER TABLE teachcopilot.memory_jobs
    DROP CONSTRAINT IF EXISTS memory_jobs_method_check;

ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_method_check
    CHECK (method IN ('profile_only','tail','summary','structured','structured_v2'));

ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_structured_v2_job_kind_check
    CHECK (method <> 'structured_v2' OR kind = 'next_step');

COMMIT;
