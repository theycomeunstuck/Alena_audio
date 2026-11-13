BEGIN;

ALTER TABLE teachcopilot.memory_jobs
    DROP CONSTRAINT IF EXISTS memory_jobs_method_check;

ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_method_check
    CHECK (method IN ('profile_only','tail','summary','structured','structured_v2','personal_graph','personal_memory'));

ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_personal_memory_job_kind_check
    CHECK (method <> 'personal_memory' OR (kind = 'next_step' AND provider = 'local'));

COMMIT;
