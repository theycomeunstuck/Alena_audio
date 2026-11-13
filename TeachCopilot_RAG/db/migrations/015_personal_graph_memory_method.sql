BEGIN;

ALTER TABLE teachcopilot.memory_jobs
    DROP CONSTRAINT IF EXISTS memory_jobs_method_check;

ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_method_check
    CHECK (method IN ('profile_only','tail','summary','structured','structured_v2','personal_graph'));

ALTER TABLE teachcopilot.memory_jobs
    ADD CONSTRAINT memory_jobs_personal_graph_job_kind_check
    CHECK (method <> 'personal_graph' OR (kind = 'next_step' AND provider = 'local'));

COMMIT;
