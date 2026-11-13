"""An uncertain external write must not become trusted Mem0 evidence."""

from research_memory.personal_memory_lab.backends import Mem0Adapter, scope


def test_search_does_not_trust_lesson_with_unfinished_index():
    class Archive:
        def lessons(self, run_id, learner_id):
            assert (run_id, learner_id) == ("run", "child")
            return [{"lesson_id": "complete"}, {"lesson_id": "uncertain"}]

        def index_references(self, run_id, learner_id, backend):
            assert (run_id, learner_id, backend) == ("run", "child", "mem0")
            # A stale completion record must not authorize an absent source.
            return [{"lesson_id": "complete", "metadata": {}},
                    {"lesson_id": "not-archived", "metadata": {}}]

    class Memory:
        def search(self, question, *, filters, top_k, threshold):
            assert filters == {"user_id": scope("run", "child"), "lesson_id": {"in": ["complete"]}}
            return {"results": [
                {"memory": "complete fact", "metadata": {"lesson_id": "complete"}},
                {"memory": "uncertain fact", "metadata": {"lesson_id": "uncertain"}},
                {"memory": "unknown fact", "metadata": {"lesson_id": "not-archived"}},
            ]}

    hits = Mem0Adapter(Memory(), Archive()).search("run", "child", "question")
    assert [hit["lesson_ids"] for hit in hits] == [["complete"], [], []]


def test_search_without_completed_lessons_never_queries_untrusted_index():
    class Archive:
        def lessons(self, run_id, learner_id):
            return [{"lesson_id": "uncertain"}]

        def index_references(self, run_id, learner_id, backend):
            return []

    class Memory:
        def search(self, *args, **kwargs):
            raise AssertionError("No completed lesson should reach the index")

    assert Mem0Adapter(Memory(), Archive()).search("run", "child", "question") == []
