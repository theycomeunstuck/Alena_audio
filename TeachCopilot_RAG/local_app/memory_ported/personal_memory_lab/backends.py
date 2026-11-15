"""Optional Graphiti and Mem0 adapters, always scoped to a research run and child."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from .archive import canonical


EXTRACTION_INSTRUCTIONS = (
    "Сохраняй дату и условия: вопрос учителя вместе с ответом ребёнка, помощь, "
    "исправление и неопределённость. Разовая просьба не является постоянным "
    "предпочтением. Слова ассистента являются только контекстом и не доказывают "
    "действий ребёнка. Не придумывай причины, навыки или устойчивые интересы. "
    "Не создавай отдельные воспоминания для каждого правильного однотипного "
    "разминочного упражнения, если там нет ошибки, помощи или личного события."
)


def scope(run_id, learner_id):
    if not run_id or not learner_id:
        raise ValueError("Both run and learner are required")
    return "nirs_" + hashlib.sha256(canonical([run_id, learner_id]).encode()).hexdigest()[:40]


def lesson_text(lesson):
    roles = {"teacher": "Учитель", "student": "Ребёнок", "assistant": "Ассистент, только контекст"}
    lines = [f"Дата: {lesson['observed_on']}; занятие: {lesson['lesson_id']}"]
    for turn in lesson["turns"]:
        marker = ", начало реплики отсутствует" if turn.get("role") == "assistant" and turn.get("truncated") else ""
        lines.append(f"[{turn['turn_id']}] {roles[turn['role']]}{marker}: {turn['text']}")
    return "\n".join(lines)


def require_archived_lesson(archive, run_id, learner_id, lesson):
    archived = archive.lessons(run_id, learner_id, [lesson["lesson_id"]])
    if len(archived) != 1 or canonical(archived[0]) != canonical(lesson):
        raise ValueError("The indexed lesson must exactly match the canonical archive")


def salient_fragment(runtime, part):
    """Select a few source-linked candidates; the archived turns remain authoritative."""
    turns = {turn["turn_id"]: turn for turn in part["turns"]}
    schema = {"type": "object", "properties": {"facts": {"type": "array", "maxItems": 4,
        "items": {"type": "object", "properties": {"summary": {"type": "string"},
            "source_ids": {"type": "array", "minItems": 1, "maxItems": 8,
                           "items": {"type": "string", "enum": list(turns)}}},
            "required": ["summary", "source_ids"]}}}, "required": ["facts"]}
    system = ("Отбери не более четырёх важных событий персональной памяти из исходного фрагмента. "
        "Сохраняй ошибки и исправления, вопросы и помощь, просьбы, случайные факты, "
        "интересы, неопределённость и незавершённые дела. Каждое событие должно ссылаться "
        "на точные ID исходных реплик ребёнка или учителя. Реплики ассистента только контекст. "
        "Изменившийся ответ после вопроса преподавателя сохраняй как возможное исправление с помощью, "
        "даже если никто не произнёс слово «ошибка». "
        "Однократную просьбу не превращай в устойчивое предпочтение. Обычные правильные "
        "разминочные вычисления не сохраняй. Если важных событий нет, верни пустой facts. "
        "Не добавляй фактов вне текста. Верни JSON.")
    payload = {"model": runtime.model, "stream": False, "think": False, "format": schema,
        "options": {"num_ctx": runtime.context, "num_predict": 1024, "temperature": 0, "seed": 42},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": lesson_text(part)}]}
    response = runtime._post("/api/chat", payload, "personal_filter")
    if response.get("done") is not True or response.get("done_reason") == "length":
        raise ValueError("Personal memory prefilter did not finish")
    parsed = json.loads(response.get("message", {}).get("content", ""))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list) or len(parsed["facts"]) > 4:
        raise ValueError("Invalid personal memory prefilter response")
    result = []
    for fact in parsed["facts"]:
        if (not isinstance(fact, dict) or not isinstance(fact.get("summary"), str)
                or not fact["summary"].strip() or len(fact["summary"]) > 600
                or not isinstance(fact.get("source_ids"), list) or not 1 <= len(fact["source_ids"]) <= 8
                or not all(isinstance(i, str) and i in turns for i in fact["source_ids"])
                or not any(turns[i]["role"] in {"teacher", "student"} for i in fact["source_ids"])):
            raise ValueError("Personal memory prefilter cited an unavailable source")
        result.append({"candidate": fact["summary"], "source_turns": [turns[i] for i in fact["source_ids"]]})
    return result


def make_graphiti(runtime, password):
    # Environment variables must be set before the libraries are imported.
    os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    class Embeddings(EmbedderClient):
        async def create(self, input_data):
            if not isinstance(input_data, str):
                if not isinstance(input_data, list) or not all(isinstance(x, str) for x in input_data):
                    raise ValueError("Expected text for BGE-M3")
                input_data = "\n".join(input_data)
            return await asyncio.to_thread(runtime.embed, input_data)

        async def create_batch(self, input_data_list):
            return [await self.create(text) for text in input_data_list]

    class LocalLLM(OpenAIGenericClient):
        async def _generate_response(self, messages, response_model=None, max_tokens=2048,
                                     model_size=None, *, model=None):
            if model and model != runtime.model:
                raise ValueError("Graphiti requested a different model")
            schema = response_model.model_json_schema() if response_model else "json"
            payload = {"model": runtime.model, "stream": False, "think": False, "format": schema,
                "options": {"num_ctx": runtime.context, "num_predict": min(max_tokens, 2048),
                            "temperature": 0, "seed": 42},
                "messages": [{"role": item.role, "content": item.content} for item in messages]}
            result = await asyncio.to_thread(runtime._post, "/api/chat", payload, "graphiti_llm")
            if result.get("done") is not True or result.get("done_reason") == "length":
                raise ValueError("Graphiti extraction did not finish")
            obj = json.loads(result["message"]["content"])
            if response_model:
                response_model.model_validate(obj)
            return obj

    class BGEReranker(CrossEncoderClient):
        async def rank(self, query, passages):
            q = await embeddings.create(query)
            ranked = []
            for passage in passages:
                v = await embeddings.create(passage)
                norm = math.sqrt(sum(x * x for x in q) * sum(x * x for x in v))
                ranked.append((passage, sum(a * b for a, b in zip(q, v)) / norm if norm else 0.0))
            return sorted(ranked, key=lambda item: item[1], reverse=True)

    embeddings = Embeddings()
    llm = LocalLLM(config=LLMConfig(api_key="unused", model=runtime.model,
        small_model=runtime.model, base_url=runtime.base_url + "/v1", temperature=0), max_tokens=2048)
    return Graphiti("bolt://127.0.0.1:17687", "neo4j", password, llm_client=llm,
                    embedder=embeddings, cross_encoder=BGEReranker(), max_coroutines=1)


class GraphitiAdapter:
    name = "graphiti"

    def __init__(self, graph, archive, runtime=None):
        self.graph, self.archive, self.runtime = graph, archive, runtime

    async def initialize(self):
        await self.graph.build_indices_and_constraints()

    async def ingest(self, run_id, learner_id, lesson):
        require_archived_lesson(self.archive, run_id, learner_id, lesson)
        if not self.archive.claim(run_id, learner_id, lesson["lesson_id"], self.name):
            return {"status": "already_indexed"}
        try:
            from .chunks import lesson_chunks
            results = []
            skipped_fragments = 0
            raw_anchor_used = False
            for position, part in enumerate(lesson_chunks(lesson, max_bytes=1800, hard_max_bytes=6000), 1):
                if self.runtime is None:
                    episode_body = lesson_text(part)
                else:
                    candidates = await asyncio.to_thread(salient_fragment, self.runtime, part)
                    if not candidates:
                        if position == 1:
                            # Do not make a model's empty prefilter the sole
                            # representation of the beginning of a lesson.
                            episode_body = lesson_text(part)
                            raw_anchor_used = True
                        else:
                            skipped_fragments += 1
                            continue
                    else:
                        episode_body = json.dumps({"observed_on": lesson["observed_on"],
                            "lesson_id": lesson["lesson_id"], "candidates": candidates}, ensure_ascii=False)
                results.append(await self.graph.add_episode(name=f"{lesson['lesson_id']}:part:{position}",
                    episode_body=episode_body, source_description="Кандидаты из фрагмента исходного занятия в PostgreSQL",
                    reference_time=datetime.fromisoformat(lesson["observed_on"]).replace(tzinfo=timezone.utc),
                    group_id=scope(run_id, learner_id),
                    previous_episode_uuids=[],
                    custom_extraction_instructions=EXTRACTION_INSTRUCTIONS))
        except Exception as exc:
            self.archive.finish(run_id, learner_id, lesson["lesson_id"], self.name, type(exc).__name__)
            raise
        self.archive.finish(run_id, learner_id, lesson["lesson_id"], self.name,
                            metadata={"episode_uuids": [result.episode.uuid for result in results]})
        nodes = [node for result in results for node in result.nodes]
        edges = [edge for result in results for edge in result.edges]
        return {"status": "indexed", "nodes": len(nodes), "edges": len(edges),
                "skipped_non_salient_fragments": skipped_fragments,
                "raw_anchor_used": raw_anchor_used,
                "episode_uuids": [result.episode.uuid for result in results],
                "graph_nodes": [{"id": node.uuid, "name": node.name, "summary": node.summary}
                                for node in nodes],
                "graph_edges": [{"id": edge.uuid, "source": edge.source_node_uuid,
                                 "target": edge.target_node_uuid, "fact": edge.fact,
                                 "episodes": edge.episodes or [],
                                 "valid_at": str(edge.valid_at) if edge.valid_at else None,
                                 "invalid_at": str(edge.invalid_at) if edge.invalid_at else None}
                                for edge in edges]}

    async def search(self, run_id, learner_id, question, limit=8):
        edges = await self.graph.search(question, group_ids=[scope(run_id, learner_id)], num_results=limit)
        references = self.archive.index_references(run_id, learner_id, self.name)
        by_episode = {}
        for row in references:
            episodes = row["metadata"].get("episode_uuids") or [row["metadata"].get("episode_uuid")]
            by_episode.update({episode: row["lesson_id"] for episode in episodes if episode})
        hits = []
        for edge in edges:
            episodes = edge.episodes or []
            complete = bool(episodes) and all(item in by_episode for item in episodes)
            hits.append({"kind": "graphiti", "text": edge.fact,
                         "lesson_ids": list(dict.fromkeys(by_episode[x] for x in episodes)) if complete else [],
                         "unresolved_episode_ids": [] if complete else episodes,
                         "valid_at": str(edge.valid_at) if edge.valid_at else None,
                         "invalid_at": str(edge.invalid_at) if edge.invalid_at else None,
                         "edge_id": edge.uuid})
        return hits


def make_mem0(runtime, database_url, work_dir, extraction="legacy"):
    if extraction not in {"legacy", "strict_v1"}:
        raise ValueError("Unknown Mem0 extraction variant")
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["MEM0_DIR"] = str(Path(work_dir).resolve())
    from mem0 import Memory
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config = {"vector_store": {"provider": "pgvector", "config": {
        "connection_string": database_url, "collection_name": "nirs_personal_mem0",
        "embedding_model_dims": 1024}},
        "llm": {"provider": "ollama", "config": {"model": runtime.model,
            "ollama_base_url": runtime.base_url, "temperature": 0, "max_tokens": 2048}},
        "embedder": {"provider": "ollama", "config": {"model": "bge-m3",
            "ollama_base_url": runtime.base_url, "embedding_dims": 1024}},
        "history_db_path": str((work_dir / "mem0-history.sqlite3").resolve()),
        "custom_instructions": EXTRACTION_INSTRUCTIONS}
    memory = Memory.from_config(config)

    class LLM:
        def generate_response(self, messages, response_format=None, tools=None, **kwargs):
            if tools:
                raise ValueError("Mem0 tools are not enabled in this experiment")
            payload = {"model": runtime.model, "stream": False, "think": False,
                "options": {"num_ctx": runtime.context, "num_predict": 2048, "temperature": 0, "seed": 42},
                "messages": messages}
            if response_format:
                payload["format"] = "json"
            result = runtime._post("/api/chat", payload, "mem0_llm")
            if result.get("done") is not True or result.get("done_reason") == "length":
                raise ValueError("Mem0 extraction did not finish")
            content = result["message"]["content"]
            if response_format:
                parsed = json.loads(content)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("memory"), list):
                    raise ValueError("Mem0 extraction did not return its expected memory list")
            return content

    class Embedder:
        def embed(self, text, memory_action=None):
            return runtime.embed(text)

        def embed_batch(self, texts, memory_action=None):
            return [runtime.embed(text) for text in texts]

    if extraction == "strict_v1":
        from .mem0_strict_v1 import StrictMem0LLM, guard_mem0_writes
        memory.llm = StrictMem0LLM(runtime)
        guard_mem0_writes(memory)
    else:
        memory.llm = LLM()
    memory.embedding_model = Embedder()
    return memory


class Mem0Adapter:
    name = "mem0"

    def __init__(self, memory, archive, extraction="legacy"):
        if extraction not in {"legacy", "strict_v1"}:
            raise ValueError("Unknown Mem0 extraction variant")
        if extraction == "strict_v1" and not hasattr(memory, "_strict_write_latch"):
            raise ValueError("strict_v1 requires a guarded Mem0 instance")
        self.memory, self.archive, self.extraction = memory, archive, extraction

    def ingest(self, run_id, learner_id, lesson):
        require_archived_lesson(self.archive, run_id, learner_id, lesson)
        if not self.archive.claim(run_id, learner_id, lesson["lesson_id"], self.name):
            return {"status": "already_indexed"}
        attempted, completed = [], []
        try:
            # Mem0's extraction includes prior messages and existing memories in
            # its session scope. Give each lesson a separate run_id so a fact is
            # never attributed only to this lesson after seeing earlier lessons.
            from .chunks import lesson_chunks
            results = []
            for position, part in enumerate(lesson_chunks(lesson, max_bytes=1800, hard_max_bytes=6000), 1):
                part_scope = scope(scope(run_id, learner_id), lesson["lesson_id"] + ":" + str(position))
                boundary = {"part": position, "first_turn_id": part["turns"][0]["turn_id"],
                            "last_turn_id": part["turns"][-1]["turn_id"]}
                attempted.append(boundary)
                latch = getattr(self.memory, "_strict_write_latch", None)
                if self.extraction == "strict_v1" and latch is not None:
                    latch.check()
                llm = getattr(self.memory, "llm", None)
                if hasattr(llm, "last_diagnostic"):
                    llm.last_diagnostic = None
                part_metadata = {"lesson_id": lesson["lesson_id"], "part": position}
                if self.extraction == "strict_v1":
                    part_metadata.update(first_turn_id=boundary["first_turn_id"],
                                         last_turn_id=boundary["last_turn_id"])
                result = self.memory.add(lesson_text(part), user_id=scope(run_id, learner_id), run_id=part_scope,
                                         metadata=part_metadata, infer=True)
                if self.extraction == "strict_v1" and latch is not None:
                    latch.check()
                results.extend(result.get("results", []))
                completed.append(boundary)
        except Exception as exc:
            latch = getattr(self.memory, "_strict_write_latch", None)
            diagnostic = (latch.diagnostic() if latch is not None and latch.uncertain else
                          getattr(getattr(self.memory, "llm", None), "last_diagnostic", None))
            metadata = {"extraction": self.extraction, "attempted_parts": attempted,
                        "completed_parts": completed, "diagnostic": diagnostic}
            self.archive.finish(run_id, learner_id, lesson["lesson_id"], self.name, type(exc).__name__,
                                metadata=metadata)
            raise
        self.archive.finish(run_id, learner_id, lesson["lesson_id"], self.name,
                            metadata={"extraction": self.extraction, "attempted_parts": attempted,
                                      "completed_parts": completed})
        return {"status": "indexed", "results": results}

    def search(self, run_id, learner_id, question, limit=8):
        archived_ids = {lesson["lesson_id"] for lesson in self.archive.lessons(run_id, learner_id)}
        completed_ids = {row["lesson_id"] for row in self.archive.index_references(run_id, learner_id, self.name)}
        valid_ids = archived_ids & completed_ids
        if not valid_ids:
            return []
        # Apply the completed-lesson filter before ranking so uncertain writes
        # cannot crowd out trusted memories. Recheck results defensively below.
        found = self.memory.search(question, filters={"user_id": scope(run_id, learner_id),
                                   "lesson_id": {"in": sorted(valid_ids)}}, top_k=limit, threshold=0)
        hits = []
        for row in found.get("results", []):
            metadata = row.get("metadata") or {}
            lesson_id = metadata.get("lesson_id") or row.get("lesson_id")
            hits.append({"kind": "mem0", "text": row.get("memory", ""), "memory_id": row.get("id"),
                         "score": row.get("score"),
                         "lesson_ids": [lesson_id] if lesson_id in valid_ids else []})
        return hits
