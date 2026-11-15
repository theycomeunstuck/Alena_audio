"""Local PostgreSQL store isolated from the legacy public schema."""
from __future__ import annotations
import hashlib, json, re, os, sys, time, uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import psycopg2
from psycopg2.extensions import parse_dsn
from psycopg2.extras import Json, RealDictCursor
from local_app.memory_store import MemoryStoreMixin

MIGRATIONS = Path(__file__).resolve().parent.parent / "db" / "migrations"
LEARNER_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

class RevisionConflict(RuntimeError):
    """Card revision did not match the caller's expected revision."""

class IndexNotReadyError(RuntimeError):
    """The selected embedding model has not indexed the full filtered corpus."""

class StaleMaterialSnapshotError(RuntimeError):
    """Materials changed after embeddings were calculated for their snapshot."""

@contextmanager
def connect(dsn: str):
    options = parse_dsn(dsn).get("options", "") + " -c timezone=UTC"
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor, options=options.strip())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _json(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value

def _day(value):
    if not value: return None
    if isinstance(value, datetime): return value.date()
    if isinstance(value, date): return value
    try: return date.fromisoformat(str(value)[:10])
    except ValueError: return None

class _ReuseConnectionContext:
    """Yields one already-open connection and commits per use without closing it."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        return False

class _SingleConnectionClaims(MemoryStoreMixin):
    """Runs the durable chat claim against one open connection (F-035).

    Every attempt commits (or rolls back) on the shared connection, which
    releases the queue advisory lock between attempts exactly like a fresh
    connection per attempt, minus the per-iteration connect cost.
    """

    def __init__(self, conn):
        self._conn = conn

    def _connect(self):
        return _ReuseConnectionContext(self._conn)

class Store(MemoryStoreMixin):
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        return connect(self.dsn)

    def claim_interactive_chat_job_polling(self, job_id, worker_id, lease_seconds=600,
                                           poll_interval=0.05, timeout=180.0):
        """Poll for an interactive chat slot reusing one connection per waiter (F-035).

        Returns the claimed job row, or None once the timeout elapses without a
        claim; the caller remains responsible for cancelling the job on None.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._connect() as conn:
            waiting = _SingleConnectionClaims(conn)
            while True:
                claimed = waiting.claim_interactive_chat_job(job_id, worker_id, lease_seconds=lease_seconds)
                if claimed:
                    return claimed
                if time.monotonic() >= deadline:
                    return None
                time.sleep(max(0.0, float(poll_interval)))

    def ping(self) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            return cur.fetchone()["ok"] == 1

    def migrate(self) -> int:
        files = sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS teachcopilot")
            cur.execute("CREATE TABLE IF NOT EXISTS teachcopilot.schema_version(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_migrations',0))")
            cur.execute("SELECT COALESCE(max(version),0) AS v FROM teachcopilot.schema_version")
            version, applied = cur.fetchone()["v"], 0
            for path in files:
                next_version = int(path.name.split("_",1)[0])
                if next_version <= version: continue
                cur.execute(path.read_text(encoding="utf-8"))
                cur.execute("INSERT INTO teachcopilot.schema_version(version) VALUES (%s)", (next_version,))
                version, applied = next_version, applied + 1
            return applied

    def schema_version(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COALESCE(max(version),0) AS version FROM teachcopilot.schema_version")
            return cur.fetchone()["version"]

    def import_repository(self, repo_root: Path) -> dict[str,int]:
        root = Path(repo_root).resolve()
        data_root = root / "learner-data"
        rag_root = root
        if not (data_root / "learners").is_dir() and (root / "TeachCopilot_RAG").is_dir():
            data_root, rag_root = root / "learner-data", root / "TeachCopilot_RAG"
        elif (root / "learner-data").is_dir():
            data_root = root / "learner-data"
            if not (root / "knowledge-data").is_dir() and (root / "TeachCopilot_RAG" / "knowledge-data").is_dir():
                rag_root = root / "TeachCopilot_RAG"
        cards = []
        learners_dir = data_root / "learners"
        if learners_dir.is_dir():
            cards = [json.loads(p.read_text(encoding="utf-8-sig")) for p in sorted(learners_dir.glob("*.json")) if not p.name.startswith("_")]
        catalog_files = sorted((data_root / "catalog").glob("*.json")) if (data_root / "catalog").is_dir() else []
        catalogs = [json.loads(p.read_text(encoding="utf-8-sig")) for p in catalog_files]
        task_files = sorted((rag_root / "knowledge-data").glob("*.json"))
        tasks = []
        for file in task_files:
            file_text = file.read_text(encoding="utf-8-sig")
            source_version = hashlib.sha256(file_text.encode("utf-8")).hexdigest()
            for task in json.loads(file_text).get("tasks", []):
                tasks.append((task, file.relative_to(root).as_posix() if file.is_relative_to(root) else file.name, source_version))
        with self._connect() as conn, conn.cursor() as cur:
            for catalog_path,catalog in zip(catalog_files,catalogs):
                for topic in catalog.get("topics", []):
                    cur.execute("""INSERT INTO teachcopilot.catalog_topics(topic_id,subject_id,grade,title,parent_topic_id,payload)
                    VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(topic_id) DO UPDATE SET subject_id=EXCLUDED.subject_id,
                    grade=EXCLUDED.grade,title=EXCLUDED.title,parent_topic_id=EXCLUDED.parent_topic_id,payload=EXCLUDED.payload""",
                    (topic["id"],catalog.get("subject","math"),int(topic["grade"]) if str(topic.get("grade","")).isdigit() else None,topic.get("title_ru",topic.get("title",topic["id"])),topic.get("parent"),Json(topic)))
                raw_catalog = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
                self._save_snapshot(cur,"catalog",str(catalog.get("subject","catalog")),hashlib.sha256(raw_catalog.encode()).hexdigest(),catalog_path.relative_to(root).as_posix(),catalog)
            imported, skipped, conflicts = 0, 0, []
            for card in cards:
                lid=card["learner_id"]
                card_text=json.dumps(card,ensure_ascii=False,sort_keys=True)
                card_version=hashlib.sha256(card_text.encode()).hexdigest()
                source_path=learners_dir / f"{lid}.json"
                self._save_snapshot(cur,"learner_card",lid,card_version,source_path.relative_to(root).as_posix(),card)
                cur.execute("SELECT card FROM teachcopilot.learners WHERE learner_id=%s",(lid,))
                existing=cur.fetchone()
                if existing:
                    skipped += 1
                    if _json(existing["card"]) != card: conflicts.append(lid)
                else:
                    self._save_card(cur,card,"migration",None,audit=False)
                    imported += 1
            for task, source, source_version in tasks:
                tid = str(task.get("task_id") or hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest())
                # Keep the answer only in the retained private raw snapshot.
                content = str(task.get("content", ""))
                meta = {k:v for k,v in task.items() if k not in ("content","answer")}
                grade = int(task["grade"]) if str(task.get("grade","")).isdigit() else None
                subject = task.get("subject_id") or task.get("subject") or ("programming" if source.lower().find("programming") >= 0 else "math")
                cur.execute("""INSERT INTO teachcopilot.materials(id,content,source_file,topic_id,grade,subject_id,source_version,metadata,raw_snapshot,active)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE) ON CONFLICT(id) DO UPDATE SET content=EXCLUDED.content,
                    source_file=EXCLUDED.source_file,topic_id=EXCLUDED.topic_id,grade=EXCLUDED.grade,subject_id=EXCLUDED.subject_id,
                    embedding=CASE WHEN teachcopilot.materials.source_version<>EXCLUDED.source_version THEN NULL ELSE teachcopilot.materials.embedding END,
                    embedding_dimension=CASE WHEN teachcopilot.materials.source_version<>EXCLUDED.source_version THEN NULL ELSE teachcopilot.materials.embedding_dimension END,
                    embedding_model=CASE WHEN teachcopilot.materials.source_version<>EXCLUDED.source_version THEN NULL ELSE teachcopilot.materials.embedding_model END,
                    source_version=EXCLUDED.source_version,metadata=EXCLUDED.metadata,raw_snapshot=EXCLUDED.raw_snapshot,active=TRUE,updated_at=now()""",
                    (tid,content,source,task.get("topic_id"),grade,subject,source_version,Json(meta),Json(task)))
                self._save_snapshot(cur,"material",tid,source_version,source,task)
            for file in task_files:
                source_text=file.read_text(encoding="utf-8-sig")
                source_payload=json.loads(source_text)
                source_version=hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                source_name=file.relative_to(root).as_posix() if file.is_relative_to(root) else file.name
                current_ids=[str(task.get("task_id") or hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()) for task in source_payload.get("tasks",[])]
                cur.execute("UPDATE teachcopilot.materials SET active=FALSE,updated_at=now() WHERE source_file=%s AND NOT (id=ANY(%s))",(source_name,current_ids))
                self._save_snapshot(cur,"source_file",file.name,source_version,source_name,source_payload)
        return {"cards":len(cards),"cards_imported":imported,"cards_skipped":skipped,"card_conflicts":conflicts,"tasks":len(tasks),"topics":sum(len(c.get("topics",[])) for c in catalogs)}

    @staticmethod
    def _save_snapshot(cur,kind,key,version,source_file,payload):
        cur.execute("""INSERT INTO teachcopilot.source_snapshots(source_kind,source_key,source_version,source_file,payload)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT(source_kind,source_key,source_version) DO NOTHING""",
            (kind,key,version,source_file,Json(payload)))

    def import_materials(self,repo_root: Path) -> dict[str,int]:
        """Import curriculum materials only; learner state is never rewritten."""
        root=Path(repo_root).resolve()
        rag_root=root if (root/"knowledge-data").is_dir() else root/"TeachCopilot_RAG"
        files=sorted((rag_root/"knowledge-data").glob("*.json"))
        count=0
        with self._connect() as conn, conn.cursor() as cur:
            for file in files:
                raw=file.read_text(encoding="utf-8-sig")
                version=hashlib.sha256(raw.encode("utf-8")).hexdigest()
                for task in json.loads(raw).get("tasks",[]):
                    tid=str(task.get("task_id") or hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest())
                    content=str(task.get("content", ""))
                    grade=int(task["grade"]) if str(task.get("grade","")).isdigit() else None
                    subject=task.get("subject_id") or task.get("subject") or ("programming" if "programming" in file.name.lower() else "math")
                    meta={k:v for k,v in task.items() if k not in ("content","answer")}
                    source_name=file.relative_to(root).as_posix() if file.is_relative_to(root) else file.name
                    cur.execute("""INSERT INTO teachcopilot.materials(id,content,source_file,topic_id,grade,subject_id,source_version,metadata,raw_snapshot,active)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE) ON CONFLICT(id) DO UPDATE SET content=EXCLUDED.content,
                        source_file=EXCLUDED.source_file,topic_id=EXCLUDED.topic_id,grade=EXCLUDED.grade,subject_id=EXCLUDED.subject_id,
                        embedding=CASE WHEN teachcopilot.materials.source_version<>EXCLUDED.source_version THEN NULL ELSE teachcopilot.materials.embedding END,
                        embedding_dimension=CASE WHEN teachcopilot.materials.source_version<>EXCLUDED.source_version THEN NULL ELSE teachcopilot.materials.embedding_dimension END,
                        embedding_model=CASE WHEN teachcopilot.materials.source_version<>EXCLUDED.source_version THEN NULL ELSE teachcopilot.materials.embedding_model END,
                        source_version=EXCLUDED.source_version,metadata=EXCLUDED.metadata,raw_snapshot=EXCLUDED.raw_snapshot,active=TRUE,updated_at=now()""",
                        (tid,content,source_name,task.get("topic_id"),grade,subject,version,Json(meta),Json(task)))
                    self._save_snapshot(cur,"material",tid,version,file.name,task)
                    count+=1
                current_ids=[str(task.get("task_id") or hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()) for task in json.loads(raw).get("tasks",[])]
                cur.execute("UPDATE teachcopilot.materials SET active=FALSE,updated_at=now() WHERE source_file=%s AND NOT (id=ANY(%s))",(source_name,current_ids))
                self._save_snapshot(cur,"source_file",file.name,version,str(file),json.loads(raw))
        return {"materials":count}

    def _save_card(self, cur, card, actor_id, expected_revision, audit=True):
        lid = card.get("learner_id")
        if not isinstance(lid,str) or not LEARNER_ID.fullmatch(lid): raise ValueError("invalid learner_id")
        grade = int(card.get("grade",3))
        if not 1 <= grade <= 12: raise ValueError("grade must be 1..12")
        self._validate_card(cur,card)
        cur.execute("SELECT revision FROM teachcopilot.learners WHERE learner_id=%s FOR UPDATE",(lid,))
        old = cur.fetchone()
        if old and expected_revision is not None and old["revision"] != expected_revision:
            raise RevisionConflict(f"revision is {old['revision']}, expected {expected_revision}")
        rev = old["revision"]+1 if old else 1
        cur.execute("""INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,age_group,language,card,revision)
            VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(learner_id) DO UPDATE SET pseudonym=EXCLUDED.pseudonym,
            grade=EXCLUDED.grade,age_group=EXCLUDED.age_group,language=EXCLUDED.language,card=EXCLUDED.card,
            revision=EXCLUDED.revision,updated_at=now()""",
            (lid,card.get("pseudonym") or lid,grade,card.get("age_group"),card.get("language","russian"),Json(card),rev))
        if audit:
            cur.execute("INSERT INTO teachcopilot.card_audit(learner_id,actor_id,revision,snapshot) VALUES (%s,%s,%s,%s)",(lid,actor_id,rev,Json(card)))
        self._sync_card_relations(cur,lid,card)
        return dict(card,revision=rev)

    @staticmethod
    def _validate_card(cur,card):
        tools_dir=Path(os.getenv("TEACHCOPILOT_LEARNER_DATA_DIR",Path(__file__).resolve().parents[2]/"learner-data")) / "tools"
        if not tools_dir.is_dir(): raise ValueError(f"learner-data validation tools are unavailable: {tools_dir}")
        if str(tools_dir) not in sys.path: sys.path.insert(0,str(tools_dir))
        try:
            from validate import validate_card
        except ImportError as exc:
            raise ValueError(f"could not load learner-data validator: {exc}") from exc
        cur.execute("SELECT topic_id,payload FROM teachcopilot.catalog_topics")
        catalog={row["topic_id"]:_json(row["payload"]) for row in cur.fetchall()}
        findings=validate_card(card,catalog,f"{card.get('learner_id','card')}.json")
        errors=[finding for finding in findings if finding.is_error()]
        if errors:
            paths=", ".join(finding.json_path for finding in errors[:8])
            raise ValueError(f"learner card failed schema validation at {paths}")

    @staticmethod
    def _sync_card_relations(cur,lid,card):
        cur.execute("DELETE FROM teachcopilot.competencies WHERE learner_id=%s",(lid,))
        learner_model=card.get("learner_model") or {}
        competency_items=learner_model.get("competencies") or card.get("competence_map") or []
        by_topic={}
        for x in (card.get("knowledge") or []) + competency_items:
            if x.get("topic_id"):
                prior=by_topic.get(x["topic_id"],{})
                by_topic[x["topic_id"]]={**prior,**x}
        for x in by_topic.values():
            if x.get("topic_id"):
                cur.execute("""INSERT INTO teachcopilot.competencies(learner_id,topic_id,mastery,independence,status,last_assessed,evidence)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)""",(lid,x["topic_id"],x.get("mastery"),x.get("independence"),x.get("status"),_day(x.get("last_assessed")),Json(x)))
        cur.execute("DELETE FROM teachcopilot.error_patterns WHERE learner_id=%s",(lid,))
        for x in card.get("error_patterns",[]) or []:
            if x.get("error_tag"):
                cur.execute("""INSERT INTO teachcopilot.error_patterns(learner_id,topic_id,error_tag,subject,count,last_seen,details)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)""",(lid,x.get("topic_id","unknown"),x["error_tag"],x.get("subject"),int(x.get("count",1)),_day(x.get("last_seen")),Json(x)))
        cur.execute("DELETE FROM teachcopilot.scaffolding WHERE learner_id=%s",(lid,))
        for x in card.get("scaffolding_by_topic") or learner_model.get("scaffolding_by_topic") or []:
            if x.get("topic_id"): cur.execute("INSERT INTO teachcopilot.scaffolding(learner_id,topic_id,strategies) VALUES (%s,%s,%s)",(lid,x["topic_id"],Json(x.get("strategies",[]))))
        mvp=card.get("mvp") or {}
        event_groups=(("journal",mvp.get("journal",[])),("lesson",mvp.get("lessons") or mvp.get("lesson_history") or card.get("lesson_history",[])),("competency_history",learner_model.get("competency_history",[])))
        for event_type,items in event_groups:
            source_keys=[]
            for i,x in enumerate(items or []):
                source_key=f"{event_type}:{x.get('id') or x.get('date',i)}:{i}"
                source_keys.append(source_key)
                Store._upsert_event_projection(cur,lid,source_key,event_type,_day(x.get("date") or x.get("happened_at")),x)
            cur.execute("UPDATE teachcopilot.learner_events SET active=FALSE WHERE learner_id=%s AND event_type=%s AND active AND NOT (source_key=ANY(%s))",(lid,event_type,source_keys))
        point_keys=[]
        for i,x in enumerate((card.get("mvp") or {}).get("points_ledger",[]) or []):
            source_key=f"{x.get('id') or x.get('date',i)}:{i}"
            point_keys.append(source_key)
            Store._upsert_points_projection(cur,lid,source_key,int(x.get("points",0)),x.get("reason"),x.get("role"),x.get("topic_id"),_day(x.get("date")),x)
        cur.execute("UPDATE teachcopilot.points_ledger SET active=FALSE WHERE learner_id=%s AND active AND NOT (source_key=ANY(%s))",(lid,point_keys))
        cur.execute("DELETE FROM teachcopilot.projects WHERE learner_id=%s",(lid,))
        for i,x in enumerate(mvp.get("projects") or learner_model.get("projects") or []):
            cur.execute("INSERT INTO teachcopilot.projects(learner_id,project_key,title,payload) VALUES (%s,%s,%s,%s)",(lid,str(x.get("id",i)),x.get("title",f"project-{i}"),Json(x)))
        cur.execute("DELETE FROM teachcopilot.ai_observations WHERE learner_id=%s",(lid,))
        observations={"ai_usage":learner_model.get("ai_usage") or mvp.get("ai_usage"),"engagement":learner_model.get("engagement"),"gamification":learner_model.get("gamification"),"learning_preferences":learner_model.get("learning_preferences"),"zpd":learner_model.get("zpd")}
        for key,value in observations.items():
            if isinstance(value,dict): cur.execute("INSERT INTO teachcopilot.ai_observations(learner_id,observation_key,payload) VALUES (%s,%s,%s)",(lid,key,Json(value)))

    @staticmethod
    def _upsert_event_projection(cur,lid,source_key,event_type,happened_at,payload):
        cur.execute("SELECT id,event_key,payload FROM teachcopilot.learner_events WHERE learner_id=%s AND event_type=%s AND source_key=%s AND active FOR UPDATE",(lid,event_type,source_key))
        prior=cur.fetchone()
        if prior and _json(prior["payload"]) == payload:
            return
        if prior:
            cur.execute("UPDATE teachcopilot.learner_events SET active=FALSE WHERE id=%s",(prior["id"],))
        else:
            cur.execute("SELECT id,event_key,payload FROM teachcopilot.learner_events WHERE learner_id=%s AND event_type=%s AND source_key=%s ORDER BY created_at DESC LIMIT 1 FOR UPDATE",(lid,event_type,source_key))
            prior=cur.fetchone()
            if prior and _json(prior["payload"]) == payload:
                cur.execute("UPDATE teachcopilot.learner_events SET active=TRUE WHERE id=%s",(prior["id"],))
                return
        supersedes=prior["id"] if prior else None
        event_key=source_key if not prior else f"{source_key}:correction:{uuid.uuid4().hex}"
        cur.execute("""INSERT INTO teachcopilot.learner_events(learner_id,event_key,source_key,event_type,happened_at,payload,active,supersedes)
            VALUES (%s,%s,%s,%s,%s,%s,TRUE,%s)""",(lid,event_key,source_key,event_type,happened_at,Json(payload),supersedes))

    @staticmethod
    def _upsert_points_projection(cur,lid,source_key,points,reason,role,topic_id,happened_at,payload):
        cur.execute("SELECT id,entry_key,payload FROM teachcopilot.points_ledger WHERE learner_id=%s AND source_key=%s AND active FOR UPDATE",(lid,source_key))
        prior=cur.fetchone()
        if prior and _json(prior["payload"]) == payload:
            return
        if prior:
            cur.execute("UPDATE teachcopilot.points_ledger SET active=FALSE WHERE id=%s",(prior["id"],))
        else:
            cur.execute("SELECT id,entry_key,payload FROM teachcopilot.points_ledger WHERE learner_id=%s AND source_key=%s ORDER BY created_at DESC LIMIT 1 FOR UPDATE",(lid,source_key))
            prior=cur.fetchone()
            if prior and _json(prior["payload"]) == payload:
                cur.execute("UPDATE teachcopilot.points_ledger SET active=TRUE WHERE id=%s",(prior["id"],))
                return
        supersedes=prior["id"] if prior else None
        entry_key=source_key if not prior else f"{source_key}:correction:{uuid.uuid4().hex}"
        cur.execute("""INSERT INTO teachcopilot.points_ledger(learner_id,entry_key,source_key,points,reason,role,topic_id,happened_at,payload,active,supersedes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s)""",(lid,entry_key,source_key,points,reason,role,topic_id,happened_at,Json(payload),supersedes))

    def get_card(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT card,revision FROM teachcopilot.learners WHERE learner_id=%s",(learner_id,))
            row=cur.fetchone()
            return _json(row["card"]) if row else None

    def get_revision(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT revision FROM teachcopilot.learners WHERE learner_id=%s",(learner_id,))
            row=cur.fetchone()
            return row["revision"] if row else None

    def save_card(self, card, actor_id, expected_revision=None):
        with self._connect() as conn, conn.cursor() as cur: return self._save_card(cur,card,actor_id,expected_revision)

    def save_cards(self,cards,actor_id,expected_revisions=None,operation_id=None):
        expected_revisions=expected_revisions or {}
        with self._connect() as conn, conn.cursor() as cur:
            if operation_id:
                cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",(operation_id,))
                cur.execute("SELECT actor_id,result FROM teachcopilot.save_operations WHERE operation_id=%s FOR UPDATE",(operation_id,))
                prior=cur.fetchone()
                if prior:
                    if prior["actor_id"] != str(actor_id): raise ValueError("operation_id is already owned by another actor")
                    return _json(prior["result"])
            results=[]
            for card in cards:
                lid=card.get("learner_id")
                results.append(self._save_card(cur,card,str(actor_id),expected_revisions.get(lid)))
            if operation_id:
                cur.execute("INSERT INTO teachcopilot.save_operations(operation_id,actor_id,result) VALUES (%s,%s,%s)",(operation_id,str(actor_id),Json(results)))
            return results

    def create_proposal(self,proposal_id,learner_id=None,actor_id=None,base_revision=None,proposed_card=None,metadata=None,*,base_revisions=None,proposed_cards=None):
        # Batch form is preferred for a roster operation. The single-card form remains supported.
        if proposed_cards is None:
            proposed_cards={learner_id:proposed_card}
            base_revisions={learner_id:base_revision}
        if not proposed_cards or set(proposed_cards) != set(base_revisions or {}): raise ValueError("cards and base revisions must have the same nonempty learner IDs")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO teachcopilot.proposal_batches(proposal_id,actor_id,metadata) VALUES (%s,%s,%s)",(proposal_id,str(actor_id),Json(metadata or {})))
            for lid,card in proposed_cards.items():
                if card.get("learner_id") != lid: raise ValueError("proposal learner_id mismatch")
                cur.execute("SELECT revision FROM teachcopilot.learners WHERE learner_id=%s",(lid,))
                row=cur.fetchone()
                if not row: raise KeyError(lid)
                if row["revision"] != base_revisions[lid]: raise RevisionConflict(f"{lid} changed before proposal was created")
                cur.execute("INSERT INTO teachcopilot.proposal_batch_cards(proposal_id,learner_id,base_revision,proposed_card) VALUES (%s,%s,%s,%s)",(proposal_id,lid,base_revisions[lid],Json(card)))
            cur.execute("SELECT proposal_id,actor_id,metadata,status,created_at,confirmed_at FROM teachcopilot.proposal_batches WHERE proposal_id=%s",(proposal_id,))
            batch=dict(cur.fetchone()); batch["metadata"]=_json(batch["metadata"]); batch["cards"]=[]
            for lid,card in proposed_cards.items():
                batch["cards"].append({"learner_id":lid,"base_revision":base_revisions[lid],"proposed_card":card,"result_card":None})
            return batch

    def get_proposal(self,proposal_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT proposal_id,actor_id,metadata,status,created_at,confirmed_at FROM teachcopilot.proposal_batches WHERE proposal_id=%s",(proposal_id,))
            batch=cur.fetchone()
            if batch:
                cur.execute("SELECT learner_id,base_revision,proposed_card,result_card FROM teachcopilot.proposal_batch_cards WHERE proposal_id=%s ORDER BY learner_id",(proposal_id,))
                result=dict(batch); result["metadata"]=_json(result["metadata"]); result["cards"]=[]
                for c in cur.fetchall():
                    result["cards"].append({**dict(c),"proposed_card":_json(c["proposed_card"]),"result_card":_json(c["result_card"]) if c["result_card"] is not None else None})
                return result
            cur.execute("SELECT proposal_id,learner_id,actor_id,base_revision,proposed_card,metadata,status,result_card,created_at,confirmed_at FROM teachcopilot.card_proposals WHERE proposal_id=%s",(proposal_id,))
            row=cur.fetchone()
            if not row: return None
            result=dict(row)
            for key in ("proposed_card","metadata","result_card"):
                if result[key] is not None: result[key]=_json(result[key])
            return result

    def confirm_proposal(self,proposal_id,actor_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM teachcopilot.proposal_batches WHERE proposal_id=%s FOR UPDATE",(proposal_id,))
            batch=cur.fetchone()
            if batch:
                if str(batch["actor_id"]) != str(actor_id): raise PermissionError("proposal belongs to another actor")
                cur.execute("SELECT * FROM teachcopilot.proposal_batch_cards WHERE proposal_id=%s ORDER BY learner_id FOR UPDATE",(proposal_id,))
                rows=cur.fetchall()
                if batch["status"] == "confirmed": return [_json(r["result_card"]) for r in rows]
                if batch["status"] != "pending": raise ValueError("proposal is no longer pending")
                # All card writes and the proposal status change share this transaction.
                results=[]
                for r in rows:
                    card=self._save_card(cur,_json(r["proposed_card"]),str(actor_id),r["base_revision"])
                    cur.execute("UPDATE teachcopilot.proposal_batch_cards SET result_card=%s WHERE proposal_id=%s AND learner_id=%s",(Json(card),proposal_id,r["learner_id"]))
                    results.append(card)
                self._confirm_memory_proposal(cur,_json(batch["metadata"]),str(actor_id),results)
                cur.execute("UPDATE teachcopilot.proposal_batches SET status='confirmed',confirmed_at=now() WHERE proposal_id=%s",(proposal_id,))
                return results
            cur.execute("SELECT * FROM teachcopilot.card_proposals WHERE proposal_id=%s FOR UPDATE",(proposal_id,))
            row=cur.fetchone()
            if not row: raise KeyError(proposal_id)
            if str(row["actor_id"]) != str(actor_id): raise PermissionError("proposal belongs to another actor")
            if row["status"] == "confirmed": return _json(row["result_card"])
            if row["status"] != "pending": raise ValueError("proposal is no longer pending")
            card=self._save_card(cur,_json(row["proposed_card"]),str(actor_id),row["base_revision"])
            self._confirm_memory_proposal(cur,_json(row["metadata"]),str(actor_id),[card])
            cur.execute("UPDATE teachcopilot.card_proposals SET status='confirmed',result_card=%s,confirmed_at=now() WHERE proposal_id=%s",(Json(card),proposal_id))
            return card

    def reject_proposal(self,proposal_id,actor_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE teachcopilot.card_proposals SET status='rejected' WHERE proposal_id=%s AND actor_id=%s AND status='pending'",(proposal_id,str(actor_id)))
            return cur.rowcount>0

    def list_cards(self):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT learner_id,pseudonym,grade,age_group,language,revision,updated_at FROM teachcopilot.learners ORDER BY learner_id")
            return [dict(r) for r in cur.fetchall()]

    def catalog(self):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT topic_id,payload FROM teachcopilot.catalog_topics ORDER BY topic_id")
            return {r["topic_id"]:_json(r["payload"]) for r in cur.fetchall()}

    def create_account(self,email,display_name,password_hash,role,payload=None):
        if role not in {"teacher","student","admin"}: raise ValueError("invalid role")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO teachcopilot.accounts(email,display_name,password_hash,role,payload)
                VALUES (%s,%s,%s,%s,%s) RETURNING id,email,display_name,role,active,created_at,payload,display_timezone""",
                (email.strip().lower(),display_name,password_hash,role,Json(payload or {})))
            return dict(cur.fetchone())

    def create_student_account(self,email,display_name,password_hash,learner_id,teacher_id):
        """Create a student and bind their account in one transaction."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT a.role FROM teachcopilot.accounts a
                JOIN teachcopilot.account_learners al ON al.account_id=a.id
                WHERE a.id=%s AND al.learner_id=%s AND a.active FOR UPDATE OF a""",(teacher_id,learner_id))
            teacher=cur.fetchone()
            if not teacher or teacher["role"] not in ("teacher","admin"):
                raise PermissionError("teacher must have access to the learner")
            cur.execute("""INSERT INTO teachcopilot.accounts(email,display_name,password_hash,role)
                VALUES (%s,%s,%s,'student') RETURNING id,email,display_name,role,active,created_at,payload,display_timezone""",
                (email.strip().lower(),display_name,password_hash))
            account=dict(cur.fetchone())
            cur.execute("INSERT INTO teachcopilot.account_learners(account_id,learner_id,relationship) VALUES (%s,%s,'student')",(account["id"],learner_id))
            return account

    def create_assigned_learner(self,card,teacher_id):
        """Create a new validated learner and assign it to a teacher atomically."""
        lid=card.get("learner_id")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT role,active FROM teachcopilot.accounts WHERE id=%s FOR UPDATE",(teacher_id,))
            teacher=cur.fetchone()
            if not teacher or not teacher["active"] or teacher["role"] not in ("teacher","admin"):
                raise PermissionError("an active teacher account is required")
            cur.execute("SELECT 1 FROM teachcopilot.learners WHERE learner_id=%s",(lid,))
            if cur.fetchone(): raise ValueError("learner_id already exists")
            saved=self._save_card(cur,card,str(teacher_id),None)
            cur.execute("INSERT INTO teachcopilot.account_learners(account_id,learner_id,relationship) VALUES (%s,%s,'teacher')",(teacher_id,lid))
            return saved

    def get_account(self,email=None,account_id=None):
        if (email is None)==(account_id is None): raise ValueError("provide email or account_id")
        field,value=("email",email.strip().lower()) if email is not None else ("id",account_id)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id,email,display_name,password_hash,role,active,created_at,payload,display_timezone FROM teachcopilot.accounts WHERE {field}=%s",(value,))
            row=cur.fetchone()
            return dict(row) if row else None

    def create_session(self,account_id,token_hash,expires_at):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO teachcopilot.sessions(account_id,token_hash,expires_at) VALUES (%s,%s,%s) RETURNING id,account_id,token_hash,expires_at,created_at",(account_id,token_hash,expires_at))
            return dict(cur.fetchone())

    def get_session(self,token_hash):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT s.id AS session_id,s.account_id,s.expires_at,a.email,a.display_name,a.role,a.active,a.display_timezone
                FROM teachcopilot.sessions s JOIN teachcopilot.accounts a ON a.id=s.account_id
                WHERE s.token_hash=%s AND s.expires_at>now() AND a.active""",(token_hash,))
            row=cur.fetchone()
            return dict(row) if row else None

    def get_account_display_timezone(self, account_id):
        from .timezones import validate_timezone
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT display_timezone FROM teachcopilot.accounts WHERE id=%s", (account_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(account_id)
            return validate_timezone(row["display_timezone"])

    def set_account_display_timezone(self, account_id, display_timezone):
        from .timezones import validate_timezone
        zone = validate_timezone(display_timezone)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.accounts SET display_timezone=%s
                WHERE id=%s RETURNING display_timezone""", (zone, account_id))
            row = cur.fetchone()
            if row is None:
                raise KeyError(account_id)
            return {"display_timezone": row["display_timezone"]}

    def get_learner_learning_timezone(self, learner_id):
        from .timezones import validate_timezone
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT learning_timezone FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(learner_id)
            return validate_timezone(row["learning_timezone"])

    def revoke_session(self,token_hash):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM teachcopilot.sessions WHERE token_hash=%s",(token_hash,))
            return cur.rowcount>0

    def assign_learner(self,account_id,learner_id,relationship="teacher"):
        with self._connect() as conn, conn.cursor() as cur:
            # F-033: the guard and the insert share one transaction and the
            # account's links are locked, so concurrent links serialize. The link
            # set cannot be locked while it is empty, so the parent account row is
            # locked first (the same order create_student_account uses).
            cur.execute("SELECT id FROM teachcopilot.accounts WHERE id=%s FOR UPDATE",(account_id,))
            if cur.fetchone() is None: raise KeyError(account_id)
            cur.execute("SELECT learner_id FROM teachcopilot.account_learners WHERE account_id=%s FOR UPDATE",(account_id,))
            linked=[str(row["learner_id"]) for row in cur.fetchall()]
            if relationship=="student" and any(existing!=str(learner_id) for existing in linked):
                raise PermissionError("student account is already linked to another learner")
            cur.execute("""INSERT INTO teachcopilot.account_learners(account_id,learner_id,relationship) VALUES (%s,%s,%s)
                ON CONFLICT(account_id,learner_id) DO UPDATE SET relationship=EXCLUDED.relationship""",(account_id,learner_id,relationship))

    def authorize_learner_access(self,account_id,learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM teachcopilot.account_learners WHERE account_id=%s AND learner_id=%s",(account_id,learner_id))
            return cur.fetchone() is not None

    def upsert_material(self,material: Mapping[str,Any],embedding: Sequence[float]|None=None,embedding_model: str|None=None):
        item=dict(material)
        if embedding is not None: item["embedding"]=embedding
        if embedding_model is not None: item["embedding_model"]=embedding_model
        return self.upsert_chunks([item],embedding_model=embedding_model)[0]

    def upsert_chunks(self,chunks: Iterable[Mapping[str,Any]],embedding_model=None,embedding_dimension=None):
        results=[]
        prepared=[]
        for chunk in chunks:
            x=dict(chunk); emb=x.get("embedding") or x.get("vector")
            model=x.get("embedding_model") or (embedding_model if emb is not None else None)
            if emb is not None and embedding_dimension is not None and len(emb)!=embedding_dimension: raise ValueError("embedding dimension mismatch")
            if emb is not None and (not emb or len(emb)>2000): raise ValueError("embedding dimension must be between 1 and 2000")
            if emb is not None and not model: raise ValueError("embedding_model is required when storing a vector")
            prepared.append((x,emb,model,str(x.get("id") or hashlib.sha256((str(x.get("source_file"))+str(x.get("content"))).encode()).hexdigest()),"["+",".join(str(float(v)) for v in emb)+"]" if emb is not None else None))
        # One database transaction for every chunk in this indexing batch.
        with self._connect() as conn, conn.cursor() as cur:
            for x,emb,model,cid,vec in prepared:
                meta=dict(x.get("metadata") or {})
                raw=x.get("raw_snapshot") or x
                if emb is not None:
                    cur.execute("INSERT INTO teachcopilot.embedding_models(model_name,dimension) VALUES (%s,%s) ON CONFLICT(model_name) DO NOTHING",(model,len(emb)))
                    cur.execute("SELECT dimension FROM teachcopilot.embedding_models WHERE model_name=%s FOR UPDATE",(model,))
                    if cur.fetchone()["dimension"] != len(emb): raise ValueError("embedding dimension differs from the registered model dimension")
                cur.execute("""INSERT INTO teachcopilot.materials(id,content,source_file,topic_id,grade,subject_id,source_version,embedding,embedding_dimension,embedding_model,metadata,raw_snapshot,active)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,TRUE) ON CONFLICT(id) DO UPDATE SET content=EXCLUDED.content,
                    source_file=EXCLUDED.source_file,topic_id=EXCLUDED.topic_id,grade=EXCLUDED.grade,subject_id=EXCLUDED.subject_id,
                    embedding=CASE WHEN teachcopilot.materials.source_version IS DISTINCT FROM EXCLUDED.source_version OR teachcopilot.materials.content IS DISTINCT FROM EXCLUDED.content THEN EXCLUDED.embedding ELSE COALESCE(EXCLUDED.embedding,teachcopilot.materials.embedding) END,
                    embedding_dimension=CASE WHEN teachcopilot.materials.source_version IS DISTINCT FROM EXCLUDED.source_version OR teachcopilot.materials.content IS DISTINCT FROM EXCLUDED.content THEN EXCLUDED.embedding_dimension ELSE COALESCE(EXCLUDED.embedding_dimension,teachcopilot.materials.embedding_dimension) END,
                    embedding_model=CASE WHEN teachcopilot.materials.source_version IS DISTINCT FROM EXCLUDED.source_version OR teachcopilot.materials.content IS DISTINCT FROM EXCLUDED.content THEN EXCLUDED.embedding_model ELSE COALESCE(EXCLUDED.embedding_model,teachcopilot.materials.embedding_model) END,
                    source_version=EXCLUDED.source_version,metadata=EXCLUDED.metadata,
                    raw_snapshot=EXCLUDED.raw_snapshot,active=TRUE,updated_at=now()
                    RETURNING id,content,source_file,topic_id,grade,subject_id,source_version,embedding_model,metadata""",
                    (cid,str(x.get("content","")),str(x.get("source_file","unknown")),x.get("topic_id"),x.get("grade"),x.get("subject_id"),str(x.get("source_version","1")),vec,len(emb) if emb is not None else None,model,Json(meta),Json(raw)))
                results.append(dict(cur.fetchone()))
        return results

    def update_embeddings(self,chunks: Iterable[Mapping[str,Any]],embedding_model: str,embedding_dimension: int) -> int:
        """Atomically attach vectors to the exact material snapshots embedded."""
        if not embedding_model: raise ValueError("embedding_model is required")
        if not isinstance(embedding_dimension,int) or not 1 <= embedding_dimension <= 2000:
            raise ValueError("embedding dimension must be between 1 and 2000")
        prepared=[]
        for chunk in chunks:
            x=dict(chunk); emb=x.get("embedding") or x.get("vector")
            if not x.get("id") or x.get("source_version") is None or x.get("content") is None:
                raise ValueError("embedding snapshots require id, source_version, and content")
            if emb is None or len(emb)!=embedding_dimension: raise ValueError("embedding dimension mismatch")
            vec="["+",".join(str(float(v)) for v in emb)+"]"
            prepared.append((str(x["id"]),str(x["source_version"]),str(x["content"]),vec))
        if not prepared: return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO teachcopilot.embedding_models(model_name,dimension) VALUES (%s,%s) ON CONFLICT(model_name) DO NOTHING",(embedding_model,embedding_dimension))
            cur.execute("SELECT dimension FROM teachcopilot.embedding_models WHERE model_name=%s FOR UPDATE",(embedding_model,))
            if cur.fetchone()["dimension"] != embedding_dimension:
                raise ValueError("embedding dimension differs from the registered model dimension")
            # Lock and validate the complete batch before writing a vector.
            for material_id,source_version,content,_ in prepared:
                cur.execute("SELECT source_version,content,active FROM teachcopilot.materials WHERE id=%s FOR UPDATE",(material_id,))
                row=cur.fetchone()
                if not row or not row["active"] or row["source_version"] != source_version or row["content"] != content:
                    raise StaleMaterialSnapshotError(f"material snapshot changed during embedding: {material_id}")
            updated=0
            for material_id,source_version,content,vec in prepared:
                cur.execute("""UPDATE teachcopilot.materials SET embedding=%s::vector,embedding_dimension=%s,
                    embedding_model=%s,updated_at=now() WHERE id=%s AND source_version=%s AND content=%s AND active""",
                    (vec,embedding_dimension,embedding_model,material_id,source_version,content))
                if cur.rowcount != 1: raise StaleMaterialSnapshotError(f"material snapshot changed during embedding: {material_id}")
                updated+=1
            return updated

    def material_rows(self,filters=None):
        filters=filters or {}; allowed=("source_file","topic_id","grade","subject_id","source_version")
        where=[]; params=[]
        for key in allowed:
            if filters.get(key) is not None: where.append(f"{key}=%s"); params.append(filters[key])
        sql="SELECT id,content,source_file,topic_id,grade,subject_id,source_version,embedding_model,metadata,raw_snapshot FROM teachcopilot.materials WHERE active"
        if where: sql+=" AND "+" AND ".join(where)
        sql+=" ORDER BY source_file,id"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql,params)
            return [dict(r) for r in cur.fetchall()]

    def embedding_index_status(self,embedding_model):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) AS total FROM teachcopilot.materials WHERE active")
            total=cur.fetchone()["total"]
            cur.execute("SELECT dimension FROM teachcopilot.embedding_models WHERE model_name=%s",(embedding_model,))
            model=cur.fetchone()
            dimension=model["dimension"] if model else None
            cur.execute("SELECT count(*) AS indexed FROM teachcopilot.materials WHERE active AND embedding_model=%s AND embedding IS NOT NULL",(embedding_model,))
            indexed=cur.fetchone()["indexed"]
            return {"embedding_model":embedding_model,"dimension":dimension,"total":total,"indexed":indexed,"ready":total==0 or indexed==total}

    def hybrid_search(self,query,query_embedding,filters=None,limit=5,embedding_model=None):
        filters=filters or {}; where=[]; pred_params=[]
        for key in ("subject_id","grade","topic_id","source_file"):
            val=filters.get(key)
            if val is not None:
                if isinstance(val,(list,tuple,set)): where.append(f"{key}=ANY(%s)"); pred_params.append(list(val))
                else: where.append(f"{key}=%s"); pred_params.append(val)
        prefix = filters.get('topic_prefix')
        if prefix is not None:
            if prefix not in {'programming.python.', 'programming.scratch.'}:
                raise ValueError('Unsupported curriculum prefix')
            where.append('topic_id LIKE %s')
            pred_params.append(prefix + '%')
        lim=max(1,min(int(limit),100))
        if query_embedding is not None:
            if not query_embedding or len(query_embedding)>2000: raise ValueError("query embedding dimension must be between 1 and 2000")
            vec="["+",".join(str(float(v)) for v in query_embedding)+"]"
            if not embedding_model: raise ValueError("embedding_model is required for vector search")
            sql="""WITH eligible AS (SELECT id,content,source_file,topic_id,grade,subject_id,source_version,metadata,
                (1-(embedding <=> %s::vector)) AS similarity,ts_rank(search_document,plainto_tsquery('russian',%s)||plainto_tsquery('simple',%s)) AS text_score
                FROM teachcopilot.materials WHERE active AND embedding_model=%s AND embedding_dimension=%s"""
            params=[vec,query,query,embedding_model,len(query_embedding)]
            if where: sql+=" AND "+" AND ".join(where); params.extend(pred_params)
            # id is the final tie-break everywhere, so equal scores give the same order on every run.
            sql+="""), vr AS (SELECT id,row_number() OVER (ORDER BY similarity DESC,id) AS rank FROM eligible),
                tr AS (SELECT id,row_number() OVER (ORDER BY text_score DESC,id) AS rank FROM eligible WHERE text_score>0),
                fused AS (SELECT vr.id,1.0/(60+vr.rank)+COALESCE(1.0/(60+tr.rank),0) AS rrf FROM vr LEFT JOIN tr USING(id))
                SELECT e.*,f.rrf AS score FROM eligible e JOIN fused f USING(id) ORDER BY f.rrf DESC,e.id LIMIT %s"""
            params.append(lim)
        else:
            sql="SELECT id,content,source_file,topic_id,grade,subject_id,source_version,metadata,ts_rank(search_document,plainto_tsquery('russian',%s)||plainto_tsquery('simple',%s)) AS score FROM teachcopilot.materials WHERE active"
            if where: sql+=" AND "+" AND ".join(where)
            sql+=" ORDER BY score DESC,id LIMIT %s"
            params=[query,query,*pred_params,lim]
        with self._connect() as conn, conn.cursor() as cur:
            if query_embedding is not None:
                # Signal stale/partial indices instead of silently serving an incomplete corpus.
                cur.execute("SELECT count(*) AS total FROM teachcopilot.materials WHERE active"+(" AND "+" AND ".join(where) if where else ""),pred_params)
                total=cur.fetchone()["total"]
                cur.execute("SELECT count(*) AS indexed FROM teachcopilot.materials WHERE active AND embedding_model=%s AND embedding_dimension=%s AND embedding IS NOT NULL",(embedding_model,len(query_embedding)))
                indexed=cur.fetchone()["indexed"]
                # With hard subject/grade/topic filters, compare only matching material rows.
                if where:
                    cur.execute("SELECT count(*) AS indexed FROM teachcopilot.materials WHERE active AND embedding_model=%s AND embedding_dimension=%s AND embedding IS NOT NULL AND "+" AND ".join(where),[embedding_model,len(query_embedding),*pred_params])
                    indexed=cur.fetchone()["indexed"]
                if total and indexed < total:
                    raise IndexNotReadyError(f"embedding index is incomplete: {indexed} of {total} matching materials indexed for {embedding_model}")
            cur.execute(sql,params)
            return [dict(r) for r in cur.fetchall()]

