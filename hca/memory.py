"""Memory modules.

* VectorStore – a persisted FAISS index (nomic-embed-text embeddings).
* Patient memory – long-term, per-patient context: profile summaries, clinical record chunks and
  memories of past interactions, all embedded in one FAISS index filtered by patient_id.
* Medical knowledge base – chunks of MedlinePlus / WHO pages fetched by the search tool (RAG corpus).
Short-term (conversation) memory lives in the agent state / Streamlit session.
"""
import threading

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config, db
from .llm import get_embeddings

_lock = threading.RLock()
_stores: dict[str, "VectorStore"] = {}
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


class VectorStore:
    def __init__(self, name: str):
        self.name = name
        self.path = config.FAISS_DIR / name
        self._index: FAISS | None = None

    @property
    def index(self) -> FAISS | None:
        if self._index is None and (self.path / "index.faiss").exists():
            self._index = FAISS.load_local(str(self.path), get_embeddings(), allow_dangerous_deserialization=True)
        return self._index

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        with _lock:
            if self.index is None:
                self._index = FAISS.from_documents(docs, get_embeddings())
            else:
                self._index.add_documents(docs)
            self._index.save_local(str(self.path))

    def search(self, query: str, k: int = 4, filter=None, fetch_k: int = 50) -> list[tuple[Document, float]]:
        if self.index is None:
            return []
        with _lock:
            return self.index.similarity_search_with_score(query, k=k, filter=filter, fetch_k=fetch_k)

    def reset(self) -> None:
        with _lock:
            self._index = None
            for f in ("index.faiss", "index.pkl"):
                (self.path / f).unlink(missing_ok=True)

    def all_documents(self) -> list[Document]:
        if self.index is None:
            return []
        return list(self.index.docstore._dict.values())

    def __len__(self) -> int:
        return self.index.index.ntotal if self.index is not None else 0


def store(name: str) -> VectorStore:
    with _lock:
        if name not in _stores:
            _stores[name] = VectorStore(name)
        return _stores[name]


def patient_store() -> VectorStore:
    return store("patient_memory")


def knowledge_store() -> VectorStore:
    return store("medical_kb")


# ---------------------------------------------------------------- patient memory

def profile_text(p: dict) -> str:
    conds = ", ".join(p.get("conditions") or []) or "none recorded"
    meds = ", ".join(p.get("medications") or []) or "none recorded"
    return (f"Patient profile: {p['name']}, {p.get('age')} y, {p.get('gender')}. "
            f"Conditions: {conds}. Medications: {meds}. Allergies: {', '.join(p.get('allergies') or []) or 'none recorded'}. "
            f"Summary: {p.get('summary') or 'n/a'}")


def record_text(r: dict) -> str:
    dx = "; ".join(f"{d['name']} ({d.get('code') or 'no code'})" for d in r.get("diagnoses") or [])
    parts = [f"{r.get('record_type', 'record')} dated {r.get('visit_date') or 'unknown'} (source: {r.get('source')})"]
    for label in ("subjective", "objective", "assessment", "plan"):
        if r.get(label):
            parts.append(f"{label.title()}: {r[label]}")
    if dx:
        parts.append(f"Diagnoses: {dx}")
    if not any(r.get(l) for l in ("subjective", "objective", "assessment", "plan")) and r.get("raw_text"):
        parts.append(r["raw_text"])
    return "\n".join(parts)


def index_patient(patient_id: int) -> int:
    """(Re)embed a patient's profile and all records. Returns number of chunks added."""
    p = db.query_one("SELECT * FROM patients WHERE id=?", (patient_id,))
    if not p:
        return 0
    docs = [Document(profile_text(p), metadata={"patient_id": patient_id, "kind": "profile", "name": p["name"]})]
    for r in db.query("SELECT * FROM medical_records WHERE patient_id=?", (patient_id,)):
        for chunk in splitter.split_text(record_text(r)):
            docs.append(Document(chunk, metadata={"patient_id": patient_id, "kind": "record", "record_id": r["id"],
                                                  "name": p["name"], "date": r.get("visit_date")}))
    for e in db.query("SELECT * FROM memory_events WHERE patient_id=?", (patient_id,)):
        docs.append(_event_doc(e, p["name"]))
    _remove_patient_docs(patient_id)
    patient_store().add(docs)
    return len(docs)


def _remove_patient_docs(patient_id: int) -> None:
    s = patient_store()
    if s.index is None:
        return
    ids = [i for i, d in s.index.docstore._dict.items() if d.metadata.get("patient_id") == patient_id]
    if ids:
        with _lock:
            s.index.delete(ids)
            s.index.save_local(str(s.path))


def rebuild_patient_index() -> int:
    patient_store().reset()
    return sum(index_patient(p["id"]) for p in db.query("SELECT id FROM patients"))


def _event_doc(e: dict, name: str) -> Document:
    return Document(f"Memory ({e['kind']}, {e['created_at'][:10]}): {e['content']}",
                    metadata={"patient_id": e["patient_id"], "kind": "memory", "event_kind": e["kind"],
                              "name": name, "date": e["created_at"], "run_id": e.get("run_id")})


def remember(patient_id: int | None, kind: str, content: str, run_id: str | None = None) -> None:
    """Persist a long-term memory event (SQL for audit + FAISS for semantic recall)."""
    event_id = db.execute(
        "INSERT INTO memory_events (patient_id, kind, content, run_id, created_at) VALUES (?,?,?,?,?)",
        (patient_id, kind, content, run_id, db.now()))
    if patient_id is not None:
        e = db.query_one("SELECT * FROM memory_events WHERE id=?", (event_id,))
        p = db.query_one("SELECT name FROM patients WHERE id=?", (patient_id,))
        patient_store().add([_event_doc(e, p["name"] if p else "unknown")])


def forget_run(run_id: str) -> None:
    """Remove all memory written by one agent run (SQL + vectors), e.g. after an evaluation run."""
    db.execute("DELETE FROM memory_events WHERE run_id=?", (run_id,))
    s = patient_store()
    if s.index is None:
        return
    ids = [i for i, d in s.index.docstore._dict.items() if d.metadata.get("run_id") == run_id]
    if ids:
        with _lock:
            s.index.delete(ids)
            s.index.save_local(str(s.path))


def recall(query: str, patient_id: int | None = None, k: int = 5, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """Semantic lookup in patient memory, optionally restricted to one patient / document kinds."""
    def keep(meta: dict) -> bool:
        return (patient_id is None or meta.get("patient_id") == patient_id) and (kinds is None or meta.get("kind") in kinds)

    hits = patient_store().search(query, k=k, filter=keep, fetch_k=200)
    return [{"text": d.page_content, "score": float(score), **d.metadata} for d, score in hits]


def recent_events(patient_id: int, limit: int = 5) -> list[dict]:
    return db.query("SELECT * FROM memory_events WHERE patient_id=? ORDER BY id DESC LIMIT ?", (patient_id, limit))
