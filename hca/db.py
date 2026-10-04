"""SQLite persistence: patient DB (EHR), doctor schedule, appointments, memory events and LLMOps logs."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    dob TEXT,
    phone TEXT,
    email TEXT,
    address TEXT,
    conditions TEXT DEFAULT '[]',
    medications TEXT DEFAULT '[]',
    allergies TEXT DEFAULT '[]',
    summary TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS medical_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    visit_date TEXT,
    record_type TEXT,          -- visit_note | document | attendant_note
    source TEXT,               -- file name or 'ui' / 'agent'
    subjective TEXT,
    objective TEXT,
    assessment TEXT,
    plan TEXT,
    diagnoses TEXT DEFAULT '[]',
    medications TEXT DEFAULT '[]',
    raw_text TEXT,
    created_by TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL,
    location TEXT,
    bio TEXT
);
CREATE TABLE IF NOT EXISTS slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER NOT NULL REFERENCES doctors(id),
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'   -- open | booked
);
CREATE INDEX IF NOT EXISTS idx_slots_doctor ON slots(doctor_id, start_ts);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    doctor_id INTEGER NOT NULL REFERENCES doctors(id),
    slot_id INTEGER NOT NULL REFERENCES slots(id),
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'booked', -- booked | cancelled | completed
    booked_by TEXT,
    run_id TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    kind TEXT,                 -- interaction | reported_condition | preference | history_summary
    content TEXT,
    run_id TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    query TEXT,
    patient_id INTEGER,
    plan TEXT,
    trace TEXT,
    response TEXT,
    success INTEGER,
    latency_ms INTEGER,
    source TEXT,               -- chat | scenario | eval
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS tool_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    tool TEXT,
    args TEXT,
    success INTEGER,
    error TEXT,
    latency_ms INTEGER,
    output_preview TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS medical_searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    topic TEXT,
    question TEXT,
    summary TEXT,
    sources TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS web_cache (
    url TEXT PRIMARY KEY,
    content TEXT,
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metrics TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_run_id INTEGER REFERENCES eval_runs(id),
    module TEXT,
    case_id TEXT,
    query TEXT,
    expected TEXT,
    predicted TEXT,
    grade TEXT,
    score REAL,
    details TEXT
);
"""

JSON_COLUMNS = {"conditions", "medications", "allergies", "diagnoses", "args", "plan", "trace", "sources", "metrics", "details"}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with session() as conn:
        conn.executescript(SCHEMA)


def to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key in JSON_COLUMNS & out.keys():
        if isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out


def query(sql: str, params: tuple = ()) -> list[dict]:
    with session() as conn:
        return [to_dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    """Run a write statement and return lastrowid (or rowcount for UPDATE/DELETE)."""
    with session() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid if sql.lstrip().upper().startswith("INSERT") else cur.rowcount


def dumps(value) -> str:
    return json.dumps(value, default=str)
