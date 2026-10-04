"""Patient DB / EHR API: lookup, registration, record add/update, PDF ingestion and LLM history summaries."""
import difflib
import json
import re
from datetime import date

from pydantic import BaseModel, Field

from .. import db, memory, prompts
from ..ingest import parse_pdf
from ..llm import get_llm
from ..tracking import logged_tool

UPDATABLE_FIELDS = {"name", "age", "gender", "dob", "phone", "email", "address", "conditions", "medications",
                    "allergies", "summary"}


# ---------------------------------------------------------------- structured LLM outputs

class Diagnosis(BaseModel):
    condition: str
    code: str | None = Field(None, description="ICD-10 code if known")
    date: str | None = None
    status: str | None = Field(None, description="active, resolved, chronic, or reported-unconfirmed")


class Visit(BaseModel):
    date: str | None
    reason: str
    outcome: str


class HistorySummary(BaseModel):
    overview: str = Field(description="2-3 sentence overview of the patient")
    diagnoses: list[Diagnosis]
    treatments: list[str] = Field(description="treatments, procedures and care plans")
    medications: list[str]
    alerts: list[str]
    recent_visits: list[Visit]
    follow_ups: list[str]


class ExtractedRecord(BaseModel):
    visit_date: str | None = None
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    diagnoses: list[Diagnosis] = []
    medications: list[str] = []
    allergies: list[str] = []


class ProfileEnrichment(BaseModel):
    conditions: list[str]
    medications: list[str]
    allergies: list[str]
    summary: str


# ---------------------------------------------------------------- lookups

def get_patient(patient_id: int) -> dict | None:
    return db.query_one("SELECT * FROM patients WHERE id=?", (patient_id,))


def get_records(patient_id: int) -> list[dict]:
    return db.query("SELECT * FROM medical_records WHERE patient_id=? ORDER BY visit_date DESC, id DESC", (patient_id,))


def list_patients() -> list[dict]:
    return db.query("SELECT * FROM patients ORDER BY name")


@logged_tool("identify_patient")
def find_patient(name: str | None = None, description: str | None = None) -> dict:
    """Exact name -> fuzzy name -> semantic search over patient profiles."""
    patients = list_patients()
    if name:
        n = name.strip().lower()
        exact = [p for p in patients if p["name"].lower() == n]
        if exact:
            return {"ok": True, "match": "exact", "patient": exact[0]}
        scored = []
        for p in patients:
            full = difflib.SequenceMatcher(None, n, p["name"].lower()).ratio()
            parts = max(difflib.SequenceMatcher(None, t, pt).ratio()
                        for t in n.split() for pt in p["name"].lower().split())
            scored.append((max(full, parts if len(n.split()) == 1 else full), p))
        scored.sort(key=lambda x: -x[0])
        if scored and scored[0][0] >= 0.85 and (len(scored) == 1 or scored[0][0] - scored[1][0] > 0.05):
            return {"ok": True, "match": "fuzzy", "score": round(scored[0][0], 2), "patient": scored[0][1]}
        candidates = [p["name"] for s, p in scored if s >= 0.6]
        if candidates:
            return {"ok": False, "error": f"Ambiguous patient name '{name}'", "candidates": candidates}
    if description:
        # Search profiles and clinical notes; rank patients by their closest chunk (L2 distance, lower = closer).
        best: dict[int, dict] = {}
        for h in memory.recall(description, k=12, kinds=("profile", "record")):
            if h["patient_id"] not in best or h["score"] < best[h["patient_id"]]["score"]:
                best[h["patient_id"]] = h
        ranked = sorted(best.values(), key=lambda h: h["score"])
        if ranked and ranked[0]["score"] < 0.9:
            return {"ok": True, "match": "semantic", "score": round(ranked[0]["score"], 3),
                    "patient": get_patient(ranked[0]["patient_id"]), "candidates": [h["name"] for h in ranked[:3]]}
    return {"ok": False, "error": f"No patient found for name={name!r}", "candidates": []}


# ---------------------------------------------------------------- writes

@logged_tool("register_patient")
def register_patient(name: str, age: int | None = None, gender: str | None = None, phone: str | None = None,
                     email: str | None = None, address: str | None = None, conditions: list[str] | None = None,
                     summary: str | None = None) -> dict:
    existing = db.query_one("SELECT * FROM patients WHERE lower(name)=lower(?)", (name,))
    if existing:
        return {"ok": True, "created": False, "patient": existing}
    pid = db.execute(
        "INSERT INTO patients (name, age, gender, phone, email, address, conditions, summary, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (name, age, gender, phone, email, address, db.dumps(conditions or []), summary, db.now(), db.now()))
    memory.index_patient(pid)
    return {"ok": True, "created": True, "patient": get_patient(pid)}


@logged_tool("update_patient")
def update_patient(patient_id: int, **fields) -> dict:
    fields = {k: v for k, v in fields.items() if k in UPDATABLE_FIELDS}
    if not fields or not get_patient(patient_id):
        return {"ok": False, "error": "Nothing to update or unknown patient"}
    for k in ("conditions", "medications", "allergies"):
        if k in fields and not isinstance(fields[k], str):
            fields[k] = db.dumps(fields[k])
    sets = ", ".join(f"{k}=?" for k in fields)
    db.execute(f"UPDATE patients SET {sets}, updated_at=? WHERE id=?", (*fields.values(), db.now(), patient_id))
    memory.index_patient(patient_id)
    return {"ok": True, "patient": get_patient(patient_id)}


JUNK_VALUES = {"none", "none reported", "n/a", "na", "unknown", "nil", "nkda", "no known allergies", "not specified"}


def _key(value: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", value).strip().lower()


def _merge(existing: list, new: list) -> list:
    out, seen = [], set()
    for x in list(existing) + list(new):
        if x and _key(x) not in JUNK_VALUES and _key(x) not in seen:
            seen.add(_key(x))
            out.append(x)
    return out


def insert_record(patient_id: int, record_type: str, source: str, created_by: str, visit_date=None, subjective="",
                  objective="", assessment="", plan="", diagnoses=None, medications=None, raw_text="") -> int:
    return db.execute(
        "INSERT INTO medical_records (patient_id, visit_date, record_type, source, subjective, objective, assessment, "
        "plan, diagnoses, medications, raw_text, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (patient_id, visit_date, record_type, source, subjective, objective, assessment, plan,
         db.dumps(diagnoses or []), db.dumps(medications or []), raw_text, created_by, db.now()))


@logged_tool("add_medical_record")
def add_record(patient_id: int, note: str, author: str = "attendant", structured: dict | None = None,
               use_llm: bool = True) -> dict:
    """Add structured or unstructured history. Free text is structured by the LLM, then merged into the profile."""
    patient = get_patient(patient_id)
    if not patient:
        return {"ok": False, "error": f"Unknown patient {patient_id}"}
    if structured is None and use_llm:
        chain = prompts.RECORD_EXTRACTION | get_llm().with_structured_output(ExtractedRecord)
        rec = chain.invoke({"today": date.today().isoformat(), "patient": patient["name"], "note": note})
        structured = rec.model_dump()
    structured = structured or {"subjective": note}
    diagnoses = [{"name": d.get("condition") or d.get("name"), "code": d.get("code")}
                 for d in structured.get("diagnoses", [])]
    rid = insert_record(patient_id, "attendant_note", "agent" if author == "agent" else "ui", author,
                        visit_date=structured.get("visit_date") or date.today().isoformat(),
                        subjective=structured.get("subjective", ""), objective=structured.get("objective", ""),
                        assessment=structured.get("assessment", ""), plan=structured.get("plan", ""),
                        diagnoses=diagnoses, medications=structured.get("medications", []), raw_text=note)
    db.execute("UPDATE patients SET conditions=?, medications=?, allergies=?, updated_at=? WHERE id=?", (
        db.dumps(_merge(patient["conditions"], [d["name"] for d in diagnoses])),
        db.dumps(_merge(patient["medications"], structured.get("medications", []))),
        db.dumps(_merge(patient["allergies"], structured.get("allergies", []))), db.now(), patient_id))
    memory.index_patient(patient_id)
    return {"ok": True, "record_id": rid, "structured": structured, "patient": get_patient(patient_id)}


@logged_tool("ingest_pdf")
def ingest_pdf_document(source, filename: str | None = None, patient_id: int | None = None,
                        created_by: str = "ui") -> dict:
    """Parse a clinical PDF, attach it to the named (or given) patient, and index it in memory."""
    doc = parse_pdf(source, name=filename)
    if patient_id is None:
        if not doc.patient_name:
            return {"ok": False, "error": "Could not find a patient name in the PDF; select a patient explicitly"}
        p = db.query_one("SELECT * FROM patients WHERE lower(name)=lower(?)", (doc.patient_name,))
        if p is None:
            p = register_patient(doc.patient_name, gender=doc.gender, phone=doc.phone, address=doc.address)["patient"]
        patient_id = p["id"]
    patient = get_patient(patient_id)
    updates = {k: v for k, v in (("dob", doc.dob), ("gender", doc.gender), ("phone", doc.phone),
                                 ("address", doc.address)) if v and not patient.get(k)}
    if updates:
        db.execute(f"UPDATE patients SET {', '.join(f'{k}=?' for k in updates)} WHERE id=?", (*updates.values(), patient_id))
    for enc in doc.encounters:
        insert_record(patient_id, "visit_note", doc.source, created_by, enc.visit_date, enc.subjective, enc.objective,
                      enc.assessment, enc.plan, enc.diagnoses, enc.medications, enc.raw_text)
    for supp in doc.supplementary:
        insert_record(patient_id, "document", f"{doc.source} (pages {supp['pages']})", created_by,
                      raw_text=supp["text"])
    dx = [d["name"] for e in doc.encounters for d in e.diagnoses
          if not d["code"].startswith("Z")]  # Z-codes are screenings / well visits, not conditions
    meds = [m for e in doc.encounters for m in e.medications]
    patient = get_patient(patient_id)
    db.execute("UPDATE patients SET conditions=?, medications=?, updated_at=? WHERE id=?",
               (db.dumps(_merge(patient["conditions"], dx + doc.past_history)),
                db.dumps(_merge(patient["medications"], meds)), db.now(), patient_id))
    memory.index_patient(patient_id)
    return {"ok": True, "patient_id": patient_id, "patient": patient["name"], "encounters": len(doc.encounters),
            "documents": len(doc.supplementary)}


def enrich_profile(patient_id: int) -> dict:
    """LLM pass over all records to fill conditions / medications / allergies / summary."""
    patient = get_patient(patient_id)
    records = get_records(patient_id)
    if not records:
        return patient
    text = "\n\n".join(memory.record_text(r) for r in records)[:12000]
    chain = prompts.PROFILE_ENRICHMENT | get_llm().with_structured_output(ProfileEnrichment)
    out = chain.invoke({"patient": f"{patient['name']}, {patient.get('age')} y, {patient.get('gender')}", "records": text})
    db.execute("UPDATE patients SET conditions=?, medications=?, allergies=?, summary=?, updated_at=? WHERE id=?", (
        db.dumps(_merge(patient["conditions"], out.conditions)), db.dumps(_merge([], out.medications) or patient["medications"]),
        db.dumps(_merge(patient["allergies"], out.allergies)), patient.get("summary") or out.summary, db.now(),
        patient_id))
    return get_patient(patient_id)


# ---------------------------------------------------------------- summarization (prompt chain + memory)

@logged_tool("get_medical_history")
def summarize_history(patient_id: int, focus: str | None = None) -> dict:
    patient = get_patient(patient_id)
    if not patient:
        return {"ok": False, "error": f"Unknown patient {patient_id}"}
    records = get_records(patient_id)
    # Every visit/attendant note goes in, compacted and clearly dated so the LLM keeps encounters apart.
    notes = [_compact_note(r) for r in records if r["record_type"] != "document"]
    # Long supplementary documents (labs, CCD exports) are only included via vector recall of relevant chunks.
    doc_ids = {r["id"] for r in records if r["record_type"] == "document"}
    if doc_ids:
        hits = memory.recall(focus or "lab results medications problems", patient_id, k=10, kinds=("record",))
        chunks = [h["text"] for h in hits if h.get("record_id") in doc_ids][:3]
        if chunks:
            notes.append("SUPPLEMENTARY DOCUMENT EXCERPTS (labs / chart export):\n" + "\n---\n".join(chunks))
    record_text = "\n\n".join(notes)
    memories = memory.recall(focus or "recent interactions and reported conditions", patient_id, k=4, kinds=("memory",))
    chain = prompts.HISTORY_SUMMARY | get_llm().with_structured_output(HistorySummary)
    summary = chain.invoke({
        "today": date.today().isoformat(),
        "profile": memory.profile_text(patient),
        "records": record_text or "No clinical records on file.",
        "memories": "\n".join(m["text"] for m in memories) or "None.",
        "focus": focus or "general overview",
    })
    # Guard-rail: keep only ICD codes that literally occur in the source records (LLMs invent plausible codes).
    source = record_text + " " + memory.profile_text(patient)
    for d in summary.diagnoses:
        if d.code and d.code not in source:
            d.code = None
    return {"ok": True, "patient": patient["name"], "summary": summary.model_dump(), "markdown": render_summary(summary),
            "records_used": len(records), "memories_used": len(memories)}


def _compact_note(r: dict, limit: int = 700) -> str:
    def cut(text: str | None) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= limit else text[:limit] + " ..."
    dx = "; ".join(f"{d['name']} ({d.get('code') or 'no code'})" for d in r.get("diagnoses") or []) or "none coded"
    lines = [f"=== {r['record_type'].upper()} on {r.get('visit_date') or 'unknown date'} (record #{r['id']}) ===",
             f"Diagnoses at this visit: {dx}"]
    for label in ("subjective", "assessment", "plan"):
        if r.get(label):
            lines.append(f"{label.title()}: {cut(r[label])}")
    if r.get("objective"):
        lines.append(f"Objective: {cut(r['objective'])[:300]}")
    if r.get("medications"):
        lines.append(f"Medications listed: {', '.join(r['medications'])}")
    if not any(r.get(k) for k in ("subjective", "assessment", "plan")) and r.get("raw_text"):
        lines.append(cut(r["raw_text"]))
    return "\n".join(lines)


def render_summary(s: HistorySummary) -> str:
    lines = [s.overview, ""]
    if s.diagnoses:
        lines.append("**Diagnoses**")
        lines += [f"- {d.condition}" + (f" ({d.code})" if d.code else "") + (f", {d.date}" if d.date else "")
                  + (f" [{d.status}]" if d.status else "") for d in s.diagnoses]
    for title, items in (("Treatments", s.treatments), ("Medications", s.medications), ("Alerts", s.alerts),
                         ("Follow-ups", s.follow_ups)):
        if items:
            lines.append(f"**{title}**")
            lines += [f"- {i}" for i in items]
    if s.recent_visits:
        lines.append("**Recent visits**")
        lines += [f"- {v.date or 'n/a'}: {v.reason} → {v.outcome}" for v in s.recent_visits]
    return "\n".join(lines)


def patient_brief(patient: dict) -> str:
    return json.dumps({k: patient.get(k) for k in ("id", "name", "age", "gender", "conditions", "medications",
                                                   "allergies")}, default=str)
