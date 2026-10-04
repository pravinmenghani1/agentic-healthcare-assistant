"""Deterministic tests (no LLM / Ollama required): parsing, scheduling, plan guard-rails, response composition."""
from datetime import date

from hca import config, db
from hca.agent.graph import Plan, PlanStep, _repair_plan, compose_status, resolve_date
from hca.ingest import load_records_xlsx, parse_pdf
from hca.tools import appointments as appt
from hca.tools import records


def _plan(**kw) -> Plan:
    base = dict(intent="t", patient_name=None, patient_relation=None, patient_age=None, reported_conditions=[],
                specialty=None, preferred_date=None, preferred_time=None, appointment_reason=None,
                appointment_id=None, search_topic=None, search_question=None, record_note=None, steps=[])
    base.update(kw)
    return Plan(**base)


# ---------------------------------------------------------------- ingestion
def test_xlsx_deduplicates_patients():
    patients = load_records_xlsx(config.DATA_DIR / "records.xlsx")
    assert [p["name"] for p in patients] == ["Rahul Negi", "Rebeca Nagle", "Ramesh Kulkarni", "Anjali Mehra",
                                             "David Thompson"]


def test_parse_short_report():
    doc = parse_pdf(config.DATA_DIR / "sample_report_david.pdf")
    assert doc.patient_name == "David Thompson" and doc.dob == "1972-03-14"
    (enc,) = doc.encounters
    assert enc.visit_date == "2024-01-10"
    assert enc.diagnoses == [{"name": "Type 2 Diabetes Mellitus", "code": "E11.9"}]
    assert enc.plan.startswith("Increase metformin")


def test_parse_multi_encounter_pdf():
    doc = parse_pdf(config.DATA_DIR / "sample_patient.pdf")
    assert doc.patient_name == "Rebeca Nagle"
    assert [e.visit_date for e in doc.encounters] == ["2022-09-07", "2022-09-20"]
    assert doc.encounters[1].diagnoses[0]["code"] == "M94.0"
    assert "PCOS" in doc.past_history
    assert doc.supplementary and doc.supplementary[0]["pages"].startswith("4")


# ---------------------------------------------------------------- scheduling
def test_specialty_normalization():
    assert appt.normalize_specialty("kidney specialist") == "Nephrologist"
    assert appt.normalize_specialty("chronic kidney disease") == "Nephrologist"
    assert appt.normalize_specialty("Endocrinologist") == "Endocrinologist"
    assert appt.normalize_specialty("dermatology") is None


def test_booking_prevents_double_booking(fresh_db):
    pid = records.register_patient("Test Patient", 50)["patient"]["id"]
    slots = appt.find_slots("Nephrologist", limit=2)
    assert slots["ok"]
    sid = slots["slots"][0]["slot_id"]
    assert appt.book_appointment(pid, sid, "test")["ok"]
    assert not appt.book_appointment(pid, sid, "again")["ok"]          # slot already taken
    assert sid not in [s["slot_id"] for s in appt.find_slots("Nephrologist", limit=50)["slots"]]


def test_cancel_reopens_slot(fresh_db):
    pid = records.register_patient("Test Patient", 50)["patient"]["id"]
    sid = appt.find_slots("Cardiologist")["slots"][0]["slot_id"]
    aid = appt.book_appointment(pid, sid)["appointment"]["appointment_id"]
    assert appt.cancel_appointment(aid)["ok"]
    assert db.query_one("SELECT status FROM slots WHERE id=?", (sid,))["status"] == "open"


def test_slots_prefer_nearby_doctor(fresh_db):
    res = appt.find_slots("Nephrologist", near="52 Residency Road, Chennai", limit=3)
    assert {s["doctor"] for s in res["slots"]} == {"Dr. Priya Raman"}


def test_tool_calls_are_logged(fresh_db):
    before = db.query_one("SELECT COUNT(*) n FROM tool_logs")["n"]
    appt.find_slots("Dermatology")
    row = db.query_one("SELECT * FROM tool_logs ORDER BY id DESC LIMIT 1")
    assert db.query_one("SELECT COUNT(*) n FROM tool_logs")["n"] == before + 1
    assert row["tool"] == "find_doctor_slots" and row["success"] == 0


# ---------------------------------------------------------------- planning guard-rails
def test_resolve_date():
    sunday = date(2026, 10, 4)
    assert resolve_date("next friday afternoon", sunday) == "2026-10-09"
    assert resolve_date("tomorrow", sunday) == "2026-10-05"
    assert resolve_date("in 3 days", sunday) == "2026-10-07"
    assert resolve_date("book a nephrologist", sunday) is None


def test_repair_plan_fixes_dependencies_and_hallucinations():
    plan = _plan(patient_name="Father", reported_conditions=["chronic kidney disease"], preferred_date="2026-01-01",
                 preferred_time="morning", steps=[PlanStep(tool="book_appointment", goal="book"),
                                                  PlanStep(tool="medical_search", goal="search")])
    plan, repairs = _repair_plan(plan, "My father has chronic kidney disease, book a doctor")
    assert [s.tool for s in plan.steps] == ["identify_patient", "find_doctor_slots", "book_appointment",
                                            "medical_search"]
    assert plan.patient_name is None and plan.preferred_date is None and plan.preferred_time is None
    assert plan.specialty == "Nephrologist" and plan.search_topic == "chronic kidney disease"
    assert repairs


def test_compose_status_reports_only_tool_facts():
    state = {"results": [
        {"tool": "identify_patient", "ok": False, "result": {"ok": False, "error": "no patient name given"}},
        {"tool": "book_appointment", "ok": False, "result": {"ok": False, "skipped": True}}],
        "alerts": [], "clarification": "Could you tell me your father's full name?", "patient": None}
    text = compose_status(state)
    assert "on hold" in text and "full name" in text and "booked:" not in text
