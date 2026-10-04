"""Doctor Schedule API: specialty lookup, slot discovery, booking, cancellation and rescheduling."""
import re
from datetime import date, datetime, timedelta

from .. import db
from ..tracking import logged_tool

# Free-text phrases -> canonical specialty. Order matters: first match wins.
SPECIALTY_SYNONYMS = {
    "Nephrologist": ["nephro", "kidney", "renal", "ckd", "dialysis"],
    "Cardiologist": ["cardio", "heart", "chest pain", "palpitation", "hypertension", "blood pressure"],
    "Endocrinologist": ["endocrin", "diabet", "thyroid", "hormone", "metformin", "insulin"],
    "Pulmonologist": ["pulmo", "lung", "asthma", "copd", "breath", "respiratory", "cough"],
    "Neurologist": ["neuro", "migraine", "headache", "seizure", "stroke"],
    "Gynecologist": ["gyn", "pcos", "pap", "menstru", "pregnan", "obstetric"],
    "Orthopedist": ["ortho", "bone", "joint", "fracture", "costochondritis", "back pain"],
    "Nutritionist": ["nutrition", "dietitian", "diet"],
    "General Physician": ["general", "gp", "family", "physician", "primary care", "fever", "check", "routine"],
}


def normalize_specialty(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    for specialty in SPECIALTY_SYNONYMS:
        if specialty.lower() in t:
            return specialty
    for specialty, keys in SPECIALTY_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(k)}", t) for k in keys):
            return specialty
    return None


def list_specialties() -> list[str]:
    return [r["specialty"] for r in db.query("SELECT DISTINCT specialty FROM doctors ORDER BY specialty")]


def _slot_row(r: dict) -> dict:
    start = datetime.fromisoformat(r["start_ts"])
    return {"slot_id": r["id"], "doctor_id": r["doctor_id"], "doctor": r["doctor_name"], "specialty": r["specialty"],
            "location": r["location"], "start": r["start_ts"], "end": r["end_ts"],
            "display": start.strftime("%a %d %b %Y, %I:%M %p")}


@logged_tool("find_doctor_slots")
def find_slots(specialty: str, date_from: str | None = None, date_to: str | None = None,
               time_of_day: str | None = None, doctor_name: str | None = None, near: str | None = None,
               limit: int = 5) -> dict:
    """Open slots for a specialty, earliest first. Prefers doctors located near `near` (a patient address) and
    widens the window by two weeks if the preferred dates are full."""
    canonical = normalize_specialty(specialty) or specialty
    if canonical not in list_specialties():
        return {"ok": False, "error": f"No doctors for specialty '{specialty}'", "available_specialties": list_specialties()}
    start = max(datetime.fromisoformat(date_from) if date_from else datetime.now(), datetime.now())
    end = datetime.fromisoformat(date_to) + timedelta(days=1) if date_to else start + timedelta(days=7)

    def fetch(lo: datetime, hi: datetime) -> list[dict]:
        sql = ("SELECT s.*, d.name AS doctor_name, d.specialty, d.location FROM slots s JOIN doctors d ON d.id=s.doctor_id "
               "WHERE d.specialty=? AND s.status='open' AND s.start_ts>=? AND s.start_ts<?")
        params: list = [canonical, lo.isoformat(timespec="seconds"), hi.isoformat(timespec="seconds")]
        if doctor_name:
            sql += " AND d.name LIKE ?"
            params.append(f"%{doctor_name.replace('Dr.', '').strip()}%")
        if time_of_day == "morning":
            sql += " AND time(s.start_ts) < '12:00'"
        elif time_of_day == "afternoon":
            sql += " AND time(s.start_ts) >= '12:00'"
        rows = db.query(sql + " ORDER BY s.start_ts", tuple(params))
        if near:
            words = {w for w in re.findall(r"[a-z]{4,}", near.lower())} - {"road", "street", "apartments", "hill"}
            local = [r for r in rows if words & set(re.findall(r"[a-z]{4,}", (r["location"] or "").lower()))]
            rows = local or rows
        return rows[:limit]

    rows, widened = fetch(start, end), False
    if not rows:
        rows, widened = fetch(start, end + timedelta(days=14)), True
    if not rows:
        return {"ok": False, "error": f"No open {canonical} slots in the next weeks", "specialty": canonical}
    return {"ok": True, "specialty": canonical, "widened_search": widened, "near": near,
            "slots": [_slot_row(r) for r in rows]}


@logged_tool("book_appointment")
def book_appointment(patient_id: int, slot_id: int, reason: str = "", booked_by: str = "agent",
                     run_id: str | None = None) -> dict:
    with db.session() as conn:
        # Atomic claim: only succeeds if the slot is still open (prevents double-booking).
        claimed = conn.execute("UPDATE slots SET status='booked' WHERE id=? AND status='open'", (slot_id,)).rowcount
        if not claimed:
            return {"ok": False, "error": f"Slot {slot_id} is no longer available"}
        slot = conn.execute("SELECT doctor_id FROM slots WHERE id=?", (slot_id,)).fetchone()
        cur = conn.execute(
            "INSERT INTO appointments (patient_id, doctor_id, slot_id, reason, status, booked_by, run_id, created_at, updated_at) "
            "VALUES (?,?,?,?, 'booked', ?,?,?,?)",
            (patient_id, slot["doctor_id"], slot_id, reason, booked_by, run_id, db.now(), db.now()))
        appointment_id = cur.lastrowid
    return {"ok": True, "appointment": get_appointment(appointment_id)}


def get_appointment(appointment_id: int) -> dict | None:
    return db.query_one(
        "SELECT a.id AS appointment_id, a.status, a.reason, a.booked_by, a.created_at, a.patient_id, p.name AS patient, "
        "d.name AS doctor, d.specialty, d.location, s.start_ts AS start, s.end_ts AS end "
        "FROM appointments a JOIN patients p ON p.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id "
        "JOIN slots s ON s.id=a.slot_id WHERE a.id=?", (appointment_id,))


@logged_tool("cancel_appointment")
def cancel_appointment(appointment_id: int) -> dict:
    appt = db.query_one("SELECT * FROM appointments WHERE id=?", (appointment_id,))
    if not appt or appt["status"] != "booked":
        return {"ok": False, "error": f"No active appointment #{appointment_id}"}
    with db.session() as conn:
        conn.execute("UPDATE appointments SET status='cancelled', updated_at=? WHERE id=?", (db.now(), appointment_id))
        conn.execute("UPDATE slots SET status='open' WHERE id=?", (appt["slot_id"],))
    return {"ok": True, "appointment": get_appointment(appointment_id)}


@logged_tool("reschedule_appointment")
def reschedule_appointment(appointment_id: int, new_slot_id: int) -> dict:
    old = get_appointment(appointment_id)
    if not old or old["status"] != "booked":
        return {"ok": False, "error": f"No active appointment #{appointment_id}"}
    booked = book_appointment(old["patient_id"], new_slot_id, reason=old["reason"], booked_by="reschedule")
    if not booked["ok"]:
        return booked
    cancel_appointment(appointment_id)
    return {"ok": True, "old": old, "appointment": booked["appointment"]}


@logged_tool("list_appointments")
def list_appointments(patient_id: int | None = None, doctor_id: int | None = None, status: str | None = None,
                      upcoming_only: bool = False) -> list[dict]:
    sql = ("SELECT a.id AS appointment_id, a.status, a.reason, a.booked_by, a.created_at, a.patient_id, p.name AS patient, "
           "a.doctor_id, d.name AS doctor, d.specialty, d.location, s.start_ts AS start "
           "FROM appointments a JOIN patients p ON p.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id "
           "JOIN slots s ON s.id=a.slot_id WHERE 1=1")
    params: list = []
    for col, val in (("a.patient_id", patient_id), ("a.doctor_id", doctor_id), ("a.status", status)):
        if val is not None:
            sql += f" AND {col}=?"
            params.append(val)
    if upcoming_only:
        sql += " AND s.start_ts>=?"
        params.append(datetime.now().isoformat(timespec="seconds"))
    return db.query(sql + " ORDER BY s.start_ts", tuple(params))


def doctor_day_schedule(doctor_id: int, day: date) -> list[dict]:
    return db.query(
        "SELECT s.id AS slot_id, s.start_ts, s.status, a.id AS appointment_id, p.name AS patient, a.reason "
        "FROM slots s LEFT JOIN appointments a ON a.slot_id=s.id AND a.status='booked' "
        "LEFT JOIN patients p ON p.id=a.patient_id WHERE s.doctor_id=? AND date(s.start_ts)=? ORDER BY s.start_ts",
        (doctor_id, day.isoformat()))
