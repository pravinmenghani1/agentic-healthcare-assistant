"""Agentic workflow (LangGraph):

    load_context ──► plan ──► execute ⟲ (one sub-goal per pass) ──► respond ──► END

* load_context – short-term chat memory + long-term patient memory lookups
* plan         – LLM planner decomposes the request into sub-goals mapped to tools (structured output),
                 followed by deterministic validation/repair of the plan
* execute      – runs each sub-goal's tool, passing results forward (patient -> slots -> booking)
* respond      – composes the reply, writes long-term memory and the run log (trace) for LLMOps
"""
import re
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .. import db, memory, prompts
from ..llm import get_llm
from ..tools import appointments as appt
from ..tools import medical_search as search
from ..tools import records
from ..tracking import current_run_id

ToolName = Literal["identify_patient", "register_patient", "get_medical_history", "add_medical_record",
                   "find_doctor_slots", "book_appointment", "cancel_appointment", "list_appointments",
                   "medical_search"]
TOOL_ORDER = list(ToolName.__args__)
PATIENT_TOOLS = {"register_patient", "get_medical_history", "add_medical_record", "book_appointment",
                 "cancel_appointment", "list_appointments"}


class PlanStep(BaseModel):
    tool: ToolName
    goal: str = Field(description="the sub-goal this step achieves")


class Plan(BaseModel):
    # Every field is required (nullable) so small local models are forced to consider each one.
    intent: str = Field(description="one-line summary of what the user wants")
    patient_name: str | None = Field(description="full name of the patient the request is about, else null")
    patient_relation: str | None = Field(description="self, father, mother, child, ... or null")
    patient_age: int | None = Field(description="age if stated, else null")
    reported_conditions: list[str] = Field(description="conditions the user says the patient has")
    specialty: str | None = Field(description="doctor specialty needed for booking, e.g. Nephrologist, else null")
    preferred_date: str | None = Field(description="YYYY-MM-DD or null")
    preferred_time: Literal["morning", "afternoon", "any"] | None
    appointment_reason: str | None
    appointment_id: int | None
    search_topic: str | None = Field(description="disease/condition to look up, in plain English, or null")
    search_question: str | None = Field(description="exact information wanted about the topic, or null")
    record_note: str | None = Field(description="clinical information to add to the patient's record, or null")
    steps: list[PlanStep]


class AgentState(TypedDict, total=False):
    run_id: str
    query: str
    source: str
    chat_history: list[dict]
    active_patient_id: int | None
    pending_request: str | None
    context: dict
    plan: dict
    plan_repairs: list[str]
    patient: dict | None
    step_idx: int
    results: list[dict]
    alerts: list[str]
    clarification: str | None
    trace: list[dict]
    response: str
    started: float


# ---------------------------------------------------------------- helpers

def _emit(config: RunnableConfig, event: dict) -> None:
    cb: Callable | None = (config or {}).get("configurable", {}).get("on_event")
    if cb:
        cb(event)


def _trace(state: AgentState, node: str, detail: Any, t0: float) -> list[dict]:
    return state.get("trace", []) + [{"node": node, "detail": detail, "ms": int((time.perf_counter() - t0) * 1000),
                                      "at": db.now()}]


def _calendar(days: int = 14) -> str:
    today = date.today()
    return ", ".join(f"{(today + timedelta(d)).strftime('%a')} {(today + timedelta(d)).isoformat()}" for d in range(days))


def _heuristic_plan(query: str) -> Plan:
    """Keyword fallback used when the LLM planner fails, so the agent still acts."""
    q = query.lower()
    steps, name = [], None
    for p in records.list_patients():
        if p["name"].lower() in q or p["name"].split()[0].lower() in q.split():
            name = p["name"]
    specialty = appt.normalize_specialty(q)
    if re.search(r"\b(book|appointment|schedule|see a|consult)", q):
        steps += [PlanStep(tool="identify_patient", goal="identify patient"),
                  PlanStep(tool="find_doctor_slots", goal="find slots"),
                  PlanStep(tool="book_appointment", goal="book")]
    if re.search(r"\b(history|records?|summar(y|ize) (his|her|the patient))", q):
        steps.insert(1 if steps else 0, PlanStep(tool="get_medical_history", goal="summarize history"))
    topic = None
    m = re.search(r"(?:has|with|about|for|of)\s+([a-z0-9 \-]+?(?:disease|diabetes|hypertension|cancer|infection|asthma|failure))", q)
    if m:
        topic = m.group(1).strip()
    if re.search(r"\b(treatment|symptom|what is|information|latest|cause)", q):
        steps.append(PlanStep(tool="medical_search", goal="search medical information"))
    return Plan(intent="(heuristic plan)", patient_name=name, patient_relation=None, patient_age=None,
                reported_conditions=[topic] if topic else [], specialty=specialty, preferred_date=None,
                preferred_time=None, appointment_reason=None, appointment_id=None, search_topic=topic,
                search_question=None, record_note=None,
                steps=steps or [PlanStep(tool="medical_search", goal="answer")])


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
RELATION_WORDS = {"father", "dad", "mother", "mom", "mum", "son", "daughter", "wife", "husband", "brother", "sister",
                  "grandfather", "grandmother", "child", "patient", "me", "myself", "self", "him", "her", "unknown"}


def resolve_date(text: str, today: date | None = None) -> str | None:
    """Resolve explicit/relative dates in user text deterministically (small LLMs are unreliable at date math)."""
    today = today or date.today()
    t = text.lower()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        return m.group(1)
    if re.search(r"\btoday\b", t):
        return today.isoformat()
    if re.search(r"\btomorrow\b", t):
        return (today + timedelta(days=1)).isoformat()
    m = re.search(r"\bin (\d+) days?\b", t)
    if m:
        return (today + timedelta(days=int(m.group(1)))).isoformat()
    m = re.search(r"\b(" + "|".join(WEEKDAYS) + r")\b", t)
    if m:  # "Friday" / "next Friday" -> the first upcoming Friday (never today)
        ahead = (WEEKDAYS.index(m.group(1)) - today.weekday()) % 7 or 7
        return (today + timedelta(days=ahead)).isoformat()
    if re.search(r"\bnext week\b", t):
        return (today + timedelta(days=7 - today.weekday())).isoformat()
    return None


def _repair_plan(plan: Plan, query: str, pending: str | None = None) -> tuple[Plan, list[str]]:
    """Deterministic guard-rails on the LLM plan: dependencies, ordering, missing or hallucinated parameters."""
    repairs = []
    text = f"{pending or ''} {query}"
    if plan.patient_name and (plan.patient_name.lower().strip() in RELATION_WORDS or len(plan.patient_name) < 3):
        repairs.append(f"patient_name '{plan.patient_name}' is not a name -> null")
        plan.patient_name = None
    resolved = resolve_date(text)
    if resolved != plan.preferred_date:
        repairs.append(f"preferred_date {plan.preferred_date} -> {resolved} (deterministic date resolution)")
        plan.preferred_date = resolved
    tod = "morning" if "morning" in text.lower() else "afternoon" if re.search(r"afternoon|evening", text.lower()) else None
    if tod != (plan.preferred_time if plan.preferred_time != "any" else None):
        repairs.append(f"preferred_time {plan.preferred_time} -> {tod}")
        plan.preferred_time = tod
    tools = []
    for s in plan.steps:
        if s.tool not in [t.tool for t in tools]:
            tools.append(s)
    names = [s.tool for s in tools]
    if "book_appointment" in names and "find_doctor_slots" not in names:
        tools.append(PlanStep(tool="find_doctor_slots", goal="find available slots before booking"))
        repairs.append("added find_doctor_slots before book_appointment")
    if any(s.tool in PATIENT_TOOLS for s in tools) and "identify_patient" not in [s.tool for s in tools]:
        tools.append(PlanStep(tool="identify_patient", goal="identify the patient"))
        repairs.append("added identify_patient (patient-specific tools present)")
    tools.sort(key=lambda s: TOOL_ORDER.index(s.tool))
    plan.steps = tools
    if "find_doctor_slots" in [s.tool for s in tools]:
        spec = appt.normalize_specialty(plan.specialty) or appt.normalize_specialty(" ".join(plan.reported_conditions)) \
            or appt.normalize_specialty(query)
        if spec != plan.specialty:
            repairs.append(f"specialty '{plan.specialty}' -> '{spec}'")
            plan.specialty = spec
    if "medical_search" in [s.tool for s in tools] and not plan.search_topic:
        plan.search_topic = (plan.reported_conditions or [None])[0] or query[:80]
        repairs.append(f"search_topic defaulted to '{plan.search_topic}'")
    if plan.preferred_date:
        try:
            date.fromisoformat(plan.preferred_date)
        except ValueError:
            repairs.append(f"dropped invalid preferred_date '{plan.preferred_date}'")
            plan.preferred_date = None
    return plan, repairs


def _brief(result: Any) -> Any:
    """Trim large tool outputs for traces / prompts."""
    if isinstance(result, dict):
        return {k: _brief(v) for k, v in result.items() if k not in ("context", "patient_raw")}
    if isinstance(result, list):
        return [_brief(v) for v in result[:6]]
    if isinstance(result, str) and len(result) > 1500:
        return result[:1500] + " ..."
    return result


# ---------------------------------------------------------------- nodes

def load_context(state: AgentState, config: RunnableConfig) -> AgentState:
    t0 = time.perf_counter()
    _emit(config, {"node": "load_context", "status": "running"})
    ctx: dict = {"chat_history": state.get("chat_history", [])[-6:]}
    pid = state.get("active_patient_id")
    if pid:
        p = records.get_patient(pid)
        ctx["active_patient"] = records.patient_brief(p) if p else None
        ctx["patient_memories"] = [m["text"] for m in memory.recall(state["query"], pid, k=3, kinds=("memory",))]
    if state.get("pending_request"):
        ctx["pending_request"] = state["pending_request"]
    detail = {k: v for k, v in ctx.items() if v}
    _emit(config, {"node": "load_context", "status": "done", "detail": detail})
    return {"context": ctx, "trace": _trace(state, "load_context", detail, t0)}


def plan_node(state: AgentState, config: RunnableConfig) -> AgentState:
    t0 = time.perf_counter()
    _emit(config, {"node": "plan", "status": "running"})
    ctx = state["context"]
    lines = []
    if ctx.get("active_patient"):
        lines.append(f"ACTIVE PATIENT (selected in the app; use if the request is about them): {ctx['active_patient']}")
    if ctx.get("patient_memories"):
        lines.append("PATIENT MEMORY: " + " | ".join(ctx["patient_memories"]))
    if ctx.get("chat_history"):
        lines.append("RECENT CONVERSATION:\n" + "\n".join(f"{m['role']}: {m['content'][:300]}" for m in ctx["chat_history"]))
    if ctx.get("pending_request"):
        lines.append(f"PENDING REQUEST (awaiting missing info): {ctx['pending_request']}")
    used_fallback = False
    try:
        planner = prompts.PLANNER | get_llm().with_structured_output(Plan)
        plan = planner.invoke({"today": date.today().isoformat(), "calendar": _calendar(),
                               "context": "\n".join(lines), "query": state["query"]})
    except Exception as exc:  # noqa: BLE001 - any planner failure falls back to heuristics
        used_fallback = True
        plan = _heuristic_plan(state["query"])
        plan.intent += f" [LLM planner failed: {type(exc).__name__}]"
    plan, repairs = _repair_plan(plan, state["query"], state.get("pending_request"))
    if used_fallback:
        repairs.insert(0, "heuristic fallback planner used")
    plan_dict = plan.model_dump()
    _emit(config, {"node": "plan", "status": "done", "detail": {"plan": plan_dict, "repairs": repairs}})
    return {"plan": plan_dict, "plan_repairs": repairs, "step_idx": 0, "results": [], "alerts": [],
            "trace": _trace(state, "plan", {"plan": plan_dict, "repairs": repairs}, t0)}


def execute_node(state: AgentState, config: RunnableConfig) -> AgentState:
    t0 = time.perf_counter()
    plan = state["plan"]
    step = plan["steps"][state["step_idx"]]
    _emit(config, {"node": "execute", "status": "running", "step": step})
    updates: AgentState = {}
    patient = state.get("patient")
    alerts = list(state.get("alerts", []))
    try:
        if step["tool"] != "identify_patient" and step["tool"] in PATIENT_TOOLS and step["tool"] != "register_patient" \
                and not patient:
            result = {"ok": False, "skipped": True, "error": "patient not identified"}
        else:
            result = EXECUTORS[step["tool"]](state, plan, patient, alerts, updates)
    except Exception as exc:  # noqa: BLE001 - a failing tool must not crash the agent
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    entry = {"tool": step["tool"], "goal": step["goal"], "ok": bool(result.get("ok")), "result": result}
    _emit(config, {"node": "execute", "status": "done", "step": step, "ok": entry["ok"], "detail": _brief(result)})
    return {**updates, "alerts": alerts, "results": state["results"] + [entry], "step_idx": state["step_idx"] + 1,
            "trace": _trace(state, f"execute:{step['tool']}", {"goal": step["goal"], "ok": entry["ok"],
                                                                "result": _brief(result)}, t0)}


# ---------------------------------------------------------------- tool executors

def _identify(state, plan, patient, alerts, updates):
    name = plan.get("patient_name")
    active = state.get("active_patient_id")
    relation = (plan.get("patient_relation") or "").lower()
    if name:
        res = records.find_patient(name=name)
        if not res["ok"] and "register_patient" in [s["tool"] for s in plan["steps"]]:
            res = records.register_patient(name, plan.get("patient_age"), conditions=plan.get("reported_conditions"))
    elif active and relation in ("", "self", "patient", "me", "him", "her"):
        res = {"ok": True, "match": "active_patient", "patient": records.get_patient(active)}
    else:
        who = f"your {relation}" if relation and relation != "self" else "the patient"
        res = {"ok": False, "error": "no patient name given"}
        updates["clarification"] = f"Could you tell me {who}'s full name so I can find their record?"
    if not res["ok"]:
        if res.get("candidates"):
            updates["clarification"] = f"I found several possible matches: {', '.join(res['candidates'])}. Which one?"
        elif "clarification" not in updates:
            updates["clarification"] = (f"I couldn't find a patient named '{name}'. Please check the spelling, or ask me "
                                        f"to register them as a new patient.")
        updates["pending_request"] = state.get("pending_request") or state["query"]
        return res
    p = res["patient"]
    updates["patient"] = p
    updates["pending_request"] = None
    if plan.get("patient_age") and p.get("age") and abs(plan["patient_age"] - p["age"]) > 1:
        alerts.append(f"Stated age {plan['patient_age']} differs from age on file ({p['age']}); please verify.")
    known = " ".join(p.get("conditions") or []).lower()
    adding_record = "add_medical_record" in [s["tool"] for s in plan["steps"]]  # the note itself goes to the EHR
    for cond in [] if adding_record else plan.get("reported_conditions") or []:
        if cond and cond.lower() not in known:
            alerts.append(f"'{cond}' was reported by the user but is not in the EHR (recorded as unconfirmed).")
            memory.remember(p["id"], "reported_condition", f"User reported (unconfirmed): {cond}", state["run_id"])
    return {"ok": True, "match": res.get("match"), "patient": records.patient_brief(p)}


def _register(state, plan, patient, alerts, updates):
    if patient:
        return {"ok": True, "created": False, "note": "patient already identified"}
    if not plan.get("patient_name"):
        updates["clarification"] = "Please provide the new patient's full name to register them."
        return {"ok": False, "error": "no name to register"}
    res = records.register_patient(plan["patient_name"], plan.get("patient_age"),
                                   conditions=plan.get("reported_conditions"))
    updates["patient"] = res["patient"]
    return {"ok": True, "created": res["created"], "patient": records.patient_brief(res["patient"])}


def _history(state, plan, patient, alerts, updates):
    focus = plan.get("search_topic") or ", ".join(plan.get("reported_conditions") or []) or None
    return records.summarize_history(patient["id"], focus)


def _add_record(state, plan, patient, alerts, updates):
    note = plan.get("record_note") or state["query"]
    res = records.add_record(patient["id"], note, author="agent")
    if res.get("ok"):
        updates["patient"] = res["patient"]
        return {"ok": True, "record_id": res["record_id"], "structured": res["structured"]}
    return res


def _find_slots(state, plan, patient, alerts, updates):
    specialty = plan.get("specialty")
    if not specialty:
        return {"ok": False, "error": "could not determine the doctor specialty",
                "available_specialties": appt.list_specialties()}
    return appt.find_slots(specialty, date_from=plan.get("preferred_date"), date_to=plan.get("preferred_date"),
                           time_of_day=None if plan.get("preferred_time") == "any" else plan.get("preferred_time"),
                           near=(patient or {}).get("address"))


def _book(state, plan, patient, alerts, updates):
    slots_res = next((r["result"] for r in reversed(state["results"]) if r["tool"] == "find_doctor_slots"), None)
    if not slots_res or not slots_res.get("ok"):
        return {"ok": False, "error": "no available slots to book" + (f": {slots_res.get('error')}" if slots_res else "")}
    specialty = slots_res["specialty"]
    existing = [a for a in appt.list_appointments(patient_id=patient["id"], status="booked", upcoming_only=True)
                if a["specialty"] == specialty]
    if existing:
        return {"ok": True, "already_booked": True, "appointment": existing[0],
                "note": "patient already has an upcoming appointment with this specialty; not double-booking"}
    reason = plan.get("appointment_reason") or ", ".join(plan.get("reported_conditions") or []) or plan.get("intent")
    for slot in slots_res["slots"]:  # fall through to the next slot if one was taken concurrently
        res = appt.book_appointment(patient["id"], slot["slot_id"], reason, booked_by="agent", run_id=state["run_id"])
        if res["ok"]:
            if slots_res.get("widened_search"):
                alerts.append("No slots on the preferred date; booked the earliest available instead.")
            return res
    return {"ok": False, "error": "all candidate slots were taken; please retry"}


def _cancel(state, plan, patient, alerts, updates):
    upcoming = appt.list_appointments(patient_id=patient["id"], status="booked", upcoming_only=True)
    if plan.get("appointment_id"):
        return appt.cancel_appointment(plan["appointment_id"])
    spec = appt.normalize_specialty(plan.get("specialty"))
    matches = [a for a in upcoming if not spec or a["specialty"] == spec]
    if not matches:
        return {"ok": False, "error": "no matching upcoming appointment to cancel"}
    return appt.cancel_appointment(matches[0]["appointment_id"])


def _list_appts(state, plan, patient, alerts, updates):
    return {"ok": True, "appointments": appt.list_appointments(patient_id=patient["id"])}


def _search(state, plan, patient, alerts, updates):
    p = updates.get("patient") or patient
    ctx = None
    if p:
        conds = ", ".join((p.get("conditions") or []) + (plan.get("reported_conditions") or []))
        ctx = f"{p.get('age')}-year-old {p.get('gender') or ''} with {conds or 'no recorded conditions'}"
    elif plan.get("patient_age"):
        ctx = f"{plan['patient_age']}-year-old with {', '.join(plan.get('reported_conditions') or [])}"
    return search.medical_search(plan["search_topic"], plan.get("search_question"), ctx, run_id=state["run_id"])


EXECUTORS = {"identify_patient": _identify, "register_patient": _register, "get_medical_history": _history,
             "add_medical_record": _add_record, "find_doctor_slots": _find_slots, "book_appointment": _book,
             "cancel_appointment": _cancel, "list_appointments": _list_appts, "medical_search": _search}


# ---------------------------------------------------------------- respond

def _when(ts: str) -> str:
    return datetime.fromisoformat(ts).strftime("%a %d %b %Y, %I:%M %p")


def _appt_line(a: dict) -> str:
    return f"**{a['doctor']}** ({a['specialty']}) — {_when(a['start'])} at {a['location']} · appointment #{a['appointment_id']}"


def compose_status(state: AgentState) -> str:
    """Deterministic action report built directly from tool results, so outcomes are never hallucinated."""
    lines: list[str] = []
    tools = [r["tool"] for r in state["results"]]
    for r in state["results"]:
        res, tool = r["result"], r["tool"]
        if tool == "identify_patient" and r["ok"]:
            p = state["patient"]
            lines.append(f"👤 Patient: **{p['name']}** ({p.get('age')} y, {p.get('gender') or 'n/a'}) — "
                         f"record found ({res.get('match')} match)")
        elif tool == "register_patient" and r["ok"] and res.get("created"):
            lines.append(f"🆕 Registered new patient **{state['patient']['name']}**")
        elif tool == "find_doctor_slots":
            booked_ok = any(x["tool"] == "book_appointment" and x["ok"] for x in state["results"])
            if r["ok"] and not booked_ok:
                lines.append(f"🗓️ Available **{res['specialty']}** slots:")
                lines += [f"   - {sl['doctor']} — {sl['display']} ({sl['location']})" for sl in res["slots"]]
            elif not r["ok"]:
                lines.append(f"❌ Slot search failed: {res.get('error')}")
        elif tool == "book_appointment":
            if r["ok"] and res.get("already_booked"):
                lines.append(f"ℹ️ An upcoming appointment already exists, so I did not double-book: {_appt_line(res['appointment'])}")
            elif r["ok"]:
                lines.append(f"✅ Appointment booked: {_appt_line(res['appointment'])}")
            elif res.get("skipped"):
                lines.append("⏸️ Booking is on hold until the patient is identified.")
            else:
                lines.append(f"❌ Booking failed: {res.get('error')}")
        elif tool == "cancel_appointment":
            lines.append(f"✅ Cancelled: {_appt_line(res['appointment'])}" if r["ok"]
                         else f"❌ Cancellation failed: {res.get('error')}")
        elif tool == "list_appointments" and r["ok"]:
            appts = res["appointments"]
            lines.append(f"📅 Appointments ({len(appts)}):" if appts else "📅 No appointments on file.")
            lines += [f"   - [{a['status']}] {_appt_line(a)}" for a in appts]
        elif tool == "add_medical_record":
            if r["ok"]:
                st = res["structured"]
                dx = ", ".join(d.get("condition") or d.get("name") for d in st.get("diagnoses", [])) or "—"
                lines.append(f"📝 Added record #{res['record_id']} — diagnoses: {dx}; medications: "
                             f"{', '.join(st.get('medications', [])) or '—'}; allergies: {', '.join(st.get('allergies', [])) or '—'}")
            elif not res.get("skipped"):
                lines.append(f"❌ Could not update the record: {res.get('error')}")
        elif tool in ("get_medical_history", "medical_search") and not r["ok"] and not res.get("skipped"):
            lines.append(f"⚠️ {tool.replace('_', ' ').capitalize()} failed: {res.get('error')}")
    lines += [f"⚠️ {a}" for a in state.get("alerts", [])]
    if state.get("clarification"):
        lines.append(f"❓ **{state['clarification']}**")
    if not tools:
        return ""
    return "\n".join(lines) if lines else "Done."


def respond_node(state: AgentState, config: RunnableConfig) -> AgentState:
    t0 = time.perf_counter()
    _emit(config, {"node": "respond", "status": "running"})
    patient = state.get("patient")
    results = state["results"]
    if results:
        reply = compose_status(state)
    else:  # no tool needed (greeting / capability question)
        reply = (prompts.GENERAL_REPLY | get_llm(temperature=0.3)).invoke({"query": state["query"]}).content.strip()

    # Detailed sections are appended verbatim (not paraphrased) to keep them faithful to the tool output.
    sections = [reply] if reply else []
    for r in results:
        if r["tool"] == "get_medical_history" and r["ok"]:
            sections.append(f"### 📋 Medical history — {r['result']['patient']}\n{r['result']['markdown']}")
        if r["tool"] == "medical_search" and r["ok"]:
            src = "\n".join(f"{s['n']}. [{s['title']}]({s['url']}) — {s['source']}" for s in r["result"]["sources"])
            sections.append(f"### 📚 {r['result']['topic'].title()} — trusted sources\n{r['result']['summary']}\n\n"
                            f"**Sources**\n{src}")
    response = "\n\n".join(sections)

    success = all(r["ok"] for r in results) and not state.get("clarification")
    if patient:
        actions = "; ".join(f"{r['tool']}={'ok' if r['ok'] else 'failed'}" for r in results)
        booked = [r["result"]["appointment"] for r in results if r["tool"] == "book_appointment" and r["ok"]]
        note = f"Request: {state['query'][:200]} | Actions: {actions}"
        if booked:
            note += f" | Appointment: {booked[0]['doctor']} ({booked[0]['specialty']}) at {booked[0]['start']}"
        memory.remember(patient["id"], "interaction", note, state["run_id"])
    trace = _trace(state, "respond", {"success": success, "alerts": state.get("alerts")}, t0)
    latency = int((time.perf_counter() - state["started"]) * 1000)
    db.execute("INSERT OR REPLACE INTO agent_runs (id, query, patient_id, plan, trace, response, success, latency_ms, "
               "source, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (state["run_id"], state["query"], patient["id"] if patient else None,
                db.dumps({**state["plan"], "repairs": state.get("plan_repairs")}), db.dumps(trace), response,
                int(success), latency, state.get("source", "chat"), db.now()))
    _emit(config, {"node": "respond", "status": "done"})
    return {"response": response, "trace": trace}


# ---------------------------------------------------------------- graph

def _route(state: AgentState) -> str:
    return "execute" if state["step_idx"] < len(state["plan"]["steps"]) else "respond"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_context", load_context)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("respond", respond_node)
    g.set_entry_point("load_context")
    g.add_edge("load_context", "plan")
    g.add_conditional_edges("plan", _route, {"execute": "execute", "respond": "respond"})
    g.add_conditional_edges("execute", _route, {"execute": "execute", "respond": "respond"})
    g.add_edge("respond", END)
    return g.compile()


_GRAPH = None


def run_agent(query: str, chat_history: list[dict] | None = None, active_patient_id: int | None = None,
              pending_request: str | None = None, source: str = "chat",
              on_event: Callable[[dict], None] | None = None) -> AgentState:
    """Run one user turn. Returns the final state (response, plan, trace, results, pending_request)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    run_id = uuid.uuid4().hex[:12]
    token = current_run_id.set(run_id)
    try:
        state = _GRAPH.invoke(
            {"run_id": run_id, "query": query, "source": source, "chat_history": chat_history or [],
             "active_patient_id": active_patient_id, "pending_request": pending_request, "patient": None,
             "clarification": None, "trace": [], "started": time.perf_counter()},
            config={"recursion_limit": 40, "configurable": {"on_event": on_event}})
    finally:
        current_run_id.reset(token)
    return state
