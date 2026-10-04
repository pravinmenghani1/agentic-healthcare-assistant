"""LLMOps evaluation.

Modules
* history  – QAEvalChain grades LLM history summaries against gold answers written from the dataset.
* search   – QAEvalChain (correctness vs. reference) + ContextQAEvalChain (groundedness in the retrieved sources)
             + citation check for the RAG medical search.
* planner  – tool-selection precision / recall / F1 and entity accuracy of the planner.
* booking  – end-to-end agent bookings (success rate); bookings made by the eval are rolled back afterwards.
* tools    – success rate and latency per tool from the tool-usage logs.
"""
import re
import time
from datetime import date
from typing import Callable

from langchain_classic.evaluation.qa import ContextQAEvalChain, QAEvalChain

from . import config, db, memory
from .agent.graph import Plan, _calendar, _repair_plan, run_agent
from .llm import get_llm
from .prompts import PLANNER
from .tools import medical_search, records
from .tracking import current_run_id

HISTORY_CASES = [
    ("Rahul Negi", "Summarize Rahul Negi's medical history.",
     "Rahul Negi is a 31-year-old male who is fit and healthy, with no recorded diagnoses, medications or clinical concerns."),
    ("Rebeca Nagle", "What are Rebeca Nagle's diagnoses and treatments?",
     "Rebeca Nagle (36, female) was diagnosed with costochondritis (M94.0) in September 2022 and advised ibuprofen 600 mg "
     "three times daily for 5-7 days with stretching and rest. She has a history of PCOS and migraines (magnesium and "
     "vitamin B2) and uses a combined oral contraceptive; a well-woman exam with Pap smear and screening labs was done "
     "on 2022-09-07."),
    ("Ramesh Kulkarni", "What are Ramesh Kulkarni's diagnoses, medications and follow-up plan?",
     "Ramesh Kulkarni (65, male) has essential hypertension (I10) treated with Telmisartan 40 mg once daily plus lifestyle "
     "modifications; routine labs were ordered and he should return 6 months after the 2024-02-28 visit."),
    ("Anjali Mehra", "What was Anjali Mehra diagnosed with and how is it being treated?",
     "Anjali Mehra (33, female) was diagnosed with an upper respiratory infection (J06.9) on 2024-03-15 after 5 days of dry "
     "cough and mild fever; treatment is symptomatic with antihistamines, fluids and rest, with follow-up in 5 days if "
     "no improvement."),
    ("David Thompson", "What are David Thompson's diagnosis and current treatment plan?",
     "David Thompson (51, male) has type 2 diabetes mellitus (E11.9); metformin was increased to 1000 mg twice daily, "
     "HbA1c and a lipid profile were ordered, a nutritionist follow-up was scheduled, and he should return in 3 months."),
]

SEARCH_CASES = [
    ("chronic kidney disease", "What are the treatment options for chronic kidney disease?",
     "Treatment aims to slow progression: controlling blood pressure and blood sugar, lowering cholesterol, diet changes "
     "such as less salt, and healthy lifestyle; kidney failure is treated with dialysis or a kidney transplant."),
    ("type 2 diabetes", "How is type 2 diabetes managed?",
     "Healthy eating, regular physical activity and weight control, monitoring blood glucose, and medicines such as "
     "metformin or insulin when needed."),
    ("high blood pressure", "How is high blood pressure treated?",
     "Lifestyle changes (heart-healthy diet with less salt, exercise, limiting alcohol, not smoking, healthy weight) and "
     "blood pressure medicines."),
    ("migraine", "What are the symptoms of migraine?",
     "A throbbing or pulsing headache, often on one side, with nausea or vomiting and sensitivity to light and sound; "
     "some people have an aura beforehand."),
]

# (query, required tools, optional tools, expected entities)
PLANNER_CASES = [
    ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize "
     "latest treatment methods?", {"identify_patient", "find_doctor_slots", "book_appointment", "medical_search"},
     {"get_medical_history"}, {"specialty": "Nephrologist", "patient_relation": "father", "patient_age": 70}),
    ("Book a cardiologist for Ramesh Kulkarni next Monday morning",
     {"identify_patient", "find_doctor_slots", "book_appointment"}, {"get_medical_history"},
     {"patient_name": "Ramesh Kulkarni", "specialty": "Cardiologist", "preferred_time": "morning"}),
    ("Show David Thompson's medical history", {"identify_patient", "get_medical_history"}, set(),
     {"patient_name": "David Thompson"}),
    ("Add to Anjali Mehra's record: cough resolved, allergic to penicillin",
     {"identify_patient", "add_medical_record"}, set(), {"patient_name": "Anjali Mehra"}),
    ("What are the symptoms of migraine?", {"medical_search"}, set(), {"search_topic": "migraine"}),
    ("Cancel Anjali Mehra's appointment", {"identify_patient", "cancel_appointment"}, {"list_appointments"},
     {"patient_name": "Anjali Mehra"}),
    ("Which appointments does David Thompson have coming up?", {"identify_patient", "list_appointments"}, set(),
     {"patient_name": "David Thompson"}),
    ("Rebeca Nagle has chest pain again. Book an orthopedist and tell me how costochondritis is treated.",
     {"identify_patient", "find_doctor_slots", "book_appointment", "medical_search"}, {"get_medical_history"},
     {"patient_name": "Rebeca Nagle", "specialty": "Orthopedist"}),
]

BOOKING_CASES = [
    ("Book an endocrinologist for David Thompson next Wednesday afternoon", "David Thompson", "Endocrinologist"),
    ("Please schedule a gynecologist appointment for Rebeca Nagle", "Rebeca Nagle", "Gynecologist"),
]

MODULES = ("history", "search", "planner", "booking")


def _grade(output: dict) -> str:
    text = (output.get("results") or output.get("text") or "").upper()
    m = re.search(r"GRADE:\s*(CORRECT|INCORRECT)", text)
    return m.group(1) if m else ("INCORRECT" if "INCORRECT" in text else "CORRECT" if "CORRECT" in text else "UNKNOWN")


def _norm(v) -> str:
    return str(v).strip().lower() if v is not None else ""


class Evaluator:
    """Generation and grading are separate phases: all predictions are produced with the agent model first, then
    graded in one batch by the (independent) grader model, so a 16 GB machine loads each model only once."""

    def __init__(self, on_progress: Callable[[str], None] | None = None):
        self.grader = get_llm(model=config.EVAL_MODEL)
        self.qa = QAEvalChain.from_llm(self.grader)
        self.ctx_qa = ContextQAEvalChain.from_llm(self.grader)
        self.rows: list[dict] = []
        self.to_grade: list[tuple[dict, str, dict, dict]] = []   # (row, kind 'qa'|'context', example, prediction)
        self.log = on_progress or (lambda msg: None)

    def _row(self, module, case_id, query, expected, predicted, grade, score, **details) -> dict:
        row = {"module": module, "case_id": case_id, "query": query, "expected": expected,
               "predicted": predicted, "grade": grade, "score": score, "details": details}
        self.rows.append(row)
        return row

    # ------------------------------------------------------------ modules (generation phase)
    def eval_history(self) -> None:
        for name, question, gold in HISTORY_CASES:
            self.log(f"history: {name}")
            p = db.query_one("SELECT id FROM patients WHERE name=?", (name,))
            t0 = time.perf_counter()
            pred = records.summarize_history(p["id"])["markdown"] if p else "patient not found"
            row = self._row("history", name, question, gold, pred, None, None,
                            latency_ms=int((time.perf_counter() - t0) * 1000))
            self.to_grade.append((row, "qa", {"query": question, "answer": gold}, {"result": pred}))

    def eval_search(self) -> None:
        for topic, question, gold in SEARCH_CASES:
            self.log(f"search: {topic}")
            t0 = time.perf_counter()
            res = medical_search.medical_search(topic, question)
            pred = res.get("summary", res.get("error", ""))
            row = self._row("search", topic, question, gold, pred, None, None,
                            cited=bool(re.search(r"\[\d+\]", pred)), grounded=None if res.get("ok") else "INCORRECT",
                            sources=[s["url"] for s in res.get("sources", [])],
                            latency_ms=int((time.perf_counter() - t0) * 1000))
            self.to_grade.append((row, "qa", {"query": question, "answer": gold}, {"result": pred}))
            if res.get("ok"):
                self.to_grade.append((row, "context", {"query": question, "context": res["context"]}, {"result": pred}))

    def calibrate_grader(self) -> None:
        """Positive and negative controls: a trustworthy grader must pass the first and fail the second."""
        _, question, gold = HISTORY_CASES[4]
        for case_id, pred, expect in (
                ("positive-control", gold, "CORRECT"),
                ("negative-control", "David has type 1 diabetes treated with insulin glargine only.", "INCORRECT")):
            row = self._row("grader", case_id, question, expect, pred, None, None)
            self.to_grade.append((row, "qa", {"query": question, "answer": gold}, {"result": pred}))

    # ------------------------------------------------------------ grading phase
    def grade_all(self) -> None:
        for kind, chain in (("qa", self.qa), ("context", self.ctx_qa)):
            items = [t for t in self.to_grade if t[1] == kind]
            if not items:
                continue
            self.log(f"grading {len(items)} {kind} items with {config.EVAL_MODEL}")
            outputs = chain.evaluate([t[2] for t in items], [t[3] for t in items])
            for (row, _, _, _), out in zip(items, outputs):
                grade = _grade(out)
                if kind == "context":
                    row["details"]["grounded"] = grade
                elif row["module"] == "grader":
                    row["grade"] = "CORRECT" if grade == row["expected"] else "INCORRECT"
                    row["score"] = float(row["grade"] == "CORRECT")
                    row["details"]["grader_output"] = grade
                else:
                    row["grade"], row["score"] = grade, float(grade == "CORRECT")

    def graded_metrics(self) -> dict:
        def rate(module, pred) -> float | None:
            rows = [r for r in self.rows if r["module"] == module]
            return sum(map(pred, rows)) / len(rows) if rows else None
        out = {
            "history_accuracy": rate("history", lambda r: r["grade"] == "CORRECT"),
            "search_accuracy": rate("search", lambda r: r["grade"] == "CORRECT"),
            "search_groundedness": rate("search", lambda r: r["details"].get("grounded") == "CORRECT"),
            "search_citation_rate": rate("search", lambda r: r["details"]["cited"]),
            "grader_calibration": rate("grader", lambda r: r["grade"] == "CORRECT"),
        }
        return {k: v for k, v in out.items() if v is not None}

    def eval_planner(self) -> dict:
        tp = fp = fn = exact = ent_ok = ent_total = 0
        planner = PLANNER | get_llm().with_structured_output(Plan)
        for i, (query, required, optional, entities) in enumerate(PLANNER_CASES):
            self.log(f"planner: case {i + 1}")
            t0 = time.perf_counter()
            try:
                plan = planner.invoke({"today": date.today().isoformat(), "calendar": _calendar(), "context": "",
                                       "query": query})
                plan, repairs = _repair_plan(plan, query)
                d = plan.model_dump()
            except Exception as exc:  # noqa: BLE001
                d, repairs = {"steps": [], "error": str(exc)}, []
            latency = int((time.perf_counter() - t0) * 1000)
            got = {s["tool"] for s in d["steps"]}
            c_tp, c_fn = len(got & required), len(required - got)
            c_fp = len(got - required - optional)
            tp, fp, fn = tp + c_tp, fp + c_fp, fn + c_fn
            ok = c_fn == 0 and c_fp == 0
            exact += ok
            ent_hits = {k: _norm(d.get(k)) == _norm(v) or (_norm(v) in _norm(d.get(k)) and _norm(v) != "")
                        for k, v in entities.items()}
            ent_ok, ent_total = ent_ok + sum(ent_hits.values()), ent_total + len(ent_hits)
            self._row("planner", f"P{i + 1}", query, sorted(required), sorted(got), "CORRECT" if ok else "INCORRECT",
                      c_tp / max(1, c_tp + c_fp + c_fn), entities=ent_hits, repairs=repairs, latency_ms=latency,
                      plan={k: v for k, v in d.items() if k != "steps" and v})
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        return {"planner_tool_precision": precision, "planner_tool_recall": recall,
                "planner_tool_f1": 2 * precision * recall / max(1e-9, precision + recall),
                "planner_exact_match": exact / len(PLANNER_CASES), "planner_entity_accuracy": ent_ok / max(1, ent_total)}

    def eval_booking(self) -> dict:
        success, latencies = 0, []
        for query, name, specialty in BOOKING_CASES:
            self.log(f"booking: {name}")
            t0 = time.perf_counter()
            state = run_agent(query, source="eval")
            latencies.append(int((time.perf_counter() - t0) * 1000))
            booked = [r["result"] for r in state["results"] if r["tool"] == "book_appointment" and r["ok"]]
            appt = booked[0]["appointment"] if booked else None
            ok = bool(appt and appt["patient"] == name and appt["specialty"] == specialty)
            success += ok
            self._row("booking", name, query, f"{name} / {specialty}",
                      f"{appt['patient']} / {appt['specialty']} @ {appt['start']}" if appt else "no booking",
                      "CORRECT" if ok else "INCORRECT", float(ok), run_id=state["run_id"], latency_ms=latencies[-1])
            self._rollback(state["run_id"], booked)
        return {"booking_success_rate": success / len(BOOKING_CASES),
                "agent_avg_latency_s": sum(latencies) / len(latencies) / 1000}

    @staticmethod
    def _rollback(run_id: str, booked: list[dict]) -> None:
        """Eval bookings must not consume real capacity: delete them and reopen slots."""
        for b in booked:
            if b.get("already_booked"):
                continue
            a = db.query_one("SELECT * FROM appointments WHERE id=?", (b["appointment"]["appointment_id"],))
            if a:
                db.execute("UPDATE slots SET status='open' WHERE id=?", (a["slot_id"],))
                db.execute("DELETE FROM appointments WHERE id=?", (a["id"],))
        memory.forget_run(run_id)

    # ------------------------------------------------------------ orchestration
    def run(self, modules=MODULES) -> dict:
        token = current_run_id.set("eval")
        metrics: dict = {"modules": list(modules), "agent_model": config.LLM_MODEL, "grader_model": config.EVAL_MODEL}
        try:
            for m in modules:
                metrics.update(getattr(self, f"eval_{m}")() or {})
            if self.to_grade:
                self.calibrate_grader()
                self.grade_all()
        finally:
            current_run_id.reset(token)
        metrics.update(self.graded_metrics())
        metrics.update(tool_metrics())
        run_id = db.execute("INSERT INTO eval_runs (metrics, created_at) VALUES (?,?)", (db.dumps(metrics), db.now()))
        with db.session() as conn:
            conn.executemany(
                "INSERT INTO eval_results (eval_run_id, module, case_id, query, expected, predicted, grade, score, details) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [(run_id, r["module"], r["case_id"], r["query"], db.dumps(r["expected"]) if not isinstance(r["expected"], str)
                  else r["expected"], db.dumps(r["predicted"]) if not isinstance(r["predicted"], str) else r["predicted"],
                  r["grade"], r["score"], db.dumps(r["details"])) for r in self.rows])
        self.log("done")
        return {"eval_run_id": run_id, **metrics}


def tool_metrics() -> dict:
    """All-time tool reliability from the usage logs (agent + UI + eval)."""
    rows = db.query("SELECT tool, COUNT(*) n, AVG(success) rate, AVG(latency_ms) lat FROM tool_logs GROUP BY tool")
    total = sum(r["n"] for r in rows)
    return {"tool_calls": total,
            "tool_success_rate": sum(r["rate"] * r["n"] for r in rows) / total if total else None,
            "per_tool": {r["tool"]: {"calls": r["n"], "success_rate": round(r["rate"], 3), "avg_ms": int(r["lat"])}
                         for r in rows}}
