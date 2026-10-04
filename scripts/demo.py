"""Run the problem statement's sample scenario end-to-end from the command line.

    python scripts/demo.py                 # sample scenario (2 turns: request, then the father's name)
    python scripts/demo.py "your query"    # any single query
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hca.agent.graph import run_agent  # noqa: E402

SAMPLE = ["My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
          "Also, can you summarize latest treatment methods?",
          "His name is Ramesh Kulkarni."]


def show(event: dict) -> None:
    if event["status"] == "done":
        label = event["node"] + (f":{event['step']['tool']}" if "step" in event else "")
        ok = "" if "ok" not in event else (" ✔" if event["ok"] else " ✘")
        print(f"  · {label}{ok}")
        if event["node"] == "plan":
            plan = event["detail"]["plan"]
            print("    plan:", [s["tool"] for s in plan["steps"]], "| repairs:", event["detail"]["repairs"])
            print("    entities:", json.dumps({k: plan[k] for k in ("patient_name", "patient_relation", "patient_age",
                                                                    "reported_conditions", "specialty", "search_topic")}))


if __name__ == "__main__":
    turns = sys.argv[1:] or SAMPLE
    history, pending = [], None
    for q in turns:
        print(f"\n>>> USER: {q}")
        state = run_agent(q, chat_history=history, pending_request=pending, on_event=show)
        pending = state.get("pending_request")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": state["response"]}]
        print(f"\n<<< ASSISTANT:\n{state['response']}\n" + "-" * 80)
