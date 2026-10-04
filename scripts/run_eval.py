"""Run the LLMOps evaluation suite and store results (viewable in the Streamlit Evaluation page).

    python scripts/run_eval.py                      # all modules
    python scripts/run_eval.py planner search       # selected modules
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hca.evaluation import MODULES, Evaluator  # noqa: E402

if __name__ == "__main__":
    modules = tuple(sys.argv[1:]) or MODULES
    result = Evaluator(on_progress=lambda m: print("  ·", m, flush=True)).run(modules)
    print(json.dumps(result, indent=2, default=str))
