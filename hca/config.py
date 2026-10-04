"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("HCA_DATA_DIR", ROOT / "data"))
STATE_DIR = Path(os.getenv("HCA_STATE_DIR", ROOT / "state"))
DB_PATH = Path(os.getenv("HCA_DB_PATH", STATE_DIR / "healthcare.db"))
FAISS_DIR = Path(os.getenv("HCA_FAISS_DIR", STATE_DIR / "faiss"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("HCA_LLM_MODEL", "qwen2.5:7b")
# Independent, larger grader model for QAEvalChain (avoids a model grading its own output).
EVAL_MODEL = os.getenv("HCA_EVAL_MODEL", "gemma4:12b")
EMBED_MODEL = os.getenv("HCA_EMBED_MODEL", "nomic-embed-text")

# How many days of doctor slots to generate when seeding.
SCHEDULE_DAYS = int(os.getenv("HCA_SCHEDULE_DAYS", "21"))
HTTP_TIMEOUT = float(os.getenv("HCA_HTTP_TIMEOUT", "15"))

STATE_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)
