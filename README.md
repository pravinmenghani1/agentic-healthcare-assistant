# 🏥 Agentic Healthcare Assistant

Capstone: *Agentic Healthcare Assistant for Medical Task Automation* (Applied Generative AI Specialisation).

A virtual medical assistant that **books appointments, manages patient records, summarizes medical histories and
searches trusted medical sources**. It runs **entirely on local Ollama models**: no API keys, and patient data never
leaves the machine.

| Requirement | Where |
|---|---|
| Planner / goal decomposition | `hca/agent/graph.py` (LangGraph: `load_context → plan → execute⟲ → respond`), `hca/prompts.py::PLANNER` |
| Doctor Schedule API | `hca/tools/appointments.py` (slot discovery, atomic booking, cancel/reschedule) |
| EHR / Patient DB | `hca/db.py` (SQLite), `hca/tools/records.py`, `hca/ingest.py` (xlsx + clinical PDF parsing) |
| Disease search (Medline, WHO) | `hca/tools/medical_search.py` (MedlinePlus Web Service + WHO fact sheets → FAISS RAG) |
| Vector DB + long-term memory | `hca/memory.py` (FAISS `patient_memory` and `medical_kb`, memory events) |
| Prompt engineering / chaining | `hca/prompts.py` (planner → history summary / record extraction / RAG summary) |
| QAEvalChain evaluation | `hca/evaluation.py`, `scripts/run_eval.py` |
| Streamlit dashboard | `app.py`, `ui/` |

## Architecture

```
User ─► load_context ─► plan (LLM, structured output) ─► guard-rails ─► execute (one sub-goal per pass) ─► respond
          │ short-term chat       │ intent, entities, ordered steps    │ dependencies, date      │                     │
          │ long-term FAISS mem   │ (tool per sub-goal)                │ resolution, specialty   │ tools ▼             │ memory write +
          ▼                                                                                     │                     │ run trace log
   identify_patient · register_patient · get_medical_history · add_medical_record · find_doctor_slots ·
   book_appointment · cancel_appointment · list_appointments · medical_search
          │                    │                           │                              │
     SQLite EHR          FAISS patient memory      SQLite schedule (atomic)     MedlinePlus + WHO → FAISS KB → LLM
```

**Design choices**
- **Structured planning with guard-rails.** The LLM planner (qwen2.5:7b) returns a typed `Plan` (all fields
  required, so a small model must consider each one). Code then enforces tool dependencies, resolves relative dates
  (LLMs are poor at date arithmetic), maps conditions to specialties and drops invented values such as
  `patient_name="Father"`. If the LLM call fails, a keyword planner takes over.
- **Reports outcomes exactly.** Booking and cancellation results are written straight from tool outputs. The history
  summary and the cited RAG answer are appended verbatim. The LLM never rewrites the outcome of an action.
- **Safety for clinical use.** The agent asks for a name instead of guessing which patient someone means. It flags
  an age mismatch (stated 70 vs. 65 on file) and stores caregiver-reported conditions as *unconfirmed* memory
  instead of writing them into the EHR. ICD codes not found in the source records are removed. Booking is atomic
  (no double-booking), and an existing upcoming appointment for the same specialty is never booked twice.
- **Nearby doctors.** Slot search prefers doctors in the patient's city (Ramesh in Chennai gets the Chennai
  nephrologist).
- **Independent grader.** QAEvalChain grading uses `gemma4:12b`, a different model from the agent. Predictions are
  generated first, then graded in one batch so each model loads once on a 16 GB laptop. Positive and negative
  control answers check that the grader can be trusted.

## Data

`data/` holds the capstone dataset ([source repo](https://github.com/pravinmenghani1/Capstone-3-Agentic-Healthcare-Assistant/tree/main/dataset)):
`records.xlsx` (7 rows covering 5 patients) and 4 clinical PDFs. One of them, `sample_patient.pdf`, is a 12-page
chart with 2 encounters, labs and a CCD export. The PDFs are parsed into SOAP visit notes with ICD-10 codes and
medications. An LLM pass then fills in each patient profile. The dataset has no doctors, so `hca/seed.py` generates
a directory of 11 doctors (across 9 specialties) and 3 weeks of 30-minute slots.

> The `Datasets/` folder (sales_data.csv and BI papers) belongs to a different capstone and is not used.

## Setup

```bash
ollama pull qwen2.5:7b && ollama pull nomic-embed-text && ollama pull gemma4:12b   # agent, embeddings, grader
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/seed.py          # build SQLite EHR + schedules + FAISS memory (≈1 min)
./run.sh                                  # http://localhost:8501 (uses .venv, not a global streamlit)
```

Other entry points:
```bash
.venv/bin/python scripts/demo.py          # the problem statement's sample scenario in the terminal
.venv/bin/python scripts/run_eval.py      # full evaluation suite (history search planner booking)
.venv/bin/python -m pytest tests          # deterministic unit tests (no Ollama needed)
```
To use different models, see [Using other models](#using-other-models).

## Using other models

### Other Ollama models (no code changes)

Every model is chosen through an environment variable (defaults are in `hca/config.py`):

| Variable | Default | Used for |
|---|---|---|
| `HCA_LLM_MODEL` | `qwen2.5:7b` | agent: planner, history summaries, note structuring, search answers |
| `HCA_EVAL_MODEL` | `gemma4:12b` | QAEvalChain grader (evaluation only) |
| `HCA_EMBED_MODEL` | `nomic-embed-text` | embeddings for the FAISS patient memory and medical knowledge base |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (can be a remote GPU machine) |
| `HCA_STATE_DIR` | `./state` | where the SQLite DB and FAISS indexes are stored |

```bash
ollama pull llama3.1                                   # 1. pull the model
HCA_LLM_MODEL=llama3.1 ./run.sh                        # 2. start the dashboard with it
HCA_LLM_MODEL=llama3.1 .venv/bin/python scripts/demo.py
```

Or set it once for the whole terminal session:
```bash
export HCA_LLM_MODEL=llama3.1 HCA_EVAL_MODEL=gemma4:12b
./run.sh
```

The sidebar shows the active models and warns if one has not been pulled. Models are read at startup, so restart
the app after changing a variable.

**Changing the embedding model requires a rebuild.** A FAISS index only works with the model that created it, so
reseed after switching (about 1 minute). A separate `HCA_STATE_DIR` keeps the old index intact:
```bash
ollama pull mxbai-embed-large
HCA_EMBED_MODEL=mxbai-embed-large HCA_STATE_DIR=state-mxbai .venv/bin/python scripts/seed.py
HCA_EMBED_MODEL=mxbai-embed-large HCA_STATE_DIR=state-mxbai ./run.sh
```
Chat and grader models need no rebuild.

### Choosing a model

- **The agent model must produce reliable structured (JSON) output.** The planner, history summaries and note
  structuring require output matching a fixed format. Plan-checking code and a keyword fallback catch some mistakes,
  but models below about 7B parameters produce noticeably worse plans. 7B–14B instruction-tuned models are the
  practical range on a 16 GB machine.
- **Keep the grader different from the agent model**, and ideally stronger. A model grading its own answers is biased:
  with `qwen2.5:7b` grading itself, a correct summary was marked wrong. The grader controls in each evaluation run
  show whether a grader can be trusted (one known-correct and one known-wrong answer).
- **Memory:** agent and grader models are loaded one after the other during evaluation, not at the same time. The
  defaults use about 5 GB and 8 GB.
- **Context size** is fixed at 8,192 tokens in `hca/llm.py` (`num_ctx`). Increase it there for long patient histories,
  if the model supports it.
- **Tested combination:** `qwen2.5:7b` (agent) + `gemma4:12b` (grader) + `nomic-embed-text`. Other models use the same
  interface but have not been evaluated, so run the evaluation suite before relying on one.

### Comparing models

Use the evaluation suite to measure a model before switching to it:
```bash
HCA_LLM_MODEL=qwen2.5:7b .venv/bin/python scripts/run_eval.py
HCA_LLM_MODEL=llama3.1   .venv/bin/python scripts/run_eval.py
HCA_LLM_MODEL=mistral:7b .venv/bin/python scripts/run_eval.py planner search   # selected modules only
```
Each run records its agent and grader model. The **📊 Evaluation** page lists runs by model and charts every metric
across runs.

### Cloud providers (OpenAI, Anthropic, ...)

All model creation lives in `hca/llm.py`, so switching provider only changes that file. Any LangChain chat model
that supports `with_structured_output` works. Example for Claude:

```bash
uv pip install --python .venv/bin/python langchain-anthropic
export ANTHROPIC_API_KEY=...
```
```python
# hca/llm.py — replace the body of get_llm()
from langchain_anthropic import ChatAnthropic

@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0, model: str | None = None, num_ctx: int = 8192):
    return ChatAnthropic(model=model or config.LLM_MODEL, temperature=temperature, max_tokens=4096)
```
```bash
HCA_LLM_MODEL=claude-sonnet-5-5 HCA_EVAL_MODEL=claude-opus-5-5 ./run.sh
```
For OpenAI, use `langchain-openai` and `ChatOpenAI(model=..., temperature=...)` in the same place.

Notes for cloud models:
- **Embeddings can stay on Ollama** (`get_embeddings()` is separate). If you change them to a cloud embedding model,
  reseed as described above.
- The sidebar's Ollama check reports cloud model names as "missing". This is harmless and can be ignored.
- **Privacy:** with a cloud model, patient records, notes and queries are sent to that provider's API. The default
  all-local setup keeps them on your machine.

## Dashboard pages

| Page | What it shows |
|---|---|
| 🤖 Assistant | Chat with a live progress view of each plan step and an expandable plan breakdown and trace for every answer; optional active-patient context; quick examples |
| 🧪 Scenario lab | Runs single- or multi-turn scenarios (including the sample one), compares the tools used with the expected tools, optional rollback |
| 🧑 Patients | Patient overview, FAISS semantic patient search, registration, profile editing, records, free-text note entry (structured by the LLM), structured SOAP form, PDF upload, AI history summary, appointments, memory |
| 🩺 Doctors | Daily schedule, utilisation, upcoming patients with their EHR context, 14-day capacity chart |
| 📅 Appointments | Real-time tracker (refreshes every 5 s): KPIs, table, timeline, recent activity, manual booking |
| 📚 Medical info | Latest retrieved summaries with sources, ad-hoc MedlinePlus/WHO search, list of indexed sources |
| 📊 Evaluation | Run the suite and view QAEvalChain accuracy, groundedness, citation rate, planner F1 and entity accuracy, booking success, grader calibration, trends across runs, per-tool success and latency, agent run latency |
| 🧠 Memory & logs | Each agent run's plan and trace timeline, node details, tool calls and memory written; filterable tool-usage log; memory events; FAISS explorer |

## Evaluation suite (`hca/evaluation.py`)

| Module | Method | Metric |
|---|---|---|
| history | QAEvalChain vs. gold answers written from the dataset (5 patients) | accuracy |
| search | QAEvalChain vs. reference + ContextQAEvalChain vs. retrieved sources + citation regex (4 topics) | accuracy, groundedness, citation rate |
| planner | Tools chosen vs. required/optional tools; extracted entities (8 queries) | precision / recall / F1, exact plan, entity accuracy |
| booking | End-to-end agent runs; bookings rolled back afterwards (2 cases) | success rate, latency |
| grader | Positive and negative control answers | calibration |
| tools | Aggregated from `tool_logs` | success rate and latency per tool |

## Project layout

```
app.py                 Streamlit entry point (st.navigation)
hca/
  config.py            models, paths (env-overridable)
  db.py                SQLite schema + helpers
  ingest.py            records.xlsx + clinical PDF → structured encounters
  llm.py               ChatOllama / nomic embeddings
  memory.py            FAISS stores, long-term memory (remember / recall / forget_run)
  prompts.py           prompt templates for every sub-task
  tracking.py          @logged_tool — tool-usage logging
  seed.py              builds the local state
  tools/               appointments.py · records.py · medical_search.py
  agent/graph.py       LangGraph agent
  evaluation.py        QAEvalChain suite
ui/                    one module per dashboard page
scripts/               seed.py · demo.py · run_eval.py
tests/                 unit tests
data/                  dataset (xlsx + PDFs)
state/                 generated: healthcare.db + FAISS indexes (delete to reset)
```

*Educational project. Not a medical device: summaries and search results support staff and do not replace
clinical judgement.*
