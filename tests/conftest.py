import os
import sys
import tempfile
from pathlib import Path

import pytest

# Isolated state dir *before* hca.config is imported, so tests never touch the real database.
os.environ["HCA_STATE_DIR"] = tempfile.mkdtemp(prefix="hca-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hca import config, db, memory, seed  # noqa: E402


@pytest.fixture()
def fresh_db(monkeypatch):
    """Empty schema + doctors/slots; vector indexing is stubbed out (no Ollama needed)."""
    monkeypatch.setattr(memory, "index_patient", lambda pid: 0)
    monkeypatch.setattr(memory, "remember", lambda *a, **k: None)
    config.DB_PATH.unlink(missing_ok=True)
    db.init_db()
    seed.seed_doctors(days=10, busy_ratio=0.0)
    yield
    config.DB_PATH.unlink(missing_ok=True)
