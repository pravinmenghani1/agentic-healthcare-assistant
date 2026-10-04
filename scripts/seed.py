"""Initialise the database and vector memory from the dataset.

    python scripts/seed.py            # with LLM profile enrichment (needs Ollama)
    python scripts/seed.py --no-llm   # deterministic parsing only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hca import seed  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip LLM profile enrichment")
    seed.run(use_llm=not parser.parse_args().no_llm)
