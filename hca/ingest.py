"""Parse the raw dataset (records.xlsx + clinical PDFs) into structured EHR records.

PDF parsing is deterministic (regex over SOAP sections) so seeding works without an LLM;
the LLM is only used afterwards to enrich patient profiles.
"""
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

SECTION_NAMES = ["Subjective Notes", "Objective Notes", "Assessment Notes", "Plan Notes"]
# Text that marks the end of the plan section / an encounter's footer.
PLAN_END = re.compile(r"\n\s*(Encounter\s*\n|Patient MRN:|Signed electronically|History and Physical Note|---PAGE)", re.I)
DATE_PATTERNS = [
    re.compile(r"Visit Date:\s*(\d{1,2}/\d{1,2}/\d{4})"),
    re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}\s*[AP]M", re.M),
]
DIAG_ICD10 = re.compile(r"([A-Za-z][^\[\]:]*?)\s*\[ICD-10:\s*([A-Z0-9.]+)\]")
DIAG_PAREN = re.compile(r"Diagnosis:\s*(.+?)\s*\(([A-Z]\d[\w.]*)\)", re.I)
MEDS_BLOCK = re.compile(r"MEDICATIONS:\s*(.+?)(?=\n\s*\n|\nEncounter|$)", re.S)
# Split a run-on medication list before each "<DrugName> <dose> MG" token.
MED_SPLIT = re.compile(r"\s+(?=(?!Oral\b|Capsule\b|Tablet\b)[A-Z][A-Za-z\-]+(?:\s[A-Z][\w\-]*)?\s[\d.]+(?:-[\d.]+)?\s?MG)")
PATIENT_NAME = re.compile(r"^\s*Patient(?: Name)?:?[ \t]+([A-Z][a-z]+(?:[ \t][A-Z][a-zA-Z'-]+)+)[ \t]*$", re.M)
PMH = re.compile(r"PMH:\s*([^.]+?)\s+(?:FH|PSH|FMH):", re.S)


@dataclass
class Encounter:
    visit_date: str | None
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    diagnoses: list[dict] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class ParsedDocument:
    source: str
    patient_name: str | None
    dob: str | None
    gender: str | None
    phone: str | None
    address: str | None
    encounters: list[Encounter]
    supplementary: list[dict]          # [{pages: "4-12", text: "..."}] non-SOAP pages (labs, CCD, PMH)
    past_history: list[str]


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def _iso_date(us_date: str | None) -> str | None:
    if not us_date:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(us_date, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _field(text: str, label: str) -> str | None:
    m = re.search(rf"^\s*{label}:?\s+(.+)$", text, re.M | re.I)
    return m.group(1).strip() if m else None


def extract_diagnoses(text: str) -> list[dict]:
    text = " ".join(text.split())
    found = [{"name": n.strip(" -,"), "code": c} for n, c in DIAG_ICD10.findall(text)]
    found += [{"name": n.strip(), "code": c} for n, c in DIAG_PAREN.findall(text)]
    clean = []
    for d in found:
        d["name"] = re.sub(r"^(DIAGNOSIS:?\s*)", "", d["name"], flags=re.I).strip()
        if d["name"] and d not in clean:
            clean.append(d)
    return clean


def extract_medications(text: str) -> list[str]:
    meds = []
    for block in MEDS_BLOCK.findall(text):
        block = " ".join(block.split())
        meds += [m.strip() for m in MED_SPLIT.split(block) if m.strip()]
    return meds


def _section(block: str, name: str) -> str:
    idx = SECTION_NAMES.index(name)
    start = block.find(name)
    if start < 0:
        return ""
    start += len(name)
    end = len(block)
    for nxt in SECTION_NAMES[idx + 1:]:
        pos = block.find(nxt, start)
        if pos >= 0:
            end = pos
            break
    text = block[start:end]
    if name == "Plan Notes":
        m = PLAN_END.search(text)
        # The plan can continue on the next page after a footer; keep text up to "Encounter"/signature.
        if m and "Patient MRN" in m.group(0):
            rest = text[m.end():]
            rest = re.sub(r"^.*?---PAGE \d+---", "", rest, count=1, flags=re.S)
            m2 = PLAN_END.search(rest)
            text = text[: m.start()] + "\n" + (rest[: m2.start()] if m2 else rest)
        elif m:
            text = text[: m.start()]
    return _clean(re.sub(r"Encounter ID:.*|Page \d+ of \d+|---PAGE \d+---", "", text)).lstrip(": \n")


def parse_pdf(source: str | Path | bytes, name: str | None = None) -> ParsedDocument:
    if isinstance(source, (bytes, bytearray)):
        reader, src_name = PdfReader(io.BytesIO(source)), name or "upload.pdf"
    else:
        reader, src_name = PdfReader(str(source)), name or Path(source).name
    pages = [p.extract_text() or "" for p in reader.pages]
    full = "".join(f"\n---PAGE {i + 1}---\n{t}" for i, t in enumerate(pages))

    m = PATIENT_NAME.search(full)
    patient_name = m.group(1).strip() if m else None
    if not patient_name:
        first_lines = [l.strip() for l in pages[0].splitlines() if l.strip()] if pages else []
        if len(first_lines) > 1 and re.match(r"\d{1,2}/\d{1,2}/\d{4}", first_lines[1]):
            patient_name = first_lines[0]
    dob = _iso_date(_field(full, "DOB"))
    gender = _field(full, "Gender")
    phone = _field(full, "Phone")
    address = _field(full, "Address")

    # Each encounter is a "Subjective Notes ... Plan Notes" block; its date is the closest header above it.
    encounters, soap_pages = [], set()
    starts = [m.start() for m in re.finditer(r"Subjective Notes", full)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(full)
        block = full[start:end]
        # An encounter ends at the page break following its signature line.
        signed = re.search(r"Signed electronically.*?(?=---PAGE)", block, re.S)
        if signed:
            block = block[: signed.end()]
        header = full[max(0, start - 600):start]
        visit_date = None
        for pat in DATE_PATTERNS:
            hits = pat.findall(header)
            if hits:
                visit_date = _iso_date(hits[-1])
                break
        enc = Encounter(
            visit_date=visit_date,
            subjective=_section(block, "Subjective Notes"),
            objective=_section(block, "Objective Notes"),
            assessment=_section(block, "Assessment Notes"),
            plan=_section(block, "Plan Notes"),
            raw_text=_clean(re.sub(r"---PAGE \d+---", "", header[-300:] + block)),
        )
        enc.diagnoses = extract_diagnoses(enc.assessment)
        enc.medications = extract_medications(block)
        encounters.append(enc)
        first_page = full[:start].count("---PAGE")
        last_page = full[:start + len(block)].count("---PAGE")
        soap_pages.update(range(first_page, last_page + 1))

    supplementary_pages = [i + 1 for i, t in enumerate(pages) if (i + 1) not in soap_pages and t.strip()]
    supplementary = []
    if supplementary_pages:
        text = "\n".join(pages[p - 1] for p in supplementary_pages)
        supplementary.append({"pages": f"{supplementary_pages[0]}-{supplementary_pages[-1]}", "text": _clean(text)})

    past_history = []
    for m in PMH.findall(full):
        past_history += [p.strip() for p in re.split(r",|;", m) if p.strip()]

    return ParsedDocument(src_name, patient_name, dob, gender, phone, address, encounters, supplementary, past_history)


def load_records_xlsx(path: str | Path) -> list[dict]:
    """Load records.xlsx and collapse duplicate rows (same name + phone) into one patient."""
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    patients: dict[tuple, dict] = {}
    for row in df.to_dict("records"):
        row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        key = (str(row.get("name", "")).strip().lower(), str(row.get("phone_number", "")).strip())
        entry = patients.setdefault(key, {
            "name": str(row["name"]).strip(),
            "age": int(row["age"]) if row.get("age") is not None else None,
            "gender": row.get("gender"),
            "phone": str(row["phone_number"]).strip() if row.get("phone_number") else None,
            "email": row.get("email"),
            "address": row.get("address"),
            "summary": None,
        })
        if row.get("summary"):
            entry["summary"] = (entry["summary"] + " " if entry["summary"] else "") + str(row["summary"]).strip()
    return list(patients.values())
