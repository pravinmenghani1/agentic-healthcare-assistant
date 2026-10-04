"""Disease information search over trusted sources (MedlinePlus Web Service + WHO fact sheets) with a RAG
pipeline: fetched pages are chunked into a FAISS knowledge base, the most relevant chunks are retrieved for the
question and summarized by the LLM with citations. Raw fetches are cached so the system degrades gracefully offline.
"""
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .. import config, db, memory, prompts
from ..llm import get_llm
from ..tracking import logged_tool

HEADERS = {"User-Agent": "Mozilla/5.0 (AgenticHealthcareAssistant; educational capstone)"}
MEDLINE_URL = "https://wsearch.nlm.nih.gov/ws/query"
WHO_INDEX = "https://www.who.int/news-room/fact-sheets"
CACHE_TTL = timedelta(days=7)
STOPWORDS = {"the", "and", "of", "for", "in", "a", "an", "to", "disease", "diseases", "latest", "treatment", "treatments",
             "methods", "options", "what", "is", "are", "chronic", "type", "about", "how", "with"}


def _fetch(url: str, params: dict | None = None) -> str | None:
    """GET with a SQLite cache; serves stale cache when the network fails."""
    key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "")
    cached = db.query_one("SELECT * FROM web_cache WHERE url=?", (key,))
    if cached and datetime.fromisoformat(cached["fetched_at"]) > datetime.now() - CACHE_TTL:
        return cached["content"]
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return cached["content"] if cached else None
    db.execute("INSERT OR REPLACE INTO web_cache (url, content, fetched_at) VALUES (?,?,?)", (key, resp.text, db.now()))
    return resp.text


def _strip_html(text: str) -> str:
    return " ".join(BeautifulSoup(html.unescape(text), "html.parser").get_text(" ").split())


def search_medlineplus(term: str, k: int = 3) -> list[dict]:
    xml = _fetch(MEDLINE_URL, {"db": "healthTopics", "term": term, "retmax": k})
    if not xml:
        return []
    out = []
    for doc in ET.fromstring(xml).iter("document"):
        fields = {c.get("name"): c.text or "" for c in doc.findall("content")}
        text = _strip_html(fields.get("FullSummary", ""))
        if text:
            out.append({"source": "MedlinePlus (NIH/NLM)", "title": _strip_html(fields.get("title", term)),
                        "url": doc.get("url"), "text": text})
    return out[:k]


def _who_index() -> dict[str, str]:
    page = _fetch(WHO_INDEX)
    if not page:
        return {}
    slugs = set(re.findall(r"/news-room/fact-sheets/detail/([a-z0-9\-()]+)", page))
    return {slug: f"{WHO_INDEX}/detail/{slug}" for slug in slugs}


def search_who(term: str) -> list[dict]:
    """Pick the WHO fact sheet whose slug best overlaps the topic words."""
    words = {w for w in re.findall(r"[a-z0-9]+", term.lower()) if w not in STOPWORDS and len(w) > 2}
    if not words:
        return []
    best, best_score = None, 0.0
    for slug, url in _who_index().items():
        slug_words = set(re.findall(r"[a-z0-9]+", slug))
        overlap = len(words & slug_words)
        score = overlap / len(words) - 0.01 * len(slug_words - words)
        if overlap and score > best_score:
            best, best_score = url, score
    if not best or best_score < 0.5:
        return []
    page = _fetch(best)
    if not page:
        return []
    soup = BeautifulSoup(page, "html.parser")
    article = soup.select_one("article") or soup.select_one("main") or soup.body
    title = soup.title.get_text(strip=True) if soup.title else best
    text = " ".join(article.get_text(" ").split()) if article else ""
    return [{"source": "World Health Organization", "title": title, "url": best, "text": text[:15000]}] if text else []


def _index_sources(sources: list[dict]) -> None:
    kb = memory.knowledge_store()
    known = {d.metadata.get("url") for d in kb.all_documents()}
    docs = []
    for s in sources:
        if s["url"] in known:
            continue
        for i, chunk in enumerate(memory.splitter.split_text(s["text"])):
            docs.append(Document(chunk, metadata={"url": s["url"], "title": s["title"], "source": s["source"],
                                                  "chunk": i, "fetched_at": db.now()}))
    kb.add(docs)


@logged_tool("medical_search")
def medical_search(topic: str, question: str | None = None, patient_context: str | None = None,
                   k: int = 6, run_id: str | None = None) -> dict:
    question = question or f"What is {topic} and what are the current treatment options?"
    sources = search_medlineplus(topic) + search_who(topic)
    offline = not sources
    if sources:
        _index_sources(sources)
    urls = {s["url"] for s in sources}
    # Retrieve from the knowledge base; prefer chunks from this search, else anything relevant cached earlier.
    hits = memory.knowledge_store().search(f"{topic}: {question}", k=k,
                                           filter=(lambda m: m.get("url") in urls) if urls else None)
    if not hits:
        return {"ok": False, "error": f"No trusted sources found for '{topic}' (offline={offline})"}

    numbered, refs = [], {}
    for doc, _ in hits:
        url = doc.metadata["url"]
        if url not in refs:
            refs[url] = {"n": len(refs) + 1, "title": doc.metadata["title"], "url": url, "source": doc.metadata["source"]}
        numbered.append(f"[{refs[url]['n']}] ({doc.metadata['source']}: {doc.metadata['title']})\n{doc.page_content}")
    context = "\n\n".join(numbered)
    summary = (prompts.SEARCH_SUMMARY | get_llm()).invoke({
        "question": question, "sources": context,
        "patient_context": f"Tailor relevance notes to this patient where appropriate: {patient_context}" if patient_context else "",
    }).content
    source_list = sorted(refs.values(), key=lambda r: r["n"])
    db.execute("INSERT INTO medical_searches (run_id, topic, question, summary, sources, created_at) VALUES (?,?,?,?,?,?)",
               (run_id, topic, question, summary, db.dumps(source_list), db.now()))
    return {"ok": True, "topic": topic, "question": question, "summary": summary, "sources": source_list,
            "context": context, "from_cache_only": offline}
