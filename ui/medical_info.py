"""Latest medical information retrieved by the RAG pipeline + ad-hoc search."""
import streamlit as st

from hca import db, memory
from hca.tools.medical_search import medical_search


def page() -> None:
    st.title("📚 Medical information")
    kb = memory.knowledge_store()
    docs = kb.all_documents()
    c = st.columns(3)
    c[0].metric("Knowledge-base chunks", len(kb))
    c[1].metric("Indexed sources", len({d.metadata.get("url") for d in docs}))
    c[2].metric("Searches", db.query_one("SELECT COUNT(*) n FROM medical_searches")["n"])

    with st.form("search"):
        topic = st.text_input("Disease / condition", placeholder="chronic kidney disease")
        question = st.text_input("Question (optional)", placeholder="What are the latest treatment options?")
        context = st.text_input("Patient context (optional)", placeholder="70-year-old with hypertension")
        if st.form_submit_button("Search MedlinePlus + WHO", type="primary") and topic:
            with st.spinner("Fetching sources, indexing into FAISS, retrieving and summarizing ..."):
                res = medical_search(topic, question or None, context or None)
            if res["ok"]:
                st.markdown(res["summary"])
                for s in res["sources"]:
                    st.markdown(f"{s['n']}. [{s['title']}]({s['url']}) — {s['source']}")
            else:
                st.error(res["error"])

    st.subheader("Latest retrieved information")
    for s in db.query("SELECT * FROM medical_searches ORDER BY id DESC LIMIT 15"):
        with st.expander(f"{s['created_at'][:16].replace('T', ' ')} · {s['topic']} — {s['question']}"):
            st.markdown(s["summary"])
            st.markdown("\n".join(f"{x['n']}. [{x['title']}]({x['url']}) — {x['source']}" for x in s["sources"]))
            if s["run_id"]:
                st.caption(f"agent run {s['run_id']}")

    if docs:
        with st.expander("Indexed sources"):
            seen = {}
            for d in docs:
                seen.setdefault(d.metadata["url"], (d.metadata["title"], d.metadata["source"], 0))
                t, src, n = seen[d.metadata["url"]]
                seen[d.metadata["url"]] = (t, src, n + 1)
            for url, (t, src, n) in seen.items():
                st.markdown(f"- [{t}]({url}) — {src} · {n} chunks")
