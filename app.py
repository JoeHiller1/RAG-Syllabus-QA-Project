import os
import streamlit as st
from rag_mvp import load_or_build_index, answer_query

st.set_page_config(page_title="Syllabus RAG MVP", layout="wide")

st.title("Syllabus RAG MVP")
st.caption("Ask questions about course syllabi. Answers are grounded in retrieved chunks.")

docs_dir = st.text_input("Docs folder (in repo)", value="docs")
query = st.text_input("Question", value="What is covered in week 5?")
top_k = st.slider("top_k", 1, 10, 5)

if st.button("Run"):
    index, chunks, model = load_or_build_index(docs_dir)
    ans, retrieved = answer_query(query, index, chunks, model, top_k=top_k)

    st.subheader("Answer")
    st.write(ans)

    st.subheader("Citations / Retrieved chunks")
    for r in retrieved:
        st.markdown(f"**{r['doc_id']} — chunk {r['chunk_id']} (score={r['score']:.3f})**")
        st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
        st.divider()
