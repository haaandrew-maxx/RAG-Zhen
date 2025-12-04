from typing import Any, Dict, List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from graph.state import GraphState
from ingestion import retriever as base_retriever  # your original retriever


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")

    question = state["question"]
    bom = state["bom"]

    # ============================
    # 1) Base retriever (your existing one)
    # ============================
    docs_query_base = base_retriever.invoke(question)
    docs_bom_base = base_retriever.invoke(bom)

    merged = docs_query_base + docs_bom_base

    # ============================
    # 2) Session retriever (optional)
    # ============================
    session_docs = state.get("session_docs", [])

    if session_docs:
        print(f"---SESSION DOCS FOUND: {len(session_docs)}---")
        # Build temporary FAISS vectorstore for the session
        session_vs = FAISS.from_documents(session_docs, embeddings)
        session_retriever = session_vs.as_retriever(search_kwargs={"k": 4})

        # Retrieve from session store using your .invoke API
        docs_query_session = session_retriever.invoke(question)
        docs_bom_session = session_retriever.invoke(bom)

        merged.extend(docs_query_session + docs_bom_session)

    # ============================
    # 3) Deduplicate by page_content (keep your logic)
    # ============================
    unique = {}
    for d in merged:
        unique[d.page_content] = d

    merged_docs = list(unique.values())

    print(f"---MERGED DOC COUNT: {len(merged_docs)}---")

    # Return merged docs + question unchanged
    return {
        "documents": merged_docs,
        "question": question,
    }