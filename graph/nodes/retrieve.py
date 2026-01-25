from typing import Any, Dict, List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from graph.state import GraphState
from ingestion import retriever as base_retriever  # your original retriever


# Explicitly use OpenAI for embeddings (not Ollama)
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url="https://api.openai.com/v1"
)


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves relevant documents based on the operation mode.
    
    Mode A: Retrieves from knowledge base + session docs (description + BOM)
    Mode B: Retrieves only from knowledge base (no session docs)
    """
    print("---RETRIEVE---")

    question = state["question"]
    bom = state.get("bom", "")
    mode = state.get("mode", "B")
    
    print(f"---OPERATING IN MODE {mode}---")

    merged = []

    # ============================
    # 1) Base retriever (knowledge base) - ALWAYS USED
    # ============================
    docs_query_base = base_retriever.invoke(question)
    merged.extend(docs_query_base)
    print(f"---BASE RETRIEVER (Question): {len(docs_query_base)} docs---")

    # ============================
    # 2) Mode A: Also retrieve using BOM + session docs
    # ============================
    if mode == "A":
        # Retrieve using BOM as query
        if bom and bom.strip():
            docs_bom_base = base_retriever.invoke(bom)
            merged.extend(docs_bom_base)
            print(f"---BASE RETRIEVER (BOM): {len(docs_bom_base)} docs---")

        # Session retriever for uploaded files
        session_docs = state.get("session_docs", [])
        if session_docs:
            print(f"---SESSION DOCS FOUND: {len(session_docs)}---")
            # Build temporary FAISS vectorstore for the session
            session_vs = FAISS.from_documents(session_docs, embeddings)
            session_retriever = session_vs.as_retriever(search_kwargs={"k": 4})

            # Retrieve from session store
            docs_query_session = session_retriever.invoke(question)
            merged.extend(docs_query_session)
            print(f"---SESSION RETRIEVER (Question): {len(docs_query_session)} docs---")
            
            if bom and bom.strip():
                docs_bom_session = session_retriever.invoke(bom)
                merged.extend(docs_bom_session)
                print(f"---SESSION RETRIEVER (BOM): {len(docs_bom_session)} docs---")
    
    # ============================
    # 3) Mode B: Only knowledge base (already added above)
    # ============================
    else:  # mode == "B"
        print("---MODE B: Using only knowledge base documents---")

    # ============================
    # 4) Deduplicate by page_content
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