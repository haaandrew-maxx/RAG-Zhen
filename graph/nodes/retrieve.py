import json
import os
import re
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from graph.state import GraphState
from ingestion import retriever as base_retriever, vectorstore_chroma


embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url="https://api.openai.com/v1",
)
openai_client = OpenAI()

# ─────────────────────────────────────────────
# Entity extraction from user question
# ─────────────────────────────────────────────

ENTITY_EXTRACT_PROMPT = """Extract specific named entities from the following question related to industrial machinery documentation.

Return a JSON object with:
- "entities": list of objects {"name": "...", "type": "..."}

The "type" field MUST be EXCLUSIVELY one of these values:
  maquina | componente | sistema | parametro | proceso | accion | error | entorno | proveedor | cliente | normativa | herramienta

Rules:
- Only extract concrete named elements that are explicitly mentioned in the question.
- DO NOT invent or infer entities not present in the text.
- DO NOT use types outside the allowed list above.
- DO NOT include generic terms without a concrete name (e.g. "machine", "system" alone).
- Normalize names: lowercase, singular, no articles.
- Maximum 5 entities.
- If no clear entities, return {"entities": []}.

Return ONLY the JSON, no other text."""


ALLOWED_ENTITY_TYPES = {
    "maquina", "componente", "sistema", "parametro", "proceso",
    "accion", "error", "entorno", "proveedor", "cliente", "normativa", "herramienta",
}


def extract_entities_from_question(question: str) -> List[str]:
    """Use GPT-4.1 to extract entity names from the user question.
    Only returns names whose type is in the closed allowed list.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": ENTITY_EXTRACT_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        entities = result.get("entities", [])
        # Filter: only keep entities with a valid type
        valid = [
            e["name"].lower().strip()
            for e in entities
            if isinstance(e, dict)
            and e.get("type", "").lower().strip() in ALLOWED_ENTITY_TYPES
            and e.get("name", "").strip()
        ]
        return valid
    except Exception as e:
        print(f"  [WARN] Entity extraction failed: {e}")
        return []


# ─────────────────────────────────────────────
# Entity-filtered retrieval helpers
# ─────────────────────────────────────────────

def _extract_search_tokens(entities: List[str]) -> dict:
    """Split entity strings into two buckets:
      - machine_ids : purely numeric tokens of 5+ digits (e.g. '1005808')
      - text_tokens : alphabetic/alphanumeric tokens of 4+ chars (e.g. 'dguv', 'sinamics')
    Both buckets are de-duplicated and lower-cased.
    """
    machine_ids: list = []
    text_tokens: list = []
    for e in entities:
        for part in re.split(r"[\s_\-\.]+", e.lower()):
            if re.fullmatch(r"\d{5,}", part):
                machine_ids.append(part)
            elif len(part) >= 4 and not re.fullmatch(r"\d+", part):
                text_tokens.append(part)
    return {
        "machine_ids": list(dict.fromkeys(machine_ids)),
        "text_tokens": list(dict.fromkeys(text_tokens)),
    }


def _token_matches_entities(doc, entities: List[str]) -> bool:
    """Token-level match: each query entity is split into tokens; a doc matches
    if ANY token appears inside entity_names metadata OR the source filename.
    Much more robust than whole-string substring matching.
    """
    stored_raw = doc.metadata.get("entity_names", "").lower()
    source_fname = os.path.basename(doc.metadata.get("source", "")).lower()
    haystack = stored_raw + " " + source_fname

    for q_entity in entities:
        for tok in re.split(r"[\s_\-\.]+", q_entity.lower()):
            if len(tok) >= 3 and tok in haystack:
                return True
    return False


def _where_search_by_machine_id(query: str, machine_id: str, k: int) -> List:
    """Chroma similarity_search filtered to a specific machine ID.

    Uses the dedicated scalar 'machine_id' metadata field (written by
    patch_chroma_meta.py) with the $eq operator, which is supported by all
    Chroma versions.  Falls back to [] on any error.
    """
    try:
        return vectorstore_chroma.similarity_search(
            query,
            k=k,
            filter={"machine_id": {"$eq": machine_id}},
        )
    except Exception as e:
        print(f"  [WARN] Chroma machine_id filter failed for '{machine_id}': {e}")
        return []


def retrieve_with_entity_filter(question: str, entities: List[str], k: int = 8) -> List:
    """
    Three-strategy retrieval (tried in order, returns as soon as ≥2 docs found):

    Strategy 1 — Chroma $eq filter on 'machine_id' scalar field
                 Deterministic, language-agnostic, works with all Chroma versions.
                 Directly pins the semantic search to chunks that belong to a
                 specific machine (e.g. all docs for machine 1005808).
                 Requires patch_chroma_meta.py to have been run to populate
                 the 'machine_id' field.

    Strategy 2 — Semantic search pool + token-level metadata filter
                 Builds a large candidate pool from the question AND each entity
                 string, then filters by token-level match against entity_names
                 metadata and source filename.  Handles text entities (dguv,
                 sinamics, etc.) for which $eq filtering isn't applicable.

    Strategy 3 — Last resort: unfiltered MMR on full corpus.
    """
    if not entities:
        return base_retriever.invoke(question)

    print(f"  Entity filter applied: {entities}")
    tokens = _extract_search_tokens(entities)
    machine_ids = tokens["machine_ids"]

    seen_ids: set = set()
    where_docs: list = []

    def _add_unique(docs: list, target: list) -> None:
        for doc in docs:
            uid = doc.metadata.get("source", "") + doc.page_content[:80]
            if uid not in seen_ids:
                seen_ids.add(uid)
                target.append(doc)

    # ── Strategy 1: $eq filter on machine_id scalar field ─────────────────
    for mid in machine_ids:
        _add_unique(_where_search_by_machine_id(question, mid, k * 6), where_docs)

    if where_docs:
        print(f"  Machine-ID WHERE results: {len(where_docs)} docs")
        if len(where_docs) >= 2:
            return where_docs[:k]

    # ── Strategy 2: Semantic pool + token-level filter ────────────────────
    pool: list = list(where_docs)   # start with whatever WHERE gave us

    for doc in vectorstore_chroma.similarity_search(question, k=k * 10):
        uid = doc.metadata.get("source", "") + doc.page_content[:80]
        if uid not in seen_ids:
            seen_ids.add(uid)
            pool.append(doc)

    for entity in entities:
        for doc in vectorstore_chroma.similarity_search(entity, k=k * 4):
            uid = doc.metadata.get("source", "") + doc.page_content[:80]
            if uid not in seen_ids:
                seen_ids.add(uid)
                pool.append(doc)

    matched = [d for d in pool if _token_matches_entities(d, entities)]
    print(f"  Token-filter results: {len(matched)} / {len(pool)} candidates")

    if len(matched) >= 2:
        return matched[:k]

    # ── Strategy 3: Unfiltered MMR ─────────────────────────────────────
    print("  All entity strategies exhausted → fallback to unfiltered MMR")
    return base_retriever.invoke(question)


# ─────────────────────────────────────────────
# Main retrieve node
# ─────────────────────────────────────────────

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves relevant documents based on the operation mode.

    Mode A: knowledge base + session docs (description + BOM)
    Mode B: knowledge base only

    Entity-aware: extracts entities from question, filters ChromaDB first,
    falls back to plain MMR if filter yields too few results.
    """
    print("---RETRIEVE---")

    question = state["question"]
    bom = state.get("bom", "")
    mode = state.get("mode", "B")

    print(f"---OPERATING IN MODE {mode}---")

    merged = []

    # ── 1) Extract entities from question ──────────────────────────────
    entities = extract_entities_from_question(question)
    if entities:
        print(f"---EXTRACTED ENTITIES: {entities}---")
    else:
        print("---NO ENTITIES EXTRACTED---")

    # ── 2) Knowledge-base retrieval (entity-filtered or plain MMR) ─────
    docs_base = retrieve_with_entity_filter(question, entities, k=8)
    merged.extend(docs_base)
    print(f"---BASE RETRIEVER (Question): {len(docs_base)} docs---")

    # ── 3) Mode A extras ───────────────────────────────────────────────
    if mode == "A":
        if bom and bom.strip():
            # BOM query: use plain MMR (entity filter not useful here)
            docs_bom_base = base_retriever.invoke(bom)
            merged.extend(docs_bom_base)
            print(f"---BASE RETRIEVER (BOM): {len(docs_bom_base)} docs---")

        session_docs = state.get("session_docs", [])
        if session_docs:
            print(f"---SESSION DOCS FOUND: {len(session_docs)}---")
            session_vs = FAISS.from_documents(session_docs, embeddings)
            session_retriever = session_vs.as_retriever(search_kwargs={"k": 8})

            docs_query_session = session_retriever.invoke(question)
            merged.extend(docs_query_session)
            print(f"---SESSION RETRIEVER (Question): {len(docs_query_session)} docs---")

            if bom and bom.strip():
                docs_bom_session = session_retriever.invoke(bom)
                merged.extend(docs_bom_session)
                print(f"---SESSION RETRIEVER (BOM): {len(docs_bom_session)} docs---")

    else:
        print("---MODE B: Knowledge base only---")

    # ── 4) Deduplicate by content ──────────────────────────────────────
    unique: Dict[str, Any] = {}
    for d in merged:
        unique[d.page_content] = d
    merged_docs = list(unique.values())

    print(f"---MERGED DOC COUNT: {len(merged_docs)}---")

    return {
        "documents": merged_docs,
        "question": question,
    }
