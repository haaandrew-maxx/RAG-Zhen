import os
import re
import json
import pdfplumber
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

DOC_DIR = "./docs"
client = OpenAI()


# # ─────────────────────────────────────────────
# # Filename-based machine ID extraction
# # ─────────────────────────────────────────────

def machine_id_from_filename(filename: str) -> str | None:
    """Extract leading numeric machine ID from a filename, e.g.:
      '1005808 07 DGUV.pdf'  →  '1005808'
      'FB0010.01 ...'        →  None  (not a machine ID pattern)
    Returns the ID as a lowercase string, or None if not found.
    """
    m = re.match(r"^(\d{5,})", os.path.basename(filename))
    return m.group(1) if m else None


def sample_doc_pages(docs: list, max_pages: int = 8) -> str:
    """Return a text sample covering the full document by evenly picking up to
    max_pages pages (first, last, and evenly-spaced middle pages).
    This avoids the blind-spot of only reading the first 3 pages, which often
    miss machine IDs, component names, and norms buried deeper in the document.
    """
    if not docs:
        return ""
    if len(docs) <= max_pages:
        pages = docs
    else:
        # Always include the first and last page; fill the rest evenly
        step = (len(docs) - 1) / (max_pages - 1)
        indices = sorted(set(round(i * step) for i in range(max_pages)))
        pages = [docs[i] for i in indices]
    return " ".join(d.page_content for d in pages)

# ─────────────────────────────────────────────
# Metadata extraction via LLM
# ─────────────────────────────────────────────

SYSTEM_PROMPT_METADATA = """Eres un sistema de enriquecimiento de metadatos para documentos técnicos industriales.
Analiza el fragmento proporcionado (puede incluir texto normal Y tablas en formato Markdown) y responde
EXCLUSIVAMENTE con un objeto JSON válido con estas tres claves:

────────────────────────────────────────────────────────
1. "document_type" — UNA de estas categorías exactas (elige la más específica):
   "Manual técnico" | "Procedimiento" | "Informe" | "Especificación" |
   "Incidencia / Ticket" | "Email" | "Contrato" | "Presentación" | "Otro"

────────────────────────────────────────────────────────
2. "tags" — lista de 12 a 20 keywords técnicas de dominio industrial.

   CRITERIOS:
   ✓ Resumir aspectos técnicos del contenido (procesos, acciones, conceptos de seguridad, parámetros).
   ✓ Ser útiles para búsqueda y clasificación.
   ✓ Preferir términos técnicos específicos sobre genéricos.
   ✓ Añadir el idioma principal del documento (ej: "german", "spanish", "english").
   ✓ Si hay tablas, incluir keywords que describan qué datos contienen (ej: "lista de repuestos", "tabla de parámetros").
   ✗ NO usar frases largas, nombres propios ni términos totalmente genéricos ("documento", "sistema").

────────────────────────────────────────────────────────
3. "entities" — lista de objetos {"name": "...", "type": "..."}

   TIPOS PERMITIDOS (usar EXACTAMENTE uno de estos):
   maquina | componente | sistema | parametro | proceso | accion | error | entorno | proveedor | cliente | normativa | herramienta

   REGLAS CRÍTICAS:
   ✓ Extraer SOLO entidades con nombre CONCRETO y ESPECÍFICO presente en el texto o en las tablas.
   ✓ Incluir SIEMPRE identificadores numéricos que aparezcan junto a nombres de máquinas o componentes
     (e.g. "panel táctil 1005808", no solo "panel táctil" ni solo "1005808").
   ✓ Leer las tablas Markdown en busca de números de referencia, modelos, normas y proveedores.
   ✓ Incluir normas con su código completo ("iso 13849-1", "dguv vorschrift 3").
   ✓ Incluir proveedores/fabricantes con nombre legal completo.
   ✗ NO inventar entidades no presentes en el texto.
   ✗ NO usar tipos fuera de la lista.
   ✗ Máximo 10 entidades. Nombres en singular, sin artículos, en minúsculas.

   EJEMPLOS VÁLIDOS:
   {"name": "panel táctil 1005808",           "type": "maquina"}
   {"name": "mtpe machine dag br 449",         "type": "maquina"}
   {"name": "3con anlagenbau gmbh",            "type": "proveedor"}
   {"name": "iec 61439-2",                     "type": "normativa"}
   {"name": "dguv vorschrift 3",              "type": "normativa"}
   {"name": "sinamics g120",                  "type": "componente"}
   {"name": "presostático de aceite kp36",    "type": "componente"}

Devuelve SOLO el JSON, sin texto adicional, sin bloques de código."""


def extract_doc_metadata(text_sample: str, filename: str) -> dict:
    """Call GPT-4.1 to extract document_type, tags and entities for a document.
    text_sample should be produced by sample_doc_pages() for full coverage.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_METADATA},
                {"role": "user", "content": f"Archivo: {filename}\n\nFragmento (páginas representativas):\n{text_sample[:5000]}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [WARN] Metadata extraction failed for {filename}: {e}")
        return {"document_type": "Otro", "tags": [], "entities": []}


ALLOWED_ENTITY_TYPES = {
    "maquina", "componente", "sistema", "parametro", "proceso",
    "accion", "error", "entorno", "proveedor", "cliente", "normativa", "herramienta",
}


def metadata_to_flat(meta: dict, filename: str = "") -> dict:
    """Flatten nested metadata into ChromaDB-compatible strings.
    Only keeps entities whose type is in the closed allowed list.
    Also injects the numeric machine ID from the filename (e.g. '1005808')
    into entity_names so that all document types for the same machine
    (DGUV, maintenance plans, spare-parts lists, etc.) are discoverable
    by entity-filtered retrieval.
    """
    entities = meta.get("entities", [])
    tags = meta.get("tags", [])
    valid_entities = [
        e for e in entities
        if isinstance(e, dict)
        and e.get("type", "").lower().strip() in ALLOWED_ENTITY_TYPES
        and e.get("name", "").strip()
    ]
    entity_names_list = [e["name"].lower().strip() for e in valid_entities]
    # Inject machine ID from filename if not already present
    machine_id = machine_id_from_filename(filename)
    if machine_id and machine_id not in entity_names_list:
        entity_names_list.append(machine_id)
    return {
        "document_type": meta.get("document_type", "Otro"),
        "tags": ",".join(tags),
        "entity_names": ",".join(entity_names_list),
        "entity_types": ",".join(e["type"].lower().strip() for e in valid_entities),
    }


# ─────────────────────────────────────────────
# File loading (table-aware for PDFs)
# ─────────────────────────────────────────────

def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table (list of row lists) to a Markdown table string.
    Handles None cells, multi-line cells, and ragged rows gracefully.
    """
    if not table:
        return ""
    # Normalise: replace None → "", flatten multi-line cell content
    rows = [
        [str(cell or "").strip().replace("\n", " ") for cell in row]
        for row in table
    ]
    # Drop completely empty rows
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""

    # Determine column count from the widest row
    ncols = max(len(row) for row in rows)
    # Pad every row to ncols
    rows = [row + [""] * (ncols - len(row)) for row in rows]

    header = rows[0]
    data_rows = rows[1:]
    sep = ["---"] * ncols

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in data_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def load_pdf_with_tables(path: str) -> list:
    """Table-aware PDF loader.

    For each page:
    1. Uses pdfplumber to detect and extract tables → Markdown format.
    2. Extracts prose text from regions OUTSIDE table bounding boxes.
    3. Combines prose + markdown tables into the page Document.
    4. Tags pages that contain tables with metadata has_tables='true'.
    5. Falls back to plain PyPDFLoader if pdfplumber raises any error.

    Why this matters:
    - PyPDFLoader / pypdf extracts PDF text in reading order but treats table
      cells as a stream of words, jumbling columns together.
    - pdfplumber detects grid-based table regions and maps cells to rows and
      columns deterministically, yielding clean Markdown that the LLM can
      reason over (e.g. look up a part number, read a parameter range, etc.).
    """
    try:
        base_docs = PyPDFLoader(path).load()
    except Exception:
        base_docs = []

    # Map page index → (text, metadata) from PyPDFLoader
    base_by_page: dict[int, tuple[str, dict]] = {
        doc.metadata.get("page", i): (doc.page_content, doc.metadata)
        for i, doc in enumerate(base_docs)
    }

    result_docs: list[Document] = []

    try:
        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                base_text, meta = base_by_page.get(
                    page_idx, ("", {"source": path, "page": page_idx})
                )

                tables = page.find_tables()
                if not tables:
                    if base_text.strip():
                        result_docs.append(Document(page_content=base_text, metadata=meta))
                    continue

                # Collect table bounding boxes for exclusion
                table_bboxes = [t.bbox for t in tables]  # (x0, top, x1, bottom)

                # Build prose from words that fall OUTSIDE every table bbox
                words_outside: list[str] = []
                for word in page.extract_words():
                    wx0, wtop, wx1, wbot = (
                        word["x0"], word["top"], word["x1"], word["bottom"]
                    )
                    in_table = any(
                        bx0 <= wx0 and wtop >= btop and wx1 <= bx1 and wbot <= bbot
                        for bx0, btop, bx1, bbot in table_bboxes
                    )
                    if not in_table:
                        words_outside.append(word["text"])
                prose = " ".join(words_outside).strip()

                # Convert each detected table to Markdown
                md_tables: list[str] = []
                for tbl in tables:
                    data = tbl.extract()
                    if data:
                        md = _table_to_markdown(data)
                        if md:
                            md_tables.append(md)

                parts = ([prose] if prose else []) + md_tables
                combined = "\n\n".join(parts)

                if combined.strip():
                    enriched_meta = dict(meta)
                    if md_tables:
                        enriched_meta["has_tables"] = "true"
                    result_docs.append(
                        Document(page_content=combined, metadata=enriched_meta)
                    )

    except Exception as e:
        print(f"  [WARN] pdfplumber failed for {path}: {e} — using PyPDFLoader fallback")
        return base_docs

    return result_docs if result_docs else base_docs


def load_single_file(path):
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return load_pdf_with_tables(path)   # table-aware
    elif ext == "txt":
        return TextLoader(path).load()
    elif ext in ["md", "markdown"]:
        return UnstructuredMarkdownLoader(path).load()
    else:
        return UnstructuredFileLoader(path).load()


visible_files = [f for f in sorted(os.listdir(DOC_DIR)) if not f.startswith(".")]
files = visible_files[:100]

# ─────────────────────────────────────────────
# Step 1: Load files and extract metadata per file
# ─────────────────────────────────────────────

file_docs_map: dict = {}  # filename → list[Document]
all_docs = []

for f in files:
    path = os.path.join(DOC_DIR, f)
    print(f"Loading: {path}")
    docs = load_single_file(path)
    file_docs_map[f] = docs
    all_docs.extend(docs)

# ─────────────────────────────────────────────
# Step 2: Extract metadata from each file (LLM)
# ─────────────────────────────────────────────

print("\n--- Extracting document metadata ---")
file_metadata_map: dict = {}  # filename → flat metadata dict

for f, docs in file_docs_map.items():
    sample_text = sample_doc_pages(docs, max_pages=8)  # even sampling across full doc
    print(f"  [{f}]")
    raw_meta = extract_doc_metadata(sample_text, f)
    flat = metadata_to_flat(raw_meta, f)  # pass filename to inject machine ID
    file_metadata_map[f] = flat
    print(f"    type={flat['document_type']} | entities={flat['entity_names'][:80]}")

# ─────────────────────────────────────────────
# Step 3: Split and attach metadata to each chunk
# ─────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600, chunk_overlap=80
)
doc_splits = text_splitter.split_documents(all_docs)

for chunk in doc_splits:
    source = chunk.metadata.get("source", "")
    filename = os.path.basename(source)
    if filename in file_metadata_map:
        chunk.metadata.update(file_metadata_map[filename])

# ─────────────────────────────────────────────
# Step 4: Build vectorstore
# ─────────────────────────────────────────────

print("\n--- Building vectorstore ---")
Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
    persist_directory="./.chroma",
)

# Export raw Chroma instance (used by retrieve.py for filtered search)
vectorstore_chroma = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
)

# Export retriever (backward-compatible)
retriever = vectorstore_chroma.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.6,
    },
)