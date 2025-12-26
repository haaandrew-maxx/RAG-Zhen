import os
from dotenv import load_dotenv  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# DOC_DIR = "./docs"

# def load_single_file(path):
#     ext = path.lower().split(".")[-1]

#     if ext == "pdf":
#         return PyPDFLoader(path).load()
#     elif ext in ["txt"]:
#         return TextLoader(path).load()
#     elif ext in ["md", "markdown"]:
#         return UnstructuredMarkdownLoader(path).load()
#     else:
#         return UnstructuredFileLoader(path).load()

# all_docs = []
# visible_files = [
#     f for f in sorted(os.listdir(DOC_DIR))
#     if not f.startswith(".")
# ]

# files = visible_files[:100]
# for f in files:
#     path = os.path.join(DOC_DIR, f)
#     print(f"Loading: {path}")
#     docs = load_single_file(path)
#     all_docs.extend(docs)

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=400, chunk_overlap=50
# )

# doc_splits = text_splitter.split_documents(all_docs)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
#     persist_directory="./.chroma",
# ).as_retriever()

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
).as_retriever(
    search_type="mmr",  # Back to MMR for better relevance
    search_kwargs={
        "k": 6,  # Rolled back from 10 to reduce noise        
        "fetch_k": 20,
        "score_threshold": 0.2,
        "lambda_mult": 0.6,  # Higher relevance weight
    },
)