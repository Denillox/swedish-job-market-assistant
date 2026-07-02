# assistant/retriever.py
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = [PROJECT_ROOT / "docs" / "market_notes.md", PROJECT_ROOT / "README.md"]
PERSIST_DIR = PROJECT_ROOT / "embeddings"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_documents():

    documents = []
    for path in DOCS:
        text_loader = TextLoader(path, encoding="utf-8")
        documents.extend(text_loader.load())
    return documents


def _split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def _build_vectorstore():
    documents = _load_documents()
    chunks = _split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    return vectorstore

_vectorstore = None 


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore

def retrieve_context(query: str, k: int = 3) -> str:
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return "\n".join([result.page_content for result in results])