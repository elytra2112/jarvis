from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "Astrology-e-book-compressed.pdf"
DB_DIR = BASE_DIR / "chroma_db"

# PDF_PATH = "data/Astrology-e-book-compressed.pdf"
# DB_DIR = "chroma_db"

print("Loading PDF...")

loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

for chunk in chunks:
    if "page" in chunk.metadata:
        chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    chunk.metadata["source_file"] = "Astrology-e-book-compressed.pdf"

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Creating Chroma vector database...")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
    collection_name="astrology_book"
)

print("\n==============================")
print("VECTOR DATABASE CREATED!")
print("==============================")
print(f"Pages : {len(documents)}")
print(f"Chunks: {len(chunks)}")
print(f"DB    : {DB_DIR}")