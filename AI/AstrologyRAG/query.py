from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory=str(DB_DIR),
    collection_name="astrology_book",
    embedding_function=embeddings
)


while True:
    question = input("\nAsk about the book (or type 'exit'): ")

    if question.lower() == "exit":
        break

    results = db.similarity_search(question, k=5)

    print("\n" + "=" * 60)
    print("RELEVANT PASSAGES")
    print("=" * 60)

    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page_number", "Unknown")

        print(f"\n--- Result {i} | Page {page} ---")
        print(doc.page_content)