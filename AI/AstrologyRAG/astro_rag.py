from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ============================================================
# CHROMA DATABASE
# ============================================================

db = Chroma(
    persist_directory=str(DB_DIR),
    collection_name="astrology_book",
    embedding_function=embeddings
)


# ============================================================
# ASTROLOGY RAG
# ============================================================

def get_astrology_context(question, k=5):

    results = db.similarity_search(
        question,
        k=k
    )

    if not results:
        return None

    context_parts = []

    for i, doc in enumerate(results, 1):

        source = doc.metadata.get(
            "source_file",
            "Unknown book"
        )

        page = doc.metadata.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {i}
Book: {source}
Page: {page}

{doc.page_content}
"""
        )

    return "\n".join(context_parts)










# from pathlib import Path

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma


# # ============================================================
# # PATHS
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent
# DB_DIR = BASE_DIR / "chroma_db"


# # ============================================================
# # EMBEDDING MODEL
# # Must be the SAME model used in ingest.py
# # ============================================================

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={
#         "device": "cpu"
#     },
#     encode_kwargs={
#         "normalize_embeddings": True
#     }
# )


# # ============================================================
# # LOAD EXISTING CHROMA DATABASE
# # ============================================================

# db = Chroma(
#     persist_directory=str(DB_DIR),
#     collection_name="astrology_book",
#     embedding_function=embeddings
# )


# # ============================================================
# # ASTROLOGY RAG SEARCH
# # ============================================================

# def get_astrology_context(question, k=5):
#     """
#     Search the astrology knowledge base and return
#     the most relevant passages.
#     """

#     results = db.similarity_search(
#         question,
#         k=k
#     )

#     if not results:
#         return None

#     context_parts = []

#     for i, doc in enumerate(results, 1):

#         source = doc.metadata.get(
#             "source_file",
#             "Unknown book"
#         )

#         page = doc.metadata.get(
#             "page_number",
#             "Unknown"
#         )

#         context_parts.append(
#             f"""
# SOURCE {i}
# Book: {source}
# Page: {page}

# {doc.page_content}
# """
#         )

#     return "\n".join(context_parts)

# # ============================================================
# # TEST
# # ============================================================

# if __name__ == "__main__":

#     question = input(
#         "\nAsk an astrology question (or type 'exit'): "
#     )

#     if question.lower() != "exit":

#         context = get_astrology_context(
#             question,
#             k=5
#         )

#         print("\n" + "=" * 70)
#         print("ASTROLOGY RAG RESULTS")
#         print("=" * 70)

#         if context:
#             print(context)
#         else:
#             print("No relevant passages found.")