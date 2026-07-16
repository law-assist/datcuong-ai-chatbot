"""
Build ChromaDB vector store from MongoDB laws collection.
Run from the project root: python scripts/build_vectorstore.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.data_processing import build_chroma_document_from_mongo_document

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "law_linking")
PERSIST_DIR = "./api/database/all-MiniLM-L6-v2/3000_300"
COLLECTION_NAME = "legislation"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300

def main():
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DBNAME]["laws"]
    docs = list(collection.find({"isDeleted": {"$ne": True}}))
    print(f"Found {len(docs)} laws")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    langchain_docs = []
    for mongo_doc in docs:
        try:
            processed = build_chroma_document_from_mongo_document(mongo_doc)
            meta = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in processed["metadata"].items()}
            chunks = splitter.split_text(processed["documents"])
            for chunk in chunks:
                langchain_docs.append(
                    Document(page_content=chunk, metadata=meta)
                )
        except Exception as e:
            print(f"  Skipping {mongo_doc.get('name', '?')}: {e}")

    print(f"Total chunks to embed: {len(langchain_docs)}")

    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Building ChromaDB at {PERSIST_DIR}...")
    vector_store = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Done. Vector store has {vector_store._collection.count()} chunks.")
    client.close()

if __name__ == "__main__":
    main()
