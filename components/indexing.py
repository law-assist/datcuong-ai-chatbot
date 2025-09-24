import time
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_chroma import Chroma


def text_splitter_recursive_config(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter: 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap,  
        add_start_index=True,  # track index in original document
    )
    return text_splitter

def text_splitter_semantic_config(embedding_model) -> SemanticChunker:
    text_splitter = SemanticChunker(
        embedding_model, breakpoint_threshold_type="percentile", add_start_index=True
    )
    return text_splitter

def indexing_docs(docs: List[Document], chunk_size: int, chunk_overlap: int, embedding_model: OllamaEmbeddings , vector_store: Chroma):
    # text_splitter = text_splitter_recursive_config(chunk_size, chunk_overlap)
    text_splitter = text_splitter_semantic_config(embedding_model)
    total_success = 0
    error_list = {}
    for doc in docs:
        chunked_docs = text_splitter.split_documents([doc])
        doc_id = str(doc.metadata["id"])
        print(f"Number of chunks {doc.metadata['name']} splited into: {len(chunked_docs)}")
        
        for attempt in range(3):  # Try up to 3 times
            try:
                # Operation that might fail
                vector_store.add_documents(
                    documents=chunked_docs,
                    collection_name="legislation",
                    ids=[(doc_id + "_" + str(doc.metadata["start_index"])) for doc in chunked_docs],
                )
                total_success += 1
                if doc_id in error_list:
                    error_list.pop(doc_id)
                break  # Exit the loop on success
            except Exception as e:
                if not doc_id in error_list:
                    error_list[doc_id] = e
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)  # Wait before retrying
        else:
            print("Failed to add documents after 3 attempts. Skipping this document.")
            
    return {"message": f"Successfully add {total_success}/{len(docs)} documents!",
            "errors": [f"Document ID {failed_doc_id}: {error_list[failed_doc_id]}" for failed_doc_id in error_list.keys()]}