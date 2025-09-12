from langchain.load import dumps, loads
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda 
from sentence_transformers import CrossEncoder
from typing import Union, Any
import requests
import os
from dotenv import load_dotenv

# Get JINAAI URL and API_KEY from environment variable or use default
load_dotenv(dotenv_path=".env")
JINAAI_API_URL = os.getenv("JINAAI_API_URL")
JINAAI_API_KEY = os.getenv("JINAAI_API_KEY")

def get_unique_union(documents: list[list[Document]]) -> list[Document]:
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [doc for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs_dict = {doc.id:doc for doc in flattened_docs}
    return [doc for doc in unique_docs_dict.values()] 

def retrieve_and_rerank(retriever: VectorStoreRetriever, reranker_model: Any, raw_question: str, queries: list[str], top_k: int=3) -> list[tuple[Document, Any]]:
    """ Retrieve documents for each query and rerank them"""
    # Define runnabale in retrieve and rerank chain
    def first_stage_retrieve(queries: list[str]) -> list[Document]:
        all_results = []
        for query in queries:
            results = retriever.invoke(query)
            all_results.append(results)
        unique_results = get_unique_union(all_results)
        print(f"Retrieved {len(unique_results)} unique documents.")
        print("-"* 50)  # Separator for readability
        return unique_results
    
    def rerank_local(retrieved_docs: list[Document]) -> list[tuple[Document, Any]]:    
        tokenized_raw_question = raw_question
        tokenized_retrieved_docs = [doc.page_content for doc in retrieved_docs]
        
        query_and_docs = [[tokenized_raw_question, tokenized_doc] for tokenized_doc in tokenized_retrieved_docs]
        
        scores = reranker_model.compute_score(query_and_docs, max_length=4096)
        return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)
    
    def rerank_api_call(retrieved_docs: list[Document]) -> list[tuple[Document, Any]]:
        url = JINAAI_API_URL
        headers = {
            "Authorization": f"Bearer {JINAAI_API_KEY}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                url,
                json={
                    "model": reranker_model,
                    "query": raw_question,
                    "documents": [doc.page_content for doc in retrieved_docs]
                },
                headers=headers,
                timeout=60
            )
            response.raise_for_status()  # Raise an error for bad responses
            scores = response.json().get("scores", [])
            return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)
        except requests.RequestException as e:
            raise ValueError("Failed to get reranking scores from the API")
    
    def rerank_docs(retrieved_docs: list[Document]) -> list[tuple[Document, Any]]:
        if isinstance(reranker_model, str):
            result = rerank_api_call(retrieved_docs)
        else:
            result = rerank_local(retrieved_docs)
        for i, doc in enumerate(result):
            print(f"Document {i + 1}:")
            print("Name:", doc[0].metadata.get("name", "N/A"))
            print("Id:", doc[0].metadata.get("id", "N/A"))
            print("Document number:", doc[0].metadata.get("numberDoc", "N/A"))
            print("Fields:", doc[0].metadata.get("fields", "N/A"))
            print("Score:", doc[1])
            print("-"* 50)  # Separator for readability
        return result
    
    FirstStageRetrieve = RunnableLambda(first_stage_retrieve)
    RerankDocs = RunnableLambda(rerank_docs)
       
    retrieve_rerank_chain = (
        FirstStageRetrieve
        | RerankDocs
    )
    final_results = retrieve_rerank_chain.invoke(queries)
    return final_results[:top_k]