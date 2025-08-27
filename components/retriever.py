from typing import Optional, Union, Any
from langchain.load import dumps, loads
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification
import requests

def reciprocal_rank_fusion(results: list[list], k=60) -> list[tuple[Document, float]]:
    """ 
    Reciprocal_rank_fusion that takes multiple lists of ranked documents and an optional parameter k used in the RRF formula 
    """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def rerank_docs(raw_question, retrieved_docs: list[Document], reranker_model: Union[CrossEncoder, Any]) -> list[tuple[Document, Any]]:
    # if isinstance(tokenizer, type(None)):
    #     tokenized_raw_question = raw_question
    #     tokenized_retrieved_docs = [doc.page_content for doc in retrieved_docs]
    # else:
    #     tokenized_raw_question = tokenizer.word_segment(raw_question)
    #     tokenized_retrieved_docs = [tokenizer.word_segment(doc.page_content) for doc in retrieved_docs]
        
    tokenized_raw_question = raw_question
    tokenized_retrieved_docs = [doc.page_content for doc in retrieved_docs]
    
    query_and_docs = [[tokenized_raw_question, tokenized_doc] for tokenized_doc in tokenized_retrieved_docs]
    
    if isinstance(reranker_model, CrossEncoder):
        scores = reranker_model.predict(query_and_docs)
    else:
        scores = reranker_model.compute_score(query_and_docs, max_length=1024)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)

def rerank_api_call(raw_question, retrieved_docs: list[Document], reranker_model: Any) -> list[tuple[Document, Any]]:
    url = "https://api.jina.ai/v1/rerank"
    try:
        response = requests.post(
            url,
            json={
                "model": reranker_model,
                "data": [[raw_question, doc.page_content] for doc in retrieved_docs]
            },
            timeout=60
        )
        response.raise_for_status()  # Raise an error for bad responses
        scores = response.json().get("scores", [])
        return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)
    except requests.RequestException as e:
        raise ValueError("Failed to get reranking scores from the API")
    

def retrieve_and_rerank(retriever: VectorStoreRetriever, reranker_model: Union[CrossEncoder, Any], raw_question: str, queries: list[str], top_k: int=3) -> list[tuple[Document, Any]]:
    """ Retrieve documents for each query and rerank them"""
    all_results = []
    # model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    # compressor = CrossEncoderReranker(model=model, top_n=3)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=retriever
    # )
    
    for query in queries:
        results = retriever.invoke(query)
        all_results.append(results)
        
    # fused_results = reciprocal_rank_fusion(all_results)
    # if isinstance(tokenizer, type(None)):
    #     fused_results = rerank_docs(raw_question, [doc for docs in all_results for doc in docs], reranker_model)
    # else:
    #     fused_results = rerank_docs(raw_question, [doc for docs in all_results for doc in docs], reranker_model, tokenizer)
    if isinstance(reranker_model, str):
        fused_results = rerank_api_call(raw_question, [doc for docs in all_results for doc in docs], reranker_model)
    else:
        fused_results = rerank_docs(raw_question, [doc for docs in all_results for doc in docs], reranker_model)
    return fused_results[:top_k]