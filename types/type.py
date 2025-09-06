from typing import Any, List, TypedDict
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


# Langgraph state definition
class State(TypedDict):
    raw_question: str
    query: List[str]
    structured_query: List[str]
    context: List[tuple[Document, Any]]
    answer: str

# Chatbot component for api handler
class ChatbotComponents(BaseModel):
    llm: ChatOllama
    embedding_model: OllamaEmbeddings
    reranker_model: Any
    vector_store: Chroma
    retriever: VectorStoreRetriever
    graph: CompiledStateGraph[State, None, State, State]