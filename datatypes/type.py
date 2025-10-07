from typing import Any, List, TypedDict, Optional
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph.state import StateGraph
from langgraph.graph import MessagesState

from components.retriever import RerankResults

# Langgraph state definition
class State(MessagesState):
    query: List[str]
    structured_query: List[str]
    context: List[RerankResults]

# Chatbot component for api handler
class ChatbotComponents(TypedDict):    
    llm: Optional[ChatOllama] 
    embedding_model: Optional[OllamaEmbeddings] 
    reranker_model: Optional[Any] 
    vector_store: Optional[Chroma] 
    retriever: Optional[VectorStoreRetriever] 
    graph_builder: Optional[StateGraph[State, None, State, State]] 