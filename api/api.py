# Langchain import
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

# Transformer import
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification
# import py_vncorenlp

# Common import
import time
from bson import ObjectId

# FastAPI import
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager

# System import
import sys
sys.path.insert(0, "..")

# Local import
from utils.data_processing import build_chroma_document_from_mongo_document
from utils.mongo_handler import get_legislation_by_query
from utils.dto import IndexMongoId, QueryQuestion
from types.type import State, ChatbotComponents
from components.query_translation import query_translation
from components.query_analysis import query_analysis
from components.retriever import retrieve_and_rerank
from components.indexing import indexing_docs

# Enviroment import 
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=".env")
# Load environment variables
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
# RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH"))

VECTOR_STORE_COLLECTION = os.getenv("VECTOR_STORE_COLLECTION")
VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST")
VECTOR_STORE_PORT = int(os.getenv("VECTOR_STORE_PORT"))

# Chatbot components initialization
def components_initialize():
    # Ollama model initialization
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"LLM model loaded: {llm.model}")

    # Embedding model
    # embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"Embedding model loaded: {embedding_model.model}")
    
    # Reranker model
    # reranker_model = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL,
        torch_dtype="auto",
        trust_remote_code=True,
        use_flash_attn=False,
    )
    reranker_model.to('cpu')
    # reranker_model = RERANK_MODEL
    print(f"Reranker model loaded: {RERANK_MODEL}")

    # Vector store initialization
    vector_store = Chroma(
        # persist_directory=persist_directory,
        host=VECTOR_STORE_HOST,
        port=VECTOR_STORE_PORT,
        collection_name=VECTOR_STORE_COLLECTION,
        embedding_function=embedding_model,
    )
    print(f"Total number of documents in the vector store: {vector_store._collection.count()}")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
    
    return llm, embedding_model, reranker_model, vector_store, retriever

# Chatbot build function
def chatbot_build(llm: ChatOllama, reranker_model, retriever: VectorStoreRetriever):
    rag_template = """
    Bạn là một luật sư giàu kinh nghiệm với vai trò tư vấn pháp lý cho khách hàng. Hãy trả lời câu hỏi của khách hàng bằng tiếng Việt CHỈ dựa trên các tài liệu đã cung cấp:
    {context}
    Câu hỏi: {question}
    """
    prompt = ChatPromptTemplate.from_template(rag_template)
    
    # Langgraph node definition
    def translation(state: State):
        query = query_translation(state["raw_question"], llm)
        return {"query": query}
    
    def analysis(state: State):
        structured_query = query_analysis(state["query"])
        return {"structured_query": structured_query}
    
    def retrieve(state: State):
        # retrieved_docs = vector_store.similarity_search(state["structured_query"], k=5)
        retrieved_docs = retrieve_and_rerank(retriever, reranker_model, state["raw_question"], state["structured_query"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["raw_question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    graph_builder = StateGraph(State).add_sequence([translation, analysis, retrieve, generate])
    graph_builder.add_edge(START, "translation")
    graph = graph_builder.compile()
    return graph

# Sanitizing metadata function
def sanitize_metadata(metadata):
    """Sanitize metadata by removing keys that are not serializable."""
    for k, v in metadata.items():
        if isinstance(v, (dict, list)):
            metadata[k] = ", ".join(map(str, v))  # Convert to string
    return metadata

# Query MongoDB and index to vector store
def query_mongo_and_index(query):
    legislations = get_legislation_by_query(query)
    chroma_documents = [build_chroma_document_from_mongo_document(doc) for doc in legislations["data"]]
    print(f"Number of documents: {len(chroma_documents)}")
    
    # Text splitter configuration
    chunk_size = 3000  # chunk size (characters)
    chunk_overlap = 300  # chunk overlap (characters)  
    
    # Vector store
    vector_store = chatbot_componets.vector_store
    # Convert the documents to Chroma Document format
    docs = [Document(page_content=doc["documents"], metadata=sanitize_metadata(doc["metadata"])) for doc in chroma_documents]
    
    return indexing_docs(docs, chunk_size, chunk_overlap, vector_store)

chatbot_componets: ChatbotComponents

@asynccontextmanager
async def app_initialization(app: FastAPI):
    # Initialize components
    llm, embedding_model, reranker_model, vector_store, retriever = components_initialize()
    graph = chatbot_build(llm, reranker_model, retriever)
    chatbot_componets.llm = llm
    chatbot_componets.embedding_model = embedding_model
    chatbot_componets.reranker_model = reranker_model
    chatbot_componets.vector_store = vector_store
    chatbot_componets.retriever = retriever
    chatbot_componets.graph = graph
    yield
    
app = FastAPI(lifespan=app_initialization)

@app.post("/agents/question-answering")
async def question_answering(question: QueryQuestion):
    raw_query = question.query
    chatbot = chatbot_componets.graph
    result = chatbot.invoke({"raw_question": raw_query})
    response = {"context": [{num: doc.page_content} for num, doc in enumerate(result['context'])], "answer": result['answer']}
    return response

@app.post("/indexing/id")
async def mongoid_indexing(param: IndexMongoId):
    legislation_id = param.indexing_id
    object_id = ObjectId(legislation_id)
    query = {"_id": object_id}
    response = query_mongo_and_index(query)
    return response
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)