# Langchain import
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver


# Transformer import
from transformers import AutoModelForSequenceClassification

# Common import
from types import NoneType
from bson import ObjectId
from datetime import date, datetime

# FastAPI import
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager

# System import
import sys
sys.path.insert(0, "..")

# Local import
from components.indexing import indexing_docs
from components.query_translation import query_translation
from components.query_analysis import query_analysis
from components.router import route_tool
from components.retriever import retrieve_and_rerank, RerankResults
from utils.data_processing import build_chroma_document_from_mongo_document
from utils.mongo_handler import get_legislation_by_query
from datatypes.type import State, ChatbotComponents
from datatypes.dto import IndexingIdParam, QuestionAnsweringParam

# Enviroment import 
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=".env")
# Load environment variables
MONGO_URI = os.getenv("MONGO_URI")
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RERANK_MODEL = os.getenv("RERANK_MODEL")

VECTOR_STORE_COLLECTION = os.getenv("VECTOR_STORE_COLLECTION")
VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST")
VECTOR_STORE_PORT = int(os.getenv("VECTOR_STORE_PORT"))

chatbot_componets: ChatbotComponents = {}

# Chatbot components initialization
def components_initialize():
    # Ollama model initialization
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"LLM model loaded: {llm.model}")

    # Embedding model
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"Embedding model loaded: {embedding_model.model}")
    
    # Reranker model
    # reranker_model = AutoModelForSequenceClassification.from_pretrained(
    #     RERANK_MODEL,
    #     torch_dtype="auto",
    #     trust_remote_code=True,
    #     use_flash_attn=False,
    # )
    # reranker_model.to('cpu')
    reranker_model = RERANK_MODEL
    print(f"Reranker model loaded: {RERANK_MODEL[7:]}")  # Print model name without huggingface prefix

    # Vector store initialization
    vector_store = Chroma(
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
    Cung cấp tên tài liệu tham khảo trong câu trả lời của bạn.
    """
    prompt = ChatPromptTemplate.from_template(rag_template)
    
    # Langgraph node definition
    def translation(state: State):
        query = query_translation(state["messages"][-1].content, llm)
        return {"query": query}
        
    def analysis(state: State):
        structured_query = query_analysis(state["query"])
        return {"structured_query": structured_query}
        
    def retrieve(state: State):
        retrieved_docs = retrieve_and_rerank(retriever, reranker_model, state["messages"][-1].content, state["structured_query"], 6)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_contents = "\n\n".join(doc["document"].page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["messages"][-1].content, "context": docs_contents})
        response = llm.invoke(messages)
        return {"messages": {"role": "assistant", "content": [response.content]}}

    def quick_reply(state: State):
        response = llm.invoke(state["messages"][-1].content)
        return {"messages": {"role": "assistant", "content": [response.content]}}

    # Langgraph edge definition
    def router(state: State) -> str:
        return route_tool(state["messages"][-1].content, llm)
    
    graph_builder = StateGraph(State).add_sequence([translation, analysis, retrieve, generate])
    graph_builder.add_node(quick_reply)
    graph_builder.add_conditional_edges(START, router, {
        "có": "translation",
        "không": "quick_reply"
    })
    return graph_builder

# Sanitizing metadata function
def sanitize_metadata(metadata):
    """Sanitize metadata by removing keys that are not serializable."""
    for k, v in metadata.items():
        if isinstance(v, (dict, list)):
            metadata[k] = ", ".join(map(str, v))  # Convert to string
        if isinstance(v, (datetime, date)):
            metadata[k] = v.strftime("%Y-%m-%dT%H:%M:%S")  # Convert to ISO format string
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
    vector_store = chatbot_componets["vector_store"]
    # Embedding model
    embedding_model = chatbot_componets["embedding_model"]
    # Convert the documents to Chroma Document format
    docs = [Document(page_content=doc["documents"], metadata=sanitize_metadata(doc["metadata"])) for doc in chroma_documents]
    
    return indexing_docs(docs, chunk_size, chunk_overlap, embedding_model, vector_store)

# Checkpointer configuration
def checkpointer_config(user_id: str, chat_id: str):
    return {"configurable": {"thread_id": f"{user_id}_{chat_id}"}}

@asynccontextmanager
async def app_initialization(app: FastAPI):
    # Initialize components
    global chatbot_componets
    llm, embedding_model, reranker_model, vector_store, retriever = components_initialize()
    graph_builder = chatbot_build(llm, reranker_model, retriever)
    chatbot_componets["llm"] = llm
    chatbot_componets["embedding_model"] = embedding_model
    chatbot_componets["reranker_model"] = reranker_model
    chatbot_componets["vector_store"] = vector_store
    chatbot_componets["retriever"] = retriever
    chatbot_componets["graph_builder"] = graph_builder
    yield
    
app = FastAPI(lifespan=app_initialization)

@app.post("/agents/question-answering")
async def question_answering(param: QuestionAnsweringParam):
    raw_query = param.query
    user_id = param.user_id
    chat_id = param.chat_id
    print(f"User {user_id} - Chat {chat_id}: {raw_query}")
    
    graph_builder = chatbot_componets["graph_builder"]
    with MongoDBSaver.from_conn_string(f"{MONGO_URI}{CHECKPOINT_DB}") as checkpointer:
        chatbot = graph_builder.compile(checkpointer=checkpointer)
        config = checkpointer_config(user_id, chat_id)
              
        result = chatbot.invoke({"messages": {"type": "human" , "content" : raw_query}}, config=config)
    response = {"context": [{num: doc["document"].page_content} for num, doc in enumerate(result['context'])], "answer": result['messages'][-1].content[0]}
    return response



@app.get("/agents/chat-history")
async def chat_history(user_id: str = "1", chat_id: str = "1"):
    
    with MongoDBSaver.from_conn_string(f"{MONGO_URI}{CHECKPOINT_DB}") as checkpointer:
        config = checkpointer_config(user_id, chat_id)
        
        last_state = checkpointer.get_tuple(config)
        if isinstance(last_state, NoneType) or "channel_values" not in last_state.checkpoint:
            return {"history": [], "errors": "No chat history found."}
        channel_values = last_state.checkpoint["channel_values"]
        messages = channel_values["messages"]
        history = [{"human" : msg.content} if isinstance(msg, HumanMessage) else {"ai" : msg.content[0]} for msg in messages]
    response = {"history": history, "errors": None}
    return response
        
        
        
@app.post("/indexing/id")
async def mongoid_indexing(param: IndexingIdParam):
    legislation_id = param.indexing_id
    
    object_id = ObjectId(legislation_id)
    query = {"_id": object_id}
    response = query_mongo_and_index(query)
    return response
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)