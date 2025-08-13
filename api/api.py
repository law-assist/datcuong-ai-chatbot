# Langchain import
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

# FastAPI import
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Common import
from typing import List, TypedDict

# System import
import sys
sys.path.insert(0, "..")

# Local import
from utils.dto import QueryQuestion
from components.query_translation import query_translation
from components.query_analysis import query_analysis

# Enviroment import 
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=".env")
# Load environment variables
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

def components_initialize():
    # Ollama model initialization
    local_model = "llama3"
    llm = ChatOllama(model=local_model)
    print(f"LLM model loaded: {llm.model}")

    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"Embedding model loaded: {embedding_model.model_name}")

    # Vector store initialization
    persist_directory = "./database/all-MiniLM-L6-v2/3000_300"
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        collection_name="legislation",
        embedding_function=embedding_model,
    )
    print(f"Total number of documents in the vector store: {vector_store._collection.count()}")
    
    return llm, embedding_model, vector_store

def chatbot_build(llm, embedding_model, vector_store):
    rag_template = """
    Bạn là một luật sư giàu kinh nghiệm với vai trò tư vấn pháp lý cho khách hàng. Hãy trả lời câu hỏi của khách hàng bằng tiếng Việt CHỈ dựa trên các tài liệu đã cung cấp:
    {context}
    Câu hỏi: {question}
    """
    prompt = ChatPromptTemplate.from_template(rag_template)
    
    # Langgraph state definition
    class State(TypedDict):
        raw_question: str
        query: str
        structured_query: str
        context: List[Document]
        answer: str
    
    # Langgraph node definition
    def translation(state: State):
        query = query_translation(state["raw_question"])
        return {"query": query}
    
    def analysis(state: State):
        structured_query = query_analysis(state["query"])
        return {"structured_query": structured_query}
    
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["structured_query"], k=5)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["structured_query"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    graph_builder = StateGraph(State).add_sequence([translation, analysis, retrieve, generate])
    graph_builder.add_edge(START, "translation")
    graph = graph_builder.compile()
    return graph

chatbot_components = {}    

@asynccontextmanager
async def app_initialization(app: FastAPI):
    # Initialize components
    llm, embedding_model, vector_store = components_initialize()
    graph = chatbot_build(llm, embedding_model, vector_store)
    chatbot_components["llm"] = llm
    chatbot_components["embedding_model"] = embedding_model
    chatbot_components["vector_store"] = vector_store
    chatbot_components["graph"] = graph
    yield
    
app = FastAPI(lifespan=app_initialization)

@app.post("/agents/question-answering")
async def question_answering(question: QueryQuestion):
    raw_query = question.query
    chatbot = chatbot_components["graph"]
    result = chatbot.invoke({"raw_question": raw_query})
    response = {"context": [{num: doc.page_content} for num, doc in enumerate(result['context'])], "answer": result['answer']}
    return response
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)