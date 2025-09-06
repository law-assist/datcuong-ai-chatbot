from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 
from langchain_ollama import ChatOllama

def check_first_number_in_string(s: str) -> bool:
    """ Check if a string first character is numeric"""
    return s[0].isdigit()

def query_extraction(queries: list[str]) -> list[str]:
    """
    Extract queries from a list of strings, ensuring each query is non-empty and contain one number character.
    """
    extracted_queries = []
    for query in queries:
        query = query.strip()
        if query and check_first_number_in_string(query):
            extracted_queries.append(query)
    return extracted_queries

def query_translation(query, llm: ChatOllama):
    """
    Translate a query into a format suitable for Document retrieval.
    """
    template = """
    Bạn là một trợ lý hữu ích với nhiệm vụ sinh ra các câu hỏi truy vấn dựa trên một câu hỏi đầu vào. \n
    Hãy tạo các câu hỏi truy vấn bằng Tiếng Việt liên quan đến: {question} \n
    Đầu ra KHÔNG có bất kỳ lời giới thiệu hay chào hỏi nào, CHỈ có duy nhất danh sách các câu truy vấn (4 kết quả):
    """
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    QueryExtraction = RunnableLambda(query_extraction)

    generate_queries_chain = (
        prompt_rag_fusion 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
        | QueryExtraction
    )
    translated_query = generate_queries_chain.invoke({"question": query})
    return translated_query