from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def check_number_in_string(s: str) -> bool:
    """ Check if a string contains any numeric characters """
    return any(char.isdigit() for char in s)

def query_extraction(queries: list[str]) -> list[str]:
    """
    Extract queries from a list of strings, ensuring each query is non-empty and contain one number character.
    """
    extracted_queries = []
    for query in queries:
        query = query.strip()
        if query and check_number_in_string(query):
            extracted_queries.append(query)
    return extracted_queries

def query_translation(query, llm: ChatOllama):
    """
    Translate a query into a format suitable for Document retrieval.
    """
    template = """Bạn là một trợ lý hữu ích với nhiệm vụ sinh ra các câu hỏi truy vấn dựa trên một câu hỏi đầu vào. \n
    Hãy tạo các câu hỏi truy vấn bằng Tiếng Việt liên quan đến: {question} \n
    Đầu ra không có bất kỳ lời giới thiệu hay giải thích nào (4 kết quả):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_rag_fusion 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
        | query_extraction
    )
    translated_query = generate_queries.invoke({"question": query})
    return translated_query