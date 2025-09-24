from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 
from langchain_ollama import ChatOllama

def check_first_number_in_string(s: str) -> bool:
    """ Check if a string first character is numeric"""
    return s[0].isdigit()

def query_extraction(queries: list[str]) -> list[str]:
    """
    Extract queries from a list of strings, ensuring each query is non-empty
    """
    extracted_queries = []
    for query in queries:
        if query and check_first_number_in_string(query):
            extracted_query = query[2:].strip()  # Remove the leading number and dot
            extracted_queries.append(extracted_query)
    return extracted_queries

def query_translation(query: str, llm: ChatOllama) -> list[str]:
    """
    Translate a query into a format suitable for Document retrieval.
    """
    template = """
    Bạn là một trợ lý hữu ích với nhiệm vụ đề xuất các câu hỏi truy vấn dựa trên một câu hỏi đầu vào. \n
Hãy đề xuất các câu hỏi truy vấn bằng Tiếng Việt tương tự (4 kết quả): {question} \n
Trả lời dưới dạng đánh số thứ tự (1., 2., 3., 4.) mỗi câu một dòng. Không cần lời chào hỏi hay giới thiệu nào.
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
    translated_query.append(query) # Include original query as well
    print("Translated query:\n", "\n".join(translated_query))
    return translated_query